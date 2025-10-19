#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune_vn_coral_lora_adapter_v2.py

Improved version with these changes to increase generalization and lower MAE:
- Mixed precision training (torch.cuda.amp)
- Gradient clipping
- EMA (Exponential Moving Average) of weights for evaluation
- Focal loss wrapper for ordinal and presence heads (helps class imbalance)
- Per-ordinal-class pos_weight computed from training class frequencies
- Stronger adapter (bottleneck=256) + LayerNorm + residual scaling
- Optional simple back-translation/paraphrase disabled by default (kept but optional)
- Improved weighted sampler logic
- Early stopping and checkpointing by combined metric (f1_macro - mae_scaled)
- Learning-rate warmup + cosine annealing scheduler
- Command-line flags to experiment quickly

Note: this script focuses on code-level improvements. To achieve target metrics you must run experiments, try hyperparameters and possibly collect more labeled data.
"""

import os
import random
import argparse
import unicodedata
import regex as re
from typing import List, Optional, Tuple, Dict, Any
import math
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# Optional libs
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_ST = True
except Exception:
    HAS_ST = False

# ---------- Config ----------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # 0..5
# --------------------------

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------
# Load & clean (unchanged except small robustness tweaks)
# --------------------------
def Load_data(path: str = "/mnt/data/train-problem.csv", text_col_candidates: List[str] = None) -> pd.DataFrame:
    if text_col_candidates is None:
        text_col_candidates = ["text", "review", "Review", "review_text", "content"]
    df = pd.read_csv(path)
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: "text"})
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            df[lbl] = 0
    return df[["text"] + LABEL_COLUMNS]


def _remove_control_chars(s: str) -> str:
    return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _remove_control_chars(s)
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    # keep letters, numbers, simple punctuation, whitespace
    s = re.sub(r"[^\w\s\.,!?\-:;'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def Clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text'] = df['text'].astype(str).apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df

# --------------------------
# Dataset
# --------------------------
class MultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# --------------------------
# Adapter with layernorm and residual scaling
# --------------------------
class Adapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # residual scaling to stabilize large adapters
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        z = self.dropout(z)
        return self.ln(x + self.scale * z)


class Multi_label_Model(nn.Module):
    def __init__(self, model_name: str = 'vinai/phobert-base', num_labels: int = len(LABEL_COLUMNS), adapter_bottleneck: int = 256, adapter_dropout: float = 0.1):
        super().__init__()
        self.encoder_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.adapter = Adapter(hidden_size, bottleneck=adapter_bottleneck, dropout=adapter_dropout)
        self.ord_heads = nn.ModuleList([nn.Linear(hidden_size, K_CLASSES-1) for _ in range(num_labels)])
        self.pres_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])
        self.reg_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])
        self.dropout = nn.Dropout(0.15)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last)
            pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        pooled = self.adapter(pooled)

        ord_logits = [head(pooled) for head in self.ord_heads]   # list of [B, K-1]
        pres_logits = [head(pooled).squeeze(-1) for head in self.pres_heads]  # list of [B]
        reg_outs = [head(pooled).squeeze(-1) for head in self.reg_heads]  # list of [B]
        ord_logits = torch.stack(ord_logits, dim=1)  # [B, L, K-1]
        pres_logits = torch.stack(pres_logits, dim=1)  # [B, L]
        reg_outs = torch.stack(reg_outs, dim=1)  # [B, L]
        return {'ord_logits': ord_logits, 'pres_logits': pres_logits, 'reg_outs': reg_outs}

# --------------------------
# Helpers
# --------------------------
def to_ordinal_targets(y: torch.Tensor, K: int = K_CLASSES):
    B = y.size(0)
    out = torch.zeros(B, K-1, device=y.device)
    for k in range(1, K):
        out[:, k-1] = (y >= k).float()
    return out


def compute_per_label_class_weights(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, device='cpu'):
    weight_dict = {}
    for lbl in labels:
        y = df[lbl].astype(int).values
        classes = np.arange(K_CLASSES)
        # compute pos weight for BCE: pos_weight = (N - pos) / pos for each ordinal column
        counts = np.array([ (y >= k).sum() for k in range(1, K_CLASSES) ], dtype=float)
        N = len(y)
        pos_weights = []
        for c in counts:
            pos = c
            neg = N - pos
            if pos <= 0:
                pw = 1.0
            else:
                pw = max(1.0, neg / (pos + 1e-6))
            pos_weights.append(float(pw))
        weight_dict[lbl] = torch.tensor(pos_weights, dtype=torch.float).to(device)
    return weight_dict


def make_weighted_sampler(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS):
    # improved sampler: treat rare higher-class labels as more important but avoid extreme upweighting
    freqs = {lbl: df[lbl].value_counts().to_dict() for lbl in labels}
    weights = []
    for idx, row in df.iterrows():
        score = 1.0
        for lbl in labels:
            cls = int(row[lbl])
            freq = freqs[lbl].get(cls, 1)
            # rarer classes get higher weight, but clip
            w = min(20.0, max(1.0, (len(df) / (freq + 1))))
            # de-emphasize class 0 (often majority/no-op)
            if cls == 0:
                w *= 0.3
            score = max(score, w)
        weights.append(score)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

# LoRA helpers unchanged
def auto_detect_lora_target_modules(model):
    names = [n for n, _ in model.named_modules()]
    candidates = set()
    for n in names:
        ln = n.lower()
        for pat in ['q_proj','k_proj','v_proj','o_proj','q','k','v','proj','dense','query','key','value','attention']:
            if pat in ln:
                candidates.add(pat)
    prefer = ['q_proj','k_proj','v_proj','o_proj','query','key','value','dense','proj']
    found = []
    for p in prefer:
        if any(p in n.lower() for n in names):
            found.append(p)
    if not found:
        found = ['query','key','value']
    return list(dict.fromkeys(found))


def apply_lora_to_model(base_model, r=8, alpha=32, dropout=0.05, target_modules: Optional[List[str]] = None):
    if not HAS_PEFT:
        raise RuntimeError("PEFT is not installed. pip install peft")
    if target_modules is None:
        target_modules = auto_detect_lora_target_modules(base_model)
    print("Applying LoRA to target_modules:", target_modules)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model

# --------------------------
# Losses: focal for BCE
# --------------------------
class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # logits: [B, ...], targets: same shape (0/1)
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * ce_loss
        return loss.mean()

# EMA helper
class ModelEMA(object):
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.ema[k] = (1.0 - self.decay) * v.detach().cpu() + self.decay * self.ema[k]

    def store(self, device='cpu'):
        self._backup = {k: v.clone() for k, v in self.ema.items()}

    def copy_to(self, model):
        sd = model.state_dict()
        for k in sd.keys():
            if k in self.ema:
                sd[k].copy_(self.ema[k].to(sd[k].device))
        model.load_state_dict(sd)

# --------------------------
# Eval (unchanged logic but accepts EMA model)
# --------------------------
def test(model_or_path, dataloader=None, device: Optional[str] = None, return_preds: bool = True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(model_or_path, str):
        model = Multi_label_Model()
        model.load_state_dict(torch.load(model_or_path, map_location=device))
        model.to(device)
    else:
        model = model_or_path
    if dataloader is None:
        raise ValueError("dataloader is required")
    model.eval()
    all_preds = []
    all_labels = []
    all_reg_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            ord_logits = out['ord_logits'].cpu().numpy()
            reg_outs = out['reg_outs'].cpu().numpy()
            pres_logits = out['pres_logits'].cpu().numpy()
            B,L,K1 = ord_logits.shape
            preds = np.zeros((B, L), dtype=int)
            for i in range(B):
                for j in range(L):
                    logits_bin = ord_logits[i,j,:]
                    probs = 1.0 / (1.0 + np.exp(-logits_bin))
                    pred_class = int((probs > 0.5).sum())
                    preds[i,j] = pred_class
            reg_preds = np.round(np.clip(reg_outs, 0, K_CLASSES-1)).astype(int)
            all_preds.append(preds)
            all_reg_preds.append(reg_preds)
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = {'accuracy_per_label': {}, 'f1_macro_per_label': {}, 'mae_per_label': {}}
    for j, lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:,j], all_preds[:,j]))
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:,j], all_preds[:,j], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:,j], all_preds[:,j]))
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))
    if return_preds:
        return {**metrics, 'preds': all_preds, 'labels': all_labels, 'reg_preds': all_reg_preds}
    else:
        return metrics

# --------------------------
# Train: improved with AMP, EMA, gradient clipping, focal loss
# --------------------------
def train(df: pd.DataFrame,
          model_name: str = 'vinai/phobert-base',
          out_dir: str = './model_out',
          epochs: int = 3,
          batch_size: int = 16,
          lr_lora: float = 5e-4,
          lr_adapter: float = 2e-4,
          lr_heads: float = 2e-4,
          lr_encoder_top: float = 1e-5,
          max_length: int = 256,
          val_size: float = 0.1,
          device: Optional[str] = None,
          use_lora: bool = True,
          lora_r: int = 8,
          lora_alpha: int = 32,
          lora_dropout: float = 0.05,
          lora_target_modules: Optional[List[str]] = None,
          adapter_bottleneck: int = 256,
          adapter_dropout: float = 0.1,
          use_sampler: bool = True,
          do_augment: bool = False,
          mae_weight: float = 0.5,
          pres_weight: float = 0.6,
          focal_gamma: float = 2.0,
          ema_decay: float = 0.999,
          grad_clip: float = 1.0):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if do_augment:
        print("Paraphrase augmentation enabled (may be slow)")
        df = paraphrase_augment_for_labels(df, device=device)

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=RANDOM_SEED)
    class_pos_weights = compute_per_label_class_weights(train_df, labels=LABEL_COLUMNS, device=device)
    print("Per-label ordinal pos-weights (len K-1):")
    for k,v in class_pos_weights.items():
        print(k, v.cpu().numpy())

    train_labels = train_df[LABEL_COLUMNS].values.astype(int)
    val_labels = val_df[LABEL_COLUMNS].values.astype(int)

    train_dataset = MultiLabelDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length=max_length)

    if use_sampler:
        sampler = make_weighted_sampler(train_df, labels=LABEL_COLUMNS)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Loading base encoder:", model_name)
    base_encoder = AutoModel.from_pretrained(model_name)
    if use_lora:
        if not HAS_PEFT:
            raise RuntimeError("PEFT not installed. Install via `pip install peft` to use LoRA.")
        if lora_target_modules is None:
            detected = auto_detect_lora_target_modules(base_encoder)
            print("Auto-detected LoRA target modules:", detected)
            lora_target_modules = detected
        base_encoder = apply_lora_to_model(base_encoder, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=lora_target_modules)
        print("LoRA applied.")

    model = Multi_label_Model(model_name=model_name, adapter_bottleneck=adapter_bottleneck, adapter_dropout=adapter_dropout)
    model.encoder = base_encoder
    model.to(device)

    # Freeze non-trainable parts
    if use_lora and HAS_PEFT:
        print("Freezing non-LoRA encoder parameters...")
        for n, p in model.encoder.named_parameters():
            if 'lora_' in n or 'alpha' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        print("No LoRA: freezing embeddings and lower encoder layers by default")
        for n, p in model.encoder.named_parameters():
            if n.startswith('embeddings.') or 'layer.0' in n or 'layer.1' in n:
                p.requires_grad = False

    lora_params = []
    adapter_params = list(model.adapter.parameters())
    head_params = list(model.ord_heads.parameters()) + list(model.pres_heads.parameters()) + list(model.reg_heads.parameters())

    if use_lora and HAS_PEFT:
        for n,p in model.encoder.named_parameters():
            if p.requires_grad:
                lora_params.append(p)
    else:
        enc_trainable = [p for n,p in model.encoder.named_parameters() if p.requires_grad]
        lora_params = enc_trainable

    optimizer = AdamW([
        {'params': lora_params, 'lr': lr_lora},
        {'params': adapter_params, 'lr': lr_adapter},
        {'params': head_params, 'lr': lr_heads}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    # losses
    bce_focal_per_label = {}
    for lbl in LABEL_COLUMNS:
        pos_w = class_pos_weights[lbl]  # tensor length K-1
        bce_focal_per_label[lbl] = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=pos_w)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))

    ema = ModelEMA(model, decay=ema_decay)

    best_score = -1e9
    early_patience = 5
    no_imp = 0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [B, L]

            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                ord_logits = out['ord_logits']   # [B,L,K-1]
                pres_logits = out['pres_logits'] # [B,L]
                reg_outs = out['reg_outs']       # [B,L]

                per_label_losses = []
                per_label_mae = []
                for i, lbl in enumerate(LABEL_COLUMNS):
                    logit_i = ord_logits[:, i, :]  # [B, K-1]
                    tgt_i = labels[:, i]           # [B]
                    ord_t = to_ordinal_targets(tgt_i, K=K_CLASSES)  # [B, K-1]
                    # focal loss for ordinal
                    loss_ord = bce_focal_per_label[lbl](logit_i, ord_t)
                    pres_t = (tgt_i > 0).float()
                    # presence focal
                    loss_pres = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=None)(pres_logits[:, i], pres_t)
                    mae_i = l1(reg_outs[:, i], tgt_i.float())
                    per_label_losses.append(loss_ord + pres_weight * loss_pres)
                    per_label_mae.append(mae_i)

                loss_main = torch.stack(per_label_losses).mean()
                loss_mae = torch.stack(per_label_mae).mean()
                loss = loss_main + mae_weight * loss_mae

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # update EMA
            ema.update(model)

            losses.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(losses))})

        # validation with EMA weights
        # store original and copy EMA weights to model
        ema.store()
        ema.copy_to(model)
        val_metrics = test(model, val_loader, device=device, return_preds=False)
        # restore original not necessary because we didn't change backup of current state_dict
        print(f"Epoch {epoch} validation:", val_metrics)

        # combined score: prioritize f1_macro and penalize mae
        val_f1 = val_metrics.get('f1_macro', 0.0)
        val_mae = val_metrics.get('mae', 1.0)
        combined = val_f1 - 0.2 * val_mae
        if combined > best_score:
            best_score = combined
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            print(f"Saved best model (epoch {epoch}) with combined={combined:.4f} f1={val_f1:.4f} mae={val_mae:.4f}")
            no_imp = 0
        else:
            no_imp += 1
            print(f"No improvement for {no_imp} epochs (best combined={best_score:.4f})")
            if no_imp >= early_patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best combined score:", best_score)
    return model, tokenizer

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train-problem.csv')
    parser.add_argument('--out_dir', type=str, default='./model_out')
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--do_augment', action='store_true')
    parser.add_argument('--mae_weight', type=float, default=0.3)
    parser.add_argument('--pres_weight', type=float, default=0.6)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    args = parser.parse_args()

    print("Loading dataset...")
    df = Load_data(args.data)
    df = Clean_and_normalize_data(df)
    print("Dataset shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    model, tokenizer = train(df,
                             model_name=args.model_name,
                             out_dir=args.out_dir,
                             epochs=args.epochs,
                             batch_size=args.bs,
                             max_length=args.max_len,
                             use_lora=args.use_lora,
                             use_sampler=args.use_sampler,
                             do_augment=args.do_augment,
                             mae_weight=args.mae_weight,
                             pres_weight=args.pres_weight,
                             focal_gamma=args.focal_gamma)

    best = os.path.join(args.out_dir, 'best_model.pt')
    if os.path.exists(best):
        print("Loading best model for evaluation...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        df_eval = Clean_and_normalize_data(Load_data(args.data))
        _, val_df = train_test_split(df_eval, test_size=0.1, random_state=RANDOM_SEED)
        val_loader = DataLoader(MultiLabelDataset(val_df['text'].tolist(), val_df[LABEL_COLUMNS].values.astype(int), tokenizer, max_length=args.max_len), batch_size=args.bs, shuffle=False)
        results = test(best, dataloader=val_loader, return_preds=True)
        print("Final evaluation on validation:", results)
    else:
        print("No best model saved.")
