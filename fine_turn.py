#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast medium model variant for Hackathon:
- Default backbone: vinai/phobert-base (Vietnamese)
- Freeze encoder except last N layers (light fine-tune)
- Smaller MLP heads for speed
- Mixed precision, EMA, gradient clipping
- Early stopping + max_time cutoff (sec) — default 1800s (30 min)
"""

import os, time, argparse, random, unicodedata
import regex as re
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # ordinal 0..5

def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    s = re.sub(r"[^\p{L}\p{N}\s\.,!?\-:;'\"()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# Dataset
# -------------------------
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer(txt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -------------------------
# Model (light / medium)
# -------------------------
class MultiTaskOrdinalModel(nn.Module):
    def __init__(self, backbone_name='vinai/phobert-base', fc_dim=512, mid_dim=256, hidden_dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size

        # small MLP blocks
        self.proj = nn.Linear(hidden_size, fc_dim)
        self.ln_proj = nn.LayerNorm(fc_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout)

        self.fc2 = nn.Linear(fc_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)

        # heads per label
        self.class_heads = nn.ModuleList([nn.Linear(mid_dim, K_CLASSES) for _ in LABEL_COLUMNS])
        self.reg_heads = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in LABEL_COLUMNS])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pooled
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            last = out.last_hidden_state
            if attention_mask is None:
                pooled = last.mean(dim=1)
            else:
                att = attention_mask.unsqueeze(-1).type_as(last)
                pooled = (last * att).sum(dim=1) / att.sum(dim=1).clamp(min=1e-9)
        x = self.proj(pooled); x = self.ln_proj(x); x = self.act(x); x = self.dropout(x)
        x = self.fc2(x); x = self.ln2(x); x = self.act(x); x = self.dropout(x)
        class_logits = torch.stack([h(x) for h in self.class_heads], dim=1)  # [B, L, K]
        reg_outs = torch.stack([h(x).squeeze(-1) for h in self.reg_heads], dim=1)  # [B, L]
        return {"class_logits": class_logits, "reg_outs": reg_outs}

# -------------------------
# Helpers: weights, sampler, loss
# -------------------------
def compute_class_weights(df, device='cpu'):
    weights = {}
    N = len(df)
    for lbl in LABEL_COLUMNS:
        counts = df[lbl].value_counts().reindex(range(K_CLASSES), fill_value=0).values.astype(float)
        inv = (N / (counts + 1e-6))
        # mạnh hoá: bình phương inverse frequency, sau đó normalize mean=1
        inv = (inv ** 2)
        inv = inv / inv.mean()
        weights[lbl] = torch.tensor(inv, dtype=torch.float, device=device)
    return weights

def make_weighted_sampler(df):
    N = len(df)
    weights = np.ones(N, dtype=float)
    for lbl in LABEL_COLUMNS:
        counts = df[lbl].value_counts().reindex(range(K_CLASSES), fill_value=0)
        freq_map = {cls: max(1, counts.get(cls, 0)) for cls in range(K_CLASSES)}
        col_scores = df[lbl].map(lambda x: N / (freq_map.get(int(x), 1))).values.astype(float)
        col_scores = col_scores / col_scores.mean()
        weights = np.maximum(weights, col_scores)
    weights = np.clip(weights, 0.1, 50.0)
    weights = weights / np.mean(weights)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def focal_ce_loss(logits, targets, gamma=0.0, weight=None):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')  # [B]
    if gamma == 0.0:
        return ce.mean()
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
    focal = ((1.0 - p_t) ** gamma) * ce
    return focal.mean()

# -------------------------
# Freeze last N layers helper
# -------------------------
def freeze_encoder_except_last(model, unfreeze_last=2):
    # model.encoder is a RobertaModel-like; detect number layers
    try:
        num_layers = model.encoder.config.num_hidden_layers
    except:
        num_layers = None
    for n, p in model.encoder.named_parameters():
        p.requires_grad = False
        if num_layers is not None:
            for i in range(num_layers - unfreeze_last, num_layers):
                if f'layer.{i}.' in n:
                    p.requires_grad = True
                    break
        # allow pooler and layernorm and embeddings to remain frozen to speed
        if 'pooler' in n or 'layernorm' in n.lower() or 'layer_norm' in n.lower():
            # keep frozen (could be left frozen)
            pass
    return model

# -------------------------
# Evaluate
# -------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds=[]; all_labels=[]; all_regs=[]
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['class_logits'].cpu().numpy()
            regs = out['reg_outs'].cpu().numpy()
            B,L,K = logits.shape
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds); all_regs.append(np.round(np.clip(regs,0,K_CLASSES-1)).astype(int)); all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = {'accuracy_per_label':{}, 'f1_macro_per_label':{}, 'mae_per_label':{}}
    for j,lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:,j], all_preds[:,j]))
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:,j], all_preds[:,j], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:,j], all_preds[:,j]))
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))
    return metrics

# -------------------------
# EMA
# -------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k] = (1.0 - self.decay)*v.detach().cpu() + self.decay*self.shadow[k]
    def copy_to(self, model):
        sd = model.state_dict()
        for k in sd.keys():
            if k in self.shadow:
                sd[k].copy_(self.shadow[k].to(sd[k].device))
        model.load_state_dict(sd)

# -------------------------
# Train loop with max_time cutoff
# -------------------------
def train_loop(df, model_name='vinai/phobert-base', out_dir='./out', epochs=6, bs=16, max_len=128,
               lr=2e-5, focal_gamma=0.0, mae_weight=0.3, use_sampler=True, device=None,
               unfreeze_last=2, max_time=1800, patience=2):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df, val_df = train_test_split(df, test_size=0.12, random_state=RANDOM_SEED)
    train_labels = train_df[LABEL_COLUMNS].values.astype(int)
    val_labels = val_df[LABEL_COLUMNS].values.astype(int)
    train_ds = MultiLabelTextDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length=max_len)
    val_ds = MultiLabelTextDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length=max_len)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=make_weighted_sampler(train_df)) if use_sampler else DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    class_weights = compute_class_weights(train_df, device=device)

    model = MultiTaskOrdinalModel(backbone_name=model_name, fc_dim=512, mid_dim=256, hidden_dropout=0.2)
    model = freeze_encoder_except_last(model, unfreeze_last=unfreeze_last)  # freeze many params
    # ensure heads trainable
    for n,p in model.named_parameters():
        if any(x in n for x in ['class_heads', 'reg_heads', 'proj', 'fc2']):
            p.requires_grad = True

    model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))
    ema = ModelEMA(model, decay=0.999)

    best_score = -1e9; no_imp = 0
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, epochs+1):
        model.train(); losses=[]
        pbar = tqdm(train_loader, desc=f"Ep{epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            # time cutoff
            if time.time() - start_time > max_time:
                print("Max time reached — stopping training loop.")
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda'))):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out['class_logits']  # [B,L,K]
                regs = out['reg_outs']  # [B,L]
                loss_ce_list=[]; loss_mae_list=[]
                for j,lbl in enumerate(LABEL_COLUMNS):
                    logit_j = logits[:, j, :]
                    tgt_j = labels[:, j]
                    w = class_weights[lbl] if class_weights is not None else None
                    loss_ce = focal_ce_loss(logit_j, tgt_j, gamma=focal_gamma, weight=w)
                    loss_ce_list.append(loss_ce)
                    loss_mae_list.append(F.l1_loss(regs[:,j], tgt_j.float()))
                loss = torch.stack(loss_ce_list).mean() + mae_weight * torch.stack(loss_mae_list).mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], 1.0 )
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            ema.update(model)
            losses.append(loss.item()); pbar.set_postfix({'loss':float(np.mean(losses))})
        # if time cutoff reached
        if time.time() - start_time > max_time:
            print("Stopping due to max_time.")
            break
        # validation with EMA
        backup = {k:v.clone() for k,v in model.state_dict().items()}
        ema.copy_to(model)
        val_metrics = evaluate_model(model, val_loader, device)
        model.load_state_dict(backup)
        print(f"Epoch {epoch} val: {val_metrics}")
        combined = val_metrics['f1_macro'] - 0.2 * val_metrics['mae']
        if combined > best_score:
            best_score = combined; no_imp = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print(f"Saved new best combined={combined:.4f}")
        else:
            no_imp += 1
            print(f"No improve {no_imp}/{patience}")
            if no_imp >= patience:
                print("Early stopping patience reached.")
                break

    print("Training finished. Best combined:", best_score)
    return model, tokenizer

# -------------------------
# I/O and main
# -------------------------
def load_and_prepare_data(path):
    df = pd.read_csv(path)
    text_cols = ['text','review','content','Review']
    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c; break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: 'text'})
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            df[lbl] = 0
    df['text'] = df['text'].astype(str).apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df[['text'] + LABEL_COLUMNS]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/mnt/data/train-problem.csv')
    parser.add_argument('--out', default='./out')
    parser.add_argument('--model', default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--focal_gamma', type=float, default=0.0)
    parser.add_argument('--mae_weight', type=float, default=0.3)
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--unfreeze_last', type=int, default=2)
    parser.add_argument('--max_time', type=int, default=1800)  # seconds
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    print("Loading data:", args.data)
    df = load_and_prepare_data(args.data)
    print("Shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    model, tokenizer = train_loop(df, model_name=args.model, out_dir=args.out, epochs=args.epochs,
                                  bs=args.bs, max_len=args.max_len, lr=args.lr,
                                  focal_gamma=args.focal_gamma, mae_weight=args.mae_weight,
                                  use_sampler=args.use_sampler, device=device,
                                  unfreeze_last=args.unfreeze_last, max_time=args.max_time,
                                  patience=args.patience)
    print("Done. Best model saved at", os.path.join(args.out, 'best.pt'))

if __name__ == "__main__":
    main()
