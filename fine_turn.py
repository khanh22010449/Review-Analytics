#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_model_multilabel_vn.py

Mục tiêu: tối ưu F1_macro và giảm MAE trên dữ liệu tiếng Việt (multi-label ordinal 0..K-1).
Thiết kế:
 - Backbone (configurable, default 'xlm-roberta-large' phù hợp RTX 4090)
 - After encoder: two FC layers with LayerNorm + Dropout
 - Heads per label: classification (K classes) + regression (single scalar)
 - Weighted CrossEntropy per label (class weights from train distribution)
 - Optional focal factor for multi-class (applied multiplicatively to CE loss)
 - Mixed precision, EMA, Gradient clipping, Scheduler
 - WeightedRandomSampler to upsample rare classes
"""

import os
import argparse
import random
import math
import time
import unicodedata
import regex as re
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup, AdamW
from tqdm import tqdm

# ----------------------
# Config & utilities
# ----------------------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # ordinal classes 0..5

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    s = re.sub(r"[^\\w\\s\\.,!?\\-:;'\"àáảãạăằắẵặâầấẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵñçươạêăđ0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------
# Dataset
# ----------------------
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels  # shape [N, L]
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

# ----------------------
# Model
# ----------------------
class MultiTaskOrdinalModel(nn.Module):
    def __init__(self,
                 backbone_name: str = "xlm-roberta-large",
                 hidden_dropout: float = 0.2,
                 fc_dim: int = 1024,
                 mid_dim: int = 768,
                 num_labels: int = len(LABEL_COLUMNS),
                 k_classes: int = K_CLASSES):
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size

        # projection and MLP with residual
        self.proj = nn.Linear(hidden_size, fc_dim)
        self.ln_proj = nn.LayerNorm(fc_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout)

        # second FC block
        self.fc2 = nn.Linear(fc_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)

        # Heads: classification and regression per label
        self.num_labels = num_labels
        self.k_classes = k_classes
        # classification heads: one linear mapping to K classes per label
        self.class_heads = nn.ModuleList([nn.Linear(mid_dim, k_classes) for _ in range(num_labels)])
        # regression heads: predict scalar (for MAE auxiliary)
        self.reg_heads = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in range(num_labels)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # get pooled representation
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # mean pooling by attention mask
            last = out.last_hidden_state  # [B, T, H]
            if attention_mask is None:
                pooled = last.mean(dim=1)
            else:
                attn = attention_mask.unsqueeze(-1).type_as(last)
                pooled = (last * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
        # MLP
        x = self.proj(pooled)
        x = self.ln_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.dropout(x)
        # heads
        class_logits = []
        reg_outs = []
        for i in range(self.num_labels):
            class_logits.append(self.class_heads[i](x))  # [B, K]
            reg_outs.append(self.reg_heads[i](x).squeeze(-1))  # [B]
        class_logits = torch.stack(class_logits, dim=1)  # [B, L, K]
        reg_outs = torch.stack(reg_outs, dim=1)  # [B, L]
        return {"class_logits": class_logits, "reg_outs": reg_outs}

# ----------------------
# Loss helpers
# ----------------------
def compute_class_weights(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, k: int = K_CLASSES, device='cpu'):
    # returns dict[label] = tensor(length K)
    weights = {}
    N = len(df)
    for lbl in labels:
        counts = df[lbl].value_counts().reindex(range(k), fill_value=0).values.astype(float)
        # avoid zero counts -> add eps
        eps = 1e-6
        # class weight inverse proportional to frequency
        inv = (N / (counts + eps))
        # normalize to mean 1
        inv = inv / inv.mean()
        w = torch.tensor(inv, dtype=torch.float, device=device)
        weights[lbl] = w
    return weights

def focal_ce_loss(logits, targets, gamma=0.0, weight=None, reduction='mean'):
    """
    Multi-class focal-like variant applied on CrossEntropy:
     - logits: [B, C]
     - targets: [B] (long)
     - weight: class weights tensor [C] or None
    When gamma==0 reduces to normal CrossEntropyLoss.
    """
    ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')  # [B]
    if gamma == 0.0:
        if reduction == 'mean':
            return ce.mean()
        elif reduction == 'sum':
            return ce.sum()
        else:
            return ce
    # compute p_t
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
    focal_factor = (1.0 - p_t) ** gamma
    loss = focal_factor * ce
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

# ----------------------
# Sampler helper
# ----------------------
def make_weighted_sampler(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, k: int = K_CLASSES):
    # compute sample weight based on rarity across labels: take max rarity across labels per sample
    N = len(df)
    weights = np.ones(N, dtype=float)
    for lbl in labels:
        counts = df[lbl].value_counts().reindex(range(k), fill_value=0)
        # rarity score = N / freq
        freq_map = {cls: max(1, counts.get(cls, 0)) for cls in range(k)}
        col_scores = df[lbl].map(lambda x: N / (freq_map.get(int(x), 1)))
        # scale to ~1 mean
        col_scores = col_scores.values.astype(float)
        col_scores = col_scores / col_scores.mean()
        weights = np.maximum(weights, col_scores)  # emphasize rare across any label
    # reduce extreme upsampling
    weights = np.clip(weights, 0.1, 50.0)
    weights = weights / np.mean(weights)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

# ----------------------
# Eval
# ----------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_reg_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['class_logits'].cpu().numpy()  # [B,L,K]
            regs = out['reg_outs'].cpu().numpy()  # [B,L]
            B,L,K = logits.shape
            preds = np.zeros((B,L), dtype=int)
            for i in range(B):
                for j in range(L):
                    p = logits[i,j,:]
                    preds[i,j] = int(np.argmax(p))
            all_preds.append(preds)
            all_reg_preds.append(np.round(np.clip(regs, 0, K_CLASSES-1)).astype(int))
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
    return metrics

# ----------------------
# Training loop
# ----------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = (1.0 - self.decay) * v.detach().cpu() + self.decay * self.shadow[k]

    def copy_to(self, model):
        sd = model.state_dict()
        for k in sd.keys():
            if k in self.shadow:
                sd[k].copy_(self.shadow[k].to(sd[k].device))
        model.load_state_dict(sd)

def train_loop(df,
               model_name="xlm-roberta-large",
               out_dir="./out_model",
               epochs=6,
               batch_size=8,
               max_len=256,
               lr=2e-5,
               weight_decay=1e-2,
               warmup_ratio=0.06,
               focal_gamma=0.0,
               mae_weight=0.3,
               use_sampler=True,
               device=None,
               grad_clip=1.0,
               gradient_accumulation_steps=1):
    set_seed()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df = df.copy()
    # split
    train_df, val_df = train_test_split(df, test_size=0.12, random_state=RANDOM_SEED)
    # label arrays
    train_labels = train_df[LABEL_COLUMNS].values.astype(int)
    val_labels = val_df[LABEL_COLUMNS].values.astype(int)
    # datasets
    train_ds = MultiLabelTextDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length=max_len)
    val_ds = MultiLabelTextDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length=max_len)
    # dataloaders
    if use_sampler:
        sampler = make_weighted_sampler(train_df, labels=LABEL_COLUMNS, k=K_CLASSES)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # compute class weights for CE per label
    class_weights = compute_class_weights(train_df, labels=LABEL_COLUMNS, k=K_CLASSES, device=device)

    # model
    model = MultiTaskOrdinalModel(backbone_name=model_name)
    model.to(device)

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs // max(1, gradient_accumulation_steps)
    num_warmup = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    ema = ModelEMA(model, decay=0.999)

    best_score = -1e9
    os.makedirs(out_dir, exist_ok=True)
    global_step = 0

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [B, L]
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out['class_logits']  # [B, L, K]
                regs = out['reg_outs']  # [B, L]
                # compute loss per label
                loss_ce_list = []
                loss_mae_list = []
                for j, lbl in enumerate(LABEL_COLUMNS):
                    logit_j = logits[:, j, :]  # [B, K]
                    tgt_j = labels[:, j]  # [B]
                    w = class_weights[lbl] if class_weights is not None else None
                    # CE or focal CE
                    loss_ce = focal_ce_loss(logit_j, tgt_j, gamma=focal_gamma, weight=w, reduction='mean')
                    loss_ce_list.append(loss_ce)
                    # regression MAE
                    pred_reg = regs[:, j]
                    loss_mae = F.l1_loss(pred_reg, tgt_j.float())
                    loss_mae_list.append(loss_mae)
                loss_ce_mean = torch.stack(loss_ce_list).mean()
                loss_mae_mean = torch.stack(loss_mae_list).mean()
                loss = loss_ce_mean + mae_weight * loss_mae_mean
            scaler.scale(loss).backward()
            # gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                ema.update(model)
            losses.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(losses))})
        # validation using EMA weights
        # backup state
        backup_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val_metrics = evaluate_model(model, val_loader, device)
        # restore original
        model.load_state_dict(backup_state)
        print(f"Epoch {epoch} Val metrics: {val_metrics}")
        # combined scoring: prioritize f1_macro then penalize mae
        combined = val_metrics['f1_macro'] - 0.2 * val_metrics['mae']
        if combined > best_score:
            best_score = combined
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"Saved best model at epoch {epoch} combined={combined:.4f}")
        else:
            print(f"No improvement. Best combined={best_score:.4f}")
    print("Training finished. Best combined:", best_score)
    return model, tokenizer

# ----------------------
# CLI & run
# ----------------------
def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    # detect text column
    text_cols = ['text', 'review', 'content', 'Review', df.columns[0]]
    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: 'text'})
    # ensure labels exist
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            df[lbl] = 0
    # clean text
    df['text'] = df['text'].astype(str).apply(clean_text)
    # cast labels
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df[['text'] + LABEL_COLUMNS]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train-problem.csv')
    parser.add_argument('--out', type=str, default='./out_model')
    parser.add_argument('--model', type=str, default='xlm-roberta-large')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--mae_weight', type=float, default=0.3)
    parser.add_argument('--focal_gamma', type=float, default=0.0)
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    args = parser.parse_args()

    print("Loading data:", args.data)
    df = load_and_prepare_data(args.data)
    print("Dataset shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model, tokenizer = train_loop(df,
                                 model_name=args.model,
                                 out_dir=args.out,
                                 epochs=args.epochs,
                                 batch_size=args.bs,
                                 max_len=args.max_len,
                                 lr=args.lr,
                                 mae_weight=args.mae_weight,
                                 focal_gamma=args.focal_gamma,
                                 use_sampler=args.use_sampler,
                                 device=device,
                                 gradient_accumulation_steps=args.accum)
    # final evaluation: load best
    best = os.path.join(args.out, "best_model.pt")
    if os.path.exists(best):
        print("Load best model for final eval...")
        model.load_state_dict(torch.load(best, map_location=device))
        # build val split again
        df_all = load_and_prepare_data(args.data)
        _, val_df = train_test_split(df_all, test_size=0.12, random_state=RANDOM_SEED)
        val_ds = MultiLabelTextDataset(val_df['text'].tolist(), val_df[LABEL_COLUMNS].values.astype(int), tokenizer, max_length=args.max_len)
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)
        metrics = evaluate_model(model, val_loader, device)
        print("Final evaluation:", metrics)
    else:
        print("No best model saved.")

if __name__ == "__main__":
    main()
