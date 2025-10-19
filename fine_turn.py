#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_turn.py - Phiên bản chỉnh sửa với collate_fn bảo vệ khỏi NoneType trong DataLoader
- Giữ các cải tiến: backbone, MLP, focal loss, sampler, lr_head/lr_encoder, classification_only,...
- Thêm collate_fn tùy chỉnh để tránh lỗi default_collate khi có trường raw_text hoặc None.
"""
import os
import time
import argparse
import random
import unicodedata
import regex as re
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ---------- Config labels ----------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # 0..5

# ---------- Utils ----------
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    """Sửa lỗi regex safe, giữ unicode letters & numbers và ký tự câu cơ bản."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    # Escape '-' and include parentheses/quotes explicitly
    pattern = r"[^\p{L}\p{N}\s\.,!?:;'\-\(\)\"/]+"
    s = re.sub(pattern, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Dataset ----------
class MultiLabelTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: AutoTokenizer, max_length: int = 128, return_text: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_text = return_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw = self.texts[idx]
        txt = clean_text(raw)
        enc = self.tokenizer(txt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.return_text:
            item['raw_text'] = raw
        return item

# ---------- collate_fn ----------
def collate_fn(batch):
    """
    Collate that:
     - stacks torch.Tensor fields (input_ids, attention_mask, labels)
     - keeps other fields (like raw_text) as list (strings)
    Also checks required fields not None.
    """
    if len(batch) == 0:
        return {}
    first = batch[0]
    coll = {}
    for key in first.keys():
        vals = [d.get(key, None) for d in batch]
        # sanity: required tensor fields must not contain None
        if key in ('input_ids', 'attention_mask', 'labels'):
            for i, v in enumerate(vals):
                if v is None:
                    raise ValueError(f"Found None for required field {key} in batch index {i}.")
            # all should be tensors -> stack
            if isinstance(vals[0], torch.Tensor):
                coll[key] = torch.stack(vals, dim=0)
            else:
                # handle numpy arrays
                try:
                    coll[key] = torch.tensor(np.stack(vals, axis=0))
                except Exception:
                    # fallback: keep list
                    coll[key] = vals
        else:
            # keep list as-is (strings or None)
            coll[key] = vals
    return coll

# ---------- Model ----------
class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name='vinai/phobert-base', proj_dim=512, mid_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(proj_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)
        self.class_heads = nn.ModuleList([nn.Linear(mid_dim, K_CLASSES) for _ in LABEL_COLUMNS])
        self.reg_heads = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in LABEL_COLUMNS])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            last = out.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last) if attention_mask is not None else None
            if attn is None:
                pooled = last.mean(dim=1)
            else:
                pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        x = self.proj(pooled)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.drop(x)
        logits = torch.stack([h(x) for h in self.class_heads], dim=1)  # [B, L, K]
        regs = torch.stack([h(x).squeeze(-1) for h in self.reg_heads], dim=1)  # [B, L]
        return {'logits': logits, 'regs': regs}

# ---------- Loss & helpers ----------
class FocalCELoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None):
        ce = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        if self.gamma == 0.0:
            return ce.mean()
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        loss = ((1 - p_t) ** self.gamma) * ce
        return loss.mean()

def compute_class_weights_per_label(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, k: int = K_CLASSES, device='cpu'):
    weights = {}
    N = len(df)
    for lbl in labels:
        counts = df[lbl].value_counts().reindex(range(k), fill_value=0).values.astype(float)
        inv = (N / (counts + 1e-6))
        inv = inv ** 1.2
        inv = inv / inv.mean()
        weights[lbl] = torch.tensor(inv, dtype=torch.float, device=device)
    return weights

def make_weighted_sampler(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, k: int = K_CLASSES):
    N = len(df)
    weights = np.ones(N, dtype=float)
    for lbl in labels:
        counts = df[lbl].value_counts().reindex(range(k), fill_value=0)
        freq_map = {cls: max(1, counts.get(cls, 0)) for cls in range(k)}
        col_scores = df[lbl].map(lambda x: (N / (freq_map.get(int(x), 1)))).values.astype(float)
        col_scores = col_scores / col_scores.mean()
        weights = np.maximum(weights, col_scores)
    weights = np.clip(weights, 0.1, 50.0)
    weights = weights / np.mean(weights)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# ---------- Evaluation ----------
def evaluate_comp(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            # collate_fn may keep raw_text as list; input tensors stacked
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['logits'].cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    micro_f1 = float(f1_score(all_labels.flatten(), all_preds.flatten(), average='micro', zero_division=0))
    mask = (all_labels > 0)
    if mask.sum() == 0:
        sentiment_acc = 0.0
    else:
        correct = ((all_preds == all_labels) & mask).sum()
        sentiment_acc = float(correct / mask.sum())
    return {'micro_f1': micro_f1, 'sentiment_acc': sentiment_acc}

def inspect_val_samples(model, val_loader, device, n=10):
    model.eval()
    printed = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            raws = batch.get('raw_text', [None] * input_ids.size(0))
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['logits'].cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            B = preds.shape[0]
            for i in range(B):
                text = raws[i] if raws and raws[i] is not None else "<no-text>"
                print("TEXT:", text)
                print("GT:", labels[i].cpu().numpy())
                print("PRED:", preds[i])
                print("-" * 60)
                printed += 1
                if printed >= n:
                    return

# ---------- EMA ----------
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

# ---------- Train loop ----------
def train_loop(df: pd.DataFrame,
               model_name: str = 'vinai/phobert-base',
               out_dir: str = './out',
               epochs: int = 6,
               bs: int = 16,
               max_len: int = 128,
               lr_head: float = 1e-3,
               lr_encoder: float = 1e-5,
               focal_gamma: float = 1.0,
               mae_weight: float = 0.1,
               use_sampler: bool = True,
               unfreeze_last: int = 0,
               max_time: int = 1800,
               patience: int = 2,
               classification_only: bool = False,
               inspect_flag: bool = False,
               device: Optional[str] = None):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # split
    train_df, val_df = train_test_split(df, test_size=0.12, random_state=RANDOM_SEED)
    train_labels = train_df[LABEL_COLUMNS].values.astype(int)
    val_labels = val_df[LABEL_COLUMNS].values.astype(int)

    train_ds = MultiLabelTextDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length=max_len, return_text=False)
    val_ds = MultiLabelTextDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length=max_len, return_text=True)

    if use_sampler:
        sampler = make_weighted_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weights_per_label(train_df, device=device)

    model = MultiTaskModel(backbone_name=model_name)
    # freeze encoder except last N layers
    try:
        num_layers = model.encoder.config.num_hidden_layers
    except:
        num_layers = None
    if num_layers is not None:
        for n, p in model.encoder.named_parameters():
            p.requires_grad = False
            if unfreeze_last > 0:
                for i in range(num_layers - unfreeze_last, num_layers):
                    if f'layer.{i}.' in n:
                        p.requires_grad = True
                        break
    # ensure heads trainable
    for n, p in model.named_parameters():
        if 'class_heads' in n or 'reg_heads' in n or 'proj' in n or 'fc2' in n or 'ln' in n:
            p.requires_grad = True

    model.to(device)

    # optimizer param groups: head_params (fast LR) vs other_params (slow LR)
    head_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(x in n for x in ['class_heads', 'reg_heads', 'proj', 'fc2', 'ln']):
            head_params.append(p)
        else:
            other_params.append(p)

    optim_groups = []
    if head_params:
        optim_groups.append({'params': head_params, 'lr': lr_head})
    if other_params:
        optim_groups.append({'params': other_params, 'lr': lr_encoder})

    optimizer = AdamW(optim_groups, weight_decay=1e-2)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda')))
    ema = ModelEMA(model, decay=0.999)
    loss_cls_fn = FocalCELoss(gamma=focal_gamma)

    best_score = -1e9
    no_imp = 0
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(pbar):
            if time.time() - start_time > max_time:
                print("Max time reached — stopping training loop.")
                break
            # required fields checked in collate_fn
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda' if device.startswith('cuda') else None):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out['logits']   # [B,L,K]
                regs = out['regs']       # [B,L]
                loss_list = []
                for j, lbl in enumerate(LABEL_COLUMNS):
                    logit_j = logits[:, j, :]
                    tgt_j = labels[:, j]
                    weight = class_weights[lbl] if class_weights is not None else None
                    loss_cls = loss_cls_fn(logit_j, tgt_j, weight=weight)
                    if classification_only:
                        loss_list.append(loss_cls)
                    else:
                        loss_reg = F.l1_loss(regs[:, j], tgt_j.float())
                        loss_list.append(loss_cls + mae_weight * loss_reg)
                loss = torch.stack(loss_list).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for g in optim_groups for p in g['params']], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            losses.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(losses))})

        if time.time() - start_time > max_time:
            print("Stopping due to max_time.")
            break

        # validation with EMA weights
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        metrics = evaluate_comp(model, val_loader, device)
        model.load_state_dict(backup)

        micro_f1 = metrics['micro_f1']
        sentiment_acc = metrics['sentiment_acc']
        overall = 0.7 * micro_f1 + 0.3 * sentiment_acc

        print(f"Epoch {epoch} val -> Micro-F1: {micro_f1:.4f}, SentimentAcc: {sentiment_acc:.4f}, Overall: {overall:.4f}")

        if inspect_flag:
            print("Inspecting some validation samples:")
            inspect_val_samples(model, val_loader, device, n=6)

        if overall > best_score:
            best_score = overall
            no_imp = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print(f"Saved best model (Overall={overall:.4f})")
        else:
            no_imp += 1
            print(f"No improve {no_imp}/{patience}")
            if no_imp >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished. Best Overall:", best_score)
    return model, tokenizer

# ---------- I/O helpers ----------
def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    text_cols = ['text', 'review', 'content', 'Review']
    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: 'text'})
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            df[lbl] = 0
    # convert to str before cleaning to avoid None
    df['text'] = df['text'].astype(str).fillna("").apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df[['text'] + LABEL_COLUMNS]

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train-problem.csv')
    parser.add_argument('--out', type=str, default='./out')
    parser.add_argument('--model', type=str, default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    parser.add_argument('--focal_gamma', type=float, default=1.0)
    parser.add_argument('--mae_weight', type=float, default=0.1)
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--unfreeze_last', type=int, default=0)
    parser.add_argument('--max_time', type=int, default=1800)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--classification_only', action='store_true')
    parser.add_argument('--inspect', action='store_true', help='print sample predictions after each validation')
    args = parser.parse_args()

    print("Loading data:", args.data)
    df = load_and_prepare_data(args.data)
    print("Dataset shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    model, tokenizer = train_loop(df,
                                 model_name=args.model,
                                 out_dir=args.out,
                                 epochs=args.epochs,
                                 bs=args.bs,
                                 max_len=args.max_len,
                                 lr_head=args.lr_head,
                                 lr_encoder=args.lr_encoder,
                                 focal_gamma=args.focal_gamma,
                                 mae_weight=args.mae_weight,
                                 use_sampler=args.use_sampler,
                                 unfreeze_last=args.unfreeze_last,
                                 max_time=args.max_time,
                                 patience=args.patience,
                                 classification_only=args.classification_only,
                                 inspect_flag=args.inspect,
                                 device=device)
    print("Done. Best model saved at", os.path.join(args.out, 'best.pt'))

if __name__ == '__main__':
    main()
