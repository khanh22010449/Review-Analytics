#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_turn.py - Extended with embedding+MLP ensemble pipeline
Modes:
 - finetune: full/partial fine-tune pipeline (original)
 - emb_mlp: precompute embeddings -> train ensemble MLPs -> stack -> bias-tune -> eval
Author: updated for hackathon rapid improvement
"""
import os
import time
import argparse
import random
import unicodedata
import regex as re
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
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
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    pattern = r"[^\p{L}\p{N}\s\.,!?:;'\-\(\)\"/]+"
    s = re.sub(pattern, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compute_overall_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Return micro_f1, sentiment_acc, overall
    sentiment_acc = accuracy for positions where y_true > 0
    overall = 0.7 * micro_f1 + 0.3 * sentiment_acc
    """
    micro = float(f1_score(y_true.flatten(), y_pred.flatten(), average='micro', zero_division=0))
    mask = (y_true > 0)
    if mask.sum() == 0:
        sent_acc = 0.0
    else:
        sent_acc = float(((y_pred == y_true) & mask).sum() / mask.sum())
    overall = 0.7 * micro + 0.3 * sent_acc
    return micro, sent_acc, overall

# ---------- Dataset & collate ----------
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

def collate_fn(batch):
    if len(batch) == 0:
        return {}
    first = batch[0]
    coll = {}
    for key in first.keys():
        vals = [d.get(key, None) for d in batch]
        if key in ('input_ids', 'attention_mask', 'labels'):
            for i, v in enumerate(vals):
                if v is None:
                    raise ValueError(f"Found None for required field {key} in batch index {i}.")
            coll[key] = torch.stack(vals, dim=0)
        else:
            coll[key] = vals
    return coll

# ---------- Finetune model ----------
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
        x = self.ln1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.ln2(x); x = self.act(x); x = self.drop(x)
        logits = torch.stack([h(x) for h in self.class_heads], dim=1)  # [B,L,K]
        regs = torch.stack([h(x).squeeze(-1) for h in self.reg_heads], dim=1)  # [B,L]
        return {'logits': logits, 'regs': regs}

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

# ---------- Evaluation helpers ----------
def evaluate_comp(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['logits'].cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds); all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    micro, sent_acc, overall = compute_overall_from_preds(all_labels, all_preds)
    return {'micro_f1': micro, 'sentiment_acc': sent_acc, 'overall': overall, 'preds': all_preds, 'labels': all_labels}

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

# ---------- Finetune train loop (kept) ----------
def finetune_train_loop(df: pd.DataFrame,
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
    # freeze encoder if needed
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
    for n, p in model.named_parameters():
        if 'class_heads' in n or 'reg_heads' in n or 'proj' in n or 'fc2' in n or 'ln' in n:
            p.requires_grad = True

    model.to(device)

    # optimizer param groups
    head_params, other_params = [], []
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

    for epoch in tqdm(range(1, epochs + 1), desc="Finetune Epochs"):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(pbar):
            if time.time() - start_time > max_time:
                print("Max time reached — stopping training loop.")
                break
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

        # validation
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        metrics = evaluate_comp(model, val_loader, device)
        model.load_state_dict(backup)
        micro_f1 = metrics['micro_f1']; sentiment_acc = metrics['sentiment_acc']; overall = metrics['overall']
        print(f"Epoch {epoch} val -> Micro-F1: {micro_f1:.4f}, SentimentAcc: {sentiment_acc:.4f}, Overall: {overall:.4f}")

        if inspect_flag:
            print("Inspecting some validation samples:")
            inspect_val_samples(model, val_loader, device, n=6)

        if overall > best_score:
            best_score = overall
            no_imp = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_finetune.pt'))
            print(f"Saved best finetune model (Overall={overall:.4f})")
        else:
            no_imp += 1
            print(f"No improve {no_imp}/{patience}")
            if no_imp >= patience:
                print("Early stopping triggered.")
                break

    print("Finetune finished. Best Overall:", best_score)
    return model, tokenizer

# ---------- Embedding precompute + MLP ensemble pipeline ----------
def save_embeddings(df: pd.DataFrame, model_name: str, out_path: str, batch: int = 128, max_len: int = 128, device: Optional[str] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    texts = df['text'].astype(str).tolist()
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch), desc='Embedding'):
            batch_texts = texts[i:i+batch]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # [B,T,H]
            attn = enc['attention_mask'].unsqueeze(-1).type_as(last)
            mean = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            maxv, _ = (last * attn).max(dim=1)
            emb = torch.cat([mean, maxv], dim=1)  # [B, 2H]
            embs.append(emb.cpu().numpy())
    embs = np.vstack(embs)
    np.savez_compressed(out_path, emb=embs)
    return embs

class SimpleMLPEnsemble(nn.Module):
    def __init__(self, input_dim, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden//2, K_CLASSES) for _ in LABEL_COLUMNS])

    def forward(self, x):
        h = self.net(x)
        outs = [head(h) for head in self.heads]  # list of [B,K]
        return torch.stack(outs, dim=1)  # [B,L,K]

def train_single_mlp(X_train, y_train, X_val, y_val, input_dim, hidden, seed=0, epochs=15, bs=256, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    model = SimpleMLPEnsemble(input_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    tr = DataLoader(train_ds, batch_size=bs, shuffle=True)
    va = DataLoader(val_ds, batch_size=bs, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    best = None
    best_score = -1
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # [B,L,K]
            loss = 0
            for j in range(len(LABEL_COLUMNS)):
                loss += criterion(logits[:, j, :], yb[:, j])
            loss = loss / len(LABEL_COLUMNS)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        probs_list = []
        with torch.no_grad():
            for xb, yb in va:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()  # [B,L,K]
                probs_list.append(out)
        probs = np.vstack(probs_list)
        preds = np.argmax(probs, axis=-1)
        micro = float(f1_score(y_val.flatten(), preds.flatten(), average='micro', zero_division=0))
        if micro > best_score:
            best_score = micro
            best = model.state_dict()
            torch.save(best, f'best_mlp.pt')
    model.load_state_dict(best)
    return model, best_score

def train_ensemble_and_stack(embeddings: np.ndarray, labels: np.ndarray, n_models: int = 3,
                             hidden_sizes: List[int] = [512, 384, 256],
                             epochs:int=15, batch_size:int=256, lr:float=1e-3, device:Optional[str]=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.12, random_state=RANDOM_SEED)
    input_dim = X_train.shape[1]
    mlp_models = []
    val_probs = []  # list of [N_val, L, K]
    for i in tqdm(range(n_models), desc= "Training MLP ensemble"):
        hidden = hidden_sizes[i % len(hidden_sizes)]
        m, score = train_single_mlp(X_train, y_train, X_val, y_val, input_dim, hidden=hidden, seed=RANDOM_SEED + i,
                                    epochs=epochs, bs=batch_size, lr=lr, device=device)
        mlp_models.append(m)
        # get val probs
        m.eval()
        with torch.no_grad():
            out = []
            for idx in range(0, X_val.shape[0], batch_size):
                xb = torch.from_numpy(X_val[idx:idx+batch_size]).float().to(device)
                p = m(xb).cpu().numpy()  # [b,L,K] raw logits
                probs = np.exp(p - np.max(p, axis=-1, keepdims=True))
                probs = probs / probs.sum(axis=-1, keepdims=True)
                out.append(probs)
            out = np.vstack(out)  # [N_val, L, K]
            val_probs.append(out)
    # stacking: for each label j, train LR on concatenated probs for that label
    val_preds = np.zeros_like(y_val)
    stackers = []
    for j in range(len(LABEL_COLUMNS)):
        feats = np.concatenate([p[:, j, :] for p in val_probs], axis=1)  # [N_val, K * n_models]
        lr = LogisticRegression(max_iter=500)
        lr.fit(feats, y_val[:, j])
        stackers.append(lr)
        pred_j = lr.predict(feats)
        val_preds[:, j] = pred_j
    micro, sent_acc, overall = compute_overall_from_preds(y_val, val_preds)
    return {
        'mlp_models': mlp_models,
        'stackers': stackers,
        'X_val': X_val, 'y_val': y_val,
        'val_preds': val_preds,
        'val_probs': val_probs,
        'scores': (micro, sent_acc, overall)
    }

def tune_bias_on_logits(logits: np.ndarray, y_val: np.ndarray, biases: List[float] = [-2,-1,-0.5,0,0.5,1,2]):
    """
    logits: [N, L, K] raw logits
    For each label j, add bias to classes 1..K-1 (non-zero classes) and pick bias that maximizes Overall.
    """
    N,L,K = logits.shape
    best_bias = [0.0] * L
    best_overall = -1.0
    # We'll tune per label independently (greedy)
    current_logits = logits.copy()
    for j in range(L):
        bestb = 0.0
        best_local = -1.0
        for b in biases:
            tmp = current_logits.copy()
            tmp[:, j, 1:] = tmp[:, j, 1:] + b
            preds = np.argmax(tmp, axis=-1)
            micro, sent_acc, overall = compute_overall_from_preds(y_val, preds)
            if overall > best_local:
                best_local = overall; bestb = b
        # apply bestb to current_logits
        current_logits[:, j, 1:] += bestb
        best_bias[j] = bestb
    # final
    final_preds = np.argmax(current_logits, axis=-1)
    micro, sent_acc, overall = compute_overall_from_preds(y_val, final_preds)
    return best_bias, (micro, sent_acc, overall), final_preds

# ---------- Pipeline orchestrator ----------
def emb_mlp_pipeline(df: pd.DataFrame, model_name='vinai/phobert-base', out_dir='./out',
                     emb_path='/mnt/data/emb_meanmax.npz', force_recompute=False,
                     n_models:int=3, hidden_sizes:List[int]=[512,384,256],
                     epochs:int=12, batch_size:int=256, lr:float=1e-3, device:Optional[str]=None):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    # compute or load embeddings
    if (not force_recompute) and os.path.exists(emb_path):
        z = np.load(emb_path); embs = z['emb']
    else:
        embs = save_embeddings(df, model_name, emb_path, batch=128, max_len=128, device=device)
    labels = df[LABEL_COLUMNS].values.astype(int)
    # train ensemble and stack
    print("Training MLP ensemble and stacking...")
    res = train_ensemble_and_stack(embs, labels, n_models=n_models, hidden_sizes=hidden_sizes, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    micro, sent_acc, overall = res['scores']
    print(f"Ensemble val -> Micro-F1: {micro:.4f}, SentAcc: {sent_acc:.4f}, Overall: {overall:.4f}")
    # try bias tuning on stacked logits (we can obtain stacked logits by concatenating model logits)
    # Build stacked logits on val: for each model compute logits on X_val
    X_val = res['X_val']; y_val = res['y_val']; mlp_models = res['mlp_models']; stackers = res['stackers']
    all_logits = []
    for m in mlp_models:
        m.eval()
        cur = []
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                xb = torch.from_numpy(X_val[i:i+batch_size]).float().to(device)
                out = m(xb).cpu().numpy()  # [b,L,K] raw logits
                cur.append(out)
        cur = np.vstack(cur)
        all_logits.append(cur)
    # convert per-model logits to per-model probs
    all_probs = [np.exp(l - np.max(l, axis=-1, keepdims=True)) / np.sum(np.exp(l - np.max(l, axis=-1, keepdims=True)), axis=-1, keepdims=True) for l in all_logits]
    # create stacked logits by concatenating probs then mapping via stackers to logits (use decision function)
    stacked_logits = np.zeros((X_val.shape[0], len(LABEL_COLUMNS), K_CLASSES), dtype=float)
    for j in range(len(LABEL_COLUMNS)):
        feats = np.concatenate([p[:, j, :] for p in all_probs], axis=1)  # [N_val, K * n_models]
        lr = stackers[j]
        # Prefer predict_proba (gives probabilities for classes lr.classes_)
        if hasattr(lr, 'predict_proba'):
            prob = lr.predict_proba(feats)  # shape (N, n_classes)
            classes = lr.classes_
            cur = np.zeros((X_val.shape[0], K_CLASSES), dtype=float)
            for idx_cls, cls in enumerate(classes):
                cls_idx = int(cls)
                if 0 <= cls_idx < K_CLASSES:
                    cur[:, cls_idx] = prob[:, idx_cls]
            # convert probabilities to logit-like scores (log prob) for stacking
            eps = 1e-12
            cur_logits = np.log(cur + eps)
            stacked_logits[:, j, :] = cur_logits
        else:
            # fallback to decision_function
            decision = lr.decision_function(feats)
            if decision.ndim == 1:
                # binary case -> decision is shape (N,), form two-class scores [-d, d]
                two = np.vstack([-decision, decision]).T  # shape (N,2)
                classes = lr.classes_
                cur = np.zeros((X_val.shape[0], K_CLASSES), dtype=float)
                for idx_cls, cls in enumerate(classes):
                    cls_idx = int(cls)
                    if 0 <= cls_idx < K_CLASSES:
                        cur[:, cls_idx] = two[:, idx_cls]
                stacked_logits[:, j, :] = cur
            else:
                # multiclass decision_function -> shape (N, n_classes)
                classes = lr.classes_
                cur = np.zeros((X_val.shape[0], K_CLASSES), dtype=float)
                for idx_cls, cls in enumerate(classes):
                    cls_idx = int(cls)
                    if 0 <= cls_idx < K_CLASSES:
                        cur[:, cls_idx] = decision[:, idx_cls]
                stacked_logits[:, j, :] = cur
                
    # bias tune
    print("Tuning bias on stacked logits...")
    biases, scores, final_preds = tune_bias_on_logits(stacked_logits.copy(), y_val, biases=[-2,-1,-0.5,0,0.5,1,2])
    print("Biases per label:", biases)
    print("After bias tuning -> Micro-F1: %.4f SentAcc: %.4f Overall: %.4f" % scores)
    # Save ensemble artifacts
    np.savez_compressed(os.path.join(out_dir, 'emb_stack_artifacts.npz'),
                        val_preds=res['val_preds'], val_probs=np.array(res['val_probs'], dtype=object),
                        y_val=y_val)
    return {'ensemble_scores': res['scores'], 'bias_scores': scores, 'biases': biases}

# ---------- I/O helpers ----------
def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    text_cols = ['text', 'review', 'content', 'Review']
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
    df['text'] = df['text'].astype(str).fillna("").apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df[['text'] + LABEL_COLUMNS]

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train-problem.csv')
    parser.add_argument('--out', type=str, default='./out')
    parser.add_argument('--mode', type=str, default='finetune', choices=['finetune', 'emb_mlp'])
    # finetune args
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
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--classification_only', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    # emb_mlp args
    parser.add_argument('--emb_path', type=str, default='/mnt/data/emb_meanmax.npz')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--n_models', type=int, default=3)
    parser.add_argument('--mlp_epochs', type=int, default=12)
    parser.add_argument('--mlp_bs', type=int, default=256)
    parser.add_argument('--mlp_lr', type=float, default=1e-3)
    args = parser.parse_args()

    print("Loading data:", args.data)
    df = load_and_prepare_data(args.data)
    print("Dataset shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    if args.mode == 'finetune':
        finetune_train_loop(df,
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
    else:
        # embedding + mlp ensemble
        res = emb_mlp_pipeline(df,
                               model_name=args.model,
                               out_dir=args.out,
                               emb_path=args.emb_path,
                               force_recompute=args.force_recompute,
                               n_models=args.n_models,
                               hidden_sizes=[512,384,256],
                               epochs=args.mlp_epochs,
                               batch_size=args.mlp_bs,
                               lr=args.mlp_lr,
                               device=device)
        print("Ensemble result summary:", res)

if __name__ == '__main__':
    main()
