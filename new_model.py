# nn_bt_fixed.py
"""
Multi-task multi-label + segment training with optional offline back-translation augmentation.
Features:
 - Enhanced architecture with Attention Pooling + Adapters + per-aspect heads
 - Offline back-translation cache
 - Focal loss with ignore_index support
 - Weighted sampler to mitigate class imbalance
 - AMP training support
Usage:
  python nn_bt_fixed.py --data_dir data --do_augment True
"""

import os
import argparse
import random
import math
import json
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import (AutoTokenizer, AutoModel, AutoConfig,
                          get_linear_schedule_with_warmup,
                          MarianMTModel, MarianTokenizer)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# ---------------- Config ----------------
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
BATCH_SIZE = 16
LR_ENCODER = 1e-5
LR_HEADS = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEGMENT_CLASSES = 5
USE_FOCAL = True
FGAMMA = 1.5
CACHE_DIR = "bt_cache"
AUG_TGT_LANG = "en"   # Vi -> En -> Vi back-translation
BT_BATCH = 16         # batch size for translation generation
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 1  # increase to simulate larger batch
WEIGHT_DECAY = 0.025
FREEZE_ENCODER_EPOCHS = 1  # freeze encoder initially
# Enhanced model defaults
ENH_HIDDEN_HEAD = 512
ENH_POOL_HEADS = 8
ENH_ADAPTER_BOTTLENECK = 128
# ----------------------------------------

# ---------- Utilities ----------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ---------- Text cleaning ----------
import re, html
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = html.unescape(text)
    # keep Vietnamese letters, basic punctuation
    text = re.sub(r"[^a-zA-Z0-9À-Ỹà-ỹ\s.,!?;:'\"%&$()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Dataset ----------
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len=256, label_cols=LABEL_COLS,
                 augment_fn=None, augment_prob=0.0, augment_minority_only=False, minority_map=None):
        self.df = df.reset_index(drop=True)
        self.texts = self.df['Review'].astype(str).tolist()
        self.labels = self.df[label_cols].values  # 0..5
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_fn = augment_fn
        self.augment_prob = augment_prob
        self.augment_minority_only = augment_minority_only
        self.minority_map = minority_map or {}  # {col: set(classes_to_aug)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        row_labels = self.labels[idx].astype(np.int64)  # 0..5

        if self.augment_fn and random.random() < self.augment_prob:
            if not self.augment_minority_only:
                text = self.augment_fn(text)
            else:
                # only augment if any aspect belongs to minority classes (mapped 0..4)
                do_aug = False
                for i, col in enumerate(LABEL_COLS):
                    val = row_labels[i]
                    if val > 0 and val-1 in self.minority_map.get(col, set()):
                        do_aug = True
                        break
                if do_aug:
                    text = self.augment_fn(text)

        text = clean_text(text)
        tok = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        seg_labels_raw = row_labels
        pres_labels = (seg_labels_raw > 0).astype(np.float32)
        seg_labels = np.where(seg_labels_raw > 0, seg_labels_raw - 1, -1)

        return {
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0),
            'seg_labels': torch.tensor(seg_labels, dtype=torch.long),
            'pres_labels': torch.tensor(pres_labels, dtype=torch.float)
        }

def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    seg_labels = torch.stack([x['seg_labels'] for x in batch])
    pres_labels = torch.stack([x['pres_labels'] for x in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'seg_labels': seg_labels, 'pres_labels': pres_labels}

# ---------- Focal Loss (with ignore_index) ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, target):
        # logits: [B, C], target: [B]
        loss = self.ce(logits, target)  # shape [B]
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            loss = loss * mask
        else:
            mask = torch.ones_like(loss)
        # compute pt as exp(-CE) per-sample
        pt = torch.exp(-loss)
        focal = ((1 - pt) ** self.gamma) * loss
        if self.reduction == 'mean':
            denom = mask.sum().clamp_min(1.0)
            return focal.sum() / denom
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# ---------- Enhanced Model: Attention Pool, Adapter, per-aspect heads ----------
class AttentionPool(nn.Module):
    """
    Attention pooling: learnable queries (num_queries = num_aspects) attend over token embeddings.
    Input: token_embeddings: [B, T, H]
    Output: pooled: [B, num_queries, H]
    """
    def __init__(self, hidden_size, num_queries, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden = hidden_size
        self.num_heads = num_heads
        # learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        # multihead attention: queries (num_queries), keys/values from tokens
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
    def forward(self, token_emb, attn_mask=None):
        # token_emb: [B, T, H]
        B, T, H = token_emb.shape
        # expand queries to batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, H]
        # keys/values = token_emb
        attn_output, attn_weights = self.mha(query=q, key=token_emb, value=token_emb, key_padding_mask=None, attn_mask=None)
        out = self.ln(attn_output + q)  # residual
        out = out + self.ff(out)  # feed-forward residual
        return out  # [B, Q, H]

class Adapter(nn.Module):
    """
    small residual adapter MLP for task-specific tuning
    """
    def __init__(self, hidden, bottleneck=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: [B, H]
        res = x
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        x = self.dropout(x)
        return self.ln(x + res)

class EnhancedMultiTaskTransformer(nn.Module):
    """
    Enhanced multi-task model using attention pooling and adapters.
    Outputs:
      seg_logits: [B, A, C]
      pres_logits: [B, A]
    """
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=ENH_HIDDEN_HEAD, pool_heads=ENH_POOL_HEADS, dropout=0.2, adapter_bottleneck=ENH_ADAPTER_BOTTLENECK):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.num_aspects = num_aspects
        self.num_seg_classes = num_seg_classes

        # attention pooling: produce per-aspect vectors
        self.attn_pool = AttentionPool(hidden_size=hidden, num_queries=num_aspects, num_heads=pool_heads, dropout=dropout)

        # per-aspect adapters
        self.adapters = nn.ModuleList([Adapter(hidden, bottleneck=adapter_bottleneck, dropout=dropout) for _ in range(num_aspects)])

        # per-aspect segment heads (MLP)
        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden_head),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head, hidden_head // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head // 2, num_seg_classes)
            ) for _ in range(num_aspects)
        ])

        # presence heads: per-aspect small MLP -> single logit
        self.presence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, max(64, hidden_head // 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(64, hidden_head // 4), 1)
            ) for _ in range(num_aspects)
        ])

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.adapters) + list(self.segment_heads) + list(self.presence_heads):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state  # [B, T, H]
        token_emb = self.dropout(token_emb)

        pooled = self.attn_pool(token_emb)  # [B, A, H]

        seg_logits_per_aspect = []
        pres_logits_per_aspect = []
        for a in range(self.num_aspects):
            vec = pooled[:, a, :]
            vec = self.adapters[a](vec)
            seg_logit = self.segment_heads[a](vec)  # [B, C]
            pres_logit = self.presence_heads[a](vec).squeeze(-1)  # [B]
            seg_logits_per_aspect.append(seg_logit)
            pres_logits_per_aspect.append(pres_logit)

        seg_logits = torch.stack(seg_logits_per_aspect, dim=1)  # [B, A, C]
        pres_logits = torch.stack(pres_logits_per_aspect, dim=1)  # [B, A]
        return seg_logits, pres_logits

# ---------- Helpers: weights & sampler ----------
def compute_per_aspect_weights(train_df, label_cols, num_seg_classes):
    weights_list = []
    for col in label_cols:
        vals = train_df[col].values
        vals_pos = vals[vals > 0]
        if len(vals_pos) == 0:
            weights_list.append(None)
            continue
        mapped = (vals_pos - 1).astype(int)
        classes = np.arange(num_seg_classes)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=mapped)
        cw = torch.tensor(cw, dtype=torch.float)
        weights_list.append(cw)
    return weights_list

def make_multi_aspect_sampler(df, label_cols, num_seg_classes):
    # create sample weight = max(1/class_freq_for_each_aspect) across aspects, ignore -1 with small weight
    n = len(df)
    sample_weights = np.zeros(n, dtype=float)
    for col in label_cols:
        vals = df[col].values
        mapped = np.where(vals > 0, vals-1, -1)
        counts = {}
        for c in range(num_seg_classes):
            counts[c] = np.sum(mapped == c)
        col_weights = np.array([0.1 if v==-1 else 1.0/(counts[int(v)]+1e-9) for v in mapped])
        sample_weights = np.maximum(sample_weights, col_weights)
    sample_weights = sample_weights / (sample_weights.mean() + 1e-12)
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=n, replacement=True)

# ---------- Metrics ----------
def compute_metrics_seg_and_pres(seg_logits, pres_logits, seg_targets, pres_targets, threshold=0.5):
    seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()
    seg_true = seg_targets.cpu().numpy()
    pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
    pres_preds = (pres_probs >= threshold).astype(int)
    pres_true = pres_targets.cpu().numpy().astype(int)

    metrics = {}
    accs, f1s_seg = [], []
    for a in range(seg_true.shape[1]):
        mask = seg_true[:, a] != -1
        if np.sum(mask) == 0:
            continue
        accs.append(accuracy_score(seg_true[mask,a], seg_preds[mask,a]))
        f1s_seg.append(f1_score(seg_true[mask,a], seg_preds[mask,a], average='macro', zero_division=0))
    metrics['seg_acc_mean'] = float(np.mean(accs) if accs else 0.0)
    metrics['seg_f1_macro_mean'] = float(np.mean(f1s_seg) if f1s_seg else 0.0)
    metrics['pres_f1_micro'] = float(f1_score(pres_true.reshape(-1), pres_preds.reshape(-1), average='micro', zero_division=0))
    metrics['pres_f1_macro'] = float(f1_score(pres_true.reshape(-1), pres_preds.reshape(-1), average='macro', zero_division=0))
    return metrics

# ---------- Back-translation (offline, batched) ----------
def prepare_bt_models(src='vi', mid='en', cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    name_1 = f'Helsinki-NLP/opus-mt-{src}-{mid}'
    name_2 = f'Helsinki-NLP/opus-mt-{mid}-{src}'
    tok1 = MarianTokenizer.from_pretrained(name_1, cache_dir=cache_dir)
    m1 = MarianMTModel.from_pretrained(name_1, cache_dir=cache_dir)
    tok2 = MarianTokenizer.from_pretrained(name_2, cache_dir=cache_dir)
    m2 = MarianMTModel.from_pretrained(name_2, cache_dir=cache_dir)
    return (tok1, m1), (tok2, m2)

def back_translate_texts(texts, model_pair, batch_size=16, device=None):
    (tok1, m1), (tok2, m2) = model_pair
    if device:
        m1.to(device); m2.to(device)
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tok1(batch_texts, return_tensors="pt", padding=True, truncation=True).to(m1.device)
        with torch.no_grad():
            out = m1.generate(**inputs, num_beams=4, max_length=256)
            mids = tok1.batch_decode(out, skip_special_tokens=True)
        inputs2 = tok2(mids, return_tensors="pt", padding=True, truncation=True).to(m2.device)
        with torch.no_grad():
            out2 = m2.generate(**inputs2, num_beams=4, max_length=256)
            backs = tok2.batch_decode(out2, skip_special_tokens=True)
        results.extend(backs)
    return results

def do_offline_back_translation(train_df, cache_dir=CACHE_DIR, mid_lang=AUG_TGT_LANG, bt_batch=BT_BATCH):
    """
    Train_df must have 'Review' column.
    This function will:
      - If cached file exists return path
      - Else run BT for all train reviews and create train_aug.csv (original + augmented)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_meta = Path(cache_dir) / "bt_meta.json"
    cached_csv = Path(cache_dir) / "train_aug_bt.csv"

    # if cache exists return quickly
    if cached_csv.exists():
        print("Found cached augmented file:", cached_csv)
        return str(cached_csv)

    print("Preparing BT models (this will download models if not present)...")
    model_pair = prepare_bt_models(src='vi', mid=mid_lang, cache_dir=cache_dir)

    texts = train_df['Review'].astype(str).tolist()
    print(f"Back-translating {len(texts)} samples in batches of {bt_batch} ...")
    bt_texts = back_translate_texts(texts, model_pair, batch_size=bt_batch, device='cpu')  # use cpu by default for compatibility
    if len(bt_texts) != len(texts):
        raise RuntimeError("Mismatch bt lengths")

    # Create augmented DataFrame (one augmented per original). You can increase to more copies if desired.
    df_aug = train_df.copy()
    df_aug['Review'] = bt_texts
    df_concat = pd.concat([train_df, df_aug], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df_concat.to_csv(cached_csv, index=False)
    # save meta
    with open(cache_meta, "w") as f:
        json.dump({"src": "vi", "mid": mid_lang, "n_orig": len(train_df), "n_aug": len(df_aug)}, f)
    print("Saved augmented CSV to:", cached_csv)
    return str(cached_csv)

# ---------- Train / Validate / Predict ----------
def validate(val_loader, model, device='cpu'):
    model.eval()
    seg_logits_all, pres_logits_all, seg_t_all, pres_t_all = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Valid", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_labels = batch['seg_labels'].to(device)
            pres_labels = batch['pres_labels'].to(device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_logits_all.append(seg_logits.cpu())
            pres_logits_all.append(pres_logits.cpu())
            seg_t_all.append(seg_labels.cpu())
            pres_t_all.append(pres_labels.cpu())
    seg_logits = torch.cat(seg_logits_all)
    pres_logits = torch.cat(pres_logits_all)
    seg_t = torch.cat(seg_t_all)
    pres_t = torch.cat(pres_t_all)
    return compute_metrics_seg_and_pres(seg_logits, pres_logits, seg_t, pres_t)

def train_loop_with_weights(train_loader, val_loader, model, optimizer, scheduler=None,
               num_epochs=3, device='cpu', w_seg=1.0, w_pres=1.0,
               seg_loss_type='focal', per_aspect_weights=None, gamma=2.0,
               freeze_encoder_epochs=0, grad_accum_steps=1, fp16=True):
    model.to(device)
    scaler = GradScaler() if fp16 and device == 'cuda' else None
    bce_loss = nn.BCEWithLogitsLoss()
    # Setup loss per-aspect
    per_aspect_loss = []
    for w in per_aspect_weights:
        w_t = w.to(device) if w is not None else None
        if seg_loss_type == 'focal':
            per_aspect_loss.append(FocalLoss(gamma=gamma, weight=w_t, ignore_index=-1))
        else:
            per_aspect_loss.append(nn.CrossEntropyLoss(weight=w_t, ignore_index=-1))

    best_val = None
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        if epoch < freeze_encoder_epochs:
            # freeze encoder
            for p in model.encoder.parameters():
                p.requires_grad = False
            print(f"Epoch {epoch+1}: encoder frozen")
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_labels = batch['seg_labels'].to(device)
            pres_labels = batch['pres_labels'].to(device)

            if scaler:
                with autocast():
                    seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    B,A,C = seg_logits.shape
                    loss_seg = 0.0
                    for a in range(A):
                        loss_a = per_aspect_loss[a](seg_logits[:,a,:], seg_labels[:,a])
                        loss_seg += loss_a
                    loss_seg = loss_seg / max(1, A)
                    loss_pres = bce_loss(pres_logits, pres_labels)
                    loss = w_seg * loss_seg + w_pres * loss_pres
                scaler.scale(loss).backward()
                if (step+1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                B,A,C = seg_logits.shape
                loss_seg = 0.0
                for a in range(A):
                    loss_a = per_aspect_loss[a](seg_logits[:,a,:], seg_labels[:,a])
                    loss_seg += loss_a
                loss_seg = loss_seg / max(1, A)
                loss_pres = bce_loss(pres_logits, pres_labels)
                loss = w_seg * loss_seg + w_pres * loss_pres
                loss.backward()
                if (step+1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            total_loss += float(loss.detach().cpu().item())
            pbar.set_postfix({'loss': total_loss / (step+1)})

            global_step += 1

        val_metrics = validate(val_loader, model, device)
        print(f"Epoch {epoch+1} val metrics:", val_metrics)

        # selection metric (can be tuned)
        score = val_metrics['pres_f1_micro'] + val_metrics['seg_f1_macro_mean']
        if best_val is None or score > best_val:
            best_val = score
            torch.save(model.state_dict(), "best_multitask_weighted.pt")
            print("Saved best model.")

    print("Training finished. Best score:", best_val)

# ---------- Predict helper ----------
def predict_and_save(model, data_loader, output_path="predictions.csv", device='cpu', threshold=0.5):
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predict"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # 0..4
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            pres_preds = (pres_probs >= threshold).astype(int)
            B = seg_preds.shape[0]
            for i in range(B):
                row = {}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i,j])
                    seg = int(seg_preds[i,j]) + 1 if present else 0  # map back to 0..5 original scale
                    row[col] = seg
                rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print("Saved predictions to", output_path)
    return output_path

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--do_augment", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--bt_mid", type=str, default=AUG_TGT_LANG)
    parser.add_argument("--aug_cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--hidden_head", type=int, default=ENH_HIDDEN_HEAD)
    parser.add_argument("--adapter_bottleneck", type=int, default=ENH_ADAPTER_BOTTLENECK)
    parser.add_argument("--pool_heads", type=int, default=ENH_POOL_HEADS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "problem_train.csv"
    val_csv = data_dir / "problem_val.csv"
    test_csv = data_dir / "problem_test.csv"

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    if args.do_augment:
        aug_path = do_offline_back_translation(train, cache_dir=args.aug_cache_dir, mid_lang=args.bt_mid, bt_batch=BT_BATCH)
        train = pd.read_csv(aug_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train, tokenizer, max_len=MAX_LEN)
    val_ds = ReviewDataset(val, tokenizer, max_len=MAX_LEN)
    test_ds = ReviewDataset(test, tokenizer, max_len=MAX_LEN)

    # Optional: sampler to balance batches
    sampler = make_multi_aspect_sampler(train, LABEL_COLS, NUM_SEGMENT_CLASSES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = EnhancedMultiTaskTransformer(MODEL_NAME, num_aspects=NUM_ASPECTS, num_seg_classes=NUM_SEGMENT_CLASSES,
                                         hidden_head=args.hidden_head, pool_heads=args.pool_heads,
                                         dropout=0.2, adapter_bottleneck=args.adapter_bottleneck)

    per_aspect_weights = compute_per_aspect_weights(train, LABEL_COLS, NUM_SEGMENT_CLASSES)

    # per-parameter LR: encoder vs heads/adapters/presence
    encoder_params = list(model.encoder.parameters())
    head_params = []
    # collect heads: segment_heads + presence_heads + adapters
    for p in model.segment_heads.parameters():
        head_params.append(p)
    for p in model.presence_heads.parameters():
        head_params.append(p)
    for p in model.adapters.parameters():
        head_params.append(p)

    # flatten head_params into param groups
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEADS}
    ], weight_decay=WEIGHT_DECAY)

    total_steps = max(1, len(train_loader) * NUM_EPOCHS // max(1, args.grad_accum))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    seg_loss_type = 'focal' if USE_FOCAL else 'ce'
    train_loop_with_weights(train_loader, val_loader, model, optimizer, scheduler,
               num_epochs=NUM_EPOCHS, device=DEVICE, w_seg=1.5, w_pres=1.0,
               seg_loss_type=seg_loss_type, per_aspect_weights=per_aspect_weights, gamma=FGAMMA,
               freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS, grad_accum_steps=args.grad_accum, fp16=args.fp16)

    # Evaluate on test
    test_metrics = validate(test_loader, model, device=DEVICE)
    print("Test metrics:", test_metrics)

    # Save predictions
    predict_and_save(model, test_loader, output_path=str(data_dir/"predictions.csv"), device=DEVICE)

if __name__ == "__main__":
    main()
