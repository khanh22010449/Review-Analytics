# nn_bt.py
"""
Multi-task multi-label + segment training with optional offline back-translation augmentation.
Upgrades included:
 - option use_concat_hidden: concat last 4 hidden states for richer pooled features
 - added per-aspect regressor head (predict continuous score 1..5) and regressor loss (MSE on present)
 - score_to_tensor utility for converting continuous scores -> soft distribution (for ensemble)
 - ensemble_predict_models: example helper to average logits and regressors from multiple checkpoints
 - device-safe numpy->tensor handling
 - BT can use GPU if available
How to use:
  python nn_bt.py --data_dir data --do_augment True
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

# ---------------- Config (defaults) ----------------
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
WEIGHT_DECAY = 0.01
FREEZE_ENCODER_EPOCHS = 1  # freeze encoder initially
# --------------------------------------------------

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
    text = re.sub(r"[^a-zA-Z0-9À-Ỹà-ỹ\s.,!?;:'\"%&$()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Dataset ----------
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len=256, label_cols=LABEL_COLS,
                 augment_fn=None, augment_prob=0.0, augment_minority_only=False, minority_map=None,
                 use_segmentation_fn=None):
        self.df = df.reset_index(drop=True)
        self.texts = self.df['Review'].astype(str).tolist()
        self.labels = self.df[label_cols].values  # 0..5
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_fn = augment_fn
        self.augment_prob = augment_prob
        self.augment_minority_only = augment_minority_only
        self.minority_map = minority_map or {}
        self.use_segmentation_fn = use_segmentation_fn

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        row_labels = self.labels[idx].astype(np.int64)  # 0..5

        if self.augment_fn and random.random() < self.augment_prob:
            if not self.augment_minority_only:
                text = self.augment_fn(text)
            else:
                do_aug = False
                for i, col in enumerate(LABEL_COLS):
                    val = row_labels[i]
                    if val > 0 and val-1 in self.minority_map.get(col, set()):
                        do_aug = True
                        break
                if do_aug:
                    text = self.augment_fn(text)

        text = clean_text(text)
        if self.use_segmentation_fn:
            try:
                text = self.use_segmentation_fn(text)
            except Exception:
                pass

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
        loss = self.ce(logits, target)  # per-sample
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            loss = loss * mask
        else:
            mask = torch.ones_like(loss)
        pt = torch.exp(-loss)
        focal = ((1 - pt) ** self.gamma) * loss
        if self.reduction == 'mean':
            denom = mask.sum().clamp_min(1.0)
            return focal.sum() / denom
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# ---------- Model ----------
class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=256, dropout=0.1, use_concat_hidden=True):
        super().__init__()
        # Ensure output_hidden_states if using concat
        if use_concat_hidden:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        else:
            self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        base_hidden = self.config.hidden_size
        # If we concat last 4 layers, effective hidden becomes 4*base_hidden
        self.use_concat_hidden = use_concat_hidden
        hidden = base_hidden * 4 if use_concat_hidden else base_hidden

        self.num_aspects = num_aspects
        self.num_seg_classes = num_seg_classes

        # segment heads: per-aspect classifier (C classes)
        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden_head),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head, num_seg_classes)
            ) for _ in range(num_aspects)
        ])
        # presence head: multi-label (num_aspects)
        self.presence_head = nn.Linear(hidden, num_aspects)

        # regressor head: predict continuous score (1..5) per aspect
        self.regressor_head = nn.Sequential(
            nn.Linear(hidden, hidden_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, num_aspects),  # output raw score (we'll bound it in forward)
            # do not apply activation here; we'll apply clamp in training/inference
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_concat_hidden and hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states
            # concat [last, -2, -3, -4] CLS tokens
            pooled = torch.cat([hs[-1][:,0,:], hs[-2][:,0,:], hs[-3][:,0,:], hs[-4][:,0,:]], dim=-1)
        else:
            pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:,0,:]

        pooled = self.dropout(pooled)
        seg_logits = torch.stack([head(pooled) for head in self.segment_heads], dim=1)  # [B,A,C]
        pres_logits = self.presence_head(pooled)  # raw logits (B,A)
        reg_raw = self.regressor_head(pooled)  # (B,A) continuous raw -> we'll clamp to [1,5] where appropriate
        # map raw to [1,5] via sigmoid*4 + 1 (stable)
        reg_scores = torch.sigmoid(reg_raw) * 4.0 + 1.0  # now in (1,5)
        return seg_logits, pres_logits, reg_scores

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

# ---------- score_to_tensor (convert continuous scores to soft distribution over 5 buckets) ----------
def score_to_tensor(score_np):
    """
    score_np: numpy array of shape (N, A) with scores in [1,5] (float)
    returns: numpy array shape (N, A, 5) representing soft distribution over buckets 0..4 (meaning original 1..5)
    This is robust to slight out-of-range values.
    """
    score = np.clip(score_np, 1.0, 5.0)
    # convert to 0..4 space
    s = score - 1.0
    N, A = s.shape
    tensor = np.zeros((N, A, 5), dtype=float)
    lower = np.floor(s).astype(int)
    upper = np.ceil(s).astype(int)
    frac = s - lower
    for i in range(N):
        for j in range(A):
            l = lower[i,j]
            u = upper[i,j]
            if l == u:
                tensor[i,j,l] = 1.0
            else:
                tensor[i,j,l] += 1.0 - frac[i,j]
                tensor[i,j,u] += frac[i,j]
    return tensor  # sums to 1 on last dim

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

def do_offline_back_translation(train_df, cache_dir=CACHE_DIR, mid_lang=AUG_TGT_LANG, bt_batch=BT_BATCH, device=None):
    """
    Train_df must have 'Review' column.
    If cached file exists return path; else run BT and cache to train_aug_bt.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_meta = Path(cache_dir) / "bt_meta.json"
    cached_csv = Path(cache_dir) / "train_aug_bt.csv"

    if cached_csv.exists():
        print("Found cached augmented file:", cached_csv)
        return str(cached_csv)

    print("Preparing BT models (this will download models if not present)...")
    model_pair = prepare_bt_models(src='vi', mid=mid_lang, cache_dir=cache_dir)

    texts = train_df['Review'].astype(str).tolist()
    print(f"Back-translating {len(texts)} samples in batches of {bt_batch} ...")
    bt_texts = back_translate_texts(texts, model_pair, batch_size=bt_batch, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if len(bt_texts) != len(texts):
        raise RuntimeError("Mismatch bt lengths")

    # filter identical outputs (sometimes BT returns same)
    aug_filtered = []
    for orig, bt in zip(texts, bt_texts):
        if bt is None or str(bt).strip() == "" or str(bt).strip() == str(orig).strip():
            aug_filtered.append(orig)  # fallback to original to keep shape
        else:
            aug_filtered.append(bt)

    df_aug = train_df.copy()
    df_aug['Review'] = aug_filtered
    df_concat = pd.concat([train_df, df_aug], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df_concat.to_csv(cached_csv, index=False)
    with open(cache_meta, "w") as f:
        json.dump({"src": "vi", "mid": mid_lang, "n_orig": len(train_df), "n_aug": len(df_aug)}, f)
    print("Saved augmented CSV to:", cached_csv)
    return str(cached_csv)

# ---------- Train / Validate / Predict ----------
def validate(val_loader, model, device='cpu'):
    model.eval()
    seg_logits_all, pres_logits_all, reg_scores_all, seg_t_all, pres_t_all = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Valid", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_labels = batch['seg_labels'].to(device)
            pres_labels = batch['pres_labels'].to(device)
            seg_logits, pres_logits, reg_scores = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_logits_all.append(seg_logits.cpu())
            pres_logits_all.append(pres_logits.cpu())
            reg_scores_all.append(reg_scores.cpu())
            seg_t_all.append(seg_labels.cpu())
            pres_t_all.append(pres_labels.cpu())
    seg_logits = torch.cat(seg_logits_all)
    pres_logits = torch.cat(pres_logits_all)
    reg_scores = torch.cat(reg_scores_all)
    seg_t = torch.cat(seg_t_all)
    pres_t = torch.cat(pres_t_all)
    metrics = compute_metrics_seg_and_pres(seg_logits, pres_logits, seg_t, pres_t)
    # optionally add reg metrics (MSE on present)
    mask = (seg_t != -1)
    if mask.sum() > 0:
        preds = reg_scores.cpu().numpy()
        targets = (seg_t.cpu().numpy() + 1.0)  # map 0..4 -> 1..5
        mse = ((preds - targets)**2 * mask.numpy()).sum() / (mask.sum() + 1e-12)
        metrics['reg_mse'] = float(mse)
    else:
        metrics['reg_mse'] = None
    return metrics

def train_loop_with_weights(train_loader, val_loader, model, optimizer, scheduler=None,
               num_epochs=3, device='cpu', w_seg=1.0, w_pres=1.0, w_reg=0.5,
               seg_loss_type='focal', per_aspect_weights=None, gamma=2.0,
               freeze_encoder_epochs=0, grad_accum_steps=1, fp16=True):
    model.to(device)
    scaler = GradScaler() if fp16 and device == 'cuda' else None
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss(reduction='sum')  # we'll average manually over present entries

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
                    seg_logits, pres_logits, reg_scores = model(input_ids=input_ids, attention_mask=attention_mask)
                    B,A,C = seg_logits.shape
                    loss_seg = 0.0
                    for a in range(A):
                        loss_a = per_aspect_loss[a](seg_logits[:,a,:], seg_labels[:,a])
                        loss_seg += loss_a
                    loss_seg = loss_seg / max(1, A)
                    loss_pres = bce_loss(pres_logits, pres_labels)

                    # regressor loss: compute only where seg_labels != -1 (present)
                    mask = (seg_labels != -1).float()  # B,A
                    if mask.sum() > 0:
                        # target for present entries is seg_labels + 1 (map 0..4->1..5)
                        target = (seg_labels.float() + 1.0) * mask
                        pred = reg_scores * mask
                        # MSE computed over present entries
                        reg_sum = ((pred - target)**2).sum()
                        reg_loss = reg_sum / (mask.sum() + 1e-12)
                    else:
                        reg_loss = torch.tensor(0.0, device=device)

                    loss = w_seg * loss_seg + w_pres * loss_pres + w_reg * reg_loss
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
                seg_logits, pres_logits, reg_scores = model(input_ids=input_ids, attention_mask=attention_mask)
                B,A,C = seg_logits.shape
                loss_seg = 0.0
                for a in range(A):
                    loss_a = per_aspect_loss[a](seg_logits[:,a,:], seg_labels[:,a])
                    loss_seg += loss_a
                loss_seg = loss_seg / max(1, A)
                loss_pres = bce_loss(pres_logits, pres_labels)
                mask = (seg_labels != -1).float()
                if mask.sum() > 0:
                    target = (seg_labels.float() + 1.0) * mask
                    pred = reg_scores * mask
                    reg_sum = ((pred - target)**2).sum()
                    reg_loss = reg_sum / (mask.sum() + 1e-12)
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                loss = w_seg * loss_seg + w_pres * loss_pres + w_reg * reg_loss
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
        score = val_metrics['pres_f1_micro'] + val_metrics['seg_f1_macro_mean'] - (val_metrics['reg_mse'] or 0.0) * 0.0
        if best_val is None or score > best_val:
            best_val = score
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict()
            }, "best_multitask_weighted.pt")
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
            seg_logits, pres_logits, reg_scores = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # 0..4
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            pres_preds = (pres_probs >= threshold).astype(int)
            reg_np = reg_scores.cpu().numpy()
            B = seg_preds.shape[0]
            for i in range(B):
                row = {}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i,j])
                    # choose final segmentation using regressor if available, else class pred
                    if present:
                        # convert reg score to nearest int 1..5
                        seg_from_reg = int(np.round(np.clip(reg_np[i,j], 1.0, 5.0)))
                        # also take classifier as fallback
                        seg_cls = int(seg_preds[i,j]) + 1
                        # choose average or prefer regressor? here prefer regressor but clamp
                        final_seg = max(1, min(5, seg_from_reg))
                    else:
                        final_seg = 0
                    row[col] = final_seg
                rows.append(row)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print("Saved predictions to", output_path)
    return output_path

# ---------- Ensemble helper ----------
def ensemble_predict_models(model_paths, tokenizer, test_df, output_csv="ensemble_preds.csv", device='cuda', batch_size=16, use_concat_hidden=True):
    """
    model_paths: list of checkpoint files corresponding to same architecture (MultiTaskTransformer with same flags)
    tokenizer: tokenizer used to build DataLoader
    test_df: dataframe with 'Review' column
    This function loads models, averages seg logits & presence probs, and averages regressors (converted to soft distributions).
    """
    test_ds = ReviewDataset(test_df, tokenizer, max_len=MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    models = []
    for p in model_paths:
        m = MultiTaskTransformer(MODEL_NAME, num_aspects=NUM_ASPECTS, num_seg_classes=NUM_SEGMENT_CLASSES, use_concat_hidden=use_concat_hidden)
        ck = torch.load(p, map_location='cpu')
        # ck may be dict with 'model_state'
        if isinstance(ck, dict) and 'model_state' in ck:
            m.load_state_dict(ck['model_state'])
        else:
            m.load_state_dict(ck)
        m.to(device)
        m.eval()
        models.append(m)

    all_rows = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="EnsemblePredict"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_accum = None
            pres_accum = None
            reg_distr_accum = None
            for m in models:
                seg_logits, pres_logits, reg_scores = m(input_ids=input_ids, attention_mask=attention_mask)
                seg_logits_np = seg_logits.cpu().numpy()
                pres_probs_np = torch.sigmoid(pres_logits).cpu().numpy()
                # convert reg_scores to distribution
                reg_np = reg_scores.cpu().numpy()  # (B,A)
                reg_distr = score_to_tensor(reg_np)  # (B,A,5)
                if seg_accum is None:
                    seg_accum = seg_logits_np
                    pres_accum = pres_probs_np
                    reg_distr_accum = reg_distr
                else:
                    seg_accum += seg_logits_np
                    pres_accum += pres_probs_np
                    reg_distr_accum += reg_distr
            # average
            seg_avg = seg_accum / len(models)
            pres_avg = pres_accum / len(models)
            reg_distr_avg = reg_distr_accum / len(models)  # (B,A,5)
            seg_preds = np.argmax(seg_avg, axis=-1)  # B,A
            pres_preds = (pres_avg >= 0.5).astype(int)
            reg_from_distr = np.argmax(reg_distr_avg, axis=-1) + 1  # map 0..4 -> 1..5

            B = seg_preds.shape[0]
            for i in range(B):
                row = {}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i,j])
                    seg = int(reg_from_distr[i,j]) if present else 0
                    row[col] = seg
                all_rows.append(row)
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_csv, index=False)
    print("Saved ensemble predictions to", output_csv)
    return output_csv

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--do_augment", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--bt_mid", type=str, default=AUG_TGT_LANG)
    parser.add_argument("--aug_cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--w_reg", type=float, default=0.5, help="weight for regressor loss")
    parser.add_argument("--use_concat_hidden", type=lambda x: x.lower() == "true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "problem_train.csv"
    val_csv = data_dir / "problem_val.csv"
    test_csv = data_dir / "problem_test.csv"

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    if args.do_augment:
        device_bt = "cuda" if torch.cuda.is_available() else "cpu"
        aug_path = do_offline_back_translation(train, cache_dir=args.aug_cache_dir, mid_lang=args.bt_mid, bt_batch=BT_BATCH, device=device_bt)
        train = pd.read_csv(aug_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # optional: pass a segmentation fn using VnCoreNLP if available
    use_seg_fn = None
    # from vncorenlp import VnCoreNLP
    # rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    # use_seg_fn = lambda t: " ".join(rdrsegmenter.tokenize(t)[0])

    train_ds = ReviewDataset(train, tokenizer, max_len=MAX_LEN, use_segmentation_fn=use_seg_fn)
    val_ds = ReviewDataset(val, tokenizer, max_len=MAX_LEN, use_segmentation_fn=use_seg_fn)
    test_ds = ReviewDataset(test, tokenizer, max_len=MAX_LEN, use_segmentation_fn=use_seg_fn)

    sampler = make_multi_aspect_sampler(train, LABEL_COLS, NUM_SEGMENT_CLASSES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = MultiTaskTransformer(MODEL_NAME, num_aspects=NUM_ASPECTS, num_seg_classes=NUM_SEGMENT_CLASSES, use_concat_hidden=args.use_concat_hidden)

    per_aspect_weights = compute_per_aspect_weights(train, LABEL_COLS, NUM_SEGMENT_CLASSES)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': model.segment_heads.parameters(), 'lr': LR_HEADS},
        {'params': model.presence_head.parameters(), 'lr': LR_HEADS},
        {'params': model.regressor_head.parameters(), 'lr': LR_HEADS}
    ], weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * NUM_EPOCHS // max(1, args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    seg_loss_type = 'focal' if USE_FOCAL else 'ce'
    train_loop_with_weights(train_loader, val_loader, model, optimizer, scheduler,
               num_epochs=NUM_EPOCHS, device=DEVICE, w_seg=1.5, w_pres=1.0,
               seg_loss_type=seg_loss_type, per_aspect_weights=per_aspect_weights, gamma=FGAMMA,
               freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS, grad_accum_steps=args.grad_accum, fp16=args.fp16,
               w_reg=args.w_reg)

    # Evaluate on test
    test_metrics = validate(test_loader, model, device=DEVICE)
    print("Test metrics:", test_metrics)

    # Save predictions
    predict_and_save(model, test_loader, output_path="data/predictions.csv", device=DEVICE)

if __name__ == "__main__":
    main()
