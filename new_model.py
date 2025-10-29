# new_model_fixed.py
# Sửa đổi tích hợp: drop-all-zero, CB-weights, sum-weight sampler, CB+Focal loss, masked loss, checkpoint seg_f1_macro_mean
# Yêu cầu: transformers, torch, sklearn, pandas, numpy
# TUNE: chỉnh đường dẫn file, model_name, hyperparams ở phần CONFIG

import os
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, f1_score

# ----------------------------
# CONFIG (TUNE ở đây)
# ----------------------------
TRAIN_CSV = "problem_train.csv"
VAL_CSV = "problem_val.csv"
TEST_CSV = "problem_test.csv"
MODEL_NAME = "bert-base-multilingual-cased"  # hoặc 'vinai/phobert-base' nếu tiếng Việt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'van_chuyen', 'mua_sam']
NUM_SEGMENT_CLASSES = 5  # 1..5 -> mapped 0..4
MAX_LEN = 128
BATCH_SIZE = 16  # TUNE
EPOCHS = 6
LR_ENCODER = 1e-5
LR_HEADS = 2e-4
WEIGHT_DECAY = 1e-2
FGAMMA = 2.0  # focal gamma (TUNE)
CB_BETA = 0.9999  # effective number beta (TUNE)
AUGMENT_MINORITY_ONLY = True
AUGMENT_OFFLINE = True  # nếu True, tạo augmentation offline và append vào train df
N_AUG_PER_SAMPLE = 2  # số augment cho mỗi sample minority khi offline
SEED = 42
SAVE_DIR = "checkpoints_fixed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ----------------------------
# Data preprocessing helpers
# ----------------------------
def drop_all_zero_samples(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Giữ lại các mẫu có ít nhất 1 label != 0"""
    mask = (df[label_cols] != 0).any(axis=1)
    df_clean = df[mask].copy().reset_index(drop=True)
    return df_clean

def map_labels_1to5_to_0to4(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Chuyển 1..5 -> 0..4; giữ NaN cho 0/missing"""
    df2 = df.copy()
    for c in label_cols:
        df2[c] = df2[c].replace({0: pd.NA})
        df2[c] = df2[c].astype('float')  # allow NaN
        df2.loc[df2[c].notna(), c] = df2.loc[df2[c].notna(), c].astype(int) - 1
    return df2

# ----------------------------
# Class-Balanced weights (effective number)
# ----------------------------
def compute_cb_weights_per_aspect(df: pd.DataFrame, label_cols: List[str], num_classes: int, beta: float = CB_BETA, device='cpu'):
    """Return list of torch tensors (num_classes) for each aspect; normalized mean=1"""
    weights_list = []
    for col in label_cols:
        vals = df[col].dropna().astype(int).values  # mapped 0..4
        if len(vals) == 0:
            weights_list.append(torch.ones(num_classes, device=device))
            continue
        counts = np.array([np.sum(vals == c) for c in range(num_classes)], dtype=np.float32)
        counts = np.maximum(counts, 1.0)  # avoid zero
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-12)
        weights = weights / np.mean(weights)
        weights_list.append(torch.tensor(weights, dtype=torch.float32, device=device))
    return weights_list

# ----------------------------
# Sampler: sample weight = sum inverse-freq across aspects
# ----------------------------
def make_sample_weights_sum(df: pd.DataFrame, label_cols: List[str], num_classes: int = NUM_SEGMENT_CLASSES, eps=1e-6):
    per_col_counts = {}
    for col in label_cols:
        vals = df[col].dropna().astype(int).values
        mapped = vals
        counts = np.array([np.sum(mapped == c) for c in range(num_classes)], dtype=float)
        counts = np.maximum(counts, 1.0)
        per_col_counts[col] = counts

    sample_weights = np.zeros(len(df), dtype=float)
    for i, row in enumerate(df[label_cols].itertuples(index=False)):
        w_sum = 0.0
        for j, col in enumerate(label_cols):
            val = getattr(row, col)
            if pd.notna(val):
                cls = int(val)
                w = 1.0 / (per_col_counts[col][cls] + eps)
            else:
                w = 0.05
            w_sum += w
        sample_weights[i] = w_sum

    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-12)
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    return sampler

# ----------------------------
# Simple augmentation (placeholder)
# Bạn có thể thay bằng back-translation hoặc LLM generation offline.
# ----------------------------
def simple_synonym_augment(text: str):
    # placeholder nhẹ: shuffling words (không nên dùng cho production)
    words = text.split()
    if len(words) <= 3:
        return text
    i, j = sorted(random.sample(range(len(words)), 2))
    words[i], words[j] = words[j], words[i]
    return " ".join(words)

def augment_minority_offline(df: pd.DataFrame, label_cols: List[str], minority_map: Dict[str, set], n_aug: int = N_AUG_PER_SAMPLE):
    """Generate augmented samples offline for rows that contain minority classes (minority_map maps aspect -> set of class indices)"""
    new_rows = []
    for idx, row in df.iterrows():
        add = False
        for col in label_cols:
            val = row[col]
            if pd.notna(val) and int(val) in minority_map.get(col, set()):
                add = True
                break
        if add:
            for k in range(n_aug):
                new_row = row.copy()
                new_row['Review'] = simple_synonym_augment(str(row['Review']))
                new_rows.append(new_row)
    if len(new_rows) == 0:
        return df
    aug_df = pd.DataFrame(new_rows)
    df_aug = pd.concat([df, aug_df], ignore_index=True).reset_index(drop=True)
    return df_aug

def build_minority_map(df: pd.DataFrame, label_cols: List[str], num_classes: int, quantile: float = 0.5):
    """Return dict aspect -> set(class indices considered minority)"""
    mm = {}
    for col in label_cols:
        vals = df[col].dropna().astype(int).values
        if len(vals) == 0:
            mm[col] = set()
            continue
        counts = np.array([np.sum(vals == c) for c in range(num_classes)], dtype=float)
        q = np.quantile(counts, quantile)
        minority = set([i for i, c in enumerate(counts) if c <= q])
        mm[col] = minority
    return mm

# ----------------------------
# Dataset
# ----------------------------
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, label_cols: List[str], max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_cols = label_cols
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = str(row['Review'])
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        labels = []
        for c in self.label_cols:
            v = row[c]
            if pd.isna(v):
                labels.append(-1)
            else:
                labels.append(int(v))
        labels = torch.tensor(labels, dtype=torch.long)
        enc['labels'] = labels
        return enc

# ----------------------------
# Model: Shared encoder + per-aspect heads
# ----------------------------
class MultiTaskAspectModel(nn.Module):
    def __init__(self, model_name: str, label_cols: List[str], num_classes=NUM_SEGMENT_CLASSES, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size
        self.label_cols = label_cols
        self.num_classes = num_classes
        # create a head per aspect
        self.heads = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(enc_dim, enc_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(enc_dim//2, num_classes)
            ) for col in label_cols
        })

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        # use pooled output if available, else mean pool last_hidden_state
        pooled = getattr(outputs, 'pooler_output', None)
        if pooled is None:
            last = outputs.last_hidden_state  # (B, L, D)
            pooled = last.mean(dim=1)
        logits = {}
        for col in self.label_cols:
            logits[col] = self.heads[col](pooled)
        return logits

# ----------------------------
# Loss: Focal + Class Balanced via sample weighting for CE
# ----------------------------
def focal_ce_with_cb(logits: torch.Tensor, targets: torch.Tensor, cb_weights: torch.Tensor, gamma: float = FGAMMA, alpha: float = 1.0):
    """
    logits: (N,C)
    targets: (N,) with 0..C-1
    cb_weights: (C,)
    """
    ce = F.cross_entropy(logits, targets, reduction='none')  # (N,)
    # approx pt
    pt = torch.exp(-ce)
    sample_w = cb_weights[targets]  # (N,)
    loss = alpha * sample_w * ((1 - pt) ** gamma) * ce
    return loss.mean()

# ----------------------------
# Evaluation metrics
# ----------------------------
def eval_model(model: nn.Module, dataloader: DataLoader, label_cols: List[str], device=DEVICE):
    model.eval()
    all_probs = {c: [] for c in label_cols}
    all_preds = {c: [] for c in label_cols}
    all_trues = {c: [] for c in label_cols}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # (B, num_aspects)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            B = input_ids.size(0)
            for i, col in enumerate(label_cols):
                logit = logits[col]  # (B, C)
                probs = torch.softmax(logit, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()
                trues = labels[:, i].cpu().numpy()
                for t, p, pr in zip(trues, preds, probs.cpu().numpy()):
                    if t == -1:
                        continue
                    all_trues[col].append(int(t))
                    all_preds[col].append(int(p))
                    all_probs[col].append(pr)  # vector, not used except for AP if needed
    # compute per-aspect metrics
    per_aspect_f1 = {}
    for col in label_cols:
        if len(all_trues[col]) == 0:
            per_aspect_f1[col] = 0.0
        else:
            f1s = precision_recall_fscore_support(all_trues[col], all_preds[col], average=None, labels=list(range(NUM_SEGMENT_CLASSES)))
            # macro f1:
            per_aspect_f1[col] = f1_score(all_trues[col], all_preds[col], average='macro', zero_division=0)
    # seg_f1_macro_mean:
    seg_f1_macro_mean = np.mean(list(per_aspect_f1.values()))
    return {'per_aspect_f1': per_aspect_f1, 'seg_f1_macro_mean': seg_f1_macro_mean}

# ----------------------------
# Training loop
# ----------------------------
def train_loop(model: nn.Module, optimizer, scheduler, train_loader: DataLoader, val_loader: DataLoader,
               cb_weights_per_aspect: List[torch.Tensor], label_cols: List[str],
               device=DEVICE, epochs=EPOCHS, save_dir=SAVE_DIR):
    best_score = -1.0
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # (B, num_aspects) values in -1 or 0..4
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = 0.0
            cnt = 0
            for i, col in enumerate(label_cols):
                target = labels[:, i]  # (B,)
                mask = (target >= 0)
                if mask.sum() == 0:
                    continue
                logit = logits[col][mask]  # (M,C)
                targ = target[mask].long()  # (M,)
                loss_i = focal_ce_with_cb(logit, targ, cb_weights_per_aspect[i], gamma=FGAMMA)
                loss += loss_i
                cnt += 1
            if cnt > 0:
                loss = loss / cnt
            else:
                # nothing to learn in this batch (shouldn't happen often)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            n_batch += 1
        avg_loss = total_loss / max(1, n_batch)
        # validation
        val_metrics = eval_model(model, val_loader, label_cols, device=device)
        score = val_metrics.get('seg_f1_macro_mean', 0.0)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} val_seg_f1_macro_mean={score:.4f}")
        # save best
        if score > best_score:
            best_score = score
            save_path = os.path.join(save_dir, f"best_model_epoch{epoch}_segf1_{score:.4f}.pt")
            torch.save(model.state_dict(), save_path)
            print("Saved best model ->", save_path)
    return best_score

# ----------------------------
# Main runner
# ----------------------------
def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)

    # 1) drop all-zero samples (you asked to remove 0 which means not mentioned)
    train = drop_all_zero_samples(train, LABEL_COLS)
    val = drop_all_zero_samples(val, LABEL_COLS)
    test = drop_all_zero_samples(test, LABEL_COLS)
    print("After drop-all-zero -> Train:", train.shape, "Val:", val.shape, "Test:", test.shape)

    # 2) map 1..5 -> 0..4; keep NaN for missing
    train = map_labels_1to5_to_0to4(train, LABEL_COLS)
    val = map_labels_1to5_to_0to4(val, LABEL_COLS)
    test = map_labels_1to5_to_0to4(test, LABEL_COLS)

    # 3) build minority map & optionally augment offline (only minority)
    minority_map = build_minority_map(train, LABEL_COLS, NUM_SEGMENT_CLASSES, quantile=0.5)
    print("Minority map (per aspect):", minority_map)
    if AUGMENT_OFFLINE:
        print("Offline augment minority samples (this may increase dataset size)...")
        train = augment_minority_offline(train, LABEL_COLS, minority_map, n_aug=N_AUG_PER_SAMPLE)
        print("After augmentation train shape:", train.shape)

    # 4) compute class-balanced weights per aspect
    cb_weights = compute_cb_weights_per_aspect(train, LABEL_COLS, NUM_SEGMENT_CLASSES, beta=CB_BETA, device=DEVICE)
    for col, w in zip(LABEL_COLS, cb_weights):
        print(f"CB weights for {col}:", w.cpu().numpy())

    # 5) build sampler
    sampler = make_sample_weights_sum(train, LABEL_COLS, NUM_SEGMENT_CLASSES)

    # 6) tokenizer, dataset, dataloaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train, tokenizer, LABEL_COLS, max_len=MAX_LEN)
    val_ds = ReviewDataset(val, tokenizer, LABEL_COLS, max_len=MAX_LEN)
    test_ds = ReviewDataset(test, tokenizer, LABEL_COLS, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 7) model init
    model = MultiTaskAspectModel(MODEL_NAME, LABEL_COLS, num_classes=NUM_SEGMENT_CLASSES)
    model.to(DEVICE)

    # 8) optimizer + scheduler (different lrs for encoder and heads)
    # group params: encoder lower lr, heads higher lr
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEADS}
    ], weight_decay=WEIGHT_DECAY)
    total_steps = int(len(train_loader) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 9) train
    print("Start training...")
    best = train_loop(model, optimizer, scheduler, train_loader, val_loader, cb_weights, LABEL_COLS, device=DEVICE, epochs=EPOCHS, save_dir=SAVE_DIR)
    print("Training done. Best seg_f1_macro_mean:", best)

    # 10) load best and eval test (pick latest saved)
    saved = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".pt")])
    if len(saved) > 0:
        path = os.path.join(SAVE_DIR, saved[-1])
        print("Loading best model", path)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        test_metrics = eval_model(model, test_loader, LABEL_COLS, device=DEVICE)
        print("Test seg_f1_macro_mean:", test_metrics['seg_f1_macro_mean'])
        print("Per-aspect f1:", test_metrics['per_aspect_f1'])
    else:
        print("No saved model found.")

if __name__ == "__main__":
    main()
