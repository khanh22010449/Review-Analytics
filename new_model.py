# pseudo_labeling_safe.py
"""
Safe pseudo-labeling pipeline.
- Robust train loop (avoid NaN when many ignore targets)
- Per-aspect / soft selection rules for pseudo labels
- Limit pseudo added per round
Usage example:
  python pseudo_labeling_safe.py --data_dir data --unlabeled data/problem_test.csv --rounds 3 \
    --batch 16 --epochs_per_round 1 --pres_thres 0.90 --seg_thres 0.90 --max_add_per_round 500
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ---------------- Config ----------------
MODEL_NAME = "vinai/phobert-base"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEG_CLASSES = 5
# ----------------------------------------

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_everything(42)

# ---------- Utils ----------
import re, html
from bs4 import BeautifulSoup
def clean_text(t):
    return re.sub(r"\s+"," ", BeautifulSoup(str(t),"html.parser").get_text()).strip()

# ---------- Dataset ----------
class DatasetSimple(Dataset):
    def __init__(self, df, tokenizer, max_len=128, labeled=True):
        self.texts = [clean_text(t) for t in df['Review'].astype(str).tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labeled = labeled
        if labeled:
            self.labels = df[LABEL_COLS].values

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        tok = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {'input_ids': tok['input_ids'].squeeze(0), 'attention_mask': tok['attention_mask'].squeeze(0)}
        if self.labeled:
            seg_raw = self.labels[idx].astype(int)
            pres = (seg_raw > 0).astype(float)
            seg = np.where(seg_raw > 0, seg_raw - 1, -1)  # -1 mask
            item.update({'pres': torch.tensor(pres, dtype=torch.float), 'seg': torch.tensor(seg, dtype=torch.long)})
        return item

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    out = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'pres' in batch[0]:
        pres = torch.stack([b['pres'] for b in batch])
        seg = torch.stack([b['seg'] for b in batch])
        out.update({'pres': pres, 'seg': seg})
    return out

# ---------- Model ----------
class SimpleMultiTask(nn.Module):
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.segment_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden_head), nn.ReLU(), nn.Linear(hidden_head, num_seg_classes))
            for _ in range(num_aspects)
        ])
        self.presence_head = nn.Linear(hidden, num_aspects)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        seg_logits = torch.stack([h(pooled) for h in self.segment_heads], dim=1)  # [B,A,C]
        pres_logits = self.presence_head(pooled)
        return seg_logits, pres_logits

# ---------- Robust train one epoch ----------
def train_one_epoch_safe(model, loader, optimizer, device, clip_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n_steps = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seg = batch['seg'].to(device)   # [B,A]
        pres = batch['pres'].to(device)
        seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # compute seg loss per-aspect safely
        A = seg_logits.shape[1]
        loss_seg_sum = 0.0
        valid_aspects = 0
        for a in range(A):
            logits_a = seg_logits[:, a, :]        # [B,C]
            targ_a = seg[:, a]                   # [B]
            # compute per-sample CE (reduction none)
            loss_vec = F.cross_entropy(logits_a, targ_a, ignore_index=-1, reduction='none')
            mask = (targ_a != -1)
            if mask.sum() == 0:
                continue
            loss_a = loss_vec[mask].mean()
            loss_seg_sum = loss_seg_sum + loss_a
            valid_aspects += 1
        if valid_aspects == 0:
            loss_seg = torch.tensor(0.0, device=device)
        else:
            loss_seg = loss_seg_sum / valid_aspects
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres)
        loss = 1.5 * loss_seg + 1.0 * loss_pres

        # safeguard
        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at batch {batch_idx}: loss_seg finite? {torch.isfinite(loss_seg).item()}, loss_pres finite? {torch.isfinite(loss_pres).item()}")
            # inspect seg unique values
            for a in range(A):
                try:
                    unique_vals = torch.unique(seg[:,a]).cpu().numpy().tolist()
                except:
                    unique_vals = "?"
                print(f"  aspect {a} unique targets in batch:", unique_vals)
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_steps += 1

    return total_loss / max(1, n_steps)

# ---------- Inference on unlabeled with per-aspect selection rules ----------
def infer_on_unlabeled(model, loader, device, pres_thres=0.90, seg_thres=0.90, require_at_least_k=1):
    model.eval()
    preds = []
    raw_confidences = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()       # [B,A]
            seg_probs = torch.softmax(seg_logits, dim=-1).cpu().numpy() # [B,A,C]
            seg_preds = np.argmax(seg_probs, axis=-1)                   # [B,A]
            seg_conf = np.max(seg_probs, axis=-1)                       # [B,A]
            B = seg_preds.shape[0]
            for i in range(B):
                row_labels = [0]*NUM_ASPECTS
                confident_count = 0
                for a in range(NUM_ASPECTS):
                    if pres_probs[i,a] >= pres_thres and seg_conf[i,a] >= seg_thres:
                        row_labels[a] = int(seg_preds[i,a]) + 1  # map to 1..5
                        confident_count += 1
                    else:
                        row_labels[a] = 0
                # accept sample if at least k aspects are confident
                if confident_count >= require_at_least_k:
                    preds.append(row_labels)
                    raw_confidences.append({'pres_probs': pres_probs[i].tolist(), 'seg_conf': seg_conf[i].tolist()})
                else:
                    preds.append(None)  # mark as rejected
                    raw_confidences.append({'pres_probs': pres_probs[i].tolist(), 'seg_conf': seg_conf[i].tolist()})
    return preds, raw_confidences

# ---------- Helper: identify minority classes per aspect ----------
def find_minority_classes(df, min_count=50):
    res = {}
    for col in LABEL_COLS:
        vals = df[col].values
        mapped = np.where(vals > 0, vals - 1, -1)
        unique, counts = np.unique(mapped, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        # drop -1
        freq.pop(-1, None)
        minority = [int(k) for k,v in freq.items() if v < min_count]
        res[col] = minority
    return res

# ---------- Main pipeline ----------
def self_training_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = pd.read_csv(Path(args.data_dir)/"problem_train.csv")
    # ensure labels are ints
    train[LABEL_COLS] = train[LABEL_COLS].fillna(0).astype(int)
    unlabeled = pd.read_csv(args.unlabeled) if args.unlabeled else pd.read_csv(Path(args.data_dir)/"problem_test.csv")
    print("Train rows:", len(train), "Unlabeled rows:", len(unlabeled))
    print("Minority classes (count < {}):".format(args.min_count))
    print(find_minority_classes(train, min_count=args.min_count))

    total_pseudo = 0
    for rnd in range(args.rounds):
        print("Round", rnd+1)
        model = SimpleMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES).to(device)
        ds = DatasetSimple(train, tokenizer, labeled=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        # quick train for several epochs per round
        for ep in range(args.epochs_per_round):
            loss = train_one_epoch_safe(model, loader, opt, device)
            print(f" epoch {ep+1} loss {loss:.6f}")
        # infer on unlabeled
        ul_ds = DatasetSimple(unlabeled, tokenizer, labeled=False)
        ul_loader = DataLoader(ul_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=2)
        preds, confidences = infer_on_unlabeled(model, ul_loader, device, pres_thres=args.pres_thres, seg_thres=args.seg_thres, require_at_least_k=args.require_at_least_k)
        # collect accepted samples
        rows = []
        for i, p in enumerate(preds):
            if p is None: continue
            base = unlabeled.iloc[i].to_dict()
            for j,col in enumerate(LABEL_COLS):
                base[col] = p[j]
            rows.append(base)
        print(f"Round {rnd+1}: selected {len(rows)} pseudo samples (before cap)")
        if len(rows) > args.max_add_per_round:
            rows = rows[:args.max_add_per_round]
        if rows:
            train = pd.concat([train, pd.DataFrame(rows)], ignore_index=True).sample(frac=1).reset_index(drop=True)
            total_pseudo += len(rows)
            print(f"  added {len(rows)} pseudo samples to train.")
        else:
            print("  no pseudo added this round.")
    # save augmented train
    train.to_csv(Path(args.data_dir)/"train_pseudo_aug_safe.csv", index=False)
    print("Finished. Total pseudo added:", total_pseudo)
    print("Saved train_pseudo_aug_safe.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--unlabeled", type=str, required=False)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs_per_round", type=int, default=1)
    parser.add_argument("--pres_thres", type=float, default=0.90)
    parser.add_argument("--seg_thres", type=float, default=0.90)
    parser.add_argument("--require_at_least_k", type=int, default=1, help="Require at least k aspects confident to accept sample")
    parser.add_argument("--max_add_per_round", type=int, default=500)
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    self_training_pipeline(args)
