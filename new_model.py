# new_model.py
"""
Safe pseudo-labeling pipeline (improved).
Features:
 - Robust train loop avoiding NaN when many ignore_index per-batch
 - Inference collects pres_probs and seg_probs (loads checkpoint if provided)
 - Candidate selection supports OR rule, per-aspect pres thresholds, top-k selection
 - Saves augmented train CSV and candidates CSV for manual review

Usage example:
  python new_model.py --data_dir data --unlabeled data/problem_test.csv --rounds 1 --epochs_per_round 1 \
    --batch 16 --use_or True --use_per_aspect_pres True --pres_percentile 75 --seg_thres 0.75 \
    --top_k 400 --max_add_per_round 200 --model_path best_multitask_weighted.pt
"""

import os
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ---------------- Config ----------------
MODEL_NAME = "vinai/phobert-base"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEG_CLASSES = 5  # segments 1..5 mapped to 0..4
# ----------------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(42)

# ---------- Text clean ----------
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
            # ensure ints
            df_loc = df.copy()
            df_loc[LABEL_COLS] = df_loc[LABEL_COLS].fillna(0).astype(int)
            self.labels = df_loc[LABEL_COLS].values

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        tok = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {'input_ids': tok['input_ids'].squeeze(0), 'attention_mask': tok['attention_mask'].squeeze(0)}
        if self.labeled:
            seg_raw = self.labels[idx].astype(int)
            pres = (seg_raw > 0).astype(float)
            seg = np.where(seg_raw > 0, seg_raw - 1, -1)  # -1 indicates ignore
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
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=256, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.segment_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden_head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_head, num_seg_classes))
            for _ in range(num_aspects)
        ])
        self.presence_head = nn.Linear(hidden, num_aspects)
        self.dropout = nn.Dropout(dropout)

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
        seg = batch['seg'].to(device)   # [B,A] with -1
        pres = batch['pres'].to(device)

        seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # compute seg loss per-aspect safely
        A = seg_logits.shape[1]
        loss_seg_sum = 0.0
        valid_aspects = 0
        for a in range(A):
            logits_a = seg_logits[:, a, :]        # [B,C]
            targ_a = seg[:, a]                   # [B]
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

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at batch {batch_idx}: loss_seg finite? {torch.isfinite(loss_seg).item()}, loss_pres finite? {torch.isfinite(loss_pres).item()}")
            # print batch seg unique distribution for debugging
            for a in range(A):
                try:
                    u = torch.unique(seg[:,a]).cpu().numpy().tolist()
                except:
                    u = "?"
                print(f"  aspect {a} unique targets in batch:", u)
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

# ---------- Inference collecting pres_probs & seg_probs ----------
def collect_probs(model, loader, device):
    model.eval()
    pres_list = []
    seg_probs_list = []  # store full distribution to extract preds later
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer unlabeled"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()       # [B,A]
            seg_probs = torch.softmax(seg_logits, dim=-1).cpu().numpy() # [B,A,C]
            pres_list.append(pres_probs)
            seg_probs_list.append(seg_probs)
    pres_all = np.vstack(pres_list)
    seg_probs_all = np.vstack(seg_probs_list)  # shape [N,A,C]
    return pres_all, seg_probs_all

# ---------- Candidate selection (OR/AND rules, per-aspect thres, top-k) ----------
def select_candidates(pres_all, seg_probs_all, use_or=True, pres_thres=0.9, seg_thres=0.75,
                      use_per_aspect_pres=False, pres_percentile=75, require_at_least_k=1, top_k=None):
    N, A = pres_all.shape
    # compute per-aspect pres thresholds if requested
    if use_per_aspect_pres:
        per_aspect_thres = []
        for a in range(A):
            th = float(np.percentile(pres_all[:,a], pres_percentile))
            th = float(min(0.95, max(0.1, th)))  # clamp
            per_aspect_thres.append(th)
    else:
        per_aspect_thres = [pres_thres] * A

    candidates = []
    for i in range(N):
        row_labels = [0] * A
        confident_count = 0
        score = 0.0
        for a in range(A):
            p_pres = pres_all[i,a]
            p_seg_conf = float(np.max(seg_probs_all[i,a]))  # max softmax
            seg_pred = int(np.argmax(seg_probs_all[i,a])) + 1  # 1..5
            th_pres = per_aspect_thres[a]
            ok = False
            if use_or:
                if (p_pres >= th_pres) or (p_seg_conf >= seg_thres):
                    ok = True
            else:
                if (p_pres >= th_pres) and (p_seg_conf >= seg_thres):
                    ok = True
            if ok:
                row_labels[a] = seg_pred
                confident_count += 1
                score = max(score, max(p_pres, p_seg_conf))
        if confident_count >= require_at_least_k:
            candidates.append({'idx': i, 'score': score, 'labels': row_labels, 'confident_count': confident_count})
    # top_k selection
    if (top_k is not None) and (len(candidates) > 0):
        candidates = sorted(candidates, key=lambda x: -x['score'])[:top_k]
    return candidates, per_aspect_thres

# ---------- Minority classes helper ----------
def find_minority_classes(df, min_count=50):
    res = {}
    for col in LABEL_COLS:
        vals = df[col].values
        mapped = np.where(vals > 0, vals - 1, -1)
        unique, counts = np.unique(mapped, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        freq.pop(-1, None)
        minority = [int(k) for k,v in freq.items() if v < min_count]
        res[col] = minority
    return res

# ---------- Main pipeline ----------
def self_training_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_path = Path(args.data_dir) / "problem_train.csv"
    val_path = Path(args.data_dir) / "problem_val.csv"
    train = pd.read_csv(train_path)
    train[LABEL_COLS] = train[LABEL_COLS].fillna(0).astype(int)
    unlabeled = pd.read_csv(args.unlabeled) if args.unlabeled else pd.read_csv(Path(args.data_dir)/"problem_test.csv")
    print("Train rows:", len(train), "Unlabeled rows:", len(unlabeled))
    print("Minority classes (count < {}):".format(args.min_count))
    print(find_minority_classes(train, min_count=args.min_count))

    total_pseudo = 0
    # optionally pre-load a model for inference if user gives model_path
    model_for_infer = None
    if args.model_path:
        if Path(args.model_path).exists():
            print("Loading model checkpoint for inference from", args.model_path)
            model_for_infer = SimpleMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES)
            state = torch.load(args.model_path, map_location="cpu")
            model_for_infer.load_state_dict(state)
            model_for_infer.to(device)
            model_for_infer.eval()
        else:
            print("Warning: model_path provided but file not found:", args.model_path)
            model_for_infer = None

    for rnd in range(args.rounds):
        print("Round", rnd+1)
        # train a fresh model on current train (can be light)
        model = SimpleMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES).to(device)
        ds = DatasetSimple(train, tokenizer, labeled=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        for ep in range(args.epochs_per_round):
            loss = train_one_epoch_safe(model, loader, optimizer, device)
            print(f" epoch {ep+1} loss {loss:.6f}")

        # choose model to use for inference: prefer checkpoint-loaded model if provided, else use newly trained
        infer_model = model_for_infer if (model_for_infer is not None and rnd==0 and args.use_checkpoint_only) else model
        infer_model.to(device)

        # collect probs on unlabeled
        ul_ds = DatasetSimple(unlabeled, tokenizer, labeled=False)
        ul_loader = DataLoader(ul_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=2)
        pres_all, seg_probs_all = collect_probs(infer_model, ul_loader, device)

        # select candidates
        candidates, per_aspect_thres = select_candidates(
            pres_all, seg_probs_all,
            use_or=args.use_or,
            pres_thres=args.pres_thres,
            seg_thres=args.seg_thres,
            use_per_aspect_pres=args.use_per_aspect_pres,
            pres_percentile=args.pres_percentile,
            require_at_least_k=args.require_at_least_k,
            top_k=args.top_k
        )

        print(f"Round {rnd+1}: found {len(candidates)} candidates (before cap). Per-aspect pres thresholds:", per_aspect_thres)
        # save candidates to CSV for manual review
        cand_rows = []
        for c in candidates:
            idx = c['idx']
            row = {'idx': int(idx), 'score': float(c['score']), 'confident_count': int(c['confident_count'])}
            # original text + candidate labels
            row['Review'] = unlabeled.iloc[idx]['Review']
            for j,col in enumerate(LABEL_COLS):
                row[col] = int(c['labels'][j])
            cand_rows.append(row)
        cand_df = pd.DataFrame(cand_rows)
        cand_csv = Path(args.data_dir) / f"candidates_round{rnd+1}.csv"
        if not cand_df.empty:
            cand_df.to_csv(cand_csv, index=False)
            print("Saved candidates to", cand_csv)
        else:
            print("No candidate CSV saved (empty).")

        # cap and add to train
        if len(candidates) > args.max_add_per_round:
            candidates = sorted(candidates, key=lambda x: -x['score'])[:args.max_add_per_round]

        rows_to_add = []
        for c in candidates:
            idx = c['idx']
            base = unlabeled.iloc[idx].to_dict()
            for j,col in enumerate(LABEL_COLS):
                base[col] = int(c['labels'][j])
            rows_to_add.append(base)

        if rows_to_add:
            train = pd.concat([train, pd.DataFrame(rows_to_add)], ignore_index=True).sample(frac=1).reset_index(drop=True)
            total_pseudo += len(rows_to_add)
            print(f"  added {len(rows_to_add)} pseudo samples to train.")
        else:
            print("  no pseudo added this round.")

    out_path = Path(args.data_dir) / "train_pseudo_aug_v2.csv"
    train.to_csv(out_path, index=False)
    print("Finished. Total pseudo added:", total_pseudo)
    print("Saved augmented train to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--unlabeled", type=str, required=False, help="path to unlabeled csv")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs_per_round", type=int, default=1)
    parser.add_argument("--pres_thres", type=float, default=0.90)
    parser.add_argument("--seg_thres", type=float, default=0.75)
    parser.add_argument("--use_or", type=lambda x: x.lower() == "true", default=True, help="Use OR rule (pres OR seg) to accept aspect")
    parser.add_argument("--use_per_aspect_pres", type=lambda x: x.lower() == "true", default=False, help="Compute per-aspect pres threshold using percentile")
    parser.add_argument("--pres_percentile", type=float, default=75.0)
    parser.add_argument("--require_at_least_k", type=int, default=1)
    parser.add_argument("--max_add_per_round", type=int, default=500)
    parser.add_argument("--min_count", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=None, help="If set, keep only top_k candidates by score")
    parser.add_argument("--model_path", type=str, default=None, help="Optional path to checkpoint (state_dict) to use for inference")
    parser.add_argument("--use_checkpoint_only", type=lambda x: x.lower()=='true', default=False, help="If true and model_path provided, use checkpoint for inference (do not use newly trained model)")
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    self_training_pipeline(args)
