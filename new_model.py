# train_coral.py
"""
Train using CORAL (ordinal) for segment prediction.
Usage:
  python train_coral.py --data_dir data --epochs 6 --batch 16
"""
import os, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

# ----------------- Config -----------------
MODEL_NAME = "vinai/phobert-base"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEG_CLASSES = 5  # original 1..5 mapped to 0..4 for CORAL
# ------------------------------------------

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_everything()

# ---------- Dataset ----------
import re, html
from bs4 import BeautifulSoup
def clean_text(t): return re.sub(r"\s+"," ", BeautifulSoup(str(t),"html.parser").get_text()).strip()

class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = [clean_text(t) for t in df['Review'].astype(str).tolist()]
        self.labels_raw = df[LABEL_COLS].values  # 0..5
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        tok = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        seg_raw = self.labels_raw[idx].astype(int)  # 0..5
        # presence
        pres = (seg_raw > 0).astype(float)
        # seg for coral: -1 for absent, else 0..4
        seg = np.where(seg_raw > 0, seg_raw - 1, -1)
        return {
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0),
            'seg_raw': torch.tensor(seg, dtype=torch.long),
            'pres': torch.tensor(pres, dtype=torch.float)
        }

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    seg_raw = torch.stack([b['seg_raw'] for b in batch])
    pres = torch.stack([b['pres'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'seg_raw': seg_raw, 'pres': pres}

# ---------- Model with CORAL ----------
class CoralMultiTask(nn.Module):
    def __init__(self, model_name, num_aspects, num_classes, hidden_head=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.num_aspects = num_aspects
        self.K = num_classes  # 5
        # for CORAL we need K-1 thresholds per aspect
        self.coral_heads = nn.ModuleList([ nn.Linear(hidden, self.K - 1) for _ in range(num_aspects) ])
        self.pres_head = nn.Linear(hidden, num_aspects)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = o.pooler_output if hasattr(o,'pooler_output') and o.pooler_output is not None else o.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        # coral logits: list of [B, K-1] -> stack to [B, A, K-1]
        coral = torch.stack([head(pooled) for head in self.coral_heads], dim=1)
        pres_logits = self.pres_head(pooled)
        return coral, pres_logits

# ---------- CORAL loss & predict helpers ----------
def coral_targets_from_label(y, K):
    # y: tensor shape [B] with values 0..K-1 (or -1 for ignore)
    # return targets shape [B, K-1] with 0/1, and mask for valid rows
    B = y.shape[0]
    tgt = torch.zeros(B, K-1, dtype=torch.float, device=y.device)
    mask = (y != -1)
    for k in range(K-1):
        tgt[:, k] = (y > k).float()  # 1 if label > k
    return tgt, mask

def coral_loss_fn(coral_logits, seg_targets):
    # coral_logits: [B, K-1]; seg_targets: [B] in 0..K-1 or -1
    device = coral_logits.device
    K = coral_logits.shape[1] + 1
    tgt, valid_mask = coral_targets_from_label(seg_targets, K)
    # BCEWithLogitsLoss per threshold, reduce='none'
    bce = F.binary_cross_entropy_with_logits(coral_logits, tgt, reduction='none')
    # mask rows where seg_targets == -1 (absent)
    mask = valid_mask.unsqueeze(1).float()
    loss = (bce * mask).sum() / (mask.sum().clamp_min(1.0))
    return loss

def coral_decode(coral_logits):
    # coral_logits: [B, K-1] -> probs -> sum of positive thresholds -> label 0..K-1
    probs = torch.sigmoid(coral_logits)  # [B, K-1]
    # predicted y = number of thresholds with prob>0.5
    preds = (probs > 0.5).sum(dim=1)
    return preds  # 0..K-1

# ---------- Train & eval ----------
def compute_metrics(seg_logits_all, pres_logits_all, seg_t_all, pres_t_all):
    # seg_logits_all: list of [B,A,K-1] tensors
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    seg_logits = torch.cat(seg_logits_all, dim=0)  # [N,A,K-1]
    pres_logits = torch.cat(pres_logits_all, dim=0)  # [N,A]
    seg_t = torch.cat(seg_t_all, dim=0).numpy()  # [N,A]
    pres_t = torch.cat(pres_t_all, dim=0).numpy().astype(int)
    N,A,Km1 = seg_logits.shape
    seg_preds = torch.argmax(seg_logits, dim=-1)  # wrong: coral gives K-1 -> decode per-aspect below
    # decode per aspect:
    seg_preds_np = np.zeros((N,A), dtype=int)
    for a in range(A):
        preds_a = coral_decode(seg_logits[:,a,:]).cpu().numpy()  # 0..K-1
        seg_preds_np[:,a] = preds_a
    pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
    pres_preds = (pres_probs >= 0.5).astype(int)
    # compute metrics same as before
    accs, f1s_seg = [], []
    for a in range(A):
        mask = seg_t[:,a] != -1
        if mask.sum() == 0: continue
        accs.append(accuracy_score(seg_t[mask,a], seg_preds_np[mask,a]))
        f1s_seg.append(f1_score(seg_t[mask,a], seg_preds_np[mask,a], average='macro', zero_division=0))
    metrics = {
        'seg_acc_mean': float(np.mean(accs) if accs else 0.0),
        'seg_f1_macro_mean': float(np.mean(f1s_seg) if f1s_seg else 0.0),
        'pres_f1_micro': float(f1_score(pres_t.reshape(-1), pres_preds.reshape(-1), average='micro', zero_division=0)),
        'pres_f1_macro': float(f1_score(pres_t.reshape(-1), pres_preds.reshape(-1), average='macro', zero_division=0))
    }
    return metrics

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_csv(Path(args.data_dir)/"problem_train.csv")
    val = pd.read_csv(Path(args.data_dir)/"problem_val.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train, tokenizer, max_len=128)
    val_ds = ReviewDataset(val, tokenizer, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    model = CoralMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    best = None
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{ep+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_raw = batch['seg_raw'].to(device)  # [B,A]
            pres = batch['pres'].to(device)
            coral_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # compute coral loss per aspect
            loss_seg = 0.0
            A = coral_logits.shape[1]
            for a in range(A):
                loss_seg += coral_loss_fn(coral_logits[:,a,:], seg_raw[:,a])
            loss_seg = loss_seg / max(1, A)
            loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres)
            loss = 1.5 * loss_seg + 1.0 * loss_pres
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); scheduler.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))
        # validate
        model.eval()
        seg_logits_all, pres_logits_all, seg_t_all, pres_t_all = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                seg_raw = batch['seg_raw'].to(device)
                pres = batch['pres'].to(device)
                coral_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                seg_logits_all.append(coral_logits.cpu())
                pres_logits_all.append(pres_logits.cpu())
                seg_t_all.append(seg_raw.cpu())
                pres_t_all.append(pres.cpu())
        metrics = compute_metrics(seg_logits_all, pres_logits_all, seg_t_all, pres_t_all)
        print("Val metrics:", metrics)
        score = metrics['seg_f1_macro_mean'] + metrics['pres_f1_micro']
        if best is None or score > best:
            best = score
            torch.save(model.state_dict(), "best_coral.pt")
            print("Saved best_coral.pt")
    print("Done. Best score:", best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train(args)
