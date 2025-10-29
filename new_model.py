# pseudo_labeling.py
"""
Pseudo-labeling / self-training pipeline.
Usage:
  python pseudo_labeling.py --data_dir data --unlabeled data/unlabeled.csv --rounds 2 --thres 0.95
If you don't have unlabeled file, you can pass test.csv as unlabeled to try.
"""

import os, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

MODEL_NAME = "vinai/phobert-base"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEG_CLASSES = 5

# We'll reuse a simple multitask model (multiclass segment heads).
class SimpleMultiTask(nn.Module):
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=256):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.segment_heads = nn.ModuleList([ nn.Sequential(nn.Linear(hidden, hidden_head), nn.ReLU(), nn.Linear(hidden_head, num_seg_classes)) for _ in range(num_aspects) ])
        self.presence_head = nn.Linear(hidden, num_aspects)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        seg_logits = torch.stack([h(pooled) for h in self.segment_heads], dim=1)  # [B,A,C]
        pres_logits = self.presence_head(pooled)
        return seg_logits, pres_logits

# Dataset similar as before (no augmentation)
import re, html
from bs4 import BeautifulSoup
def clean_text(t): return re.sub(r"\s+"," ", BeautifulSoup(str(t),"html.parser").get_text()).strip()

class DatasetSimple(Dataset):
    def __init__(self, df, tokenizer, max_len=128, labeled=True):
        self.texts = [clean_text(t) for t in df['Review'].astype(str).tolist()]
        self.labeled = labeled
        self.tokenizer = tokenizer
        self.max_len = max_len
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
            seg = np.where(seg_raw > 0, seg_raw - 1, -1)
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

# train basic single-epoch/simple train to speed up for pseudo pipeline
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seg = batch['seg'].to(device)
        pres = batch['pres'].to(device)
        seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        # seg loss: per-aspect crossentropy w/ ignore_index -1
        loss_seg = 0.0
        A = seg_logits.shape[1]
        for a in range(A):
            loss_seg += F.cross_entropy(seg_logits[:,a,:], seg[:,a], ignore_index=-1)
        loss_seg = loss_seg / A
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres)
        loss = 1.5*loss_seg + 1.0*loss_pres
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += float(loss.detach().cpu().item())
    return total_loss / len(loader)

def infer_on_unlabeled(model, loader, device, pres_thres=0.95, seg_thres=0.95):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            seg_probs = torch.softmax(seg_logits, dim=-1).cpu().numpy()  # [B,A,C]
            seg_preds = np.argmax(seg_probs, axis=-1)  # 0..4
            seg_conf = np.max(seg_probs, axis=-1)  # confidence per-aspect
            B = seg_preds.shape[0]
            for i in range(B):
                # decide if we accept pseudo for this sample: require all predicted pres above pres_thres OR at least one high-confidence seg
                accepted = []
                seg_labels_for_row = []
                for a in range(NUM_ASPECTS):
                    p_pres = pres_probs[i,a]
                    if p_pres >= pres_thres and seg_conf[i,a] >= seg_thres:
                        # map back to 1..5
                        seg_val = int(seg_preds[i,a]) + 1
                    else:
                        seg_val = 0
                    seg_labels_for_row.append(seg_val)
                preds.append(seg_labels_for_row)
    return preds  # list of [A] ints 0..5

def self_training_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train = pd.read_csv(Path(args.data_dir)/"problem_train.csv")
    unlabeled = pd.read_csv(args.unlabeled) if args.unlabeled else pd.read_csv(Path(args.data_dir)/"problem_test.csv")
    # rounds of pseudo labeling
    pseudo_added = 0
    for rnd in range(args.rounds):
        print("Round", rnd+1)
        model = SimpleMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES).to(device)
        # train on current train
        ds = DatasetSimple(train, tokenizer, labeled=True)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        for ep in range(args.epochs_per_round):
            loss = train_one_epoch(model, loader, opt, device)
            print(f" epoch {ep+1} loss {loss:.4f}")
        # infer on unlabeled pool
        ul_ds = DatasetSimple(unlabeled, tokenizer, labeled=False)
        ul_loader = DataLoader(ul_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
        preds = infer_on_unlabeled(model, ul_loader, device, pres_thres=args.pres_thres, seg_thres=args.seg_thres)
        # convert preds -> DataFrame rows and select only those with any non-zero seg
        rows = []
        for i, row in enumerate(preds):
            if any([v>0 for v in row]):
                base = unlabeled.iloc[i].to_dict()
                for j,col in enumerate(LABEL_COLS):
                    base[col] = row[j]
                rows.append(base)
        print(f"Round {rnd+1}: selected {len(rows)} pseudo samples")
        # limit how many to add to avoid noise
        if len(rows) > args.max_add_per_round:
            rows = rows[:args.max_add_per_round]
        if rows:
            train = pd.concat([train, pd.DataFrame(rows)], ignore_index=True).sample(frac=1).reset_index(drop=True)
            pseudo_added += len(rows)
    print("Pseudo labeling finished. total pseudo added:", pseudo_added)
    # save augmented train
    train.to_csv(Path(args.data_dir)/"train_pseudo_aug.csv", index=False)
    print("Saved train_pseudo_aug.csv — you can now retrain fully on it.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--unlabeled", type=str, default=None)  # path to unlabeled pool (csv)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs_per_round", type=int, default=1)
    parser.add_argument("--pres_thres", type=float, default=0.95)
    parser.add_argument("--seg_thres", type=float, default=0.95)
    parser.add_argument("--max_add_per_round", type=int, default=1000)
    args = parser.parse_args()
    self_training_pipeline(args)
