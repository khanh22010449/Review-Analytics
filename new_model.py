# train_final.py
import math, random, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

MODEL_NAME = "vinai/phobert-base"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEG_CLASSES = 5

# Text cleaning (reuse)
import re
from bs4 import BeautifulSoup
def clean_text(t): return re.sub(r"\s+"," ", BeautifulSoup(str(t),"html.parser").get_text()).strip()

# Dataset
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = [clean_text(t) for t in df['Review'].astype(str).tolist()]
        self.labels = df[LABEL_COLS].astype(int).values
        # detect is_pseudo column if exists
        self.is_pseudo = df['is_pseudo'].values if 'is_pseudo' in df.columns else np.zeros(len(df), dtype=int)
        self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        tok = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        seg_raw = self.labels[idx].astype(int)
        pres = (seg_raw > 0).astype(float)
        seg = np.where(seg_raw > 0, seg_raw - 1, -1)
        return {'input_ids': tok['input_ids'].squeeze(0), 'attention_mask': tok['attention_mask'].squeeze(0),
                'pres': torch.tensor(pres, dtype=torch.float), 'seg': torch.tensor(seg, dtype=torch.long),
                'is_pseudo': int(self.is_pseudo[idx])}

def collate_fn(batch):
    import torch
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    pres = torch.stack([b['pres'] for b in batch])
    seg = torch.stack([b['seg'] for b in batch])
    is_pseudo = torch.tensor([b['is_pseudo'] for b in batch], dtype=torch.float)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'pres': pres, 'seg': seg, 'is_pseudo': is_pseudo}

# Model (reuse same architecture as new_model.py)
class SimpleMultiTask(nn.Module):
    def __init__(self, model_name, num_aspects, num_seg_classes, hidden_head=512):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.segment_heads = nn.ModuleList([ nn.Sequential(nn.Linear(hidden, hidden_head), nn.ReLU(), nn.Linear(hidden_head, num_seg_classes)) for _ in range(num_aspects) ])
        self.presence_head = nn.Linear(hidden, num_aspects)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = o.pooler_output if hasattr(o,'pooler_output') and o.pooler_output is not None else o.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        seg_logits = torch.stack([h(pooled) for h in self.segment_heads], dim=1)
        pres_logits = self.presence_head(pooled)
        return seg_logits, pres_logits

def train_epoch(model, loader, opt, device, pseudo_weight=0.3):
    model.train()
    tot=0.0; steps=0
    for b in loader:
        input_ids=b['input_ids'].to(device); attention_mask=b['attention_mask'].to(device)
        seg=b['seg'].to(device); pres=b['pres'].to(device); is_pseudo=b['is_pseudo'].to(device)
        seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        # compute seg loss per-aspect robustly
        A = seg_logits.shape[1]
        loss_seg_sum = 0.0; valid_aspects=0
        for a in range(A):
            loss_vec = F.cross_entropy(seg_logits[:,a,:], seg[:,a], ignore_index=-1, reduction='none')
            mask = (seg[:,a] != -1)
            if mask.sum() == 0: continue
            loss_a = loss_vec[mask].mean()
            loss_seg_sum += loss_a; valid_aspects += 1
        loss_seg = loss_seg_sum/valid_aspects if valid_aspects>0 else torch.tensor(0.0, device=device)
        loss_pres = F.binary_cross_entropy_with_logits(pres_logits, pres)
        # combine, weight pseudo samples
        # create per-batch weighting: samples pseudo have weight pseudo_weight, real weight 1.0
        weights = (1.0 - (1.0 - pseudo_weight)*is_pseudo)  # pseudo->pseudo_weight, real->1.0
        # We apply sample weighting only to reduce epoch-level effect by scaling loss
        loss = (1.5*loss_seg + 1.0*loss_pres) * weights.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item()); steps+=1
    return tot/steps if steps>0 else 0.0

# simple eval on val
def evaluate(model, val_loader, device):
    model.eval()
    seg_preds=[]; seg_trues=[]
    pres_preds=[]; pres_trues=[]
    with torch.no_grad():
        for b in val_loader:
            input_ids=b['input_ids'].to(device); attention_mask=b['attention_mask'].to(device)
            seg=b['seg'].numpy(); pres=b['pres'].numpy()
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_p = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # 0..C-1
            pres_p = (torch.sigmoid(pres_logits).cpu().numpy() >= 0.5).astype(int)
            seg_preds.append(seg_p); seg_trues.append(seg)
            pres_preds.append(pres_p); pres_trues.append(pres)
    import numpy as np
    seg_preds = np.vstack(seg_preds); seg_trues = np.vstack(seg_trues)
    pres_preds = np.vstack(pres_preds); pres_trues = np.vstack(pres_trues)
    # compute metrics similar to before
    A = seg_preds.shape[1]
    accs=[]; f1s=[]
    for a in range(A):
        mask = seg_trues[:,a] != -1
        if mask.sum()==0: continue
        accs.append(accuracy_score(seg_trues[mask,a], seg_preds[mask,a]))
        f1s.append(f1_score(seg_trues[mask,a], seg_preds[mask,a], average='macro', zero_division=0))
    metrics = {
        'seg_acc_mean': float(np.mean(accs) if accs else 0.0),
        'seg_f1_macro_mean': float(np.mean(f1s) if f1s else 0.0),
        'pres_f1_micro': float(f1_score(pres_trues.reshape(-1), pres_preds.reshape(-1), average='micro', zero_division=0)),
        'pres_f1_macro': float(f1_score(pres_trues.reshape(-1), pres_preds.reshape(-1), average='macro', zero_division=0))
    }
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--pseudo_weight", type=float, default=0.4, help="loss weight for pseudo samples (0..1)")
    args = parser.parse_args()

    train_all = pd.read_csv(Path(args.data_dir)/"train_pseudo_aug_v2.csv")
    # Heuristic: if no explicit is_pseudo column, assume last N rows are pseudo
    if 'is_pseudo' not in train_all.columns:
        # if file size > original_train (1927) we mark appended rows as pseudo
        orig = pd.read_csv(Path(args.data_dir)/"problem_train.csv")
        n_orig = len(orig)
        train_all['is_pseudo'] = [0]*len(train_all)
        for i in range(n_orig, len(train_all)):
            train_all.at[i,'is_pseudo'] = 1

    val_df = pd.read_csv(Path(args.data_dir)/"problem_val.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TrainDataset(train_all, tokenizer, max_len=128)
    val_ds = TrainDataset(val_df, tokenizer, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMultiTask(MODEL_NAME, NUM_ASPECTS, NUM_SEG_CLASSES, hidden_head=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader)*args.epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    best_score = None
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, opt, device, pseudo_weight=args.pseudo_weight)
        scheduler.step()
        print(f"Epoch {ep+1} train_loss {loss:.4f}")
        metrics = evaluate(model, val_loader, device)
        print("Val metrics:", metrics)
        score = metrics['seg_f1_macro_mean'] + metrics['pres_f1_micro']
        if best_score is None or score > best_score:
            best_score = score
            torch.save(model.state_dict(), Path(args.data_dir)/"final_model_best.pt")
            print("Saved final_model_best.pt")
    print("Done. Best score:", best_score)
