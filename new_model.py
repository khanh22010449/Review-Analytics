# nn_backbone_full.py
"""
Backbone-based multi-task model for segment+presence.
Features:
 - vinai/phobert-base encoder backbone
 - optional BiLSTM on token embeddings
 - per-aspect attention pooling (learnable queries)
 - adapter residual + per-aspect MLP heads (segment + presence)
 - offline back-translation augmentation (optional)
 - focal loss for segment, BCE for presence
 - weighted sampler for imbalance
Usage:
  python nn_backbone_full.py --data_dir data --do_augment False
"""

import os
import argparse
import random
import json
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from transformers import MarianMTModel, MarianTokenizer

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

# ---------------- default config ----------------
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
DEFAULT_BATCH = 16
LR_ENCODER = 1e-5
LR_HEADS = 1e-4
NUM_EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_ASPECTS = len(LABEL_COLS)
NUM_SEGMENT_CLASSES = 5
CACHE_DIR = "bt_cache"
BT_BATCH = 16
WEIGHT_DECAY = 0.01
FREEZE_ENCODER_EPOCHS = 1
# -------------------------------------------------

# ---------- utilities ----------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ---------- text cleaning ----------
import re, html
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9À-Ỹà-ỹ\s.,!?;:'\"%&$()\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- dataset ----------
class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.texts = self.df['Review'].astype(str).tolist()
        self.labels = self.df[[*LABEL_COLS]].fillna(0).astype(int).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        tok = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        seg_raw = self.labels[idx]   # values 0..5
        pres = (seg_raw > 0).astype(np.float32)
        seg = np.where(seg_raw > 0, seg_raw - 1, -1)   # map to -1 or 0..4
        return {
            'input_ids': tok['input_ids'].squeeze(0),
            'attention_mask': tok['attention_mask'].squeeze(0),
            'seg': torch.tensor(seg, dtype=torch.long),
            'pres': torch.tensor(pres, dtype=torch.float)
        }

def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    seg = torch.stack([x['seg'] for x in batch])
    pres = torch.stack([x['pres'] for x in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'seg': seg, 'pres': pres}

# ---------- Focal loss with ignore index ----------
class FocalLossIgnore(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
    def forward(self, logits, target):
        # logits: [B,C], target: [B]
        loss = self.ce(logits, target)   # [B]
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
        return focal.sum()

# ---------- model building blocks ----------
class Adapter(nn.Module):
    def __init__(self, hidden, bottleneck=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.down(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.up(x)
        x = self.drop(x)
        return self.ln(x + res)

# ---------- backbone + BiLSTM + attention pooling model ----------
class BackboneWithBiLSTMAndAttention(nn.Module):
    def __init__(self, backbone_name=MODEL_NAME, num_aspects=NUM_ASPECTS, num_seg_classes=NUM_SEGMENT_CLASSES,
                 use_bilstm=True, lstm_hidden=256, lstm_layers=1, pool_heads=8,
                 adapter_bottleneck=128, hidden_head=512, dropout=0.2, aux_token_classes=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden = self.config.hidden_size
        self.num_aspects = num_aspects
        self.num_seg_classes = num_seg_classes
        self.use_bilstm = use_bilstm

        if use_bilstm:
            self.bilstm = nn.LSTM(input_size=self.hidden, hidden_size=lstm_hidden, num_layers=lstm_layers,
                                  batch_first=True, bidirectional=True)
            post_dim = lstm_hidden * 2
        else:
            self.bilstm = None
            post_dim = self.hidden

        self.proj_back = nn.Linear(post_dim, self.hidden) if post_dim != self.hidden else nn.Identity()

        # learnable queries and multihead attention
        self.queries = nn.Parameter(torch.randn(self.num_aspects, self.hidden) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=self.hidden, num_heads=pool_heads, batch_first=True, dropout=dropout)
        self.attn_ln = nn.LayerNorm(self.hidden)
        self.attn_ff = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(dropout))

        # adapters + heads
        self.adapters = nn.ModuleList([Adapter(self.hidden, bottleneck=adapter_bottleneck, dropout=dropout) for _ in range(self.num_aspects)])
        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden, hidden_head),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head, hidden_head // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head // 2, self.num_seg_classes)
            ) for _ in range(self.num_aspects)
        ])
        self.presence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden, max(64, hidden_head // 4)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(64, hidden_head // 4), 1)
            ) for _ in range(self.num_aspects)
        ])

        self.aux_token_classes = aux_token_classes
        if aux_token_classes is not None:
            self.token_proj = nn.Sequential(
                nn.Linear(self.hidden, self.hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden // 2, aux_token_classes)
            )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.adapters) + list(self.segment_heads) + list(self.presence_heads):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def attention_pool(self, token_emb):
        # token_emb: [B, T, H]
        B, T, H = token_emb.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, A, H]
        attn_out, attn_weights = self.mha(query=q, key=token_emb, value=token_emb, key_padding_mask=None, attn_mask=None)
        out = self.attn_ln(attn_out + q)
        out = out + self.attn_ff(out)
        return out  # [B, A, H]

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = enc.last_hidden_state  # [B, T, H]
        token_emb = self.dropout(token_emb)

        if self.use_bilstm:
            lstm_out, _ = self.bilstm(token_emb)
            token_post = self.proj_back(lstm_out)
        else:
            token_post = self.proj_back(token_emb)

        pooled = self.attention_pool(token_post)  # [B, A, H]

        seg_logits_list = []
        pres_logits_list = []
        for a in range(self.num_aspects):
            vec = pooled[:, a, :]  # [B, H]
            vec = self.adapters[a](vec)  # residual inside
            seg_logits = self.segment_heads[a](vec)  # [B, C]
            pres_logit = self.presence_heads[a](vec).squeeze(-1)  # [B]
            seg_logits_list.append(seg_logits)
            pres_logits_list.append(pres_logit)

        seg_logits = torch.stack(seg_logits_list, dim=1)  # [B, A, C]
        pres_logits = torch.stack(pres_logits_list, dim=1)  # [B, A]

        token_logits = None
        if self.aux_token_classes is not None:
            token_logits = self.token_proj(token_post)  # [B, T, aux]

        return seg_logits, pres_logits, token_logits

# ---------- helpers: weights & sampler ----------
def compute_per_aspect_weights(train_df):
    weights_list = []
    for col in LABEL_COLS:
        vals = train_df[col].values
        vals_pos = vals[vals > 0]
        if len(vals_pos) == 0:
            weights_list.append(None); continue
        mapped = (vals_pos - 1).astype(int)
        classes = np.arange(NUM_SEGMENT_CLASSES)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=mapped)
        weights_list.append(torch.tensor(cw, dtype=torch.float))
    return weights_list

def make_multi_aspect_sampler(df):
    n = len(df)
    sample_weights = np.zeros(n, dtype=float)
    for col in LABEL_COLS:
        vals = df[col].values
        mapped = np.where(vals > 0, vals-1, -1)
        counts = {c: int(np.sum(mapped == c)) for c in range(NUM_SEGMENT_CLASSES)}
        col_weights = np.array([0.1 if v == -1 else 1.0 / (counts[int(v)] + 1e-9) for v in mapped])
        sample_weights = np.maximum(sample_weights, col_weights)
    sample_weights = sample_weights / (sample_weights.mean() + 1e-12)
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=n, replacement=True)

# ---------- back-translation helpers ----------
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

def do_offline_back_translation(train_df, cache_dir=CACHE_DIR, mid_lang='en', bt_batch=BT_BATCH):
    os.makedirs(cache_dir, exist_ok=True)
    cached_csv = Path(cache_dir) / "train_aug_bt.csv"
    cache_meta = Path(cache_dir) / "bt_meta.json"
    if cached_csv.exists():
        print("Found cached BT file:", cached_csv); return str(cached_csv)
    print("Preparing BT models... (download may occur)")
    model_pair = prepare_bt_models(src='vi', mid=mid_lang, cache_dir=cache_dir)
    texts = train_df['Review'].astype(str).tolist()
    bt_texts = back_translate_texts(texts, model_pair, batch_size=bt_batch, device='cpu')
    if len(bt_texts) != len(texts): raise RuntimeError("BT length mismatch")
    df_aug = train_df.copy(); df_aug['Review'] = bt_texts
    df_concat = pd.concat([train_df, df_aug], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df_concat.to_csv(cached_csv, index=False)
    with open(cache_meta, 'w') as f:
        json.dump({'src':'vi','mid':mid_lang,'n_orig':len(train_df),'n_aug':len(df_aug)}, f)
    print("Saved BT augmented:", cached_csv)
    return str(cached_csv)

# ---------- training and validation ----------
def validate(val_loader, model, device='cpu'):
    model.eval()
    seg_logits_all, pres_logits_all, seg_t_all, pres_t_all = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Valid", leave=False):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            seg_t = batch['seg'].to(device); pres_t = batch['pres'].to(device)
            seg_logits, pres_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_logits_all.append(seg_logits.cpu()); pres_logits_all.append(pres_logits.cpu())
            seg_t_all.append(seg_t.cpu()); pres_t_all.append(pres_t.cpu())
    seg_logits = torch.cat(seg_logits_all); pres_logits = torch.cat(pres_logits_all)
    seg_t = torch.cat(seg_t_all); pres_t = torch.cat(pres_t_all)
    return compute_metrics(seg_logits, pres_logits, seg_t, pres_t)

def compute_metrics(seg_logits, pres_logits, seg_t, pres_t, threshold=0.5):
    seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()
    seg_true = seg_t.cpu().numpy()
    pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
    pres_preds = (pres_probs >= threshold).astype(int)
    pres_true = pres_t.cpu().numpy().astype(int)
    accs = []; f1s = []
    for a in range(seg_true.shape[1]):
        mask = seg_true[:,a] != -1
        if mask.sum() == 0: continue
        accs.append(accuracy_score(seg_true[mask,a], seg_preds[mask,a]))
        f1s.append(f1_score(seg_true[mask,a], seg_preds[mask,a], average='macro', zero_division=0))
    metrics = {
        'seg_acc_mean': float(np.mean(accs) if accs else 0.0),
        'seg_f1_macro_mean': float(np.mean(f1s) if f1s else 0.0),
        'pres_f1_micro': float(f1_score(pres_true.reshape(-1), pres_preds.reshape(-1), average='micro', zero_division=0)),
        'pres_f1_macro': float(f1_score(pres_true.reshape(-1), pres_preds.reshape(-1), average='macro', zero_division=0))
    }
    return metrics

def train_loop(train_loader, val_loader, model, optimizer, scheduler, args, per_aspect_weights):
    device = args.device
    model.to(device)
    scaler = GradScaler() if args.fp16 and device.startswith('cuda') else None
    bce = nn.BCEWithLogitsLoss()
    per_aspect_loss = []
    for w in per_aspect_weights:
        w_t = w.to(device) if w is not None else None
        per_aspect_loss.append(FocalLossIgnore(gamma=args.fgamma, weight=w_t, ignore_index=-1))
    best_score = None
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        # freeze encoder early if requested
        if epoch < args.freeze_epochs:
            for p in model.encoder.parameters(): p.requires_grad = False
            print(f"Epoch {epoch+1}: encoder frozen")
        else:
            for p in model.encoder.parameters(): p.requires_grad = True

        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            seg_t = batch['seg'].to(device); pres_t = batch['pres'].to(device)

            if scaler:
                with autocast():
                    seg_logits, pres_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    B,A,C = seg_logits.shape
                    loss_seg = 0.0
                    for a in range(A):
                        loss_seg += per_aspect_loss[a](seg_logits[:,a,:], seg_t[:,a])
                    loss_seg = loss_seg / max(1, A)
                    loss_pres = bce(pres_logits, pres_t)
                    loss = args.w_seg * loss_seg + args.w_pres * loss_pres
                scaler.scale(loss).backward()
                if (step+1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if scheduler: scheduler.step()
            else:
                seg_logits, pres_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                B,A,C = seg_logits.shape
                loss_seg = 0.0
                for a in range(A):
                    loss_seg += per_aspect_loss[a](seg_logits[:,a,:], seg_t[:,a])
                loss_seg = loss_seg / max(1, A)
                loss_pres = bce(pres_logits, pres_t)
                loss = args.w_seg * loss_seg + args.w_pres * loss_pres
                loss.backward()
                if (step+1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()
                    if scheduler: scheduler.step()

            total_loss += float(loss.detach().cpu().item())
            pbar.set_postfix({'loss': total_loss / (step+1)})
            global_step += 1

        val_metrics = validate(val_loader, model, device=args.device)
        print(f"Epoch {epoch+1} val metrics:", val_metrics)
        score = val_metrics['pres_f1_micro'] + val_metrics['seg_f1_macro_mean']
        if best_score is None or score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_backbone_model.pt")
            print("Saved best_backbone_model.pt")
    print("Training finished. Best score:", best_score)

# ---------- predict ----------
def predict_and_save(model, loader, device='cpu', threshold=0.5, out_path="data/predictions.csv"):
    model.to(device); model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            seg_logits, pres_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # 0..C-1
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            pres_preds = (pres_probs >= threshold).astype(int)
            B = seg_preds.shape[0]
            for i in range(B):
                row = {}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i,j])
                    seg = int(seg_preds[i,j]) + 1 if present else 0
                    row[col] = seg
                rows.append(row)
    df_out = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print("Saved predictions to", out_path)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--do_augment", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--bt_mid", type=str, default="en")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--lr_encoder", type=float, default=LR_ENCODER)
    parser.add_argument("--lr_heads", type=float, default=LR_HEADS)
    parser.add_argument("--hidden_head", type=int, default=512)
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--pool_heads", type=int, default=8)
    parser.add_argument("--adapter_bottleneck", type=int, default=128)
    parser.add_argument("--use_bilstm", type=lambda x: x.lower()=='true', default=True)
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_ENCODER_EPOCHS)
    parser.add_argument("--fgamma", type=float, default=1.5)
    parser.add_argument("--w_seg", type=float, default=1.5)
    parser.add_argument("--w_pres", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(42)
    data_dir = Path(args.data_dir)
    train_csv = data_dir / "problem_train.csv"
    val_csv = data_dir / "problem_val.csv"
    test_csv = data_dir / "problem_test.csv"
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    if args.do_augment:
        aug_path = do_offline_back_translation(train, cache_dir=CACHE_DIR, mid_lang=args.bt_mid, bt_batch=BT_BATCH)
        train = pd.read_csv(aug_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train, tokenizer, max_len=MAX_LEN)
    val_ds = ReviewDataset(val, tokenizer, max_len=MAX_LEN)
    test_ds = ReviewDataset(test, tokenizer, max_len=MAX_LEN)

    sampler = make_multi_aspect_sampler(train)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = BackboneWithBiLSTMAndAttention(
        backbone_name=MODEL_NAME,
        num_aspects=NUM_ASPECTS,
        num_seg_classes=NUM_SEGMENT_CLASSES,
        use_bilstm=args.use_bilstm,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=1,
        pool_heads=args.pool_heads,
        adapter_bottleneck=args.adapter_bottleneck,
        hidden_head=args.hidden_head,
        dropout=0.2
    )

    per_aspect_weights = compute_per_aspect_weights(train)

    # optimizer with different lrs
    encoder_params = list(model.encoder.parameters())
    head_params = [p for n,p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': args.lr_encoder},
                                   {'params': head_params, 'lr': args.lr_heads}], weight_decay=WEIGHT_DECAY)
    total_steps = max(1, (len(train_loader) // max(1, args.grad_accum)) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    # attach some args to pass into train_loop
    args.fp16 = args.fp16
    args.device = args.device
    args.epochs = args.epochs
    args.grad_accum = args.grad_accum
    args.freeze_epochs = args.freeze_epochs
    args.fgamma = args.fgamma
    args.w_seg = args.w_seg
    args.w_pres = args.w_pres

    train_loop(train_loader, val_loader, model, optimizer, scheduler, args, per_aspect_weights)

    # load best model and predict
    if Path("best_backbone_model.pt").exists():
        model.load_state_dict(torch.load("best_backbone_model.pt", map_location=args.device))
    predict_and_save(model, test_loader, device=args.device, out_path=str(data_dir/"predictions.csv"))

if __name__ == "__main__":
    main()
