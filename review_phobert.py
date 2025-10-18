"""
JointPhoBERT advanced training & inference script (fixed)

- Fixes included:
  * ReviewDataset: auto-detect text column ("Review" or alternatives) and safer label handling
  * predict_and_save: robust checkpoint loading (handles wrappers, 'module.' prefix), infers LSTM params or uses args, uses provided backbone/tokenizer, and accepts lstm args from CLI
  * main(): passes lstm args into predict
  * helpful debug prints for checkpoint keys and CSV columns

Usage: same as before. Replace your old file with this one or copy/paste.
"""

import os
import math
import argparse
from typing import List
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score
import re

# --------------------------- Utilities ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------- Dataset ---------------------------
class ReviewDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len=128, aspect_cols: List[str]=None, infer_mode=False, text_col: str = None):
        self.df = pd.read_csv(csv_path)
        self.infer_mode = infer_mode
        # detect text column if not provided
        possible_text_cols = ['Review', 'review']
        if text_col is not None and text_col in self.df.columns:
            self.text_col = text_col
        else:
            found = None
            for c in possible_text_cols:
                if c in self.df.columns:
                    found = c
                    break
            if found is None:
                # fallback: try first column that's not obviously an aspect label
                lower_aspects = set(['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam'])
                non_aspect = [c for c in self.df.columns if c.lower() not in lower_aspects]
                if len(non_aspect) > 0:
                    found = non_aspect[0]
                    print(f"[Warning] Couldn't find standard text column; using '{found}' as text column.")
                else:
                    raise KeyError("No text column found in CSV. Expected one of: " + ", ".join(possible_text_cols))
            self.text_col = found

        if aspect_cols is None:
            # guess aspect cols as typical ones in your project
            aspect_cols = [c for c in self.df.columns if c not in [self.text_col]]
        self.aspect_cols = aspect_cols
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col]) if pd.notnull(row[self.text_col]) else ''
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }
        if not self.infer_mode:
            missing = [c for c in self.aspect_cols if c not in self.df.columns]
            if missing:
                # create dummy labels (won't be used in predict mode)
                item['labels'] = torch.zeros(len(self.aspect_cols), dtype=torch.long)
            else:
                labels = row[self.aspect_cols].astype(int).values
                item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

# --------------------------- CORAL helpers ---------------------------
def labels_to_coral_targets(labels: torch.LongTensor, num_classes: int):
    # labels: [B, A] with values 1..K
    B, A = labels.shape
    K = num_classes
    thresholds = torch.arange(1, K, device=labels.device).view(1, 1, -1)
    lab = labels.unsqueeze(-1)
    targets = (lab > thresholds).long()
    return targets.float()


def coral_logits_to_pred(logits: torch.Tensor):
    # logits: [B, A, K-1]
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).sum(dim=-1) + 1
    return preds

# --------------------------- Model ---------------------------
class JointPhoBERT_CoRAL(nn.Module):
    def __init__(self, backbone_name: str, num_aspects: int, num_classes: int = 5,
                 lstm_hidden: int = 256, lstm_layers: int = 1, lstm_dropout: float = 0.1,
                 freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.hidden_size = hidden_size
        self.num_aspects = num_aspects
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # BiLSTM on top of token embeddings
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout if lstm_layers>1 else 0.0)
        # project BiLSTM output back to hidden_size
        self.project = nn.Sequential(
            nn.Linear(2 * lstm_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # per-aspect attention: a learnable query vector per aspect
        self.aspect_queries = nn.Parameter(torch.randn(num_aspects, hidden_size))

        # final head: for CORAL (K-1 logits per aspect)
        self.coral_head = nn.Linear(hidden_size, num_classes - 1)

        # small dropout for pooled vectors
        self.dropout = nn.Dropout(dropout)

    def aspect_attention_pool(self, token_states: torch.Tensor, attention_mask: torch.Tensor):
        # token_states: [B, T, H], attention_mask: [B, T]
        B, T, H = token_states.shape
        attn_mask = attention_mask.unsqueeze(-1)
        queries = self.aspect_queries.unsqueeze(0).expand(B, -1, -1)
        scores = torch.einsum('bah, bth -> bat', queries, token_states)

        valid_mask = attn_mask.squeeze(-1).unsqueeze(1) != 0  # [B, 1, T]
        neg_val = torch.tensor(torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(~valid_mask, neg_val)

        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum('bat, bth -> bah', weights, token_states)
        return pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(token_states)
        proj = self.project(lstm_out)
        per_aspect_vecs = self.aspect_attention_pool(proj, attention_mask)
        per_aspect_vecs = self.dropout(per_aspect_vecs)
        B, A, H = per_aspect_vecs.shape
        flat = per_aspect_vecs.view(B * A, H)
        logits = self.coral_head(flat)
        logits = logits.view(B, A, -1)
        return logits

# --------------------------- Training / Eval ---------------------------

def compute_metrics_all_aspects(y_true: np.ndarray, y_pred: np.ndarray):
    A = y_true.shape[1]
    res = {}
    qwk_list = []
    mae_list = []
    acc_list = []
    for a in range(A):
        t = y_true[:, a]
        p = y_pred[:, a]
        try:
            qwk = cohen_kappa_score(t, p, weights='quadratic')
        except Exception:
            qwk = float('nan')
        mae = mean_absolute_error(t, p)
        acc = accuracy_score(t, p)
        res[f'aspect_{a}_qwk'] = qwk
        res[f'aspect_{a}_mae'] = mae
        res[f'aspect_{a}_acc'] = acc
        qwk_list.append(qwk)
        mae_list.append(mae)
        acc_list.append(acc)
    res['mean_qwk'] = float(np.nanmean([x for x in qwk_list if not np.isnan(x)])) if len(qwk_list)>0 else float('nan')
    res['mean_mae'] = float(np.mean(mae_list))
    res['mean_acc'] = float(np.mean(acc_list))
    return res


def train_loop(model, dataloader, optimizer, scaler, device, num_classes, loss_fn, scheduler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc='train'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # [B, A]
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B,A,K-1]
            targets = labels_to_coral_targets(labels, num_classes=num_classes).to(device)
            # Label smoothing
            if hasattr(loss_fn, 'label_smoothing') and loss_fn.label_smoothing > 0.0:
                targets = targets * (1.0 - loss_fn.label_smoothing) + 0.5 * loss_fn.label_smoothing
            loss = loss_fn(logits.view(-1, num_classes-1), targets.view(-1, num_classes-1))
        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and not isinstance(scheduler, tuple):
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_loop(model, dataloader, device, num_classes):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B,A,K-1]
            preds = coral_logits_to_pred(logits)
            all_true.append(labels)
            all_pred.append(preds.cpu().numpy())
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    return compute_metrics_all_aspects(y_true, y_pred), y_true, y_pred

# --------------------------- Robust Prediction Function ---------------------------

def _extract_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            return ckpt['state_dict']
        for k in ['model_state_dict', 'model', 'state_dict']:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(sd: dict):
    if any(k.startswith('module.') for k in sd.keys()):
        return {k.replace('module.', '', 1): v for k, v in sd.items()}
    return sd


def _infer_lstm_from_state_dict(sd: dict):
    lstm_keys = [k for k in sd.keys() if k.startswith('lstm.')]
    if not lstm_keys:
        return None
    max_layer = -1
    bidir = False
    hidden = None
    for k in lstm_keys:
        m = re.search(r'lstm\.weight_ih_l(\d+)(_reverse)?$', k)
        if m:
            idx = int(m.group(1))
            if idx > max_layer:
                max_layer = idx
            if m.group(2) == '_reverse':
                bidir = True
    layers = max_layer + 1 if max_layer >= 0 else 1
    for cand in ['lstm.weight_ih_l0', 'lstm.weight_ih_l0_reverse']:
        if cand in sd:
            try:
                hidden = sd[cand].shape[0] // 4
                break
            except Exception:
                hidden = None
    if hidden is None:
        for k in lstm_keys:
            if 'weight_ih' in k:
                try:
                    hidden = sd[k].shape[0] // 4
                    break
                except Exception:
                    pass
    if hidden is None:
        hidden = 256
    return int(hidden), int(layers), bool(bidir)


def predict_and_save(model_path: str, test_csv: str, output_csv: str, batch_size=8, max_len=128,
                     lstm_layers: int = None, lstm_hidden: int = None, backbone_name: str = 'muhtasham/bert-small-finetuned-ner-to-multilabel-finer-139'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=False)
    aspect_cols = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
    ds = ReviewDataset(test_csv, tokenizer, max_len=max_len, aspect_cols=aspect_cols, infer_mode=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    raw = torch.load(model_path, map_location='cpu')
    sd = _extract_state_dict_from_checkpoint(raw)
    sd = _strip_module_prefix(sd)

    # debug prints
    lstm_keys = [k for k in sd.keys() if k.startswith('lstm.')]
    print(f"[Debug] checkpoint has {len(lstm_keys)} lstm keys. Example (first 20): {lstm_keys[:20]}")

    inferred = _infer_lstm_from_state_dict(sd)
    if inferred is not None:
        inferred_hidden, inferred_layers, inferred_bidir = inferred
        print(f"[Info] Inferred LSTM from ckpt: hidden={inferred_hidden}, layers={inferred_layers}, bidirectional={inferred_bidir}")
    else:
        inferred_hidden, inferred_layers, inferred_bidir = 256, 1, True
        print("[Info] No LSTM keys found in checkpoint; using defaults.")

    use_lstm_hidden = lstm_hidden if lstm_hidden is not None else inferred_hidden
    use_lstm_layers = lstm_layers if lstm_layers is not None else inferred_layers

    model = JointPhoBERT_CoRAL(backbone_name, num_aspects=len(aspect_cols),
                               num_classes=5, lstm_hidden=use_lstm_hidden, lstm_layers=use_lstm_layers)

    try:
        model.load_state_dict(sd, strict=True)
        print("[Info] Loaded checkpoint with strict=True")
    except RuntimeError as e:
        print("[Warning] strict load failed:", e)
        print("[Info] Retrying load with strict=False (will ignore missing/extra keys).")
        model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()

    preds_all = []
    with torch.no_grad():
        for batch in tqdm(dl, desc='Predict'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = coral_logits_to_pred(logits)
            preds_all.append(preds.cpu().numpy())

    preds_all = np.vstack(preds_all)
    df_out = pd.read_csv(test_csv)
    for i, c in enumerate(aspect_cols):
        df_out[c] = preds_all[:, i]
    # Ensure output columns are: 'stt','giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam'
    output_columns = ['stt'] + aspect_cols
    # If 'stt' does not exist, create a default index
    if 'stt' not in df_out.columns:
        df_out['stt'] = np.arange(1, len(df_out) + 1)
    df_out = df_out[output_columns]
    df_out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# --------------------------- Main & Argparse ---------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    tokenizer = AutoTokenizer.from_pretrained('muhtasham/bert-small-finetuned-ner-to-multilabel-finer-139', use_fast=False)
    aspect_cols = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']

    if args.mode == 'train':
        dataset = ReviewDataset(args.csv, tokenizer, max_len=args.max_len, aspect_cols=aspect_cols)
        n = len(dataset)
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(n * args.train_frac)
        train_idx, val_idx = idx[:split], idx[split:]

        from torch.utils.data import Subset
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model = JointPhoBERT_CoRAL(backbone_name='muhtasham/bert-small-finetuned-ner-to-multilabel-finer-139', num_aspects=len(aspect_cols),
                                   num_classes=args.num_classes, lstm_hidden=args.lstm_hidden,
                                   lstm_layers=args.lstm_layers, freeze_backbone=args.freeze_backbone,
                                   dropout=args.dropout)
        model.to(device)

        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr_backbone},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': args.lr_head},
        ]
        optimizer = AdamW(param_groups, weight_decay=1e-2)

        num_training_steps = math.ceil(len(train_loader) * args.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * num_training_steps), num_training_steps=num_training_steps)
        # Add ReduceLROnPlateau for dynamic LR adjustment
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
        # Optional: label smoothing
        class SmoothBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
            def __init__(self, label_smoothing=0.05, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.label_smoothing = label_smoothing
        loss_fn = SmoothBCEWithLogitsLoss(label_smoothing=0.05)

        best_qwk = -1e9
        best_loss = float('inf')
        patience = 6  # Early stopping patience
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss = train_loop(model, train_loader, optimizer, scaler, device, args.num_classes, loss_fn, scheduler)
            print(f'Train loss: {train_loss:.5f}')
            metrics, y_true, y_pred = eval_loop(model, val_loader, device, args.num_classes)
            print(f'Val metrics: {metrics}')
            val_qwk = metrics['mean_qwk']
            val_loss = metrics['mean_mae']  # Optionally use MAE as proxy for val loss

            # Save best QWK model
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                torch.save(model.state_dict(), args.output)
                print(f'Saved best QWK model to {args.output}')
                patience_counter = 0
            # Save best loss model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), args.output.replace('.pt', '_bestloss.pt'))
                print(f"Saved best loss model to {args.output.replace('.pt', '_bestloss.pt')}")

            if val_qwk <= best_qwk:
                patience_counter += 1
                print(f"No improvement in QWK. Patience: {patience_counter}/{patience}")

            # Step ReduceLROnPlateau
            plateau_scheduler.step(val_qwk)

            if patience_counter >= patience:
                print(f"Early stopping: no improvement in QWK for {patience} epochs.")
                break

            if train_loss < 0.01:
                print('Training loss below 0.01, stopping early.')
                break

        print('Training finished. Best mean QWK:', best_qwk)
    elif args.mode == 'predict':
        if args.model_path is None or args.test_csv is None:
            raise ValueError('model_path and test_csv required for predict mode')
        predict_and_save(args.model_path, args.test_csv, args.output_csv, batch_size=args.batch_size, max_len=args.max_len,
                         lstm_layers=args.lstm_layers, lstm_hidden=args.lstm_hidden,
                         backbone_name='muhtasham/bert-small-finetuned-ner-to-multilabel-finer-139')
    else:
        raise ValueError('mode must be one of [train, predict]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train')
    parser.add_argument('--csv', type=str, default='./train-problem.csv')
    parser.add_argument('--test_csv', type=str, default="./gt_reviews.csv")
    parser.add_argument('--output', type=str, default='../best_checkpoint.pt')
    parser.add_argument('--output_csv', type=str, default='predict.csv')
    parser.add_argument('--model_path', type=str, default='../best_checkpoint.pt')
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr_backbone', type=float, default=5e-5)
    parser.add_argument('--lr_head', type=float, default=2e-4)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--train_frac', type=float, default=0.85)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
