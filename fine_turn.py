#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune_coral_paraphrase.py

Giải pháp tích hợp:
 - Load_data(), Clean_and_normalize_data()
 - Paraphrase augmentation (T5/mT5) targeted cho label>0 (semantic filter optional)
 - Multi-label ordinal model (CORAL) + presence head (multi-task)
 - Loss = BCE ordinal (CORAL) + presence BCE + MAE (on expectation) weighted
 - Class weights per-head + WeightedRandomSampler
 - Test & metrics same as trước
"""

import os
import random
import argparse
import unicodedata
import regex as re
from typing import List, Optional, Tuple
import math
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# Optional semantic filter (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_ST = True
except Exception:
    HAS_ST = False

# -----------------------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # 0..5
# -----------------------

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========== Load & Clean ==========
def Load_data(path: str = "/mnt/data/train-problem.csv", text_col_candidates: List[str] = None) -> pd.DataFrame:
    if text_col_candidates is None:
        text_col_candidates = ["text", "review", "Review", "review_text", "content"]
    df = pd.read_csv(path)
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    df = df.rename(columns={text_col: "text"})
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            df[lbl] = 0
    return df[["text"] + LABEL_COLUMNS]

def _remove_control_chars(s: str) -> str:
    return ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'C')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _remove_control_chars(s)
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    # keep letters, numbers, punctuation, whitespace
    s = re.sub(r"[^\w\s\.\,\!\?\-:;'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def Clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text'] = df['text'].astype(str).apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df

# ========== Paraphrase augmentation (targeted) ==========
def paraphrase_generate(texts: List[str], model_name: str = "VietAI/t5-small-paraphrase",
                        num_return_sequences: int = 3, max_length: int = 128,
                        do_sample: bool = True, top_p: float = 0.9, temperature: float = 0.8,
                        device: Optional[str] = None) -> List[List[str]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    results = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
            gen = model.generate(**inputs,
                                 max_length=max_length,
                                 num_return_sequences=num_return_sequences,
                                 do_sample=do_sample,
                                 top_p=top_p,
                                 temperature=temperature,
                                 num_beams=max(1, min(10, num_return_sequences*2)),
                                 early_stopping=True,
                                 no_repeat_ngram_size=3)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            grouped = [decoded[j:j+num_return_sequences] for j in range(0, len(decoded), num_return_sequences)]
            results.extend(grouped)
    return results

def semantic_filter_paraphrases(origs: List[str], cands_list: List[List[str]],
                                model_name: str = "all-mpnet-base-v2",
                                min_sim: float = 0.72, max_sim: float = 0.95,
                                device: Optional[str] = None) -> List[List[Tuple[str, float]]]:
    if not HAS_ST:
        # return all with None score if sentence-transformers not available
        return [[(c, None) for c in cands] for cands in cands_list]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer(model_name, device=device)
    results = []
    for orig, cands in zip(origs, cands_list):
        embeddings = embedder.encode([orig] + cands, convert_to_tensor=True)
        orig_emb = embeddings[0]
        cand_embs = embeddings[1:]
        sims = st_util.pytorch_cos_sim(orig_emb, cand_embs).cpu().numpy().squeeze(0)
        kept = []
        for cand, sim in zip(cands, sims):
            if sim >= min_sim and sim <= max_sim:
                kept.append((cand, float(sim)))
        results.append(kept)
    return results

def paraphrase_augment_for_labels(df: pd.DataFrame,
                                  augment_model_name: str = "VietAI/t5-small-paraphrase",
                                  sentence_emb_model: str = "all-mpnet-base-v2",
                                  min_sim: float = 0.72, max_sim: float = 0.95,
                                  num_return_sequences: int = 3,
                                  target_per_class_ratio: float = 0.20,
                                  device: Optional[str] = None,
                                  save_path: Optional[str] = None,
                                  max_samples_per_label_class: int = 500) -> pd.DataFrame:
    """
    Augment rows for classes 1..5 only (targeted). Return augmented dataframe.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df_aug_rows = []
    stats = {lbl: {i:0 for i in range(K_CLASSES)} for lbl in LABEL_COLUMNS}

    for lbl in LABEL_COLUMNS:
        vc = df[lbl].value_counts().to_dict()
        counts = [vc.get(i, 0) for i in range(K_CLASSES)]
        majority = max(counts)
        target = max(int(majority * target_per_class_ratio), 1)
        for cls in range(1, K_CLASSES):
            current = counts[cls]
            if current >= target:
                continue
            deficit = target - current
            candidates = df[df[lbl] == cls]
            if len(candidates) == 0:
                continue
            n_to_make = min(deficit, max_samples_per_label_class)
            sampled = candidates.sample(n=n_to_make, replace=True, random_state=RANDOM_SEED)
            texts = sampled['text'].tolist()
            grouped = paraphrase_generate(texts, model_name=augment_model_name,
                                          num_return_sequences=num_return_sequences, device=device)
            filtered = semantic_filter_paraphrases(texts, grouped, model_name=sentence_emb_model,
                                                  min_sim=min_sim, max_sim=max_sim, device=device)
            for (_, row), kept in zip(sampled.iterrows(), filtered):
                for cand, sim in kept:
                    new_row = row.copy()
                    new_row['text'] = cand
                    df_aug_rows.append(new_row)
                    stats[lbl][cls] += 1
    if len(df_aug_rows) == 0:
        if save_path:
            df.to_csv(save_path, index=False)
        print("No paraphrase created.")
        return df
    df_new = pd.concat([df, pd.DataFrame(df_aug_rows)], ignore_index=True)
    if save_path:
        df_new.to_csv(save_path, index=False)
        print("Saved augmented dataset to", save_path)
    print(f"Paraphrase augmented rows created: {len(df_aug_rows)}")
    return df_new

# ========== Dataset ==========
class MultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ========== Model with CORAL ordinal heads + presence ==========
class OrdinalHead(nn.Module):
    def __init__(self, hidden_size: int, K: int = K_CLASSES):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(hidden_size, K-1)  # K-1 binary logits

    def forward(self, x):
        # x: [B, hidden]
        logits = self.linear(x)  # [B, K-1]
        return logits

class Multi_label_Ordinal_Model(nn.Module):
    def __init__(self, model_name: str = 'vinai/phobert-base', num_labels: int = len(LABEL_COLUMNS), dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.ord_heads = nn.ModuleList([OrdinalHead(hidden_size, K=K_CLASSES) for _ in range(num_labels)])
        self.presence_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])
        # optional regression head (predict expectation)
        self.reg_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last)
            pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        ord_logits = [head(pooled) for head in self.ord_heads]          # list of [B, K-1]
        pres_logits = [head(pooled).squeeze(-1) for head in self.presence_heads]  # list of [B]
        reg_outs = [head(pooled).squeeze(-1) for head in self.reg_heads]  # list of [B]
        ord_logits = torch.stack(ord_logits, dim=1)   # [B, L, K-1]
        pres_logits = torch.stack(pres_logits, dim=1) # [B, L]
        reg_outs = torch.stack(reg_outs, dim=1)       # [B, L]
        return {'ord_logits': ord_logits, 'pres_logits': pres_logits, 'reg_outs': reg_outs}

# ========== Utils: convert labels to ordinal targets ==========
def to_ordinal_targets(y: torch.Tensor, K: int = K_CLASSES):
    # y: [B] ints 0..K-1
    B = y.size(0)
    out = torch.zeros(B, K-1, device=y.device)
    for k in range(1, K):
        out[:, k-1] = (y >= k).float()
    return out  # float targets for BCEWithLogits

# ========== Class weights & sampler ==========
def compute_per_label_class_weights(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, device='cpu'):
    weight_dict = {}
    for lbl in labels:
        y = df[lbl].astype(int).values
        classes = np.arange(K_CLASSES)
        present = np.unique(y)
        if len(present) < len(classes):
            cw = np.ones(len(classes), dtype=float)
            if len(present) > 0:
                cw_present = compute_class_weight(class_weight='balanced', classes=present, y=y)
                for c, w in zip(present, cw_present):
                    cw[int(c)] = float(w)
                maxw = float(np.max(cw_present)) if len(cw_present)>0 else 1.0
                for c in classes:
                    if c not in present:
                        cw[int(c)] = maxw
        else:
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y).astype(float)
        cw = cw / np.mean(cw)
        weight_dict[lbl] = torch.tensor(cw, dtype=torch.float).to(device)
    return weight_dict

def make_weighted_sampler(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS):
    freqs = {lbl: df[lbl].value_counts().to_dict() for lbl in labels}
    invfreq = {}
    for lbl in labels:
        invfreq[lbl] = {cls: (1.0 / (freqs[lbl].get(cls, 0) + 1e-9)) for cls in range(K_CLASSES)}
    weights = []
    for idx, row in df.iterrows():
        sample_weight = 0.0
        for lbl in labels:
            cls = int(row[lbl])
            w = invfreq[lbl].get(cls, 0.0)
            if cls == 0:
                w = w * 0.1
            sample_weight = max(sample_weight, w)
        if sample_weight <= 0:
            sample_weight = 1e-6
        weights.append(sample_weight)
    weights = np.array(weights, dtype=float)
    weights = weights / np.mean(weights)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

# ========== Training & Evaluation ==========
def test(model_or_path, dataloader=None, device: Optional[str] = None, return_preds: bool = True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(model_or_path, str):
        model = Multi_label_Ordinal_Model().to(device)
        model.load_state_dict(torch.load(model_or_path, map_location=device))
    else:
        model = model_or_path
    if dataloader is None:
        raise ValueError('dataloader is required')
    model.eval()
    all_ord_logits = []
    all_pres_logits = []
    all_reg = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_ord_logits.append(out['ord_logits'].cpu().numpy())
            all_pres_logits.append(out['pres_logits'].cpu().numpy())
            all_reg.append(out['reg_outs'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    ord_logits = np.concatenate(all_ord_logits, axis=0)  # [N, L, K-1]
    pres_logits = np.concatenate(all_pres_logits, axis=0)  # [N, L]
    reg_outs = np.concatenate(all_reg, axis=0)  # [N, L]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, L]
    N, L, K1 = ord_logits.shape

    # Convert ordinal logits -> predicted class
    preds = np.zeros((N, L), dtype=int)
    for i in range(N):
        for j in range(L):
            logits_bin = ord_logits[i, j, :]  # length K-1
            probs = 1.0 / (1.0 + np.exp(-logits_bin))  # sigmoid
            # predict class = sum(prob_i > 0.5)  (or threshold 0.5)
            pred_class = int((probs > 0.5).sum())
            preds[i, j] = pred_class
    # reg_post = round/reg_outs clamp
    reg_preds = np.round(np.clip(reg_outs, 0, K_CLASSES-1)).astype(int)

    # choose final pred: if presence predicted low (<0), we can force 0; here we keep ordinal pred
    # compute metrics per label
    metrics = {'accuracy_per_label': {}, 'f1_macro_per_label': {}, 'mae_per_label': {}}
    for j, lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:, j], preds[:, j]))
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:, j], preds[:, j], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:, j], preds[:, j]))
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))
    if return_preds:
        return {**metrics, 'preds': preds, 'labels': all_labels, 'reg_preds': reg_preds}
    else:
        return metrics

def train(df: pd.DataFrame,
          model_name: str = 'vinai/phobert-base',
          augment: bool = False,
          augment_model_name: str = "VietAI/t5-small-paraphrase",
          sentence_emb_model: str = "all-mpnet-base-v2",
          do_paraphrase_filter: bool = True,
          out_path: str = './model_out',
          epochs: int = 3,
          batch_size: int = 16,
          lr_encoder: float = 1e-5,
          lr_heads: float = 1e-4,
          max_length: int = 256,
          val_size: float = 0.1,
          device: Optional[str] = None,
          use_sampler: bool = True,
          mae_weight: float = 0.5,
          presence_weight: float = 0.4):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if augment:
        print("Running paraphrase augmentation (targeted label>0)...")
        df = paraphrase_augment_for_labels(df, augment_model_name, sentence_emb_model,
                                          min_sim=0.72, max_sim=0.95, num_return_sequences=3,
                                          target_per_class_ratio=0.2, device=device, save_path=None)
        print("New dataset size:", len(df))

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=RANDOM_SEED)

    class_weights = compute_per_label_class_weights(train_df, labels=LABEL_COLUMNS, device=device)
    print("Class weights (per-label):")
    for k,v in class_weights.items():
        print(k, v.cpu().numpy())

    y_train = train_df[LABEL_COLUMNS].values.astype(int)
    y_val = val_df[LABEL_COLUMNS].values.astype(int)

    train_dataset = MultiLabelDataset(train_df['text'].tolist(), y_train, tokenizer, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df['text'].tolist(), y_val, tokenizer, max_length=max_length)

    if use_sampler:
        sampler = make_weighted_sampler(train_df, labels=LABEL_COLUMNS)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Multi_label_Ordinal_Model(model_name=model_name).to(device)

    # layer-wise LR groups
    encoder_params = [p for n,p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for p in model.ord_heads.parameters()] + [p for p in model.presence_heads.parameters()] + [p for p in model.reg_heads.parameters()]
    optimizer = AdamW([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': head_params, 'lr': lr_heads}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.L1Loss()  # MAE

    best_val_f1 = 0.0
    os.makedirs(out_path, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        losses_epoch = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [B, L]
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            ord_logits = out['ord_logits']   # [B,L,K-1]
            pres_logits = out['pres_logits'] # [B,L]
            reg_outs = out['reg_outs']       # [B,L] continuous

            per_label_losses = []
            per_label_mae = []
            per_label_pres = []
            for i, lbl in enumerate(LABEL_COLUMNS):
                logit_i = ord_logits[:, i, :]  # [B, K-1]
                target_i = labels[:, i]        # [B]
                # ordinal targets
                ord_t = to_ordinal_targets(target_i, K=K_CLASSES)  # [B, K-1]
                # weighted BCE per threshold: use class weight derived from original distribution:
                # we approximate weight per threshold by weight of class >k vs <=k (simple)
                # but here we use uniform BCE and rely on class weights via sampler and MAE term
                loss_ord = bce(logit_i, ord_t)
                # presence loss: binary label if >0
                pres_target = (target_i > 0).float()
                loss_pres = F.binary_cross_entropy_with_logits(pres_logits[:, i], pres_target)
                # regression MAE
                mae_i = mse(reg_outs[:, i], target_i.float())
                per_label_losses.append(loss_ord + presence_weight * loss_pres)
                per_label_mae.append(mae_i)
            loss_ord_pres = torch.stack(per_label_losses).mean()
            loss_mae = torch.stack(per_label_mae).mean()
            loss = loss_ord_pres + mae_weight * loss_mae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses_epoch.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(losses_epoch))})

        # validation
        val_metrics = test(model, val_loader, device=device, return_preds=False)
        print(f"Epoch {epoch} validation:", val_metrics)
        val_f1 = val_metrics.get('f1_macro', 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out_path, 'best_model.pt'))
            print(f"Saved best model at epoch {epoch} with f1_macro={val_f1:.4f}")

    print("Training complete. Best f1_macro:", best_val_f1)
    return model, tokenizer

# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./train-problem.csv')
    parser.add_argument('--out_dir', type=str, default='./model_out')
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr_enc', type=float, default=1e-5)
    parser.add_argument('--lr_heads', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--do_augment', action='store_true')
    parser.add_argument('--augment_model', type=str, default='VietAI/t5-small-paraphrase')
    parser.add_argument('--sentence_emb_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--mae_weight', type=float, default=0.5)
    parser.add_argument('--presence_weight', type=float, default=0.4)
    args = parser.parse_args()

    print("Loading data...")
    df = Load_data(args.data)
    df = Clean_and_normalize_data(df)
    print("Data shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    model, tokenizer = train(df,
                             model_name=args.model_name,
                             augment=args.do_augment,
                             augment_model_name=args.augment_model,
                             sentence_emb_model=args.sentence_emb_model,
                             out_path=args.out_dir,
                             epochs=args.epochs,
                             batch_size=args.bs,
                             lr_encoder=args.lr_enc,
                             lr_heads=args.lr_heads,
                             max_length=args.max_len,
                             use_sampler=args.use_sampler,
                             mae_weight=args.mae_weight,
                             presence_weight=args.presence_weight)
    # Evaluate best
    best_path = os.path.join(args.out_dir, 'best_model.pt')
    if os.path.exists(best_path):
        print("Evaluating best model on validation split...")
        df_eval = Clean_and_normalize_data(Load_data(args.data))
        _, val_df = train_test_split(df_eval, test_size=0.1, random_state=RANDOM_SEED)
        val_loader = DataLoader(MultiLabelDataset(val_df['text'].tolist(), val_df[LABEL_COLUMNS].values.astype(int),
                                                 tokenizer, max_length=args.max_len),
                                batch_size=args.bs, shuffle=False)
        results = test(best_path, dataloader=val_loader, return_preds=True)
        print("Final evaluation on validation:", results)
    else:
        print("No best_model.pt found.")
