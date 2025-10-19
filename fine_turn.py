#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune_multilabel_updated.py

- Load_data()
- Clean_and_normalize_data()
- paraphrase_augment_for_labels()  # uses T5/mT5
- Class Multi_label_Model
- train()
- test()

Usage examples (basic):
 python fine_tune_multilabel_updated.py --data /mnt/data/train-problem.csv --do_augment --augment_model_name VietAI/t5-small-paraphrase
 python fine_tune_multilabel_updated.py --data /mnt/data/train-problem-paraphrased.csv --no_augment
"""

import os
import random
import argparse
import unicodedata
import regex as re
from typing import List, Dict, Tuple, Optional
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# Try optional sentence-transformers for semantic filtering
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_ST = True
except Exception:
    HAS_ST = False

RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#########################
# 1. Load_data
#########################
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

#########################
# 2. Clean_and_normalize_data
#########################
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
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0,5)
    return df

#########################
# 3. Paraphrase augmentation (T5 / mT5) - only for samples with label>0
#########################
def paraphrase_generate(texts: List[str], model_name: str = "VietAI/t5-small-paraphrase",
                        num_return_sequences: int = 3, max_length: int = 128,
                        do_sample: bool = True, top_p: float = 0.9, temperature: float = 0.8,
                        device: Optional[str] = None) -> List[List[str]]:
    """
    Generate paraphrases for a list of texts. Returns a list (len=texts) of lists of paraphrases.
    Requires model available locally or will download from HF.
    """
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
            # gen shape: (batch_size * num_return_sequences, seq_len)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            # group per input
            grouped = [decoded[j:j+num_return_sequences] for j in range(0, len(decoded), num_return_sequences)]
            results.extend(grouped)
    return results  # list of lists

def semantic_filter_paraphrases(origs: List[str], cands_list: List[List[str]],
                                model_name: str = "all-mpnet-base-v2",
                                min_sim: float = 0.72, max_sim: float = 0.95,
                                device: Optional[str] = None) -> List[List[Tuple[str, float]]]:
    """
    If sentence-transformers is installed, compute cosine similarity and keep paraphrases
    with min_sim <= sim <= max_sim. Returns list of list of (paraphrase, sim).
    """
    if not HAS_ST:
        # If not available, return all with sim=None
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
                                  max_samples_per_label_class: int = 1000) -> pd.DataFrame:
    """
    Augment only rows where any label > 0. Strategy:
     - For each label, for classes 1..5 that are under target (target = majority_count * ratio),
       sample existing rows of that (label==cls), generate paraphrases, filter semantically,
       and append paraphrase rows (keep same labels).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df_aug_rows = []
    stats_created = {lbl: {i:0 for i in range(6)} for lbl in LABEL_COLUMNS}

    # compute counts
    for lbl in LABEL_COLUMNS:
        vc = df[lbl].value_counts().to_dict()
        counts = [vc.get(i, 0) for i in range(6)]
        majority = max(counts)
        target = max(int(majority * target_per_class_ratio), 1)
        for cls in range(1,6):
            current = counts[cls]
            if current >= target:
                continue
            deficit = target - current
            candidates = df[df[lbl] == cls]
            if len(candidates) == 0:
                continue
            # limit samples we will process to avoid explosion
            n_to_make = min(deficit, max_samples_per_label_class)
            # sample rows with replacement
            sampled = candidates.sample(n=n_to_make, replace=True, random_state=RANDOM_SEED)
            texts = sampled['text'].tolist()
            # generate paraphrases
            grouped_pars = paraphrase_generate(texts, model_name=augment_model_name,
                                               num_return_sequences=num_return_sequences,
                                               device=device)
            # semantic filter
            filtered = semantic_filter_paraphrases(texts, grouped_pars,
                                                  model_name=sentence_emb_model,
                                                  min_sim=min_sim, max_sim=max_sim,
                                                  device=device)
            # for each original, append kept paraphrases as new rows
            for (idx, row), kept in zip(sampled.iterrows(), filtered):
                for cand, sim in kept:
                    new_row = row.copy()
                    new_row['text'] = cand
                    df_aug_rows.append(new_row)
                    stats_created[lbl][cls] += 1
    if len(df_aug_rows) == 0:
        print("No paraphrase augmentation created (maybe sentence-transformer not available or filter strict).")
        if save_path:
            df.to_csv(save_path, index=False)
            print("Saved original to", save_path)
        return df
    df_aug = pd.concat([df, pd.DataFrame(df_aug_rows)], ignore_index=True)
    if save_path:
        df_aug.to_csv(save_path, index=False)
        print("Saved augmented dataset to", save_path)
    print(f"Created {len(df_aug_rows)} paraphrase augmented rows.")
    return df_aug

#########################
# Dataset & Model
#########################
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

class Multi_label_Model(nn.Module):
    def __init__(self, model_name: str = 'vinai/phobert-base', num_labels: int = len(LABEL_COLUMNS), hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 6) for _ in range(num_labels)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last)
            pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        logits = [head(pooled) for head in self.heads]
        logits = torch.stack(logits, dim=1)  # [batch, num_labels, 6]
        loss = None
        if labels is not None:
            loss = 0.0  # computed externally in train loop to support custom CE+MAE per head
        return {'loss': loss, 'logits': logits}

#########################
# Utils: class weights & sampler
#########################
def compute_per_label_class_weights(df: pd.DataFrame, labels: List[str] = LABEL_COLUMNS, device='cpu'):
    weight_dict = {}
    for lbl in labels:
        y = df[lbl].astype(int).values
        classes = np.arange(6)
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
        total = len(df)
        invfreq[lbl] = {cls: (1.0 / (freqs[lbl].get(cls, 0) + 1e-9)) for cls in range(6)}
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

#########################
# Train & Test
#########################
def test(model_or_path, dataloader=None, device: str = None, return_preds: bool = True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(model_or_path, str):
        model = Multi_label_Model().to(device)
        model.load_state_dict(torch.load(model_or_path, map_location=device))
    else:
        model = model_or_path
    if dataloader is None:
        raise ValueError('dataloader is required')
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs['logits']
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = np.argmax(all_logits, axis=-1)
    metrics = {'accuracy_per_label': {}, 'f1_macro_per_label': {}, 'mae_per_label': {}}
    for i, lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:, i], preds[:, i]))
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:, i], preds[:, i], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:, i], preds[:, i]))
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))
    if return_preds:
        return {**metrics, 'preds': preds, 'labels': all_labels}
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
          lr: float = 2e-5,
          max_length: int = 256,
          val_size: float = 0.1,
          device: str = None,
          use_focal: bool = False,
          use_sampler: bool = True,
          mae_weight: float = 0.5):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # optionally augment with paraphrase for label>0 only
    if augment:
        print("Starting paraphrase augmentation (only for label>0). This may take time and GPU/CPU resources.")
        df = paraphrase_augment_for_labels(df,
                                          augment_model_name=augment_model_name,
                                          sentence_emb_model=sentence_emb_model,
                                          min_sim=0.72, max_sim=0.95,
                                          num_return_sequences=3,
                                          device=device,
                                          save_path=None)
        print("Augmented dataset size:", len(df))

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=RANDOM_SEED)

    class_weights = compute_per_label_class_weights(train_df, labels=LABEL_COLUMNS, device=device)
    print("Per-label class weights (mean normalized):")
    for k, v in class_weights.items():
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

    model = Multi_label_Model(model_name=model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    # prepare focal if requested
    focal_losses = {}
    if use_focal:
        for lbl in LABEL_COLUMNS:
            focal_losses[lbl] = lambda logits, targets, w=class_weights[lbl]: focal_loss_multiclass(logits, targets, gamma=2.0, weight=w)

    best_val_f1 = 0.0
    os.makedirs(out_path, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        train_losses = []
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [B, num_labels]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs['logits']  # [B, num_labels, 6]

            per_head_losses = []
            per_head_mae = []
            for i, lbl in enumerate(LABEL_COLUMNS):
                logit_i = logits[:, i, :]  # [B,6]
                target_i = labels[:, i]    # [B]
                # CrossEntropy with class weights
                ce_loss_f = nn.CrossEntropyLoss(weight=class_weights[lbl])
                ce_i = ce_loss_f(logit_i, target_i)
                # MAE on soft expectation
                probs = F.softmax(logit_i, dim=-1)
                exp = (probs * torch.arange(6.0).to(probs.device)).sum(dim=-1)
                mae_i = torch.abs(exp - target_i.float()).mean()
                per_head_losses.append(ce_i)
                per_head_mae.append(mae_i)
            loss_ce = torch.stack(per_head_losses).mean()
            loss_mae = torch.stack(per_head_mae).mean()
            loss = loss_ce + mae_weight * loss_mae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(train_losses))})

        # Validation
        val_metrics = test(model, val_loader, device=device, return_preds=False)
        print(f"Epoch {epoch} validation: ", val_metrics)
        val_f1 = val_metrics.get('f1_macro', 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out_path, 'best_model.pt'))
            print(f"Saved new best model at epoch {epoch} with f1_macro={val_f1:.4f}")

    print('Training complete. Best f1_macro:', best_val_f1)
    return model, tokenizer

#########################
# Focal loss helper (multiclass)
#########################
def focal_loss_multiclass(logits, targets, gamma=2.0, weight=None, reduction='mean'):
    """
    logits: [B, C], targets: [B]
    weight: tensor [C] or None
    """
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    nll = F.nll_loss(logp, targets, weight=weight, reduction='none')
    pt = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = ((1 - pt) ** gamma) * nll
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

#########################
# Main entry
#########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./train-problem.csv')
    parser.add_argument('--out_dir', type=str, default='./model_out')
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--do_augment', action='store_true', help='Run paraphrase augmentation (only for label>0)')
    parser.add_argument('--augment_model_name', type=str, default='VietAI/t5-small-paraphrase')
    parser.add_argument('--sentence_emb_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--mae_weight', type=float, default=0.5)
    args = parser.parse_args()

    print("Loading data...")
    df = Load_data(args.data)
    df = Clean_and_normalize_data(df)
    print("Data shape:", df.shape)
    # quick label distribution
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    model, tokenizer = train(df,
                             model_name=args.model_name,
                             augment=args.do_augment,
                             augment_model_name=args.augment_model_name,
                             sentence_emb_model=args.sentence_emb_model,
                             do_paraphrase_filter=HAS_ST,
                             out_path=args.out_dir,
                             epochs=args.epochs,
                             batch_size=args.bs,
                             lr=args.lr,
                             max_length=args.max_len,
                             use_focal=args.use_focal,
                             use_sampler=args.use_sampler,
                             mae_weight=args.mae_weight)

    # Evaluate best model on validation split
    best_path = os.path.join(args.out_dir, 'best_model.pt')
    if os.path.exists(best_path):
        print("Evaluating best model on validation split...")
        df_for_eval = Clean_and_normalize_data(Load_data(args.data))
        # use same split as train(): we cannot reconstruct, so make a new split for quick eval
        _, val_df = train_test_split(df_for_eval, test_size=0.1, random_state=RANDOM_SEED)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        val_loader = DataLoader(MultiLabelDataset(val_df['text'].tolist(), val_df[LABEL_COLUMNS].values.astype(int), tokenizer, max_length=args.max_len),
                                batch_size=args.bs, shuffle=False)
        results = test(best_path, dataloader=val_loader, return_preds=True)
        print("Final evaluation on validation:", results)
    else:
        print("No best_model.pt found (training may not have improved).")
