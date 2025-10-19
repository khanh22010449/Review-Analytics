#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_tune_vn_coral_lora_adapter.py

Backbone: vinai/phobert-base
Features:
 - Adapter on pooled output
 - Optional LoRA (PEFT) applied to encoder
 - CORAL ordinal heads + presence + regression heads
 - Combined loss: ordinal BCE + presence BCE + alpha * MAE
 - WeightedRandomSampler for imbalance
 - Optional paraphrase augmentation (T5/mT5) + semantic filter
 - Functions: Load_data(), Clean_and_normalize_data(), class Multi_label_Model, train(), test()

Usage example:
 python fine_tune_vn_coral_lora_adapter.py --data /mnt/data/train-problem.csv --model_name vinai/phobert-base --epochs 3 --bs 16 --use_sampler --use_lora --do_augment
"""

import os
import random
import argparse
import unicodedata
import regex as re
from typing import List, Optional, Tuple, Dict, Any
import math
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# Optional libs
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_ST = True
except Exception:
    HAS_ST = False

# ---------- Config ----------
RANDOM_SEED = 42
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6  # 0..5
# --------------------------

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------
# 1) Load & Clean
# --------------------------
def Load_data(path: str = "/mnt/data/train-problem.csv", text_col_candidates: List[str] = None) -> pd.DataFrame:
    """
    Load CSV and normalize column name for text -> 'text'.
    Ensure LABEL_COLUMNS exist (if absent create with zeros).
    """
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
    # keep letters, numbers, simple punctuation, whitespace
    s = re.sub(r"[^\w\s\.\,\!\?\-:;'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def Clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text'] = df['text'].astype(str).apply(clean_text)
    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int).clip(0, K_CLASSES-1)
    return df

# --------------------------
# 2) Paraphrase augmentation (optional)
# --------------------------
def paraphrase_generate(texts: List[str], model_name: str = "VietAI/t5-small-paraphrase",
                        num_return_sequences: int = 3, max_length: int = 128,
                        do_sample: bool = True, top_p: float = 0.9, temperature: float = 0.8,
                        device: Optional[str] = None) -> List[List[str]]:
    """
    Generate paraphrases using seq2seq model.
    Returns list-of-lists aligned with texts.
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
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            grouped = [decoded[j:j+num_return_sequences] for j in range(0, len(decoded), num_return_sequences)]
            results.extend(grouped)
    return results

def semantic_filter_paraphrases(origs: List[str], cands_list: List[List[str]],
                                model_name: str = "all-mpnet-base-v2",
                                min_sim: float = 0.72, max_sim: float = 0.95,
                                device: Optional[str] = None) -> List[List[Tuple[str, float]]]:
    """
    Use sentence-transformers to keep paraphrases with similarity in [min_sim, max_sim].
    If sentence-transformers not installed, returns all with sim=None.
    """
    if not HAS_ST:
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
    Targeted paraphrase augmentation: only augment samples where label > 0 (for each label).
    Return augmented dataframe.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows_aug = []
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
            grouped = paraphrase_generate(texts, model_name=augment_model_name, num_return_sequences=num_return_sequences, device=device)
            filtered = semantic_filter_paraphrases(texts, grouped, model_name=sentence_emb_model, min_sim=min_sim, max_sim=max_sim, device=device)
            for (_, row), kept in zip(sampled.iterrows(), filtered):
                for cand, sim in kept:
                    new_row = row.copy()
                    new_row['text'] = cand
                    rows_aug.append(new_row)
    if len(rows_aug) == 0:
        if save_path:
            df.to_csv(save_path, index=False)
        print("No paraphrase augmentation created.")
        return df
    df_new = pd.concat([df, pd.DataFrame(rows_aug)], ignore_index=True)
    if save_path:
        df_new.to_csv(save_path, index=False)
        print("Saved augmented data to:", save_path)
    print(f"Created {len(rows_aug)} augmented rows.")
    return df_new

# --------------------------
# 3) Dataset
# --------------------------
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

# --------------------------
# 4) Model: Adapter + LoRA + CORAL + presence + regression
# --------------------------
class Adapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        return x + self.dropout(z)

class Multi_label_Model(nn.Module):
    """
    Encoder (AutoModel) with optional LoRA applied externally (model passed may already be peft-wrapped).
    Heads:
     - ord_heads: K-1 logits per label (CORAL)
     - pres_heads: single logit per label (presence)
     - reg_heads: scalar per label (regression)
    """
    def __init__(self, model_name: str = 'vinai/phobert-base', num_labels: int = len(LABEL_COLUMNS), adapter_bottleneck: int = 128, adapter_dropout: float = 0.1):
        super().__init__()
        self.encoder_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.adapter = Adapter(hidden_size, bottleneck=adapter_bottleneck, dropout=adapter_dropout)
        self.ord_heads = nn.ModuleList([nn.Linear(hidden_size, K_CLASSES-1) for _ in range(num_labels)])
        self.pres_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])
        self.reg_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_labels)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last)
            pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        pooled = self.adapter(pooled)  # adapter on pooled representation

        ord_logits = [head(pooled) for head in self.ord_heads]   # list of [B, K-1]
        pres_logits = [head(pooled).squeeze(-1) for head in self.pres_heads]  # list of [B]
        reg_outs = [head(pooled).squeeze(-1) for head in self.reg_heads]  # list of [B]
        ord_logits = torch.stack(ord_logits, dim=1)  # [B, L, K-1]
        pres_logits = torch.stack(pres_logits, dim=1)  # [B, L]
        reg_outs = torch.stack(reg_outs, dim=1)  # [B, L]
        return {'ord_logits': ord_logits, 'pres_logits': pres_logits, 'reg_outs': reg_outs}

# --------------------------
# Helpers: ordinal conversion, sampler, class weights
# --------------------------
def to_ordinal_targets(y: torch.Tensor, K: int = K_CLASSES):
    """
    Convert y [B] ints 0..K-1 to ordinal targets [B, K-1] where t_k = 1 if y >= k
    """
    B = y.size(0)
    out = torch.zeros(B, K-1, device=y.device)
    for k in range(1, K):
        out[:, k-1] = (y >= k).float()
    return out

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
                maxw = float(np.max(cw_present)) if len(cw_present) > 0 else 1.0
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

# --------------------------
# LoRA utility: auto-detect target modules and apply
# --------------------------
def auto_detect_lora_target_modules(model):
    """
    Try to find likely attention projection module names inside a HF model.
    Returns a list of substrings to match module names, e.g. ['q_proj','v_proj','k_proj'].
    This is heuristic — you can pass explicit target_modules if detection fails.
    """
    names = [n for n, _ in model.named_modules()]
    candidates = set()
    for n in names:
        ln = n.lower()
        # common patterns
        for pat in ['q_proj','k_proj','v_proj','o_proj','q','k','v','proj','dense','query','key','value','attention']:
            if pat in ln:
                candidates.add(pat)
    # create prioritized list (unique)
    # choose a small set to avoid matching many unrelated modules
    prefer = ['q_proj','k_proj','v_proj','o_proj','query','key','value','dense','proj']
    found = []
    for p in prefer:
        if any(p in n.lower() for n in names):
            found.append(p)
    # fallback
    if not found:
        found = ['query','key','value']
    # deduplicate
    return list(dict.fromkeys(found))

def apply_lora_to_model(base_model, r=8, alpha=32, dropout=0.05, target_modules: Optional[List[str]] = None):
    """
    Wrap base_model with PEFT LoRA. Returns peft-model.
    """
    if not HAS_PEFT:
        raise RuntimeError("PEFT is not installed. pip install peft")
    # detect targets if not provided
    if target_modules is None:
        target_modules = auto_detect_lora_target_modules(base_model)
    print("Applying LoRA to target_modules:", target_modules)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model

# --------------------------
# 5) Test / Eval
# --------------------------
def test(model_or_path, dataloader=None, device: Optional[str] = None, return_preds: bool = True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(model_or_path, str):
        model = Multi_label_Model()  # requires args default, but user normally passes object
        model.load_state_dict(torch.load(model_or_path, map_location=device))
        model.to(device)
    else:
        model = model_or_path
    if dataloader is None:
        raise ValueError("dataloader is required")
    model.eval()
    all_preds = []
    all_labels = []
    all_reg_preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            ord_logits = out['ord_logits'].cpu().numpy()  # [B,L,K-1]
            reg_outs = out['reg_outs'].cpu().numpy()      # [B,L]
            pres_logits = out['pres_logits'].cpu().numpy()  # [B,L]
            B,L,K1 = ord_logits.shape
            preds = np.zeros((B, L), dtype=int)
            for i in range(B):
                for j in range(L):
                    logits_bin = ord_logits[i,j,:]
                    probs = 1.0 / (1.0 + np.exp(-logits_bin))
                    pred_class = int((probs > 0.5).sum())
                    preds[i,j] = pred_class
            reg_preds = np.round(np.clip(reg_outs, 0, K_CLASSES-1)).astype(int)
            # combine strategies: prefer ordinal pred, but can fallback to regression if needed
            all_preds.append(preds)
            all_reg_preds.append(reg_preds)
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = {'accuracy_per_label': {}, 'f1_macro_per_label': {}, 'mae_per_label': {}}
    for j, lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:,j], all_preds[:,j]))
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:,j], all_preds[:,j], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:,j], all_preds[:,j]))
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))
    if return_preds:
        return {**metrics, 'preds': all_preds, 'labels': all_labels, 'reg_preds': all_reg_preds}
    else:
        return metrics

# --------------------------
# 6) Train
# --------------------------
def train(df: pd.DataFrame,
          model_name: str = 'vinai/phobert-base',
          out_dir: str = './model_out',
          epochs: int = 3,
          batch_size: int = 16,
          lr_lora: float = 1e-4,
          lr_adapter: float = 2e-4,
          lr_heads: float = 2e-4,
          lr_encoder_top: float = 1e-5,
          max_length: int = 256,
          val_size: float = 0.1,
          device: Optional[str] = None,
          use_lora: bool = True,
          lora_r: int = 8,
          lora_alpha: int = 32,
          lora_dropout: float = 0.05,
          lora_target_modules: Optional[List[str]] = None,
          adapter_bottleneck: int = 128,
          adapter_dropout: float = 0.1,
          use_sampler: bool = True,
          do_augment: bool = False,
          augment_model_name: str = "VietAI/t5-small-paraphrase",
          sentence_emb_model: str = "all-mpnet-base-v2",
          mae_weight: float = 0.5,
          pres_weight: float = 0.4):
    """
    Train pipeline integrating LoRA + Adapter + CORAL + presence + regression
    """
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if do_augment:
        print("Running paraphrase augmentation (targeted label>0)...")
        df = paraphrase_augment_for_labels(df, augment_model_name=augment_model_name, sentence_emb_model=sentence_emb_model, device=device)
        print("Augmented dataset size:", len(df))

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=RANDOM_SEED)
    class_weights = compute_per_label_class_weights(train_df, labels=LABEL_COLUMNS, device=device)
    print("Per-label class weights (normalized mean=1):")
    for k,v in class_weights.items():
        print(k, v.cpu().numpy())

    train_labels = train_df[LABEL_COLUMNS].values.astype(int)
    val_labels = val_df[LABEL_COLUMNS].values.astype(int)

    train_dataset = MultiLabelDataset(train_df['text'].tolist(), train_labels, tokenizer, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df['text'].tolist(), val_labels, tokenizer, max_length=max_length)

    if use_sampler:
        sampler = make_weighted_sampler(train_df, labels=LABEL_COLUMNS)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build base model (AutoModel) and optionally wrap with LoRA (PEFT)
    print("Loading base encoder:", model_name)
    base_encoder = AutoModel.from_pretrained(model_name)
    if use_lora:
        if not HAS_PEFT:
            raise RuntimeError("PEFT not installed. Install via `pip install peft` to use LoRA.")
        if lora_target_modules is None:
            detected = auto_detect_lora_target_modules(base_encoder)
            print("Auto-detected LoRA target modules:", detected)
            lora_target_modules = detected
        base_encoder = apply_lora_to_model(base_encoder, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=lora_target_modules)
        print("LoRA applied. Trainable parameters should be LoRA adapters + heads + adapter module.")
    # wrap into our full model structure
    model = Multi_label_Model(model_name=model_name, adapter_bottleneck=adapter_bottleneck, adapter_dropout=adapter_dropout)
    # replace encoder weights with base_encoder weights (peft-wrapped or not)
    model.encoder = base_encoder
    model.to(device)

    # Freeze original encoder parameters except LoRA params (peft will keep lora params trainable)
    if use_lora and HAS_PEFT:
        print("Freezing non-LoRA encoder parameters...")
        for n, p in model.encoder.named_parameters():
            if 'lora_' in n or 'alpha' in n:
                p.requires_grad = True
            else:
                # PEFT might auto-handle; set requires_grad = False for safety
                p.requires_grad = False
    else:
        # optionally freeze lower layers entirely to avoid overfit
        print("No LoRA: freezing embeddings and lower encoder layers by default")
        for n, p in model.encoder.named_parameters():
            if n.startswith('embeddings.') or 'layer.0' in n or 'layer.1' in n or 'layer.2' in n:
                p.requires_grad = False

    # prepare parameter groups
    lora_params = []
    adapter_params = list(model.adapter.parameters())
    head_params = list(model.ord_heads.parameters()) + list(model.pres_heads.parameters()) + list(model.reg_heads.parameters())

    if use_lora and HAS_PEFT:
        for n,p in model.encoder.named_parameters():
            if p.requires_grad:
                lora_params.append(p)
    else:
        # nothing in encoder_trainable except maybe top layers
        enc_trainable = [p for n,p in model.encoder.named_parameters() if p.requires_grad]
        lora_params = enc_trainable

    optimizer = AdamW([
        {'params': lora_params, 'lr': lr_lora},
        {'params': adapter_params, 'lr': lr_adapter},
        {'params': head_params, 'lr': lr_heads}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    best_val_f1 = 0.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [B, L]
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            ord_logits = out['ord_logits']   # [B,L,K-1]
            pres_logits = out['pres_logits'] # [B,L]
            reg_outs = out['reg_outs']       # [B,L]

            per_label_losses = []
            per_label_mae = []
            for i, lbl in enumerate(LABEL_COLUMNS):
                logit_i = ord_logits[:, i, :]  # [B, K-1]
                tgt_i = labels[:, i]           # [B]
                ord_t = to_ordinal_targets(tgt_i, K=K_CLASSES)  # [B, K-1]
                loss_ord = bce(logit_i, ord_t)
                pres_t = (tgt_i > 0).float()
                loss_pres = F.binary_cross_entropy_with_logits(pres_logits[:, i], pres_t)
                mae_i = l1(reg_outs[:, i], tgt_i.float())
                per_label_losses.append(loss_ord + pres_weight * loss_pres)
                per_label_mae.append(mae_i)
            loss_main = torch.stack(per_label_losses).mean()
            loss_mae = torch.stack(per_label_mae).mean()
            loss = loss_main + mae_weight * loss_mae

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix({'loss': float(np.mean(losses))})

        # validation
        val_metrics = test(model, val_loader, device=device, return_preds=False)
        print(f"Epoch {epoch} validation:", val_metrics)
        val_f1 = val_metrics.get('f1_macro', 0.0)
        # save best by f1_macro
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            print(f"Saved best model (epoch {epoch}) with f1_macro={val_f1:.4f}")

    print("Training complete. Best f1_macro:", best_val_f1)
    return model, tokenizer

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/data/train-problem.csv')
    parser.add_argument('--out_dir', type=str, default='./model_out')
    parser.add_argument('--model_name', type=str, default='vinai/phobert-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--do_augment', action='store_true')
    parser.add_argument('--augment_model', type=str, default='VietAI/t5-small-paraphrase')
    parser.add_argument('--sentence_emb_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--mae_weight', type=float, default=0.5)
    parser.add_argument('--pres_weight', type=float, default=0.4)
    args = parser.parse_args()

    print("Loading dataset...")
    df = Load_data(args.data)
    df = Clean_and_normalize_data(df)
    print("Dataset shape:", df.shape)
    for lbl in LABEL_COLUMNS:
        print(lbl, df[lbl].value_counts().sort_index().to_dict())

    model, tokenizer = train(df,
                             model_name=args.model_name,
                             out_dir=args.out_dir,
                             epochs=args.epochs,
                             batch_size=args.bs,
                             max_length=args.max_len,
                             use_lora=args.use_lora,
                             lora_r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             use_sampler=args.use_sampler,
                             do_augment=args.do_augment,
                             augment_model_name=args.augment_model,
                             sentence_emb_model=args.sentence_emb_model,
                             mae_weight=args.mae_weight,
                             pres_weight=args.pres_weight)
    # evaluate best if exists
    best = os.path.join(args.out_dir, 'best_model.pt')
    if os.path.exists(best):
        print("Loading best model for evaluation...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        df_eval = Clean_and_normalize_data(Load_data(args.data))
        _, val_df = train_test_split(df_eval, test_size=0.1, random_state=RANDOM_SEED)
        val_loader = DataLoader(MultiLabelDataset(val_df['text'].tolist(), val_df[LABEL_COLUMNS].values.astype(int), tokenizer, max_length=args.max_len), batch_size=args.bs, shuffle=False)
        results = test(best, dataloader=val_loader, return_preds=True)
        print("Final evaluation on validation:", results)
    else:
        print("No best model saved.")
