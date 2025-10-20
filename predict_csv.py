#!/usr/bin/env python3
# predict.py
# Usage:
# python predict.py --input /path/to/input.csv --out_preds /path/to/output.csv --model-dir ./out --model-type finetune
#
# Expects fine-tuned model saved as ./out/best_finetune.pt (or best.pt). Tokenizer/model_name default vinai/phobert-base
# If you used emb_mlp pipeline and saved ensemble artifacts (mlp models + stackers), you can extend this script to load them.
import os
import argparse
import unicodedata
import regex as re
from typing import List, Optional
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ----------------------------
# Config that must match training
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6

# ----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = unicodedata.normalize('NFC', s)
    s = s.lower()
    pattern = r"[^\p{L}\p{N}\s\.,!?:;'\-\(\)\"/]+"
    s = re.sub(pattern, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# Model class must match training file's class
# If you modified model architecture in training, adjust here accordingly.
class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name='vinai/phobert-base', proj_dim=512, mid_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.ln1 = nn.LayerNorm(proj_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(proj_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)
        self.class_heads = nn.ModuleList([nn.Linear(mid_dim, K_CLASSES) for _ in LABEL_COLUMNS])
        self.reg_heads = nn.ModuleList([nn.Linear(mid_dim, 1) for _ in LABEL_COLUMNS])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            last = out.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last) if attention_mask is not None else None
            if attn is None:
                pooled = last.mean(dim=1)
            else:
                pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        x = self.proj(pooled)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.drop(x)
        logits = torch.stack([h(x) for h in self.class_heads], dim=1)  # [B, L, K]
        regs = torch.stack([h(x).squeeze(-1) for h in self.reg_heads], dim=1)  # [B, L]
        return {'logits': logits, 'regs': regs}

# ----------------------------
def load_finetune_model(model_dir: str, device: str, model_name_fallback: str = 'vinai/phobert-base'):
    # Look for best_finetune.pt or best.pt
    candidates = ['best_finetune.pt', 'best.pt', 'best_model.pt']
    ckpt_path = None
    for c in candidates:
        p = os.path.join(model_dir, c)
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not find checkpoint in {model_dir}. Looked for {candidates}.")
    # load state dict
    sd = torch.load(ckpt_path, map_location='cpu')
    # try to extract model_name from saved metadata if present
    model_name = model_name_fallback
    if isinstance(sd, dict) and 'model_name' in sd:
        model_name = sd['model_name']
    # instantiate model & load state_dict
    model = MultiTaskModel(backbone_name=model_name)
    # If checkpoint saved state_dict under 'model_state' or similar, auto-detect
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        model.load_state_dict(sd['model_state_dict'])
    elif isinstance(sd, dict) and 'state_dict' in sd:
        model.load_state_dict(sd['state_dict'])
    else:
        # assume sd is raw state_dict
        try:
            model.load_state_dict(sd)
        except Exception as e:
            # try filtering keys with 'module.' prefix
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if k.startswith('module.'):
                    nk = k[len('module.'):]
                new_sd[nk] = v
            model.load_state_dict(new_sd)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# ----------------------------
def predict_finetune(model, tokenizer, texts: List[str], device: str, batch_size: int = 64, max_len: int = 128):
    preds = []
    regs = []
    N = len(texts)
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc='Predicting'):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast('cuda' if device.startswith('cuda') else None):
                out = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
                logits = out['logits'].cpu().numpy()   # [b, L, K]
                reg_out = out['regs'].cpu().numpy()    # [b, L]
                pred_batch = np.argmax(logits, axis=-1)
                # clamp/reg round
                reg_round = np.clip(np.round(reg_out), 0, K_CLASSES - 1).astype(int)
            preds.append(pred_batch)
            regs.append(reg_round)
    preds = np.vstack(preds)
    regs = np.vstack(regs)
    return preds, regs

# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input CSV file path')
    parser.add_argument('--out_preds', type=str, default='./preds.csv')
    parser.add_argument('--model-dir', type=str, default='./out', help='directory containing saved model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['finetune', 'emb_mlp'], default='finetune')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    print("Loading input:", args.input)
    df = pd.read_csv(args.input)
    # detect text column
    text_cols = ['text', 'review', 'content', 'Review']
    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c; break
    if text_col is None:
        text_col = df.columns[0]
    texts_raw = df[text_col].astype(str).fillna("").tolist()
    texts = [clean_text(t) for t in texts_raw]

    if args.model_type == 'finetune':
        print("Loading finetune model from:", args.model_dir)
        model, tokenizer = load_finetune_model(args.model_dir, device)
        print("Tokenizing and predicting ...")
        preds, regs = predict_finetune(model, tokenizer, texts, device, batch_size=args.batch, max_len=args.max_len)
        out_df = df.copy()
        for i, lbl in enumerate(LABEL_COLUMNS):
            out_df[f'pred_{lbl}'] = preds[:, i]
            out_df[f'pred_reg_{lbl}'] = regs[:, i]
        out_df.to_csv(args.out_preds, index=False)
        print("Saved predictions to", args.out_preds)
        print("Prediction distribution:")
        for lbl in LABEL_COLUMNS:
            print(lbl, out_df[f'pred_{lbl}'].value_counts().sort_index().to_dict())
    elif args.model_type == 'emb_mlp':
        import pickle
        emb_path = os.path.join(args.model_dir, 'emb_meanmax.npz')
        artifacts_path = os.path.join(args.model_dir, 'emb_stack_artifacts.npz')
        mlp_models_path = os.path.join(args.model_dir, 'emb_models.pkl')
        stackers_path = os.path.join(args.model_dir, 'emb_stackers.pkl')
        if not os.path.exists(artifacts_path):
            print(f"[WARNING] emb_mlp artifacts not found: {artifacts_path}")
            print("Please run the emb_mlp pipeline and save ensemble artifacts (emb_stack_artifacts.npz) in the model directory.")
            print("Alternatively, use --model-type finetune if you do not have emb_mlp artifacts.")
            return
        print("Computing embeddings for input texts (this may take a few minutes)...")
        model_name = 'vinai/phobert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name).to(device).eval()
        B = args.batch
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), B)):
                batch_texts = texts[i:i+B]
                enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=args.max_len, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                out = encoder(**enc)
                last = out.last_hidden_state
                attn = enc['attention_mask'].unsqueeze(-1).type_as(last)
                mean = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
                maxv, _ = (last * attn).max(dim=1)
                emb = torch.cat([mean, maxv], dim=1)
                embs.append(emb.cpu().numpy())
        embs = np.vstack(embs)
        # Load ensemble models and stackers
        if not os.path.exists(mlp_models_path) or not os.path.exists(stackers_path):
            print("[ERROR] emb_mlp requires emb_models.pkl and emb_stackers.pkl saved in model_dir.")
            print("Please save MLP ensemble models and stackers using the emb_mlp pipeline in fine_turn.py.")
            return
        with open(mlp_models_path, 'rb') as f:
            mlp_models = pickle.load(f)
        with open(stackers_path, 'rb') as f:
            stackers = pickle.load(f)
        # Get logits from each MLP model
        batch_size = args.batch
        all_logits = []
        for m in mlp_models:
            m.eval()
            cur = []
            with torch.no_grad():
                for idx in range(0, embs.shape[0], batch_size):
                    xb = torch.from_numpy(embs[idx:idx+batch_size]).float().to(device)
                    out = m(xb)
                    cur.append(out.cpu().numpy())
            cur = np.vstack(cur)
            all_logits.append(cur)
        # Convert logits to probabilities
        all_probs = [np.exp(l - np.max(l, axis=-1, keepdims=True)) / np.sum(np.exp(l - np.max(l, axis=-1, keepdims=True)), axis=-1, keepdims=True) for l in all_logits]
        # Stacking: for each label, concatenate probs and apply stacker
        N = embs.shape[0]
        stacked_logits = np.zeros((N, len(LABEL_COLUMNS), K_CLASSES), dtype=float)
        for j in range(len(LABEL_COLUMNS)):
            feats = np.concatenate([p[:, j, :] for p in all_probs], axis=1)
            lr = stackers[j]
            if hasattr(lr, 'predict_proba'):
                proba = lr.predict_proba(feats)
                # Map proba to logits
                logits = np.log(proba + 1e-9)
                stacked_logits[:, j, :proba.shape[1]] = logits
            else:
                pred = lr.predict(feats)
                for i in range(N):
                    stacked_logits[i, j, pred[i]] = 1.0
        # Bias tuning (optional, can load biases from artifacts if saved)
        # Here, just take argmax
        final_preds = np.argmax(stacked_logits, axis=-1)
        out_df = df.copy()
        for i, lbl in enumerate(LABEL_COLUMNS):
            out_df[f'pred_{lbl}'] = final_preds[:, i]
        out_df.to_csv(args.out_preds, index=False)
        print("Saved emb_mlp predictions to", args.out_preds)
        print("Prediction distribution:")
        for lbl in LABEL_COLUMNS:
            print(lbl, out_df[f'pred_{lbl}'].value_counts().sort_index().to_dict())

if __name__ == '__main__':
    main()
