#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_csv.py - Inference script
Outputs CSV in structure:
stt,giai_tri,luu_tru,nha_hang,an_uong,van_chuyen,mua_sam
1,0,4,4,0,0,0
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import joblib
import glob

# Constants (must match fine_turn)
LABEL_COLUMNS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
K_CLASSES = 6

# Try import model definitions from fine_turn if available (optional)
try:
    from fine_turn import MultiTaskModel, clean_text, SimpleMLPEnsemble
    print("Imported MultiTaskModel and SimpleMLPEnsemble from fine_turn.py")
except Exception as e:
    print("Warning: couldn't import classes from fine_turn.py ({}). Using local fallbacks.".format(e))
    # minimal fallbacks
    from transformers import AutoModel
    import torch.nn as nn
    import torch.nn.functional as F

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
            x = self.ln1(x); x = self.act(x); x = self.drop(x)
            x = self.fc2(x); x = self.ln2(x); x = self.act(x); x = self.drop(x)
            logits = torch.stack([h(x) for h in self.class_heads], dim=1)  # [B,L,K]
            regs = torch.stack([h(x).squeeze(-1) for h in self.reg_heads], dim=1)  # [B,L]
            return {'logits': logits, 'regs': regs}

    class SimpleMLPEnsemble(nn.Module):
        def __init__(self, input_dim, hidden=512, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.heads = nn.ModuleList([nn.Linear(hidden//2, K_CLASSES) for _ in LABEL_COLUMNS])
        def forward(self, x):
            h = self.net(x)
            outs = [head(h) for head in self.heads]
            return torch.stack(outs, dim=1)  # [B,L,K]

# Helper to load CSV and prepare texts
def load_texts_from_csv(path: str, textcol: str = None):
    df = pd.read_csv(path)
    # heuristics for text column
    if textcol is None:
        possible = ['text','review','content','Review']
        for c in possible:
            if c in df.columns:
                textcol = c; break
        if textcol is None:
            textcol = df.columns[0]
    texts = df[textcol].astype(str).fillna("").tolist()
    return df, texts

# Finetune inference
def predict_finetune(data_csv, out_csv, model_name, ckpt_path, device='cpu', max_len=128, bs=32):
    print("Finetune inference. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskModel(backbone_name=model_name)
    sd = torch.load(ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        if isinstance(sd, dict) and 'model' in sd:
            model.load_state_dict(sd['model'])
        else:
            model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()})
    model.to(device); model.eval()

    df, texts = load_texts_from_csv(data_csv)
    # tokenization batched
    all_preds = []
    for i in range(0, len(texts), bs):
        batch_text = [t for t in texts[i:i+bs]]
        enc = tokenizer(batch_text, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out['logits'].cpu().numpy()  # [B,L,K]
            preds = np.argmax(logits, axis=-1)  # [B,L]
        all_preds.append(preds)
    if len(all_preds) == 0:
        raise RuntimeError("No data found in CSV or batch size too large.")
    all_preds = np.vstack(all_preds)  # shape [N, L]
    N = all_preds.shape[0]
    # build output dataframe with required structure
    out_df = pd.DataFrame()
    out_df['stt'] = np.arange(1, N+1, dtype=int)
    for j, lbl in enumerate(LABEL_COLUMNS):
        out_df[lbl] = all_preds[:, j].astype(int)
    # Ensure column order exactly as requested
    cols = ['stt'] + LABEL_COLUMNS
    out_df = out_df[cols]
    out_df.to_csv(out_csv, index=False)
    print("Saved predictions to", out_csv)
    return out_csv

# Embedding utilities (reuse logic)
def compute_embeddings_from_texts(texts, model_name: str, device='cpu', batch=128, max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # [B,T,H]
            attn = enc['attention_mask'].unsqueeze(-1).type_as(last)
            mean = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            maxv, _ = (last * attn).max(dim=1)
            emb = torch.cat([mean, maxv], dim=1)  # [B, 2H]
            embs.append(emb.cpu().numpy())
    embs = np.vstack(embs)
    return embs

# emb_mlp inference
def predict_emb_mlp(data_csv, out_csv, model_name, emb_path=None, mlp_dir=None, stacker_pkl=None, device='cpu', batch=256):
    df, texts = load_texts_from_csv(data_csv)
    device = device
    # load or compute embeddings
    if emb_path and os.path.exists(emb_path):
        print("Loading embeddings from", emb_path)
        z = np.load(emb_path, allow_pickle=True)
        if 'emb' in z:
            embs = z['emb']
        else:
            raise RuntimeError("emb file does not contain 'emb' array.")
    else:
        print("No emb_path provided or file missing. Computing embeddings from texts (this may be slow).")
        embs = compute_embeddings_from_texts(texts, model_name, device=device, batch=128)
        if emb_path:
            np.savez_compressed(emb_path, emb=embs)
            print("Saved embeddings to", emb_path)

    # try to locate mlp model files
    mlp_paths = []
    if mlp_dir and os.path.isdir(mlp_dir):
        mlp_paths = sorted(glob.glob(os.path.join(mlp_dir, "mlp_model_*.pth")))
        print(f"Found {len(mlp_paths)} mlp model checkpoints in {mlp_dir}")
    else:
        print("No mlp_dir provided or directory missing; will try to find mlp_model_*.pth in current dir.")
        mlp_paths = sorted(glob.glob("mlp_model_*.pth"))
        print(f"Found {len(mlp_paths)} in cwd.")

    stackers = None
    if stacker_pkl and os.path.exists(stacker_pkl):
        stackers = joblib.load(stacker_pkl)
        print("Loaded stackers from", stacker_pkl)
    else:
        # try common filenames
        for candidate in ["stackers.pkl","stackers.joblib","stacker.pkl","emb_stackers.pkl"]:
            if os.path.exists(candidate):
                stackers = joblib.load(candidate); print("Loaded stackers from", candidate); break

    device_t = torch.device(device)
    # If we have mlp checkpoints AND stackers -> do full pipeline
    if len(mlp_paths) > 0 and stackers is not None:
        print("Full mlp+stacker inference path.")
        mlp_models = []
        probs_list = []
        for p in mlp_paths:
            sd = torch.load(p, map_location='cpu')
            # infer input dim and hidden from state_dict: find first linear weight (net.0.weight)
            keys = list(sd.keys())
            in_dim = None
            hidden = None
            for k in keys:
                if k.endswith('.weight'):
                    w = sd[k]
                    if w.ndim == 2:
                        out_dim, in_dim_candidate = w.shape
                        in_dim = in_dim_candidate
                        hidden = out_dim
                        break
            if in_dim is None or hidden is None:
                raise RuntimeError(f"Cannot infer MLP dims from state_dict keys for {p}")
            m = SimpleMLPEnsemble(input_dim=in_dim, hidden=hidden)
            try:
                m.load_state_dict(sd)
            except Exception as e:
                try:
                    sd2 = {k.replace('module.',''):v for k,v in sd.items()}
                    m.load_state_dict(sd2)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load state_dict for {p}: {e}; fallback: {e2}")
            m.to(device_t); m.eval()
            mlp_models.append(m)
            # run and get probs
            cur_probs = []
            with torch.no_grad():
                for i in range(0, embs.shape[0], batch):
                    xb = torch.from_numpy(embs[i:i+batch]).float().to(device_t)
                    logits = m(xb).cpu().numpy()  # [b,L,K]
                    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-12)
                    cur_probs.append(probs)
            cur_probs = np.vstack(cur_probs)
            probs_list.append(cur_probs)
        # Stack: for each label use stacker to predict class
        N = embs.shape[0]
        final_preds = np.zeros((N, len(LABEL_COLUMNS)), dtype=int)
        for j in range(len(LABEL_COLUMNS)):
            feats = np.concatenate([p[:, j, :] for p in probs_list], axis=1)
            lr = stackers[j]
            if hasattr(lr, 'predict'):
                pred_j = lr.predict(feats)
            else:
                mean_prob = np.mean([p[:, j, :] for p in probs_list], axis=0)
                pred_j = np.argmax(mean_prob, axis=-1)
            final_preds[:, j] = np.asarray(pred_j, dtype=int)
        # build output dataframe with required structure
        out_df = pd.DataFrame()
        out_df['stt'] = np.arange(1, N+1, dtype=int)
        for j, lbl in enumerate(LABEL_COLUMNS):
            out_df[lbl] = final_preds[:, j].astype(int)
        cols = ['stt'] + LABEL_COLUMNS
        out_df = out_df[cols]
        out_df.to_csv(out_csv, index=False)
        print("Saved emb_mlp preds to", out_csv)
        return out_csv

    # Else fallback: try to use emb_stack_artifacts.npz or val_probs if available
    possible_artifact = None
    if emb_path:
        candidate = os.path.join(os.path.dirname(emb_path), 'emb_stack_artifacts.npz')
        if os.path.exists(candidate):
            possible_artifact = candidate
    for name in ['emb_stack_artifacts.npz','out/emb_stack_artifacts.npz']:
        if os.path.exists(name):
            possible_artifact = name; break

    if possible_artifact:
        print("Using fallback artifacts from", possible_artifact)
        z = np.load(possible_artifact, allow_pickle=True)
        if 'val_probs' in z:
            try:
                probs_list = list(z['val_probs'])
                # probs_list assumed to be list of arrays [N, L, K]
                stacked = np.stack(probs_list, axis=0)  # [n_models, N, L, K]
                mean_over_models = np.mean(stacked, axis=0)  # [N,L,K]
                preds = np.argmax(mean_over_models, axis=-1)  # [N,L]
                N = preds.shape[0]
                out_df = pd.DataFrame()
                out_df['stt'] = np.arange(1, N+1, dtype=int)
                for j, lbl in enumerate(LABEL_COLUMNS):
                    out_df[lbl] = preds[:, j].astype(int)
                cols = ['stt'] + LABEL_COLUMNS
                out_df = out_df[cols]
                out_df.to_csv(out_csv, index=False)
                print("Saved fallback averaged preds to", out_csv)
                return out_csv
            except Exception as e:
                print("Fallback averaging failed:", e)
        else:
            print("Artifact doesn't contain 'val_probs'. Cannot produce reliable predictions.")
    # Last resort: produce simple baseline (all zeros) and warn
    print("WARNING: Could not find mlp checkpoints or stackers. Producing dummy zero predictions.")
    N = len(texts)
    out_df = pd.DataFrame()
    out_df['stt'] = np.arange(1, N+1, dtype=int)
    for lbl in LABEL_COLUMNS:
        out_df[lbl] = 0
    cols = ['stt'] + LABEL_COLUMNS
    out_df = out_df[cols]
    out_df.to_csv(out_csv, index=False)
    print("Saved dummy preds to", out_csv)
    return out_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['finetune','emb_mlp'], required=True)
    parser.add_argument('--data', type=str, default="gt_reviews.csv", help='CSV with texts')
    parser.add_argument('--out', type=str, default= "preds.csv", help='Output CSV path')
    parser.add_argument('--model', type=str, default='vinai/phobert-base')
    parser.add_argument('--ckpt', type=str, default='./out/best_finetune.pt', help='finetune checkpoint')
    parser.add_argument('--emb_path', type=str, default='./out_emb/emb_meanmax.npz')
    parser.add_argument('--mlp_dir', type=str, default='./out/mlps', help='folder with mlp_model_*.pth')
    parser.add_argument('--stacker', type=str, default='./out/stackers.pkl', help='joblib stackers file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.mode == 'finetune':
        predict_finetune(args.data, args.out, args.model, args.ckpt, device=args.device)
    else:
        predict_emb_mlp(args.data, args.out, args.model, emb_path=args.emb_path, mlp_dir=args.mlp_dir, stacker_pkl=args.stacker, device=args.device)

if __name__ == '__main__':
    main()
