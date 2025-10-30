#!/usr/bin/env python3
# train_then_predict.py
"""
Train MultiTaskTransformer and optional CustomModelRegressor, then run prediction and save CSV.

Usage examples:
  Train MT and predict:
    python train_then_predict.py --do_train_mt --train data/train.csv --val data/val.csv --mt_out best_mt.pt --epochs 5 \
      --do_predict --test data/problem_test.csv --out data/predict.csv --mt_checkpoint best_mt.pt

  Train regressor and predict combined:
    python train_then_predict.py --do_train_reg --train data/train.csv --val data/val.csv --reg_out best_reg.pt \
      --do_predict --test data/problem_test.csv --out data/predict.csv --mt_checkpoint best_mt.pt --reg_checkpoint best_reg.pt
"""

import argparse
from pathlib import Path
import os
import copy
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.optim as optim

# Try import from user-provided new_model.py
try:
    from new_model import MultiTaskTransformer, ReviewDataset, collate_fn, LABEL_COLS, NUM_SEGMENT_CLASSES, MODEL_NAME, MAX_LEN, BATCH_SIZE
except Exception as e:
    # Provide fallback definitions if some names are missing
    print("Warning importing from new_model.py:", e)
    # MultiTaskTransformer must exist in new_model.py or training will fail
    try:
        from new_model import MultiTaskTransformer
    except Exception as e2:
        raise ImportError("MultiTaskTransformer not found in new_model.py. Please ensure new_model.py defines MultiTaskTransformer.") from e2

    # Fallback small dataset that expects label columns present in train CSV
    class ReviewDataset(Dataset):
        def __init__(self, df, tokenizer, max_len=256, label_cols=None, is_train=True):
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.texts = self.df.get("Review", self.df.get("review", self.df.get("text", [""] * len(self.df)))).astype(str).tolist()
            self.is_train = is_train
            self.label_cols = label_cols or ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
            if self.is_train:
                # Expect label columns exist and are 0..K ints
                self.labels = self.df[self.label_cols].fillna(0).astype(int).values
            else:
                self.labels = None
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            t = self.texts[idx]
            tok = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            item = {
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0),
            }
            if self.is_train:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "labels" in batch[0]:
            labels = torch.stack([b["labels"] for b in batch], dim=0)
            out["labels"] = labels
        return out

    LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
    NUM_SEGMENT_CLASSES = 5
    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 32

# Define CustomModelRegressor from user's snippet (frozen encoder + linear head)
class CustomModelRegressor(nn.Module):
    def __init__(self, checkpoint, num_outputs):
        super(CustomModelRegressor, self).__init__()
        self.num_outputs = num_outputs
        config = AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(checkpoint, config=config)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        hidden_size = getattr(self.model.config, "hidden_size", 768)
        self.output1 = nn.Linear(hidden_size * 4, num_outputs)
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = outputs.hidden_states  # tuple of (layer_count, B, seq_len, hidden)
        # take last 4 layers' [CLS] token
        pooled = torch.cat((hs[-1][:,0,:], hs[-2][:,0,:], hs[-3][:,0,:], hs[-4][:,0,:]), dim=-1)
        pooled = self.dropout(pooled)
        out = self.output1(pooled)  # shape (B, num_outputs)
        out = torch.sigmoid(out) * 5.0
        return out

# Utilities for checkpoint loading/saving
def load_state_dict_flexible(model, path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    # common wrappers
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    # remove module. prefix
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model

# Training function for MultiTaskTransformer
def train_mt(train_csv, val_csv, model_out, epochs=3, batch_size=32, lr=2e-5, device="cuda"):
    print("Train MT: train=", train_csv, "val=", val_csv, "out=", model_out)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(df_train, tokenizer, max_len=MAX_LEN, label_cols=LABEL_COLS, is_train=True)
    val_ds = ReviewDataset(df_val, tokenizer, max_len=MAX_LEN, label_cols=LABEL_COLS, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskTransformer(MODEL_NAME, num_aspects=len(LABEL_COLS), num_seg_classes=NUM_SEGMENT_CLASSES)
    model.to(device)

    # losses: seg -> CrossEntropy per aspect, pres -> BCEWithLogits (multi-label)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_score = -1e9
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # shape (B, A) ints 0..K

            optim.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # try extract seg_logits, pres_logits (robust)
            seg_logits = None
            pres_logits = None
            if isinstance(outputs, dict):
                seg_logits = outputs.get("seg_logits") or outputs.get("seg")
                pres_logits = outputs.get("pres_logits") or outputs.get("pres")
            elif isinstance(outputs, (list, tuple)):
                tensors = [o for o in outputs if torch.is_tensor(o)]
                if len(tensors) >= 2:
                    seg_logits, pres_logits = tensors[0], tensors[1]
                else:
                    # try first two positions
                    try:
                        seg_logits, pres_logits = outputs[0], outputs[1]
                    except Exception:
                        raise RuntimeError("Cannot parse outputs from MultiTaskTransformer during train.")
            else:
                raise RuntimeError("Unsupported outputs type from MultiTaskTransformer.")

            # compute losses
            # seg_logits shape: (B, A, C) ; labels: (B, A)
            loss_seg = 0.0
            B, A = labels.shape
            for a in range(A):
                loss_seg = loss_seg + ce(seg_logits[:, a, :], labels[:, a])
            loss_seg = loss_seg / float(A)

            # pres_logits shape: (B, A) logits
            pres_target = (labels > 0).float()  # presence if label > 0
            loss_pres = bce(pres_logits, pres_target)

            loss = loss_seg + loss_pres
            # if model returns reg losses/outputs you can add MSE here (skip for now)
            loss.backward()
            optim.step()
            running += loss.item()

        # validation
        model.eval()
        total_acc = []
        total_mae = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict):
                    seg_logits = outputs.get("seg_logits") or outputs.get("seg")
                    pres_logits = outputs.get("pres_logits") or outputs.get("pres")
                else:
                    tensors = [o for o in outputs if torch.is_tensor(o)]
                    seg_logits, pres_logits = tensors[0], tensors[1]
                seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # (B,A)
                pres_preds = (torch.sigmoid(pres_logits).cpu().numpy() >= 0.5).astype(int)
                labels_np = labels.cpu().numpy()
                # decode to final scores like inference: if pres==0 -> 0 else seg+1
                B = labels_np.shape[0]
                pred_scores = np.zeros_like(labels_np)
                for i in range(B):
                    for j in range(labels_np.shape[1]):
                        if pres_preds[i, j] == 0:
                            pred_scores[i, j] = 0
                        else:
                            pred_scores[i, j] = int(seg_preds[i, j]) + 1
                acc = (pred_scores == labels_np).mean()
                mae = np.mean(np.abs(pred_scores - labels_np))
                total_acc.append(acc)
                total_mae.append(mae)
        mean_acc = float(np.mean(total_acc))
        mean_mae = float(np.mean(total_mae))
        score = mean_acc - 0.5 * mean_mae
        print(f"Epoch {ep}/{epochs} train_loss={running/len(train_loader):.4f} val_acc={mean_acc:.4f} val_mae={mean_mae:.4f} score={score:.4f} time={(time.time()-t0):.1f}s")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), model_out)
            print("Saved best MT checkpoint to", model_out)
    return model_out

# Training function for CustomModelRegressor (train linear head)
def train_regressor(train_csv, val_csv, reg_out, epochs=3, batch_size=32, lr=1e-3, device="cuda"):
    print("Train regressor:", train_csv, val_csv, "->", reg_out)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # For regressor we can reuse ReviewDataset but with labels as float targets (0..5)
    train_ds = ReviewDataset(df_train, tokenizer, max_len=MAX_LEN, label_cols=LABEL_COLS, is_train=True)
    val_ds = ReviewDataset(df_val, tokenizer, max_len=MAX_LEN, label_cols=LABEL_COLS, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    reg_model = CustomModelRegressor(MODEL_NAME, num_outputs=len(LABEL_COLS))
    reg_model.to(device)
    # Only train linear head
    params = [p for n,p in reg_model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr)
    loss_fn = nn.L1Loss()

    best_mae = 1e9
    for ep in range(1, epochs+1):
        reg_model.train()
        running = 0.0
        t0 = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)  # (B,A) expected 0..5
            optimizer.zero_grad()
            preds = reg_model(input_ids=input_ids, attention_mask=attention_mask)  # (B,A)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        # val
        reg_model.eval()
        maes = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)
                preds = reg_model(input_ids=input_ids, attention_mask=attention_mask)
                mae = torch.mean(torch.abs(preds - labels)).item()
                maes.append(mae)
        mean_mae = float(np.mean(maes))
        print(f"Epoch {ep}/{epochs} train_loss={running/len(train_loader):.4f} val_mae={mean_mae:.4f} time={(time.time()-t0):.1f}s")
        if mean_mae < best_mae:
            best_mae = mean_mae
            torch.save(reg_model.state_dict(), reg_out)
            print("Saved best regressor to", reg_out)
    return reg_out

# Combined prediction (MT +/- regressor) -> output CSV format required
def predict(mt_checkpoint, test_csv, out_csv, device="cuda", reg_checkpoint=None):
    print("Predict: mt_checkpoint=", mt_checkpoint, "reg_checkpoint=", reg_checkpoint, "test=", test_csv, "out=", out_csv)
    df_test = pd.read_csv(test_csv)
    if "stt" not in df_test.columns:
        df_test.insert(0, "stt", range(1, len(df_test) + 1))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # PredictDataset lightweight
    class _PredDS(Dataset):
        def __init__(self, df, tokenizer, max_len=MAX_LEN):
            self.df = df.reset_index(drop=True)
            self.texts = self.df.get("Review", self.df.get("review", self.df.get("text", [""] * len(self.df)))).astype(str).tolist()
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.stt = self.df["stt"].astype(int).tolist()
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            tok = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            return {"input_ids": tok["input_ids"].squeeze(0), "attention_mask": tok["attention_mask"].squeeze(0), "stt": self.stt[idx]}
    ds = _PredDS(df_test, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: {"input_ids": torch.stack([x["input_ids"] for x in b]), "attention_mask": torch.stack([x["attention_mask"] for x in b]), "stt": [x["stt"] for x in b]})

    # load MT model
    mt_model = MultiTaskTransformer(MODEL_NAME, num_aspects=len(LABEL_COLS), num_seg_classes=NUM_SEGMENT_CLASSES)
    mt_model = load_state_dict_flexible(mt_model, mt_checkpoint, map_location=device)
    mt_model.to(device)
    mt_model.eval()

    reg_model = None
    if reg_checkpoint:
        # instantiate regressor and load state if checkpoint exists
        reg_model = CustomModelRegressor(MODEL_NAME, num_outputs=len(LABEL_COLS))
        try:
            reg_model = load_state_dict_flexible(reg_model, reg_checkpoint, map_location=device)
        except Exception:
            # maybe reg_checkpoint is model id - then CustomModelRegressor will load AutoModel weights from that id
            try:
                reg_model = CustomModelRegressor(reg_checkpoint, num_outputs=len(LABEL_COLS))
            except Exception as e:
                print("Warning: couldn't load regressor from", reg_checkpoint, "->", e)
                reg_model = None
        if reg_model is not None:
            reg_model.to(device)
            reg_model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out_mt = mt_model(input_ids=input_ids, attention_mask=attention_mask)
            # extract seg_logits and pres_logits robustly
            seg_logits = None; pres_logits = None; reg_scores_mt = None
            if isinstance(out_mt, dict):
                seg_logits = out_mt.get("seg_logits") or out_mt.get("seg")
                pres_logits = out_mt.get("pres_logits") or out_mt.get("pres")
                reg_scores_mt = out_mt.get("reg_scores") or out_mt.get("reg")
            elif isinstance(out_mt, (list, tuple)):
                tensors = [o for o in out_mt if torch.is_tensor(o)]
                if len(tensors) >= 2:
                    seg_logits, pres_logits = tensors[0], tensors[1]
                    if len(tensors) >= 3:
                        reg_scores_mt = tensors[2]
                else:
                    # fallback: try positions
                    try:
                        seg_logits = out_mt[0]; pres_logits = out_mt[1]
                        if len(out_mt) >= 3:
                            reg_scores_mt = out_mt[2]
                    except Exception:
                        raise RuntimeError("Cannot parse mt_model outputs during predict.")

            if seg_logits is None or pres_logits is None:
                raise RuntimeError("seg_logits or pres_logits not found from MT model during predict.")

            seg_logits = seg_logits.detach().cpu()
            pres_logits = pres_logits.detach().cpu()
            seg_preds = torch.argmax(seg_logits, dim=-1).numpy()  # (B, A)
            pres_probs = torch.sigmoid(pres_logits).numpy()       # (B, A)
            pres_preds = (pres_probs >= 0.5).astype(int)

            # reg predictions from reg_model if provided, else from MT reg_scores if available
            reg_preds_arr = None
            if reg_model is not None:
                reg_out = reg_model(input_ids=input_ids, attention_mask=attention_mask)  # (B, A)
                reg_preds_arr = reg_out.detach().cpu().numpy()
            elif reg_scores_mt is not None and torch.is_tensor(reg_scores_mt):
                reg_preds_arr = reg_scores_mt.detach().cpu().numpy()

            B = seg_preds.shape[0]
            for i in range(B):
                stt = int(batch["stt"][i])
                row = {"stt": stt}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i, j])
                    cls_val = int(seg_preds[i, j]) + 1  # classifier -> 1..C
                    if reg_preds_arr is not None:
                        reg_val = float(np.clip(reg_preds_arr[i, j], 0.0, 5.0))
                    else:
                        reg_val = float(cls_val)
                    reg_round = int(np.round(reg_val))
                    reg_round = max(0, min(5, reg_round))
                    if present == 0:
                        outv = 0
                    else:
                        diff = abs(reg_round - cls_val)
                        if diff <= 1:
                            outv = int(round((reg_round + cls_val) / 2.0))
                        else:
                            outv = int(cls_val)
                        outv = max(1, min(NUM_SEGMENT_CLASSES, outv))
                    row[col] = outv
                rows.append(row)
    df_out = pd.DataFrame(rows, columns=["stt"] + list(LABEL_COLS))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved predictions to:", out_csv)
    return out_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--do_train_mt", action="store_true")
    p.add_argument("--do_train_reg", action="store_true")
    p.add_argument("--do_predict", action="store_true")
    p.add_argument("--train", type=str, help="Training CSV (must contain label cols)")
    p.add_argument("--val", type=str, help="Validation CSV")
    p.add_argument("--test", type=str, help="Test CSV (for prediction)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--epochs_reg", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr_reg", type=float, default=1e-3)
    p.add_argument("--mt_out", type=str, default="best_mt.pt")
    p.add_argument("--reg_out", type=str, default="best_reg.pt")
    p.add_argument("--mt_checkpoint", type=str, default=None)
    p.add_argument("--reg_checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default="predict.csv")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    if args.do_train_mt:
        if not args.train or not args.val:
            raise ValueError("Train and val CSV must be provided for training MT.")
        train_mt(args.train, args.val, args.mt_out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
        args.mt_checkpoint = args.mt_out

    if args.do_train_reg:
        if not args.train or not args.val:
            raise ValueError("Train and val CSV must be provided for training regressor.")
        train_regressor(args.train, args.val, args.reg_out, epochs=args.epochs_reg, batch_size=args.batch_size, lr=args.lr_reg, device=device)
        args.reg_checkpoint = args.reg_out

    if args.do_predict:
        if args.mt_checkpoint is None:
            raise ValueError("mt_checkpoint must be provided for prediction (use --mt_checkpoint or run --do_train_mt).")
        predict(args.mt_checkpoint, args.test, args.out, device=device, reg_checkpoint=args.reg_checkpoint)

if __name__ == "__main__":
    main()
