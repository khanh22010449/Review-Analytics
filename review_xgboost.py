#!/usr/bin/env python3
# review_xgboost.py  (multi-task BERT training + inference + overall score)
# Usage:
# Train: python review_xgboost.py --train train-problem.csv --model_name vinai/phobert-base --output_dir ./mt_model --epochs 3
# Predict: python review_xgboost.py --predict --model_dir ./mt_model --test_csv gt_reviews.csv --output_csv predict.csv

import os
import argparse
import numpy as np
import pandas as pd
import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------
ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
NUM_CLASSES_PER_ASPECT = 6  # 0..5
# ---------------------------

def safe_read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def prepare_multitask_dataset(df, text_col='Review'):
    records = []
    for _, row in df.iterrows():
        text = row[text_col] if text_col in row.index else ""
        labels = []
        for a in ASPECT_COLS:
            raw = row.get(a, 0)
            if pd.isna(raw):
                val = 0
            else:
                try:
                    val = int(raw)
                except:
                    val = 0
            val = max(0, min(5, val))
            labels.append(val)
        records.append({'text': str(text), 'labels': labels})
    ds = Dataset.from_pandas(pd.DataFrame(records))
    if 'index' in ds.column_names:
        ds = ds.remove_columns('index')
    return ds

class MultiAspectModel(nn.Module):
    def __init__(self, encoder_name, n_aspects=len(ASPECT_COLS), num_classes=NUM_CLASSES_PER_ASPECT, dropout=0.1):
        super().__init__()
        # encoder_name may be a model id or a local dir with pretrained encoder weights
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_aspects * num_classes)
        self.n_aspects = n_aspects
        self.num_classes = num_classes

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        include 'labels' in signature so Trainer won't drop labels column
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last * mask).sum(1) / (mask.sum(1).clamp(min=1e-9))
        x = self.dropout(pooled)
        logits = self.classifier(x)
        logits = logits.view(-1, self.n_aspects, self.num_classes)
        return logits

# robust metric compute
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """
    eval_pred: (logits, labels)
    flatten and compute accuracy; safe if mismatch lengths
    """
    logits, labels = eval_pred
    if logits is None:
        return {}
    preds = np.argmax(logits, axis=-1).reshape(-1)
    refs = np.array(labels).reshape(-1) if labels is not None else None

    if refs is None:
        return {}
    if preds.shape[0] != refs.shape[0]:
        n = min(preds.shape[0], refs.shape[0])
        print(f"Warning: compute_metrics truncating preds/ref from {preds.shape[0]}/{refs.shape[0]} to {n}.")
        preds = preds[:n]
        refs = refs[:n]
    acc = accuracy_metric.compute(predictions=preds, references=refs)['accuracy']
    return {"accuracy": acc}

# Robust MTTrainer
class MTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # find labels robustly
        labels = None
        if "labels" in inputs:
            labels = inputs.pop("labels")
        elif "label" in inputs:
            labels = inputs.pop("label")
        elif "labels" in kwargs:
            labels = kwargs.pop("labels")
        elif "label" in kwargs:
            labels = kwargs.pop("label")

        if labels is None:
            # try to locate tensor shaped (batch, n_aspects)
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.size(1) == len(ASPECT_COLS):
                    labels = inputs.pop(k)
                    break

        if labels is None:
            raise ValueError(
                "Could not find 'labels' in inputs passed to compute_loss(). "
                "Ensure your HF Dataset has a 'labels' column (list length == n_aspects) and "
                "you called dataset.set_format(type='torch', columns=[...,'labels']). "
                f"Current input keys: {list(inputs.keys())}."
            )

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        labels = labels.to(next(model.parameters()).device)

        logits = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        total_loss = 0.0
        for i in range(logits.size(1)):
            total_loss = total_loss + loss_fct(logits[:, i, :], labels[:, i].long())
        total_loss = total_loss / logits.size(1)
        return (total_loss, logits) if return_outputs else total_loss

def make_safe_training_args(args):
    ta_all = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "evaluation_strategy": "epoch" if args.eval_steps == 0 else "steps",
        "eval_steps": args.eval_steps if args.eval_steps > 0 else None,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_accuracy",
        "fp16": torch.cuda.is_available(),
        "logging_dir": os.path.join(args.output_dir, "logs")
    }

    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())
    valid_params.discard('self')
    ta_filtered = {k: v for k, v in ta_all.items() if k in valid_params}
    ignored = set(ta_all.keys()) - set(ta_filtered.keys())
    if ignored:
        print("Note: the following TrainingArguments keys are unsupported in this environment and will be ignored:", ignored)

    if 'evaluation_strategy' in ta_filtered:
        eval_strat = ta_filtered.get('evaluation_strategy')
        if 'save_strategy' in valid_params:
            ta_filtered['save_strategy'] = eval_strat
    else:
        if 'load_best_model_at_end' in ta_filtered and ta_filtered['load_best_model_at_end']:
            print("Warning: evaluation_strategy unsupported -> setting load_best_model_at_end=False to avoid errors.")
            ta_filtered['load_best_model_at_end'] = False
        if 'metric_for_best_model' in ta_filtered:
            ta_filtered.pop('metric_for_best_model', None)

    return TrainingArguments(**ta_filtered)

def train_main(args):
    df = safe_read_csv(args.train)
    text_col = args.text_col if args.text_col in df.columns else ('Review' if 'Review' in df.columns else df.columns[0])
    print("Using text column:", text_col)

    ds_all = prepare_multitask_dataset(df, text_col=text_col)
    ds = ds_all.train_test_split(test_size=args.test_size, seed=args.seed)
    train_ds = ds['train']
    val_ds = ds['test']

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = MultiAspectModel(args.model_name, n_aspects=len(ASPECT_COLS), num_classes=NUM_CLASSES_PER_ASPECT, dropout=args.dropout)

    # Tokenize datasets ahead and return labels so labels persist
    def tokenize_fn(batch):
        enc = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=args.max_length)
        enc['labels'] = batch['labels']
        return enc

    print("Tokenizing train dataset...")
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    print("Tokenizing val dataset...")
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['text'])

    print("train_ds columns:", train_ds.column_names)
    print("val_ds columns:", val_ds.column_names)

    columns_to_set = [c for c in ['input_ids', 'attention_mask', 'token_type_ids', 'labels'] if c in train_ds.column_names]
    train_ds.set_format(type='torch', columns=columns_to_set)
    val_ds.set_format(type='torch', columns=columns_to_set)

    training_args = make_safe_training_args(args)

    trainer = MTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=None
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved multi-task model to", args.output_dir)

    # Evaluate / save per-aspect predictions on validation set and compute overall score
    print("Running prediction on validation set...")
    preds_out = trainer.predict(val_ds)
    logits = preds_out.predictions  # shape (N_pred, n_aspects, n_classes)
    label_ids = getattr(preds_out, "label_ids", None)

    if logits is None:
        raise RuntimeError("trainer.predict returned no predictions (logits is None).")

    pred_labels = np.argmax(logits, axis=-1)  # (N_pred, n_aspects)

    # True labels: prefer label_ids returned by trainer.predict
    if label_ids is not None:
        lab = np.array(label_ids)
        if lab.ndim == 1:
            if lab.size % len(ASPECT_COLS) == 0:
                lab = lab.reshape(-1, len(ASPECT_COLS))
            else:
                raise RuntimeError("label_ids returned but cannot reshape to (n_examples, n_aspects).")
        true_labels = lab
    else:
        # fallback: build from val_ds
        true_list = [x['labels'] for x in val_ds]
        true_labels = np.vstack(true_list) if len(true_list) > 0 else np.zeros((0, len(ASPECT_COLS)), dtype=int)

    # Align pred and true lengths
    if pred_labels.shape[0] != true_labels.shape[0]:
        n_pred = pred_labels.shape[0]
        n_true = true_labels.shape[0]
        n = min(n_pred, n_true)
        print(f"Warning: Aligning pred ({n_pred}) and true ({n_true}) to {n} examples.")
        pred_labels = pred_labels[:n]
        true_labels = true_labels[:n]

    # Save val preds CSV
    rows = []
    for i in range(pred_labels.shape[0]):
        row = {'idx': int(i)}
        for j, a in enumerate(ASPECT_COLS):
            row[f'{a}_gt'] = int(true_labels[i, j])
            row[f'{a}_pred'] = int(pred_labels[i, j])
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "val_preds.csv")
    out_df.to_csv(out_csv, index=False, encoding='utf-8')
    print("Saved validation predictions to", out_csv)

    # --- Compute metrics according to your rules ---
    # 1) Micro-F1 for presence (presence = label>0)
    gt_presence = (true_labels > 0).astype(int).reshape(-1)
    pred_presence = (pred_labels > 0).astype(int).reshape(-1)
    # avoid zero division: if no positive labels at all, micro f1 set to 0
    try:
        micro_f1 = f1_score(gt_presence, pred_presence, average='micro', zero_division=0)
    except Exception:
        micro_f1 = 0.0

    # 2) Sentiment accuracy: only positions where GT>0
    mask = true_labels > 0  # boolean array shape (N, n_aspects)
    total_positions = int(mask.sum())
    correct_sent = 0
    if total_positions > 0:
        # compare pred_labels to true_labels only where mask True
        correct_sent = int(((pred_labels == true_labels) & mask).sum())
        sent_acc = correct_sent / total_positions
    else:
        sent_acc = 0.0

    overall_score = 0.7 * micro_f1 + 0.3 * sent_acc

    # Also compute per-aspect presence acc and sentiment acc (for diagnostics)
    per_aspect = {}
    for j, a in enumerate(ASPECT_COLS):
        gt_p = (true_labels[:, j] > 0).astype(int)
        pred_p = (pred_labels[:, j] > 0).astype(int)
        pres_f1 = f1_score(gt_p, pred_p, average='micro', zero_division=0)
        # sentiment acc (only where GT>0)
        mask_j = gt_p.astype(bool)
        if mask_j.sum() > 0:
            sent_acc_j = accuracy_score(true_labels[mask_j, j], pred_labels[mask_j, j])
        else:
            sent_acc_j = None
        per_aspect[a] = {"presence_f1": float(pres_f1), "sent_acc": (None if sent_acc_j is None else float(sent_acc_j)), "n_pos": int(mask_j.sum())}

    # Save results to results.txt
    res_path = os.path.join(args.output_dir, "results.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write("Validation evaluation results\n")
        f.write("=================================\n")
        f.write(f"n_examples (aligned): {pred_labels.shape[0]}\n")
        f.write(f"Micro-F1 (presence, flattened): {micro_f1:.6f}\n")
        f.write(f"Sentiment Accuracy (only GT>0 positions): {sent_acc:.6f} (correct {correct_sent}/{total_positions})\n")
        f.write(f"Overall Score = 0.7 * Micro-F1 + 0.3 * SentimentAcc = {overall_score:.6f}\n\n")
        f.write("Per-aspect details:\n")
        for a in ASPECT_COLS:
            info = per_aspect[a]
            f.write(f"{a}: presence_f1={info['presence_f1']:.6f}, sent_acc={info['sent_acc']}, n_pos={info['n_pos']}\n")
        f.write("\nNotes:\n")
        f.write("- Presence is defined as label>0.\n")
        f.write("- Sentiment accuracy only counts positions where GT>0, as requested.\n")
    print(f"Saved results to {res_path}")
    print(f"Micro-F1 (presence)={micro_f1:.6f}, SentimentAcc={sent_acc:.6f}, Overall={overall_score:.6f}")

def load_saved_model(model_dir, device=None):
    """
    Load MultiAspectModel from model_dir. Assumes trainer.save_model(model_dir) was called previously,
    which creates a folder with a pytorch_model.bin. We build model with encoder loaded from model_dir,
    then load pytorch_model.bin (if present) into model state_dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiAspectModel(model_dir, n_aspects=len(ASPECT_COLS), num_classes=NUM_CLASSES_PER_ASPECT)
    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_sd = {}
            for k, v in state_dict.items():
                nk = k
                if k.startswith('model.'):
                    nk = k[len('model.'):]
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)
    else:
        print(f"Warning: {pt_path} not found. Using encoder weights from {model_dir} (if available).")
    model.to(device)
    model.eval()
    return model

def predict_file(model_dir, test_csv, output_csv, text_col_candidates=('review','Review','text'), batch_size=32, max_length=256):
    """
    Run inference using saved model at model_dir and tokenizer saved there.
    Outputs CSV with columns: stt, giai_tri, luu_tru, nha_hang, an_uong, van_chuyen, mua_sam
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = load_saved_model(model_dir, device=device)

    df = safe_read_csv(test_csv)
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == 'object']
        text_col = obj_cols[0] if obj_cols else df.columns[0]
    if 'stt' not in df.columns:
        df.insert(0, 'stt', range(1, len(df)+1))

    texts = df[text_col].fillna("").astype(str).tolist()
    n = len(texts)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc)  # (batch, n_aspects, num_classes)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()  # shape (b, n_aspects)
            all_preds.append(preds)
    if len(all_preds) == 0:
        preds_arr = np.zeros((0, len(ASPECT_COLS)), dtype=int)
    else:
        preds_arr = np.vstack(all_preds)

    out_df = pd.DataFrame(preds_arr, columns=ASPECT_COLS)
    out_df.insert(0, 'stt', df['stt'].astype(int).values[:out_df.shape[0]])
    out_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved predictions to {output_csv}")

def main(args):
    if args.predict:
        if not args.model_dir:
            raise ValueError("Please provide --model_dir for prediction")
        predict_file(args.model_dir, args.test_csv, args.output_csv, batch_size=args.pred_batch_size, max_length=args.max_length)
        return
    train_main(args)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="train-problem.csv")
    p.add_argument("--model_name", type=str, default="vinai/phobert-base")
    p.add_argument("--output_dir", type=str, default="./mt_bert_model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--text_col", default="Review")

    # predict args
    p.add_argument("--predict", action="store_true", help="Run prediction with saved model")
    p.add_argument("--model_dir", type=str, default=None, help="Directory containing saved model + tokenizer (for predict)")
    p.add_argument("--test_csv", type=str, default="gt_reviews.csv", help="CSV to predict")
    p.add_argument("--output_csv", type=str, default="predict.csv", help="Output CSV path")
    p.add_argument("--pred_batch_size", type=int, default=32, help="Batch size for prediction")

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
