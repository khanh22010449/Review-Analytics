# br_transformer.py
"""
Binary Relevance với Transformer (Hugging Face).
- Mỗi label được huấn luyện bằng 1 mô hình sequence-classification (num_labels=6 => classes 0..5).
- Yêu cầu: transformers, datasets, torch, scikit-learn
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ---------------------------
LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
# ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def detect_text_column(df: pd.DataFrame):
    if 'Review' in df.columns:
        return 'Review'
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    if not obj_cols:
        return df.columns[0]
    avg_lens = {c: df[c].dropna().astype(str).map(len).mean() for c in obj_cols}
    return max(avg_lens, key=avg_lens.get)

def prepare_datasets(df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length=256):
    """
    Trả về datasets.Dataset có fields: input_ids, attention_mask, label
    label: int (0..5)
    """
    sub = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    # convert to str and int
    sub['text'] = sub['text'].fillna('').astype(str)
    sub['label'] = sub['label'].fillna(0).astype(int).clip(0,5)
    ds = Dataset.from_pandas(sub)
    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=max_length)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=['text', 'idx'] if 'idx' in ds.column_names else None)
    return ds

def compute_metrics_for_preds(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    # Micro-F1 on presence (>0)
    y_true_bin = (y_true_df.values > 0).astype(int).flatten()
    y_pred_bin = (y_pred_df.values > 0).astype(int).flatten()
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)

    # Sentiment accuracy (only where GT>0)
    mask = (y_true_df.values > 0)
    sentiment_acc = 0.0
    if mask.sum() > 0:
        correct = (y_true_df.values == y_pred_df.values) & mask
        sentiment_acc = correct.sum() / mask.sum()
    overall = 0.7 * micro_f1 + 0.3 * sentiment_acc
    return {'micro_f1': micro_f1, 'sentiment_acc': sentiment_acc, 'overall': overall}

def train_label_model(train_df, val_df, text_col, label_col, model_name, output_dir, args):
    """
    Train 1 transformer model for a single label.
    Returns the trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # prepare datasets
    train_ds = prepare_datasets(train_df, text_col, label_col, tokenizer, max_length=args.max_length)
    val_ds = prepare_datasets(val_df, text_col, label_col, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("TrainingArguments debug:")
    print("output_dir:", os.path.join(output_dir, f'tmp_{label_col}'))
    print("num_train_epochs:", args.epochs)
    print("per_device_train_batch_size:", args.batch_size)
    print("per_device_eval_batch_size:", args.batch_size)
    print("evaluation_strategy:", 'epoch')
    print("save_strategy:", 'epoch')
    print("learning_rate:", args.learning_rate)
    print("weight_decay:", 0.01)
    print("logging_dir:", os.path.join(output_dir, 'logs'))
    print("logging_steps:", 50)
    print("load_best_model_at_end:", True)
    print("metric_for_best_model:", 'eval_loss')
    print("save_total_limit:", 2)
    print("seed:", args.seed)
    print("fp16:", args.fp16 and torch.cuda.is_available())
    try:
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f'tmp_{label_col}'),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=50,
            seed=args.seed,
            fp16=args.fp16 and torch.cuda.is_available()
        )
    except Exception as e:
        print("Error initializing TrainingArguments:", e)
        raise

    # define metrics for Trainer evaluation (simple accuracy)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).astype(float).mean()
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # save final model
    label_dir = os.path.join(output_dir, f'model_{label_col}')
    os.makedirs(label_dir, exist_ok=True)
    trainer.save_model(label_dir)
    tokenizer.save_pretrained(label_dir)
    return label_dir  # path to saved model

def predict_with_model(model_dir, texts, device='cpu', batch_size=32, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
            enc = {k: v.to(device) for k,v in enc.items()}
            out = model(**enc)
            logits = out.logits
            batch_preds = torch.argmax(logits, axis=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())
    return np.array(preds, dtype=int)

def main(args):
    set_seed(args.seed)
    df = pd.read_csv(args.input)
    text_col = detect_text_column(df)
    print("Detected text column:", text_col)

    # check label columns
    for c in LABEL_COLS:
        if c not in df.columns:
            raise ValueError(f"Label column '{c}' không tìm thấy trong file input.")

    # split train/val (stratify not trivial for multi-label; do simple random split)
    df = df.reset_index(drop=True)
    n = len(df)
    val_frac = args.val_ratio
    val_n = int(n * val_frac)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = df.iloc[:val_n].reset_index(drop=True)
    train_df = df.iloc[val_n:].reset_index(drop=True)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Train one model per label
    model_dirs = {}
    for label in LABEL_COLS:
        print(f"\n=== Training for label: {label} ===")
        label_dir = train_label_model(
            train_df, val_df, text_col, label,
            model_name=args.model_name,
            output_dir=args.output_dir,
            args=args
        )
        model_dirs[label] = label_dir
        print(f"Saved model for {label} to {label_dir}")

    # Predict on val set with each model
    texts_val = val_df[text_col].fillna('').astype(str).tolist()
    preds = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for label, mdir in model_dirs.items():
        print(f"Predicting val for {label} ...")
        p = predict_with_model(mdir, texts_val, device=device, batch_size=args.pred_batch_size, max_length=args.max_length)
        preds[label] = p

    y_pred_df = pd.DataFrame(preds)
    y_true_df = val_df[LABEL_COLS].fillna(0).astype(int).reset_index(drop=True)

    metrics = compute_metrics_for_preds(y_true_df, y_pred_df)
    print("\n=== EVALUATION (on val) ===")
    print(f"Micro-F1 (presence >0): {metrics['micro_f1']:.4f}")
    print(f"Sentiment Accuracy (only GT>0): {metrics['sentiment_acc']:.4f}")
    print(f"Overall Score: {metrics['overall']:.4f}")

    print("\nPer-label classification reports (val):")
    for label in LABEL_COLS:
        print(f"--- {label} ---")
        print(classification_report(y_true_df[label], y_pred_df[label], zero_division=0))

    # Save predictions CSV in required format
    out_df = y_pred_df.copy()
    out_df.insert(0, 'stt', range(1, len(out_df)+1))
    out_csv = os.path.join(args.output_dir, 'predictions_val.csv')
    out_df.to_csv(out_csv, index=False)
    print(f"Saved validation predictions to {out_csv}")

    # Also save mapping of models
    meta = {'model_dirs': model_dirs, 'text_col': text_col, 'labels': LABEL_COLS}
    import json
    with open(os.path.join(args.output_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='train-problem.csv')
    p.add_argument('--output_dir', type=str, default='./br_transformer_models')
    p.add_argument('--model_name', type=str, default='vinai/phobert-base', help='Tên model HF (hoặc path)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--pred_batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fp16', action='store_true', help='Enable fp16 if CUDA available')
    args = p.parse_args()
    main(args)
