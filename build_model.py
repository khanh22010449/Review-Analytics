# -*- coding: utf-8 -*-
"""
train_multilabel_bert_with_overall.py
Phiên bản mở rộng: tính 'overall = 0.7*micro_f1 + 0.3*seg'
seg có thể là F1 trung bình của một nhóm nhãn hoặc Accuracy của nhóm nhãn.
Chạy:
  python train_multilabel_bert_with_overall.py --create_example --use_cuda
"""

import os
import random
import argparse
import re
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, hamming_loss, classification_report, accuracy_score

# -------------------------
# Cấu hình mặc định
# -------------------------
MODEL_NAME = "bert-base-uncased"
LABEL_NAMES = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]

# -------------------------
# Tạo dữ liệu ví dụ (CSV)
# -------------------------
def create_example_csv(path="data_example.csv"):
    rows = [
        ("Tối nay đi xem phim và ăn tối với bạn bè", [1,0,0,1,0,0]),
        ("Máy ảnh mới 4k, lưu trữ lên đám mây rất tốt", [0,1,0,0,0,0]),
        ("Nhà hàng này phục vụ ngon, giá hợp lý", [0,0,1,1,0,0]),
        ("Mua sắm online, giao hàng nhanh", [0,0,0,0,1,1]),
        ("Chuyến du lịch kết hợp ẩm thực và giải trí", [1,0,0,1,0,1]),
        ("Dịch vụ vận chuyển an toàn, chất lượng", [0,0,0,0,1,0]),
        ("Lưu trữ dữ liệu quan trọng cho công ty", [0,1,0,0,0,0]),
        ("Tối nay chỉ ở nhà xem phim", [1,0,0,0,0,0]),
        ("Mua đồ ăn online và nhận hàng ngay", [0,0,0,1,1,1]),
        ("Review về nhà hàng và dịch vụ giao đồ ăn", [0,0,1,1,1,0]),
    ]
    df = pd.DataFrame(columns=["Review"] + LABEL_NAMES)
    for t, labels in rows:
        row = {"Review": t}
        for i, name in enumerate(LABEL_NAMES):
            row[name] = labels[i]
        df = df.append(row, ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Created example CSV at {path} with {len(df)} rows.")
    return path

# -------------------------
# Dataset class
# -------------------------
class MultiLabelTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_names: List[str], max_length=128):
        self.texts = df["Review"].tolist()
        self.labels = df[label_names].values.astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float)  # float for BCEWithLogits
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = labels
        return item

# -------------------------
# Train / Eval functions
# -------------------------
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    loop = tqdm(dataloader, desc="train", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape (batch, n_labels)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)
    all_logits = np.vstack(all_logits) if len(all_logits) > 0 else np.zeros((0, model.config.num_labels))
    all_labels = np.vstack(all_labels) if len(all_labels) > 0 else np.zeros((0, model.config.num_labels))
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= threshold).astype(int)
    metrics = {}
    # micro / macro F1
    metrics["micro_f1"] = f1_score(all_labels, preds, average="micro", zero_division=0) if all_labels.size else 0.0
    metrics["macro_f1"] = f1_score(all_labels, preds, average="macro", zero_division=0) if all_labels.size else 0.0
    metrics["hamming_loss"] = hamming_loss(all_labels, preds) if all_labels.size else 0.0
    # classification report per label (as text)
    if all_labels.size:
        metrics["report"] = classification_report(all_labels, preds, target_names=LABEL_NAMES, zero_division=0)
    else:
        metrics["report"] = ""
    metrics["y_true"] = all_labels
    metrics["y_pred"] = preds
    metrics["probs"] = probs
    return metrics

# -------------------------
# Helper: lấy F1 từ classification_report
# -------------------------
def extract_f1_from_report(report_text, label_names):
    """
    Trích F1-score cho từng nhãn từ classification_report (text).
    Trả về dict: {label: f1}
    """
    f1s = {}
    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s{2,}", line)
        # parts thường là [label, precision, recall, f1-score, support]
        if len(parts) >= 4 and parts[0] in label_names:
            try:
                f1 = float(parts[3])
            except:
                f1 = 0.0
            f1s[parts[0]] = f1
    return f1s

# -------------------------
# In kết quả đẹp theo định dạng yêu cầu
# -------------------------
def print_and_log_eval(log_fpath, metrics, seg_labels, seg_mode):
    # classification summary
    # Tính tổng đúng labels: ở multi-label, "Đúng 6/6 labels" ý bạn có thể hiểu là
    # với 1 sample bạn dự đoán đúng về toàn bộ 6 nhãn hay ở tổng dataset. Ở đây ta in Micro-F1.
    micro = metrics["micro_f1"]
    # seg: tính theo mode
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    seg_f1 = None
    seg_acc = None

    if y_true.size:
        # Nếu seg_mode == 'f1' -> tính trung bình F1 trong seg_labels
        if seg_mode == "f1":
            # dùng sklearn f1_score trên từng nhãn rồi trung bình (macro) cho nhóm seg
            idxs = [LABEL_NAMES.index(l) for l in seg_labels if l in LABEL_NAMES]
            if len(idxs) == 0:
                seg_f1 = 0.0
            else:
                # lấy cột tương ứng
                seg_y_true = y_true[:, idxs]
                seg_y_pred = y_pred[:, idxs]
                # tính F1 cho từng nhãn, sau đó trung bình
                per_label_f1 = []
                for j in range(seg_y_true.shape[1]):
                    per_label_f1.append(f1_score(seg_y_true[:, j], seg_y_pred[:, j], zero_division=0))
                seg_f1 = float(np.mean(per_label_f1))
        else:
            # seg_mode == 'accuracy': tính accuracy trên từng nhãn sau đó trung bình,
            # hoặc tính "exact match" trên chỉ những nhãn seg? Ở đây ta tính per-label accuracy trung bình.
            idxs = [LABEL_NAMES.index(l) for l in seg_labels if l in LABEL_NAMES]
            if len(idxs) == 0:
                seg_acc = 0.0
            else:
                seg_y_true = y_true[:, idxs]
                seg_y_pred = y_pred[:, idxs]
                per_label_acc = []
                for j in range(seg_y_true.shape[1]):
                    per_label_acc.append(accuracy_score(seg_y_true[:, j], seg_y_pred[:, j]))
                seg_acc = float(np.mean(per_label_acc))

    # Tính overall theo công thức: 0.7 * micro_f1 + 0.3 * seg
    seg_for_overall = seg_f1 if seg_mode == "f1" else seg_acc
    seg_for_overall = seg_for_overall if seg_for_overall is not None else 0.0
    overall = 0.7 * micro + 0.3 * seg_for_overall

    # Format giống ví dụ:
    # • Classification: Đúng 6/6 labels → Micro-F1 = 1.0 (100%)
    # • Sentiment: Đúng 2/2 điểm (luu_tru và nha_hang) → Accuracy = 1.0 (100%)
    # • Overall: 0.7 × 1.0 + 0.3 × 1.0 = 1.0 (100%)

    # Dòng Classification
    classification_line = f"• Classification: Micro-F1 = {micro:.4f} ({micro*100:.1f}%)"
    # Dòng Sentiment / SEG
    if seg_mode == "f1":
        seg_line = f"• SEG (F1 trung bình trên {len(seg_labels)} labels): F1 = {seg_for_overall:.4f} ({seg_for_overall*100:.1f}%)"
    else:
        seg_line = f"• SEG (Accuracy trung bình trên {len(seg_labels)} labels): Accuracy = {seg_for_overall:.4f} ({seg_for_overall*100:.1f}%)"
    overall_line = f"• Overall: 0.7 × {micro:.4f} + 0.3 × {seg_for_overall:.4f} = {overall:.4f} ({overall*100:.1f}%)"

    summary = "\n".join([classification_line, seg_line, overall_line])
    print(summary)
    with open(log_fpath, "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")
    return overall

# -------------------------
# Main
# -------------------------
def main(args):
    # tạo dữ liệu ví dụ nếu file không tồn tại
    if args.create_example and not os.path.exists(args.data_csv):
        create_example_csv(args.data_csv)

    df = pd.read_csv(args.data_csv)
    # tách train/val đơn giản
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(df) * args.val_ratio)
    df_train = df[n_val:].reset_index(drop=True)
    df_val = df[:n_val].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_NAMES),
        problem_type="multi_label_classification"
    )
    model.to(device)

    train_ds = MultiLabelTextDataset(df_train, tokenizer, LABEL_NAMES, max_length=args.max_length)
    val_ds = MultiLabelTextDataset(df_val, tokenizer, LABEL_NAMES, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    best_overall = -1.0
    log_fpath = args.log_file
    # khởi tạo log file
    with open(log_fpath, "w", encoding="utf-8") as f:
        f.write("Eval log\n\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"[Epoch {epoch}] train_loss = {train_loss:.4f}")

        metrics = eval_model(model, val_loader, device, threshold=args.threshold)
        # ghi classification_report đầy đủ vào log
        with open(log_fpath, "a", encoding="utf-8") as f:
            f.write(f"=== Epoch {epoch} ===\n")
            f.write(metrics["report"] + "\n")

        # tính overall và in/log
        overall = print_and_log_eval(log_fpath, metrics, args.seg_labels, args.seg_mode)

        # in report tóm tắt
        print("=== Classification Report (per-label) ===")
        print(metrics["report"])

        # save best theo overall
        if overall > best_overall:
            best_overall = overall
            save_dir = args.save_dir
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[INFO] Saved best model to {save_dir} (overall={best_overall:.4f})")
            with open(log_fpath, "a", encoding="utf-8") as f:
                f.write(f"[INFO] Saved best model at epoch {epoch} with overall={best_overall:.4f}\n\n")

    print("[DONE] Training finished. Best overall: {:.4f}".format(best_overall))

# -------------------------
# CLI arguments
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data_example.csv", help="CSV file with 'text' and label columns")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Pretrained transformer model")
    parser.add_argument("--create_example", action="store_true", help="If set, create example CSV")
    parser.add_argument("--save_dir", type=str, default="saved_model_overall", help="Where to save best model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--log_file", type=str, default="eval_log.txt", help="Evaluation log file")
    parser.add_argument("--seg_labels", nargs="+", default=["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"], help="Labels considered as SEG")
    parser.add_argument("--seg_mode", choices=["f1", "accuracy"], default="f1", help="SEG metric mode: 'f1' or 'accuracy'")
    args = parser.parse_args()

    main(args)
