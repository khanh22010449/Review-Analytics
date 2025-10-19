"""
Fine-tuning pipeline cho bài toán Multi-label + Sentiment (6 nhãn, điểm 0-5)
File này cung cấp các hàm/ lớp yêu cầu:
 - Load_data()
 - Clean_and_normalize_data()
 - class Multi_label_Model
 - train()
 - test()

Hướng dẫn nhanh:
 - Đặt file dữ liệu tại: /mnt/data/train-problem.csv
 - Cấu trúc CSV kỳ vọng có cột: "text" (hoặc "review") và 6 cột nhãn: giai_tri, luu_tru, nha_hang, an_uong, van_chuyen, mua_sam
 - Chạy: python fine_tune_multilabel_sentiment.py

Yêu cầu: transformers, torch, pandas, scikit-learn, tqdm
"""

import os
import random
import argparse
import unicodedata
import regex as re
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

from torch.optim import AdamW

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
    """Đọc CSV và chuẩn hoá tên cột.
    Trả về DataFrame chứa cột 'text' và các cột label trong LABEL_COLUMNS.
    """
    if text_col_candidates is None:
        text_col_candidates = ["text", "review", "review_text", "content"]

    df = pd.read_csv(path)
    # tìm cột text
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # nếu không có, lấy cột đầu tiên
        text_col = df.columns[0]

    df = df.rename(columns={text_col: "Review"})

    # đảm bảo có đủ nhãn
    for lbl in LABEL_COLUMNS:
        if lbl not in df.columns:
            # tạo cột mặc định 0 nếu không tồn tại
            df[lbl] = 0

    # giữ chỉ các cột cần thiết
    keep = ["Review"] + LABEL_COLUMNS
    return df[keep]


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
    # unicode normalize
    s = unicodedata.normalize('NFC', s)
    # chuyển về thường
    s = s.lower()
    # thay thế nhiều khoảng trắng
    s = re.sub(r"\s+", " ", s)
    # loại ký tự đặc biệt trừ dấu câu đơn giản
    s = re.sub(r"[^\w\s\p{P}\p{L}]", " ", s)
    s = s.strip()
    return s


def Clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý văn bản cơ bản cho tiếng Việt/đa ngôn ngữ.
    - Loại control char, chuẩn hoá unicode, lower-case, collapse spaces.
    - Chuyển labels thành int và đảm bảo trong [0,5]
    """
    df = df.copy()
    df['Review'] = df['Review'].astype(str).apply(clean_text)

    for lbl in LABEL_COLUMNS:
        df[lbl] = pd.to_numeric(df[lbl], errors='coerce').fillna(0).astype(int)
        df[lbl] = df[lbl].clip(0, 5)

    return df


#########################
# Dataset
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
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


#########################
# 3. Model
#########################

class Multi_label_Model(nn.Module):
    """Shared encoder + per-label classification head (6-way classification each: classes 0..5)
    """
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_labels: int = len(LABEL_COLUMNS), hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # one linear head per label
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 6) for _ in range(num_labels)])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs[0] is last_hidden_state; use pooled output if available
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # fallback: mean pooling (attention aware)
            last = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).type_as(last)
            pooled = (last * attn).sum(1) / attn.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        logits = [head(pooled) for head in self.heads]  # list of [batch, 6]
        logits = torch.stack(logits, dim=1)  # [batch, num_labels, 6]

        loss = None
        if labels is not None:
            # labels: [batch, num_labels], each an int 0..5
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for i in range(logits.size(1)):
                losses.append(loss_fct(logits[:, i, :], labels[:, i].long()))
            loss = torch.stack(losses).mean()

        return {'loss': loss, 'logits': logits}


#########################
# 4. train
#########################

def train(
    df: pd.DataFrame,
    model_name: str = 'bert-base-multilingual-cased',
    output_dir: str = './model_out',
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256,
    val_size: float = 0.1,
    device: str = None
):
    set_seed()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=RANDOM_SEED)

    y_train = train_df[LABEL_COLUMNS].values.astype(int)
    y_val = val_df[LABEL_COLUMNS].values.astype(int)

    train_dataset = MultiLabelDataset(train_df['Review'].tolist(), y_train, tokenizer, max_length=max_length)
    val_dataset = MultiLabelDataset(val_df['Review'].tolist(), y_val, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Multi_label_Model(model_name=model_name).to(device)

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        train_losses = []
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(train_losses)})

        # Validation
        val_metrics = test(model, val_loader, device=device, return_preds=False)
        val_loss = val_metrics.get('loss', None)
        print(f"Epoch {epoch} validation: ", val_metrics)

        # save
        model_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))

    print('Training complete. Models saved to', output_dir)
    return model, tokenizer


#########################
# 5. test
#########################

def test(model_or_path, dataloader=None, device: str = None, return_preds: bool = True) -> Dict:
    """
    Nếu model_or_path là đường dẫn -> tải model. Nếu là object model -> dùng trực tiếp.
    Nếu dataloader không được cung cấp, hàm sẽ raise error.

    Trả về dict chứa: loss, accuracy_per_label, f1_macro_per_label, mae_per_label, preds, labels (nếu return_preds=True)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # nếu model_or_path là đường dẫn tới state_dict
    if isinstance(model_or_path, str):
        # cần tokenizer bên ngoài để tạo dataloader; user phải truyền dataloader
        state_dict_path = model_or_path
        # attempt to infer model architecture from state dict keys
        # đơn giản: khởi tạo default model và load
        model = Multi_label_Model().to(device)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    else:
        model = model_or_path

    if dataloader is None:
        raise ValueError('dataloader is required')

    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    loss_fct = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']  # [batch, num_labels, 6]
            loss = outputs['loss']
            losses.append(loss.item() if loss is not None else 0.0)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # predictions
    preds = np.argmax(all_logits, axis=-1)  # [n, num_labels]

    metrics = {
        'loss': float(np.mean(losses)),
        'accuracy_per_label': {},
        'f1_macro_per_label': {},
        'mae_per_label': {}
    }

    for i, lbl in enumerate(LABEL_COLUMNS):
        metrics['accuracy_per_label'][lbl] = float(accuracy_score(all_labels[:, i], preds[:, i]))
        # f1 macro for multiclass
        metrics['f1_macro_per_label'][lbl] = float(f1_score(all_labels[:, i], preds[:, i], average='macro', zero_division=0))
        metrics['mae_per_label'][lbl] = float(mean_absolute_error(all_labels[:, i], preds[:, i]))

    # global metrics
    metrics['accuracy_macro'] = float(np.mean(list(metrics['accuracy_per_label'].values())))
    metrics['f1_macro'] = float(np.mean(list(metrics['f1_macro_per_label'].values())))
    metrics['mae'] = float(np.mean(list(metrics['mae_per_label'].values())))

    if return_preds:
        return {**metrics, 'preds': preds, 'labels': all_labels}
    else:
        return metrics


#########################
# Utility: prepare dataloader from DataFrame
#########################

def prepare_dataloader_from_df(df: pd.DataFrame, tokenizer_name: str, batch_size: int = 16, max_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    labels = df[LABEL_COLUMNS].values.astype(int)
    ds = MultiLabelDataset(df['Review'].tolist(), labels, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


#########################
# main (example)
#########################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./train-problem.csv')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--out', type=str, default='./model_out')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    df = Load_data(args.data)
    df = Clean_and_normalize_data(df)

    model, tokenizer = train(df, model_name=args.model_name, output_dir=args.out, epochs=args.epochs, batch_size=args.bs, lr=args.lr)

    # Evaluate on validation split (recreate val loader)
    # load best model
    best_path = os.path.join(args.out, 'best_model.pt')
    val_df = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)[1]
    val_loader = prepare_dataloader_from_df(val_df, tokenizer_name=args.model_name, batch_size=args.bs)
    results = test(best_path, dataloader=val_loader)
    print('Final evaluation on validation:', results)
