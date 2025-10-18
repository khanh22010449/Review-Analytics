import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(t,
                             truncation=True,
                             padding='max_length',
                             max_length=self.max_length,
                             return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_aspects, num_rating_classes=5, hidden_dropout_prob=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.aspect_head = nn.Linear(hidden_size, num_aspects)
        self.sentiment_head = nn.Linear(hidden_size, num_aspects * num_rating_classes)
        self.num_aspects = num_aspects
        self.num_rating_classes = num_rating_classes

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        presence_logits = self.aspect_head(pooled)                     # (B, K)
        sentiment_logits = self.sentiment_head(pooled)                 # (B, K * C)
        sentiment_logits = sentiment_logits.view(-1, self.num_aspects, self.num_rating_classes)  # (B, K, C)
        return presence_logits, sentiment_logits

def predict_dataframe(model, tokenizer, texts, device, batch_size=32, threshold=0.5, mask_rating_when_absent=True):
    ds = TestDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)

    all_pres = []
    all_ratings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pres_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_prob = torch.sigmoid(pres_logits).cpu().numpy()   # (B, K)
            pres_pred = (pres_prob >= threshold).astype(int)       # 0/1
            sent_pred = torch.argmax(sent_logits, dim=-1).cpu().numpy()  # (B, K) values 0..C-1
            sent_pred_plus1 = sent_pred + 1  # convert to 1..5

            all_pres.append(pres_pred)
            all_ratings.append(sent_pred_plus1)

    all_pres = np.vstack(all_pres)
    all_ratings = np.vstack(all_ratings)

    # If mask_rating_when_absent -> set rating to 1 where presence==0
    if mask_rating_when_absent:
        masked_ratings = all_ratings.copy()
        masked_ratings[all_pres == 0] = 1
    else:
        masked_ratings = all_ratings

    # Build DataFrame in required format
    df_out = pd.DataFrame(masked_ratings, columns=ASPECT_COLS)
    # ensure integer dtype
    df_out = df_out.astype(int)
    return df_out

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_aspects = len(ASPECT_COLS)
    num_rating_classes = 5

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskModel(args.model_name, num_aspects=num_aspects, num_rating_classes=num_rating_classes)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    state = torch.load(args.model_path, map_location='cpu')
    # support both plain state_dict or dict with model_state_dict
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            # try to be helpful: if user saved entire model object, try load keys that match
            if isinstance(state, dict):
                # try to find key that looks like state_dict
                for k,v in state.items():
                    if isinstance(v, dict) and set(v.keys()) & {"aspect_head.weight", "sentiment_head.weight"}:
                        model.load_state_dict(v)
                        break
                else:
                    raise RuntimeError("Không thể load weights: định dạng file lạ. Hãy kiểm tra cách bạn lưu model.")
            else:
                raise RuntimeError("Không thể load weights: định dạng file lạ.")

    # read test csv
    df_test = pd.read_csv(args.test_csv)
    if 'reviews' not in df_test.columns:
        raise ValueError("Test CSV phải có cột 'Review' chứa văn bản.")
    texts = df_test['reviews'].astype(str).tolist()

    preds_df = predict_dataframe(model, tokenizer, texts, device,
                                 batch_size=args.batch_size,
                                 threshold=args.threshold,
                                 mask_rating_when_absent=not args.no_mask)

    # build final output with stt starting at 1 and aspect order exactly as requested
    out_df = preds_df.reset_index(drop=True).copy()
    out_df.insert(0, 'stt', range(1, len(out_df)+1))
    # reorder columns to: stt, giai_tri, luu_tru, nha_hang, an_uong, van_chuyen, mua_sam
    cols = ['stt'] + ASPECT_COLS
    out_df = out_df[cols]

    # save
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="../lora_multitask_best.pth")
    parser.add_argument("--test-csv", type=str, default="./gt_reviews.csv")
    parser.add_argument("--out", type=str, default="./predict.csv")
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="ngưỡng sigmoid để coi 1 khía cạnh là present")
    parser.add_argument("--no-mask", action="store_true",
                        help="nếu set, không gán rating=1 khi presence==0 (tức giữ rating dự đoán bất kể presence)")
    args = parser.parse_args()
    main(args)
