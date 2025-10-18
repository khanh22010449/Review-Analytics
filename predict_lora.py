# predict_lora_0to5.py
"""
Predict for LoRA-trained multi-task model with ratings 0..5.
Outputs CSV: stt,giai_tri,luu_tru,nha_hang,an_uong,van_chuyen,mua_sam
"""

import argparse, os
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = str(self.texts[idx])
        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        return item

class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states, mask):
        scores = self.linear(hidden_states).squeeze(-1)
        mask_bool = mask.to(dtype=torch.bool, device=scores.device)
        if scores.dtype in (torch.float16, torch.bfloat16):
            neg_inf_val = -1e4
        else:
            neg_inf_val = -1e9
        neg_inf = torch.tensor(neg_inf_val, dtype=scores.dtype, device=scores.device)
        scores = scores.masked_fill(~mask_bool, neg_inf)
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return pooled

def safe_encoder_forward(encoder_module, **kwargs):
    try:
        return encoder_module(**kwargs)
    except TypeError as e:
        msg = str(e).lower()
        if "unexpected keyword argument 'labels'" in msg or 'labels' in kwargs:
            kwargs_filtered = {k:v for k,v in kwargs.items() if k != 'labels'}
            if hasattr(encoder_module, "base_model"):
                try:
                    return encoder_module.base_model(**kwargs_filtered)
                except Exception:
                    pass
            return encoder_module(**kwargs_filtered)
        else:
            raise

class MultiTaskModelForPEFT(nn.Module):
    def __init__(self, encoder, hidden_size, num_aspects=len(ASPECT_COLS), num_rating_classes=6, hidden_dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.att_pool = AttentionPool(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.shared_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(hidden_dropout))
        self.presence_head = nn.Linear(hidden_size, num_aspects)
        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Dropout(hidden_dropout),
                          nn.Linear(hidden_size//2, num_rating_classes))
            for _ in range(num_aspects)
        ])
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = safe_encoder_forward(self.encoder, input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        pooled = self.att_pool(last, attention_mask)
        pooled = self.dropout(pooled)
        shared = self.shared_proj(pooled)
        pres_logits = self.presence_head(shared)
        sent_logits = torch.stack([h(shared) for h in self.sentiment_heads], dim=1)
        return pres_logits, sent_logits

def predict(model, tokenizer, texts, device, batch_size=32, threshold=0.5, mask_rating_when_absent=True, include_presence=False):
    ds = TestDataset(texts, tokenizer, max_length=tokenizer.model_max_length if hasattr(tokenizer,"model_max_length") else 256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    all_pres = []; all_ratings = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pres_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pres_prob = torch.sigmoid(pres_logits).cpu().numpy()
            pres_pred = (pres_prob >= threshold).astype(int)
            sent_pred = torch.argmax(sent_logits, dim=-1).cpu().numpy()  # 0..5
            all_pres.append(pres_pred); all_ratings.append(sent_pred)
    all_pres = np.vstack(all_pres); all_ratings = np.vstack(all_ratings)
    if mask_rating_when_absent:
        masked = all_ratings.copy()
        masked[all_pres == 0] = 0  # set to 0 when absent
    else:
        masked = all_ratings
    df_ratings = pd.DataFrame(masked, columns=ASPECT_COLS)
    df_presence = pd.DataFrame(all_pres, columns=[f"presence_{c}" for c in ASPECT_COLS])
    if include_presence:
        return pd.concat([df_presence, df_ratings], axis=1)
    else:
        return df_ratings

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="xlm-roberta-base")
    parser.add_argument("--peft-dir", type=str, required=True)
    parser.add_argument("--model-state", type=str, default=None)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="../predict_lora.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-rating", action="store_true")
    parser.add_argument("--include-presence", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    base_encoder = AutoModel.from_pretrained(args.model_name)
    hidden_size = base_encoder.config.hidden_size

    print("Loading PEFT adapter from:", args.peft_dir)
    try:
        encoder = PeftModel.from_pretrained(base_encoder, args.peft_dir)
    except Exception as e:
        print("PeftModel.from_pretrained failed:", e)
        encoder = base_encoder

    model = MultiTaskModelForPEFT(encoder=encoder, hidden_size=hidden_size, num_aspects=len(ASPECT_COLS), num_rating_classes=6)
    model.to(device)

    if args.model_state:
        s = torch.load(args.model_state, map_location="cpu")
        try:
            model.load_state_dict(s)
            print("Loaded full model state.")
        except Exception:
            if isinstance(s, dict) and 'model_state_dict' in s:
                model.load_state_dict(s['model_state_dict'])
                print("Loaded model_state_dict.")
            else:
                own = model.state_dict()
                filtered = {k:v for k,v in s.items() if k in own and v.shape == own[k].shape}
                own.update(filtered)
                model.load_state_dict(own)
                print(f"Partially loaded {len(filtered)} keys.")

    df = pd.read_csv(args.test_csv)
    if 'review' not in df.columns:
        raise ValueError("Test CSV must contain column 'Review'")
    texts = df['review'].astype(str).tolist()

    preds = predict(model, tokenizer, texts, device, batch_size=args.batch_size, threshold=args.threshold,
                    mask_rating_when_absent=args.mask_rating, include_presence=args.include_presence)

    if args.include_presence:
        out_df = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    else:
        out_df = preds.reset_index(drop=True).copy()
        out_df.insert(0, 'stt', range(1, len(out_df)+1))
        cols = ['stt'] + ASPECT_COLS
        out_df = out_df[cols]

    out_df.to_csv(args.out, index=False)
    print("Saved predictions to:", args.out)
    print(out_df.head(10).to_string(index=False))

if __name__ == "__main__":
    _main()
