# predict.py
"""
Predict using best_multitask_weighted.pt and save CSV.

Usage:
  python predict.py --input data/problem_test.csv --output data/predict.csv --model best_multitask_weighted.pt
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Try import model and config from new_model
try:
    from new_model import MultiTaskTransformer, LABEL_COLS, NUM_SEGMENT_CLASSES, MODEL_NAME, MAX_LEN, BATCH_SIZE
except Exception as e:
    # Fallback defaults if some names missing in new_model
    print("Warning importing from new_model:", e)
    # MultiTaskTransformer must exist, otherwise cannot proceed
    try:
        from new_model import MultiTaskTransformer
    except Exception as e2:
        raise ImportError("Cannot import MultiTaskTransformer from new_model.py. Fix new_model.py or place model class there.") from e2
    # reasonable defaults (you can override via CLI)
    LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
    NUM_SEGMENT_CLASSES = 5  # segments 0..4 -> we'll add +1 when present
    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 256
    BATCH_SIZE = 32

class PredictDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="Review", max_len=256):
        # df: pandas DataFrame. May or may not contain label columns.
        self.df = df.reset_index(drop=True)
        # detect text column
        if text_col in df.columns:
            self.texts = df[text_col].fillna("").astype(str).tolist()
        else:
            # try common alternatives
            if "review" in df.columns:
                self.texts = df["review"].fillna("").astype(str).tolist()
            elif "text" in df.columns:
                self.texts = df["text"].fillna("").astype(str).tolist()
            else:
                # if no text column, use empty strings
                self.texts = [""] * len(df)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # stt handling
        if "stt" in df.columns:
            self.stt = df["stt"].astype(int).tolist()
        else:
            self.stt = list(range(1, len(df) + 1))
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        txt = self.texts[idx]
        tok = self.tokenizer(txt, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "stt": self.stt[idx]
        }
        return item

def collate_predict(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    stt = [b["stt"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "stt": stt}

def load_model_weights(model_path, device):
    # instantiate model (expects MultiTaskTransformer in new_model)
    model = MultiTaskTransformer(MODEL_NAME, num_aspects=len(LABEL_COLS), num_seg_classes=NUM_SEGMENT_CLASSES)
    state = torch.load(model_path, map_location=device)
    # If saved dict contains 'model' or is wrapped, try common fixes
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state
    # If it's a checkpoint with keys like 'model_state', try to find it
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/predict.csv")
    parser.add_argument("--model", type=str, default="best_multitask_weighted.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # ensure stt exists or will be created in dataset
    if "stt" not in df.columns:
        df.insert(0, "stt", range(1, len(df) + 1))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = PredictDataset(df, tokenizer, text_col="Review", max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_predict)

    model = load_model_weights(args.model, device=args.device)

    all_rows = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            # model expected signature: (input_ids=..., attention_mask=...) -> seg_logits, pres_logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # support either tuple or dict return
            if isinstance(outputs, dict):
                seg_logits = outputs.get("seg_logits") or outputs.get("seg")
                pres_logits = outputs.get("pres_logits") or outputs.get("pres")
            else:
                # assume tuple (seg_logits, pres_logits)
                seg_logits, pres_logits = outputs
            seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()  # shape (B, num_aspects)
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()      # shape (B, num_aspects)
            pres_preds = (pres_probs >= 0.5).astype(int)
            B = seg_preds.shape[0]
            for i in range(B):
                stt_val = batch["stt"][i]
                out_row = {"stt": int(stt_val)}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i, j])
                    seg = int(seg_preds[i, j]) + 1 if present else 0
                    # ensure seg in 0..(NUM_SEGMENT_CLASSES) maybe clamp
                    if seg < 0:
                        seg = 0
                    # clamp upper bound to NUM_SEGMENT_CLASSES (if argmax returned 0..K-1)
                    seg = int(min(seg, NUM_SEGMENT_CLASSES))
                    out_row[col] = seg
                all_rows.append(out_row)

    df_out = pd.DataFrame(all_rows, columns=["stt"] + list(LABEL_COLS))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print("Saved predictions to:", args.output)

if __name__ == "__main__":
    main()
