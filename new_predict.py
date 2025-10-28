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
from transformers import AutoTokenizer
from new_model import MultiTaskTransformer, ReviewDataset, collate_fn, LABEL_COLS, NUM_SEGMENT_CLASSES, MODEL_NAME, MAX_LEN, BATCH_SIZE
from torch.utils.data import DataLoader

def load_model_weights(model_path, device):
    model = MultiTaskTransformer(MODEL_NAME, num_aspects=len(LABEL_COLS), num_seg_classes=NUM_SEGMENT_CLASSES)
    state = torch.load(model_path, map_location=device)
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
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ReviewDataset(df, tokenizer, max_len=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = load_model_weights(args.model, device=args.device)

    rows = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            seg_logits, pres_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            seg_preds = torch.argmax(seg_logits, dim=-1).cpu().numpy()
            pres_probs = torch.sigmoid(pres_logits).cpu().numpy()
            pres_preds = (pres_probs >= 0.5).astype(int)
            B = seg_preds.shape[0]
            for i in range(B):
                out_row = {}
                for j, col in enumerate(LABEL_COLS):
                    present = int(pres_preds[i,j])
                    seg = int(seg_preds[i,j]) + 1 if present else 0
                    out_row[col] = seg
                rows.append(out_row)
    df_out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print("Saved predictions to:", args.output)

if __name__ == "__main__":
    main()
