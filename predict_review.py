"""
Dự đoán và lưu kết quả từ mô hình Binary Relevance (BR)
- Đọc model đã huấn luyện (thư mục model_dir)
- Đọc file test.csv chứa cột Review
- Xuất file CSV dự đoán theo format:
  stt,giai_tri,luu_tru,nha_hang,an_uong,van_chuyen,mua_sam

Cách dùng:
python predict_br.py --model_dir ./br_models --test_csv /path/to/test.csv --output_csv ./pred_result.csv
"""

import argparse
import os
import pandas as pd
import joblib

LABEL_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']


def detect_text_column(df):
    if 'Review' in df.columns:
        return 'Review'
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    if not obj_cols:
        return df.columns[0]
    avg_lens = {c: df[c].dropna().astype(str).map(len).mean() for c in obj_cols}
    return max(avg_lens, key=avg_lens.get)


def load_models(model_dir, label_cols=LABEL_COLS):
    models = {}
    for label in label_cols:
        path = os.path.join(model_dir, f"br_{label}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model {path}")
        models[label] = joblib.load(path)
    return models


def predict_and_save(model_dir, test_csv, output_csv):
    df = pd.read_csv(test_csv)
    text_col = detect_text_column(df)
    print(f"Detected text column: {text_col}")

    models = load_models(model_dir)

    preds = {}
    for label, model in models.items():
        print(f"Predicting for label: {label}")
        preds[label] = model.predict(df[text_col].astype(str).fillna(''))

    pred_df = pd.DataFrame(preds)
    pred_df.insert(0, 'stt', range(1, len(pred_df) + 1))

    pred_df.to_csv(output_csv, index=False)
    print(f"Đã lưu kết quả dự đoán tại: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./br_models', help='Thư mục chứa các model BR')
    parser.add_argument('--test_csv', type=str, required=True, help='Đường dẫn tới file test.csv')
    parser.add_argument('--output_csv', type=str, default='./pred_result.csv', help='File CSV đầu ra')
    args = parser.parse_args()

    predict_and_save(args.model_dir, args.test_csv, args.output_csv)
