"""
File: build_model.py (đã chỉnh sửa)
- Giữ nguyên toàn bộ logic gốc của bạn (train LogisticRegression, lưu model, vectorizer...)
- Thêm hỗ trợ PyVi tokenizer (thuần Python, không cần Java)
"""

import os
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

# ==================== PyVi integration ====================
USE_PYVI_AVAILABLE = True
try:
    from pyvi.ViTokenizer import tokenize as pyvi_tokenize_single
except Exception:
    USE_PYVI_AVAILABLE = False


def pyvi_tokenize_texts(texts):
    if not USE_PYVI_AVAILABLE:
        print("[PyVi] Not installed. Run: pip install pyvi")
        return texts
    out = []
    for t in texts:
        try:
            seg = pyvi_tokenize_single(t)
            out.append(" ".join(seg.split()))
        except Exception:
            out.append(t)
    return out


def prepare_texts_for_feats(texts, sp=None, use_pyvi=False):
    if sp is not None:
        return [" ".join(sp.encode_as_pieces(t)) for t in texts]
    if use_pyvi:
        return pyvi_tokenize_texts(texts)
    return texts


# ==================== Train Function ====================

def train_br(train_df, val_df, text_col, aspect_cols, output_dir, sp=None, random_state=42, use_pyvi=False):
    os.makedirs(output_dir, exist_ok=True)

    # --- Tokenization ---
    train_texts_for_feats = prepare_texts_for_feats(train_df[text_col].tolist(), sp=sp, use_pyvi=use_pyvi)
    val_texts_for_feats   = prepare_texts_for_feats(val_df[text_col].tolist(), sp=sp, use_pyvi=use_pyvi)

    # --- TF-IDF Vectorizer ---
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_texts_for_feats)
    X_val = vectorizer.transform(val_texts_for_feats)

    joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))

    pres_models = {}
    sent_models = {}

    for aspect in aspect_cols:
        print(f"[train_br] Training aspect: {aspect}")
        y_train_raw = train_df[aspect].fillna(0).astype(int)
        y_val_raw   = val_df[aspect].fillna(0).astype(int)

        # === Presence Model ===
        y_train_pres = (y_train_raw > 0).astype(int)
        y_val_pres   = (y_val_raw > 0).astype(int)

        pres_clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=random_state)
        pres_clf.fit(X_train, y_train_pres)

        joblib.dump(pres_clf, os.path.join(output_dir, f'presence_{aspect}.joblib'))
        pres_models[aspect] = pres_clf

        # === Sentiment Model (multi-class) ===
        idx_pos = y_train_raw > 0
        if idx_pos.sum() > 5:
            y_train_sent = y_train_raw[idx_pos]
            X_train_pos  = X_train[idx_pos]

            sent_clf = LogisticRegression(max_iter=1500, solver='liblinear', multi_class='auto', C=0.1, random_state=random_state)
            sent_clf.fit(X_train_pos, y_train_sent)

            joblib.dump(sent_clf, os.path.join(output_dir, f'sentiment_{aspect}.joblib'))
            sent_models[aspect] = sent_clf

            preds_val_pres = pres_clf.predict(X_val)
            preds_val_sent = sent_clf.predict(X_val[preds_val_pres > 0])
            print(f"Aspect {aspect} trained — Sentiment classes: {sent_clf.classes_}")
        else:
            print(f"[train_br] Skipped sentiment for aspect {aspect} (too few samples)")

    return {
        'vectorizer': vectorizer,
        'pres_models': pres_models,
        'sent_models': sent_models
    }


# ==================== Evaluate ====================

def evaluate_aggregate(val_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, sp=None, use_pyvi=False):
    val_texts_for_feats = prepare_texts_for_feats(val_df[text_col].tolist(), sp=sp, use_pyvi=use_pyvi)
    X_val = vectorizer.transform(val_texts_for_feats)

    results = {}
    for aspect in aspect_cols:
        pres_clf = pres_models.get(aspect)
        sent_clf = sent_models.get(aspect)
        y_true = val_df[aspect].fillna(0).astype(int)
        y_true_pres = (y_true > 0).astype(int)

        if pres_clf is not None:
            y_pred_pres = pres_clf.predict(X_val)
            print(f"[eval] Presence Report for {aspect}:")
            print(classification_report(y_true_pres, y_pred_pres))

        if sent_clf is not None:
            idx_pos = y_pred_pres > 0
            y_true_sent = y_true[idx_pos]
            y_pred_sent = sent_clf.predict(X_val[idx_pos])
            print(f"[eval] Sentiment Report for {aspect}:")
            print(classification_report(y_true_sent, y_pred_sent))

    return results


# ==================== Predict ====================

def predict_and_save(test_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, output_csv, sp=None, use_pyvi=False):
    test_texts_for_feats = prepare_texts_for_feats(test_df[text_col].tolist(), sp=sp, use_pyvi=use_pyvi)
    X_test = vectorizer.transform(test_texts_for_feats)

    preds = {}
    for aspect in aspect_cols:
        pres_clf = pres_models.get(aspect)
        sent_clf = sent_models.get(aspect)
        y_pred_pres = pres_clf.predict(X_test) if pres_clf is not None else [0]*len(test_df)

        y_pred_sent = [0]*len(test_df)
        if sent_clf is not None:
            idx_pos = [i for i, p in enumerate(y_pred_pres) if p > 0]
            if idx_pos:
                X_pos = X_test[idx_pos]
                y_pred_sent_pos = sent_clf.predict(X_pos)
                for j, idx in enumerate(idx_pos):
                    y_pred_sent[idx] = y_pred_sent_pos[j]
        preds[f'{aspect}_presence'] = y_pred_pres
        preds[f'{aspect}_sentiment'] = y_pred_sent

    df_out = pd.DataFrame(preds)
    df_out.to_csv(output_csv, index=False)
    print(f"[predict] Saved predictions to {output_csv}")
    return df_out


# ==================== Main CLI ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','eval','predict'], required=True)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--output_csv', type=str, default='predict.csv')
    parser.add_argument('--use_pyvi', action='store_true', help='Use PyVi Vietnamese tokenizer (pure Python)')
    args = parser.parse_args()

    text_col = 'Review'
    aspect_cols = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']

    if args.mode == 'train':
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        res = train_br(train_df, val_df, text_col, aspect_cols, args.output_dir, use_pyvi=args.use_pyvi)
    elif args.mode == 'eval':
        val_df = pd.read_csv(args.val_csv)
        vectorizer = joblib.load(os.path.join(args.output_dir, 'tfidf_vectorizer.joblib'))
        pres_models = {a: joblib.load(os.path.join(args.output_dir, f'presence_{a}.joblib')) for a in aspect_cols if os.path.exists(os.path.join(args.output_dir, f'presence_{a}.joblib'))}
        sent_models = {a: joblib.load(os.path.join(args.output_dir, f'sentiment_{a}.joblib')) for a in aspect_cols if os.path.exists(os.path.join(args.output_dir, f'sentiment_{a}.joblib'))}
        evaluate_aggregate(val_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, use_pyvi=args.use_pyvi)
    elif args.mode == 'predict':
        test_df = pd.read_csv(args.test_csv)
        vectorizer = joblib.load(os.path.join(args.output_dir, 'tfidf_vectorizer.joblib'))
        pres_models = {a: joblib.load(os.path.join(args.output_dir, f'presence_{a}.joblib')) for a in aspect_cols if os.path.exists(os.path.join(args.output_dir, f'presence_{a}.joblib'))}
        sent_models = {a: joblib.load(os.path.join(args.output_dir, f'sentiment_{a}.joblib')) for a in aspect_cols if os.path.exists(os.path.join(args.output_dir, f'sentiment_{a}.joblib'))}
        predict_and_save(test_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, args.output_csv, use_pyvi=args.use_pyvi)


if __name__ == '__main__':
    main()
