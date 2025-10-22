"""
Train_and_Test_fixed.py
Bản sửa hoàn chỉnh của Train_and_Test.py với hỗ trợ Word2Vec và alignment features.
- Đảm bảo mọi bước (train / eval / predict) ghép W2V giống nhau.
- Hàm align_features sẽ trim hoặc pad features để phù hợp với model đã lưu.

Cách dùng giống file trước.
"""
import argparse
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import joblib
from tqdm import tqdm

# Optional W2V
USE_W2V = True
try:
    if USE_W2V:
        from gensim.models import Word2Vec
except Exception:
    USE_W2V = False


# ===== helpers =====

def safe_read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def preprocess_df(df, text_col_candidate=None):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if text_col_candidate and text_col_candidate in df.columns:
        text_col = text_col_candidate
    else:
        text_cols = [c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()]
        text_col = text_cols[0] if len(text_cols) > 0 else df.columns[0]
    df[text_col] = df[text_col].fillna("").astype(str)
    if 'stt' not in df.columns:
        df.insert(0, 'stt', range(1, len(df) + 1))
    return df, text_col


# ===== feature =====

def fit_vectorizer(texts, output_dir, use_tfidf=True, max_features=30000):
    if use_tfidf:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,3), min_df=2)
    else:
        vec = CountVectorizer(max_features=max_features, ngram_range=(1,3), min_df=2)
    vec.fit(texts)
    joblib.dump(vec, os.path.join(output_dir, 'vectorizer.joblib'))
    return vec


def transform_texts(vectorizer, texts):
    return vectorizer.transform(texts)


def fit_w2v(texts, output_dir, vector_size=800, window=10, min_count=2):
    tokenized = [t.split() for t in texts]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, min_count=min_count, epochs=500, sg = 1)
    model.save(os.path.join(output_dir, 'w2v.model'))
    return model


def texts_to_w2v_avg(model, texts, vector_size=None):
    if vector_size is None:
        vector_size = getattr(model, 'vector_size', None) or getattr(model.wv, 'vector_size', None)
    arr = np.zeros((len(texts), vector_size), dtype=float)
    for i, t in enumerate(texts):
        toks = t.split()
        vecs = [model.wv[w] for w in toks if w in model.wv]
        if len(vecs) > 0:
            arr[i] = np.mean(vecs, axis=0)
    return arr


# ===== alignment helper =====

def align_features(X, expected_dim):
    from scipy import sparse
    if expected_dim is None:
        return X
    if sparse.issparse(X):
        X = X.tocsr()
        cur = X.shape[1]
        if cur == expected_dim:
            return X
        if cur > expected_dim:
            return X[:, :expected_dim]
        pad = sparse.csr_matrix((X.shape[0], expected_dim - cur))
        return sparse.hstack([X, pad]).tocsr()
    else:
        cur = X.shape[1]
        if cur == expected_dim:
            return X
        if cur > expected_dim:
            return X[:, :expected_dim]
        pad = np.zeros((X.shape[0], expected_dim - cur), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)


# ===== Train BR =====

def train_br(train_df, val_df, text_col, aspect_cols, output_dir, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    vectorizer = fit_vectorizer(train_df[text_col].tolist(), output_dir, use_tfidf=False)
    w2v_model = None
    if USE_W2V:
        w2v_model = fit_w2v(train_df[text_col].tolist(), output_dir)

    X_train = transform_texts(vectorizer, train_df[text_col].tolist())
    X_val = transform_texts(vectorizer, val_df[text_col].tolist())

    if USE_W2V and w2v_model is not None:
        Xw_train = texts_to_w2v_avg(w2v_model, train_df[text_col].tolist(), vector_size=getattr(w2v_model, 'vector_size', None))
        Xw_val = texts_to_w2v_avg(w2v_model, val_df[text_col].tolist(), vector_size=getattr(w2v_model, 'vector_size', None))
        from scipy.sparse import hstack, csr_matrix
        X_train = hstack([X_train, csr_matrix(Xw_train)])
        X_val = hstack([X_val, csr_matrix(Xw_val)])

    # ensure CSR
    from scipy import sparse
    if sparse.issparse(X_train) and not isinstance(X_train, sparse.csr_matrix):
        X_train = X_train.tocsr()
    if sparse.issparse(X_val) and not isinstance(X_val, sparse.csr_matrix):
        X_val = X_val.tocsr()

    results = {}
    for aspect in aspect_cols:
        print(f"\nTraining aspect: {aspect}")
        y_train_raw = train_df[aspect].fillna("0").astype(int)
        y_val_raw = val_df[aspect].fillna("0").astype(int)

        y_train_pres = (y_train_raw > 0).astype(int)
        y_val_pres = (y_val_raw > 0).astype(int)

        pres_clf = LogisticRegression(max_iter=1000, solver='liblinear')
        pres_clf.fit(X_train, y_train_pres)
        joblib.dump(pres_clf, os.path.join(output_dir, f'presence_{aspect}.joblib'))

        y_val_pres_pred = pres_clf.predict(X_val)
        pres_acc = accuracy_score(y_val_pres, y_val_pres_pred)
        pres_f1_micro = f1_score(y_val_pres, y_val_pres_pred, average='micro')
        pres_f1_macro = f1_score(y_val_pres, y_val_pres_pred, average='macro')

        pos_idx = np.where(y_train_raw > 0)[0]
        sentiment_clf = None
        sent_acc = None
        sent_mae = None
        if len(pos_idx) >= 10:
            X_train_pos = X_train[pos_idx]
            y_train_sent = y_train_raw.iloc[pos_idx].astype(int)
            sentiment_clf = LogisticRegression(max_iter=1500, solver='saga', multi_class='auto')
            sentiment_clf.fit(X_train_pos, y_train_sent)
            joblib.dump(sentiment_clf, os.path.join(output_dir, f'sentiment_{aspect}.joblib'))

            val_pos_idx = np.where(y_val_raw > 0)[0]
            if len(val_pos_idx) > 0:
                X_val_pos = X_val[val_pos_idx]
                y_val_sent = y_val_raw.iloc[val_pos_idx].astype(int)
                y_val_sent_pred = sentiment_clf.predict(X_val_pos)
                sent_acc = accuracy_score(y_val_sent, y_val_sent_pred)
                sent_mae = mean_absolute_error(y_val_sent, y_val_sent_pred)

        results[aspect] = {
            'presence_acc': float(pres_acc),
            'presence_f1_micro': float(pres_f1_micro),
            'presence_f1_macro': float(pres_f1_macro),
            'sentiment_acc': float(sent_acc) if sent_acc is not None else None,
            'sentiment_mae': float(sent_mae) if sent_mae is not None else None,
            'n_pos_train': int((y_train_raw > 0).sum()),
            'n_pos_val': int((y_val_raw > 0).sum())
        }

    with open(os.path.join(output_dir, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('\nSaved models and summary to', output_dir)
    return results


# ===== Inference / Predict =====

def load_models_and_vectorizer(model_dir, aspect_cols):
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
    pres_models = {}
    sent_models = {}
    for aspect in aspect_cols:
        pth = os.path.join(model_dir, f'presence_{aspect}.joblib')
        if os.path.exists(pth):
            pres_models[aspect] = joblib.load(pth)
        spth = os.path.join(model_dir, f'sentiment_{aspect}.joblib')
        if os.path.exists(spth):
            sent_models[aspect] = joblib.load(spth)

    w2v_model = None
    w2v_path = os.path.join(model_dir, 'w2v.model')
    try:
        if os.path.exists(w2v_path):
            from gensim.models import Word2Vec
            w2v_model = Word2Vec.load(w2v_path)
            print('Loaded w2v model from', w2v_path)
    except Exception as e:
        print('Could not load w2v model:', e)

    return vectorizer, pres_models, sent_models, w2v_model


def evaluate_aggregate(val_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, w2v_model=None):
    from scipy import sparse
    X_val = transform_texts(vectorizer, val_df[text_col].tolist())

    if w2v_model is not None:
        Xw_val = texts_to_w2v_avg(w2v_model, val_df[text_col].tolist(), vector_size=getattr(w2v_model, 'vector_size', None))
        from scipy.sparse import hstack, csr_matrix
        if sparse.issparse(X_val) and not isinstance(X_val, csr_matrix):
            X_val = X_val.tocsr()
        X_val = hstack([X_val, csr_matrix(Xw_val)])
        if sparse.issparse(X_val) and not isinstance(X_val, csr_matrix):
            X_val = X_val.tocsr()
    else:
        if sparse.issparse(X_val) and not isinstance(X_val, sparse.csr_matrix):
            X_val = X_val.tocsr()

    presence_trues = []
    presence_preds = []
    sentiment_trues = []
    sentiment_preds = []

    for aspect in aspect_cols:
        y_true_raw = val_df[aspect].fillna("0").astype(int)
        y_true_pres = (y_true_raw > 0).astype(int)

        if aspect in pres_models:
            model = pres_models[aspect]
            expected = getattr(model, 'n_features_in_', None)
            if expected is None and hasattr(model, 'coef_'):
                try:
                    expected = int(model.coef_.shape[1])
                except Exception:
                    expected = None
            X_for_pres = align_features(X_val, expected)
            y_pred_pres = model.predict(X_for_pres)
        else:
            y_pred_pres = np.zeros(len(val_df), dtype=int)

        presence_trues.append(y_true_pres)
        presence_preds.append(y_pred_pres)

        pos_idx = np.where(y_true_raw > 0)[0]
        if len(pos_idx) > 0 and aspect in sent_models:
            sent_model = sent_models[aspect]
            expected_s = getattr(sent_model, 'n_features_in_', None)
            if expected_s is None and hasattr(sent_model, 'coef_'):
                try:
                    expected_s = int(sent_model.coef_.shape[1])
                except Exception:
                    expected_s = None
            X_full_aligned = align_features(X_val, expected_s)
            X_sub = X_full_aligned[pos_idx]
            y_true_sent = y_true_raw.iloc[pos_idx].astype(int)
            y_pred_sent = sent_model.predict(X_sub)
            sentiment_trues.append(y_true_sent)
            sentiment_preds.append(y_pred_sent)

    presence_trues = np.stack(presence_trues, axis=1)
    presence_preds = np.stack(presence_preds, axis=1)

    pres_micro = f1_score(presence_trues.flatten(), presence_preds.flatten(), average='micro')
    pres_macro = f1_score(presence_trues.flatten(), presence_preds.flatten(), average='macro')

    sent_accs = []
    sent_maes = []
    for t, p in zip(sentiment_trues, sentiment_preds):
        sent_accs.append(accuracy_score(t, p))
        sent_maes.append(mean_absolute_error(t, p))

    mean_sent_acc = float(np.mean(sent_accs)) if len(sent_accs) > 0 else None
    mean_sent_mae = float(np.mean(sent_maes)) if len(sent_maes) > 0 else None

    agg = {
        'presence_f1_micro': float(pres_micro),
        'presence_f1_macro': float(pres_macro),
        'mean_sent_acc': mean_sent_acc,
        'mean_sent_mae': mean_sent_mae
    }
    return agg


def predict_and_save(test_df, text_col, vectorizer, pres_models, sent_models, aspect_cols, output_csv, w2v_model=None):
    from scipy import sparse
    X_test = transform_texts(vectorizer, test_df[text_col].tolist())

    if w2v_model is not None:
        Xw_test = texts_to_w2v_avg(w2v_model, test_df[text_col].tolist(), vector_size=getattr(w2v_model, 'vector_size', None))
        from scipy.sparse import hstack, csr_matrix
        if sparse.issparse(X_test) and not isinstance(X_test, csr_matrix):
            X_test = X_test.tocsr()
        X_test = hstack([X_test, csr_matrix(Xw_test)])
        if sparse.issparse(X_test) and not isinstance(X_test, csr_matrix):
            X_test = X_test.tocsr()
    else:
        from scipy.sparse import csr_matrix
        if sparse.issparse(X_test) and not isinstance(X_test, csr_matrix):
            X_test = X_test.tocsr()

    out = pd.DataFrame()
    out['stt'] = test_df['stt']

    for aspect in aspect_cols:
        preds = np.zeros(len(test_df), dtype=int)
        if aspect in pres_models:
            model = pres_models[aspect]
            expected = getattr(model, 'n_features_in_', None)
            if expected is None and hasattr(model, 'coef_'):
                try:
                    expected = int(model.coef_.shape[1])
                except Exception:
                    expected = None
            X_for_pres = align_features(X_test, expected)
            pres_pred = model.predict(X_for_pres)
            preds = pres_pred.copy()
            if aspect in sent_models:
                idx_pos = np.where(pres_pred == 1)[0]
                if len(idx_pos) > 0:
                    sent_model = sent_models[aspect]
                    expected_s = getattr(sent_model, 'n_features_in_', None)
                    if expected_s is None and hasattr(sent_model, 'coef_'):
                        try:
                            expected_s = int(sent_model.coef_.shape[1])
                        except Exception:
                            expected_s = None
                    X_for_sent = align_features(X_test, expected_s)
                    X_sub = X_for_sent[idx_pos]
                    sent_pred = sent_model.predict(X_sub)
                    preds[idx_pos] = sent_pred
        out[aspect] = preds

    out.to_csv(output_csv, index=False)
    print('Saved predictions to', output_csv)
    return out


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(description='Train & Predict BR presence + sentiment')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--train_csv', type=str, help='Train CSV path (train-problem.csv)')
    parser.add_argument('--test_csv', type=str, help='Test CSV path (gt_reviews.csv)')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_dir', type=str, help='If predict mode, load models from this dir')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--aspect_cols', type=str, default='giai_tri,luu_tru,nha_hang,an_uong,van_chuyen,mua_sam',
                        help='Comma-separated aspect column names')
    parser.add_argument('--output_csv', type=str, default='predict.csv')
    args = parser.parse_args()

    aspect_cols = [c.strip() for c in args.aspect_cols.split(',') if c.strip()]

    if args.mode == 'train':
        if not args.train_csv:
            raise ValueError('train_csv is required for train mode')
        df = safe_read_csv(args.train_csv)
        df, text_col = preprocess_df(df)
        for c in aspect_cols:
            if c not in df.columns:
                df[c] = 0
        train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=args.random_seed)
        os.makedirs(args.output_dir, exist_ok=True)

        results = train_br(train_df.reset_index(drop=True), val_df.reset_index(drop=True), text_col, aspect_cols, args.output_dir)

        vectorizer, pres_models, sent_models, w2v_model = load_models_and_vectorizer(args.output_dir, aspect_cols)
        agg = evaluate_aggregate(val_df.reset_index(drop=True), text_col, vectorizer, pres_models, sent_models, aspect_cols, w2v_model=w2v_model)
        print('\nAggregate on validation:', agg)

    elif args.mode == 'predict':
        model_dir = args.model_dir if args.model_dir else args.output_dir
        if not args.test_csv:
            raise ValueError('test_csv is required for predict mode')
        test_df = safe_read_csv(args.test_csv)
        test_df, text_col = preprocess_df(test_df)

        vectorizer, pres_models, sent_models, w2v_model = load_models_and_vectorizer(model_dir, aspect_cols)
        out = predict_and_save(test_df.reset_index(drop=True), text_col, vectorizer, pres_models, sent_models, aspect_cols, args.output_csv, w2v_model=w2v_model)


if __name__ == '__main__':
    main()
