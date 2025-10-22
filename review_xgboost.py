#!/usr/bin/env python3
"""
improved_review_xgb.py
Phiên bản dùng XGBoost để cải thiện mean_sent_acc và classification.
- Dùng XGBoost (XGBClassifier cho presence và sentiment; XGBRegressor làm ordinal fallback)
- Tiền xử lý TF-IDF giống trước (ngram 1..3, min_df=2)
- Upsampling balanced cho sentiment
- Chọn mô hình giữa XGBClassifier và XGBRegressor dựa trên validation accuracy cho mỗi aspect

Chạy:
python improved_review_xgb.py --train train-problem.csv --test gt_reviews.csv --out_predict predict.csv --out_results results.txt

Yêu cầu:
pip install xgboost scikit-learn pandas numpy
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

# XGBoost
try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("Xin cài đặt xgboost: pip install xgboost")

SEED = 42
np.random.seed(SEED)

ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
TEXT_COL_TRAIN = "Review"
TEXT_COL_TEST = "review"


def safe_read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def upsample_indices(y):
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    idxs = np.arange(len(y))
    resampled = []
    for cls in classes:
        cls_idx = idxs[y == cls]
        res = np.random.choice(cls_idx, size=max_count, replace=True)
        resampled.append(res)
    all_idx = np.concatenate(resampled)
    np.random.shuffle(all_idx)
    return all_idx


def train_and_predict(train_csv, test_csv, out_predict, out_results,
                      tfidf_max_features=30000):
    df = safe_read_csv(train_csv)
    df_test = safe_read_csv(test_csv)

    # ensure text columns
    if TEXT_COL_TRAIN not in df.columns:
        if 'review' in df.columns:
            df[TEXT_COL_TRAIN] = df['review']
        else:
            raise ValueError(f"Missing text column in train: {TEXT_COL_TRAIN}")

    if TEXT_COL_TEST not in df_test.columns:
        if 'Review' in df_test.columns:
            df_test[TEXT_COL_TEST] = df_test['Review']
        elif 'text' in df_test.columns:
            df_test[TEXT_COL_TEST] = df_test['text']
        else:
            df_test[TEXT_COL_TEST] = ""

    if 'stt' not in df_test.columns:
        df_test.insert(0, 'stt', range(1, len(df_test) + 1))

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,3), min_df=2)
    vectorizer.fit(train_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_train = vectorizer.transform(train_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_val = vectorizer.transform(val_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_test = vectorizer.transform(df_test[TEXT_COL_TEST].fillna("").astype(str))

    presence_models = {}
    sent_models = {}
    results = {}

    # common XGBoost parameters (can tune)
    xgb_clf_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'verbosity': 0
    }
    xgb_multi_params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'verbosity': 0
    }
    xgb_reg_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': SEED,
        'verbosity': 0
    }

    for c in ASPECT_COLS:
        # Presence model (binary)
        y_train_pres = (train_df[c].fillna(0) > 0).astype(int).values
        y_val_pres = (val_df[c].fillna(0) > 0).astype(int).values

        clf_pres = xgb.XGBClassifier(**xgb_clf_params)
        clf_pres.fit(X_train, y_train_pres)
        presence_models[c] = clf_pres

        val_pred_pres = clf_pres.predict(X_val)
        results[f"{c}_presence"] = {
            'acc': float(accuracy_score(y_val_pres, val_pred_pres)),
            'f1_micro': float(f1_score(y_val_pres, val_pred_pres, average='micro', zero_division=0)),
            'n_val': int(len(y_val_pres))
        }

        # Sentiment models
        train_idx = train_df[train_df[c].fillna(0) > 0].index
        val_idx = val_df[val_df[c].fillna(0) > 0].index

        if len(train_idx) < 8:
            sent_models[c] = None
            results[f"{c}_sent"] = {'acc': None, 'mae': None, 'n_val': int(len(val_idx))}
            continue

        y_train_sent = train_df.loc[train_idx, c].astype(int).values
        y_val_sent = val_df.loc[val_idx, c].astype(int).values
        X_train_sent = X_train[train_df.index.get_indexer(train_idx)]
        X_val_sent = X_val[val_df.index.get_indexer(val_idx)]

        # Upsample to balance classes
        try:
            up_idx = upsample_indices(y_train_sent)
            X_train_sent_bal = X_train_sent[up_idx]
            y_train_sent_bal = y_train_sent[up_idx]
        except Exception:
            X_train_sent_bal = X_train_sent
            y_train_sent_bal = y_train_sent

        # XGBoost multiclass classifier (labels 1..5 -> convert to 0..4)
        clf_multi = xgb.XGBClassifier(**{**xgb_multi_params, 'num_class': 5})
        clf_multi.fit(X_train_sent_bal, (y_train_sent_bal - 1))

        # XGBoost regressor (ordinal-aware)
        reg = xgb.XGBRegressor(**xgb_reg_params)
        reg.fit(X_train_sent_bal, y_train_sent_bal.astype(float))

        # Evaluate on val
        val_pred_clf = clf_multi.predict(X_val_sent) + 1 if len(y_val_sent) > 0 else np.array([])
        val_pred_reg = np.clip(np.rint(reg.predict(X_val_sent)), 1, 5).astype(int) if len(y_val_sent) > 0 else np.array([])

        acc_clf = accuracy_score(y_val_sent, val_pred_clf) if len(y_val_sent) > 0 else -1
        acc_reg = accuracy_score(y_val_sent, val_pred_reg) if len(y_val_sent) > 0 else -1

        # Choose best model (prefer reg when close)
        if acc_reg + 1e-9 >= acc_clf:
            sent_models[c] = ('reg', reg)
            chosen_acc = acc_reg
            chosen_mae = mean_absolute_error(y_val_sent, val_pred_reg) if len(y_val_sent) > 0 else None
        else:
            sent_models[c] = ('clf', clf_multi)
            chosen_acc = acc_clf
            chosen_mae = mean_absolute_error(y_val_sent, val_pred_clf) if len(y_val_sent) > 0 else None

        results[f"{c}_sent"] = {'acc': float(chosen_acc) if chosen_acc is not None else None,
                                   'mae': float(chosen_mae) if chosen_mae is not None else None,
                                   'n_val': int(len(y_val_sent))}

    # Aggregate multi-label metrics on val
    val_pres_preds = np.vstack([presence_models[c].predict(X_val) for c in ASPECT_COLS]).T
    val_pres_tgts = np.vstack([(val_df[c].fillna(0) > 0).astype(int).values for c in ASPECT_COLS]).T
    micro_f1 = float(f1_score(val_pres_tgts.reshape(-1), val_pres_preds.reshape(-1), average='micro', zero_division=0))

    # Sentiment aggregate
    sent_accs = []
    sent_maes = []
    for c in ASPECT_COLS:
        m = sent_models.get(c)
        if m is None:
            continue
        mask_idx = val_df[val_df[c].fillna(0) > 0].index
        if len(mask_idx) == 0:
            continue
        X_val_sent = X_val[val_df.index.get_indexer(mask_idx)]
        y_val_sent = val_df.loc[mask_idx, c].astype(int).values
        if m[0] == 'clf':
            y_pred = m[1].predict(X_val_sent) + 1
        else:
            y_pred = np.clip(np.rint(m[1].predict(X_val_sent)), 1, 5).astype(int)
        sent_accs.append(accuracy_score(y_val_sent, y_pred))
        sent_maes.append(mean_absolute_error(y_val_sent, y_pred))

    mean_sent_acc = float(np.mean(sent_accs)) if len(sent_accs) > 0 else None
    mean_sent_mae = float(np.mean(sent_maes)) if len(sent_maes) > 0 else None

    # Write results
    with open(out_results, 'w', encoding='utf-8') as f:
        f.write('Per-aspect presence:\n')
        for c in ASPECT_COLS:
            r = results.get(f"{c}_presence", {})
            f.write(f"{c}: acc={r.get('acc',0):.4f}, f1_micro={r.get('f1_micro',0):.4f}, n_val={r.get('n_val',0)}\n")
        f.write('\nPer-aspect sentiment:\n')
        for c in ASPECT_COLS:
            r = results.get(f"{c}_sent", {})
            if r.get('acc') is None:
                f.write(f"{c}: no sent model (too few samples), val_n={r.get('n_val',0)}\n")
            else:
                f.write(f"{c}: acc={r['acc']:.4f}, mae={r['mae']:.4f}, val_n={r['n_val']}\n")
        f.write('\nAggregates:\n')
        f.write(f"micro_f1_presence: {micro_f1:.4f}\n")
        f.write(f"mean_sent_acc: {mean_sent_acc}\n")
        f.write(f"mean_sent_mae: {mean_sent_mae}\n")

    # Predict on test
    final_pred = pd.DataFrame()
    final_pred['stt'] = df_test['stt'].astype(int)
    for c in ASPECT_COLS:
        pres = presence_models[c].predict(X_test)
        if sent_models.get(c) is None:
            pred_rating_masked = np.zeros(X_test.shape[0], dtype=int)
        else:
            m = sent_models[c]
            if m[0] == 'clf':
                pred_all = m[1].predict(X_test) + 1
            else:
                pred_all = np.clip(np.rint(m[1].predict(X_test)), 1, 5).astype(int)
            pred_rating_masked = np.where(pres == 1, pred_all, 0)
        final_pred[c] = pred_rating_masked.astype(int)

    final_pred.to_csv(out_predict, index=False, encoding='utf-8')
    print(f"Saved predict: {out_predict}")
    print(f"Saved results: {out_results}")
    return out_predict, out_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train-problem.csv')
    parser.add_argument('--test', default='gt_reviews.csv')
    parser.add_argument('--out_predict', default='predict.csv')
    parser.add_argument('--out_results', default='results.txt')
    args = parser.parse_args()
    train_and_predict(args.train, args.test, args.out_predict, args.out_results)
