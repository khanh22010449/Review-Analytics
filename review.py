#!/usr/bin/env python3
# train_and_predict_final.py
# Yêu cầu: python3, pandas, scikit-learn, tqdm
# pip install pandas scikit-learn tqdm

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

ASPECT_COLS = ['giai_tri','luu_tru','nha_hang','an_uong','van_chuyen','mua_sam']
TEXT_COL_TRAIN = "Review"
TEXT_COL_TEST = "review"  # column in gt_reviews.csv expected

def safe_read_csv(path):
    df = pd.read_csv(path)
    # Strip BOM or extra spaces in column names
    df.columns = [c.strip() for c in df.columns]
    return df

def train_and_predict(train_csv, test_csv, out_predict, out_results,
                      tfidf_max_features=20000):
    # --- Load ---
    df = safe_read_csv(train_csv)
    df_test = safe_read_csv(test_csv)

    # Validate columns
    for c in ASPECT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing aspect column in train file: {c}")
    if TEXT_COL_TRAIN not in df.columns:
        # try common alternatives
        if "review" in df.columns:
            df[TEXT_COL_TRAIN] = df["review"]
        else:
            raise ValueError(f"Missing text column in train file: {TEXT_COL_TRAIN}")

    if TEXT_COL_TEST not in df_test.columns:
        # try alternative 'Review' or 'text'
        if "Review" in df_test.columns:
            df_test[TEXT_COL_TEST] = df_test["Review"]
        elif "text" in df_test.columns:
            df_test[TEXT_COL_TEST] = df_test["text"]
        else:
            # if none, create empty reviews
            df_test[TEXT_COL_TEST] = ""

    # Ensure stt exists in test for output indexing
    if "stt" not in df_test.columns:
        df_test.insert(0, "stt", range(1, len(df_test) + 1))

    # --- Split train/val ---
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2))
    vectorizer.fit(train_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_train = vectorizer.transform(train_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_val = vectorizer.transform(val_df[TEXT_COL_TRAIN].fillna("").astype(str))
    X_test = vectorizer.transform(df_test[TEXT_COL_TEST].fillna("").astype(str))

    # --- Models containers ---
    presence_models = {}
    sent_models = {}
    results = {}

    # --- Train per-aspect ---
    for c in ASPECT_COLS:
        # Presence target: 0 if not mentioned, 1 if rating>0
        y_train_pres = (train_df[c].fillna(0) > 0).astype(int)
        y_val_pres = (val_df[c].fillna(0) > 0).astype(int)

        # train presence classifier
        clf_pres = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        clf_pres.fit(X_train, y_train_pres)
        presence_models[c] = clf_pres

        val_pred_pres = clf_pres.predict(X_val)
        results[f"{c}_presence"] = {
            "acc": float(accuracy_score(y_val_pres, val_pred_pres)),
            "f1_micro": float(f1_score(y_val_pres, val_pred_pres, average='micro', zero_division=0)),
            "f1_macro": float(f1_score(y_val_pres, val_pred_pres, average='macro', zero_division=0)),
            "n_val": int(len(y_val_pres))
        }

        # Sentiment classifier: train only on rows with rating > 0
        train_idx = train_df[train_df[c].fillna(0) > 0].index
        val_idx = val_df[val_df[c].fillna(0) > 0].index

        if len(train_idx) < 10:
            # not enough samples to train reliable sentiment model
            sent_models[c] = None
            results[f"{c}_sent"] = {"acc": None, "mae": None, "n_val": int(len(val_idx))}
            continue

        y_train_sent = train_df.loc[train_idx, c].astype(int)
        y_val_sent = val_df.loc[val_idx, c].astype(int)

        X_train_sent = X_train[train_df.index.get_indexer(train_idx)]
        X_val_sent = X_val[val_df.index.get_indexer(val_idx)]

        clf_sent = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
        clf_sent.fit(X_train_sent, y_train_sent)
        sent_models[c] = clf_sent

        if len(y_val_sent) > 0:
            val_pred_sent = clf_sent.predict(X_val_sent)
            results[f"{c}_sent"] = {
                "acc": float(accuracy_score(y_val_sent, val_pred_sent)),
                "mae": float(mean_absolute_error(y_val_sent, val_pred_sent)),
                "n_val": int(len(y_val_sent))
            }
        else:
            results[f"{c}_sent"] = {"acc": None, "mae": None, "n_val": 0}

    # --- Aggregate multi-label metrics (on val) ---
    val_pres_preds = np.vstack([presence_models[c].predict(X_val) for c in ASPECT_COLS]).T
    val_pres_tgts = np.vstack([(val_df[c].fillna(0) > 0).astype(int).values for c in ASPECT_COLS]).T
    micro_f1 = float(f1_score(val_pres_tgts.reshape(-1), val_pres_preds.reshape(-1), average='micro', zero_division=0))
    macro_f1 = float(f1_score(val_pres_tgts.reshape(-1), val_pres_preds.reshape(-1), average='macro', zero_division=0))
    overall_row_equal_acc = float((val_pres_preds == val_pres_tgts).mean())

    # --- Aggregate sentiment metrics ---
    sent_accs = []
    sent_maes = []
    for c in ASPECT_COLS:
        if sent_models.get(c) is None:
            continue
        mask_idx = val_df[val_df[c].fillna(0) > 0].index
        if len(mask_idx) == 0:
            continue
        X_val_sent = X_val[val_df.index.get_indexer(mask_idx)]
        y_val_sent = val_df.loc[mask_idx, c].astype(int).values
        y_pred_sent = sent_models[c].predict(X_val_sent)
        sent_accs.append(accuracy_score(y_val_sent, y_pred_sent))
        sent_maes.append(mean_absolute_error(y_val_sent, y_pred_sent))
    mean_sent_acc = float(np.mean(sent_accs)) if len(sent_accs) > 0 else None
    mean_sent_mae = float(np.mean(sent_maes)) if len(sent_maes) > 0 else None

    # --- Save results to out_results ---
    with open(out_results, "w", encoding="utf-8") as f:
        f.write("Multi-label presence metrics (per aspect):\n")
        for c in ASPECT_COLS:
            r = results.get(f"{c}_presence", {})
            f.write(f"{c}: acc={r.get('acc',0):.4f}, f1_micro={r.get('f1_micro',0):.4f}, f1_macro={r.get('f1_macro',0):.4f}, n_val={r.get('n_val',0)}\n")
        f.write("\nAggregated multi-label (val set):\n")
        f.write(f"overall_rows_exact_match_acc: {overall_row_equal_acc:.4f}\n")
        f.write(f"micro_f1: {micro_f1:.4f}\n")
        f.write(f"macro_f1: {macro_f1:.4f}\n\n")
        f.write("Sentiment (per-aspect) metrics:\n")
        for c in ASPECT_COLS:
            r = results.get(f"{c}_sent", {})
            if r.get("acc") is None:
                f.write(f"{c}: no sentiment model (too few train samples), val_n={r.get('n_val',0)}\n")
            else:
                f.write(f"{c}: acc={r['acc']:.4f}, mae={r['mae']:.4f}, val_n={r['n_val']}\n")
        f.write("\nAggregated sentiment:\n")
        f.write(f"mean_sent_acc: {mean_sent_acc}\n")
        f.write(f"mean_sent_mae: {mean_sent_mae}\n")

    # --- Predict on test set and format final predict.csv ---
    final_pred = pd.DataFrame()
    final_pred["stt"] = df_test["stt"].astype(int)

    # For each aspect: predict presence; if presence==1 then predict rating (if model exists) else 0
    for c in ASPECT_COLS:
        pres = presence_models[c].predict(X_test)  # 0/1 array
        if sent_models.get(c) is not None:
            # predict rating for all test rows, then mask by pres
            pred_rating_all = sent_models[c].predict(X_test)  # values 1..5
            pred_rating_masked = np.where(pres == 1, pred_rating_all, 0)
        else:
            pred_rating_masked = np.zeros(X_test.shape[0], dtype=int)
        final_pred[c] = pred_rating_masked.astype(int)

    # Save final predict.csv in requested compact format
    final_pred.to_csv(out_predict, index=False, encoding="utf-8")
    print(f"Saved predict file: {out_predict}")
    print(f"Saved results: {out_results}")

    return out_predict, out_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train-problem.csv", help="path to train csv")
    parser.add_argument("--test", default="gt_reviews.csv", help="path to test csv")
    parser.add_argument("--out_predict", default="predict.csv", help="output predict csv path")
    parser.add_argument("--out_results", default="results.txt", help="output results txt path")
    args = parser.parse_args()

    train_and_predict(args.train, args.test, args.out_predict, args.out_results)
