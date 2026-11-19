#!/usr/bin/env python3
"""
compute_shapley.py
==================
SHAP values on the test set for the enhancer baselines trained with baselines_enhancers.py.

  xgb          - TreeExplainer (exact, fast). Values are in log-odds space.
  svm_linear,
  svm_rbf      - KernelExplainer on predict_proba (slow, runs in parallel chunks).
                 Values are in probability space for the positive class.

Uses the same --dataset / --features flags as baselines_enhancers.py, so the data
matches what the model was trained on. Warns if they don't match args.json in the results dir.

Usage
  python compute_shapley.py --dataset c1 --features histone
  python compute_shapley.py --dataset all --features dna --models xgb
  python compute_shapley.py --dataset c1 --models svm_rbf --n_workers 32 --background_size 10

  # older results dirs
  python compute_shapley.py --dataset c1 --features histone --results_dir ../results/final_enhancers_histone_c1

Output (in the results dir)
  {linear,rbf,xgb}_shap_values.csv         raw SHAP values, test samples x features
  {linear_svm,rbf_svm,xgb}_shap_values_rank.csv
      per-feature ranking: mean SHAP on positives / negatives, separation, mean |SHAP|,
      factor name and factor type (sorted by signed_pos)
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import joblib
import shap
from joblib import Parallel, delayed

from dataloaders import EnhancersDatasetClassicalSimple
from baselines_enhancers import (
    DATA_DIR, TRACK_MATRIX, DATASETS, FEATURES, MODELS, MODEL_FILE, SRC_DIR, get_split_data,
)

METADATA = "/s/chromatin/c/nobackup/Saira/BERTModel/merged_metadata_AT_OCR.csv"

# output names (kept same as older runs / feature_importance.ipynb)
SHAP_FILE = {"svm_linear": "linear_shap_values.csv", "svm_rbf": "rbf_shap_values.csv", "xgb": "xgb_shap_values.csv"}
RANK_FILE = {"svm_linear": "linear_svm_shap_values_rank.csv", "svm_rbf": "rbf_svm_shap_values_rank.csv",
             "xgb": "xgb_shap_values_rank.csv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SHAP values for SVM and XGBoost enhancer models.")
    parser.add_argument("--dataset", choices=list(DATASETS), default="c1")
    parser.add_argument("--features", default="all",
                        help=f"feature set: {list(FEATURES)} or a path to a .npy mask")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--results_dir", default=None,
                        help="dir with saved models. default: ../results/enhancers_<dataset>_<features>")
    # overrides
    parser.add_argument("--track_matrix", default=os.path.join(DATA_DIR, TRACK_MATRIX))
    parser.add_argument("--labels", default=None, help="labels file. Overrides --dataset")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--metadata", default=METADATA, help="for factor name / type in the rank tables")
    # kernel SHAP
    parser.add_argument("--n_workers", type=int, default=None, help="default: all CPUs")
    parser.add_argument("--background_size", type=int, default=10,
                        help="number of training samples used as Kernel SHAP background")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = os.path.join(DATA_DIR, DATASETS[args.dataset])
    args.filter_by = FEATURES.get(args.features, args.features)
    if args.filter_by is not None and not os.path.isabs(args.filter_by):
        args.filter_by = os.path.join(SRC_DIR, args.filter_by)
    if args.results_dir is None:
        feat_tag = os.path.splitext(os.path.basename(args.features))[0]
        args.results_dir = os.path.join(SRC_DIR, "..", "results", f"enhancers_{args.dataset}_{feat_tag}")
    return args


def check_against_training_args(args):
    """Warn if the data we load here is not what the model was trained on."""
    path = os.path.join(args.results_dir, "args.json")
    if not os.path.exists(path):
        print(f"  (no args.json in {args.results_dir}, can't check data matches training)")
        return
    with open(path) as f:
        train_args = json.load(f)

    def base(p):
        return os.path.basename(p) if p else None

    for key in ["track_matrix", "labels", "filter_by"]:
        if base(train_args.get(key)) != base(getattr(args, key)):
            print(f"  WARNING: {key} differs from training run: "
                  f"trained on {base(train_args.get(key))}, now using {base(getattr(args, key))}")


def kernel_shap(model, background, X, n_workers):
    explainer = shap.KernelExplainer(model.predict_proba, data=background)

    # split test rows into chunks and run them in parallel
    chunk_size = max(1, len(X) // (n_workers * 4))
    chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
    n_workers = min(n_workers, len(chunks))
    print(f"  Kernel SHAP: {len(X)} rows, {len(chunks)} chunks, {n_workers} workers, "
          f"background={len(background)}")

    results = Parallel(n_jobs=n_workers)(delayed(explainer.shap_values)(c) for c in chunks)
    shap_values = np.concatenate(results, axis=0)

    # (n_samples, n_features, n_classes) -> keep positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def rank_features(shap_values, y, feature_names, metadata):
    # same ranking as feature_importance.ipynb
    signed_pos = shap_values[y == 1].mean(axis=0)
    signed_neg = shap_values[y == 0].mean(axis=0)
    rank = pd.DataFrame({
        "feature":    feature_names,
        "signed_pos": signed_pos,
        "signed_neg": signed_neg,
        "separation": signed_pos - signed_neg,
        "abs_infl":   np.abs(shap_values).mean(axis=0),
    }).sort_values("signed_pos", ascending=False)

    acc_to_factor = dict(zip(metadata["Sample accession number"], metadata["Factor_y"]))
    acc_to_type = dict(zip(metadata["Sample accession number"], metadata["Factor_type"]))
    acc = rank["feature"].str.split(".").str[0]
    rank["factor name"] = [acc_to_factor.get(a, a) for a in acc]
    rank["factor type"] = [acc_to_type.get(a, "unknown") for a in acc]
    return rank


def main():
    args = parse_args()
    print(f"Results dir: {args.results_dir}")
    check_against_training_args(args)

    # Dataset
    dataset = EnhancersDatasetClassicalSimple(
        track_matrix_path=args.track_matrix,
        labels_file_path=args.labels,
        split_file=args.split_file,
        filter_by=args.filter_by,
    )
    data = get_split_data(dataset, use_split_file=args.split_file is not None)
    X_train, _, _ = data["train"]
    X_test, y_test, _ = data["test"]

    # feature names = track matrix header (minus 'actual'), filtered by the mask
    feature_names = pd.read_csv(args.track_matrix, index_col=0, nrows=0).columns.drop("actual")
    if args.filter_by is not None:
        feature_names = feature_names[np.load(args.filter_by).ravel().astype(bool)]
    feature_names = list(feature_names)
    assert len(feature_names) == X_test.shape[1], (len(feature_names), X_test.shape)

    metadata = pd.read_csv(args.metadata, index_col=0)

    # Compute SHAP
    for name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Computing SHAP values for: {name}")
        print(f"{'=' * 60}")

        model = joblib.load(os.path.join(args.results_dir, MODEL_FILE[name]))["model"]
        t0 = time.time()

        if name == "xgb":
            shap_values = shap.TreeExplainer(model).shap_values(X_test)
        else:
            # random training samples as background
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(X_train), size=min(args.background_size, len(X_train)), replace=False)
            n_workers = args.n_workers or os.cpu_count() or 1
            shap_values = kernel_shap(model, X_train[idx], X_test, n_workers)

        print(f"  done in {time.time() - t0:.1f}s, shape {shap_values.shape}")

        # Save SHAP values + ranking
        out = os.path.join(args.results_dir, SHAP_FILE[name])
        pd.DataFrame(shap_values, columns=feature_names).to_csv(out, index=False)
        print(f"  Saved {out}")

        rank = rank_features(shap_values, y_test, feature_names, metadata)
        out = os.path.join(args.results_dir, RANK_FILE[name])
        rank.to_csv(out, index=False)
        print(f"  Saved {out}")
        print(rank.head(10)[["feature", "factor name", "factor type", "signed_pos", "abs_infl"]].to_string(index=False))


if __name__ == "__main__":
    main()
