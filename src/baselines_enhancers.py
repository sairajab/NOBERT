#!/usr/bin/env python3
"""
baselines_enhancers.py
======================
SVM (linear + rbf) and XGBoost baselines for enhancer classification, to compare against ChromBERT.

Input  : discretized chromatin tokens (1-4) for the 500bp window centered on each region, 2689 tracks
Target : enhancer (1) vs background (0)

Datasets (--dataset)
  c1   - C1 enhancers vs background       DATA/enhancers/C1_2500_enhancer_dataset.txt
  all  - all enhancers vs background      DATA/enhancers/enhancers_2500_labels_no_duplicates.txt

Feature sets (--features), masks over the 2689 tracks
  all      - every track
  dna      - DNA-binding factors   (dna_binding_mask.npy, 1509 tracks)
  histone  - histone marks         (histone_mask.npy, 1014 tracks)
  tf       - TFs                   (tf_mask.npy, 1211 tracks)
  or a path to any other .npy mask, e.g. --features new_dna_mask.npy

Usage
  python baselines_enhancers.py --dataset c1
  python baselines_enhancers.py --dataset all --features dna
  python baselines_enhancers.py --dataset c1  --features histone --models xgb
  python baselines_enhancers.py --dataset all --grid           # redo grid search (CV on train) instead of saved best params
  python baselines_enhancers.py --dataset c1  --eval_only      # reload saved models from outdir, redo metrics + plots

Defaults reproduce ../results/final_enhancers_* (best params from earlier grid search,
final models trained on train+val, evaluated on test).

Output (in --output_dir, default ../results/enhancers_<dataset>_<features>)
  svm_model_linear.joblib, svm_model_rbf.joblib, xgb_model_l2.joblib
  metrics_summary.json   test metrics + train/inference time per model, and params
  test_predictions.csv   region, label, prob per model (for ROC/PR comparisons with ChromBERT)
  roc_pr_test.png, confusion_test.png, args.json
"""

import os
import sys
import time
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from dataloaders import EnhancersDatasetClassicalSimple


DATA_DIR = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers"
TRACK_MATRIX = "enhancers_500bp_centered_final.csv"  # tokens 1-4, used for final_enhancers_* runs

DATASETS = {
    "c1":  "C1_2500_enhancer_dataset.txt",
    "all": "enhancers_2500_labels_no_duplicates.txt",
}
FEATURES = {
    "all":     None,
    "dna":     "dna_binding_mask.npy",
    "histone": "histone_mask.npy",
    "tf":      "tf_mask.npy",
}
MODELS = ["svm_linear", "svm_rbf", "xgb"]

# key names in metrics_summary.json (kept same as older runs)
METRIC_KEY = {"svm_linear": "SVM_test_linear", "svm_rbf": "SVM_test_rbf", "xgb": "XGB_test"}
PARAM_KEY  = {"svm_linear": "SVM_linear_params", "svm_rbf": "SVM_rbf_params", "xgb": "XGB_params"}
MODEL_FILE = {"svm_linear": "svm_model_linear.joblib", "svm_rbf": "svm_model_rbf.joblib", "xgb": "xgb_model_l2.joblib"}

# ---------- best params (from earlier grid search, used for final_enhancers_*) ----------
BEST_PARAMS = {
    "svm_linear": {"kernel": "linear", "C": 0.1},
    "svm_rbf":    {"kernel": "rbf", "C": 10, "gamma": "scale"},
    "xgb": {
        "learning_rate":    0.05,
        "max_depth":        8,
        "min_child_weight": 5,
        "n_estimators":     979,
        "subsample":        0.8,
        "colsample_bytree": 1.0,
        "reg_lambda":       1.0,
        "tree_method":      "exact",
    },
}

# ---------- grids for --grid (CV on train, scored by average precision) ----------
GRIDS = {
    "svm_linear": {"C": [0.01, 0.1, 1, 10]},
    "svm_rbf":    {"C": [0.1, 1, 10, 100], "gamma": ["scale", 1e-6, 1e-4, 1e-2, 1]},
    "xgb": {
        "max_depth":        [4, 6, 8, 10],
        "learning_rate":    [0.05, 0.1, 0.2],
        "min_child_weight": [1, 2, 5, 7, 10],
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════

def get_split_data(dataset, use_split_file=False):
    """Returns {split: (X, y, regions)}. Regions missing from the track matrix are dropped."""
    if use_split_file:
        splits = dataset.create_splits_from_data()
    else:
        splits = dataset.create_splits_from_labels_data()

    out = {}
    for split_name, idxs in splits.items():
        keep = [i for i in idxs if dataset.sample_row_indices[i] is not None]
        rows = [dataset.sample_row_indices[i] for i in keep]
        lab = dataset.labels_data.iloc[keep]

        X = dataset.feature_values[rows]
        y = lab["Label"].to_numpy(dtype=np.int64)
        regions = (lab["Chromosome"].astype(str) + ":" + lab["Start"].astype(int).astype(str)
                   + "-" + lab["End"].astype(int).astype(str)).tolist()
        out[split_name] = (X, y, regions)

        dropped = len(idxs) - len(keep)
        print(f"  {split_name:5s}: X{X.shape}, pos={y.sum()}, neg={len(y) - y.sum()}"
              + (f"  (dropped {dropped} not in track matrix)" if dropped else ""))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════════

def make_model(name, params, scale_pos_weight, seed=42):
    if name.startswith("svm"):
        return SVC(probability=True, class_weight="balanced", random_state=seed, **params)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def grid_search(name, X, y, scale_pos_weight, args):
    """CV grid search on train only, returns best params (base params + grid winners)."""
    cv = StratifiedKFold(n_splits=min(args.cv_folds, min(np.bincount(y))), shuffle=True, random_state=args.seed)

    if name.startswith("svm"):
        base_params = {"kernel": BEST_PARAMS[name]["kernel"]}
        # no probability=True here, AP is scored from decision_function (much faster)
        estimator = SVC(class_weight="balanced", cache_size=8000, max_iter=10000,
                        random_state=args.seed, **base_params)
    else:
        base_params = {**BEST_PARAMS["xgb"], "tree_method": "hist"}  # hist is a lot faster for the search
        for k in GRIDS["xgb"]:
            base_params.pop(k)
        estimator = make_model("xgb", base_params, scale_pos_weight, args.seed)
        estimator.set_params(n_jobs=1)  # parallelize over grid points instead

    print(f"  Grid search {name}: {GRIDS[name]}")
    t0 = time.time()
    gs = GridSearchCV(estimator, GRIDS[name], cv=cv, scoring="average_precision",
                      n_jobs=args.n_jobs, verbose=1, refit=False)
    gs.fit(X, y)
    print(f"  Grid search done in {time.time() - t0:.1f}s")
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV AP:  {gs.best_score_:.4f}")
    return {**base_params, **gs.best_params_}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, split_name="test", model_name=""):
    """Compute full classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        metrics["auroc"] = roc_auc_score(y, y_prob)
        metrics["auprc"] = average_precision_score(y, y_prob)

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — {split_name.upper()} SET RESULTS")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    print(f"\n{classification_report(y, y_pred, target_names=['Negative', 'Positive'], zero_division=0)}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y, y_pred)}\n")

    return metrics, y_pred, y_prob


def plot_roc_pr(y_true, y_prob_dict, output_dir, split_name="test"):
    """ROC and PR curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, y_prob in y_prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, y_prob):.3f})")
    baseline = y_true.sum() / len(y_true)
    ax.axhline(baseline, color="k", ls="--", alpha=0.4, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {split_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"roc_pr_{split_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_confusion_matrices(y_true, predictions_dict, output_dir, split_name="test"):
    """Side-by-side confusion matrices."""
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} — {split_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_{split_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM & XGBoost on enhancers data")
    parser.add_argument("--dataset", choices=list(DATASETS), default="c1",
                        help="c1 = C1 enhancers, all = all enhancers")
    parser.add_argument("--features", default="all",
                        help=f"feature set: {list(FEATURES)} or a path to a .npy mask")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--grid", action="store_true",
                        help="grid search (CV on train) instead of using saved best params")
    parser.add_argument("--eval_only", action="store_true",
                        help="load saved models from output_dir and just evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="default: ../results/enhancers_<dataset>_<features>")
    # overrides
    parser.add_argument("--track_matrix", type=str, default=os.path.join(DATA_DIR, TRACK_MATRIX))
    parser.add_argument("--labels", type=str, default=None,
                        help="labels file (.txt, tab-separated, no header). Overrides --dataset")
    parser.add_argument("--split_file", type=str, default=None,
                        help="optional split CSV. Default: use Split column in labels")
    # grid search
    parser.add_argument("--n_jobs", type=int, default=4, help="parallel jobs for grid search")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = os.path.join(DATA_DIR, DATASETS[args.dataset])

    args.filter_by = FEATURES.get(args.features, args.features)
    if args.filter_by is not None and not os.path.isabs(args.filter_by):
        args.filter_by = os.path.join(SRC_DIR, args.filter_by)

    if args.output_dir is None:
        feat_tag = os.path.splitext(os.path.basename(args.features))[0]
        args.output_dir = os.path.join(SRC_DIR, "..", "results", f"enhancers_{args.dataset}_{feat_tag}")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.eval_only:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    #  1. Load dataset
    print("=" * 70)
    print(f"  LOADING DATASET  (dataset={args.dataset}, features={args.features})")
    print("=" * 70)
    dataset = EnhancersDatasetClassicalSimple(
        track_matrix_path=args.track_matrix,
        labels_file_path=args.labels,
        split_file=args.split_file,
        filter_by=args.filter_by,
    )
    data = get_split_data(dataset, use_split_file=args.split_file is not None)

    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, test_regions = data["test"]

    # final models are trained on train+val (val is only used for picking params)
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    #  2. Train / load + evaluate each model
    metrics_path = os.path.join(args.output_dir, "metrics_summary.json")
    all_metrics = {}
    if args.eval_only and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)  # keep train times from the original run

    probs, preds = {}, {}
    for name in args.models:
        print("\n" + "=" * 70)
        print(f"  {name.upper()}")
        print("=" * 70)
        model_path = os.path.join(args.output_dir, MODEL_FILE[name])

        if args.eval_only:
            saved = joblib.load(model_path)
            model, params = saved["model"], saved.get("params")
            train_time = all_metrics.get(METRIC_KEY[name], {}).get("train_time_s")
        else:
            params = grid_search(name, X_train, y_train, scale_pos_weight, args) if args.grid else BEST_PARAMS[name]
            model = make_model(name, params, scale_pos_weight, args.seed)

            print(f"  Training on train+val: {X_full.shape}, params={params}")
            t0 = time.time()
            model.fit(X_full, y_full)
            train_time = time.time() - t0
            print(f"  {name} training time: {train_time:.1f}s")

            joblib.dump({"model": model, "params": params}, model_path)
            print(f"  Saved {model_path}")

        t0 = time.time()
        metrics, preds[name], probs[name] = evaluate(model, X_test, y_test, "test", model_name=name)
        metrics["inference_time_s"] = time.time() - t0
        metrics["train_time_s"] = train_time
        all_metrics[METRIC_KEY[name]] = metrics
        all_metrics[PARAM_KEY[name]] = params

    #  3. Save predictions + plots
    pred_df = pd.DataFrame({"region": test_regions, "label": y_test})
    for name in args.models:
        pred_df[f"prob_{name}"] = probs[name]
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved {pred_path}")

    plot_roc_pr(y_test, probs, args.output_dir, "test")
    plot_confusion_matrices(y_test, preds, args.output_dir, "test")

    #  4. Summary
    print("\n" + "=" * 70)
    print("  SUMMARY (test)")
    print("=" * 70)
    for metric_name in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "auroc", "auprc"]:
        print(f"\n{metric_name.upper()}:")
        for name in args.models:
            print(f"  {name:12s}: {all_metrics[METRIC_KEY[name]][metric_name]:.4f}")

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4, default=float)
    print(f"\nSaved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
