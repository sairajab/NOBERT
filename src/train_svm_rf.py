#!/usr/bin/env python3
"""
train_svm_rf.py
===============
"""


from asyncio import run
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
 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

import joblib
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter


class EnhancersDatasetFast(Dataset):
 
    def __init__(self, track_matrix_path, labels_file_path, n_windows=3, stride=200, window_size=200, pick_closest_bin=False, pretrain_datapath=None,
                 split_file=None):
 
        self.split_file = split_file
        self.window_size = window_size
        self.stride = stride
        self.num_labels = 2
        self.n_windows = n_windows
 
        self.labels_data = pd.read_csv(
            labels_file_path, sep='\t',
            names=['Chromosome', 'Start', 'End', 'Split', 'FeatureType', 'Label']
        )
 
        print("Labels loaded:")
        print(self.labels_data.head())
        print(f"Shape: {self.labels_data.shape}")
        print(f"NaN labels: {self.labels_data['Label'].isna().sum()}")
        non_finite = self.labels_data[~np.isfinite(self.labels_data['Label'])]
        if len(non_finite) > 0:
            print(f"Non-finite labels: {len(non_finite)}")
            print(non_finite.head())
        self.labels_data['Label'] = self.labels_data['Label'].astype(int)
 
        label_counts = self.labels_data['Label'].value_counts().sort_index()
        print(f"Label distribution: {dict(label_counts)}")
 
        # ── Load track matrix ──
        # CSV: 'window' col as index, feature cols in middle, 'actual' as last col
        print(f"\nLoading track matrix from {track_matrix_path}...")
        t0 = time.time()
 
        if track_matrix_path.endswith('.parquet'):
            track_df = pd.read_parquet(track_matrix_path)
        else:
            track_df = pd.read_csv(track_matrix_path, index_col=0)
 
        print(f"  Loaded in {time.time()-t0:.1f}s: {track_df.shape}")
        print(f"  First 5 columns: {track_df.columns[:5].tolist()}")
        print(f"  Last 3 columns:  {track_df.columns[-3:].tolist()}")
 
        # ── Separate 'actual' column from features ──
        if 'actual' not in track_df.columns:
            raise ValueError(
                "Track matrix must have an 'actual' column. "
                "Found columns: " + str(track_df.columns[:5].tolist())
            )
 
        actual_col = track_df['actual'].values
        window_index = track_df.index.values
        feature_cols = [c for c in track_df.columns if c != 'actual']
        self.n_features = len(feature_cols)
        self.feature_names = feature_cols
        self.feature_values = track_df[feature_cols].values.astype(np.float32)
 
        print(f"  Features: {self.n_features}")
        print(f"  Sample: window={window_index[0]} -> actual={actual_col[0]}")
 
        # ── Build lookup: actual_key -> sorted list of (window_start, row_idx) ──
        print("Building lookup...")
        t0 = time.time()
        raw_lookup = defaultdict(list)
 
        for row_idx, (win_key, act_key) in enumerate(zip(window_index, actual_col)):
            win_str = str(win_key)
            try:
                coords = win_str.split(':')[1]
                win_start = int(coords.split('-')[0])
            except (IndexError, ValueError):
                win_start = row_idx
 
            raw_lookup[str(act_key)].append((win_start, row_idx))
 
        self.actual_to_rows = {}
        for act_key, window_list in raw_lookup.items():
            window_list.sort(key=lambda x: x[0])
            self.actual_to_rows[act_key] = [row_idx for _, row_idx in window_list]
 
        window_counts = Counter(len(v) for v in self.actual_to_rows.values())
        print(f"  Built in {time.time()-t0:.2f}s")
        print(f"  Unique regions: {len(self.actual_to_rows):,}")
        print(f"  Windows per region: {dict(sorted(window_counts.items()))}")
 
        # ── Precompute sample -> row indices ──
        print("Precomputing sample indices...")
        self.sample_row_indices = []
        matched = 0
        missing = 0
 
        for i in range(len(self.labels_data)):
            row = self.labels_data.iloc[i]
            chrom = str(row['Chromosome'])
            start = int(row['Start'])
            end = int(row['End'])
            actual_key = f"{chrom}:{start}-{end}"
 
            row_indices = self.actual_to_rows.get(actual_key, [])
            self.sample_row_indices.append(row_indices)
 
            if row_indices:
                matched += 1
            else:
                missing += 1
                if missing <= 5:
                    print(f"  No match: {actual_key}")
 
        print(f"  Matched: {matched:,}/{len(self.labels_data):,} "
              f"({100*matched/len(self.labels_data):.1f}%)")
        if missing > 0:
            print(f"  Missing: {missing:,} ({100*missing/len(self.labels_data):.1f}%)")
 
        self._zero = np.zeros(self.n_features, dtype=np.float32)
 
    def __len__(self):
        return len(self.labels_data)
 
    def create_splits_from_labels_data(self):
        """Use the 'Split' column (Train/Test/Val) to create splits."""
        splits = {"train": [], "val": [], "test": []}
        for idx in range(len(self.labels_data)):
            row = self.labels_data.iloc[idx]
            split = str(row['Split']).strip().lower()
            if split in ["val", "validation"]:
                split = "val"
            if split in splits:
                splits[split].append(idx)
            else:
                splits["train"].append(idx)
 
        for k, v in splits.items():
            print(f"  {k}: {len(v)} samples")
        return splits
 
    def create_splits_from_data(self, split_file=None):
        if split_file is None:
            split_file = self.split_file
        split_df = pd.read_csv(split_file)
        seq_to_split = {}
        for _, row in split_df.iterrows():
            seq_to_split[(row['chrom'], row['start'], row['end'])] = row['split']
 
        splits = {"train": [], "val": [], "test": []}
        missed = 0
        for idx in range(len(self.labels_data)):
            data = self.labels_data.iloc[idx]
            seq_key = (data['Chromosome'], data['Start'], data['End'])
            split = seq_to_split.get(seq_key, "train")
            if split in splits:
                splits[split].append(idx)
            else:
                missed += 1
        if missed:
            print(f"Warning: {missed} sequences not in split file, assigned to train")
        return splits
 
    def get_labels(self, idx):
        return self.labels_data.iloc[idx]['Label']
 
    # ─────────────────────────────────────────────────────────────────────
    #  sklearn-compatible methods
    # ─────────────────────────────────────────────────────────────────────
 
    def _aggregate_windows(self, row_indices, method="mean"):
        if not row_indices:
            base = self._zero
            if method == "concat":
                return np.tile(base, self.n_windows)
            elif method == "stats":
                return np.concatenate([base, base, base, base])
            return base.copy()
 
        mat = self.feature_values[row_indices]
 
        if method == "mean":
            return mat.mean(axis=0)
        elif method == "max":
            return mat.max(axis=0)
        elif method == "middle":
            return mat[len(mat) // 2]
        elif method == "concat":
            if len(mat) >= self.n_windows:
                mid = len(mat) // 2
                half = self.n_windows // 2
                start = mid - half
                mat = mat[start:start + self.n_windows]
            else:
                pad = np.zeros((self.n_windows - len(mat), self.n_features), dtype=np.float32)
                mat = np.vstack([mat, pad])
            return mat.ravel()
        elif method == "stats":
            return np.concatenate([
                mat.mean(axis=0), mat.std(axis=0),
                mat.min(axis=0), mat.max(axis=0),
            ])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
 
    def get_sklearn_data(self, indices=None, aggregation="mean"):
        if indices is None:
            indices = list(range(len(self.labels_data)))
        X_list, y_list = [], []
        for idx in indices:
            feat = self._aggregate_windows(self.sample_row_indices[idx], method=aggregation)
            X_list.append(feat)
            y_list.append(int(self.labels_data.iloc[idx]['Label']))
        return np.stack(X_list), np.array(y_list, dtype=np.int64)
 
    def get_split_sklearn_data(self, aggregation="mean", use_split_file=False):
        if use_split_file and self.split_file is not None:
            splits = self.create_splits_from_data()
        else:
            splits = self.create_splits_from_labels_data()
 
        result = {}
        for split_name, idxs in splits.items():
            X, y = self.get_sklearn_data(indices=idxs, aggregation=aggregation)
            result[split_name] = (X, y)
            print(f"  {split_name}: X{X.shape}, y{y.shape}, "
                  f"pos={y.sum()}, neg={len(y)-y.sum()}")
        return result
 
    # ─────────────────────────────────────────────────────────────────────
    #  __getitem__ for DataLoader (neural nets)
    # ─────────────────────────────────────────────────────────────────────
 
    def __getitem__(self, idx):
        row = self.labels_data.iloc[idx]
        chrom = str(row['Chromosome'])
        start = int(row['Start'])
        end = int(row['End'])
        label = int(row['Label'])
        name = f"{chrom}:{start}-{end}"
 
        row_indices = self.sample_row_indices[idx]
 
        if row_indices:
            windows = [self.feature_values[ri] for ri in row_indices]
            if len(windows) > self.n_windows:
                pick_mid = len(windows) // 2
                windows = [windows[pick_mid]]
        else:
            windows = [self._zero]
 
        data = torch.from_numpy(np.stack(windows))
        label = torch.tensor(label, dtype=torch.long)
        return name, data, label
 
 
 
# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
 
def evaluate(model, X, y, split_name="test"):
    """Compute full classification metrics."""
    y_pred = model.predict(X)
 
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = None
 
    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
    }
 
    if y_prob is not None and len(np.unique(y)) == 2:
        metrics["auroc"] = roc_auc_score(y, y_prob)
        metrics["auprc"] = average_precision_score(y, y_prob)
 
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
    print(f"\n{classification_report(y, y_pred, target_names=['Negative', 'Positive'], zero_division=0)}")
 
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:\n{cm}\n")
 
    return metrics, y_pred, y_prob
 
 
def plot_roc_pr(y_true, y_prob_dict, output_dir, split_name="test"):
    """ROC and PR curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    ax = axes[0]
    for name, y_prob in y_prob_dict.items():
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name}")
    ax.legend()
    ax.grid(alpha=0.3)
 
    ax = axes[1]
    for name, y_prob in y_prob_dict.items():
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
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
 
 
def plot_feature_importance(importances, feature_names, output_dir, model_name="RF", top_k=30):
    """Bar chart of top-k feature importances."""
    idx = np.argsort(importances)[::-1][:top_k]
 
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.25)))
    ax.barh(range(len(idx)), importances[idx][::-1])
    if feature_names is not None:
        labels = [feature_names[i] for i in idx][::-1]
    else:
        labels = [f"feat_{i}" for i in idx][::-1]
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {top_k} Feature Importances")
    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")
 
 
# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(description="Train SVM & RF on enhancers data")
    parser.add_argument("--track_matrix", type=str, required=True,
                        help="Path to track matrix (.csv or .parquet)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to labels file (.txt, tab-separated, no header)")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional split CSV file. Default: use Split column in labels.")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "max", "middle", "concat", "stats"],
                        help="How to aggregate windows per region (default: mean)")
    parser.add_argument("--output_dir", type=str, default="./results_svm_rf",
                        help="Directory to save models, plots, metrics")
    parser.add_argument("--n_windows", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Parallel jobs for GridSearch & RF (-1 = all cores)")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of CV folds for hyperparameter search")
    parser.add_argument("--skip_svm_grid", action="store_true",
                        help="Skip SVM grid search and use defaults")
    parser.add_argument("--skip_gb_grid", action="store_true",
                        help="Skip GB grid search and use defaults")
    args = parser.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    #  1. Load dataset 
    print("=" * 70)
    print("  LO" \
    "ADING DATASET")
    print("=" * 70)
 
    dataset = EnhancersDatasetFast(
        track_matrix_path=args.track_matrix,
        labels_file_path=args.labels,
        n_windows=args.n_windows,
        split_file=args.split_file,
    )
 
    #  2. Extract sklearn arrays 
    print("\n" + "=" * 70)
    print(f"  EXTRACTING FEATURES (aggregation={args.aggregation})")
    print("=" * 70)
 
    # Use Split column from labels (Train/Test) by default
    use_split_file = args.split_file is not None
    data = dataset.get_split_sklearn_data(
        aggregation=args.aggregation,
        use_split_file=use_split_file,
    )
 
    X_train, y_train = data["train"]
    X_val, y_val = data.get("val", (np.array([]).reshape(0, X_train.shape[1]),
                                     np.array([], dtype=np.int64)))
    X_test, y_test = data["test"]
     
    print(f"\nFinal splits:")
    print(f"  Train: {X_train.shape}, pos={y_train.sum()}, neg={len(y_train)-y_train.sum()}")
    print(f"  Val:   {X_val.shape},   pos={y_val.sum()},   neg={len(y_val)-y_val.sum()}")
    print(f"  Test:  {X_test.shape},  pos={y_test.sum()},  neg={len(y_test)-y_test.sum()}")
 


    
    feature_names = dataset.feature_names if hasattr(dataset, "feature_names") else None
    if args.aggregation in ("concat", "stats") and feature_names is not None:
        feature_names = None 


    print("\nScaling features (StandardScaler fit on train)...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
 
    all_metrics = {}
 

    print("\n" + "=" * 70)
    print("  TRAINING SVM")
    print("=" * 70)
 
    cv = StratifiedKFold(n_splits=min(args.cv_folds, min(np.bincount(y_train))),
                         shuffle=True, random_state=42)
    
    svm_param_grid = {
            "kernel": ["rbf", "linear"],
            "C": [0.01, 0.1, 1, 10],
            "gamma": ["scale", "auto"],
        }
    print(f"  Grid search: {svm_param_grid}")
    t0 = time.time()
 
    svm_grid = GridSearchCV(
            SVC(probability=True, class_weight="balanced", random_state=42),
            svm_param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=args.n_jobs,
            verbose=1,
            refit=True,
        )
    print("SKIPPING SVM GRID SEARCH (using default hyperparameters)...")

    if not args.skip_svm_grid:
        svm_grid.fit(X_train, y_train)
    
        print(f"  Grid search done in {time.time()-t0:.1f}s")
        print(f"  Best params: {svm_grid.best_params_}")
        print(f"  Best CV F1:  {svm_grid.best_score_:.4f}")
    
        svm_model = svm_grid.best_estimator_
        svm_best_params = svm_grid.best_params_

        svm_val_metrics, svm_val_pred, svm_val_prob = evaluate(svm_model, X_val, y_val, "val")

    
        svm_path = os.path.join(args.output_dir, "svm_model.joblib")
        joblib.dump({"model": svm_model, "scaler": scaler, "params": svm_best_params}, svm_path)
        print(f"  Saved SVM model to {svm_path}")

    else:
        print("SKIPPING SVM GRID SEARCH (using default hyperparameters)...")
    #     svm_model = SVC(probability=True, class_weight="balanced", random_state=42)
    #     full_train_data = np.vstack([X_train_sc, X_val_sc])
    #     full_train_labels = np.hstack([y_train, y_val])
    #     svm_model.fit(full_train_data, full_train_labels)
    #     svm_best_params = {"kernel": "rbf", "C": 1, "gamma": "scale"}


    svm_test_metrics, svm_test_pred, svm_test_prob = evaluate(svm_model, X_test, y_test, "test")
    all_metrics["SVM_test"] = svm_test_metrics
    all_metrics["SVM_params"] = svm_best_params


    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    if not args.skip_gb_grid:
        print("\n" + "=" * 70)
        print("  TRAINING XGBOOST FOREST")
        print("=" * 70)
    


        xgb_param_grid = {
            "max_depth":        [4, 6, 8],
            "learning_rate":    [0.05, 0.1, 0.2],
            "min_child_weight": [1, 5, 10],
            
            #  multiplies grid size:
            # "subsample":        [0.8, 1.0],
            # "colsample_bytree": [0.8, 1.0],
        }
        print(f"  Grid search: {xgb_param_grid}")
        t0 = time.time()

        xgb_estimator = XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            eval_metric='aucpr',            
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,                        
            random_state=42,
            verbosity=0,
            early_stopping_rounds=None,           
            n_estimators=2000,                  

        )

        xgb_grid = GridSearchCV(
            xgb_estimator,
            xgb_param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=args.n_jobs,
            verbose=1,
            refit=True,
        )
        xgb_grid.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    
        print(f"  Grid search done in {time.time()-t0:.1f}s")
        print(f"  Best params: {xgb_grid.best_params_}")
        print(f"  Best CV F1:  {xgb_grid.best_score_:.4f}")
    
        #Best params: {'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 5}                                                                                                                 

        xgb_model = xgb_grid.best_estimator_
        xgb_best_params = xgb_grid.best_params_
    
    else:
        print("SKIPPING XGBOOST GRID SEARCH (using best params from earlier run)...")
        best ={'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 5, 'n_estimators': 979, 'colsample_bytree': 1.0, 'subsample': 0.8} 
        xgb_model = XGBClassifier(
            objective="binary:logistic",
            tree_method="exact",
            eval_metric="aucpr",
            scale_pos_weight=neg / pos,
            reg_lambda=1,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            early_stopping_rounds=None,     # ON now, so it actually picks the tree count
            **best,
        )
        total_train_data = np.vstack([X_train, X_val])
        total_train_labels = np.hstack([y_train, y_val])
        xgb_model.fit(
            total_train_data, total_train_labels,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        
    print("\n" + "=" * 70)
    #print(xgb_model.best_iteration)

    # Evaluate RF
    #xgb_val_metrics, xgb_val_pred, xgb_val_prob = evaluate(xgb_model, X_val, y_val, "val")
    xgb_test_metrics, xgb_test_pred, xgb_test_prob = evaluate(xgb_model, X_test, y_test, "test")
    #all_metrics["XGB_val"] = xgb_val_metrics
    all_metrics["XGB_test"] = xgb_test_metrics
    all_metrics["XGB_params"] = xgb_best_params

    print(all_metrics)
 
    xgb_path = os.path.join(args.output_dir, "xgb_model_l2.joblib")
    joblib.dump({"model": xgb_model, "scaler": scaler, "params": xgb_best_params}, xgb_path)
    print(f"  Saved XGB model to {xgb_path}")
 
    # Feature importance plot
    plot_feature_importance(
        xgb_model.feature_importances_, feature_names,
        args.output_dir, model_name="XGB_l2", top_k=min(30, X_train_sc.shape[1]),
    )
 
    if svm_best_params.get("kernel") == "linear":
        plot_feature_importance(
            np.abs(svm_model.coef_[0]), feature_names,
            args.output_dir, model_name="SVM_linear", top_k=min(30, X_train_sc.shape[1]),
        )
 
    # ═════════════════════════════════════════════════════════════════
    #  6. Comparison plots
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  GENERATING PLOTS")
    print("=" * 70)
 
    # Only plot if we have both classes in the split
    if len(np.unique(y_val)) == 2:
        plot_roc_pr(y_val, {"SVM": svm_val_prob, "XGB": xgb_val_prob}, args.output_dir, "val")
        plot_confusion_matrices(y_val, {"SVM": svm_val_pred, "XGB": xgb_val_pred}, args.output_dir, "val")
 
    if len(np.unique(y_test)) == 2:
        plot_roc_pr(y_test, {"SVM": svm_test_prob, "XGB": xgb_test_prob}, args.output_dir, "test")
        plot_confusion_matrices(y_test, {"SVM": svm_test_pred, "XGB": xgb_test_pred}, args.output_dir, "test")
 
    # ═════════════════════════════════════════════════════════════════
    #  7. Summary
    # ════════════════════════════════════
    print("  SUMMARY")
    print("=" * 70)
 
    summary_rows = []
    for key in ["SVM_val", "SVM_test", "RF_val", "RF_test"]:
        m = all_metrics[key]
        summary_rows.append({
            "Model": key.split("_")[0],
            "Split": m["split"],
            "N": m["n_samples"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Bal.Acc": f"{m['balanced_accuracy']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "AUROC": f"{m.get('auroc', 'N/A'):.4f}" if isinstance(m.get('auroc'), float) else "N/A",
            "AUPRC": f"{m.get('auprc', 'N/A'):.4f}" if isinstance(m.get('auprc'), float) else "N/A",
        })
 
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
 
    # Save metrics
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
 
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=convert)
 
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
 
    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
 
    print(f"\n  All outputs saved to: {args.output_dir}/")
    print(f"    - svm_model.joblib, rf_model.joblib")
    print(f"    - scaler.joblib")
    print(f"    - metrics.json, summary.csv")
    print(f"    - roc_pr_*.png, confusion_*.png, feature_importance_*.png")
    print("  Done!")
 
 
if __name__ == "__main__":
    main()