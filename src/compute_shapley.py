# track_matrix = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/enhances_full_500_5_chrombert_style_final.csv"
# labels_file = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/C1_2500_enhancer_dataset.txt"

import argparse
import xgboost as xgb
from train_svm_rf import EnhancersDatasetClassical, EnhancersDatasetClassicalSimple
import shap
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from joblib import Parallel, delayed


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute SHAP values for SVM and XGBoost enhancer models.")

parser.add_argument("--track-matrix", default="/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/enhances_full_500_5_chrombert_style_final.csv",help="Path to track matrix CSV.")
parser.add_argument("--labels-file",default="/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/C1_2500_enhancer_dataset.txt",help="Path to labels file.")
parser.add_argument("--split-file",default=None, help="Optional split file. Default: use Train/Test/Val information from labels.")
parser.add_argument("--filter-by",default=None,help="Path to feature mask .npy file. Use 'none' to disable filtering.")
parser.add_argument("--out-dir",default="5_embeddings_C1_xgb_HM",help="Output directory name under ../results/.")
parser.add_argument("--models",nargs="+",default=["rbf", "linear", "xgb"],choices=["rbf", "linear", "xgb"],help="Models for which to compute SHAP values.")
parser.add_argument("--n-workers",type=int,default=None,help="Number of parallel workers for Kernel SHAP. Default: CPU count.")
parser.add_argument("--background-size",type=int,default=10,help="Number of training samples used as Kernel SHAP background.")
parser.add_argument("--seed",type=int,default=42,help="Random seed for selecting SHAP background samples.")

args = parser.parse_args()


# Configuration
track_matrix = args.track_matrix
labels_file = args.labels_file
split_file = args.split_file
filter_by = None if args.filter_by.lower() == "none" else args.filter_by
out_dir = args.out_dir
models = args.models

results_dir = f"../results/{out_dir}"
os.makedirs(results_dir, exist_ok=True)

# Dataset
dataset = EnhancersDatasetClassicalSimple(
    track_matrix_path=track_matrix,
    labels_file_path=labels_file,
    split_file=split_file,
    filter_by=filter_by
)

# Use Split column from labels (Train/Test) by default
use_split_file = split_file is not None

data = dataset.get_split_sklearn_data(use_split_file=use_split_file,)

X_train, y_train = data["train"]
X_test, y_test = data["test"]
X_val, y_val = data["val"]

track_matrix_df = pd.read_csv(track_matrix, index_col=0)
if 'actual' in track_matrix_df.columns:
    track_matrix_df = track_matrix_df.drop(columns=['actual'])
all_columns = track_matrix_df.columns

if filter_by is not None:
    
    filter_columns = np.load(filter_by).ravel()
    print("loaded mask for filtering features ",filter_columns.sum(),filter_columns.shape)
    track_matrix_df = track_matrix_df.loc[:, filter_columns.astype(bool)]

print("track_matrix_df shape: ", track_matrix_df.shape)

index_to_col = dict(zip(range(track_matrix_df.shape[1]), track_matrix_df.columns))
feature_names = [index_to_col[i] for i in range(len(index_to_col))]
X_test_df = pd.DataFrame(X_test,columns=feature_names,)

# SHAP helper
def _shap_for_chunk(chunk):
    return explainer.shap_values(chunk)

# Compute SHAP
for model_name in models:

    print(f"\n{'=' * 60}")
    print(f"Computing SHAP values for: {model_name}")
    print(f"{'=' * 60}")

    if model_name == "xgb":

        bst = joblib.load(f"{results_dir}/xgb_model_l2.joblib")["model"]
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test_df)

    else:

        model = joblib.load(f"{results_dir}/svm_model_{model_name}.joblib")["model"]

        # Sample random features / background samples
        rng = np.random.default_rng(args.seed)

        background_size = min(args.background_size,X_train.shape[0])

        idx = rng.choice(X_train.shape[0],size=background_size,replace=False)

        background = X_train[idx]

        explainer = shap.KernelExplainer(model.predict_proba,data=background)

        # Number of workers
        n_workers = args.n_workers or os.cpu_count() or 1

        # Do not use more workers than chunks
        chunk_size = max(1,len(X_test_df) // (n_workers * 4))

        row_chunks = [X_test_df.iloc[i:i + chunk_size]
            for i in range(0, len(X_test_df), chunk_size)]

        n_workers = min(n_workers, len(row_chunks))

        print(
            f"Computing SHAP values for {len(X_test_df)} rows "
            f"across {len(row_chunks)} chunks "
            f"on {n_workers} workers"
        )

        chunk_results = Parallel(n_jobs=n_workers)(
            delayed(_shap_for_chunk)(chunk)
            for chunk in row_chunks
        )

        shap_values = np.concatenate(
            chunk_results,
            axis=0
        )

        # shap_values = (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    # Save SHAP values
    shap_values_df = pd.DataFrame(shap_values,columns=feature_names)
    output_file = (f"{results_dir}/{model_name}_shap_values.csv")
    shap_values_df.to_csv(output_file,index=False)
    print(f"Saved SHAP values to: {output_file}")
    

#Usage  
# python compute_shap.py \
#     --filter-by new_hm_mask.npy \
#     --out-dir 5_embeddings_C1_xgb_HM \
#     --models rbf linear xgb \
#     --n-workers 32 \
#     --background-size 10