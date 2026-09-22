import xgboost as xgb
from train_svm_rf import EnhancersDatasetClassical,EnhancersDatasetClassicalSimple
import shap
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from joblib import Parallel, delayed
track_matrix = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/enhances_full_500_5_chrombert_style_final.csv"
labels_file = "/s/chromatin/c/nobackup/Saira/BERTModel/DATA/enhancers/C1_2500_enhancer_dataset.txt"
win = 3
split_file = None
filter_by = "new_hm_mask.npy"#".npy"#dna_binding_mask, "histone_mask.npy"
dataset = EnhancersDatasetClassicalSimple(
        track_matrix_path=track_matrix,
        labels_file_path=labels_file,
        split_file=split_file,
        filter_by=filter_by
    )
 
    # Use Split column from labels (Train/Test) by default
use_split_file = split_file is not None
data = dataset.get_split_sklearn_data(
        use_split_file=use_split_file,
    )

out_dir = "5_embeddings_C1_xgb_HM"#"final_enhancers_histone_c1"

X_train, y_train = data["train"]
X_test, y_test = data["test"]
X_val, y_val = data["val"]

print(X_train[0:5], y_train[0:5])
#svm_model_rbf = joblib.load(f"../results/{out_dir}/svm_model_rbf.joblib")["model"]
svm_model_linear = joblib.load(f"../results/{out_dir}/svm_model_linear.joblib")["model"]
model = svm_model_linear
track_matrix_df = pd.read_csv(track_matrix, index_col=0)
if 'actual' in track_matrix_df.columns:
    track_matrix_df = track_matrix_df.drop(columns=['actual'])
all_columns = track_matrix_df.columns
if filter_by is not None:
    filter_columns =  np.load(filter_by).ravel()
    print("loaded mask for filtering features ", filter_columns.sum(), filter_columns.shape)
    track_matrix_df = track_matrix_df.loc[:, filter_columns.astype(bool)]
print("track_matrix_df shape: ", track_matrix_df.shape)
index_to_col = dict(zip(range(track_matrix_df.shape[1]), track_matrix_df.columns))


feature_names = [index_to_col[i] for i in range(len(index_to_col))]
X_test_df = pd.DataFrame(
    X_test,
    columns=feature_names,
)
# sample random features
rng = np.random.default_rng(42)

idx = rng.choice(X_train.shape[0], size=10, replace=False)
background = X_train[idx]

explainer = shap.KernelExplainer(
    model.predict_proba,
    data = background
)
n_workers = os.cpu_count()
chunk_size = max(1, len(X_test_df) // (n_workers * 4))
row_chunks = [
    X_test_df.iloc[i:i + chunk_size]
    for i in range(0, len(X_test_df), chunk_size)
]
print(f"Computing SHAP values for {len(X_test_df)} rows "
      f"across {len(row_chunks)} chunks on {n_workers} workers")

def _shap_for_chunk(chunk):
    return explainer.shap_values(chunk)

print("Number of workers" , n_workers)
chunk_results = Parallel(n_jobs=n_workers)(
    delayed(_shap_for_chunk)(chunk) for chunk in row_chunks
)
shap_values = np.concatenate(chunk_results, axis=0)

# shap_values = (n_samples, n_features, n_classes) for
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif shap_values.ndim == 3:
    shap_values = shap_values[:, :, 1]

# save shap values to csv
shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
shap_values_df.to_csv(f"../results/{out_dir}/linear_shap_values.csv", index=False)
