import numpy as np
import time
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from train_gene_expression import GeneExpressionDataset
import os
import xgboost as xgb

def featurize(dataset, indices):
    X, Y = [], []
    n_feat = dataset.disc_matrix.shape[1]
    for i in indices:
        data, label = dataset[i]
        arr = data.numpy().astype(np.float32)
        if arr.shape[0] == 0:
            feats = np.zeros(n_feat * 3, dtype=np.float32)
        else:
            feats = np.max(arr, axis=0)
        X.append(feats)
        Y.append(label.numpy())

    return np.asarray(X, dtype=np.int8), np.log1p(np.asarray(Y, dtype=np.float32))

base_dir ="/s/chromatin/c/nobackup/Saira/BERTModel"
dataset = GeneExpressionDataset(
    cont_matrix_path=None,
    disc_matrix_path=os.path.join(base_dir, "DATA/gene expression/gene_Exp_3w_disc.csv"),
    labels_file=os.path.join(base_dir, "gene expression/arabidopsis_expression_data.h5"),
)
splits = dataset.load_splits(split_path=os.path.join(base_dir, "gene expression/"))

X_train, Y_train = featurize(dataset, splits["train"])
X_val,   Y_val   = featurize(dataset, splits["val"])
X_test,  Y_test  = featurize(dataset, splits["test"])
print("X:", X_train.shape, "Y:", Y_train.shape)

print("Data loaded. Sample features and labels:")
print("X[0]:", X_train[0][:10], "...")
print("Y[0]:", Y_train[0][:10], "...")
# ---------- Metrics ----------
def report(name, Y_true, Y_pred, tag=""):
    # reverse log1p transform to get back to original scale for metrics
    Y_true = np.expm1(Y_true)
    Y_pred = np.expm1(Y_pred)

    pcc = np.array([
        pearsonr(Y_true[:, k], Y_pred[:, k])[0]
        if np.std(Y_pred[:, k]) > 0 and np.std(Y_true[:, k]) > 0 else 0.0
        for k in range(Y_true.shape[1])
    ])
    rmse = np.sqrt(((Y_true - Y_pred) ** 2).mean())
    gene_pcc = np.array([
        pearsonr(Y_true[i], Y_pred[i])[0]
        if np.std(Y_pred[i]) > 0 else 0.0
        for i in range(Y_true.shape[0])
    ])
    if tag:
        np.savetxt(f"{tag}_{name.strip()}_per_target_pcc.txt", pcc, fmt="%.4f")
    print(f"[{tag}] {name}: RMSE={rmse:.4f}  "
          f"mean per-target PCC={np.nanmean(pcc):.4f}  "
          f"mean per-gene PCC={np.nanmean(gene_pcc):.4f}")


# ---------- Model zoo ----------
def build_models():
    models = {}

    # 1. Random Forest — native multi-output
    models["rf"] = RandomForestRegressor(
        n_estimators=1000, max_depth=40, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, verbose=0,
    )


    # 5. MLP — native multi-output, scale inputs first
    models["mlp"] = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )),
    ])

    return models


# ---------- Train / evaluate ----------
# Trim this list to whatever you want to run — gb and ada can be slow on 3k targets.
to_run = [ "hgb", "mlp", "gb", "ada"]

# models = build_models()
# results = {}

# for tag in to_run:
#     model = models[tag]
#     print(f"\n=== Training {tag} ===")
#     t0 = time.time()
#     model.fit(X_train, Y_train)
#     print(f"  fit time: {time.time() - t0:.1f}s")

#     report("Val ", Y_val,  model.predict(X_val),  tag=tag)
#     report("Test", Y_test, model.predict(X_test), tag=tag)
#     results[tag] = model


# # model = MultiOutputRegressor(
# HistGradientBoostingRegressor(
#     max_iter=500,
#     max_depth=6,
#     learning_rate=0.05,      # slower, often better
#     l2_regularization=0.1,
#     subsample=0.8,            # adds stochasticity like XGBoost
#     max_features=0.8,         # equivalent to colsample_bytree
# )

# model.fit(X_train, Y_train)
# report("Val ", Y_val,  model.predict(X_val),  tag="hgb2")
# report("Test", Y_test, model.predict(X_test), tag="hgb2")



dtrain = xgb.DMatrix(X_train, label=Y_train)   # Y_train shape: (n_samples, 3000)
dval  = xgb.DMatrix(X_val,   label=Y_val)
dtest  = xgb.DMatrix(X_test,  label=Y_test)

params = {
    'objective':        'reg:squarederror',
    'multi_strategy':   'multi_output_tree',  # KEY: one tree per output
    'max_depth':        6,
    'eta':              0.1,
    'subsample':        0.8,
    'tree_method': 'hist',   # or 'hist' + 'device': 'cuda' in XGBoost >= 2.0
    'device': 'cuda:0',

}

params = {
    'objective':      'reg:squarederror',
    'tree_method':    'hist',
    # no device: cuda
    'multi_strategy': 'multi_output_tree',
    'max_depth':      8,
    'eta':            0.05,
    'nthread':        -1,       # all CPU cores
    'subsample':      0.8,
    'colsample_bytree': 0.8,
    'lambda': 0.1,           # L2 regularization on leaf weights, default 1.0
}

import os
import joblib

print(f"\n=== Training xgboost ===")
t0 = time.time()

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dval, 'eval')],
    early_stopping_rounds=None,
    verbose_eval=50,
)
print(f"  fit time: {time.time() - t0:.1f}s")
Y_pred = bst.predict(dtest)
# Evaluate
report("Test", Y_test, Y_pred, tag="exp1_max_xgb")
print("Done in ", time.time() - t0, "seconds")
#####################################################
############ Random Forest (sklearn) ############
bst.save_model('xgb_model_max.ubj') 
print(f"  Saved XGB model")


contribs = bst.get_score(importance_type='gain')   # dict: feature -> score
contribs_2 = bst.get_score(importance_type='weight')   # dict: feature -> score

# save feature importances to file
with open("xgb_max_feature_importances.txt", "w") as f:
    for feat, score in contribs.items():
        f.write(f"{feat}\t{score}\n")
with open("xgb_max_feature_importances_weight.txt", "w") as f:
    for feat, score in contribs_2.items():
        f.write(f"{feat}\t{score}\n")

# model = RandomForestRegressor(
#         n_estimators=1000, max_depth=None, min_samples_leaf=5,
#         max_features="sqrt", n_jobs=-1, random_state=42, verbose=0,
#     )

# print(f"\n=== Training rf ===")
# t0 = time.time()
# model.fit(X_train, Y_train)
# print(f"  fit time: {time.time() - t0:.1f}s")
# report("Val ", Y_val,  model.predict(X_val),  tag="rf")
# report("Test", Y_test, model.predict(X_test), tag="rf")


# # ---------- Simple ensemble: average predictions of all fitted models ----------
# if len(results) > 1:
#     val_pred  = np.mean([m.predict(X_val)  for m in results.values()], axis=0)
#     test_pred = np.mean([m.predict(X_test) for m in results.values()], axis=0)
#     report("Val ", Y_val,  val_pred,  tag="ensemble")
#     report("Test", Y_test, test_pred, tag="ensemble")