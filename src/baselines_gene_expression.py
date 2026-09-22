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
from train_gene_expression import compute_all_metrics, summarize, _json_default
import os
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import optuna
from functools import partial
import json
import os
from dataloaders import GeneExpressionDataset
import os
import joblib

def featurize(dataset, indices):
    X, Y = [], []
    n_feat = dataset.disc_matrix.shape[1]
    indx_to_gene = {}
    for i in indices:
        gene_name , data, label = dataset[i]
        arr = data.numpy().astype(np.float32)
        if arr.shape[0] == 0:
            feats = np.zeros(n_feat * 3, dtype=np.float32)
        else:
            feats = np.max(arr, axis=0)
        X.append(feats)
        Y.append(label.numpy())
        indx_to_gene[i] = gene_name

    return np.asarray(X, dtype=np.int8), np.asarray(Y, dtype=np.float32), indx_to_gene


# ---------- Metrics ----------
def report(name, Y_true, Y_pred, outdir, test_indices, tag="",  gene_names_dict=None):
    # reverse log1p transform to get back to original scale for metrics
    # Y_true = np.expm1(Y_true)
    # Y_pred = np.expm1(Y_pred)
    os.makedirs(outdir, exist_ok=True)
    metrics = compute_all_metrics(Y_true, Y_pred)
    summary = summarize(metrics)

    def _print(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"    {prefix}{k}:")
                _print(v, prefix + "  ")
            else:
                print(f"    {prefix}{k}: {v:.4f}" if isinstance(v, float) else f"    {prefix}{k}: {v}")
    _print(summary)

    
    dump = {
                "per_gene": {
                    f"gene_{gene_names_dict[test_indices[i]]}": {
                        "pearson":  float(metrics["pearson_per_gene"][i]),
                        "spearman": float(metrics["spearman_per_gene"][i]),
                        "r2":       float(metrics["r2_per_gene"][i]),
                        "mae":      float(metrics["mae_per_gene"][i]),
                    }
                    for i in range(Y_true.shape[0])
                },
                "per_experiment": {
                    f"experiment_{j}": {
                        "pearson":          float(metrics["pearson_per_exp"][j]),
                        "pearson_centered": float(metrics["pearson_per_exp_centered"][j]),
                        "spearman":         float(metrics["spearman_per_exp"][j]),
                        "r2":               float(metrics["r2_per_exp"][j]),
                        "mae":              float(metrics["mae_per_exp"][j]),
                    }
                    for j in range(Y_pred.shape[1])
                },
                "summary": summary,
            }
    with open(os.path.join(outdir, f"{name}_{tag}_metrics.json"), "w") as f:
        json.dump(dump, f, indent=2, default=_json_default)
                

    # pcc = np.array([
    #     pearsonr(Y_true[:, k], Y_pred[:, k])[0]
    #     if np.std(Y_pred[:, k]) > 0 and np.std(Y_true[:, k]) > 0 else 0.0
    #     for k in range(Y_true.shape[1])
    # ])
    # rmse = np.sqrt(((Y_true - Y_pred) ** 2).mean())
    # gene_pcc = np.array([
    #     pearsonr(Y_true[i], Y_pred[i])[0]
    #     if np.std(Y_pred[i]) > 0 else 0.0
    #     for i in range(Y_true.shape[0])
    # ])
    
    # pcc = np.array([
    #         pearsonr(Y_true[:, k], Y_pred[:, k])[0]
    #         if np.std(Y_pred[:, k]) > 0 and np.std(Y_true[:, k]) > 0 else 0.0
    #         for k in range(Y_true.shape[1])
    #     ])
    # rmse = np.sqrt(((Y_true - Y_pred) ** 2).mean())
    # gene_pcc = np.array([
    #         pearsonr(Y_true[i], Y_pred[i])[0]
    #         if np.std(Y_pred[i]) > 0 else 0.0
    #         for i in range(Y_true.shape[0])
    #     ])

    # if tag:
    #     np.savetxt(f"{outdir}/{tag}_{name.strip()}_per_target_pcc.txt", pcc, fmt="%.4f")
    #     np.savetxt(f"{outdir}/{tag}_{name.strip()}_per_gene_pcc.txt", gene_pcc, fmt="%.4f")
    #     np.savetxt(f"{outdir}/{tag}_{name.strip()}_rmse.txt", [rmse], fmt="%.4f")
    # print(f"[{tag}] {name}: RMSE={rmse:.4f}  "
    #       f"mean per-target PCC={np.nanmean(pcc):.4f}  "
    #       f"mean per-gene PCC={np.nanmean(gene_pcc):.4f}")



class MLPMultiRegressor(nn.Module):

    def __init__(
        self,
        input_dim,
        num_labels,
        hidden_dims,
        dropout=0.1
    ):
        super().__init__()

        self.input_tokenizer = nn.Embedding(6, input_dim)
        self.exp_embedding = nn.Embedding(2689, input_dim) # 947 histone

        mlp_input_dim = input_dim * 2689

        layers = []

        current_dim = mlp_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_labels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        token_emb = self.input_tokenizer(x)

        exp_ids = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        exp_emb = self.exp_embedding(exp_ids)

        out = token_emb + exp_emb
        out = out.view(out.size(0), -1)  # flatten

        out = self.model(out)

        return out
        

# ---------- Model zoo ----------
def build_models():
    models = {}

    # 1. Random Forest — native multi-output
    models["rf"] = RandomForestRegressor(
        n_estimators=1000, max_depth=40, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, verbose=0,
    )

    # 2. MLP — native multi-output, scale inputs first
    models["mlp"] = MLPMultiRegressor(input_dim=128, num_labels=Y_train.shape[1], hidden_dims=[1024, 512, 256, 128], dropout=0.06)

    #3. XGBoost
    models["xgb"] = params = {
            'objective':      'reg:squarederror',
            'tree_method':    'hist',
            # no device: cuda
            'multi_strategy': 'one_output_per_tree' , #'multi_output_tree',
            'max_depth':      8,
            'eta':            0.05,
            'nthread':        -1,       # all CPU cores
            'subsample':      0.8,
            'colsample_bytree': 0.8,
            'lambda': 0.1,           # L2 regularization on leaf weights, default 1.0
            'min_child_weight': 20,    # minimum sum of instance weight (hessian) needed in a child, default 1
        }
    
    return models




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


# model = MultiOutputRegressor(
# HistGradientBoostingRegressor(
#     max_iter=500,
#     max_depth=6,
#     learning_rate=0.05,      # slower, often better
#     l2_regularization=0.1,
#     subsample=0.8,            # adds stochasticity like XGBoost
#     max_features=0.8,         # equivalent to colsample_bytree
# )
# )
# model.fit(X_train, Y_train)
# report("Val ", Y_val,  model.predict(X_val),  tag="hgb2")
# report("Test", Y_test, model.predict(X_test), tag="hgb2")


def objective(trial, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = trial.suggest_categorical("input_dim",[32, 64, 128, 256])

    num_layers = trial.suggest_int("num_layers",1, 4)

    hidden_dims = []

    for i in range(num_layers):
        hidden_dims.append(trial.suggest_categorical(f"hidden_dim_{i}",[128, 256, 512, 1024])
        )
    
        # Learning rate
    lr = trial.suggest_float("lr",1e-5,1e-3,log=True)

    # Weight decay
    weight_decay = trial.suggest_float("weight_decay",1e-6,1e-2,log=True)

    # Dropout
    dropout = trial.suggest_float("dropout",0.05,0.3 )


    # Number of epochs
    epochs = trial.suggest_int("epochs",30,100)

    model = MLPMultiRegressor(input_dim=input_dim,num_labels=3142,hidden_dims=hidden_dims,dropout=dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)

    criterion = nn.MSELoss()

    for epoch in range(epochs):

        model.train()

        for x, y in train_loader:

            x = x.to(device).long()
            y = y.to(device).float()

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

    model.eval()

    total_val_loss = 0
    n = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device).long()
            y = y.to(device).float()

            pred = model(x)

            loss = criterion(pred, y)

            total_val_loss += loss.item() * x.size(0)
            n += x.size(0)

    val_loss = total_val_loss / n

    return val_loss
    
def select_mlp_hyperparameters(train_loader, val_loader):
    objective_fn = partial(objective,train_loader=train_loader,val_loader=val_loader)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn,n_trials=50)
    print(study.best_params)
    print(study.best_value)


def train_mlp(model, dataset, outdir, epochs=100, batch_size=32, lr=1e-3):
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    splits  = dataset.load_splits(split_path=os.path.join(base_dir, "gene expression/"))

    train_ds = Subset(dataset, splits["train"])
    val_ds   = Subset(dataset, splits["val"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    
    #select_mlp_hyperparameters(train_loader, val_loader)
    #return  # Stop execution after hyperparameter selection for now
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    current_best_val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch in train_loader:
            x, y = batch
            x = x.to(device).long()
            y = y.to(device).float()
            optimizer.zero_grad()

            # Forward pass
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device).long()
                y = y.to(device).float()

                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_ds):.4f}, Val Loss: {val_loss / len(val_ds):.4f}")

        if val_loss < current_best_val_loss:
            current_best_val_loss = val_loss
            torch.save(model.state_dict(), f"{outdir}/best_model.pth")
            print("New best model saved.")
    
        

def evaluate_mlp(model, dataset,outdir , batch_size = 32, device = "cuda", gene_names_dict=None):
    
    splits  = dataset.load_splits(split_path=os.path.join(base_dir, "gene expression/"))
    
    test_ds  = Subset(dataset, splits["test"])

    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


    model.load_state_dict(torch.load(f"{outdir}/best_model.pth"))
    model.eval()
    model.to(device)
    test_loss = 0.0
    Y_pred = []
    Y_true = []
    with torch.no_grad():
        for batch in test_loader:
            names , x, y = batch
            x = x.to(device).long()
            y = y.to(device).float()

            out = model(x)
            Y_pred.append(out.cpu().numpy())
            Y_true.append(y.cpu().numpy())

    print(f"Test Loss: {test_loss / len(test_ds):.4f}")
    
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)
    
    report("MLP", Y_true, Y_pred, os.path.join(outdir ,"results"),splits["test"], tag="test", gene_names_dict= gene_names_dict)
    
    return Y_pred, Y_true

    
    
if __name__ == "__main__":
    
    base_dir ="/s/chromatin/c/nobackup/Saira/BERTModel"
    dataset = GeneExpressionDataset(
        cont_matrix_path=None,
        disc_matrix_path=os.path.join(base_dir, "DATA/gene expression/gene_Exp_500__5emb_disc.csv"), #gene_Exp_500_disc
        labels_file=os.path.join(base_dir, "gene expression/arabidopsis_expression_data.h5"),
        filter_by=None
    )#new_hm_mask.npy"
    splits = dataset.load_splits(split_path=os.path.join(base_dir, "gene expression/"))

    X_train, Y_train, indx_to_gene_train = featurize(dataset, splits["train"])
    X_val,   Y_val , indx_to_gene_val  = featurize(dataset, splits["val"])
    X_test,  Y_test, indx_to_gene_test  = featurize(dataset, splits["test"])
    print("X:", X_train.shape, "Y:", Y_train.shape)


    print("Data loaded. Sample features and labels:")
    print("X[0]:", X_train[0][:10], "...")
    print("Y[0]:", Y_train[0][:10], "...")

    print("X[5]:", X_train[5][:10], "...")
    print("Y[5]:", Y_train[5][:10], "...")

        
    models = build_models()
    results = {}
    
    ## run MLP model
    train_mlp(models["mlp"], dataset, outdir="mlp_output", epochs=50, batch_size=32, lr=1e-4)
    
    # evaluate MLP 
    evaluate_mlp(models["mlp"], dataset, outdir="mlp_output", gene_names_dict = indx_to_gene_test)

    

    # ## run Random Forest model
    rf_outdir = "rf_output_final"
    rf_model = RandomForestRegressor(
       n_estimators=1000, max_depth=40, min_samples_leaf=5,
       max_features="sqrt", n_jobs=4, random_state=42, verbose=0,
    )
    rf_model.fit(X_train, Y_train)
    if not os.path.exists(rf_outdir):
        os.makedirs(rf_outdir)
        
    # save RF model
    joblib.dump(rf_model, f"{rf_outdir}/rf_model.joblib")
    
    report("RF", Y_val,  rf_model.predict(X_val),  outdir=rf_outdir, tag="Val")
    report("RF", Y_test, rf_model.predict(X_test), outdir=rf_outdir, tag="Test")
    
    # ## run XGBoost model


    dtrain = xgb.DMatrix(X_train, label=Y_train)   # Y_train shape: (n_samples, 3000)
    dval  = xgb.DMatrix(X_val,   label=Y_val)
    dtest  = xgb.DMatrix(X_test,  label=Y_test)
    
    xgb_dir = "xgb_5emb"

    print(f"\n=== Training xgboost ===")
    t0 = time.time()

    bst = xgb.train(
        models["xgb"],
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=2,
    )
    print(f"  fit time: {time.time() - t0:.1f}s")
    Y_pred = bst.predict(dtest)

    
    if not os.path.exists(xgb_dir):
        os.mkdir(xgb_dir)
    # Evaluate
    
    bst.save_model(f'{xgb_dir}/xgb_model.ubj') 
    print(f"  Saved XGB model")
    print("Done in ", time.time() - t0, "seconds")


    # load xgb_model
    bst = xgb.Booster()
    bst.load_model(f'{xgb_dir}/xgb_model.ubj')
    Y_pred = bst.predict(dtest)
    report("XGB", Y_test, Y_pred, xgb_dir, splits["test"], tag="5emb_test", gene_names_dict= indx_to_gene_test)
    print("test done")
    

