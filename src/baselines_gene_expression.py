"""
Baselines for gene expression prediction, to compare against ChromBERT.

Input  : discretized chromatin tokens per gene (max over the gene's bins), 2689 features
Target : log1p(FPKM) across 3142 experiments (multi-output regression)

Models
  rf    - Random Forest
  xgb   - XGBoost (multi-output trees)
  mlp   - MLP with token + experiment embeddings
  head  - ChromBERT classification head only (same head, no transformer backbone)

Usage
  python baselines_gene_expression.py --model rf
  python baselines_gene_expression.py --model xgb
  python baselines_gene_expression.py --model mlp  --epochs 50
  python baselines_gene_expression.py --model head --epochs 50

  # reload a saved model and just report metrics
  python baselines_gene_expression.py --model xgb --eval_only

  # optuna search for MLP hyperparameters
  python baselines_gene_expression.py --model mlp --tune

Output (in --outdir, default ../results/gene_expression_baselines/<model>)
  saved model + {MODEL}_Val_metrics.json, {MODEL}_Test_metrics.json
  (per-gene, per-experiment and summary metrics, same format as ChromBERT runs)
"""

import os
import json
import time
import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataloaders import GeneExpressionDataset
from models import ClassificationHeadOnly
from train_gene_expression import compute_all_metrics, summarize, _json_default


BASE_DIR = "/s/chromatin/c/nobackup/Saira/BERTModel"

# default learning rates for the torch models
DEFAULT_LR = {"mlp": 1e-4, "head": 1e-5}

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=40,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
)

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "multi_strategy":   "multi_output_tree",  # one tree predicts all experiments (one_output_per_tree is much slower)
    "max_depth":        8,
    "eta":              0.05,
    "nthread":          8,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "lambda":           0.1,   # L2 on leaf weights, default 1.0
    "min_child_weight": 20,    # min hessian sum in a child, default 1
    "max_bin":          16,    # tokens are small ints, 16 bins is plenty
}


# ---------- Data ----------
def load_data(base_dir, disc_file):
    dataset = GeneExpressionDataset(
        cont_matrix_path=None,
        disc_matrix_path=os.path.join(base_dir, "DATA/gene expression", disc_file),
        labels_file=os.path.join(base_dir, "gene expression/arabidopsis_expression_data.h5"),
        filter_by=None,
    )
    splits = dataset.load_splits(split_path=os.path.join(base_dir, "gene expression/"))
    return dataset, splits


def featurize(dataset, indices):
    # dataset already takes the max over a gene's bins, so each gene is one row of tokens
    X, Y, names = [], [], []
    for i in indices:
        gene_name, data, label = dataset[i]
        X.append(data.numpy().max(axis=0))
        Y.append(label.numpy())
        names.append(gene_name)
    return np.asarray(X, dtype=np.int8), np.asarray(Y, dtype=np.float32), names


# ---------- Metrics ----------
def _print(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"    {prefix}{k}:")
            _print(v, prefix + "  ")
        else:
            print(f"    {prefix}{k}: {v:.4f}" if isinstance(v, float) else f"    {prefix}{k}: {v}")


def report(name, Y_true, Y_pred, gene_names, outdir, tag=""):
    # labels are log1p(FPKM), metrics are on that scale
    os.makedirs(outdir, exist_ok=True)
    metrics = compute_all_metrics(Y_true, Y_pred)
    summary = summarize(metrics)

    print(f"\n[{name} | {tag}]")
    _print(summary)

    dump = {
        "per_gene": {
            f"gene_{gene_names[i]}": {
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
            for j in range(Y_true.shape[1])
        },
        "summary": summary,
    }
    path = os.path.join(outdir, f"{name}_{tag}_metrics.json")
    with open(path, "w") as f:
        json.dump(dump, f, indent=2, default=_json_default)
    print(f"  saved {path}")


# ---------- MLP model ----------
class MLPMultiRegressor(nn.Module):
    # token embedding + experiment embedding, flattened into an MLP

    def __init__(self, input_dim, num_labels, hidden_dims, n_feats=2689, vocab_size=6, dropout=0.1):
        super().__init__()

        self.input_tokenizer = nn.Embedding(vocab_size, input_dim)
        self.exp_embedding = nn.Embedding(n_feats, input_dim)

        layers = []
        current_dim = input_dim * n_feats
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_labels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 1, n_feats) -> (B, n_feats)
        exp_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)

        out = self.input_tokenizer(x) + self.exp_embedding(exp_ids)
        out = out.view(out.size(0), -1)  # flatten
        return self.model(out)


def build_torch_model(name, n_feats, n_labels):
    if name == "mlp":
        return MLPMultiRegressor(input_dim=128, num_labels=n_labels, hidden_dims=[1024, 512, 256, 128],
                                 dropout=0.06, n_feats=n_feats)
    if name == "head":
        return ClassificationHeadOnly(num_labels=n_labels)
    raise ValueError(name)


# ---------- Torch train / predict ----------
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    # one pass over loader, trains if optimizer is given; returns mean loss
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for _, x, y in loader:
            x = x.to(device).long()
            y = y.to(device).float()
            out = model(x)
            loss = loss_fn(out, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


def train_nn(model, train_loader, val_loader, outdir, device, epochs=50, lr=1e-4, weight_decay=1e-4):
    os.makedirs(outdir, exist_ok=True)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
            print("  new best model saved")


def predict_nn(model, loader, device):
    model.eval()
    model.to(device)
    Y_pred, Y_true, names = [], [], []
    with torch.no_grad():
        for gene_names, x, y in loader:
            out = model(x.to(device).long())
            Y_pred.append(out.cpu().numpy())
            Y_true.append(y.numpy())
            names.extend(gene_names)
    return np.concatenate(Y_true), np.concatenate(Y_pred), names


# ---------- Optuna search for MLP ----------
def objective(trial, train_loader, val_loader, n_feats, n_labels, device):
    input_dim   = trial.suggest_categorical("input_dim", [32, 64, 128, 256])
    num_layers  = trial.suggest_int("num_layers", 1, 4)
    hidden_dims = [trial.suggest_categorical(f"hidden_dim_{i}", [128, 256, 512, 1024]) for i in range(num_layers)]
    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout", 0.05, 0.3)
    epochs       = trial.suggest_int("epochs", 30, 100)

    model = MLPMultiRegressor(input_dim=input_dim, num_labels=n_labels, hidden_dims=hidden_dims,
                              dropout=dropout, n_feats=n_feats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        run_epoch(model, train_loader, loss_fn, device, optimizer)
    return run_epoch(model, val_loader, loss_fn, device)


def select_mlp_hyperparameters(train_loader, val_loader, n_feats, n_labels, device, n_trials=50):
    import optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, train_loader=train_loader, val_loader=val_loader,
                           n_feats=n_feats, n_labels=n_labels, device=device), n_trials=n_trials)
    print("best params:", study.best_params)
    print("best val loss:", study.best_value)
    return study.best_params


# ---------- Runners ----------
def run_trees(args, dataset, splits):
    print("featurizing ...")
    X_train, Y_train, _          = featurize(dataset, splits["train"])
    X_val,   Y_val,   val_names  = featurize(dataset, splits["val"])
    X_test,  Y_test,  test_names = featurize(dataset, splits["test"])
    print("X:", X_train.shape, "Y:", Y_train.shape)

    if args.model == "rf":
        import joblib
        from sklearn.ensemble import RandomForestRegressor

        path = os.path.join(args.outdir, "rf_model.joblib")
        if args.eval_only:
            model = joblib.load(path)
        else:
            model = RandomForestRegressor(**RF_PARAMS, random_state=args.seed)
            t0 = time.time()
            model.fit(X_train, Y_train)
            print(f"  fit time: {time.time() - t0:.1f}s")
            joblib.dump(model, path)
        pred_val, pred_test = model.predict(X_val), model.predict(X_test)

    else:  # xgb
        import xgboost as xgb

        dtrain = xgb.QuantileDMatrix(X_train, label=Y_train, max_bin=XGB_PARAMS["max_bin"])
        dval   = xgb.QuantileDMatrix(X_val,   label=Y_val,   ref=dtrain, max_bin=XGB_PARAMS["max_bin"])
        dtest  = xgb.QuantileDMatrix(X_test,  label=Y_test,  ref=dtrain, max_bin=XGB_PARAMS["max_bin"])

        path = os.path.join(args.outdir, "xgb_model.ubj")
        if args.eval_only:
            bst = xgb.Booster()
            bst.load_model(path)
        else:
            params = {**XGB_PARAMS, "device": "cuda" if torch.cuda.is_available() else "cpu", "seed": args.seed}
            t0 = time.time()
            bst = xgb.train(params, dtrain, num_boost_round=args.xgb_rounds,
                            evals=[(dval, "eval")], early_stopping_rounds=50, verbose_eval=10)
            print(f"  fit time: {time.time() - t0:.1f}s")
            bst.save_model(path)

        # predict with the best round from early stopping
        best = getattr(bst, "best_iteration", None)
        it = (0, best + 1) if best is not None else (0, 0)
        pred_val, pred_test = bst.predict(dval, iteration_range=it), bst.predict(dtest, iteration_range=it)

    name = args.model.upper()
    report(name, Y_val,  pred_val,  val_names,  args.outdir, tag="Val")
    report(name, Y_test, pred_test, test_names, args.outdir, tag="Test")


def run_nn(args, dataset, splits):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = dataset.disc_matrix.shape[1]
    n_labels = dataset.fpkm_vals.shape[1]

    train_loader = DataLoader(Subset(dataset, splits["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, splits["val"]),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, splits["test"]),  batch_size=args.batch_size, shuffle=False)

    if args.tune:
        select_mlp_hyperparameters(train_loader, val_loader, n_feats, n_labels, device)
        return

    model = build_torch_model(args.model, n_feats, n_labels)
    if not args.eval_only:
        lr = args.lr or DEFAULT_LR[args.model]
        train_nn(model, train_loader, val_loader, args.outdir, device, epochs=args.epochs, lr=lr)

    model.load_state_dict(torch.load(os.path.join(args.outdir, "best_model.pth"), map_location=device))

    name = args.model.upper()
    for tag, loader in [("Val", val_loader), ("Test", test_loader)]:
        Y_true, Y_pred, names = predict_nn(model, loader, device)
        report(name, Y_true, Y_pred, names, args.outdir, tag=tag)


def parse_args():
    p = argparse.ArgumentParser(description="Gene expression baselines (RF / XGB / MLP / head-only)")
    p.add_argument("--model", required=True, choices=["rf", "xgb", "mlp", "head"])
    p.add_argument("--outdir", default=None, help="default: ../results/gene_expression_baselines/<model>")
    p.add_argument("--base_dir", default=BASE_DIR)
    p.add_argument("--disc_file", default="gene_Exp_500__5emb_disc.csv",
                   help="token matrix in DATA/gene expression/ (e.g. gene_Exp_500_disc.csv)")
    p.add_argument("--eval_only", action="store_true", help="load saved model from outdir and report")
    p.add_argument("--seed", type=int, default=42)
    # trees
    p.add_argument("--xgb_rounds", type=int, default=300)
    # torch
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-4 for mlp, 1e-5 for head")
    p.add_argument("--tune", action="store_true", help="optuna search for MLP (only with --model mlp)")
    args = p.parse_args()
    if args.tune and args.model != "mlp":
        p.error("--tune only works with --model mlp")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.outdir is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        args.outdir = os.path.join(src_dir, "..", "results", "gene_expression_baselines", args.model)
    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset, splits = load_data(args.base_dir, args.disc_file)

    if args.model in ("rf", "xgb"):
        run_trees(args, dataset, splits)
    else:
        run_nn(args, dataset, splits)
