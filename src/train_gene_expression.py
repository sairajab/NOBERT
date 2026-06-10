import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from downstream_model import CistromeEmbeddingManager, ResidualBlock, SinusoidalPositionalEncoding, LayerNorm
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import json
import os
import wandb
from train_mlm import GenomicTransformerV2 as ChromBERT  # adjust import to your actual class
from matplotlib import pyplot as plt

"""
Evaluation + plotting for gene expression prediction.

Convention: y_true, y_pred have shape (n_genes, n_experiments).
    axis=1  -> across experiments  (per-gene metric)
    axis=0  -> across genes        (per-experiment metric)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

MAX_SEQ_LEN = 2689
VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
BATCH_SIZE = 256  # Reduced from 32 to fit in 24GB GPU memory
ACCUMULATION_STEPS = 256 // BATCH_SIZE  # = 64, keeps effective batch size at 256
MASK_PROB = 0.15
EPOCHS = 200
LEARNING_RATE = 1e-4
PAD_TOKEN = 0
MASK_TOKEN = 5

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GeneExpressionDataset(Dataset):
    def __init__(self, cont_matrix_path, disc_matrix_path, labels_file):
        super().__init__()

        self.cont_matrix_path = cont_matrix_path
        self.disc_matrix_path = disc_matrix_path
        self.labels_file = labels_file

        with h5py.File(self.labels_file, 'r') as f:
            self.fpkm_vals = f['FPKM_matrix'][:]
            self.gene_chrom = f['chrom_geneid'][:]

        assert len(self.fpkm_vals) == len(self.gene_chrom), \
            f"Label/index length mismatch: {len(self.fpkm_vals)} labels vs {len(self.gene_chrom)} gene_chrom entries"

        self.load_input_matrices(cont=False)

    def load_input_matrices(self, cont=False):
        if cont:
            self.cont_matrix = pd.read_parquet(self.cont_matrix_path)
        else:
            self.cont_matrix = None

        self.disc_matrix = pd.read_csv(self.disc_matrix_path, index_col=0)
        print("Original disc_matrix columns:", self.disc_matrix.columns.tolist())
        self.disc_matrix = self.disc_matrix.drop(columns=['window.1'])
        self.index = self.disc_matrix.index.tolist()

        self.gene_to_start = defaultdict(int)
        self.map_index = defaultdict(list)

        for idx in self.index:
            chrom, other, rest = idx.split(":")
            strand = int(rest.split("|")[-1])
            gene, start = other.split("|")
            if strand == -1:
                start = rest.split("|")[0]
            self.gene_to_start[(gene, chrom)] = start
            self.map_index[(gene, chrom)].append(idx)

        print("Sample gene_to_start keys:", list(self.gene_to_start.items())[:5])

    def load_splits(self, split_path="gene expression/"):
        train_path = split_path + "Train.txt"
        val_path   = split_path + "Val.txt"
        test_path  = split_path + "Test.txt"

        train_ids, val_ids, test_ids = set(), set(), set()

        for path, id_set in [(train_path, train_ids), (val_path, val_ids), (test_path, test_ids)]:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    gene, chrom, start = line.split("_")
                    id_set.add((gene, f"Chr{chrom}", start))

        train_indices, val_indices, test_indices = [], [], []
        missing = 0
        print("split size", len(train_ids), len(val_ids), len(test_ids))

        for j in range(len(self.gene_chrom)):
            arr = self.gene_chrom[j]
            chrom   = f"Chr{arr[0].decode()}"
            gene_id = arr[1].decode()
            start   = self.gene_to_start.get((gene_id, chrom))
            key     = (gene_id, chrom, start)
            matched = False

            if key in train_ids:
                train_indices.append(j); matched = True
            elif key in val_ids:
                val_indices.append(j);   matched = True
            elif key in test_ids:
                test_indices.append(j);  matched = True

            if not matched:
                if missing < 10:
                    print("Sample lookup key:", key)
                missing += 1

        print("Missing", missing)
        return {"train": train_indices, "val": val_indices, "test": test_indices}

    def __len__(self):
        return len(self.gene_chrom)

    def __getitem__(self, idx):
        sample  = self.gene_chrom[idx]
        chrom   = f"Chr{sample[0].decode()}"
        gene_id = sample[1].decode()

        matrix_indices = self.map_index.get((gene_id, chrom), [])

        if len(matrix_indices) == 0:
            print(f"Warning: Gene {gene_id} on {chrom} not found in map_index.")
            n_features = self.disc_matrix.shape[1]
            data = torch.zeros((1, n_features), dtype=torch.long)
        else:
            data = torch.tensor(
                self.disc_matrix.loc[matrix_indices].values,
                dtype=torch.long
            )

        labels = torch.log1p(torch.tensor(self.fpkm_vals[idx], dtype=torch.float32))
        return data, labels



# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MultiWindowHeaderFusion(nn.Module):

    def __init__(self, chrombert_model, num_labels,
                 mtx_mask="/s/chromatin/c/nobackup/Saira/BERTModel/mtx_mask.txt",
                 n_windows=3, dropout=0.1, medium_dim=256):
        super().__init__()
        self.chrombert  = chrombert_model
        self.windows    = n_windows
        H = 768 * n_windows

        if self.windows != 1:
            self.pos_embedding = SinusoidalPositionalEncoding(hidden_dim=768, num_slots=n_windows)

        self.interface = CistromeEmbeddingManager(mtx_mask=mtx_mask)
        n_factors =  self.interface.normalized_mtx_mask.shape[1] 
        self.res1 = ResidualBlock(n_factors, 1024, dropout=dropout)
        self.res2 = ResidualBlock(1024, H,    dropout=dropout)
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, H), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.activation  = nn.ReLU()
        self.res3 = ResidualBlock(H, medium_dim, dropout=dropout)
        self.fc   = nn.Linear(medium_dim, num_labels)

    def _post_transformer(self, x):

        x = self.interface(x)  # (B, L, n_factors)
        x = x.permute(0, 2, 1)
        x = self.res1(x)
        x = self.res2(x)
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x, mask=None, return_emb=False):
        """
        x   : (B, n_windows, MAX_SEQ_LEN)  long tokens
        mask: (B, n_windows)               bool — True = real window
        """
        B, W, L = x.shape
        x = x.view(B * W, L)                          # flatten for BERT
        x = self.chrombert(x)[0]                      # (B*W, L, 768)
        x = x.reshape(B, W, L, -1).permute(0, 2, 1, 3)  # (B, L, W, 768)
        x = x.reshape(B, L, W * 768)                  # (B, L, W*768)

        if self.windows != 1:
            x = self.pos_embedding(x)

        x   = self._post_transformer(x)
        x   = self.res3(x)
        out = self.fc(x)

        if return_emb:
            return out, x
        return out, None

    def freeze_backbone(self):
        for p in self.chrombert.parameters():
            p.requires_grad = False
        for p in self.interface.parameters():
            p.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self, unfreeze_layers=3):
        for p in self.chrombert.encoder_layers[-unfreeze_layers:].parameters():
            p.requires_grad = True
        print(f"Last {unfreeze_layers} encoder layers unfrozen")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

# def evaluate(y_true, y_pred, split_name="", plot = False, outdir="results_chrombert_gene_expression_log1p"):
#     mse_per_label = mean_squared_error(y_true, y_pred, multioutput='raw_values')
#     r2_per_label  = r2_score(y_true, y_pred,           multioutput='raw_values')
#     pearson_per_label = np.array([
#         pearsonr(y_true[:, i], y_pred[:, i])[0]
#         for i in range(y_true.shape[1])
#     ])
    
#     mae_per_label = np.array([np.mean(np.abs(y_true[:, i] - y_pred[:, i])) for i in range(y_true.shape[1])])

#     metrics = {
#         f"{split_name}/mse_mean":     float(mse_per_label.mean()),
#         f"{split_name}/rmse_mean":    float(np.sqrt(mse_per_label).mean()),
#         f"{split_name}/r2_mean":      float(r2_per_label.mean()),
#         f"{split_name}/pearson_mean": float(np.nanmean(pearson_per_label)),
#         f"{split_name}/mae_mean":     float(mae_per_label.mean()),
#     }
#     if plot:
#         # save per experiment pearson correlation to a json file
#         per_experiment_pearson = {
#             f"experiment_{i}": float(pearson_per_label[i]), float(mae_per_label[i])) for i in range(len(pearson_per_label))
#         }
#         with open(os.path.join(outdir, f"{split_name}_per_experiment_pearson.json"), "w") as f:
#             json.dump(per_experiment_pearson, f, indent=2)
        
#         os.makedirs(outdir, exist_ok=True)
#         plot_predictions(y_true, y_pred, outdir)
    
    


#     print(f"\n  [{split_name}]")
#     for k, v in metrics.items():
#         print(f"    {k}: {v:.4f}")
#     return metrics





# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _safe_pearson(a, b):
    """Pearson that returns nan if either vector is constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]
 
 
def _row_pearson(A, B):
    """Row-wise Pearson between two matrices of the same shape."""
    return np.array([_safe_pearson(A[i], B[i]) for i in range(A.shape[0])])
 
 
def _row_spearman(A, B):
    return np.array([
        spearmanr(A[i], B[i]).correlation if np.std(A[i]) > 0 and np.std(B[i]) > 0 else np.nan
        for i in range(A.shape[0])
    ])
 
 
def compute_all_metrics(y_true, y_pred):
    """
    Returns a dict of arrays and scalars covering both axes.
 
    Per-gene:        across experiments (one score per gene)
    Per-experiment:  across genes       (one score per experiment)
    Centered per-experiment: same, but with each gene's mean subtracted first.
        This removes the 'genes have different baselines' signal and
        measures whether the model predicts *deviations* from each gene's
        typical expression -- usually the biologically meaningful metric.
    """
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"
    n_genes, n_experiments = y_true.shape
 
    # Per-gene (across experiments): iterate rows
    pearson_per_gene  = _row_pearson(y_true, y_pred)
    spearman_per_gene = _row_spearman(y_true, y_pred)
    mse_per_gene      = np.mean((y_true - y_pred) ** 2, axis=1)
    mae_per_gene      = np.mean(np.abs(y_true - y_pred), axis=1)
    # r2 per gene: use sklearn with transpose so each gene is a "sample-set"
    r2_per_gene = r2_score(y_true.T, y_pred.T, multioutput='raw_values')
 
    # Per-experiment (across genes): iterate columns -> transpose then row-wise
    pearson_per_exp  = _row_pearson(y_true.T, y_pred.T)
    spearman_per_exp = _row_spearman(y_true.T, y_pred.T)
    mse_per_exp      = np.mean((y_true - y_pred) ** 2, axis=0)
    mae_per_exp      = np.mean(np.abs(y_true - y_pred), axis=0)
    r2_per_exp       = r2_score(y_true, y_pred, multioutput='raw_values')
 
    # Centered per-experiment: subtract each gene's mean across experiments
    gene_means = y_true.mean(axis=1, keepdims=True)
    y_true_c = y_true - gene_means
    y_pred_c = y_pred - gene_means  # subtract the same thing so we measure deviation prediction
    pearson_per_exp_centered = _row_pearson(y_true_c.T, y_pred_c.T)
 
    # Baseline: predict each gene's mean for every experiment
    baseline_pred = np.broadcast_to(gene_means, y_true.shape)
    baseline_pearson_per_exp = _row_pearson(y_true.T, baseline_pred.T)  # will be nan (constant preds)
    # More useful baseline number: MSE of the mean predictor
    baseline_mse = np.mean((y_true - baseline_pred) ** 2)
 
    # Global / flattened
    global_pearson = _safe_pearson(y_true.ravel(), y_pred.ravel())
    global_mse     = float(np.mean((y_true - y_pred) ** 2))
    global_rmse    = float(np.sqrt(global_mse))
 
    # MSE in log1p space -- useful when training on raw FPKM, since raw MSE
    # is dominated by a few highly expressed genes. Clamp negatives at 0
    # before log1p in case predictions go slightly negative.
    yt_log = np.log1p(np.clip(y_true, 0, None))
    yp_log = np.log1p(np.clip(y_pred, 0, None))
    global_mse_log1p  = float(np.mean((yt_log - yp_log) ** 2))
    global_rmse_log1p = float(np.sqrt(global_mse_log1p))
 
    return {
        # arrays
        "pearson_per_gene":         pearson_per_gene,
        "spearman_per_gene":        spearman_per_gene,
        "r2_per_gene":              r2_per_gene,
        "mse_per_gene":             mse_per_gene,
        "mae_per_gene":             mae_per_gene,
        "pearson_per_exp":          pearson_per_exp,
        "spearman_per_exp":         spearman_per_exp,
        "r2_per_exp":               r2_per_exp,
        "mse_per_exp":              mse_per_exp,
        "mae_per_exp":              mae_per_exp,
        "pearson_per_exp_centered": pearson_per_exp_centered,
        # scalars
        "global_pearson":           global_pearson,
        "global_mse":               global_mse,
        "global_rmse":              global_rmse,
        "global_mse_log1p":         global_mse_log1p,
        "global_rmse_log1p":        global_rmse_log1p,
        "baseline_mse":             baseline_mse,
        "skill_over_mean_predictor": 1.0 - global_mse / baseline_mse,
    }
 
 
def summarize(metrics):
    """Reduce array metrics to headline scalars for logging."""
    def s(arr):
        arr = np.asarray(arr)
        return {
            "mean":   float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "frac_gt_0.5": float(np.nanmean(arr > 0.5)),
        }
 
    out = {
        "pearson_per_gene":         s(metrics["pearson_per_gene"]),
        "spearman_per_gene":        s(metrics["spearman_per_gene"]),
        "r2_per_gene":              {"mean": float(np.nanmean(metrics["r2_per_gene"])),
                                     "median": float(np.nanmedian(metrics["r2_per_gene"])),
                                     "frac_positive": float(np.nanmean(metrics["r2_per_gene"] > 0))},
        "pearson_per_exp":          s(metrics["pearson_per_exp"]),
        "spearman_per_exp":         s(metrics["spearman_per_exp"]),
        "pearson_per_exp_centered": s(metrics["pearson_per_exp_centered"]),
        "global_pearson":           metrics["global_pearson"],
        "global_rmse":              metrics["global_rmse"],
        "global_rmse_log1p":        metrics["global_rmse_log1p"],
        "baseline_mse":             metrics["baseline_mse"],
        "skill_over_mean_predictor": metrics["skill_over_mean_predictor"],
    }
    return out
 
 
# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
 
def _json_default(o):
    """Coerce numpy scalars/arrays so json.dump never chokes."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
 
 
def _savefig(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
 
 
def plot_correlation_distributions(metrics, outdir, scale="log1p"):
    """Histogram + ECDF of per-gene and per-experiment correlations.
 
    On raw FPKM, leads with Spearman (rank-based, robust to outliers).
    On log1p, leads with Pearson.
    """
    if scale == "raw":
        keys = [
            ("spearman_per_gene",        "per-gene (Spearman)"),
            ("spearman_per_exp",         "per-experiment (Spearman)"),
            ("pearson_per_gene",         "per-gene (Pearson, may be outlier-driven)"),
        ]
        title_suffix = " — raw FPKM, Spearman recommended"
    else:
        keys = [
            ("pearson_per_gene",         "per-gene"),
            ("pearson_per_exp",          "per-experiment"),
            ("pearson_per_exp_centered", "per-experiment (centered)"),
        ]
        title_suffix = ""
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    ax = axes[0]
    for key, label in keys:
        ax.hist(metrics[key], bins=50, alpha=0.5, label=label)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("correlation")
    ax.set_ylabel("count")
    ax.set_title("Distribution of correlations" + title_suffix)
    ax.legend(fontsize=8)
 
    ax = axes[1]
    for key, label in keys:
        arr = np.sort(metrics[key][~np.isnan(metrics[key])])
        ax.plot(arr, np.linspace(0, 1, len(arr)), label=label)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("correlation")
    ax.set_ylabel("fraction of items ≤ r")
    ax.set_title("ECDF of correlations" + title_suffix)
    ax.legend(fontsize=8)
 
    return _savefig(fig, outdir, "correlation_distributions.png")
 
 
def plot_hexbin_pred_vs_true(y_true, y_pred, outdir, global_pearson=None, scale="log1p"):
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y_true.ravel(), y_pred.ravel(), gridsize=80, bins='log', mincnt=1, cmap='viridis',
                   xscale=('symlog' if scale == "raw" else 'linear'),
                   yscale=('symlog' if scale == "raw" else 'linear'))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='y = x')
    unit = "FPKM" if scale == "raw" else "log1p(FPKM)"
    ax.set_xlabel(f"true {unit}")
    ax.set_ylabel(f"predicted {unit}")
    title = "Predicted vs true (all gene × experiment pairs)"
    if global_pearson is not None:
        title += f"\nglobal Pearson = {global_pearson:.3f}"
    ax.set_title(title)
    ax.legend(loc='upper left')
    fig.colorbar(hb, ax=ax, label='log10(count)')
    return _savefig(fig, outdir, "hexbin_pred_vs_true.png")
 
 
def plot_rank_ordered_performance(metrics, outdir, scale="log1p"):
    if scale == "raw":
        panels = [
            ("spearman_per_gene", "per-gene Spearman (sorted)"),
            ("spearman_per_exp",  "per-experiment Spearman (sorted)"),
        ]
    else:
        panels = [
            ("pearson_per_gene",         "per-gene Pearson (sorted)"),
            ("pearson_per_exp_centered", "per-experiment centered Pearson (sorted)"),
        ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (key, title) in zip(axes, panels):
        arr = np.sort(metrics[key])[::-1]
        ax.plot(arr)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(0.5, color='r', lw=0.5, ls='--', label='r=0.5')
        ax.set_xlabel("rank")
        ax.set_ylabel("correlation")
        ax.set_title(title)
        ax.legend(fontsize=8)
    return _savefig(fig, outdir, "rank_ordered_performance.png")
 
 
def plot_performance_vs_variance(y_true, metrics, outdir, scale="log1p"):
    """Is performance just a function of how variable the target is?"""
    gene_var = y_true.var(axis=1)
    exp_var  = y_true.var(axis=0)
 
    # On raw FPKM, variance scales with mean (heteroscedastic) and spans
    # many orders of magnitude. Use spearman + log x-axis.
    if scale == "raw":
        gene_key, exp_key = "spearman_per_gene", "spearman_per_exp"
        ylab_g, ylab_e = "per-gene Spearman", "per-experiment Spearman"
        xscale = "log"
        # add small epsilon to avoid log(0)
        gene_var = gene_var + 1e-9
        exp_var  = exp_var  + 1e-9
    else:
        gene_key, exp_key = "pearson_per_gene", "pearson_per_exp_centered"
        ylab_g, ylab_e = "per-gene Pearson", "per-experiment centered Pearson"
        xscale = "linear"
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    ax = axes[0]
    ax.scatter(gene_var, metrics[gene_key], s=4, alpha=0.4)
    ax.set_xlabel("gene variance (across experiments)")
    ax.set_ylabel(ylab_g)
    ax.set_xscale(xscale)
    ax.set_title("Per-gene performance vs gene variance")
    ax.axhline(0, color='k', lw=0.5)
 
    ax = axes[1]
    ax.scatter(exp_var, metrics[exp_key], s=4, alpha=0.4)
    ax.set_xlabel("experiment variance (across genes)")
    ax.set_ylabel(ylab_e)
    ax.set_xscale(xscale)
    ax.set_title("Per-experiment performance vs experiment variance")
    ax.axhline(0, color='k', lw=0.5)
 
    return _savefig(fig, outdir, "performance_vs_variance.png")
 
 
def plot_calibration(y_true, y_pred, outdir, n_bins=20, scale="log1p"):
    """Bin by predicted value, plot mean true vs mean predicted per bin."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    bins = np.quantile(yp, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    idx = np.digitize(yp, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred = np.array([yp[idx == i].mean() if (idx == i).any() else np.nan for i in range(n_bins)])
    mean_true = np.array([yt[idx == i].mean() if (idx == i).any() else np.nan for i in range(n_bins)])
 
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, mean_true, 'o-', label='binned')
    lo = np.nanmin([mean_pred.min(), mean_true.min()])
    hi = np.nanmax([mean_pred.max(), mean_true.max()])
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='y=x')
    unit = "FPKM" if scale == "raw" else "log1p(FPKM)"
    ax.set_xlabel(f"mean predicted {unit} (per bin)")
    ax.set_ylabel(f"mean true {unit} (per bin)")
    if scale == "raw":
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
    ax.set_title("Calibration: is the model compressed toward the mean?")
    ax.legend()
    return _savefig(fig, outdir, "calibration.png")
 
 
def plot_stratified_examples(y_true, y_pred, metrics, outdir, scale="log1p"):
    """
    Instead of random genes/experiments, show best / median / worst.
    On raw FPKM, rank by Spearman and use symlog y-axis.
    """
    if scale == "raw":
        gene_key, exp_key = "spearman_per_gene", "spearman_per_exp"
    else:
        gene_key, exp_key = "pearson_per_gene", "pearson_per_exp_centered"
 
    unit = "FPKM" if scale == "raw" else "log1p(FPKM)"
 
    for axis_name, key, slicer in [
        ("gene",       gene_key, lambda i: (y_true[i, :], y_pred[i, :], "experiment index")),
        ("experiment", exp_key,  lambda i: (y_true[:, i], y_pred[:, i], "gene index")),
    ]:
        scores = metrics[key]
        valid = np.where(~np.isnan(scores))[0]
        if len(valid) == 0:
            continue
        sorted_valid = valid[np.argsort(scores[valid])]
        worst   = sorted_valid[0]
        median_ = sorted_valid[len(sorted_valid) // 2]
        best    = sorted_valid[-1]
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, idx, label in zip(axes, [worst, median_, best], ["worst", "median", "best"]):
            yt, yp, xlabel = slicer(idx)
            order = np.argsort(yt)
            ax.plot(yt[order], label='true', lw=1)
            ax.plot(yp[order], label='pred', lw=1, alpha=0.8)
            ax.set_title(f"{label} {axis_name} (idx={idx}, r={scores[idx]:.3f})")
            ax.set_xlabel(f"{xlabel} (sorted by true)")
            ax.set_ylabel(unit)
            if scale == "raw":
                ax.set_yscale("symlog")
            ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, outdir, f"stratified_examples_{axis_name}.png")
 
 
# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
 
def evaluate(y_true, y_pred, split_name="", plot=False,
             outdir="results_chrombert_gene_expression_log1p",
             scale="log1p"):
    """
    y_true, y_pred: (n_genes, n_experiments)
    scale: "log1p" or "raw". Controls plot axes and which correlations are
           emphasized. Use "raw" if y is raw FPKM (heavy-tailed).
    """
    assert scale in ("log1p", "raw"), f"scale must be 'log1p' or 'raw', got {scale}"
    os.makedirs(outdir, exist_ok=True)
    metrics = compute_all_metrics(y_true, y_pred)
    summary = summarize(metrics)
 
    # Pretty print
    print(f"\n  [{split_name}]")
    def _print(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"    {prefix}{k}:")
                _print(v, prefix + "  ")
            else:
                print(f"    {prefix}{k}: {v:.4f}" if isinstance(v, float) else f"    {prefix}{k}: {v}")
    _print(summary)
 
    if plot:
        # Dump per-gene and per-experiment arrays
        dump = {
            "per_gene": {
                f"gene_{i}": {
                    "pearson":  float(metrics["pearson_per_gene"][i]),
                    "spearman": float(metrics["spearman_per_gene"][i]),
                    "r2":       float(metrics["r2_per_gene"][i]),
                    "mae":      float(metrics["mae_per_gene"][i]),
                }
                for i in range(y_true.shape[0])
            },
            "per_experiment": {
                f"experiment_{j}": {
                    "pearson":          float(metrics["pearson_per_exp"][j]),
                    "pearson_centered": float(metrics["pearson_per_exp_centered"][j]),
                    "spearman":         float(metrics["spearman_per_exp"][j]),
                    "r2":               float(metrics["r2_per_exp"][j]),
                    "mae":              float(metrics["mae_per_exp"][j]),
                }
                for j in range(y_true.shape[1])
            },
            "summary": summary,
        }
        with open(os.path.join(outdir, f"{split_name}_metrics.json"), "w") as f:
            json.dump(dump, f, indent=2, default=_json_default)
 
        plot_correlation_distributions(metrics, outdir, scale=scale)
        plot_hexbin_pred_vs_true(y_true, y_pred, outdir,
                                 global_pearson=metrics["global_pearson"], scale=scale)
        plot_rank_ordered_performance(metrics, outdir, scale=scale)
        plot_performance_vs_variance(y_true, metrics, outdir, scale=scale)
        plot_calibration(y_true, y_pred, outdir, scale=scale)
        plot_stratified_examples(y_true, y_pred, metrics, outdir, scale=scale)
 
    return summary
 


def plot_predictions(y_true, y_pred, outdir, num_genes=5, num_experiments=5):
    # Implementation for plotting predictions
        # plot some genes and some experiments 
    no_of_genes = y_true.shape[0]
    no_of_experiments = y_true.shape[1]
    print(f"Plotting predictions for {num_genes} random genes and {num_experiments} random experiments...")
    gene_indices = np.random.choice(no_of_genes, num_genes, replace=False)
    experiment_indices = np.random.choice(no_of_experiments, num_experiments, replace=False)

    for gene_idx in gene_indices:
        plt.figure(figsize=(10, 5))
        plt.plot(y_true[gene_idx, :], label='True', marker='o')
        plt.plot(y_pred[gene_idx, :], label='Predicted', marker='x')
        plt.title(f'Gene {gene_idx} Expression Across Experiments')
        plt.xlabel('Experiment Index')
        plt.ylabel('Log1p(FPKM)')
        plt.legend()
        plt.savefig(os.path.join(outdir, f'gene_{gene_idx}_expression.png'))
        plt.close()
    for exp_idx in experiment_indices:
        plt.figure(figsize=(10, 5))
        plt.plot(y_true[:, exp_idx], label='True', marker='o')
        plt.plot(y_pred[:, exp_idx], label='Predicted', marker='x')
        plt.title(f'Experiment {exp_idx} Expression Across Genes')
        plt.xlabel('Gene Index')
        plt.ylabel('Log1p(FPKM)')
        plt.legend()
        plt.savefig(os.path.join(outdir, f'experiment_{exp_idx}_expression.png'))
        plt.close()

    


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for padded, labels in loader:
            padded = padded.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs, _ = model(padded)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * padded.size(0)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss  = total_loss / len(loader.dataset)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return avg_loss, all_preds, all_labels


def finetune_chrombert(dataset, splits, chrombert_model,
                       num_labels=3140,
                       output_dir="results_chrombert",
                       device="cuda",
                       num_epochs=30,
                       batch_size=16,
                       lr=1e-4,
                       freeze_epochs=2, 
                       mode = "train"):

    os.makedirs(output_dir, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project="gene-expression-chrombert",
        config=dict(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            freeze_epochs=freeze_epochs,
            num_labels=num_labels,
            model="MultiWindowHeaderFusion",
            mode=mode
        )
    )

    model = MultiWindowHeaderFusion(chrombert_model, num_labels=num_labels).to(device)
    loader_kwargs = dict(batch_size=batch_size, num_workers=4,
                            pin_memory=True)
    train_loader = DataLoader(Subset(dataset, splits["train"]), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(Subset(dataset, splits["val"]),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(Subset(dataset, splits["test"]),  shuffle=False, **loader_kwargs)
    
    print("=======Running in mode:", mode)

    model.freeze_backbone() 
    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-5
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    all_metrics   = {}
    if mode == "train":

        # start with frozen backbone

        for epoch in range(num_epochs):

            # Unfreeze backbone after warm-up
            if epoch == freeze_epochs:
                model.unfreeze_backbone(unfreeze_layers=3)
                # Re-init optimizer to include newly unfrozen params
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr * 0.1, weight_decay=1e-5
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs - freeze_epochs
                )
                print("Backbone unfrozen, optimizer re-initialised with lr=", lr * 0.1)

            # ── Train ──────────────────────────────────────────────────────────────
            train_loss, train_preds, train_labels = run_epoch(
                model, train_loader, optimizer, criterion, device, train=True
            )
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────────────
            val_loss, val_preds, val_labels = run_epoch(
                model, val_loader, optimizer, criterion, device, train=False
            )

            print(f"\nEpoch {epoch+1}/{num_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            train_metrics = evaluate(train_labels, train_preds, "train")
            val_metrics   = evaluate(val_labels,   val_preds,   "val")

            log_dict = {
                "epoch":      epoch + 1,
                "train/loss": train_loss,
                "val/loss":   val_loss,
                "lr":         scheduler.get_last_lr()[0],
                **train_metrics,
                **val_metrics,
            }
            wandb.log(log_dict)
            all_metrics.update({**train_metrics, **val_metrics})

            # ── Checkpoint ────────────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f" Best model saved (val_loss={val_loss:.4f})")
                wandb.save(ckpt_path)

    # ── Test (load best checkpoint) ────────────────────────────────────────────
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")), strict=False)
    test_loss, test_preds, test_labels = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False
    )

    test_metrics = evaluate(test_labels, test_preds, "test", plot=True, outdir=output_dir, scale="raw")

    wandb.log({"test/loss": test_loss, **test_metrics})
    all_metrics.update(test_metrics)

    # ── Save metrics ───────────────────────────────────────────────────────────
    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=_convert)

    wandb.finish()
    print(f"\nAll outputs saved to: {output_dir}/")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
num_labels = 3142  # adjust if needed based on your dataset
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cont_data",    default="DATA/gene expression/gene_Exp_3w_cont.parquet")
    parser.add_argument("--disc_data",    default="DATA/gene expression/gene_Exp_3w_disc.csv")
    parser.add_argument("--labels",       default="gene expression/arabidopsis_expression_data.h5")
    parser.add_argument("--split_path",   default="gene expression/")
    parser.add_argument("--chrombert_ckpt", required=True, help="Path to pretrained ChromBERT checkpoint")
    parser.add_argument("--output_dir",   default="results_chrombert_gene_expression_log1p_mtx")
    parser.add_argument("--num_epochs",   type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--freeze_epochs",type=int,   default=2,
                        help="Epochs to keep backbone frozen before unfreezing last 3 layers")
    parser.add_argument("--wandb_project",default="gene-expression-chrombert")
    parser.add_argument("--mode",         choices=["train", "eval"], default="train",
                        help="Whether to run training or just evaluation (requires --chrombert_ckpt and saved metrics)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load pretrained ChromBERT ──────────────────────────────────────────────
    chrombert_model = ChromBERT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS)          # instantiate your model class
    model_path = args.chrombert_ckpt

    if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...")
            state_dict = torch.load(model_path, map_location="cpu")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

            chrombert_model.load_state_dict(new_state_dict)

    print("ChromBERT loaded from", args.chrombert_ckpt)

    # ── Dataset & splits ──────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = GeneExpressionDataset(args.cont_data, args.disc_data, args.labels)
    splits  = dataset.load_splits(args.split_path)
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    finetune_chrombert(
        dataset, splits, chrombert_model,
        num_labels=num_labels,
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        mode = args.mode
    )