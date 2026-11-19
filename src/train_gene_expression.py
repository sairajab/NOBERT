import time
import json
import os

import h5py
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import wandb

from models import ChromBERTModel as ChromBERT, MultiWindowHeaderFusion, ClassificationHeadOnly
from distributed_utils import (
    is_distributed,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    get_raw_model,
    gather_predictions,
    is_better,
    broadcast_stop,
)

"""
Multi-task gene expression regression: ChromBERT backbone -> MultiWindowHeaderFusion
head (shared with chrombert_finetune_enhancers.py's binary classification) -> linear output over
n_experiments continuous log1p(FPKM) targets, trained with MSELoss.

Convention: y_true, y_pred have shape (n_genes, n_experiments).
    axis=1  -> across experiments  (per-gene metric)
    axis=0  -> across genes        (per-experiment metric)
"""

MAX_SEQ_LEN = 2689
VOCAB_SIZE = 7  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
PAD_TOKEN = 0
MASK_TOKEN = 5
NUM_LABELS = 3142



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
# Top-level metrics entry point
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
    """Plot a handful of random genes and experiments, true vs predicted."""
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


def _log_summary(summary, prefix):
    if not is_main_process():
        return
    wandb.log({
        f"{prefix}/pearson_per_gene_mean":         summary["pearson_per_gene"]["mean"],
        f"{prefix}/pearson_per_exp_mean":           summary["pearson_per_exp"]["mean"],
        f"{prefix}/pearson_per_exp_centered_mean":  summary["pearson_per_exp_centered"]["mean"],
        f"{prefix}/r2_per_gene_mean":                summary["r2_per_gene"]["mean"],
        f"{prefix}/global_pearson":                  summary["global_pearson"],
        f"{prefix}/global_rmse":                     summary["global_rmse"],
        f"{prefix}/skill_over_mean_predictor":       summary["skill_over_mean_predictor"],
    })


# ──────────────────────────────────────────────────────────────
# Training / evaluation loops
#
# Structurally mirrors chrombert_finetune_enhancers.py's train_one_epoch / evaluate_epoch /
# run_training so the two finetuning scripts are easy to read side by side. The
# task-specific pieces are: MSELoss instead of CrossEntropyLoss, float (not long)
# labels, no softmax on the output, and Pearson correlation (not AUPR) as the
# "higher is better" metric driving checkpoint/early-stop decisions.
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0, scheduler=None, epoch=0, phase=1):
    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(train_one_epoch._current_epoch)

    total_loss = 0.0
    all_preds, all_labels = [], []
    global_step = epoch * len(loader)

    for i, batch in enumerate(loader):
        s_time = time.time()
        input_ids, labels = batch
        input_ids = input_ids.to(device).long()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids)[0].float()
            loss = criterion(outputs, labels)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_time = time.time() - s_time
        if i % 100 == 0:
            print(f"Processing batch {i} /{len(loader)} - batch time: {batch_time:.4f}s", flush=True)

        if is_main_process():
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            wandb.log({
                f"phase{phase}/batch/loss": loss.item(),
                f"phase{phase}/batch/grad_norm": grad_norm.item(),
                f"phase{phase}/batch/lr": current_lr,
                f"phase{phase}/batch/batch_time_s": batch_time,
                "global_step": global_step + i,
            })

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    gathered_preds, gathered_labels = gather_predictions(all_preds, all_labels)
    pearson_mean = 0.0
    if is_main_process():
        summary = evaluate(gathered_labels, gathered_preds, split_name=f"phase{phase}/train", plot=False)
        _log_summary(summary, f"phase{phase}/train")
        pearson_mean = summary["pearson_per_gene"]["mean"]

    avg_loss = total_loss / len(loader)

    if is_main_process():
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        print(f"  Train pearson (per-gene mean): {pearson_mean:.4f} | Current LR (Head): {current_lr:.2e}")

    return avg_loss, pearson_mean


# Static variable to track epoch for sampler
train_one_epoch._current_epoch = 0


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, outdir=None, save_predictions=False,
                    epoch=0, phase=1, split="val", scale="log1p", gather=True):
    """Run one evaluation pass. `gather=True` performs a distributed all_gather so the
    returned metric reflects the full split rather than just this rank's local shard —
    use this for validation, where every rank calls evaluate_epoch together. Pass
    `gather=False` for a single-process (rank-0-only) final test evaluation, where no
    other rank is present to participate in a collective op.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device).long()
        labels = labels.to(device).float()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids)[0].float()
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if gather:
        metric_preds, metric_labels = gather_predictions(all_preds, all_labels)
    else:
        metric_preds, metric_labels = all_preds, all_labels

    avg_loss = total_loss / len(loader)
    pearson_mean = 0.0
    if is_main_process():
        summary = evaluate(metric_labels, metric_preds, split_name=split,
                           plot=save_predictions, outdir=outdir or ".", scale=scale)
        _log_summary(summary, f"phase{phase}/{split}")
        wandb.log({f"phase{phase}/{split}/loss": avg_loss, f"phase{phase}/epoch": epoch})
        pearson_mean = summary["pearson_per_gene"]["mean"]

    return avg_loss, pearson_mean


# ──────────────────────────────────────────────────────────────
# Main training driver
# ──────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, test_loader=None, device='cuda', local_rank=0, outdir=None,
                 phase1_epochs=2, phase2_epochs=10,
                 lr_phase1=1e-5, lr_phase2_backbone=1e-5, lr_phase2_head=1e-4,
                 weight_decay_phase1=0.01, weight_decay_phase2=0.01,
                 patience_phase1=3, patience_phase2=7, unfreeze_layers=3):

    criterion = nn.MSELoss()

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = get_raw_model(model)

    # if is_main_process():
    #     wandb.watch(raw_model, log="all", log_freq=200)

    # ============================================
    # PHASE 1: Frozen backbone, train head only
    # ============================================
    if is_main_process():
        print("=" * 60)
        print("PHASE 1: Training head with frozen backbone")
        print("=" * 60)
        wandb.log({"phase": 1})

    raw_model.freeze_backbone()

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_phase1 = AdamW(head_params, lr=lr_phase1, weight_decay=weight_decay_phase1)
    scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=max(phase1_epochs, 1))

    best_val_pearson = -float("inf")
    patience_counter = 0

    for epoch in range(phase1_epochs):
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch._current_epoch = epoch
        epoch_start_time = time.time()

        train_loss, train_pearson = train_one_epoch(
            model, train_loader, optimizer_phase1, criterion, device,
            epoch=epoch, phase=1,
        )

        val_loss, val_pearson = evaluate_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, phase=1, split="val",
        )
        scheduler_phase1.step()

        # --- Early stopping logic (rank 0 decides, then broadcasts) ---
        should_stop = False
        if is_main_process():
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1:2d} | Train Pearson: {train_pearson:.4f} | Val Pearson: {val_pearson:.4f} | Val Loss: {val_loss:.4f}")

            wandb.log({
                "phase1/epoch/train_loss": train_loss,
                "phase1/epoch/train_pearson": train_pearson,
                "phase1/epoch/val_loss": val_loss,
                "phase1/epoch/val_pearson": val_pearson,
                "phase1/epoch/patience": patience_counter,
                "phase1/epoch/epoch_time_s": epoch_time,
                "phase1/epoch": epoch,
            })

            if is_better(val_pearson, best_val_pearson, mode="max"):
                best_val_pearson = val_pearson
                ckpt_path = os.path.join(outdir, 'best_phase_1.pt')
                torch.save(raw_model.state_dict(), ckpt_path)
                patience_counter = 0
                wandb.log({"phase1/best_val_pearson": best_val_pearson})
                # artifact = wandb.Artifact("phase1_checkpoint", type="model",
                #                           description=f"Phase 1 best (epoch {epoch+1})",
                #                           metadata={"epoch": epoch + 1, "val_pearson": val_pearson})
                # artifact.add_file(ckpt_path)
                # wandb.log_artifact(artifact)
            else:
                patience_counter += 1

            if patience_counter >= patience_phase1:
                print("  Early stopping phase 1")
                wandb.log({"phase1/early_stop_epoch": epoch + 1})
                should_stop = True

        should_stop = broadcast_stop(should_stop, device)
        if should_stop:
            break

    if is_main_process():
        print(f"  Best Phase 1 Val Pearson: {best_val_pearson:.4f}")
        wandb.summary["phase1/best_val_pearson"] = best_val_pearson

    # ============================================
    # PHASE 2: Unfreeze backbone, discriminative LR
    # ============================================
    if is_main_process():
        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning full model with discriminative LR")
        print("=" * 60)
        wandb.log({"phase": 2})

    # --- Destroy old DDP wrapper before modifying the model ---
    if is_distributed():
        del model
        torch.cuda.empty_cache()
        dist.barrier()  # wait for rank 0 to finish saving the phase-1 checkpoint

    # ALL ranks load the best phase 1 checkpoint (not just rank 0)
    best_ckpt_path = os.path.join(outdir, 'best_phase_1.pt')
    if is_main_process():
        print("Loading best phase 1 checkpoint for fine-tuning...")

    raw_model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    raw_model = raw_model.to(device)
    raw_model.unfreeze_backbone(unfreeze_layers=unfreeze_layers)

    # --- Re-wrap in DDP with the updated model ---
    if is_distributed():
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = raw_model

    optimizer_phase2 = AdamW([
        {'params': raw_model.chrombert.encoder_layers[-unfreeze_layers:].parameters(), 'lr': lr_phase2_backbone},
        {'params': raw_model.res1.parameters(), 'lr': lr_phase2_head},
        {'params': raw_model.res2.parameters(), 'lr': lr_phase2_head},
        {'params': raw_model.conv.parameters(), 'lr': lr_phase2_head},
        {'params': raw_model.res3.parameters(), 'lr': lr_phase2_head},
        {'params': raw_model.fc.parameters(), 'lr': lr_phase2_head},
    ], weight_decay=weight_decay_phase2)

    best_val_pearson_phase2 = best_val_pearson
    patience_counter = 0

    for epoch in range(phase2_epochs):
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch + phase1_epochs)  # offset to avoid same seed as phase 1

        train_one_epoch._current_epoch = epoch + phase1_epochs
        epoch_start_time = time.time()

        train_loss, train_pearson = train_one_epoch(
            model, train_loader, optimizer_phase2, criterion, device,
            max_grad_norm=1.0,
            scheduler=None,
            epoch=epoch,
            phase=2,
        )

        val_loss, val_pearson = evaluate_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, phase=2, split="val",
        )

        # --- Early stopping logic (rank 0 decides, then broadcasts) ---
        should_stop = False
        if is_main_process():
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1:2d} | Train Pearson: {train_pearson:.4f} | Val Pearson: {val_pearson:.4f} | Val Loss: {val_loss:.4f}")

            wandb.log({
                "phase2/epoch/train_loss": train_loss,
                "phase2/epoch/train_pearson": train_pearson,
                "phase2/epoch/val_loss": val_loss,
                "phase2/epoch/val_pearson": val_pearson,
                "phase2/epoch/patience": patience_counter,
                "phase2/epoch/epoch_time_s": epoch_time,
                "phase2/epoch": epoch,
            })

            if is_better(val_pearson, best_val_pearson_phase2, mode="max"):
                best_val_pearson_phase2 = val_pearson
                ckpt_path = os.path.join(outdir, 'best_phase_2.pt')
                torch.save(raw_model.state_dict(), ckpt_path)
                patience_counter = 0
                wandb.log({"phase2/best_val_pearson": best_val_pearson_phase2})
                # artifact = wandb.Artifact("phase2_checkpoint", type="model",
                #                           description=f"Phase 2 best (epoch {epoch+1})",
                #                           metadata={"epoch": epoch + 1, "val_pearson": val_pearson})
                #artifact.add_file(ckpt_path)
                #wandb.log_artifact(artifact)
            else:
                patience_counter += 1

            if patience_counter >= patience_phase2:
                print("  Early stopping phase 2")
                wandb.log({"phase2/early_stop_epoch": epoch + 1})
                should_stop = True

        should_stop = broadcast_stop(should_stop, device)
        if should_stop:
            break

    if is_main_process():
        print(f"  Best Phase 2 Val Pearson: {best_val_pearson_phase2:.4f}")
        wandb.summary["phase2/best_val_pearson"] = best_val_pearson_phase2

    # ============================================
    # FINAL EVALUATION (rank 0 only, single process, full test set)
    # ============================================
    if test_loader is not None and is_main_process():
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        raw_model.load_state_dict(torch.load(os.path.join(outdir, 'best_phase_2.pt'), map_location="cpu"))
        raw_model = raw_model.to(device)
        test_loss, test_pearson = evaluate_epoch(
            raw_model, test_loader, criterion, device,
            phase=2, split="test", save_predictions=True, outdir=outdir, scale="raw", gather=False,
        )
        print(f"  Test Pearson (per-gene mean): {test_pearson:.4f}")
        wandb.summary["test/pearson_per_gene_mean"] = test_pearson
        wandb.summary["test/loss"] = test_loss

        with open(os.path.join(outdir, "test_results.json"), "w") as f:
            json.dump({"test_loss": test_loss, "test_pearson_per_gene_mean": test_pearson}, f, indent=2)

    return raw_model


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cont_data",    default="/s/chromatin/c/nobackup/Saira/BERTModel//DATA/gene expression/gene_Exp_3w_cont.parquet")
    parser.add_argument("--disc_data",    default="/s/chromatin/c/nobackup/Saira/BERTModel//DATA/gene expression/gene_Exp_500__5emb_disc.csv")
    parser.add_argument("--labels",       default="/s/chromatin/c/nobackup/Saira/BERTModel/gene expression/arabidopsis_expression_data.h5")
    parser.add_argument("--split_path",   default="/s/chromatin/c/nobackup/Saira/BERTModel//gene expression/")
    parser.add_argument("--chrombert_ckpt", required=True, help="Path to pretrained ChromBERT checkpoint")
    parser.add_argument("--output_dir",   default="/s/chromatin/c/nobackup/Saira/BERTModel/NOBERT/results/results_chrombert_gene_expression_500_log1p_mtx_5emb_new")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--phase1_epochs", type=int, default=2)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--lr_phase1", type=float, default=1e-5)
    parser.add_argument("--lr_phase2_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_phase2_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay_phase1", type=float, default=0.01)
    parser.add_argument("--weight_decay_phase2", type=float, default=0.01)
    parser.add_argument("--patience_phase1", type=int, default=3)
    parser.add_argument("--patience_phase2", type=int, default=7)
    parser.add_argument("--unfreeze_layers", type=int, default=3)
    parser.add_argument("--wandb_project", default="gene-expression-chrombert")
    parser.add_argument("--mode",         choices=["train", "eval"], default="train",
                        help="Whether to run training or just evaluate an existing best_phase_2 checkpoint")
    args = parser.parse_args()

    device, local_rank = setup_distributed()

    # save args to output_dir (rank 0 only)
    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process():
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved arguments to {os.path.join(args.output_dir, 'args.json')}")

    # ── Init WandB (rank 0 only; other ranks get a disabled no-op run) ──
    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=f"gene_expression_h{N_HEADS}_l{N_LAYERS}_b{args.batch_size}",
            config={
                **vars(args),
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "max_seq_len": MAX_SEQ_LEN,
                "num_labels": NUM_LABELS,
                "model": "MultiWindowHeaderFusion",
                "criterion": "MSELoss",
            },
            dir=args.output_dir,
        )
    else:
        # Non-zero DDP ranks: disable wandb entirely to avoid duplicate logs
        wandb.init(mode="disabled")

    # ── Load pretrained ChromBERT ──────────────────────────────────────────────
    chrombert_model = ChromBERT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS)
    model_path = args.chrombert_ckpt

    if os.path.exists(model_path):
        if is_main_process():
            print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        chrombert_model.load_state_dict(new_state_dict)

    print("ChromBERT loaded from", args.chrombert_ckpt)
    chrombert_model = chrombert_model.to(device)

    # ── Dataset & splits ──────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = GeneExpressionDataset(args.cont_data, args.disc_data, args.labels)
    splits  = dataset.load_splits(args.split_path)
    if is_main_process():
        print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    train_ds = Subset(dataset, splits["train"])
    val_ds   = Subset(dataset, splits["val"])
    test_ds  = Subset(dataset, splits["test"])

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    # Test set is only ever evaluated on rank 0 as a single process (see run_training),
    # so it deliberately does NOT use a DistributedSampler.

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = MultiWindowHeaderFusion(chrombert_model, num_labels=NUM_LABELS, n_windows=1).to(device)
    #model = ClassificationHeadOnly(num_labels=NUM_LABELS).to(device)
    if args.mode == "train":
        model = run_training(
            model, train_loader, val_loader, test_loader, device, local_rank, args.output_dir,
            phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
            lr_phase1=args.lr_phase1, lr_phase2_backbone=args.lr_phase2_backbone, lr_phase2_head=args.lr_phase2_head,
            weight_decay_phase1=args.weight_decay_phase1, weight_decay_phase2=args.weight_decay_phase2,
            patience_phase1=args.patience_phase1, patience_phase2=args.patience_phase2,
            unfreeze_layers=args.unfreeze_layers,
        )
    else:
        # mode == "eval": load an existing checkpoint (rank 0 only) and report test metrics
        ckpt_path = os.path.join(args.output_dir, "best_phase_2.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, "best_phase_1.pt")
        if is_main_process():
            print(f"Eval-only mode: loading {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = model.to(device)
            criterion = nn.MSELoss()
            evaluate_epoch(model, test_loader, criterion, device,
                          phase=2, split="test", save_predictions=True,
                          outdir=args.output_dir, scale="raw", gather=False)

    if is_main_process():
        wandb.finish()

    cleanup_distributed()
