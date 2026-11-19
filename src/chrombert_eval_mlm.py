import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from train_mlm_2 import GenomicTransformer  
from train_mlm_2 import ChromBERTDataset
# ================= CONFIGURATION =================
VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
BATCH_SIZE = 8
MASK_PROB = 0.15
PAD_TOKEN = 0
MASK_TOKEN = 5

LABEL_NAMES = {
    1: "Label_1",
    2: "Label_2",
    3: "Label_3",
    4: "Label_4",
}





# ================= FOCAL LOSS =================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        if logits.ndim != 2:
            raise ValueError(f"logits must be [N, C], got {tuple(logits.shape)}")

        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            if valid.sum() == 0:
                return logits.new_tensor(0.0)
            logits = logits[valid]
            targets = targets[valid]

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        idx = torch.arange(logits.size(0), device=logits.device)
        p_t = probs[idx, targets]
        log_p_t = log_probs[idx, targets]
        loss = -((1.0 - p_t) ** self.gamma) * log_p_t

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


# ================= SPLIT LOADER =================
def load_splits_data(dataset, train_split, val_split, test_split):
    def load_ids(path):
        if not os.path.exists(path):
            return set()
        return set(open(path, "r").read().strip().split("\n"))

    train_file = load_ids(train_split)
    val_file = load_ids(val_split)
    test_file = load_ids(test_split)

    train_indices, val_indices, test_indices = [], [], []
    for i, idx in enumerate(dataset.ids):
        idx_str = str(idx)
        if idx_str in train_file:
            train_indices.append(i)
        elif idx_str in val_file:
            val_indices.append(i)
        elif idx_str in test_file:
            test_indices.append(i)

    return train_indices, val_indices, test_indices


# ================= EVALUATION FUNCTION =================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_runs=5):
    """
    Evaluate the model over multiple masking runs to get stable metrics.

    Since masking is stochastic, each run masks different positions.
    Averaging over multiple runs gives more reliable per-class accuracy.

    Args:
        model: Trained GenomicTransformer
        dataloader: DataLoader yielding (masked_seq, target, original)
        criterion: Loss function (FocalLoss)
        device: torch device
        num_runs: Number of stochastic masking passes to average over

    Returns:
        dict with overall metrics and per-class accuracy breakdown
    """
    model.eval()

    # Accumulators across all runs
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_masked = 0

    # Per-class accumulators: class_id -> {correct, total}
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for run in range(num_runs):
        run_loss = 0.0
        run_batches = 0

        for seq, target in dataloader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(seq)  # (B, S, V)
                loss = criterion(logits.view(-1, VOCAB_SIZE), target.view(-1))

            run_loss += loss.item()
            run_batches += 1

            # Predictions at masked positions only
            preds = logits.argmax(dim=-1)  # (B, S)
            masked_positions = target != -100  # (B, S)

            if masked_positions.sum() == 0:
                continue

            pred_masked = preds[masked_positions]
            true_masked = target[masked_positions]  # Ground truth labels

            total_correct += (pred_masked == true_masked).sum().item()
            total_masked += masked_positions.sum().item()

            # Per-class breakdown
            for cls_id in LABEL_NAMES.keys():
                cls_mask = true_masked == cls_id
                if cls_mask.sum() == 0:
                    continue
                per_class_correct[cls_id] += (pred_masked[cls_mask] == cls_id).sum().item()
                per_class_total[cls_id] += cls_mask.sum().item()

        total_loss += run_loss / max(1, run_batches)
        total_batches += 1

        print(f"  Run {run + 1}/{num_runs} complete.", flush=True)

    # --- Aggregate Results ---
    avg_loss = total_loss / num_runs
    overall_accuracy = total_correct / max(1, total_masked) * 100

    per_class_results = {}
    for cls_id, name in LABEL_NAMES.items():
        correct = per_class_correct[cls_id]
        total = per_class_total[cls_id]
        acc = correct / max(1, total) * 100
        per_class_results[name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

    results = {
        "avg_loss": avg_loss,
        "overall_accuracy": overall_accuracy,
        "total_masked_tokens": total_masked,
        "total_correct": total_correct,
        "num_runs": num_runs,
        "per_class": per_class_results,
    }

    return results


def print_results(results, split_name="Test"):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {split_name} Set ({results['num_runs']} masking runs)")
    print("=" * 60)
    print(f"  Average Loss:       {results['avg_loss']:.4f}")
    print(f"  Overall Accuracy:   {results['overall_accuracy']:.2f}%")
    print(f"  Total Masked Tokens Evaluated: {results['total_masked_tokens']:,}")
    print("-" * 60)
    print(f"  {'Class':<12} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 60)
    for name, stats in results["per_class"].items():
        print(
            f"  {name:<12} {stats['accuracy']:>9.2f}% {stats['correct']:>10,} {stats['total']:>10,}"
        )
    print("=" * 60)


# ================= MAIN =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    window_200 = False  # Set to True if using the 200bp window dataset, False for 500bp bins dataset
    model = GenomicTransformer(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS).to(device)

    # --- Load data ---
    if window_200:
        dataset = ChromBERTDataset("out.parquet", mask_mode="precomputed_all", num_masks=7, cache_dir="mask_cache_200")
        train_indices, val_indices, test_indices = load_splits_data(
            dataset, "ftrain_idxs.txt", "fval_idxs.txt", "ftest_idxs.txt"
        )

        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        if dataset.mask_mode == "precomputed_all":
            num_masks = dataset.num_masks
            val_indices = [i * num_masks + m for i in val_indices for m in range(num_masks)]
            test_indices = [i * num_masks + m for i in test_indices for m in range(num_masks)]

        test_subset = Subset(dataset, test_indices)
        val_subset = Subset(dataset, val_indices)


        test_loader = DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # --- Load model ---

        checkpoint_path = "chromBERT_models2/best_genomic_bert.pt"
    else:
        dataset = ChromBERTDataset("bins500bp_labels.parquet",  mask_mode="precomputed_all", num_masks=7, cache_dir="mask_cache_500")
            # Create splits from data
        splits = dataset.create_splits_from_data("/s/chromatin/a/nobackup/Saira/SPLICE_SITE_PREDICTION/src/bins_500bp.csv")

        val_indices = splits["val"]
        test_indices = splits["test"]
        if dataset.mask_mode == "precomputed_all":
            num_masks = dataset.num_masks
            val_indices = [i * num_masks + m for i in val_indices for m in range(num_masks)]
            test_indices = [i * num_masks + m for i in test_indices for m in range(num_masks)]

        print(f"Val: {len(val_indices)}, Test: {len(test_indices)}")
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True )
        test_loader = DataLoader(
            test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True )
        checkpoint_path = "chromBERT_models_all_clusters500/best_genomic_bert.pt"


    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    # state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")

    criterion = FocalLoss(gamma=2.0, ignore_index=-100)

    # --- Evaluate on validation set ---
    print("\nEvaluating on Validation set...")
    val_results = evaluate(model, val_loader, criterion, device, num_runs=1)
    print_results(val_results, split_name="Validation")

    # --- Evaluate on test set ---
    print("\nEvaluating on Test set...")
    test_results = evaluate(model, test_loader, criterion, device, num_runs=1)
    print_results(test_results, split_name="Test")


if __name__ == "__main__":
    main()