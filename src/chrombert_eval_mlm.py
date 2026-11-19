#!/usr/bin/env python3
"""
chrombert_eval_mlm.py
=====================
Evaluate pre-trained ChromBERT on held-out bins: how well does it recover masked chromatin states?

Uses the same splits and the same cluster-structured masking as chrombert_pretrain.py, so test bins were
never trained on. Masks are generated on the fly with a fixed seed (no precomputed cache needed,
--masks_per_bin random masks per bin).

Reports, at masked positions
  - focal loss
  - overall accuracy and per-token accuracy (tokens 1..n_tokens)
  - confusion matrix (true token x predicted token)
  - baseline: accuracy of always predicting each experiment's most common token in the train bins

Usage
  python chrombert_eval_mlm.py                               # full test split, default checkpoint
  python chrombert_eval_mlm.py --split val
  python chrombert_eval_mlm.py --max_bins 2000               # random 2000 test bins (quicker)
  python chrombert_eval_mlm.py --checkpoint path/to/best_genomic_bert.pt

Output: <out_dir>/mlm_eval_<split>.json  (out_dir defaults to the checkpoint's folder)
"""

import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import PRETRAIN_DATA, PRETRAIN_SPLITS, CLUSTER_FILE, CHROMBERT_CKPT, N_TOKENS, vocab_for
from dataloaders import ChromBERTDataset
from models import load_chrombert
from utils import FocalLoss


def parse_args():
    p = argparse.ArgumentParser(description="Masked-token evaluation of pre-trained ChromBERT")
    p.add_argument("--checkpoint", default=CHROMBERT_CKPT)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--data", default=PRETRAIN_DATA)
    p.add_argument("--split_file", default=PRETRAIN_SPLITS)
    p.add_argument("--cluster_file", default=CLUSTER_FILE)
    p.add_argument("--masks_per_bin", type=int, default=1, help="random masks per bin")
    p.add_argument("--n_tokens", type=int, default=N_TOKENS)
    p.add_argument("--max_bins", type=int, default=None, help="random subset of bins (default: all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", default=None, help="default: folder of --checkpoint")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def column_majority(data, train_bins, n_tokens):
    # most common token per experiment (column) over the train bins
    counts = np.stack([(data[train_bins] == t).sum(axis=0) for t in range(1, n_tokens + 1)])  # (n_tokens, n_exp)
    return counts.argmax(axis=0) + 1


@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab_size, n_tokens, majority):
    model.eval()
    total_loss, n_batches = 0.0, 0
    confusion = np.zeros((n_tokens + 1, vocab_size), dtype=np.int64)  # row = true token, col = predicted
    base_correct, n_masked = 0, 0
    majority = torch.as_tensor(majority, device=device)

    for bidx, (seq, target) in enumerate(loader):
        seq = seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, logits = model(seq)  # (B, S, V)
            loss = criterion(logits.view(-1, vocab_size), target.view(-1))
        total_loss += loss.item()
        n_batches += 1

        masked = target != -100
        true = target[masked]
        pred = logits.argmax(dim=-1)[masked]
        np.add.at(confusion, (true.cpu().numpy(), pred.cpu().numpy()), 1)

        # baseline: column majority token at the same masked positions
        base_pred = majority.expand_as(target)[masked]
        base_correct += (base_pred == true).sum().item()
        n_masked += masked.sum().item()

        if bidx % 200 == 0:
            print(f"  batch {bidx}/{len(loader)}", flush=True)

    correct = np.trace(confusion[:, :n_tokens + 1])
    per_token = {}
    for t in range(1, n_tokens + 1):
        total = int(confusion[t].sum())
        per_token[f"token_{t}"] = {
            "accuracy": 100 * confusion[t, t] / max(1, total),
            "correct": int(confusion[t, t]),
            "total": total,
        }

    return {
        "avg_loss": total_loss / max(1, n_batches),
        "overall_accuracy": 100 * correct / max(1, n_masked),
        "baseline_column_majority_accuracy": 100 * base_correct / max(1, n_masked),
        "total_masked_tokens": int(n_masked),
        "per_token": per_token,
        "confusion_true_x_pred": confusion[1:, 1:n_tokens + 1].tolist(),  # tokens 1..n_tokens only
    }


def print_results(results, split):
    print("\n" + "=" * 60)
    print(f"  MLM EVALUATION — {split} set")
    print("=" * 60)
    print(f"  Average loss:            {results['avg_loss']:.4f}")
    print(f"  Overall accuracy:        {results['overall_accuracy']:.2f}%")
    print(f"  Baseline (col majority): {results['baseline_column_majority_accuracy']:.2f}%")
    print(f"  Masked tokens evaluated: {results['total_masked_tokens']:,}")
    print("-" * 60)
    print(f"  {'Token':<10} {'Accuracy':>10} {'Correct':>12} {'Total':>12}")
    for name, s in results["per_token"].items():
        print(f"  {name:<10} {s['accuracy']:>9.2f}% {s['correct']:>12,} {s['total']:>12,}")
    print("=" * 60)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size, mask_token = vocab_for(args.n_tokens)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    print(f"Evaluating {args.checkpoint} on {device}")

    # ── Data (masks made on the fly, seeded) ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = ChromBERTDataset(
        args.data, mask_mode="dynamic",
        mask_token=mask_token, random_tokens=range(1, args.n_tokens + 1), cluster_file=args.cluster_file,
    )
    splits = dataset.create_splits_from_data(args.split_file)
    bins = splits[args.split]
    if args.max_bins and args.max_bins < len(bins):
        rng = np.random.default_rng(args.seed)
        bins = sorted(rng.choice(bins, size=args.max_bins, replace=False).tolist())
    idx = [i for i in bins for _ in range(args.masks_per_bin)]
    print(f"{args.split}: {len(bins)} bins x {args.masks_per_bin} masks = {len(idx)} sequences")

    # num_workers=0 keeps the random masks reproducible
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    majority = column_majority(dataset.data, splits["train"], args.n_tokens)

    # ── Model ──
    model = load_chrombert(args.checkpoint, vocab_size).to(device)
    criterion = FocalLoss(gamma=2.0, ignore_index=-100)

    results = evaluate(model, loader, criterion, device, vocab_size, args.n_tokens, majority)
    results.update({"checkpoint": args.checkpoint, "split": args.split, "n_bins": len(bins),
                    "masks_per_bin": args.masks_per_bin, "seed": args.seed})
    print_results(results, args.split)

    out = os.path.join(out_dir, f"mlm_eval_{args.split}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
