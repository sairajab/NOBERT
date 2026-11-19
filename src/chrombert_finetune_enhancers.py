#!/usr/bin/env python3
"""
chrombert_finetune_enhancers.py
===============================
Finetune pre-trained ChromBERT for enhancer classification (one 500bp window centered on each region).

Model : ChromBERT backbone -> cistrome interface (experiments -> factors, mtx_mask.txt)
        -> residual MLP head -> 2 logits.  With --features dna/histone/tf the head uses only those
        experiments instead of the cistrome interface.
Train : phase 1 - backbone frozen, train head
        phase 2 - unfreeze last --unfreeze_layers encoder layers, lower LR for backbone than head
        class-weighted cross entropy, model selection / early stopping on val AUPRC
Test  : best checkpoint (phase 2, or phase 1 if phase 2 never beat it) on the test split

Datasets (--dataset)
  c1   - C1 enhancers vs background       (C1_2500_enhancer_dataset.txt)
  all  - all enhancers vs background      (enhancers_2500_labels_no_duplicates.txt)

Defaults reproduce results/chrombert_finetune/5emb_C1_enh_final (c1) and 5emb_enhancers_all (all).

Usage
  python chrombert_finetune_enhancers.py --dataset c1
  python chrombert_finetune_enhancers.py --dataset all
  python chrombert_finetune_enhancers.py --dataset c1 --features dna
  python chrombert_finetune_enhancers.py --dataset c1 --eval_only --output_dir ../results/chrombert_finetune/5emb_C1_enh_final
  torchrun --nproc_per_node 2 chrombert_finetune_enhancers.py --dataset all    # multi GPU

Output (in --output_dir, default ../results/chrombert_finetune/enhancers_<dataset>_<features>)
  best_phase_1.pt, best_phase_2.pt
  phase2_test_predictions.csv   Name, Label, Predicted_Prob
  test_metrics.json             auroc, auprc, accuracy, f1, ... (same metrics as baselines_enhancers.py)
  test_results.txt, args.json
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler, Subset
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score, recall_score)
import wandb

from config import (ENH_DATASETS, ENH_TRACKS_5TOK, CHROMBERT_CKPT, RESULTS_DIR, N_TOKENS,
                    vocab_for, resolve_mask, feature_tag)
from dataloaders import EnhancersDatasetFast
from models import MultiWindowHeaderFusion, load_chrombert, N_HEADS, N_LAYERS
from distributed_utils import (
    is_distributed, is_main_process, setup_distributed, cleanup_distributed,
    get_raw_model, gather_predictions, is_better, broadcast_stop,
)


# ──────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0, epoch=0, phase=1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    global_step = epoch * len(loader)

    for i, (_, input_ids, labels) in enumerate(loader):
        s_time = time.time()
        input_ids = input_ids.to(device).long()
        labels = labels.to(device).long()

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)[0]  # (B, 2)
            loss = criterion(logits, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_time = time.time() - s_time
        if i % 100 == 0 and is_main_process():
            print(f"  batch {i}/{len(loader)} - loss {loss.item():.4f} - {batch_time:.3f}s", flush=True)

        if is_main_process():
            wandb.log({
                f"phase{phase}/batch/loss": loss.item(),
                f"phase{phase}/batch/grad_norm": grad_norm.item(),
                f"phase{phase}/batch/lr": optimizer.param_groups[0]["lr"],
                "global_step": global_step + i,
            })

        total_loss += loss.item()
        all_preds.append(torch.softmax(logits, dim=-1)[:, 1].float().detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds, labels = gather_predictions(np.concatenate(all_preds), np.concatenate(all_labels))
    aupr = average_precision_score(labels, preds) if is_main_process() else 0.0
    return total_loss / len(loader), aupr


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, gather=True):
    """One pass over loader. Returns (loss, aupr, names, labels, probs).
    gather=True collects predictions from all ranks (validation); use gather=False for the
    rank-0-only test evaluation, where no other rank takes part."""
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_names = [], [], []

    for names, input_ids, labels in loader:
        input_ids = input_ids.to(device).long()
        labels = labels.to(device).long()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)[0]
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_names.extend(names)

    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)
    if gather:
        preds, labels = gather_predictions(preds, labels)
    aupr = average_precision_score(labels, preds) if is_main_process() else 0.0
    return total_loss / len(loader), aupr, all_names, labels, preds


def compute_class_weights(dataset, indices, device):
    """Inverse-frequency class weights from the train split (normalized to mean 1)."""
    counts = np.bincount(dataset.labels_data.iloc[indices]["Label"].values)
    if is_main_process():
        print(f"  Class counts in train: {dict(enumerate(counts))} (neg:pos = {counts[0]/counts[1]:.1f}:1)")
    weights = 1.0 / counts.astype(np.float32)
    return torch.tensor(weights / weights.mean(), device=device)


def run_phase(phase, model, raw_model, optimizer, train_loader, val_loader, criterion, device, outdir,
              epochs, patience, best_val_aupr, scheduler=None, epoch_offset=0):
    """Train for up to `epochs`, save best_phase_<phase>.pt when val AUPRC beats best_val_aupr."""
    ckpt_path = os.path.join(outdir, f"best_phase_{phase}.pt")
    patience_counter = 0

    for epoch in range(epochs):
        if is_distributed() and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch + epoch_offset)
        t0 = time.time()

        train_loss, train_aupr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                 epoch=epoch, phase=phase)
        val_loss, val_aupr, *_ = evaluate_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        # rank 0 decides early stopping, then broadcasts
        should_stop = False
        if is_main_process():
            print(f"  Phase {phase} epoch {epoch+1:2d} | Train AUPR: {train_aupr:.4f} | "
                  f"Val AUPR: {val_aupr:.4f} | Val Loss: {val_loss:.4f} | {time.time() - t0:.0f}s")
            wandb.log({
                f"phase{phase}/epoch/train_loss": train_loss,
                f"phase{phase}/epoch/train_aupr": train_aupr,
                f"phase{phase}/epoch/val_loss": val_loss,
                f"phase{phase}/epoch/val_aupr": val_aupr,
                f"phase{phase}/epoch": epoch,
            })
            if is_better(val_aupr, best_val_aupr, mode="max"):
                best_val_aupr = val_aupr
                torch.save(raw_model.state_dict(), ckpt_path)
                patience_counter = 0
                print(f"  --> saved {ckpt_path}")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping phase {phase}")
                should_stop = True

        if broadcast_stop(should_stop, device):
            break

    if is_main_process():
        print(f"  Best val AUPR after phase {phase}: {best_val_aupr:.4f}")
        wandb.summary[f"phase{phase}/best_val_aupr"] = best_val_aupr
    return best_val_aupr


def run_training(model, train_loader, val_loader, device, local_rank, outdir, class_weights, args):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    raw_model = model

    # ============ PHASE 1: frozen backbone, train head ============
    if is_main_process():
        print("=" * 60 + "\nPHASE 1: training head with frozen backbone\n" + "=" * 60)
    raw_model.freeze_backbone()
    if is_distributed():
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)

    head_params = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=args.lr_phase1, weight_decay=args.weight_decay_phase1)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.phase1_epochs, 1))
    best_val_aupr = run_phase(1, model, raw_model, optimizer, train_loader, val_loader, criterion, device,
                              outdir, args.phase1_epochs, args.patience_phase1, best_val_aupr=0.0,
                              scheduler=scheduler)

    # ============ PHASE 2: unfreeze last layers, discriminative LR ============
    if is_main_process():
        print("\n" + "=" * 60 + "\nPHASE 2: finetuning with last layers unfrozen\n" + "=" * 60)
    if is_distributed():
        del model
        torch.cuda.empty_cache()
        dist.barrier()  # wait for rank 0 to finish saving phase 1

    # all ranks continue from the best phase 1 checkpoint
    raw_model.load_state_dict(torch.load(os.path.join(outdir, "best_phase_1.pt"), map_location="cpu"))
    raw_model = raw_model.to(device)
    raw_model.unfreeze_backbone(unfreeze_layers=args.unfreeze_layers)
    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True) if is_distributed() else raw_model

    optimizer = AdamW([
        {"params": raw_model.chrombert.encoder_layers[-args.unfreeze_layers:].parameters(), "lr": args.lr_phase2_backbone},
        {"params": raw_model.res1.parameters(), "lr": args.lr_phase2_head},
        {"params": raw_model.res2.parameters(), "lr": args.lr_phase2_head},
        {"params": raw_model.conv.parameters(), "lr": args.lr_phase2_head},
        {"params": raw_model.res3.parameters(), "lr": args.lr_phase2_head},
        {"params": raw_model.fc.parameters(), "lr": args.lr_phase2_head},
    ], weight_decay=args.weight_decay_phase2)
    # phase 2 only saves if it beats the best phase 1 val AUPR
    run_phase(2, model, raw_model, optimizer, train_loader, val_loader, criterion, device, outdir,
              args.phase2_epochs, args.patience_phase2, best_val_aupr=best_val_aupr,
              epoch_offset=args.phase1_epochs)
    return raw_model, criterion


def best_checkpoint(outdir):
    path = os.path.join(outdir, "best_phase_2.pt")
    if not os.path.exists(path):
        print("  phase 2 never beat phase 1 on val, using best_phase_1.pt")
        path = os.path.join(outdir, "best_phase_1.pt")
    return path


def test_and_save(model, test_loader, criterion, device, outdir):
    """Load the best checkpoint, evaluate on test (rank 0 only), save predictions + metrics."""
    ckpt = best_checkpoint(outdir)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)

    test_loss, test_aupr, names, labels, probs = evaluate_epoch(model, test_loader, criterion, device, gather=False)
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "checkpoint": os.path.basename(ckpt),
        "n_samples": len(labels),
        "loss": test_loss,
        "accuracy": accuracy_score(labels, pred),
        "balanced_accuracy": balanced_accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, zero_division=0),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "auroc": roc_auc_score(labels, probs),
        "auprc": test_aupr,
    }

    print("\n" + "=" * 60 + "\nTEST RESULTS\n" + "=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    pd.DataFrame({"Name": names, "Label": labels, "Predicted_Prob": probs}).to_csv(
        os.path.join(outdir, "phase2_test_predictions.csv"), index=False)
    with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, default=float)
    with open(os.path.join(outdir, "test_results.txt"), "w") as f:
        f.write(f"Test AUPR: {test_aupr:.4f}\n")
    wandb.summary["test/aupr"] = test_aupr
    wandb.summary["test/auroc"] = metrics["auroc"]
    print(f"  Saved predictions + metrics to {outdir}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Finetune ChromBERT for enhancer classification")
    p.add_argument("--dataset", choices=list(ENH_DATASETS), default="c1",
                   help="c1 = C1 enhancers, all = all enhancers")
    p.add_argument("--features", default="all",
                   help="all (cistrome interface) | dna | histone | tf | path to .npy mask")
    p.add_argument("--pretrained", default=CHROMBERT_CKPT, help="pre-trained ChromBERT checkpoint")
    p.add_argument("--track_matrix", default=ENH_TRACKS_5TOK)
    p.add_argument("--labels", default=None, help="labels file. Overrides --dataset")
    p.add_argument("--n_tokens", type=int, default=N_TOKENS)
    p.add_argument("--output_dir", default=None,
                   help="default: ../results/chrombert_finetune/enhancers_<dataset>_<features>")
    p.add_argument("--eval_only", action="store_true", help="just evaluate the best checkpoint in output_dir")
    # training (defaults = final runs)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--phase1_epochs", type=int, default=2)
    p.add_argument("--phase2_epochs", type=int, default=30)
    p.add_argument("--lr_phase1", type=float, default=1e-5)
    p.add_argument("--lr_phase2_backbone", type=float, default=1e-5)
    p.add_argument("--lr_phase2_head", type=float, default=1e-4)
    p.add_argument("--weight_decay_phase1", type=float, default=0.01)
    p.add_argument("--weight_decay_phase2", type=float, default=0.01)
    p.add_argument("--patience_phase1", type=int, default=3)
    p.add_argument("--patience_phase2", type=int, default=7)
    p.add_argument("--unfreeze_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_samples", type=int, default=None, help="use only N samples per split (quick test)")
    # logging
    p.add_argument("--wandb_project", default="chromBERT-finetune")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    if args.labels is None:
        args.labels = ENH_DATASETS[args.dataset]
    args.filter_by = resolve_mask(args.features)
    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_DIR, "chrombert_finetune",
                                       f"enhancers_{args.dataset}_{feature_tag(args.features)}")
    return args


def main():
    args = parse_args()
    device, local_rank = setup_distributed()
    vocab_size, _ = vocab_for(args.n_tokens)
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # ── wandb (rank 0 only) ──
    if is_main_process() and not args.no_wandb:
        wandb.init(project=args.wandb_project,
                   name=f"enh_{args.dataset}_{feature_tag(args.features)}_b{args.batch_size}",
                   config={**vars(args), "vocab_size": vocab_size, "n_heads": N_HEADS, "n_layers": N_LAYERS},
                   dir=outdir)
    else:
        wandb.init(mode="disabled")

    if is_main_process() and not args.eval_only:
        with open(os.path.join(outdir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    # ── Data: one 500bp window per region ──
    dataset = EnhancersDatasetFast(
        track_matrix_path=args.track_matrix,
        labels_file_path=args.labels,
        n_windows=1, stride=500, window_size=500,
        split_file=None,  # splits come from the Split column of the labels file
    )
    splits = dataset.create_splits_from_labels_data()
    if args.max_samples:
        splits = {k: v[:args.max_samples] for k, v in splits.items()}

    train_ds, val_ds, test_ds = (Subset(dataset, splits[k]) for k in ("train", "val", "test"))
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True)
    # test runs on rank 0 only, so no DistributedSampler (it needs the full test set)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ──
    if is_main_process():
        print(f"Loading pre-trained ChromBERT from {args.pretrained}")
    backbone = load_chrombert(args.pretrained, vocab_size)
    model = MultiWindowHeaderFusion(backbone, num_labels=2, dropout=args.dropout, n_windows=1,
                                    filter_by=args.filter_by).to(device)

    class_weights = compute_class_weights(dataset, splits["train"], device)
    if args.eval_only:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        model, criterion = run_training(model, train_loader, val_loader, device, local_rank, outdir,
                                        class_weights, args)

    if is_main_process():
        test_and_save(get_raw_model(model), test_loader, criterion, device, outdir)
    wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
