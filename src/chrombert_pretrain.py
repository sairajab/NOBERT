#!/usr/bin/env python3
"""
chrombert_pretrain.py
=====================
Pre-train ChromBERT on Arabidopsis chromatin data with masked language modeling (MLM).

Input   : 500bp genome bins x 2689 experiments, each a chromatin state token 1-5
          (DATA/pretraining/bins500_chrombert.parquet)
Masking : cluster-structured masking -- whole clusters of correlated experiments are masked
          together, ~15% of tokens, 80/10/10 (MASK / random token / keep).
          Masks are precomputed once (--num_masks per bin) and cached in --cache_dir,
          each epoch sees every bin with all of its masks.
Loss    : focal loss (gamma=2) on masked positions
Splits  : bins_500bp.csv (chrom,start,end,...,split). Bins not in the file go to train.

Defaults reproduce chromBERT_models_all_clusters500_5emb/best_genomic_bert.pt
(8 layers, 8 heads, 768 dim, vocab 7 = PAD + tokens 1-5 + MASK).

Usage
  python chrombert_pretrain.py                              # single GPU
  python chrombert_pretrain.py --build_cache_only           # build mask cache first, then:
  torchrun --nproc_per_node 4 chrombert_pretrain.py         # multi GPU
  python chrombert_pretrain.py --resume                     # continue from out_dir/best_genomic_bert.pt
  python chrombert_pretrain.py --max_bins 2000 --epochs 1 --no_compile --out_dir /tmp/test   # quick test

Output (in --out_dir)
  best_genomic_bert.pt   checkpoint with best val loss (used by finetuning / eval)
  args.json, history.json (train / val loss per epoch)
"""

import os
import json
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

from config import (PRETRAIN_DATA, PRETRAIN_SPLITS, PRETRAIN_CACHE, CLUSTER_FILE, CHROMBERT_DIR,
                    N_TOKENS, vocab_for)
from dataloaders import ChromBERTDataset
from models import ChromBERTModel, load_chrombert, EMBED_DIM, N_HEADS, N_LAYERS
from utils import FocalLoss, setup, cleanup, is_main_process

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    p = argparse.ArgumentParser(description="MLM pre-training of ChromBERT")
    # data
    p.add_argument("--data", default=PRETRAIN_DATA, help="parquet: bins x experiments, tokens 1..n_tokens")
    p.add_argument("--split_file", default=PRETRAIN_SPLITS)
    p.add_argument("--cache_dir", default=PRETRAIN_CACHE, help="precomputed masks (built if missing)")
    p.add_argument("--cluster_file", default=CLUSTER_FILE)
    p.add_argument("--num_masks", type=int, default=10, help="masks per bin, must match an existing cache")
    p.add_argument("--n_tokens", type=int, default=N_TOKENS, help="chromatin state tokens 1..n_tokens")
    p.add_argument("--max_bins", type=int, default=None, help="use only N bins per split (quick test)")
    # model / training
    p.add_argument("--out_dir", default=CHROMBERT_DIR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256, help="per GPU")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.10, help="fraction of steps for linear warmup")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="start from out_dir/best_genomic_bert.pt (weights only, optimizer restarts)")
    p.add_argument("--no_compile", action="store_true", help="skip torch.compile (faster start for tests)")
    p.add_argument("--build_cache_only", action="store_true",
                   help="only build the mask cache and exit (run with plain python before torchrun)")
    return p.parse_args()


def mask_indices(bin_indices, num_masks):
    # with precomputed_all, dataset item i*num_masks + m = bin i with mask m
    return [i * num_masks + m for i in bin_indices for m in range(num_masks)]


def raw_model(model):
    # unwrap torch.compile and DDP to get the plain ChromBERTModel
    model = getattr(model, "_orig_mod", model)
    return model.module if isinstance(model, DDP) else model


def main():
    args = parse_args()
    vocab_size, mask_token = vocab_for(args.n_tokens)

    # ── Distributed setup (works with torchrun and plain python) ──
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        setup(rank, world_size)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    if is_main_process() and not args.build_cache_only:
        print(f"{'Distributed on ' + str(world_size) + ' GPUs' if is_distributed else 'Single device: ' + str(device)}")
        print(f"vocab_size={vocab_size} (PAD=0, tokens 1-{args.n_tokens}, MASK={mask_token})")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    if is_distributed:
        dist.barrier()

    # ── Data ──
    # with several GPUs every rank would build the same cache at once, so build it first with plain python
    if is_distributed and not os.path.exists(os.path.join(args.cache_dir, "masked_seqs.npy")):
        raise SystemExit(f"No mask cache in {args.cache_dir}. Build it first (one process):\n"
                         f"  python chrombert_pretrain.py --build_cache_only --cache_dir {args.cache_dir}")
    dataset = ChromBERTDataset(
        args.data, mask_mode="precomputed_all", num_masks=args.num_masks, cache_dir=args.cache_dir,
        mask_token=mask_token, random_tokens=range(1, args.n_tokens + 1), cluster_file=args.cluster_file,
    )
    if args.build_cache_only:
        print(f"Mask cache ready in {args.cache_dir}")
        return
    splits = dataset.create_splits_from_data(args.split_file)
    if args.max_bins:
        splits = {k: v[:args.max_bins] for k, v in splits.items()}
    train_idx = mask_indices(splits["train"], args.num_masks)
    val_idx = mask_indices(splits["val"], args.num_masks)

    if is_main_process():
        print(f"Bins: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
        print(f"Masked sequences: train {len(train_idx)}, val {len(val_idx)}")
        print(f"Effective batch size: {args.batch_size * args.accum_steps * world_size}")

    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None, **loader_kw)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **loader_kw)

    # ── Model ──
    ckpt_path = os.path.join(args.out_dir, "best_genomic_bert.pt")
    if args.resume:
        if is_main_process():
            print(f"Resuming from {ckpt_path}")
        model = load_chrombert(ckpt_path, vocab_size)
    else:
        model = ChromBERTModel(vocab_size, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=True)
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FocalLoss(gamma=2.0, ignore_index=-100)

    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup * total_steps),
                                                num_training_steps=total_steps)
    if not args.no_compile:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    # ── Train ──
    best_val_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        for bidx, (seq, target) in enumerate(train_loader):
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            is_update_step = (bidx + 1) % args.accum_steps == 0 or (bidx + 1) == len(train_loader)
            # skip DDP gradient sync on accumulation steps
            sync_context = model.no_sync() if (not is_update_step and hasattr(model, "no_sync")) else nullcontext()

            with sync_context:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, logits = model(seq)  # (B, S, V)
                    loss = criterion(logits.view(-1, vocab_size), target.view(-1))
                (loss / args.accum_steps).backward()

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += loss.item()
            if is_main_process():
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {bidx+1}/{len(train_loader)} | Loss {loss.item():.4f}",
                      end="\r", flush=True)

        if is_main_process():
            print(f"\nEpoch {epoch+1} training done in {(time.time() - start_time)/60:.2f} min", flush=True)
        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for seq, target in val_loader:
                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, logits = model(seq)
                    total_val_loss += criterion(logits.view(-1, vocab_size), target.view(-1)).item()
        avg_val_loss = total_val_loss / max(1, len(val_loader))

        if is_distributed:
            t = torch.tensor([avg_train_loss, avg_val_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_train_loss, avg_val_loss = (t / world_size).tolist()

        # ── Log + save (rank 0) ──
        if is_main_process():
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)
            history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            with open(os.path.join(args.out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(raw_model(model).state_dict(), ckpt_path)
                print(f"--> Validation improved. Saved {ckpt_path} (loss {best_val_loss:.4f})", flush=True)

        if is_distributed:
            dist.barrier()

    if is_distributed:
        cleanup()


if __name__ == "__main__":
    main()
