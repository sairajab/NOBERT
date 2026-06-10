import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from contextlib import nullcontext
import torch.backends.cuda
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from dataloaders import ChromBERTDataset
from models import ChromBERTModel
from utils import FocalLoss, setup, cleanup, is_main_process, load_splits_data
# ================= CONFIGURATION =================
VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
BATCH_SIZE = 256  # Reduced from 32 to fit in 24GB GPU memory
ACCUMULATION_STEPS = 256 // BATCH_SIZE  # = 64, keeps effective batch size at 256
MASK_PROB = 0.15
EPOCHS = 200
LEARNING_RATE = 1e-4
PAD_TOKEN = 0
MASK_TOKEN = 5



def run_training(rank, world_size):

    
    print("Flash SDP enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("PyTorch version:", torch.__version__)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False) 

    is_distributed = world_size > 1

    if is_distributed:
        setup(rank, world_size)
        device = rank
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running in Single-GPU mode on {device}")

    # Load dataset
    # 500bp bins // out.parquet for 200bps 
    #dataset = GenomicBERTDataset("out.parquet", mask_mode = "vae")
    datapath = "../DATA"
    dataset = ChromBERTDataset(os.path.join(datapath, "pretraining", "out.parquet"), mask_mode="precomputed_all", num_masks=7, cache_dir="mask_cache_200")

    out_dir = "../chromBERT_models200_allclusters"
    if is_main_process() and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if is_distributed:
        dist.barrier()
    


    train_indices, val_indices, test_indices = load_splits_data(
        dataset, os.path.join(datapath, "ftrain_idxs.txt"), os.path.join(datapath,"fval_idxs.txt"), os.path.join(datapath, "ftest_idxs.txt")
    )
    if dataset.mask_mode == "precomputed_all":
        num_masks = dataset.num_masks
        train_indices = [i * num_masks + m for i in train_indices for m in range(num_masks)]
        val_indices = [i * num_masks + m for i in val_indices for m in range(num_masks)]
        test_indices = [i * num_masks + m for i in test_indices for m in range(num_masks)]

    if is_main_process():
        print(f"Total Data: {len(dataset)}")
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"Batch size per GPU: {BATCH_SIZE}, Accumulation steps: {ACCUMULATION_STEPS}")
        print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS * world_size}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    if is_distributed:
        train_sampler = DistributedSampler(
            train_subset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_subset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_loader_shuffle = True

    train_dataloader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_loader_shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- Model Setup ---
    model = ChromBERTModel(
        VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=True
    ).to(device)

    model_path = f"{out_dir}/chrombert_model.pt"
    if os.path.exists(model_path):
        # if is_main_process():
        #     print(f"Loading existing model from {model_path}...")
        # state_dict = torch.load(model_path, map_location=device)
        # model = GenomicTransformer(
        #     VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=True
        # ).to(device)
        if is_main_process():
            print(f"Loading existing model from {model_path}...")
            state_dict = torch.load(model_path, map_location=device)

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict)
    
    
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = FocalLoss(gamma=2.0, ignore_index=-100)

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- AMP GradScaler for mixed precision training ---
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # --- TRAINING ---
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        for bidx, (seq, target) in enumerate(train_dataloader):
            batch_time_start = time.time()
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            is_update_step = (
                (bidx + 1) % ACCUMULATION_STEPS == 0
                or (bidx + 1) == len(train_dataloader)
            )

            sync_context = (
                model.no_sync()
                if (not is_update_step and hasattr(model, "no_sync"))
                else nullcontext()
            )

            with sync_context:
                # --- Mixed precision forward pass ---
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, output, _, _ = model(seq)  # (B, S, V)
                    loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))

                unscaled_loss = loss.item()
                loss = loss / ACCUMULATION_STEPS

                # --- Scaled backward pass ---
                loss.backward()

            if is_update_step:
                optimizer.step()  # no scaler
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += unscaled_loss

            if is_main_process():
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Batch {bidx+1}/{len(train_dataloader)} | Loss {unscaled_loss:.4f}",
                    end="\r",
                    flush=True,
                )


        if is_main_process():
            print(
                f"\nEpoch {epoch+1} training completed in {(time.time() - start_time)/60:.2f} minutes.",
                flush=True,
            )
            

        avg_train_loss = total_train_loss / len(train_dataloader)

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for seq, target in val_dataloader:
                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, output, _, _ = model(seq)
                    loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_dataloader))

        if is_distributed:
            t = torch.tensor(
                [avg_train_loss, avg_val_loss], device=device, dtype=torch.float32
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
            avg_train_loss, avg_val_loss = float(t[0].item()), float(t[1].item())

        # --- LOGGING & SAVING (Rank 0 only) ---
        if is_main_process():
            print(
                f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}",
                flush=True,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(
                    f"--> Validation improved. Saving best model (Loss: {best_val_loss:.4f})",
                    flush=True,
                )

                model_to_save = model.module if is_distributed else model
                torch.save(
                    model_to_save.state_dict(), f"{out_dir}/chrombert_model.pt"
                )

        if is_distributed:
            dist.barrier()

    # Cleanup
    if is_distributed:
        cleanup()


if __name__ == "__main__":
    # Compatible with both `torchrun` and `python` launch
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        if world_size > 1:
            print(f"Launching Distributed Training on {world_size} GPUs...")
        else:
            print("Launching Single-GPU Training...")

    run_training(rank, world_size)