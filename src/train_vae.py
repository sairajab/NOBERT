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
from models import ChromVAE
from dataloaders import ChromBERTDataset
from contextlib import nullcontext
import torch.backends.cuda
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= CONFIGURATION =================
VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
BATCH_SIZE = 256  
ACCUMULATION_STEPS = 256 // BATCH_SIZE  # = 64, 
MASK_PROB = 0.15
EPOCHS = 200
LEARNING_RATE = 1e-4
PAD_TOKEN = 0
MASK_TOKEN = 5

# ================= SETUP UTILS =================
def setup(rank, world_size):
    """Initialize the distributed process group using torchrun env vars."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


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


# ================= FOCAL LOSS =================
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must be [N, C], got shape {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"targets must be [N], got shape {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError("logits and targets must have the same N dimension")

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

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()



# ================= MAIN TRAINING LOOP =================
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
    dataset = ChromBERTDataset("out.parquet", mask_mode="vae", num_masks=7, cache_dir="mask_cache_200")

    out_dir = "chromBERT_models200_allclusters"
    if is_main_process() and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if is_distributed:
        dist.barrier()
    


    train_indices, val_indices, test_indices = load_splits_data(
        dataset, "ftrain_idxs.txt", "fval_idxs.txt", "ftest_idxs.txt"
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
    # model = GenomicTransformerV2(
    #     VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=True
    # ).to(device)

    model = ChromVAE(
        vocab_size=VOCAB_SIZE,
        input_dim=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=128).to(device)

    model_path = f"{out_dir}/best_chrom_vae.pt"
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
                    model_to_save.state_dict(), f"{out_dir}/best_chrom_vae.pt"
                )

        if is_distributed:
            dist.barrier()

    # Cleanup
    if is_distributed:
        cleanup()


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        if world_size > 1:
            print(f"Launching Distributed Training on {world_size} GPUs...")
        else:
            print("Launching Single-GPU Training...")

    run_training(rank, world_size)