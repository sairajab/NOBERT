import os
import time
import math
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ================= CONFIGURATION =================
MAX_SEQ_LEN = 2689
EMBED_DIM = 128
HIDDEN_DIM = 512
LATENT_DIM = 512
BATCH_SIZE = 256
ACCUMULATION_STEPS = 256 // BATCH_SIZE
EPOCHS = 200
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# Loss weighting: total_loss = MSE + CORR_WEIGHT * (1 - pearson_corr)
# Correlation loss encourages the model to preserve relative ordering,
# not just minimize squared error (which can collapse to predicting the mean)
CORR_WEIGHT = 0.1

# Denoising: fraction of positions to zero out in input (0.0 = off)
INPUT_NOISE_STD = 0.0  # Gaussian noise std added to input (0.0 = off)


# ================= SETUP UTILS =================
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


# ================= DATASET =================
class ContinuousAEDataset(Dataset):
    """
    Dataset for continuous signal autoencoder.
    Applies log1p + z-score normalization.
    Stores a binary mask for which positions were originally non-zero / non-pad.
    """

    def __init__(self, parquet_path, input_noise_std=0.0):
        df = pd.read_parquet(parquet_path)
        self.ids = df.index.tolist()
        self.n_samples = df.shape[0]
        self.seq_len = df.shape[1]

        raw = df.values.astype(np.float32)

        # Build a binary mask: True where original data is present (non-zero)
        # Adjust this if your PAD convention is different
        self.pad_mask = (raw == 0).astype(np.bool_)  # True = PAD

        # Normalize: log1p then z-score per feature (column)
        X_log = np.log1p(raw)
        self.mean = X_log.mean(axis=0, keepdims=True)
        self.std = X_log.std(axis=0, keepdims=True) + 1e-8
        self.data = ((X_log - self.mean) / self.std).astype(np.float32)

        # Zero out PAD positions in normalized data too
        self.data[self.pad_mask] = 0.0

        self.input_noise_std = input_noise_std
        self.training_mode = False

        print(f"Loaded continuous dataset: {self.n_samples} samples x {self.seq_len} features")
        print(f"Non-pad fraction: {(~self.pad_mask).mean():.4f}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.data[idx].copy())       # (S,) float32
        mask = torch.from_numpy(~self.pad_mask[idx])              # (S,) bool, True = valid

        if self.training_mode and self.input_noise_std > 0:
            noisy = sequence.clone()
            noise = torch.randn_like(noisy) * self.input_noise_std
            noisy[mask] += noise[mask]  # only add noise to non-pad
            return noisy, sequence, mask

        return sequence, sequence, mask

    def set_training(self, mode=True):
        self.training_mode = mode


# ================= GELU =================
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ================= CONTINUOUS AUTOENCODER =================
class ContinuousGenomicAE(nn.Module):
    """
    Autoencoder for continuous genomic signals.
    
    Encoder: per-position projection + experiment embedding → MLP → attention pool → bottleneck
    Decoder: latent + experiment embedding → MLP → linear output (1 value per position)
    """

    def __init__(self, input_dim=MAX_SEQ_LEN, embed_dim=EMBED_DIM,
                 latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim

        # Per-position: project scalar input to embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            GELU(),
        )
        self.experiment_embedding = nn.Embedding(input_dim, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
        )

        # Attention pooling
        self.pool_attn = nn.Linear(hidden_dim, 1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
        )

        # Output: predict 1 continuous value per position
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            GELU(),
            nn.Linear(embed_dim, 1),
        )

    def encode(self, x, mask=None):
        """
        x: (B, S) continuous values
        mask: (B, S) bool, True = valid position
        Returns: (B, latent_dim)
        """
        B, S = x.shape
        exp_ids = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)

        # Project each scalar to embed_dim: (B, S) -> (B, S, 1) -> (B, S, D)
        h = self.input_proj(x.unsqueeze(-1))
        h = h + self.experiment_embedding(exp_ids)
        h = self.embed_dropout(h)
        h = self.encoder(h)  # (B, S, H)

        # Attention pool with mask
        weights = self.pool_attn(h).squeeze(-1)  # (B, S)
        if mask is not None:
            weights = weights.masked_fill(~mask, -1e9)
        weights = torch.softmax(weights, dim=-1)
        pooled = torch.einsum('bs,bsh->bh', weights, h)  # (B, H)

        z = self.bottleneck(pooled)
        return z

    def decode(self, z):
        """
        z: (B, latent_dim)
        Returns: (B, S) predicted continuous values
        """
        B = z.size(0)
        S = self.input_dim
        exp_ids = torch.arange(S, device=z.device).unsqueeze(0)
        exp_emb = self.experiment_embedding(exp_ids)  # (1, S, D)

        z_exp = z.unsqueeze(1).expand(-1, S, -1)                         # (B, S, latent)
        h = torch.cat([z_exp, exp_emb.expand(B, -1, -1)], dim=-1)        # (B, S, latent+D)

        h = self.decoder(h)               # (B, S, H)
        out = self.output_head(h)          # (B, S, 1)
        return out.squeeze(-1)             # (B, S)

    def forward(self, x, mask=None):
        z = self.encode(x, mask)
        recon = self.decode(z)
        return recon, z

    @torch.no_grad()
    def get_embeddings(self, x, mask=None):
        self.eval()
        return self.encode(x, mask)


# ================= LOSSES =================
def compute_mse_loss(pred, target, mask):
    """
    MSE loss on valid (non-pad) positions only.
    pred:   (B, S)
    target: (B, S)
    mask:   (B, S) bool, True = valid
    """
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = (pred - target) ** 2
    return diff[mask].mean()


def compute_correlation_loss(pred, target, mask):
    """
    Batch-level Pearson correlation loss: 1 - mean(per-sample correlation).
    Encourages the model to preserve relative ordering, not just minimize MSE.
    """
    B = pred.size(0)
    total_corr = 0.0
    n_valid = 0

    for i in range(B):
        m = mask[i]
        if m.sum() < 2:
            continue
        p = pred[i][m]
        t = target[i][m]

        p_centered = p - p.mean()
        t_centered = t - t.mean()

        num = (p_centered * t_centered).sum()
        den = (p_centered.norm() * t_centered.norm()).clamp(min=1e-8)
        total_corr += num / den
        n_valid += 1

    if n_valid == 0:
        return pred.new_tensor(0.0)

    avg_corr = total_corr / n_valid
    return 1.0 - avg_corr  # loss: lower is better


# ================= METRICS =================
@torch.no_grad()
def compute_metrics(pred, target, mask):
    """
    Compute MSE, R², and Pearson correlation on valid positions.
    Returns dict of scalar metrics.
    """
    if mask.sum() == 0:
        return {"mse": 0.0, "r2": 0.0, "pearson": 0.0}

    p_flat = pred[mask]
    t_flat = target[mask]

    mse = ((p_flat - t_flat) ** 2).mean().item()

    # R²
    ss_res = ((t_flat - p_flat) ** 2).sum()
    ss_tot = ((t_flat - t_flat.mean()) ** 2).sum().clamp(min=1e-8)
    r2 = (1.0 - ss_res / ss_tot).item()

    # Pearson
    p_c = p_flat - p_flat.mean()
    t_c = t_flat - t_flat.mean()
    num = (p_c * t_c).sum()
    den = (p_c.norm() * t_c.norm()).clamp(min=1e-8)
    pearson = (num / den).item()

    return {"mse": mse, "r2": r2, "pearson": pearson}


# ================= DATA SPLITS =================
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


# ================= EMBEDDING EXTRACTION =================
def extract_embeddings(model, dataset, indices, device, batch_size=512):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model.eval()
    all_embeddings = []
    all_ids = [dataset.ids[i] for i in indices]

    with torch.no_grad():
        for seq, _, mask in loader:
            seq = seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model.encode(seq, mask)
            all_embeddings.append(z.float().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0), all_ids


# ================= MAIN TRAINING LOOP =================
def run_training(rank, world_size):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    is_distributed = world_size > 1

    if is_distributed:
        setup(rank, world_size)
        device = rank
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process():
            print(f"Running in Single-GPU mode on {device}")

    # --- Dataset ---
    dataset = ContinuousAEDataset(
        "matrix_continuous_rpgc.parquet",
        input_noise_std=INPUT_NOISE_STD,
    )

    out_dir = "chromBERT_ae_continuous"
    if is_main_process() and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if is_distributed:
        dist.barrier()

    train_indices, val_indices, test_indices = load_splits_data(
        dataset, "ftrain_idxs.txt", "fval_idxs.txt", "ftest_idxs.txt"
    )

    if is_main_process():
        print(f"Total Data: {len(dataset)}")
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"Batch size per GPU: {BATCH_SIZE}, Accumulation steps: {ACCUMULATION_STEPS}")
        print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS * world_size}")
        print(f"Latent dim: {LATENT_DIM}, Hidden dim: {HIDDEN_DIM}, Embed dim: {EMBED_DIM}")
        print(f"Correlation loss weight: {CORR_WEIGHT}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    if is_distributed:
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_loader_shuffle = True

    train_dataloader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, sampler=train_sampler,
        shuffle=train_loader_shuffle, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, sampler=val_sampler,
        shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # --- Model ---
    model = ContinuousGenomicAE(
        input_dim=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    model_path = f"{out_dir}/best_genomic_ae.pt"
    if os.path.exists(model_path):
        if is_main_process():
            print(f"Loading existing model from {model_path}...")
        state_dict = torch.load(model_path, map_location=f"cuda:{device}" if isinstance(device, int) else device)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # ---- TRAINING ----
        model.train()
        dataset.set_training(True)
        total_train_mse = 0.0
        total_train_corr = 0.0
        total_train_loss = 0.0
        start_time = time.time()

        for bidx, (inp, target, mask) in enumerate(train_dataloader):
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

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
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recon, z = model(inp, mask)
                    mse_loss = compute_mse_loss(recon, target, mask)
                    corr_loss = compute_correlation_loss(recon, target, mask)
                    loss = mse_loss + CORR_WEIGHT * corr_loss

                unscaled_loss = loss.item()
                unscaled_mse = mse_loss.item()
                unscaled_corr = corr_loss.item()

                (loss / ACCUMULATION_STEPS).backward()

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += unscaled_loss
            total_train_mse += unscaled_mse
            total_train_corr += unscaled_corr

            if is_main_process():
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Batch {bidx+1}/{len(train_dataloader)} "
                    f"| Loss {unscaled_loss:.4f} (mse={unscaled_mse:.4f} corr={unscaled_corr:.4f})",
                    end="\r", flush=True,
                )

        n_batches = len(train_dataloader)
        avg_train_loss = total_train_loss / n_batches
        avg_train_mse = total_train_mse / n_batches
        avg_train_corr = total_train_corr / n_batches

        if is_main_process():
            elapsed = (time.time() - start_time) / 60
            print(f"\nEpoch {epoch+1} training done in {elapsed:.2f} min.", flush=True)

        # ---- VALIDATION ----
        model.eval()
        dataset.set_training(False)
        total_val_loss = 0.0
        total_val_mse = 0.0
        total_val_r2 = 0.0
        total_val_pearson = 0.0

        with torch.no_grad():
            for seq, target, mask in val_dataloader:
                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recon, z = model(seq, mask)
                    mse_loss = compute_mse_loss(recon, target, mask)
                    corr_loss = compute_correlation_loss(recon, target, mask)
                    loss = mse_loss + CORR_WEIGHT * corr_loss

                total_val_loss += loss.item()

                metrics = compute_metrics(recon.float(), target.float(), mask)
                total_val_mse += metrics["mse"]
                total_val_r2 += metrics["r2"]
                total_val_pearson += metrics["pearson"]

        n_val = max(1, len(val_dataloader))
        avg_val_loss = total_val_loss / n_val
        avg_val_mse = total_val_mse / n_val
        avg_val_r2 = total_val_r2 / n_val
        avg_val_pearson = total_val_pearson / n_val

        # Sync across GPUs
        if is_distributed:
            t = torch.tensor(
                [avg_train_loss, avg_val_loss, avg_val_mse, avg_val_r2, avg_val_pearson],
                device=device, dtype=torch.float32,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
            avg_train_loss = t[0].item()
            avg_val_loss = t[1].item()
            avg_val_mse = t[2].item()
            avg_val_r2 = t[3].item()
            avg_val_pearson = t[4].item()

        # ---- LOGGING & SAVING ----
        if is_main_process():
            print(
                f"Epoch {epoch+1} | "
                f"Train: loss={avg_train_loss:.4f} mse={avg_train_mse:.4f} | "
                f"Val: loss={avg_val_loss:.4f} mse={avg_val_mse:.4f} "
                f"R²={avg_val_r2:.4f} Pearson={avg_val_pearson:.4f}",
                flush=True,
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"--> Val improved. Saving best model (Loss: {best_val_loss:.4f})", flush=True)
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), f"{out_dir}/best_genomic_ae.pt")

            if (epoch + 1) % 10 == 0:
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), f"{out_dir}/ae_epoch_{epoch+1}.pt")

        if is_distributed:
            dist.barrier()

    # ---- EXTRACT & SAVE EMBEDDINGS ----
    if is_main_process():
        print("\nExtracting embeddings from best model...")
        best_model = ContinuousGenomicAE(
            input_dim=MAX_SEQ_LEN, embed_dim=EMBED_DIM,
            latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
        ).to(device if not is_distributed else 0)

        state_dict = torch.load(
            f"{out_dir}/best_genomic_ae.pt",
            map_location=device if not is_distributed else "cuda:0",
        )
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        best_model.load_state_dict(new_state_dict)

        # Save normalization params so embeddings can be used downstream
        np.savez(
            f"{out_dir}/norm_params.npz",
            mean=dataset.mean.squeeze(),
            std=dataset.std.squeeze(),
        )

        for split_name, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            if len(indices) == 0:
                continue
            dev = device if not is_distributed else torch.device("cuda:0")
            embeddings, ids = extract_embeddings(best_model, dataset, indices, dev)
            df_emb = pd.DataFrame(embeddings, index=ids)
            df_emb.to_parquet(f"{out_dir}/embeddings_{split_name}.parquet")
            print(f"Saved {split_name} embeddings: {embeddings.shape}")

        print("Done!")

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