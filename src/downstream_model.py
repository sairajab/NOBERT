import time

import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from dataloaders import EnhancersDataset, EnhancersDataset2500, EnhancersDatasetFast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
from models import (
    ChromBERTModel as ChromBERT,
    MultiWindowHeaderFusion,
    ResidualBlock,
    LayerNorm,
)
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

# ================= CONFIGURATION =================
VOCAB_SIZE = 7 # 6 for older models  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
PAD_TOKEN = 0
MASK_TOKEN = 5

# ================= DATASET CONFIGS =================
# Each entry fully specifies the paths, windowing params, and model n_windows.
DATASET_CONFIGS = {
    "200bp": {
        "dataset_cls":   EnhancersDataset,
        "data_path":     "../../DATA/enhancers/enh_600_2w_disc.csv",
        "labels_path":   "../../DATA/enhancers/enhancers_600_labels.txt",
        "split_file":    None,
        "stride":        200,
        "window_size":   200,
        "n_windows":     3,
        "pretrain_datafile": "out.parquet",
        "pretrain_modelpath": "../../chromBERT_models200_allclusters",
    },
    "2500bp": {
        "dataset_cls":   EnhancersDataset2500,
        "data_path":     "../../DATA/enhancers/Enhancers_500bp_centered_disc.csv",
        "labels_path":   "../../DATA/enhancers/C1_2500_enhancer_dataset.txt",
        "split_file":    None,   # splits come from the labels file's Split column
        "stride":        500,
        "window_size":   500,
        "n_windows":     1,
        "pretrain_datafile": "bins500bp_labels.parquet",
        "pretrain_modelpath": "../../chromBERT_models_all_clusters500_new",
    },
        "2500bp_5emb": {
        "dataset_cls":   EnhancersDataset2500,
        "data_path":     "../../DATA/enhancers/enhances_full_500_5_chrombert_style_final.csv",
        "labels_path":   "../../DATA/enhancers/C1_2500_enhancer_dataset.txt" , # C1_2500_enhancer_dataset.txt",enhancers_2500_labels_no_duplicates
        "split_file":    None,   # splits come from the labels file's Split column
        "stride":        500,
        "window_size":   500,
        "n_windows":     1,
        "pretrain_datafile": "bins500_chrombert.parquet",
        "pretrain_modelpath": "../../chromBERT_models_all_clusters500_5emb",
    },
}


# ──────────────────────────────────────────────────────────────
# Experimental / alternative heads (not used by the default training
# path below, kept for reference).
# ──────────────────────────────────────────────────────────────

class SpatialWindowFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3)
        self.norm = LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, e1, e2, e3):
        # e_i shape: [batch, hidden_dim]
        x = torch.stack([e1, e2, e3], dim=2)
        x = self.spatial_conv(x)      # Shape: [batch, hidden, 1]
        x = x.squeeze(-1)             # Shape: [batch, hidden]
        x = self.norm(x)
        return self.activation(x)


class AttnPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, Seq, H]
        weights = self.w(x).squeeze(-1).softmax(dim=-1)  # [B, Seq]
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class SimpleHeadPrediction(nn.Module):
    def __init__(self, chrombert_model, hidden_dim, num_labels, dropout=0.1, medium_dim=256):
        super().__init__()

        self.chrombert = chrombert_model

        self.attn_pool = AttnPool(hidden_dim * 3)  # if using 3 windows, input dim is hidden_dim * 3
        self.res1 = ResidualBlock(hidden_dim * 3, 1024, dropout=dropout)
        self.res2 = ResidualBlock(1024, hidden_dim, dropout=dropout)
        self.windows = 3
        self.res3 = ResidualBlock(hidden_dim, medium_dim, dropout=dropout)
        self.fc = nn.Linear(medium_dim, num_labels)

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]
        x = x.view(-1, MAX_SEQ_LEN)
        x = self.chrombert(x)[0]   # [batch_size * segments, seq_len, hidden]

        if self.windows != 1:
            x = x.reshape(batch_size, self.windows, MAX_SEQ_LEN, -1).permute(0, 2, 1, 3)
            x = x.reshape(batch_size, MAX_SEQ_LEN, -1)  # batch_size, seq_len, segments * hidden
            x = x.permute(0, 2, 1)
            x = self.res1(x)
            x = self.res2(x)
            x = self.attn_pool(x).squeeze(-1)
        x = self.res3(x)
        out = self.fc(x)

        return out, None

    def freeze_backbone(self):
        for param in self.chrombert.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self, unfreeze_layers=3):
        for param in self.chrombert.encoder_layers[-unfreeze_layers:].parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        weights = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = weights * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ──────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0, scheduler=None, epoch=0, phase=1, cont=False):
    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(train_one_epoch._current_epoch)

    total_loss = 0
    all_preds = []
    all_labels = []
    global_step = epoch * len(loader)

    for i, batch in enumerate(loader):
        s_time = time.time()
        _, input_ids, labels = batch
        if not cont:
            input_ids = input_ids.to(device).long()
        else:
            input_ids = input_ids.to(device).float()
        labels = labels.to(device).long()  # softmax needs long labels

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)[0].squeeze(0)  # keep class dim
            loss = criterion(logits, labels)

        loss.backward()

        # Compute grad norm before clipping — useful for diagnosing instability
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
        probs = torch.softmax(logits, dim=-1).float().detach().cpu().numpy()
        all_preds.append(probs[:, 1])  # positive class probability
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    gathered_preds, gathered_labels = gather_predictions(all_preds, all_labels)
    aupr = average_precision_score(gathered_labels, gathered_preds) if is_main_process() else 0.0

    if is_main_process():
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        print(f"  Train AUPR: {aupr:.4f} | Current LR (Head): {current_lr:.2e}")

    return total_loss / len(loader), aupr


# Static variable to track epoch for sampler
train_one_epoch._current_epoch = 0


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, outdir=None, save_predictions=False,
                    epoch=0, phase=1, split="val", cont=False, gather=True):
    """Run one evaluation pass. `gather=True` performs a distributed all_gather so the
    returned AUPR reflects the full split rather than just this rank's local shard —
    use this for validation, where every rank calls evaluate_epoch together. Pass
    `gather=False` for a single-process (rank-0-only) final test evaluation, where no
    other rank is present to participate in a collective op.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_names = []
    for batch in loader:
        name, input_ids, labels = batch
        if not cont:
            input_ids = input_ids.to(device).long()
        else:
            input_ids = input_ids.to(device).float()
        labels = labels.to(device).long()  # softmax needs long labels
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)[0].squeeze(0)  # keep class dim, shape: [batch, num_classes]
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1).float().detach().cpu().numpy()
        all_preds.append(probs[:, 1])  # probability of positive class
        all_labels.append(labels.float().detach().cpu().numpy())
        all_names.extend(name)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    if gather:
        metric_preds, metric_labels = gather_predictions(all_preds, all_labels)
    else:
        metric_preds, metric_labels = all_preds, all_labels
    aupr = average_precision_score(metric_labels, metric_preds) if is_main_process() else 0.0

    avg_loss = total_loss / len(loader)

    if save_predictions and is_main_process():
        pred_df = pd.DataFrame({
            "Name": all_names,
            "Label": all_labels,
            "Predicted_Prob": all_preds
        })
        outdir_path = outdir if outdir is not None else "."
        os.makedirs(outdir_path, exist_ok=True)
        pred_df.to_csv(os.path.join(outdir_path, f"phase{phase}_{split}_predictions.csv"), index=False)

    if is_main_process():
        wandb.log({
            f"phase{phase}/{split}/loss": avg_loss,
            f"phase{phase}/{split}/aupr": aupr,
            f"phase{phase}/epoch": epoch,
        })

    return avg_loss, aupr


# ──────────────────────────────────────────────────────────────
# Main training driver
# ──────────────────────────────────────────────────────────────

def compute_class_weights(dataset, indices, device):
    """
    Compute inverse-frequency class weights from the training split.
    Returns a [num_classes] tensor suitable for nn.CrossEntropyLoss(weight=...).
    For a 1:10 pos:neg ratio this gives roughly [0.18, 1.82] (normalized to mean=1).
    """
    labels = np.array(dataset.labels_data.iloc[indices]['Label'].values)
    counts = np.bincount(labels)
    if is_main_process():
        print(f"  Class counts in train split: {dict(enumerate(counts))}  "
              f"(ratio neg:pos = {counts[0]/counts[1]:.1f}:1)")
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.mean()           # normalize so mean weight = 1
    return torch.tensor(weights, device=device)


def compute_sample_weights(dataset, indices):
    """
    Per-sample weights for WeightedRandomSampler.
    Minority class samples get upweighted so every batch is ~balanced.
    """
    labels = dataset.labels_data.iloc[indices]['Label'].values
    counts = np.bincount(labels)
    class_weights = 1.0 / counts.astype(np.float32)
    return torch.tensor([class_weights[l] for l in labels], dtype=torch.double)


def run_training(model, train_loader, val_loader, test_loader=None, device='cuda', local_rank=0, outdir=None,
                 class_weights=None, phase1_epochs=2, phase2_epochs=10,
                 lr_phase1=1e-5, lr_phase2_backbone=1e-5, lr_phase2_head=1e-4,
                 weight_decay_phase1=0.01, weight_decay_phase2=0.01,
                 patience_phase1=3, patience_phase2=7, unfreeze_layers=3):

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = get_raw_model(model)

    if is_main_process():
        wandb.watch(raw_model, log="all", log_freq=200)

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

    best_val_aupr = 0
    patience_counter = 0

    for epoch in range(phase1_epochs):
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch._current_epoch = epoch
        epoch_start_time = time.time()

        train_loss, train_aupr = train_one_epoch(
            model, train_loader, optimizer_phase1, criterion, device,
            epoch=epoch, phase=1,
        )

        val_loss, val_aupr = evaluate_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, phase=1, split="val",
        )
        scheduler_phase1.step()

        # --- Early stopping logic (rank 0 decides, then broadcasts) ---
        should_stop = False
        if is_main_process():
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1:2d} | Train AUPR: {train_aupr:.4f} | Val AUPR: {val_aupr:.4f} | Val Loss: {val_loss:.4f}")

            wandb.log({
                "phase1/epoch/train_loss": train_loss,
                "phase1/epoch/train_aupr": train_aupr,
                "phase1/epoch/val_loss": val_loss,
                "phase1/epoch/val_aupr": val_aupr,
                "phase1/epoch/patience": patience_counter,
                "phase1/epoch/epoch_time_s": epoch_time,
                "phase1/epoch": epoch,
            })

            if is_better(val_aupr, best_val_aupr, mode="max"):
                best_val_aupr = val_aupr
                ckpt_path = os.path.join(outdir, 'best_phase_1.pt')
                torch.save(raw_model.state_dict(), ckpt_path)
                patience_counter = 0
                wandb.log({"phase1/best_val_aupr": best_val_aupr})
                artifact = wandb.Artifact("phase1_checkpoint", type="model",
                                          description=f"Phase 1 best (epoch {epoch+1})",
                                          metadata={"epoch": epoch + 1, "val_aupr": val_aupr})
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
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
        print(f"  Best Phase 1 Val AUPR: {best_val_aupr:.4f}")
        wandb.summary["phase1/best_val_aupr"] = best_val_aupr

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

    best_val_aupr_phase2 = best_val_aupr
    patience_counter = 0

    for epoch in range(phase2_epochs):
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch + phase1_epochs)  # offset to avoid same seed as phase 1

        train_one_epoch._current_epoch = epoch + phase1_epochs
        epoch_start_time = time.time()

        train_loss, train_aupr = train_one_epoch(
            model, train_loader, optimizer_phase2, criterion, device,
            max_grad_norm=1.0,
            scheduler=None,
            epoch=epoch,
            phase=2,
        )

        val_loss, val_aupr = evaluate_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, phase=2, split="val", outdir=outdir, save_predictions=False,
        )

        # --- Early stopping logic (rank 0 decides, then broadcasts) ---
        should_stop = False
        if is_main_process():
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch {epoch+1:2d} | Train AUPR: {train_aupr:.4f} | Val AUPR: {val_aupr:.4f} | Val Loss: {val_loss:.4f}")

            wandb.log({
                "phase2/epoch/train_loss": train_loss,
                "phase2/epoch/train_aupr": train_aupr,
                "phase2/epoch/val_loss": val_loss,
                "phase2/epoch/val_aupr": val_aupr,
                "phase2/epoch/patience": patience_counter,
                "phase2/epoch/epoch_time_s": epoch_time,
                "phase2/epoch": epoch,
            })

            if is_better(val_aupr, best_val_aupr_phase2, mode="max"):
                best_val_aupr_phase2 = val_aupr
                ckpt_path = os.path.join(outdir, 'best_phase_2.pt')
                torch.save(raw_model.state_dict(), ckpt_path)
                patience_counter = 0
                wandb.log({"phase2/best_val_aupr": best_val_aupr_phase2})
                artifact = wandb.Artifact("phase2_checkpoint", type="model",
                                          description=f"Phase 2 best (epoch {epoch+1})",
                                          metadata={"epoch": epoch + 1, "val_aupr": val_aupr})
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)
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
        print(f"  Best Phase 2 Val AUPR: {best_val_aupr_phase2:.4f}")
        wandb.summary["phase2/best_val_aupr"] = best_val_aupr_phase2

    # ============================================
    # FINAL EVALUATION (rank 0 only, single process, full test set)
    # ============================================
    if test_loader is not None and is_main_process():
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        raw_model.load_state_dict(torch.load(os.path.join(outdir, 'best_phase_2.pt'), map_location="cpu"))
        raw_model = raw_model.to(device)
        test_loss, test_aupr = evaluate_epoch(raw_model, test_loader, criterion, device,
                                              phase=2, split="test", save_predictions=True,
                                              outdir=outdir, gather=False)
        print(f"  Test AUPR: {test_aupr:.4f}")
        wandb.summary["test/aupr"] = test_aupr
        wandb.summary["test/loss"] = test_loss
        with open(os.path.join(outdir, 'test_results.txt'), 'w') as f:
            f.write(f"Test AUPR: {test_aupr:.4f}\n")

    return raw_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", choices=["200bp", "2500bp"], default="2500bp_5emb")
    parser.add_argument("--pick_closest_bin", action="store_true", default=False)
    parser.add_argument("--pretrain_modelpath", default=None,
                        help="Override the ChromBERT backbone checkpoint directory for the chosen dataset config")
    parser.add_argument("--base_dir", default="/s/chromatin/c/nobackup/Saira/BERTModel/NOBERT/results/chrombert_finetune/")
    parser.add_argument("--output_dir", default=None,
                        help="If not given, auto-named from base_dir/dataset_version/n_windows")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--phase1_epochs", type=int, default=2)
    parser.add_argument("--phase2_epochs", type=int, default=30)
    parser.add_argument("--lr_phase1", type=float, default=1e-5)
    parser.add_argument("--lr_phase2_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_phase2_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay_phase1", type=float, default=0.01)
    parser.add_argument("--weight_decay_phase2", type=float, default=0.01)
    parser.add_argument("--patience_phase1", type=int, default=3)
    parser.add_argument("--patience_phase2", type=int, default=7)
    parser.add_argument("--unfreeze_layers", type=int, default=3)
    parser.add_argument("--use_weighted_sampler", action="store_true", default=False)
    parser.add_argument("--wandb_project", default="chromBERT-finetune")
    parser.add_argument("--filter_by", type=str, default=None,
                        help="Optional filter for the dataset features")
    args = parser.parse_args()

    device, local_rank = setup_distributed()

    cfg = DATASET_CONFIGS[args.dataset_version]
    n_windows = 1  # MultiWindowHeaderFusion is used with a single-window head by default
    pretrain_modelpath = args.pretrain_modelpath or cfg["pretrain_modelpath"]

    if args.output_dir is not None:
        outdir = args.output_dir
    else:
        outdir = f"{args.base_dir}finetune_models_w{n_windows}_{args.dataset_version}_clustered_mtx_c1/"

    if os.path.exists(outdir):
        if is_main_process():
            print(f"Output directory {outdir} already exists. Checkpoint files may be overwritten.")
    else:
        os.makedirs(outdir, exist_ok=True)

    # ── Init WandB (rank 0 only; other ranks get a disabled no-op run) ──
    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=f"{args.dataset_version}_h{N_HEADS}_l{N_LAYERS}_b{args.batch_size}_w{n_windows}",
            config={
                **vars(args),
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "max_seq_len": MAX_SEQ_LEN,
                "n_windows": n_windows,
                "num_labels": 2,
                "criterion": "CrossEntropyLoss",
                "pretrain_modelpath": pretrain_modelpath,
            },
            dir=outdir,
        )
    else:
        # Non-zero DDP ranks: disable wandb entirely to avoid duplicate logs
        wandb.init(mode="disabled")

    # ── Build dataset from config ──
    if cfg["dataset_cls"] is EnhancersDataset:
        dataset_full = EnhancersDatasetFast(
            track_matrix_path=cfg["data_path"],
            labels_file_path=cfg["labels_path"],
            stride=cfg["stride"],
            window_size=cfg["window_size"],
            split_file=cfg["split_file"],
            n_windows=n_windows,
        )
    else:
        dataset_full = EnhancersDatasetFast(
            track_matrix_path=cfg["data_path"],
            labels_file_path=cfg["labels_path"],
            stride=cfg["stride"],
            window_size=cfg["window_size"],
            split_file=None,
            n_windows=n_windows,
            pick_closest_bin=args.pick_closest_bin,
            pretrain_datapath=cfg["pretrain_datafile"],
           
        )
    splits = dataset_full.create_splits_from_labels_data()

    if is_main_process():
        print("Train indices:", len(splits["train"]),
              "Val indices:", len(splits["val"]),
              "Test indices:", len(splits["test"]))

    train_ds = Subset(dataset_full, splits["train"])
    val_ds   = Subset(dataset_full, splits["val"])
    test_ds  = Subset(dataset_full, splits["test"])

    if args.use_weighted_sampler:
        class_weights_for_loss = None   # loss stays unweighted
        sample_weights = compute_sample_weights(dataset_full, splits["train"])
        if is_distributed():
            from dataloaders import DistributedWeightedSampler
            train_sampler = DistributedWeightedSampler(sample_weights, num_samples=len(splits["train"]))
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(splits["train"]), replacement=True
            )
    else:
        class_weights_for_loss = compute_class_weights(dataset_full, splits["train"], device)
        if is_main_process():
            wandb.config.update({"class_weights": class_weights_for_loss.cpu().tolist()})
        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None

    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    # Test set is only ever evaluated on rank 0 as a single process (see run_training),
    # so it deliberately does NOT use a DistributedSampler — every rank that runs the
    # final evaluation needs the full test set, not a shard of it.

    if is_main_process():
        wandb.config.update({"imbalance_strategy": "weighted_sampler" if args.use_weighted_sampler else "class_weighted_loss"})

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ── Load base model ──
    base_model = ChromBERT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS).to(device)

    model_path = f"{pretrain_modelpath}/best_genomic_bert.pt"
    if os.path.exists(model_path):
        if is_main_process():
            print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu")
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(new_state_dict)
    else:
        if is_main_process():
            print(f"No existing model found at {model_path}. Starting fresh.")

    chrombert_model = base_model.to(device)
    print("N windows used in model:", n_windows)
    model = MultiWindowHeaderFusion(chrombert_model, num_labels=2, dropout=0.2, n_windows=n_windows, filter_by=args.filter_by if hasattr(args, "filter_by") else None).to(device)

    model = run_training(
        model, train_loader, val_loader, test_loader, device, local_rank, outdir,
        class_weights=class_weights_for_loss,
        phase1_epochs=args.phase1_epochs, phase2_epochs=args.phase2_epochs,
        lr_phase1=args.lr_phase1, lr_phase2_backbone=args.lr_phase2_backbone, lr_phase2_head=args.lr_phase2_head,
        weight_decay_phase1=args.weight_decay_phase1, weight_decay_phase2=args.weight_decay_phase2,
        patience_phase1=args.patience_phase1, patience_phase2=args.patience_phase2,
        unfreeze_layers=args.unfreeze_layers,
    )

    if is_main_process():
        wandb.finish()

    cleanup_distributed()
