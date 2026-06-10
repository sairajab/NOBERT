import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
