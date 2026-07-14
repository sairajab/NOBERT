import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed():
    return dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched with torchrun / multiple GPUs."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[DDP] Rank {rank}/{world_size}, local_rank {local_rank}")
        return device, local_rank
    else:
        # Single GPU / CPU fallback
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU] Using device: {device}")
        return device, 0


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def get_raw_model(model):
    """Unwrap DDP to access the underlying model."""
    return model.module if isinstance(model, DDP) else model


def gather_predictions(preds, labels):
    """Gather predictions and labels (numpy arrays) from all ranks onto every rank.

    Used before computing metrics that must reflect the full split (e.g. AUPR,
    Pearson correlation) rather than just the local shard produced by
    DistributedSampler.
    """
    if not is_distributed():
        return preds, labels

    preds_tensor = torch.as_tensor(preds, device="cuda")
    labels_tensor = torch.as_tensor(labels, device="cuda")

    world_size = dist.get_world_size()
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    return (
        torch.cat(gathered_preds).cpu().numpy(),
        torch.cat(gathered_labels).cpu().numpy(),
    )


def is_better(candidate, best, mode="max"):
    """Single comparison function for checkpoint / early-stop decisions.

    mode="max": higher is better (e.g. AUPR, Pearson correlation).
    mode="min": lower is better (e.g. loss).
    """
    assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode}"
    return candidate > best if mode == "max" else candidate < best


def broadcast_stop(should_stop, device):
    """Rank 0 decides early stopping; broadcast the decision to all ranks."""
    if not is_distributed():
        return should_stop
    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
    dist.broadcast(stop_tensor, src=0)
    return stop_tensor.item() == 1
