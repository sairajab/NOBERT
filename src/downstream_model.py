from cProfile import label
import math
import time

import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from dataloaders import EnhancersDataset, EnhancersDataset2500, EnhancersDatasetFast
#from train_mlm_2 import GenomicTransformer as GenomicTransformerUpdated
from transformers import get_cosine_schedule_with_warmup
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import wandb
from models import ChromBERTModel as ChromBERT
# ================= CONFIGURATION =================
VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
BATCH_SIZE = 16  # Reduced from 32 to fit in 24GB GPU memory
ACCUMULATION_STEPS = 256 // BATCH_SIZE  # = 64, keeps effective batch size at 256
MASK_PROB = 0.15
EPOCHS = 200
LEARNING_RATE = 1e-4
PAD_TOKEN = 0
MASK_TOKEN = 5

# ================= DATASET CONFIGS =================
# Change dataset_version in __main__ to switch datasets.
# Each entry fully specifies the paths, windowing params, and model n_windows.
DATASET_CONFIGS = {
    "200bp": {
        "dataset_cls":   EnhancersDataset,
        "data_path":     "../../DATA/enhancers/enh_600_2w_disc.csv", #parquet",
        "labels_path":   "../../DATA/enhancers/enhancers_600_labels.txt", #"/s/chromatin/a/nobackup/Saira/SATORI_ENHANCEMENTS/satoriv2/enhancers_pos_neg_labels.txt",
        "split_file":    None, #"/s/chromatin/a/nobackup/Saira/SPLICE_SITE_PREDICTION/src/enhancers_split.csv",
        "stride":        200,
        "window_size":   200,
        "n_windows":     3,
        "pretrain_datafile" : "out.parquet",
        "pretrain_modelpath" :"../../chromBERT_models200_allclusters" 
    },
    "2500bp": {
        "dataset_cls":   EnhancersDataset2500,
        "data_path":     "../../DATA/enhancers/Enhancers_500bp_centered_disc.csv",
        "labels_path":   "../../DATA/enhancers/enhancers_2500_labels_no_duplicates.txt", #"enhancers_2500_labels.txt", #
        "split_file":    None,   # splits come from the labels file's Split column
        "stride":        500,
        "window_size":   500,
        "n_windows":     1,
        "pretrain_datafile" : "bins500bp_labels.parquet",
        "pretrain_modelpath" :"../../chromBERT_models_all_clusters500" #"chromBERT_models_500"
    },
}


class CistromeEmbeddingManager(nn.Module):
    def __init__(self, mtx_mask, ignore = False,ignore_index = None) -> None:
        super().__init__()
        assert mtx_mask is not None, "mtx_mask must be specified"
        assert isinstance(mtx_mask, str), "mtx_mask must be a path to a mtx_mask" 
        assert os.path.exists(mtx_mask), f"{mtx_mask} does not exist"
        self.mtx_mask_df = pd.read_csv(mtx_mask, sep='\t', index_col=0)
        self.mtx_mask = torch.tensor(self.mtx_mask_df.values) # (datasets, factors)
        self.gsmid_names = self.mtx_mask_df.index.tolist()
        self.regulator_names = self.mtx_mask_df.columns.tolist()

        if ignore:
            ignore_gsmid_index = ignore_index[0]
            ignore_regulator_index = ignore_index[1]
            print(f"Ignoring {len(ignore_gsmid_index)} cistromes and {len(ignore_regulator_index)} regulators")
            rows_to_keep = torch.tensor([i not in ignore_gsmid_index for i in range(self.mtx_mask.shape[0])])
            cols_to_keep = torch.tensor([j not in ignore_regulator_index for j in range(self.mtx_mask.shape[1])])
            self.mtx_mask=self.mtx_mask[rows_to_keep][:, cols_to_keep]
            self.mtx_mask_df = self.mtx_mask_df.iloc[rows_to_keep.numpy(), cols_to_keep.numpy()]
            self.gsmid_names = self.mtx_mask_df.index.tolist()
            self.regulator_names = self.mtx_mask_df.columns.tolist()

        factor_num = (self.mtx_mask != 0).sum(dim=0)
        self.normalization_factors = factor_num.clamp(min=1)
        normalized_mtx_mask = (self.mtx_mask / self.normalization_factors)
        self.register_buffer('normalized_mtx_mask', normalized_mtx_mask)
        return None 

    def forward(self, x):
        # x: [batch_size, datasets, hidden]
        # self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.device)
        # self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.dtype)
        x = x.transpose(1, 2)
        x = torch.matmul(x, self.normalized_mtx_mask)
        x = x.transpose(1, 2) # [batch_size, factors, hidden]

        return x


    def get_cistrome_embedding(self, x, gsmid):
        assert gsmid in self.gsmid_names, f"{gsmid} not found in GSMID names"
        index = self.gsmid_names.index(gsmid)
        return x[:, index, :]

    def get_regulator_embedding(self, x, regulator):
        # x: [batch_size, datasets, hidden]
        assert regulator in self.regulator_names, f"{regulator} not found in regulator names"
        index = self.regulator_names.index(regulator)
        x = x.transpose(1, 2)
        self.normalized_mtx_mask = self.normalized_mtx_mask.to(x.device)
        extracted_x = torch.matmul(x, self.normalized_mtx_mask[:, index:index+1])
        return extracted_x.transpose(1, 2)

    def get_region_embedding(self, x):
        return x.mean(dim=1)

class SpatialWindowFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3)
        self.norm = LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, e1, e2, e3):
        # e_i shape: [batch, hidden_dim]
        # Stack along a new sequence dimension, then permute for Conv1d
        # Resulting shape: [batch, channels(hidden), sequence(3)]

        x = torch.stack([e1, e2, e3], dim=2) 
        
        # Convolve across the 3 windows
        x = self.spatial_conv(x)      # Shape: [batch, hidden, 1]
        x = x.squeeze(-1)             # Shape: [batch, hidden]
        
        # Stabilize and activate
        x = self.norm(x)
        return self.activation(x)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, num_slots: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        total_dim = hidden_dim * num_slots
        
        # 1. Create PE for the 5 positions
        pe = torch.zeros(num_slots, hidden_dim)
        position = torch.arange(0, num_slots, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe_flat = pe.view(1, 1, total_dim) 
        
        # 3. Register as buffer
        self.register_buffer('pe', pe_flat)

    def forward(self, x):
        """
        x shape: (B, seq_len, hidden_dim * 5)
        """
        # self.pe (1, 1, total_dim) broadcasts over B and seq_len
        return x + self.pe

class AttnPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)
    def forward(self, x):  # x: [B, Seq, H]
        weights = self.w(x).squeeze(-1).softmax(dim=-1)  # [B, Seq]
        return (weights.unsqueeze(-1) * x).sum(dim=1)     # [B, 
    
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
            x = x.reshape(batch_size, self.windows, MAX_SEQ_LEN, -1).permute(0,2,1,3)
            # create three tensors
            x = x.reshape(batch_size, MAX_SEQ_LEN, -1) # batch_size, seq_len, segments * hidden
            #x = self.attn_pool(x).squeeze(-1) # batch_size, hidden

            x = x.permute(0, 2, 1) 
            x = self.res1(x)
            x = self.res2(x)
            #e1, e2 , e3 = x[:, :, 0, :], x[:, :, 1, :], x[:, :, 2, :]
            x = self.attn_pool(x).squeeze(-1)
            #e2 = self.attn_pool(e2).squeeze(-1)
            #e3 = self.attn_pool(e3).squeeze(-1)
            #x = torch.concat([e1, e2, e3], dim=-1) # [batch_size, 3, hidden]
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

      

class MultiWindowHeaderFusion(nn.Module):

    def __init__(self, chrombert_model, num_labels, mtx_mask = "/s/chromatin/c/nobackup/Saira/BERTModel/mtx_mask.txt", n_windows=3,
                 dropout=0.1, medium_dim=256, encoder_first=False):
        super().__init__()
        self.chrombert = chrombert_model
        H = 768 * n_windows
        self.encode_first = encoder_first
        self.windows = n_windows
        if self.windows != 1:
            # (just tagging which window)
            self.pos_embedding = SinusoidalPositionalEncoding(hidden_dim=768, num_slots=n_windows)

        self.interface = CistromeEmbeddingManager(mtx_mask=mtx_mask)
        n_factors = self.interface.normalized_mtx_mask.shape[1]      
        # Shared per-window encoder
        self.res1 = ResidualBlock(n_factors, 1024, dropout=dropout)
        self.res2 = ResidualBlock(1024, H, dropout=dropout)
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, H), stride=(1, 1))
        
        self.activation = nn.ReLU()

        #self.mlp = nn.Sequential(nn.Linear(MAX_SEQ_LEN, medium_dim), nn.ReLU(), nn.Linear(medium_dim, 1))
        

        self.fc = nn.Linear(medium_dim, num_labels)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)
        self.res3 = ResidualBlock(H , medium_dim, dropout=dropout) 

    def _post_transformer(self, x):
        x = self.interface(x)             # [batch, factors, hidden]
        x = x.permute(0, 2, 1)           # [batch, hidden, factors]
        x = self.res1(x)
        x = self.res2(x)
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        return x
    
    # def _mlp_post_transformer(self, x):
    #     #x = self.interface(x)             # [batch, factors, hidden]
    #     x = x.permute(0, 2, 1)           # [batch, hidden, factors]
    #     x = self.mlp(x)                    # [batch, hidden, 1]
    #     x = x.squeeze(-1)                 # [batch, hidden]
    #     return x

    def forward(self, x, attention_mask=None, return_emb=False):

        B, W, L = x.shape
        x = x.view(B * W, L)                          # flatten for BERT
        x = self.chrombert(x)[0]                      # (B*W, L, 768)
        x = x.reshape(B, W, L, -1).permute(0, 2, 1, 3)  # (B, L, W, 768)
        x = x.reshape(B, L, W * 768)                  # (B, L, W*768)

        if self.windows != 1:
            x = self.pos_embedding(x)

        x   = self._post_transformer(x)
        x   = self.res3(x)
        out = self.fc(x)

        if return_emb:
            return out, x
        
        return out, None
    
    def freeze_backbone(self):
        for param in self.chrombert.parameters():
            param.requires_grad = False
        for param in self.interface.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self, unfreeze_layers=3):
        for param in self.chrombert.encoder_layers[-unfreeze_layers:].parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = LayerNorm(out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Sequential()

        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.norm(self.fc2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    


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
# Helpers for distributed training
# ──────────────────────────────────────────────────────────────

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
        # Single GPU fallback
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU] Using device: {device}")
        # how to set gpu id for single GPU case? Assuming 0 for now, but this may need to be configurable

        return device, 1


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def gather_predictions(preds, labels):
    """Gather predictions and labels from all ranks for accurate metric computation."""
    if not is_distributed():
        return preds, labels

    preds_tensor = torch.tensor(preds, device="cuda")
    labels_tensor = torch.tensor(labels, device="cuda")

    world_size = dist.get_world_size()
    gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_preds, preds_tensor)
    dist.all_gather(gathered_labels, labels_tensor)

    return (
        torch.cat(gathered_preds).cpu().numpy(),
        torch.cat(gathered_labels).cpu().numpy(),
    )


# ──────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────

# def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0, scheduler=None):
#     model.train()
#     # Set epoch for DistributedSampler so shuffling differs each epoch
#     if hasattr(loader.sampler, "set_epoch"):
#         loader.sampler.set_epoch(train_one_epoch._current_epoch)

#     total_loss = 0
#     all_preds = []
#     all_labels = []
    
#     for i, batch in enumerate(loader):
#         s_time = time.time()
#         input_ids, labels = batch
#         #input_ids = tuple(f.to(device).long() for f in input_ids)
#         input_ids = input_ids.to(device).long()
#         labels = labels.to(device).float()

#         optimizer.zero_grad()
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             logits = model(input_ids)[0].squeeze(-1).squeeze(0)
#             #loss = criterion(logits, labels)
#             loss = criterion(logits.view(-1), labels.view(-1))

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#         optimizer.step()
        
#         if scheduler is not None:
#             scheduler.step()

#         if i % 100 == 0:
#             print(f"Processing batch {i} /{len(loader)} - batch time: {time.time() - s_time:.4f}s", flush=True)

#     # (Metrics aggregation remains the same)
#         total_loss += loss.item()
#         all_preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
#         all_labels.append(labels.cpu().numpy())

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     # Gather across GPUs for accurate metrics
#     all_preds, all_labels = gather_predictions(all_preds, all_labels)
#     aupr = average_precision_score(all_labels, all_preds) if is_main_process() else 0.0
    
#     # Print the current Learning Rate so you can verify the Warmup is working
#     if is_main_process():
#         current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
#         print(f"  Train AUPR: {aupr:.4f} | Current LR (Head): {current_lr:.2e}")
        
#     return total_loss / len(loader), aupr

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
        
        # logits = logits.float()  # convert to float for logging
        # labels = labels.float()  # convert to float for logging

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
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']

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

    all_preds, all_labels = gather_predictions(all_preds, all_labels)
    aupr = average_precision_score(all_labels, all_preds) if is_main_process() else 0.0

    if is_main_process():
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        print(f"  Train AUPR: {aupr:.4f} | Current LR (Head): {current_lr:.2e}")

    return total_loss / len(loader), aupr

# Static variable to track epoch for sampler
train_one_epoch._current_epoch = 0


@torch.no_grad()
def evaluate(model, loader, criterion, device, save_predicitions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        name, input_ids, labels = batch
        input_ids = input_ids.to(device).long()
        labels = labels.to(device).float()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_ids)[0].squeeze(-1).squeeze(0)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
        all_labels.append(labels.float().detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    all_preds, all_labels = gather_predictions(all_preds, all_labels)
    aupr = average_precision_score(all_labels, all_preds) if is_main_process() else 0.0
    return total_loss / len(loader), aupr

@torch.no_grad()
def evaluate_softmax(model, loader, criterion, device, outdir=None, save_predicitions=False, epoch=0, phase=1, split="val", cont=False):
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

    aupr = average_precision_score(all_labels, all_preds) if is_main_process() else 0.0

    #all_preds, all_labels = gather_predictions(all_preds, all_labels)
    print("Gathering predictions across processes for error analysis...", aupr)
    print(len(all_preds), len(all_labels), len(all_names))
    #save predictions and labels for error analysis
    if save_predicitions and is_main_process():
        pred_df = pd.DataFrame({
            "Name": all_names,
            "Label": all_labels,
            "Predicted_Prob": all_preds
        })
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            pred_df.to_csv(os.path.join(outdir, f"phase{phase}_{split}_predictions.csv"), index=False)
        else:
            pred_df.to_csv(f"phase{phase}_{split}_predictions.csv", index=False)

    avg_loss = total_loss / len(loader)

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

def get_raw_model(model):
    """Unwrap DDP to access the underlying model."""
    return model.module if isinstance(model, DDP) else model


def compute_class_weights(dataset, indices, device):
    """
    Compute inverse-frequency class weights from the training split.
    Returns a [num_classes] tensor suitable for nn.CrossEntropyLoss(weight=...).
    For a 1:10 pos:neg ratio this gives roughly [0.18, 1.82] (normalized to mean=1).
    """
    #labels = [int(dataset[i][1]) for i in indices]
    labels = np.array(dataset.labels_data.iloc[indices]['Label'].values)
    #labels = dataset.all_labels[indices]
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
                 class_weights=None):
    
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
    optimizer_phase1 = AdamW(head_params, lr=1e-5, weight_decay=0.01)
    scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=15)

    best_val_aupr = 0
    patience_counter = 5
    phase1_epochs = 2

    for epoch in range(phase1_epochs):
        # Set epoch for distributed sampler
        if is_distributed() and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch._current_epoch = epoch
        epoch_start_time = time.time()

        train_loss, train_aupr = train_one_epoch(
            model, train_loader, optimizer_phase1, criterion, device,
            epoch=epoch, phase=1,
        )

        val_loss, val_aupr = evaluate_softmax(
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

            if val_aupr > best_val_aupr:
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

            if patience_counter >= 3:
                print("  Early stopping phase 1")
                wandb.log({"phase1/early_stop_epoch": epoch + 1})
                should_stop = True

        if is_distributed():
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

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

    # Synchronize: wait for rank 0 to finish saving the checkpoint
    if is_distributed():
        dist.barrier()

    # ALL ranks load the best phase 1 checkpoint (not just rank 0)
    best_ckpt_path = os.path.join(outdir, 'best_phase_1.pt')
    if is_main_process():
        print("Loading best phase 1 checkpoint for fine-tuning...")

    raw_model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    raw_model = raw_model.to(device)
    raw_model.unfreeze_backbone(unfreeze_layers=3)

    # --- Re-wrap in DDP with the updated model ---
    if is_distributed():
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = raw_model

    from transformers import get_cosine_schedule_with_warmup

    # In run_training, replace the phase 2 optimizer setup:

    raw_model.unfreeze_backbone(unfreeze_layers=2)

    optimizer_phase2 = AdamW([
        {'params': raw_model.chrombert.encoder_layers[-2:].parameters(), 'lr': 1e-5},
        {'params': raw_model.res1.parameters(), 'lr': 1e-4},
        {'params': raw_model.res2.parameters(), 'lr': 1e-4},
        {'params': raw_model.conv.parameters(), 'lr': 1e-4},
        {'params': raw_model.res3.parameters(), 'lr': 1e-4},
        {'params': raw_model.fc.parameters(), 'lr': 1e-4},
    ], weight_decay=0.01)

    # optimizer_phase2 = AdamW((p for p in model.parameters() if p.requires_grad),
    # lr=1e-4,
    # weight_decay=0.0001)

    total_phase2_epochs = 10
    total_steps = len(train_loader) * total_phase2_epochs
    warmup_steps = int(0.1 * total_steps)

    # scheduler_phase2 = get_cosine_schedule_with_warmup(
    #     optimizer_phase2,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )

    best_val_aupr_phase2 = best_val_aupr
    patience_counter = 0

    for epoch in range(total_phase2_epochs):
        # Set epoch for distributed sampler
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

        val_loss, val_aupr = evaluate_softmax(
            model, val_loader, criterion, device,
            epoch=epoch, phase=2, split="val", outdir=outdir, save_predicitions=False,
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

            if val_aupr > best_val_aupr_phase2:
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

            if patience_counter >= 7:
                print("  Early stopping phase 2")
                wandb.log({"phase2/early_stop_epoch": epoch + 1})
                should_stop = True

        # Broadcast early stopping decision — ALL ranks must participate
        if is_distributed():
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            break

    if is_main_process():
        print(f"  Best Phase 2 Val AUPR: {best_val_aupr_phase2:.4f}")
        wandb.summary["phase2/best_val_aupr"] = best_val_aupr_phase2

    # ============================================
    # FINAL EVALUATION
    # ============================================
    if test_loader is not None and is_main_process():
        print("\n" + "=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)
        raw_model.load_state_dict(torch.load(os.path.join(outdir, 'best_phase_2.pt'), map_location="cpu"))
        raw_model = raw_model.to(device)
        test_loss, test_aupr = evaluate_softmax(raw_model, test_loader, criterion, device, phase=2, split="test", save_predicitions=True)
        print(f"  Test AUPR: {test_aupr:.4f}")
        wandb.summary["test/aupr"] = test_aupr
        wandb.summary["test/loss"] = test_loss
        with open(os.path.join(outdir, 'test_results.txt'), 'w') as f:
            f.write(f"Test AUPR: {test_aupr:.4f}\n")

    return raw_model




   
if __name__ == "__main__":

    device, local_rank = setup_distributed()

    dataset_version = "2500bp"   # options: "200bp", "2500bp"
    pick_closest_bin = False
    

    cfg = DATASET_CONFIGS[dataset_version]
    if not pick_closest_bin:
        n_windows = cfg["n_windows"]
        #n_windows = 1
    else:
        n_windows = 1
    
    base_dir = "/s/chromatin/c/nobackup/Saira/BERTModel/NOBERT/results/chrombert_finetune/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    outdir = f"{base_dir}finetune_models_w{n_windows}_{dataset_version}_clustered_mtx/"

    if os.path.exists(outdir):
        if is_main_process():
            print(f"Output directory {outdir} already exists. Checkpoint files may be overwritten.")
    else:
        os.makedirs(outdir, exist_ok=True)

    # ── Init WandB (rank 0 only; other ranks get a disabled no-op run) ──
    if is_main_process():
        wandb.init(
            project="chromBERT-finetune",
            name=f"{dataset_version}_h{N_HEADS}_l{N_LAYERS}_b{BATCH_SIZE}_w{n_windows}",
            config={
                # dataset
                "dataset_version": dataset_version,
                "data_path": cfg["data_path"],
                "stride": cfg["stride"],
                "window_size": cfg["window_size"],
                # model
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "max_seq_len": MAX_SEQ_LEN,
                "n_windows": n_windows,
                "num_labels": 2,
                "dropout": 0.1,
                # training
                "batch_size": BATCH_SIZE,
                "effective_batch_size": BATCH_SIZE,
                "epochs_phase1": 10,
                "epochs_phase2": 20,
                "lr_phase1": 1e-3,
                "lr_phase2_backbone": 1e-5,
                "lr_phase2_head": 1e-4,
                "weight_decay_phase1": 0.001,
                "warmup_fraction_phase2": 0.1,
                "patience_phase1": 5,
                "patience_phase2": 7,
                "max_grad_norm": 1.0,
                "criterion": "CrossEntropyLoss",
                "mask_prob": MASK_PROB,
                "pick_closest_bin": pick_closest_bin,
                "pretrain_datafile": cfg["pretrain_datafile"],
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
            split_file = None,
            n_windows=n_windows,
            pick_closest_bin = pick_closest_bin,
            pretrain_datapath=cfg["pretrain_datafile"],
        )
    print("+++++++++++++++")
    splits = dataset_full.create_splits_from_labels_data() #dataset_full.create_splits_from_data()

    if is_main_process():
        print("Train indices:", len(splits["train"]),
              "Val indices:", len(splits["val"]),
              "Test indices:", len(splits["test"]))

    train_ds = Subset(dataset_full, splits["train"])
    val_ds   = Subset(dataset_full, splits["val"])
    test_ds  = Subset(dataset_full, splits["test"])

    USE_WEIGHTED_SAMPLER = False   

    if USE_WEIGHTED_SAMPLER:
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
        if is_distributed():
            train_sampler = DistributedSampler(train_ds, shuffle=True)
        else:
            train_sampler = None

    val_sampler  = DistributedSampler(val_ds,  shuffle=False) if is_distributed() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed() else None

    if is_main_process():
        wandb.config.update({"imbalance_strategy": "weighted_sampler" if USE_WEIGHTED_SAMPLER else "class_weighted_loss"})

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        sampler=val_sampler,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        sampler=test_sampler,
        num_workers=0, pin_memory=True
    )

    # ── Load base model ──
    base_model_path = cfg["pretrain_modelpath"]

    base_model = ChromBERT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS).to(device)

    if dataset_version == "2500bp":
        base_model = ChromBERT(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS).to(device)

    model_path = f"{base_model_path}/best_genomic_bert.pt"
    if os.path.exists(model_path):
        if is_main_process():
            print(f"Loading model weights from {model_path}...")
            state_dict = torch.load(model_path, map_location="cpu")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

            base_model.load_state_dict(new_state_dict)
    else:
        if is_main_process():
            print(f"No existing model found at {model_path}. Starting fresh.")

    chrombert_model = base_model.to(device)
    print("N windows used in model:", n_windows)
    model = MultiWindowHeaderFusion(chrombert_model, num_labels=2, dropout=0.3, n_windows=n_windows).to(device)
    #model = SimpleHeadPrediction(chrombert_model, hidden_dim=768, num_labels=2).to(device)

    model = run_training(model, train_loader, val_loader, test_loader, device, local_rank, outdir,
                         class_weights=class_weights_for_loss)

    if is_main_process():
        wandb.finish()

    cleanup_distributed()
