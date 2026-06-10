import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os

PAD_TOKEN = 0
MASK_TOKEN = 5
MASK_PROB = 0.15

class ChromBERTDataset(Dataset):
    def __init__(self, parquet_path, mask_mode="dynamic", num_masks=7, cache_dir="mask_cache_200"):
        """
        mask_mode: 
            "dynamic"    - create new structured masks every __getitem__ call
            "precomputed" - precompute num_masks per sample, save to disk, cycle per epoch
            "precomputed_all" - precompute num_masks per sample, num_masks * x dataset size per epoch
        """
        df = pd.read_parquet(parquet_path)
        self.data = df.values.astype(np.int8)
        self.labels = torch.tensor([1, 2, 3, 4])
        self.ids = df.index.tolist()
        self.seq_len = self.data.shape[1]
        self.n_samples = self.data.shape[0]
        self.mask_mode = mask_mode
        self.num_masks = num_masks
        self.current_mask_idx = 0  # for "precomputed" mode epoch cycling

        # Load cluster info
        self.cluster_df = pd.read_csv("../DATA/pretraining/experiment_clusters_pearson.tsv", sep="\t")
        self.experiment_to_cluster = dict(
            zip(self.cluster_df["experiment"], self.cluster_df["cluster"])
        )
        self._build_cluster_index(df.columns.tolist())

        # Precompute if needed
        if mask_mode in ("precomputed", "precomputed_all"):
            self._init_precomputed(cache_dir)

    def _build_cluster_index(self, col_names):
        self.col_to_cluster = {}
        self.cluster_to_cols = {}

        for col_idx, col_name in enumerate(col_names):
            cluster = self.experiment_to_cluster.get(col_name, None)
            if cluster is not None:
                self.col_to_cluster[col_idx] = cluster
                if cluster not in self.cluster_to_cols:
                    self.cluster_to_cols[cluster] = []
                self.cluster_to_cols[cluster].append(col_idx)

        for cluster in self.cluster_to_cols:
            self.cluster_to_cols[cluster] = np.array(self.cluster_to_cols[cluster], dtype=np.int64)

        self.cluster_ids = list(self.cluster_to_cols.keys())
        self.cluster_sizes = np.array([len(self.cluster_to_cols[c]) for c in self.cluster_ids])

        print(f"Found {len(self.cluster_ids)} clusters, "
              f"sizes: min={self.cluster_sizes.min()}, max={self.cluster_sizes.max()}, "
              f"mean={self.cluster_sizes.mean():.1f}")


    def _structured_mask(self, sequence):
        seq = sequence.clone()
        target = sequence.clone()
        S = len(seq)

        non_pad_mask = seq != PAD_TOKEN
        n_maskable = non_pad_mask.sum().item()
        mask_budget = int(n_maskable * MASK_PROB)

        if mask_budget == 0:
            target[:] = -100
            return seq, target

        perm = np.random.permutation(len(self.cluster_ids))
        masked_positions = []
        total_masked = 0

        for ci in perm:
            cluster = self.cluster_ids[ci]
            cols = self.cluster_to_cols[cluster]
            valid_cols = cols[non_pad_mask.numpy()[cols]]

            if len(valid_cols) == 0:
                continue

            if total_masked + len(valid_cols) > mask_budget * 1.2:
                if total_masked > 0:
                    continue

            masked_positions.append(valid_cols)
            total_masked += len(valid_cols)

            if total_masked >= mask_budget:
                break

        if len(masked_positions) == 0:
            target[:] = -100
            return seq, target

        all_masked = np.concatenate(masked_positions)

        if len(all_masked) > mask_budget:
            all_masked = np.random.choice(all_masked, size=mask_budget, replace=False)

        all_masked = torch.from_numpy(all_masked).long()

        # 80/10/10 strategy
        probs = torch.rand(len(all_masked))

        mask_token_mask = probs < 0.8
        seq[all_masked[mask_token_mask]] = MASK_TOKEN

        random_token_mask = (probs >= 0.8) & (probs < 0.9)
        if random_token_mask.any():
            random_labels = self.labels[
                torch.randint(0, len(self.labels), (random_token_mask.sum(),))
            ]
            seq[all_masked[random_token_mask]] = random_labels

        masked_set = torch.zeros(S, dtype=torch.bool)
        masked_set[all_masked] = True
        target[~masked_set] = -100

        return seq, target

    def _init_precomputed(self, cache_dir):
        N, S = self.n_samples, self.seq_len
        os.makedirs(cache_dir, exist_ok=True)
        seq_path = os.path.join(cache_dir, "masked_seqs.npy")
        tgt_path = os.path.join(cache_dir, "targets.npy")

        if os.path.exists(seq_path) and os.path.exists(tgt_path):
            print(f"Loading precomputed masks from {cache_dir}/")
            self.masked_seqs = np.memmap(seq_path, dtype=np.int8, mode="r", shape=(N, self.num_masks, S))
            self.targets = np.memmap(tgt_path, dtype=np.int16, mode="r", shape=(N, self.num_masks, S))
        else:
            print(f"Precomputing {self.num_masks} structured masks per sample...")
            self.masked_seqs = np.memmap(seq_path, dtype=np.int8, mode="w+", shape=(N, self.num_masks, S))
            self.targets = np.memmap(tgt_path, dtype=np.int16, mode="w+", shape=(N, self.num_masks, S))

            for i in range(N):
                seq = torch.from_numpy(self.data[i]).long()
                for m in range(self.num_masks):
                    masked_seq, target = self._structured_mask(seq)
                    self.masked_seqs[i, m] = masked_seq.numpy().astype(np.int8)
                    self.targets[i, m] = target.numpy().astype(np.int16)

                if (i + 1) % 10000 == 0:
                    print(f"  {i+1}/{N} samples done")

            self.masked_seqs.flush()
            self.targets.flush()
            print("Precomputation done and saved to disk.")

    def set_epoch(self, epoch):
        """For precomputed mode: rotate which mask variant to use this epoch."""
        self.current_mask_idx = epoch % self.num_masks

    def __len__(self):
        if self.mask_mode == "precomputed_all":
            return self.n_samples * self.num_masks
        return self.n_samples

    def __getitem__(self, idx):
        if self.mask_mode == "dynamic":
            sequence = torch.from_numpy(self.data[idx]).long()
            return self._structured_mask(sequence)

        elif self.mask_mode == "precomputed":
            m = self.current_mask_idx
            seq = torch.from_numpy(self.masked_seqs[idx, m].copy()).long()
            target = torch.from_numpy(self.targets[idx, m].copy()).long()
            return seq, target

        elif self.mask_mode == "precomputed_all":
            sample_idx = idx // self.num_masks
            mask_idx = idx % self.num_masks
            seq = torch.from_numpy(self.masked_seqs[sample_idx, mask_idx].copy()).long()
            target = torch.from_numpy(self.targets[sample_idx, mask_idx].copy()).long()
            return seq, target
        else:
            seq = torch.from_numpy(self.data[idx]).long()
            target = torch.from_numpy(self.data[idx]).long()
            return seq, target


class ChromBERTDatasetOnFly(Dataset):
    """
    Simpler Dataset that creates new random masks on the fly every __getitem__ call without clusters.
    Useful for quick experiments without needing to precompute and store masks.
    """
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        self.data = df.values.astype(np.int8)
        self.labels = torch.tensor([1, 2, 3, 4])
        self.ids = df.index.tolist()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sequence = torch.from_numpy(self.data[idx]).clone().long()
        target = sequence.clone()

        # Only mask non-PAD positions
        non_pad_mask = sequence != PAD_TOKEN
        probability_matrix = torch.full(sequence.shape, MASK_PROB)
        probability_matrix[~non_pad_mask] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_masked = masked_indices.nonzero(as_tuple=True)[0]

        if len(indices_masked) > 0:
            probs = torch.rand(len(indices_masked))

            # 80% -> Replace with [MASK]
            mask_token_mask = probs < 0.8
            sequence[indices_masked[mask_token_mask]] = MASK_TOKEN

            # 10% -> Replace with random label
            random_token_mask = (probs >= 0.8) & (probs < 0.9)
            if random_token_mask.any():
                random_labels = self.labels[
                    torch.randint(0, len(self.labels), (random_token_mask.sum(),))
                ]
                sequence[indices_masked[random_token_mask]] = random_labels

            # 10% -> Keep original

        target[~masked_indices] = -100
        return sequence, target
    