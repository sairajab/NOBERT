import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
import time
from collections import defaultdict, Counter

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

class EnhancersDatasetFast(Dataset):
 
    def __init__(self, track_matrix_path, labels_file_path, n_windows=3, stride=200, window_size=200, pick_closest_bin=False,pretrain_datapath=None,
                 split_file="/s/chromatin/a/nobackup/Saira/SPLICE_SITE_PREDICTION/src/enhancers_split.csv"):
 
        self.split_file = split_file
        self.window_size = window_size
        self.stride = stride
        self.num_labels = 2
        self.n_windows = n_windows
 
        # ── Load labels ──
        if self.split_file is not None:
            self.labels_data = pd.read_csv(labels_file_path, sep='\t')
        else:
            self.labels_data = pd.read_csv(
                labels_file_path, sep='\t',
                names=['Chromosome', 'Start', 'End', 'Split', 'FeatureType', 'Label']
            )
 
        print(self.labels_data.head())
        print(f"NaN labels: {self.labels_data['Label'].isna().sum()}")
        non_finite = self.labels_data[~np.isfinite(self.labels_data['Label'])]
        if len(non_finite) > 0:
            print(f"Non-finite labels: {len(non_finite)}")
            print(non_finite.head())
        self.labels_data['Label'] = self.labels_data['Label'].astype(int)
 
        # ── Load track matrix ──
        print(f"\nLoading track matrix from {track_matrix_path}...")
        t0 = time.time()
 
        if track_matrix_path.endswith('.parquet'):
            track_df = pd.read_parquet(track_matrix_path)
        else:
            track_df = pd.read_csv(track_matrix_path, index_col=0)
 
        print(f"  Loaded in {time.time()-t0:.1f}s: {track_df.shape}")
        print(track_df.head())
 
        # ── Separate 'actual' column from features ──
        if 'actual' not in track_df.columns:
            raise ValueError(
                "Track matrix must have an 'actual' column. "
                "Found columns: " + str(track_df.columns[:5].tolist())
            )
 
        actual_col = track_df['actual'].values
        window_index = track_df.index.values  # window coordinates
        feature_cols = [c for c in track_df.columns if c != 'actual']
        self.n_features = len(feature_cols)
        self.feature_values = track_df[feature_cols].values.astype(np.float32)
 
        print(f"  Features: {self.n_features}")
        print(f"  Sample: window={window_index[0]} -> actual={actual_col[0]}")
 
        # ── Build lookup: actual_key -> sorted list of (window_start, row_idx) ──
        print("Building lookup...")
        t0 = time.time()
        raw_lookup = defaultdict(list)
 
        for row_idx, (win_key, act_key) in enumerate(zip(window_index, actual_col)):
            win_str = str(win_key)
            try:
                coords = win_str.split(':')[1]
                win_start = int(coords.split('-')[0])
            except (IndexError, ValueError):
                win_start = row_idx  # fallback: preserve insertion order
 
            raw_lookup[str(act_key)].append((win_start, row_idx))
 
        # Sort windows by genomic position within each actual region
        self.actual_to_rows = {}
        for act_key, window_list in raw_lookup.items():
            window_list.sort(key=lambda x: x[0])  # sort by window_start
            self.actual_to_rows[act_key] = [row_idx for _, row_idx in window_list]
 
        # Stats
        window_counts = Counter(len(v) for v in self.actual_to_rows.values())
        print(f"  Built in {time.time()-t0:.2f}s")
        print(f"  Unique regions: {len(self.actual_to_rows):,}")
        print(f"  Windows per region: {dict(sorted(window_counts.items()))}")
 
        # ── Precompute sample -> row indices ──
        print("Precomputing sample indices...")
        self.sample_row_indices = []
        matched = 0
        missing = 0
 
        for i in range(len(self.labels_data)):
            row = self.labels_data.iloc[i]
            chrom = str(row['Chromosome'])
            start = int(row['Start'])
            end = int(row['End'])
            actual_key = f"{chrom}:{start}-{end}"
 
            row_indices = self.actual_to_rows.get(actual_key, [])
            self.sample_row_indices.append(row_indices)
 
            if row_indices:
                matched += 1
            else:
                missing += 1
                if missing <= 5:
                    print(f"  No match: {actual_key}")
 
        print(f"  Matched: {matched:,}/{len(self.labels_data):,} "
              f"({100*matched/len(self.labels_data):.1f}%)")
        if missing > 0:
            print(f"  Missing: {missing:,} ({100*missing/len(self.labels_data):.1f}%)")
 
        # Zero vector for padding
        self._zero = np.zeros(self.n_features, dtype=np.float32)
 
    def __len__(self):
        return len(self.labels_data)
 
    def create_splits_from_labels_data(self):
        splits = {"train": [], "val": [], "test": []}
        for idx in range(len(self.labels_data)):
            row = self.labels_data.iloc[idx]
            split = str(row['Split']).strip().lower()
            if split in ["val", "validation"]:
                split = "val"
            if split in splits:
                splits[split].append(idx)
            else:
                splits["train"].append(idx)
 
        for k, v in splits.items():
            print(f"{k}: {len(v)} samples")
        return splits
 
    def create_splits_from_data(self, split_file=None):
        if split_file is None:
            split_file = self.split_file
 
        split_df = pd.read_csv(split_file)
        seq_to_split = {}
        for _, row in split_df.iterrows():
            seq_to_split[(row['chrom'], row['start'], row['end'])] = row['split']
 
        splits = {"train": [], "val": [], "test": []}
        missed = 0
        for idx in range(len(self.labels_data)):
            data = self.labels_data.iloc[idx]
            seq_key = (data['Chromosome'], data['Start'], data['End'])
            split = seq_to_split.get(seq_key, "train")
            if split in splits:
                splits[split].append(idx)
            else:
                missed += 1
 
        if missed:
            print(f"Warning: {missed} sequences not in split file, assigned to train")
        return splits
 
    def get_labels(self, idx):
        return self.labels_data.iloc[idx]['Label']
 
    def __getitem__(self, idx):
        
        row = self.labels_data.iloc[idx]
        chrom = str(row['Chromosome'])
        start = int(row['Start'])
        end = int(row['End'])
        label = int(row['Label'])
        name = f"{chrom}:{start}-{end}"
 
        row_indices = self.sample_row_indices[idx]
 
        if row_indices:
            windows = [self.feature_values[ri] for ri in row_indices]
            if len(windows) > self.n_windows:
                pick_mid = len(windows) // 2
                windows = [windows[pick_mid]]
                print(f"  Sample {name} has {len(row_indices)} windows, picking middle one")
        else:
            print(f"Warning: No windows for {name} at idx {idx}")
            windows = [self._zero]
 
        data = torch.from_numpy(np.stack(windows))
        label = torch.tensor(label, dtype=torch.long)
 
        return name, data, label

class EnhancersDataset2500(torch.utils.data.Dataset):

    def __init__(self, track_matrix_path, labels_file_path, pretrain_datapath=None,
                 pick_closest_bin=False, stride=500, window_size=500, n_windows=5):

        self.labels_data = pd.read_csv(
            labels_file_path, sep='\t',
            names=['Chromosome', 'Start', 'End', 'Split', 'FeatureType', 'Label']
        )
        #skip Mt and Pt
        self.labels_data = self.labels_data[~self.labels_data['Chromosome'].isin(['ChrMt', 'ChrPt'])]
        self.pick_closest_bin = pick_closest_bin
        if self.pick_closest_bin:
            assert pretrain_datapath is not None, "pretrain_datapath required when pick_closest_bin=True"
            self.track_matrix = pd.read_parquet(pretrain_datapath)
        else:
            self.track_matrix = pd.read_parquet(track_matrix_path)
        print(self.track_matrix.head())
        self.window_size = window_size
        self.stride = stride
        self.n_windows = n_windows
        self.num_labels = 2

    def __len__(self):
        return len(self.labels_data)

    def create_splits_from_data(self):
        splits = {"train": [], "val": [], "test": []}

        for idx in range(len(self.labels_data)):
            row = self.labels_data.iloc[idx]
            split = row['Split'].strip().lower()  

            if split == "val" or split == "validation":
                split = "val"

            if split in splits:
                splits[split].append(idx)
            else:
                splits["train"].append(idx)

        for k, v in splits.items():
            print(f"{k}: {len(v)} samples")

        return splits

    def get_labels(self, idx):
        data = self.labels_data.iloc[idx]
        label = data['Label']
        return label

    def __getitem__(self, idx):
        row = self.labels_data.iloc[idx]
        chrom = row['Chromosome']
        start = row['Start']
        end = row['End']
        label = row['Label']
        dna_seq = ""
        name = f"{chrom}:{start}-{end}"
        if self.pick_closest_bin: #try this again 

            mid_window = (start + end) // 2 #==> 400-900
            bin_start, bin_end = best_bin_fast(chrom, mid_window, bin_size = 500)
            f1 = f"{chrom}:{bin_start}-{bin_end}"
            window = self.track_matrix.loc[f1].values
            data = torch.tensor(window, dtype=torch.float32)
        else:
            windows = []
            for pos in range(start, end, self.stride):
                win_end = pos + self.window_size
                if win_end > end:
                    break
                f1 = f"{chrom}:{pos}-{win_end}"
                mask = self.track_matrix.index.to_series().str.startswith(f1)
                matched = self.track_matrix.loc[mask.values]
                if len(matched) == 0:
                    raise ValueError(f"No match in parquet for window: {f1} (sample idx={idx}, {chrom}:{start}-{end})")
                # if len(matched) > 1:
                #     print(matched.index.tolist())
                #     print(f"Warning: {len(matched)} matches for {f1}, using first")
                windows.append(matched.values[0])

            if len(windows) != self.n_windows:
                # skip samples that don't have the expected number of windows
                ## fill rest with 1s (mask token) to maintain consistent input shape
                while len(windows) < self.n_windows:
                    windows.append(np.ones(self.track_matrix.shape[1]))

                #raise ValueError(f"Expected {self.n_windows} windows, got {len(windows)} for idx={idx} {chrom}:{start}-{end}")
            if self.n_windows == 1:
                # pick middle 

                windows = windows[2] 

            data = torch.tensor(np.stack(windows), dtype=torch.float32)   
        return name, data, label

chrom_sizes = {
    "Chr1": 30427671,
    "Chr2": 19698289,
    "Chr3": 23459830,
    "Chr4": 18585056,
    "Chr5": 26975502,
    "ChrC": 154478,
    "ChrM": 366924
}

def best_bin_fast(chrom, mid_window, bin_size=500):

    chrom_len = chrom_sizes[chrom]

    # clip midpoint to chromosome bounds
    mid_window = max(0, min(mid_window, chrom_len - 1))

    bin_start = (mid_window // bin_size) * bin_size
    bin_end = min(bin_start + bin_size, chrom_len)

    return bin_start, bin_end

def best_overlapping_bin(mid_window, bin_size=500):
    bin_start = (mid_window // bin_size) * bin_size
    return bin_start, bin_start + bin_size

class EnhancersDataset(torch.utils.data.Dataset):

    def __init__(self, track_matrix_path, labels_file_path, stride=200, window_size=200,
                 split_file="/s/chromatin/a/nobackup/Saira/SPLICE_SITE_PREDICTION/src/enhancers_split.csv"):

        self.split_file = split_file
        if self.split_file is not None:
            self.labels_data = pd.read_csv(labels_file_path, sep='\t')

        else:
            self.labels_data = pd.read_csv(labels_file_path, sep='\t', names=['Chromosome', 'Start', 'End', 'Split', 'FeatureType', 'Label'])

        print(self.labels_data.head()) 
        print(self.labels_data["Label"].isna().sum())
        print(self.labels_data[~np.isfinite(self.labels_data["Label"])])
        self.labels_data["Label"] = self.labels_data["Label"].astype(int)
        self.track_matrix = pd.read_parquet(track_matrix_path)
        print(self.track_matrix.head())
        self.window_size = window_size
        # self.track_matrix.index = (
        #     self.track_matrix.index
        #     .str.replace(r'\|.*$', '', regex=True)   # remove suffix
        #     .str.replace(r'^(Chr[^-]+-[^-]+)-(Chr[^-]+-[^-]+)$', r'\2-\1', regex=True)
        # )

        # self.track_matrix.sort_index(inplace=True)
        # print(self.track_matrix.head())
        # self.track_matrix.to_parquet("enhancers_detailed.parquet")

        self.stride = stride

        self.num_labels = 2


    def __len__(self):
        return len(self.labels_data)
    
    
    def create_splits_from_labels_data(self):
        splits = {"train": [], "val": [], "test": []}

        for idx in range(len(self.labels_data)):
            row = self.labels_data.iloc[idx]
            split = row['Split'].strip().lower()  

            if split == "val" or split == "validation":
                split = "val"

            if split in splits:
                splits[split].append(idx)
            else:
                splits["train"].append(idx)

        for k, v in splits.items():
            print(f"{k}: {len(v)} samples")

        return splits


    
    def create_splits_from_data(self, split_file=None):
        if split_file is None:
            split_file = self.split_file

        split_df = pd.read_csv(split_file)
        seq_to_split = {}
        for _, row in split_df.iterrows():
            chrom = row['chrom'] #chrom,start,end,name,score,strand,split

            start = row['start']
            end = row['end']

            seq_id = (chrom, start, end)
            seq_to_split[seq_id] = row['split']
        missed_introns = set()
        splits = {"train": [], "val": [], "test": []}
        for idx in range(len(self.labels_data)):

            data = self.labels_data.iloc[idx]
            chrom = data['Chromosome']
            start = data['Start']
            end = data['End']
            seq_key = (chrom, start, end)
            
            split = seq_to_split.get(seq_key, "train")  # Default to 'train' if not found
            if split in splits:
                splits[split].append(idx)
            else:
                missed_introns.add(seq_key)

        if missed_introns:
            print(f"Warning: The following sequences were not found in the split mapping and were assigned to 'train': {len(missed_introns)}")


        return splits
    
    def get_labels(self, idx):
        data = self.labels_data.iloc[idx]
        label = data['Label']
        return label

    def __getitem__(self, idx):

        data = self.labels_data.iloc[idx]

        chrom = data['Chromosome']
        start = data['Start']
        end = data['End']
        label = data['Label']
        name = f"{chrom}:{start}-{end}"
        windows = []
        #print(f"Processing {chrom}:{start}-{end} with label {label}")
        for pos in range(start, end-self.window_size + self.stride, self.stride):
            f1 = f"{chrom}:{pos}-{pos+self.window_size}"

            if self.window_size + pos > end:
                break

            mask = self.track_matrix.index.to_series().str.startswith(f1)
            #print(f"Looking for {f1}, found {mask.sum()} matches")
            if mask.sum() == 0:
                print(f"Error: No match in parquet for window: {f1} (sample idx={idx}, {chrom}:{start}-{end})")
                return None, None, None
            windows.append(self.track_matrix.loc[mask.values].values[0])

        label = torch.tensor(label, dtype=torch.long)
        data = torch.tensor(np.stack(windows), dtype=torch.float32)
        return name,  data, label