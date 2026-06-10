import pandas as pd
from Bio import SeqIO
from typing import Literal
import numpy as np
from fastprogress import progress_bar
import torch
from Bio.Seq import reverse_complement
import gzip
import os
from torch.amp import autocast
import re
import sys

sys.path.append("/s/chromatin/a/nobackup/ahmed/phenotype")


def read_fasta_file(fasta_path: str, format: str = "fasta"):
    for record in SeqIO.parse(fasta_path, format):
        yield record.upper()


from typing import Optional
import numpy as np


def hot_encode_sequence(
    sequence: str,
    length_after_padding: Optional[int] = 0,
) -> np.ndarray:
    if not hasattr(hot_encode_sequence, "nucleotide_dict"):
        hot_encode_sequence.nucleotide_dict = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
            "U": [0, 0, 0, 1],
            "Y": [0, 0, 1 / 2, 1 / 2],
            "R": [1 / 2, 1 / 2, 0, 0],
            "W": [1 / 2, 0, 0, 1 / 2],
            "S": [0, 1 / 2, 1 / 2, 0],
            "K": [0, 1 / 2, 0, 1 / 2],
            "M": [1 / 2, 0, 1 / 2, 0],
            "D": [1 / 3, 1 / 3, 0, 1 / 3],
            "V": [1 / 3, 1 / 3, 1 / 3, 0],
            "H": [1 / 3, 0, 1 / 3, 1 / 3],
            "B": [0, 1 / 3, 1 / 3, 1 / 3],
            "N": [0, 0, 0, 0],
        }

    if length_after_padding == 0:
        hot_encoded_seq = np.zeros((4, len(sequence)), dtype=np.float32)
        start_pos = 0
        end_pos = start_pos + len(sequence)
    elif length_after_padding < len(sequence):
        hot_encoded_seq = np.zeros((4, length_after_padding), dtype=np.float32)
        sequence = sequence[:length_after_padding]
        start_pos = 0
        end_pos = start_pos + length_after_padding
    else:
        hot_encoded_seq = np.zeros((4, length_after_padding), dtype=np.float32)
        start_pos = (length_after_padding - len(sequence)) // 2
        end_pos = start_pos + len(sequence)

    hot_encoded_seq[:, start_pos:end_pos] = np.array(
        [
            hot_encode_sequence.nucleotide_dict.get(base, [0, 0, 0, 0])
            for base in sequence
        ],
        dtype=np.float32,
    ).T
    return hot_encoded_seq


def create_path(path: str) -> None:
    """
    Creates path if it does not exists
    """
    os.makedirs(name=path, exist_ok=True)


def padSeq(
    seq: str, max_length: int, side: Literal["right", "left"], pad_token: str = "N"
) -> str:
    """
    pad the sequence with 'N' if the current sequence length is smaller than the input seqyence length
    """

    if not len(seq) < max_length:
        print(len(seq), max_length)
        raise ValueError("Sequence padding is not needed")
    if side not in {"right", "left"}:
        raise ValueError("sequence padding is in the either 'right' or 'left'")
    pad_length = max_length - len(seq)
    pad_seq = pad_token * pad_length
    if side == "left":
        seq = pad_seq + seq
    else:
        seq = seq + pad_seq
    return seq


def main():
    results_dir = "/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/numpy"
    create_path(results_dir)
    cultivar_ids = pd.read_csv(
        "/s/chromatin/a/nobackup/ahmed/phenotype/phenotype_data/genome_phenotypes.csv"
    ).Genome_ID.values.tolist()
    ref_genes = pd.read_csv(
        "/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/ref_genes.csv"
    )
    device = "cuda:0"
    reference_start_chrom_pos = {
        1: 0,
        2: 30427672,
        3: 50125962,
        4: 73585793,
        5: 92170850,
    }
    window = 2500 // 2
    OUT_DIR = "/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/embeddings"
    # Avoid iterrows (VERY slow)
    ref_chrom = ref_genes["chrom"].values.astype(int)
    ref_tss = ref_genes["tss"].values.astype(int)
    ref_strand = ref_genes["strand"].values.astype(int)

    # Precompute global indices into dat_mapping
    global_tss_idx = np.array(
        [reference_start_chrom_pos[c] + t for c, t in zip(ref_chrom, ref_tss)],
        dtype=np.int64,
    )
    BATCH_SIZE = 1024
    # Precompile regex
    chr_regex = re.compile(r"Chr(\d+)")
    # genome_dict = dict()
    n_genes = 32201
    gene_dim = 2048
    
    for cultivar_id in progress_bar(cultivar_ids):
        if not os.path.exists(f"{OUT_DIR}/{cultivar_id}.npy"):
            cultivar_npy = np.zeros((n_genes, gene_dim), dtype=np.float16)
            genome = dict()
            with gzip.open(
                f"/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/dat/forwardIndex{cultivar_id}.dat.gz",
                "rt",
            ) as file:
                dat_mapping = file.read().splitlines()
            with gzip.open(
                f"/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/fasta/pseudo{cultivar_id}.fasta.gz",
                "rt",
            ) as file:
                for record in read_fasta_file(file):
                    match = chr_regex.search(record.description)
                    if match:
                        genome[int(match.group(1))] = record.seq
            batch_seqs = []
            batch_indices = []
            fasta_file = f"/s/chromatin/a/nobackup/ahmed/phenotype/cultivars_data/fasta_with_genes/pseudo{cultivar_id}.fasta"
            with open(fasta_file, "w") as f_out:
                for idx in progress_bar(range(n_genes)):
                    chrom = ref_chrom[idx]
                    strand = ref_strand[idx]
                    tss_pos = int(float(dat_mapping[global_tss_idx[idx]]))
                    if strand == 1:
                        seqStart = tss_pos - window
                        seqEnd = tss_pos + window
                    else:
                        seqStart = tss_pos - window + 1
                        seqEnd = tss_pos + window + 1
                    # Clamp
                    if seqStart < 0:
                        seqStart = 0
                        pad_side = "left"
                    elif seqEnd > len(genome[chrom]):
                        seqEnd = len(genome[chrom])
                        pad_side = "right"
                    else:
                        pad_side = None

                    seq = genome[chrom][seqStart:seqEnd]
                    if pad_side:
                        seq = padSeq(seq, window * 2, side=pad_side)
                    if strand == -1:
                        seq = reverse_complement(seq)
                    f_out.write(f">{idx}:{chrom}:{tss_pos}:{strand}\n{seq}\n")


if __name__ == "__main__":
    main()
