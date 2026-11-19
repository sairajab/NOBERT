"""
Shared paths and options for the ChromBERT / baseline scripts.
"""

import os

BASE_DIR    = "/s/chromatin/c/nobackup/Saira/BERTModel"
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "results"))

# ---------- Pretraining (500bp genome bins) ----------
PRETRAIN_DATA   = os.path.join(BASE_DIR, "DATA/pretraining/bins500_chrombert.parquet")    # tokens 1-5
PRETRAIN_SPLITS = "/s/chromatin/a/nobackup/Saira/SPLICE_SITE_PREDICTION/src/bins_500bp.csv"
# precomputed masks, 10 per bin. NOTE: the old mask_cache_500_5emb is incomplete (only bins up to
# Chr2:7879000 were ever written, the rest are zeros), so a fresh cache is built here on first run.
PRETRAIN_CACHE  = os.path.join(BASE_DIR, "mask_cache_500_5emb_full")
CLUSTER_FILE    = os.path.join(BASE_DIR, "DATA/pretraining/experiment_clusters_pearson.tsv")
CHROMBERT_DIR   = os.path.join(BASE_DIR, "chromBERT_models_all_clusters500_5emb")
CHROMBERT_CKPT  = os.path.join(CHROMBERT_DIR, "best_genomic_bert.pt")

N_TOKENS = 5  # chromatin state tokens 1..5, so vocab = PAD(0) + 1..5 + MASK(6) = 7


def vocab_for(n_tokens):
    """returns (vocab_size, mask_token) for data with tokens 1..n_tokens"""
    return n_tokens + 2, n_tokens + 1


# ---------- Enhancers (500bp window centered on each region) ----------
ENH_DIR = os.path.join(BASE_DIR, "DATA/enhancers")

ENH_DATASETS = {
    "c1":  os.path.join(ENH_DIR, "C1_2500_enhancer_dataset.txt"),             # C1 enhancers vs background
    "all": os.path.join(ENH_DIR, "enhancers_2500_labels_no_duplicates.txt"),  # all enhancers vs background
}

ENH_TRACKS_4TOK = os.path.join(ENH_DIR, "enhancers_500bp_centered_final.csv")              # used by final_enhancers_* baselines
ENH_TRACKS_5TOK = os.path.join(ENH_DIR, "enhances_full_500_5_chrombert_style_final.csv")   # used by ChromBERT finetuning

FEATURE_MASKS = {
    "all":     None,
    "dna":     "dna_binding_mask.npy",   # 1509 tracks
    "histone": "histone_mask.npy",       # 1014 tracks
    "tf":      "tf_mask.npy",            # 1211 tracks
}


def resolve_mask(features):
    """'all' / 'dna' / 'histone' / 'tf' or a path to a .npy mask -> absolute path (or None)"""
    path = FEATURE_MASKS.get(features, features)
    if path is not None and not os.path.isabs(path):
        path = os.path.join(SRC_DIR, path)
    return path


def feature_tag(features):
    """short name for output dirs, e.g. 'dna' or 'new_dna_mask'"""
    return os.path.splitext(os.path.basename(features))[0]
