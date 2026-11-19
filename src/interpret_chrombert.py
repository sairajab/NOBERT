"""
Rank regulators by how much their ChromBERT embedding shifts between active
(Label=1) and inactive (Label=0) enhancer regions from a single labels file.

For each region group: look up the region's precomputed discretized-signal row
(one token per cistrome/experiment, aligned to mtx_mask's row order), run it
through the ChromBERT backbone, project the per-cistrome hidden states onto
per-regulator embeddings via CistromeEmbeddingManager, and mean-pool over all
regions in the group. Regulators are then ranked by 1 - cosine_similarity
between the two groups' pooled embeddings: a bigger shift means that
regulator's chromatin context differs more between active and inactive regions.
"""

import os
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from models import ChromBERTModel, CistromeEmbeddingManager, MultiWindowHeaderFusion, PAD_TOKEN

VOCAB_SIZE = 7
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8

DEFAULT_PRETRAIN_CKPT = "/s/chromatin/c/nobackup/Saira/BERTModel/chromBERT_models_all_clusters500_new/best_genomic_bert.pt"
DEFAULT_MTX_MASK = "/s/chromatin/c/nobackup/Saira/BERTModel/mtx_mask.txt"
DEFAULT_FT_CKPT = ("/s/chromatin/c/nobackup/Saira/BERTModel/NOBERT/results/chrombert_finetune/"
                    "finetune_models_w1_2500bp_clustered_mtx_c1/best_phase_2.pt")


def load_chrombert_backbone(ckpt_path, device):
    """Load a ChromBERTModel backbone from a checkpoint.

    Accepts either a raw pretraining checkpoint (state dict matching
    ChromBERTModel directly) or a finetuned MultiWindowHeaderFusion checkpoint
    (state dict with a "chrombert." prefix on the backbone's keys) -- either
    way we only want the backbone weights for generating embeddings.
    """

    print("loading from ", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if any(k.startswith("chrombert.") for k in state_dict):
        state_dict = {
            k[len("chrombert."):]: v
            for k, v in state_dict.items() if k.startswith("chrombert.")
        }

    model = ChromBERTModel(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_enhancer_labels(labels_file):
    """Read a tab-separated enhancer activity/labels file.

    Accepts either a header row (columns include Chromosome, Start, End, Label)
    or the headerless 6-column convention used elsewhere in this project
    (Chromosome, Start, End, Split, FeatureType, Label -- see DATASET_CONFIGS
    in chrombert_finetune_enhancers.py).
    """
    with open(labels_file) as f:
        second_field = f.readline().split("\t")[1].strip()
    has_header = not second_field.lstrip("-").isdigit()

    if has_header:
        df = pd.read_csv(labels_file, sep="\t")
    else:
        df = pd.read_csv(labels_file, sep="\t",
                          names=["Chromosome", "Start", "End", "Split", "FeatureType", "Label"])
    
    df = df[df["Split"] == "Test"]
    print("Using Test set only", df.shape)

    required = {"Chromosome", "Start", "End", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels file {labels_file} is missing required columns: {sorted(missing)}")

    df = df.rename(columns={"Chromosome": "chrom", "Start": "start", "End": "end"})
    df["Label"] = df["Label"].astype(int)
    return df


def split_regions_by_label(labels_df):
    """Split a labels DataFrame into (Label=1, Label=0) region groups."""
    cols = ["chrom", "start", "end"]
    region1 = labels_df.loc[labels_df["Label"] == 1, cols].reset_index(drop=True)
    region0 = labels_df.loc[labels_df["Label"] == 0, cols].reset_index(drop=True)
    return region1, region0


class RegionTokenDataset(Dataset):
    """Looks up each region's precomputed token row in a discretized track matrix.

    `regions` is a DataFrame with "chrom", "start", "end" columns. `track_df`
    is indexed by "chrom:start-end" strings, with one column per
    cistrome/experiment.
    """

    def __init__(self, regions, track_df, cistrome_names, name="regions", labels=None):
        regions = regions.copy()
        regions["mid"] = (regions["start"] + regions["end"]) // 2
        keys = [f"{c}:{s}-{e}" for c, s, e in zip(regions["chrom"], regions["mid"]-250, regions["mid"] + 250)]

        found = [k in track_df.index for k in keys]
        missing = [k for k, ok in zip(keys, found) if not ok]
        if missing:
            print(f"Warning: {len(missing)}/{len(keys)} {name} regions "
                  f"not found in track matrix, skipping. First few: {missing[:5]}")
        self.keys = [k for k, ok in zip(keys, found) if ok]
        if not self.keys:
            raise ValueError(f"None of the {name} regions were found in the track matrix")

        # Reindex columns to cistrome_names (mtx_mask's row order) by name rather
        # than trusting the track matrix's on-disk column order -- position i of
        # the ChromBERT sequence must line up with row i of mtx_mask, or the
        # regulator projection silently mixes up experiments.
        self.tokens = track_df.loc[self.keys, cistrome_names].values.astype(np.int64)

        # Optional per-region ground-truth labels, filtered to line up with the
        # same regions that survived the track-matrix lookup above.
        self.labels = np.asarray(labels)[np.array(found, dtype=bool)] if labels is not None else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(self.tokens[idx])
        if self.labels is None:
            return tokens
        return tokens, int(self.labels[idx])


@torch.no_grad()
def embed_regulator_group(dataloader, chrombert_model, interface, device):
    """Mean-pool per-regulator embeddings over every region in a dataloader."""
    total = None
    n_regions = 0
    for batch in dataloader:
        batch = batch.to(device).long()
        hidden, _ = chrombert_model(batch)        # (B, L=n_cistromes, 768)
        #factors = interface(hidden)                # (B, n_factors, 768)
        batch_sum = hidden.sum(dim=0)              # (n_factors, 768)
        total = batch_sum if total is None else total + batch_sum
        n_regions += batch.size(0)

    mean_emb = (total / n_regions).cpu().numpy()
    return mean_emb, interface.regulator_names


def factor_rank(emb1, emb2, regulator_names, odir, ignore_regulator=None):
    cos_sim = [
        cosine_similarity(emb1[i].reshape(1, -1), emb2[i].reshape(1, -1))[0, 0]
        for i in range(emb1.shape[0])
    ]
    cos_sim_df = (
        pd.DataFrame({"factors": regulator_names, "similarity": cos_sim})
        .sort_values(by="similarity")
        .reset_index(drop=True)
    )

    if ignore_regulator:
        ignore = set(ignore_regulator.split(";"))
        cos_sim_df = cos_sim_df[~cos_sim_df["factors"].isin(ignore)].reset_index(drop=True)

    cos_sim_df["rank"] = cos_sim_df.index + 1
    cos_sim_df["embedding_shift"] = 1 - cos_sim_df["similarity"]

    os.makedirs(odir, exist_ok=True)
    cos_sim_df.to_csv(f"{odir}/factor_importance_rank.csv", index=False)
    return cos_sim_df


def regulator_effects_rank(track_df, chrombert_model, interface, region1, region2,
                            odir, batch_size, device, ignore_regulator=None):
    
    print(region1.head())
    print(track_df.head())
    ds1 = RegionTokenDataset(region1, track_df, interface.gsmid_names, name="Label=1")
    ds2 = RegionTokenDataset(region2, track_df, interface.gsmid_names, name="Label=0")
    dl1 = DataLoader(ds1, batch_size=batch_size, shuffle=False)
    dl2 = DataLoader(ds2, batch_size=batch_size, shuffle=False)
    print(f"Label=1: {len(ds1)} regions matched, Label=0: {len(ds2)} regions matched")

    emb1, regulator_names = embed_regulator_group(dl1, chrombert_model, interface, device)
    emb2, _ = embed_regulator_group(dl2, chrombert_model, interface, device)
    track_df = track_df.drop(columns=["actual"])
    regulator_names = track_df.columns.tolist()
    return factor_rank(emb1, emb2, regulator_names, odir, ignore_regulator)


def load_finetuned_classifier(ckpt_path, mtx_mask, device, n_windows=1, num_labels=2):
    """Load a full finetuned MultiWindowHeaderFusion (backbone + classification
    head) from a MultiWindowHeaderFusion checkpoint.

    Unlike load_chrombert_backbone (which strips the "chrombert." prefix and
    discards everything else), this keeps the classification head so we can get
    actual predicted labels -- needed to identify true positives.
    """
    print("loading finetuned classifier from ", ckpt_path)
    backbone = ChromBERTModel(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, use_gradient_checkpointing=False)
    model = MultiWindowHeaderFusion(backbone, num_labels=num_labels, mtx_mask=mtx_mask, n_windows=n_windows)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    if model.windows != 1:
        raise ValueError(
            f"integrated_gradients_attribution only supports single-window "
            f"checkpoints (RegionTokenDataset produces one window per region), "
            f"but this checkpoint has n_windows={model.windows}."
        )
    return model


def _classifier_logits_from_embeddings(model, embeds):
    """
    Mirrors MultiWindowHeaderFusion.forward for n_windows=1. Interpolation between embeddings. 
    """
    backbone = model.chrombert
    #x = backbone.embed_dropout(embeds)
    x = embeds
    for layer in backbone.encoder_layers:
        x = layer(x)
    x = model._post_transformer(x)
    x = model.res3(x)
    return model.fc(x)


def integrated_gradients_attribution(model, token_ids, device, target_class=1, steps=50,
                                      baseline_token=1):
    """Integrated Gradients attribution of each cistrome token's contribution to
    the classifier's predicted probability of `target_class`.

    token_ids: (B, L) long tensor, one window per row (L = number of cistromes).
    Returns: (B, L) numpy array of signed per-cistrome attribution scores.
    """
    backbone = model.chrombert
    token_ids = token_ids.to(device).long()
    B, L = token_ids.shape

    positions = torch.arange(L, device=device).unsqueeze(0)
    with torch.no_grad():
        input_embeds = backbone.embedding(token_ids) + backbone.pos_embedding(positions)
        baseline_ids = torch.full_like(token_ids, baseline_token)
        baseline_embeds = backbone.embedding(baseline_ids) + backbone.pos_embedding(positions)
    diff = input_embeds - baseline_embeds

    total_grads = torch.zeros_like(input_embeds)
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = (baseline_embeds + alpha * diff).detach().requires_grad_(True)

        logits = _classifier_logits_from_embeddings(model, interp)
        target_probs = torch.softmax(logits, dim=-1)[:, target_class]

        grads, = torch.autograd.grad(target_probs.sum(), interp)
        total_grads += grads

    avg_grads = total_grads / steps
    attributions = (diff * avg_grads).sum(dim=-1)  # (B, L), summed over embedding dim
    return attributions.detach().cpu().numpy()


@torch.no_grad()
def _predict_labels(model, batch_tokens, pred_threshold=0.5):
    logits, _ = model(batch_tokens.unsqueeze(1))  # add the n_windows=1 dim MultiWindowHeaderFusion expects
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return (probs >= pred_threshold).long()


def compute_true_positive_attributions(model, track_df, labels_df, cistrome_names, device,
                                        batch_size=8, steps=50, pred_threshold=0.5):
    """Run the finetuned classifier over every labeled region, keep only true
    positives (ground-truth Label=1 AND predicted positive), and compute
    Integrated Gradients attributions for each true positive's cistrome tokens.

    Returns (attributions, keys): attributions is (n_true_positives, n_cistromes),
    keys is the list of "chrom:start-end" region keys for those true positives.
    """
    cols = ["chrom", "start", "end"]
    regions = labels_df[cols].reset_index(drop=True)
    labels = labels_df["Label"].reset_index(drop=True).values

    dataset = RegionTokenDataset(regions, track_df, cistrome_names, name="all labeled", labels=labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_attributions = []
    tp_keys = []
    n_true_positive = 0
    n_total = 0
    offset = 0

    for batch_tokens, batch_labels in loader:
        batch_tokens = batch_tokens.to(device).long()
        batch_labels = batch_labels.to(device).long()
        n_total += batch_tokens.size(0)

        preds = _predict_labels(model, batch_tokens, pred_threshold)
        tp_mask = (preds == 1) & (batch_labels == 1)

        batch_keys = dataset.keys[offset:offset + batch_tokens.size(0)]
        offset += batch_tokens.size(0)

        if tp_mask.any():
            tp_tokens = batch_tokens[tp_mask]
            n_true_positive += tp_tokens.size(0)
            tp_keys.extend([k for k, keep in zip(batch_keys, tp_mask.cpu().tolist()) if keep])

            attributions = integrated_gradients_attribution(model, tp_tokens, device, steps=steps)
            all_attributions.append(attributions)

    print(f"{n_true_positive}/{n_total} regions were true positives "
          f"(Label=1, predicted positive at threshold {pred_threshold})")
    if not all_attributions:
        raise ValueError("No true positive regions found -- nothing to attribute.")

    return np.concatenate(all_attributions, axis=0), tp_keys


def true_positive_feature_rank(attributions, cistrome_names, odir):
    """Rank cistromes by mean |IG attribution| across true positive regions and
    save to CSV, mirroring factor_rank's output style.
    """
    mean_attr = attributions.mean(axis=0)
    
    mean_abs_attr = np.abs(attributions).mean(axis=0) # should i do absolute attributions or not??
    

    attr_df = (
        pd.DataFrame({
            "cistrome": cistrome_names,
            "mean_attribution": mean_attr,
            "mean_abs_attribution": mean_abs_attr,
        })
        .sort_values(by="mean_abs_attribution", ascending=False)
        .reset_index(drop=True)
    )
    attr_df["rank"] = attr_df.index + 1

    os.makedirs(odir, exist_ok=True)
    attr_df.to_csv(f"{odir}/true_positive_feature_attribution.csv", index=False)
    return attr_df


def run_true_positive_attribution(args, return_data=True, ig_steps=50, pred_threshold=0.5 ):
    """Identify true positive enhancer regions (Label=1, predicted positive) and
    rank cistromes by Integrated Gradients feature attribution on those regions.
    """
    os.makedirs(args.odir, exist_ok=True)
    results_odir = f"{args.odir}/results"

    device = torch.device(args.device)

    print(f"Loading finetuned classifier from {args.ft_ckpt}")
    model = load_finetuned_classifier(args.ft_ckpt, args.mtx_mask, device)

    print(f"Loading track matrix from {args.track_matrix}...")
    if args.track_matrix.endswith(".parquet"):
        track_df = pd.read_parquet(args.track_matrix)
    else:
        track_df = pd.read_csv(args.track_matrix, index_col=0)
    if "actual" in track_df.columns:
        track_df = track_df.drop(columns=["actual"])

    cistrome_names = model.interface.gsmid_names
    missing_cistromes = [c for c in cistrome_names if c not in track_df.columns]
    if missing_cistromes:
        raise ValueError(
            f"Track matrix is missing {len(missing_cistromes)} cistrome columns required by "
            f"mtx_mask (e.g. {missing_cistromes[:5]}). Make sure --track-matrix was built with "
            f"the same cistrome set as {args.mtx_mask}."
        )

    print(f"Loading enhancer labels from {args.labels_file}...")
    labels_df = load_enhancer_labels(args.labels_file)

    attributions, tp_keys = compute_true_positive_attributions(
        model, track_df, labels_df, cistrome_names, device,
        batch_size=args.batch_size, steps=ig_steps, pred_threshold=pred_threshold,
    )

    attr_df = true_positive_feature_rank(attributions, cistrome_names, results_odir)

    print("Top cistromes by mean |IG attribution| on true positive regions (top 25)")
    print(attr_df.head(n=25))
    print(f"Computed over {len(tp_keys)} true positive regions")
    print(f"Saved to: {results_odir}/true_positive_feature_attribution.csv")

    if return_data:
        return attr_df
    return None


def run_ES(args, return_data=True):
    os.makedirs(args.odir, exist_ok=True)
    results_odir = f"{args.odir}/results"

    device = torch.device(args.device)

    ckpt_path = args.ft_ckpt or args.pretrain_ckpt
    print(f"Loading ChromBERT backbone from {ckpt_path}")
    chrombert_model = load_chrombert_backbone(ckpt_path, device)

    interface = CistromeEmbeddingManager(mtx_mask=args.mtx_mask).to(device)

    print(f"Loading track matrix from {args.track_matrix}...")
    if args.track_matrix.endswith(".parquet"):
        track_df = pd.read_parquet(args.track_matrix)
    else:
        track_df = pd.read_csv(args.track_matrix, index_col=0)

    missing_cistromes = [c for c in interface.gsmid_names if c not in track_df.columns]
    if missing_cistromes:
        raise ValueError(
            f"Track matrix is missing {len(missing_cistromes)} cistrome columns required by "
            f"mtx_mask (e.g. {missing_cistromes[:5]}). Make sure --track-matrix was built with "
            f"the same cistrome set as {args.mtx_mask}."
        )

    print(f"Loading enhancer labels from {args.labels_file}...")
    labels_df = load_enhancer_labels(args.labels_file)
    region1_df, region2_df = split_regions_by_label(labels_df)
    print(f"Loaded {len(labels_df)} labeled regions: "
          f"{len(region1_df)} with Label=1, {len(region2_df)} with Label=0")

    cos_sim_df = regulator_effects_rank(
        track_df, chrombert_model, interface,
        region1_df, region2_df,
        results_odir, args.batch_size, device, args.ignore_regulator,
    )

    print("Identify key regulators across regions (top 25 by embedding shift)")
    print(cos_sim_df.head(n=25))

    print("Finished")
    if args.ft_ckpt is not None:
        print(f"Used fine-tuned ChromBERT checkpoint: {args.ft_ckpt}")
    else:
        print(f"Used pre-trained ChromBERT checkpoint: {args.pretrain_ckpt}")
    print(f"Key regulators across regions saved to: {results_odir}/factor_importance_rank.csv")

    if return_data:
        return cos_sim_df
    return None


def parse_args():
    
    parser = argparse.ArgumentParser( description=(
            "Identify regulators that differ between active (Label=1) and "
            "inactive (Label=0) enhancer regions via embedding shift."
        )
    )

    parser.add_argument("--labels-file",required=True, help=("Tab-separated enhancer activity/labels file " "(Chromosome, Start, End, ..., Label).Label=1 regions become group 1, Label=0 regions become group 2."),)
    parser.add_argument("--track-matrix",required=True,help=("Discretized token matrix (.csv) indexed by 'chrom:start-end', one column per cistrome/experiment."),)
    parser.add_argument("--pretrain-ckpt",default=DEFAULT_PRETRAIN_CKPT,help="Pretrained ChromBERT backbone checkpoint.",)
    parser.add_argument("--ft-ckpt",default=DEFAULT_FT_CKPT,help=("Finetuned checkpoint to use instead of --pretrain-ckpt for generating embeddings."),)
    parser.add_argument("--mtx-mask",default=DEFAULT_MTX_MASK,help="Cistrome-to-regulator mapping file.",)

    parser.add_argument("--ignore-regulator",default=None,help="Regulators to exclude. Use ';' to separate multiple regulators.",)
    parser.add_argument("--odir",default="./output",help="Output directory.",)
    parser.add_argument("--batch-size",type=int,default=8,help="Batch size.",)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu",help="Device to run inference on.",)
    parser.add_argument("--interpretation_method",default="BOTH", help="IG, ES or BOTH",)

    return parser.parse_args()


def interpret_regulator_effects_between_region_groups(args):
    '''
    Identify regulators that differ between active (Label=1) and inactive
    (Label=0) enhancer regions via embedding shift.
    '''
    run_ES(args, return_data=False)




def interpret_true_positive_feature_attribution(args):
    '''
    Rank cistromes by Integrated Gradients feature attribution, computed only on
    true positive enhancer regions (ground-truth Label=1 and predicted positive).
    '''
    pred_threshold = 0.5
    ig_steps = 50
    run_true_positive_attribution(args, return_data=False, ig_steps=ig_steps, pred_threshold=pred_threshold)


if __name__ == "__main__":
    
    args = parse_args()
    
    if args.interpretation_method == "BOTH":
        print("RUNNING ES")
        interpret_regulator_effects_between_region_groups(args)
        print("RUNNING IG")
        interpret_true_positive_feature_attribution(args)
        
    
    elif args.interpretation_method == "ES":
        interpret_regulator_effects_between_region_groups(args)
    else:
        interpret_true_positive_feature_attribution(args)
