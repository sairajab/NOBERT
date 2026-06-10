
import torch 
import torch.nn as nn
from deepplant import build_model, model
import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from train_mlm import *


vocab_size = VOCAB_SIZE

DEEPPLANT_MODEL_DIR = '/s/chromatin/a/nobackup/Saira/Deepplant_for_dPIR/deepplant_model'

# Nucleotide One-Hot Encoding Dictionary
nucleotide_dict = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "U": [0, 0, 0, 1],
    "Y": [0, 0, 1, 1],
    "R": [1, 1, 0, 0],
    "W": [1, 0, 0, 1],
    "S": [0, 1, 1, 0],
    "K": [0, 1, 0, 1],
    "M": [1, 0, 1, 0],
    "D": [1, 1, 0, 1],
    "V": [1, 1, 1, 0],
    "H": [1, 0, 1, 1],
    "B": [0, 1, 1, 1],
    "N": [1, 1, 1, 1],
}

# One-Hot Encoding Function
def nucleotide_to_one_hot(sequence, max_len=2500):
    seq_array = np.zeros((max_len, 4), dtype=np.float32)
    try:
        for i, nucleotide in enumerate(sequence[:max_len]):
            # if i < 950 or i >= 1550:  # Only encode the first 950 and last 950 nucleotides
            #     seq_array[i] = [0, 0, 0, 0]  
            # else:
            seq_array[i] = nucleotide_dict.get(nucleotide, [0, 0, 0, 0])
    except Exception as e:
        print(f"Error processing sequence: {sequence}")
        print(f"Error: {e}")
        return None 
    return seq_array

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




class ChromatinEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(ChromatinEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(2689+1, hidden_dim)  # +1 for padding token

        self.res1 = ResidualBlock(2689, 1024)
        self.res2 = ResidualBlock(1024, hidden_dim)
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, hidden_dim), stride=(1, 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding(torch.arange(x.size(1), device=x.device).unsqueeze(0))

        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        x = self.res1(x)
        x = self.res2(x)

        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)

        return x
    
class ChromatinModel(nn.Module):
    """
    Discrete cistrome features (B, num_features) with values in {0, ..., vocab_size-1}.
    Each feature is an independent track (TF binding, histone mark, etc.).
    
    Approach: embed each feature's discrete value, add a learned feature-identity
    embedding (so feature 0 and feature 1 are distinguishable), flatten, MLP.
    No positional/conv structure since features are unordered and independent.
    """
    def __init__(
        self,
        num_features=2689,
        vocab_size=vocab_size,
        embed_dim=768,
        hidden_dim=768,
        dropout=0.3,
        num_classes=2,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Value embedding: shared across all features (what "level" is this track at)
        self.value_emb = nn.Embedding(vocab_size, embed_dim)
        # Feature embedding: which track is this (learned identity per feature)
        self.feature_emb = nn.Embedding(num_features, embed_dim)
        self.register_buffer("feature_ids", torch.arange(num_features).unsqueeze(0))

        flat_dim = num_features * embed_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def encode_tokens(self, x):
        B = x.size(0)
        h = self.value_emb(x) + self.feature_emb(self.feature_ids.expand(B, -1))
        # Optional: a couple of self-attention or MLP layers over features
        return h   # (B, num_features, embed_dim)

    def forward(self, x):
        # x: (B, num_features) long
        B = x.size(0)
        # Combine value and feature-identity embeddings
        h = self.value_emb(x.unsqueeze(-1)) + self.feature_emb(self.feature_ids.expand(B, -1))
        # (B, num_features, embed_dim) -> flatten
        h = h.reshape(B, -1)
        return self.mlp(h)
        
class ChromatinFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(2689 + 1, hidden_dim)

        # Operate along the hidden dim, not the sequence length.
        # Original code applied res blocks to the seq_len axis after a permute,
        # which only works because seq_len happened to equal 2689. Cleaner to
        # process per-token features.
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)

        # Pool token features into a single vector (used by chromatin-only path)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def token_features(self, x):
        """Return per-token features: (B, L, hidden_dim). Used as KV in cross-attention."""
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)   # (B, L, D)
        h = self.res1(h)
        h = self.res2(h)
        return h


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, zero_init_output=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, context, mask=None):
        B, Lq, _ = x.shape
        Lk = context.size(1)

        Q = self.q_proj(x).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(context).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(context).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)
    


# class CrossAttentionLayer(nn.Module):
#     """One layer of cross-attention + FFN, pre-norm style, zero-init residuals."""
#     def __init__(self, d_model, num_heads, dropout=0.1, ffn_mult=4):
#         super().__init__()
#         d_model = 512

#         self.attn = CrossAttention(d_model, num_heads, dropout, zero_init_output=True)
#         self.norm_q = nn.LayerNorm(d_model)     # norm queries before attn
#         self.norm_kv = nn.LayerNorm(d_model)    # norm context before attn

#         # FFN with zero-init on output, so the whole layer = identity at init
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * ffn_mult),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * ffn_mult, d_model),
#         )
#         nn.init.zeros_(self.ffn[-1].weight)
#         nn.init.zeros_(self.ffn[-1].bias)
#         self.norm_ffn = nn.LayerNorm(d_model)

#     def forward(self, x, context, mask=None):
#         # Cross-attention with residual

#         x = x + self.attn(self.norm_q(x), self.norm_kv(context), mask=mask)
#         # FFN with residual
#         x = x + self.ffn(self.norm_ffn(x))
#         return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, context, context)   # query=x, key=value=context
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x
    

class MultiViewModel(nn.Module):
    def __init__(
        self,
        model1,
        model2,
        hidden_dim=64,
        d_model=2048,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        d_model = 512

        # Stack N cross-attention layers — all share the same chromatin context
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
        )
        self.res3 = ResidualBlock(hidden_dim, 256)
        self.output_head = nn.Linear(256, 2)
        self.seq_to_dmodel = nn.Linear(2048, d_model)
        self.chrom_to_dmodel = nn.Linear(64, d_model)

    def forward(self, input1, input2):
        seq_vec = self.model1(input1)                       # (B, 2048)
        chrom_tokens = self.model2.encode_tokens(input2)   # (B, 2689, 2048)

        # Sequence vector becomes a length-1 query
        x = seq_vec.unsqueeze(1)                            # (B, 1, 2048)
        x = self.seq_to_dmodel(x)
        chrom_tokens = self.chrom_to_dmodel(chrom_tokens)

        # Pass through each cross-attention layer.
        # Each layer re-attends to chromatin, refining the query representation.
        for layer in self.layers:
            x = layer(x, chrom_tokens)

        x = self.final_norm(x).squeeze(1)                   # (B, 2048)

        h = self.shared_mlp(x)
        h = self.res3(h)
        return self.output_head(h)


class GatedMoE(nn.Module):
    def __init__(
        self,
        model1,
        model2,
        seq_dim=2048,
        chrom_dim=768,
        d_model=768,
        num_heads=8,
        num_layers=2,
        hidden_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

        # project each modality into a shared d_model space
        self.seq_to_dmodel = nn.Linear(seq_dim, d_model)
        self.chrom_to_dmodel = nn.Linear(chrom_dim, d_model)

        # chromatin expert: seq query reads over chrom tokens
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # sequence expert: a small transform on the seq representation
        self.seq_expert = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # gate sees both expert summaries -> 2-way soft mix
        self.gate = nn.Linear(2 * d_model, 2)

        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
        )
        self.res3 = ResidualBlock(hidden_dim, 256, dropout)
        self.output_head = nn.Linear(256, 2)

    def forward(self, input1, input2):
        seq_vec = self.model1(input1)                     # (B, seq_dim)
        chrom_tokens = self.model2.encode_tokens(input2)  # (B, 2689, chrom_dim)

        seq = self.seq_to_dmodel(seq_vec)                 # (B, d_model)
        chrom = self.chrom_to_dmodel(chrom_tokens)        # (B, 2689, d_model)

        # --- chromatin expert: seq attends over chrom tokens ---
        x = seq.unsqueeze(1)                              # (B, 1, d_model) query
        for layer in self.layers:
            x = layer(x, chrom)                           # re-attend each layer
        chrom_expert = self.final_norm(x).squeeze(1)      # (B, d_model)

        # --- sequence expert ---
        seq_expert = self.seq_expert(seq)                 # (B, d_model)

        # --- gate / mix (both in d_model space) ---
        gate_in = torch.cat([seq_expert, chrom_expert], dim=-1)    # (B, 2*d_model)
        g = torch.softmax(self.gate(gate_in), dim=-1)              # (B, 2)
        fused = g[:, 0:1] * seq_expert + g[:, 1:2] * chrom_expert  # (B, d_model)

        # --- head ---
        h = self.shared_mlp(fused)                        # (B, hidden_dim)
        h = self.res3(h)                                  # (B, 256)
        return self.output_head(h)                        # (B, 2)
    
class ChromatinModel2(nn.Module):
    def __init__(self, model2, hidden_dim=64):
        super(ChromatinModel2, self).__init__()

        self.model2 = model2
        self.shared_mlp = nn.Sequential(
            nn.Linear(2689 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.res3 = ResidualBlock(2048, 256)

        self.output_head = nn.Linear(256, 2)



    def forward(self, input1):

        output2 = self.model2.token_features(input1)   # (B, 2689, 2048)
        output2 = output2.permute(0, 2, 1)  # (B, 2048, 2689)
        shared_output = self.shared_mlp(output2).squeeze(-1)  # (B, 2048)

        shared_output = self.res3(shared_output)
        final_output = self.output_head(shared_output)

        return final_output

class SequenceModel(nn.Module):
    def __init__(self, model1, hidden_dim=64):
        super(SequenceModel, self).__init__()

        self.model1 = model1
        self.linear = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                4096, 4096
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                            4096, 2
                        ),
        )

        #self.output_head = nn.Linear(hidden_dim, 2)



    def forward(self, input1, input2=None):

        output1 = self.model1(input1)

        #shared_output = self.shared_mlp(output1)
        final_output = self.linear(output1)

        return final_output


class CustomDataset(Dataset):
    def __init__(self, data, chromatin_data, mode, max_len=2500):
        self.data = data[data['dataset'] == mode].reset_index(drop=True)
        self.data = self.data[self.data['Chromosome'].isin(['Chr1', 'Chr2', 'Chr3', 'Chr4', 'Chr5'])].reset_index(drop=True)

        self.sequences = self.data['Sequence'].apply(lambda seq: nucleotide_to_one_hot(seq, max_len)).values

        # Right after building data['index_id']
        print("data index_id samples:", data['index_id'].head(3).tolist())
        print("chromatin index samples:", chromatin_data.index[:3].tolist())

        # Check overlap
        data_ids = set(data['index_id'])
        chrom_ids = set(chromatin_data.index.astype(str))
        overlap = data_ids & chrom_ids
        print(f"Overlap: {len(overlap)} / {len(data_ids)} data, {len(chrom_ids)} chrom")

        # If zero overlap, look at format differences
        print("In data but not chrom (first 3):", list(data_ids - chrom_ids)[:3])
        print("In chrom but not data (first 3):", list(chrom_ids - data_ids)[:3])

        self.chromatin_features = chromatin_data.loc[self.data['index_id']].values
        self.labels = self.data['Label'].values.astype(int)
        self.num_classes=2
        print(f"{mode} dataset: {len(self.data)} samples")
        print("Chromatin features shape:", self.chromatin_features.shape)
        # chromosome in data
        self.chromosomes = self.data['Chromosome'].unique()
        print("Chromosomes in dataset:", self.chromosomes)
        self.index_ids = self.data['index_id'].values
        ## only keep chr1- chr5 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx]).permute(1, 0)
        chromatin_features = torch.tensor(self.chromatin_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx])
        # Convert binary label (0/1) to multi-class
        #label_one_hot = F.one_hot(label, num_classes=self.num_classes).float()
        return self.index_ids[idx], sequence, chromatin_features, label



def build_multiview_model(type="multiview"):

    if type == "multiview":
        model1 = build_model(new_model=True, model_path=os.path.join(DEEPPLANT_MODEL_DIR, 'model_AT_CSP.pt'), json_path=os.path.join(DEEPPLANT_MODEL_DIR, 'config_AT_2500.json'))
        model = MultiViewModel(model1, ChromatinModel(), hidden_dim=768)
    elif type == "chromatin":
         model = ChromatinModel() #ChromatinFeatureEncoder(2048), hidden_dim=768
    elif type == "sequence":
         model1 = build_model(new_model=True, model_path=os.path.join(DEEPPLANT_MODEL_DIR, 'model_AT_CSP.pt'), json_path=os.path.join(DEEPPLANT_MODEL_DIR, 'config_AT_2500.json'))
         model = SequenceModel(model1, hidden_dim=768)
    elif type == "gated_moe":
        model1 = build_model(new_model=True, model_path=os.path.join(DEEPPLANT_MODEL_DIR, 'model_AT_CSP.pt'), json_path=os.path.join(DEEPPLANT_MODEL_DIR, 'config_AT_2500.json'))
        model = GatedMoE(model1, ChromatinModel(), hidden_dim=768)
    else:
        print("Invalid model type specified. Choose from 'multiview', 'chromatin', or 'sequence'.")
    return model


import random
def set_seed(seed: int = 42) -> None:
    """
    Code was obtained from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    The torch.backends.cudnn.benchmark is set to true to improve running time
    Set Random Seeds and improve results reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def prepare_data(csv_file, chromatin_file, aggregation = "mean"):
    ## sequence information
    data = pd.read_csv(csv_file)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle data
    # chromatin information
    chromatin_data = pd.read_csv(chromatin_file)
    chromatin_data.set_index('actual', inplace=True)
    if 'window.1' in chromatin_data.columns:
        chromatin_data.drop(columns=['window.1'], inplace=True)
    if 'window' in chromatin_data.columns:
        chromatin_data.drop(columns=['window'], inplace=True)

    print(data.head())
    print(chromatin_data.head())
    if aggregation == "mean":
        chromatin_data = chromatin_data.groupby(chromatin_data.index).mean() 
    elif aggregation == "max":
        chromatin_data = chromatin_data.groupby(chromatin_data.index).max()
    else:
        chromatin_data = (chromatin_data.groupby(chromatin_data.index).apply(lambda g: pd.Series(g.values.flatten())))

    print(chromatin_data.head())
    data["mid"] = (data["Start"] + data["End"]) // 2
    data["Start"] = data["mid"] - 300
    data["End"] = data["mid"] + 300

    data["index_id"] = data["Chromosome"] + ":" + data["Start"].astype(str) + "-" + data["End"].astype(str)

    # Create datasets for Train, Val, and Test
    train_dataset = CustomDataset(data, chromatin_data, 'Train')
    val_dataset = CustomDataset(data, chromatin_data, 'Val')
    test_dataset = CustomDataset(data, chromatin_data, 'Test')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    set_seed(42)


    seqs_path = "/s/chromatin/a/nobackup/Saira/SATORI_ENHANCEMENTS/roy/enhancer_dataset_cleaned.csv"
    chromatin_path = "../DATA/enhancers/enh_600_2w_disc.csv"

    train_loader, val_loader, test_loader = prepare_data(seqs_path, chromatin_path, aggregation="max")


    model_type = "gated_moe"  # Choose from "multiview", "chromatin", or "sequence"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    base_model_path = "chromBERT_models200_allclusters"
    chrombert = GenomicTransformerV2(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_LAYERS).to(device)

    model_path = f"../{base_model_path}/best_genomic_bert.pt"
    if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...")
            state_dict = torch.load(model_path, map_location="cpu")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

            chrombert.load_state_dict(new_state_dict)
    else:
            print(f"No existing model found at {model_path}. Starting fresh.")


    multiview_model = build_multiview_model(type=model_type)

    multiview_model.model2.value_emb.weight.data.copy_(chrombert.embedding.weight.data)
    multiview_model.model2.feature_emb.weight.data.copy_(chrombert.pos_embedding.weight.data)
    multiview_model.to(device)
    chrombert.cpu()
    del chrombert
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(multiview_model.parameters(), lr=5e-4)
    ## freeze the first model's parametersE(1, e
    if isinstance(multiview_model, MultiViewModel):
        for param in multiview_model.model1.parameters():
            param.requires_grad = True
    if isinstance(multiview_model, SequenceModel):
        for param in multiview_model.model1.parameters():
            param.requires_grad = True

    for name, param in multiview_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    optimizer = torch.optim.AdamW([
    {"params": multiview_model.model1.parameters(), "lr": 5e-6},
    {"params": multiview_model.model2.parameters(), "lr": 5e-6},
    {"params": [p for n, p in multiview_model.named_parameters()
                if not n.startswith("model1.") and not n.startswith("model2.")], "lr": 5e-4},
])



    epochs = 20
    best_loss = float('inf')
    outdir = f"../{model_type}_model_results_max"
    os.makedirs(outdir, exist_ok=True)
    best_aupr = 0.0

    for epoch in range(epochs):

        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = 0.0
        multiview_model.train()

        for batch in tqdm(train_loader, desc="Training"):

            _, sequences, chromatin_features, labels = batch
            chromatin_features = chromatin_features.long()  # Ensure chromatin features are of type long for embedding
            sequences = sequences.to(device)
            chromatin_features = chromatin_features.to(device)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            if isinstance(multiview_model, MultiViewModel):
                outputs = multiview_model(sequences, chromatin_features)
            elif isinstance(multiview_model, ChromatinModel):
                outputs = multiview_model(chromatin_features)
            elif isinstance(multiview_model, SequenceModel):
                outputs = multiview_model(sequences)
            elif isinstance(multiview_model, GatedMoE):
                outputs = multiview_model(sequences, chromatin_features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(f"Training Loss: {train_loss / len(train_loader):.4f}")

        val_loss = 0.0
        val_predictions = []
        val_targets = []
        multiview_model.eval()

        with torch.no_grad():

            for batch in tqdm(val_loader, desc="Validation"):
                _, sequences, chromatin_features, labels = batch
                chromatin_features = chromatin_features.long()  # Ensure chromatin features are of type long for embedding
                sequences = sequences.to(device)
                chromatin_features = chromatin_features.to(device)
                labels = labels.to(device, dtype=torch.long)
                if isinstance(multiview_model, MultiViewModel):
                    outputs = multiview_model(sequences, chromatin_features)
                elif isinstance(multiview_model, ChromatinModel):
                    outputs = multiview_model(chromatin_features)
                elif isinstance(multiview_model, SequenceModel):
                    outputs = multiview_model(sequences)
                elif isinstance(multiview_model, GatedMoE):
                    outputs = multiview_model(sequences, chromatin_features)
                vloss = criterion(outputs, labels)
                val_preds = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of positive class
                val_predictions.extend(val_preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                val_loss += vloss.item()
        val_loss /= len(val_loader)
        ## compute AUPR
        aupr = average_precision_score(val_targets, val_predictions)
        print(f"Validation AUPR: {aupr:.4f}")


        
        print(f"Validation Loss: {val_loss :.4f}")


        if aupr > best_aupr:
            best_aupr = aupr
            torch.save(multiview_model.state_dict(), os.path.join(outdir, f"best_{model_type}_model.pth"))
            print("Best model saved with validation AUPR: {:.4f}".format(best_aupr))

        
    # load the best model
    multiview_model.load_state_dict(torch.load(os.path.join(outdir, f"best_{model_type}_model.pth")))
    multiview_model.eval()
    test_predictions = []
    test_targets = []
    all_index_ids = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            index_ids, sequences, chromatin_features, labels = batch
            
            chromatin_features = chromatin_features.long()  # Ensure chromatin features are of type long for embedding
            sequences = sequences.to(device)
            chromatin_features = chromatin_features.to(device)
            labels = labels.to(device, dtype=torch.long)
            if isinstance(multiview_model, MultiViewModel):
                outputs = multiview_model(sequences, chromatin_features)
            elif isinstance(multiview_model, ChromatinModel):
                outputs = multiview_model(chromatin_features)
            elif isinstance(multiview_model, SequenceModel):
                outputs = multiview_model(sequences)
            test_preds = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of positive class
            test_predictions.extend(test_preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            all_index_ids.extend(list(index_ids))
    test_aupr = average_precision_score(test_targets, test_predictions)
    # save test predictions with index ids
    test_results = pd.DataFrame({
        "index_id": all_index_ids,
        "prediction": test_predictions,
        "target": test_targets
    })
    test_results.to_csv(os.path.join(outdir, f"{model_type}_test_predictions.csv"), index=False)
    print(f"Test AUPR: {test_aupr:.4f}")
    