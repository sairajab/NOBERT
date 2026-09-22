
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.checkpoint import checkpoint
import numpy as np

VOCAB_SIZE = 6  # 0: PAD, 1-4: Labels, 5: MASK
EMBED_DIM = 768
N_HEADS = 8
N_LAYERS = 8
MAX_SEQ_LEN = 2689
PAD_TOKEN = 0
MASK_TOKEN = 5


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ChromBERTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, use_gradient_checkpointing=True):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, embed_dim)
        # In __init__:
        self.embed_dropout = nn.Dropout(0.1)

        # Use ModuleList for per-layer gradient checkpointing
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                batch_first=True,
                dim_feedforward=embed_dim * 4,
                activation=GELU(),
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Sequential(
            nn.Linear(embed_dim,embed_dim, bias=True),
            GELU(),
            nn.Linear(embed_dim, vocab_size, bias=False)
        )
    def forward(self, x):
        pad_mask = x == PAD_TOKEN  # (B, S)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        for layer in self.encoder_layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        logits = self.lm_head(x)
        return x, logits  # (B, S, V)


class ChromVAE(nn.Module):
    def __init__(self, vocab_size=5, input_dim=2689, embed_dim=64, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.experiment_embedding = nn.Embedding(input_dim, embed_dim)
        self.embed_dropout = nn.Dropout(0.1)

        # Encoder: per-experiment MLP then pool
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
        )
        # Pooling attention (weighted mean, not self-attention)
        self.pool_attn = nn.Linear(hidden_dim, 1)

        # VAE bottleneck
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent + experiment_id → per-experiment prediction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            GELU(),
        )
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            GELU(),
            nn.Linear(embed_dim, vocab_size, bias=False),
        )

    def encode(self, x):
        """x: (B, F) token ids"""
        exp_ids = torch.arange(self.input_dim, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.experiment_embedding(exp_ids)  # (B, F, D)
        h = self.embed_dropout(h)
        h = self.encoder(h)  # (B, F, H) - independent per experiment

        # Attention-weighted pooling
        pad_mask = (x == PAD_TOKEN)
        weights = self.pool_attn(h).squeeze(-1)          # (B, F)
        weights = weights.masked_fill(pad_mask, -1e9)
        weights = torch.softmax(weights, dim=-1)          # (B, F)
        pooled = torch.einsum('bf,bfh->bh', weights, h)   # (B, H)

        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        """z: (B, latent_dim)"""
        exp_ids = torch.arange(self.input_dim, device=z.device).unsqueeze(0)
        exp_emb = self.experiment_embedding(exp_ids)  # (1, F, D)

        # Concat latent with each experiment embedding
        z_exp = z.unsqueeze(1).expand(-1, self.input_dim, -1)  # (B, F, lat)
        h = torch.cat([z_exp, exp_emb.expand(z.size(0), -1, -1)], dim=-1)  # (B, F, lat+D)

        h = self.decoder(h)   # (B, F, H)
        logits = self.lm_head(h)  # (B, F, V)
        return h, logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h, logits = self.decode(z)
        return h, logits, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# Shared downstream-head building blocks
#
# Used by both downstream_model.py (binary classification) and
# train_gene_expression.py (multi-task regression) so the two finetuning
# scripts share a single model definition instead of drifting copies.
# ──────────────────────────────────────────────────────────────────────────────

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
    def __init__(self, in_features, out_features, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm = LayerNorm(out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Sequential()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.norm(self.fc2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, num_slots: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        total_dim = hidden_dim * num_slots

        pe = torch.zeros(num_slots, hidden_dim)
        position = torch.arange(0, num_slots, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe_flat = pe.view(1, 1, total_dim)

        self.register_buffer('pe', pe_flat)

    def forward(self, x):
        """
        x shape: (B, seq_len, hidden_dim * num_slots)
        """
        return x + self.pe


class CistromeEmbeddingManager(nn.Module):
    def __init__(self, mtx_mask, ignore=False, ignore_index=None) -> None:
        super().__init__()
        assert mtx_mask is not None, "mtx_mask must be specified"
        assert isinstance(mtx_mask, str), "mtx_mask must be a path to a mtx_mask"
        assert os.path.exists(mtx_mask), f"{mtx_mask} does not exist"
        self.mtx_mask_df = pd.read_csv(mtx_mask, sep='\t', index_col=0)
        self.mtx_mask = torch.tensor(self.mtx_mask_df.values)  # (datasets, factors)
        self.gsmid_names = self.mtx_mask_df.index.tolist()
        self.regulator_names = self.mtx_mask_df.columns.tolist()

        if ignore:
            ignore_gsmid_index = ignore_index[0]
            ignore_regulator_index = ignore_index[1]
            print(f"Ignoring {len(ignore_gsmid_index)} cistromes and {len(ignore_regulator_index)} regulators")
            rows_to_keep = torch.tensor([i not in ignore_gsmid_index for i in range(self.mtx_mask.shape[0])])
            cols_to_keep = torch.tensor([j not in ignore_regulator_index for j in range(self.mtx_mask.shape[1])])
            self.mtx_mask = self.mtx_mask[rows_to_keep][:, cols_to_keep]
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
        x = x.transpose(1, 2)
        x = torch.matmul(x, self.normalized_mtx_mask)
        x = x.transpose(1, 2)  # [batch_size, factors, hidden]

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
        extracted_x = torch.matmul(x, self.normalized_mtx_mask[:, index:index + 1])
        return extracted_x.transpose(1, 2)

    def get_region_embedding(self, x):
        return x.mean(dim=1)


class MultiWindowHeaderFusion(nn.Module):
    """Shared finetuning head: ChromBERT backbone -> cistrome interface ->
    residual MLP stack -> linear output layer.

    `num_labels` and the loss/activation applied to its output are the only
    task-specific pieces: binary classification uses num_labels=2 logits with
    CrossEntropyLoss, multi-task regression uses num_labels=n_experiments
    continuous outputs with MSELoss.
    """

    def __init__(self, chrombert_model, num_labels,
                 mtx_mask="/s/chromatin/c/nobackup/Saira/BERTModel/mtx_mask.txt",
                 n_windows=3, dropout=0.1, medium_dim=256, filter_by = None):
        super().__init__()
        self.chrombert = chrombert_model
        self.windows = n_windows
        H = 768 * n_windows

        if self.windows != 1:
            self.pos_embedding = SinusoidalPositionalEncoding(hidden_dim=768, num_slots=n_windows)
        
        self.filter_by = filter_by
        if filter_by is None:
            self.interface = CistromeEmbeddingManager(mtx_mask=mtx_mask)
            n_factors = self.interface.normalized_mtx_mask.shape[1]
        else:
            filter_columns = np.load(filter_by)[0].astype(bool)
            n_factors = int(filter_columns.sum())

        print(f"Number of factors after filtering: {n_factors}")
        self.res1 = ResidualBlock(n_factors, 1024, dropout=dropout)
        self.res2 = ResidualBlock(1024, H, dropout=dropout)
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, H), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.activation = nn.ReLU()
        self.res3 = ResidualBlock(H, medium_dim, dropout=dropout)
        self.fc = nn.Linear(medium_dim, num_labels)

    def _post_transformer(self, x):
        if self.filter_by is None:
            x = self.interface(x)  # (B, L, n_factors)
        else:
            filter_columns = np.load(self.filter_by)[0].astype(bool)
            x = x[:, filter_columns, :]  # (B, L, n_factors)
        x = x.permute(0, 2, 1)
        x = self.res1(x)
        x = self.res2(x)
        x = x[:, None, :, :]
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x, attention_mask=None, return_emb=False):
        """
        x   : (B, n_windows, MAX_SEQ_LEN)  long tokens
        attention_mask: (B, n_windows)     bool — True = real window (unused by
            the backbone today; kept for API symmetry with other heads).
        """
        B, W, L = x.shape
        x = x.view(B * W, L)                            # flatten for BERT
        x = self.chrombert(x)[0]                        # (B*W, L, 768)
        x = x.reshape(B, W, L, -1).permute(0, 2, 1, 3)   # (B, L, W, 768)
        x = x.reshape(B, L, W * 768)                     # (B, L, W*768)

        if self.windows != 1:
            x = self.pos_embedding(x)

        x   = self._post_transformer(x)
        x   = self.res3(x)
        out = self.fc(x)

        if return_emb:
            return out, x
        return out, None

    def freeze_backbone(self):
        for p in self.chrombert.parameters():
            p.requires_grad = False
        
        if self.filter_by is None:
            for p in self.interface.parameters():
                p.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self, unfreeze_layers=3):
        for p in self.chrombert.encoder_layers[-unfreeze_layers:].parameters():
            p.requires_grad = True
        print(f"Last {unfreeze_layers} encoder layers unfrozen")


class MLPMultiRegressor:
    
    def __init__(self, input_dim, num_labels, hidden_dims=[512, 256]):
        
        self.input_tokenizer = nn.Embedding(5,input_dim) 
        self.exp_embedding = nn.Embedding(2689, input_dim)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_labels)
        )
        
    def forward(self, x):
        
        pos_emb = self.exp_embedding(torch.arange(x.size(1), device=x.device).unsqueeze(0))
        out = self.input_tokenizer(x) + pos_emb
        
        print("out shape:", out.shape)
        out = self.model(out)
        
        return out