
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    