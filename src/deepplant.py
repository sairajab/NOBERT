import torch.nn as nn
from typing import Optional
import torch

import torch.nn.functional as F
from math import log
import copy
import json


from dataclasses import dataclass, asdict, field


@dataclass
class DeepPlantConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazyLoad: bool = (False,)
    loss: str = ("mse",)
    dist: str = "cosine"
    alpha: float = (0.1,)
    max_epochs: int = (30,)
    batchSize: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_DeepPlant"
    sequences_paths: list = field(default_factory=lambda: [""])
    labels_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    bits: list = field(default_factory=lambda: [0])
    use_reverse_complement: bool = False
    n_filters: int = (240,)
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    expand_factor: int = 4
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pretrained_filter: bool = False
    n_label_embeddings: int = 0
    encoder_type: str = "mamba"
    use_pos_encoding: bool = True
    max_seq_length: int = 100
    consistency_regularization: bool = False

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class Residual(nn.Module):
    def __init__(self, layer, project=None):
        super(Residual, self).__init__()
        self.layer = layer
        self.project = project

    def forward(self, x):
        return (
            self.project(x) + self.layer(x)
            if self.project != None
            else x + self.layer(x)
        )


class Conv1dWithRC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias: Optional[bool] = True,
        complement: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, bias=bias, **kwargs
        )
        self.complement = complement

    def reverse_complement_filters(self, filters):
        rc = torch.flip(filters, dims=[-1])  # reverse along sequence axis
        swap = torch.tensor([3, 2, 1, 0], device=filters.device)  # index mapping
        rc = rc[:, swap, :]
        return rc

    def reverse_filters(self, filters):
        r = torch.flip(filters, dims=[-1])  # reverse along sequence axis
        return r

    def forward(self, x):
        # x: (batch, 4, length)
        w = self.conv.weight  # (out_channels, 4, kernel_size)
        b = self.conv.bias

        # compute RC filters

        if self.complement:
            w_rc = self.reverse_complement_filters(w)
            return F.conv1d(
                x,
                w,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            ) + F.conv1d(
                x,
                w_rc,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )

        else:
            w_r = self.reverse_filters(w)
            return F.conv1d(
                x,
                w,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            ) + F.conv1d(
                x,
                w_r,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )


class ConvNet(nn.Module):
    def __init__(self, n_filters: int, n_genomes: int):
        super(ConvNet, self).__init__()
        self.n_filters = n_filters
        self.n_genomes = n_genomes
        self.dropout = 0.2
        self.conv_features = list()
        for _ in range(self.n_genomes):
            self.conv_features.append(
                nn.Sequential(
                    Conv1dWithRC(
                        4, self.n_filters, kernel_size=11, complement=True, padding=5
                    ),
                    nn.BatchNorm1d(self.n_filters),
                    nn.MaxPool1d(kernel_size=5, stride=5),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_features = nn.ModuleList(self.conv_features)

        self.conv_network = nn.Sequential(
            Residual(
                nn.Sequential(
                    nn.Conv1d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(self.n_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        self.n_filters,
                        self.n_filters,
                        kernel_size=7,
                        padding=3,
                        # dilation=2,
                    ),
                    nn.BatchNorm1d(self.n_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        self.n_filters,
                        4 * self.n_filters,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(4 * self.n_filters),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=self.dropout),
                ),
                project=nn.Sequential(
                    nn.Conv1d(
                        self.n_filters,
                        4 * self.n_filters,
                        kernel_size=9,
                        padding=4,
                    ),
                    nn.BatchNorm1d(4 * self.n_filters),
                ),
            ),
            nn.MaxPool1d(kernel_size=5, stride=5),
            Residual(
                nn.Sequential(
                    nn.Conv1d(
                        4 * self.n_filters,
                        2 * self.n_filters,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(2 * self.n_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        2 * self.n_filters,
                        2 * self.n_filters,
                        kernel_size=7,
                        padding=3,
                        # dilation=2,
                    ),
                    nn.BatchNorm1d(2 * self.n_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        2 * self.n_filters,
                        4 * self.n_filters,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(4 * self.n_filters),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=self.dropout),
                ),
                project=nn.Sequential(
                    nn.Conv1d(
                        4 * self.n_filters,
                        4 * self.n_filters,
                        kernel_size=5,
                        padding=4,
                        dilation=2,
                    ),
                    nn.BatchNorm1d(4 * self.n_filters),
                ),
            ),
        )
        self.num_channels = 4 * self.n_filters

    def forward(self, input: torch.Tensor, bit: int = 0):
        features = self.conv_features[bit](input)
        return self.conv_network(features)


def build_ConvNet(args):
    convnet = ConvNet(n_filters=args.n_filters, n_genomes=len(args.n_features))
    return convnet


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{self.weight.shape[0]}, eps={self.variance_epsilon}"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def clone_layer(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.encoder_layers = clone_layer(encoder_layer, self.num_layers)

    def add_pos_embed(self, src, pos_embed):
        return src if pos_embed == None else (src + pos_embed)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
    ):
        encoder_output = self.add_pos_embed(src, pos_embed)
        for layer in self.encoder_layers:
            if src_mask != None or src_key_padding_mask != None:
                encoder_output = layer(encoder_output, src_mask, src_key_padding_mask)
            else:
                encoder_output = layer(encoder_output)
        if self.norm != None:
            encoder_output = self.norm(encoder_output)
        return encoder_output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.decoder_layers = clone_layer(decoder_layer, self.num_layers)

    def add_pos_embed(self, tgt, pos_embed):
        return tgt if pos_embed == None else (tgt + pos_embed)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_pos_embed: Optional[torch.Tensor] = None,
    ):

        tgt = self.add_pos_embed(tgt, tgt_pos_embed)
        for layer in self.decoder_layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
        if self.norm != None:
            tgt = self.norm(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: Optional[int] = 512,
        nhead: Optional[int] = 8,
        dim_feedforward: Optional[int] = 2048,
        dropout: Optional[float] = 0.2,
        encoder_type: Optional[str] = "mamba",
        encoder_num_layers: Optional[int] = 6,
        decoder_num_layers: Optional[int] = 6,
        use_pos_encoding: Optional[bool] = True,
        max_seq_length: Optional[int] = 100,
    ):
        super(Transformer, self).__init__()
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.use_pos_encoding = use_pos_encoding
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        if self.use_pos_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model, max_seq_length=self.max_seq_length
            )
        if self.encoder_num_layers > 0:
            if encoder_type == "mamba":
                encoder_norm = None  # MambaRMSNorm(d_model)  # nn.RMSNorm(d_model)  # nn.LayerNorm(d_model)
                """self.encoder = TransformerEncoder(
                    Mamba2(
                        d_model=d_model,
                        d_state=64,
                        d_conv=4,  # Local convolution width
                        expand=2,  # Block expansion factor
                    ),
                    encoder_num_layers,
                    encoder_norm,
                )"""
            else:
                encoder_norm = None  # nn.LayerNorm(d_model)
                self.encoder = TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        norm_first=True,
                        batch_first=True,
                    ),
                    encoder_num_layers,
                    encoder_norm,
                )
        if self.decoder_num_layers > 0:
            decoder_norm = None  # nn.LayerNorm(d_model)

            self.decoder = TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=True,
                    batch_first=True,
                ),
                decoder_num_layers,
                decoder_norm,
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,  # Query embeddings
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_pos_embed: Optional[torch.Tensor] = None,
    ):
        if self.use_pos_encoding:
            src = self.pos_encoder(src)
        if self.encoder_type == "mamba":
            encoder_embed = self.encoder(src)
        else:
            encoder_embed = self.encoder(src, src_mask, src_key_padding_mask)
        if self.decoder_num_layers > 0:
            decoder_output = self.decoder(
                tgt,
                encoder_embed,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
                tgt_pos_embed,
            )
            return decoder_output
        return encoder_embed


def build_transformer(args):
    return Transformer(
        d_model=args.embed_dim,
        nhead=args.num_heads,
        dim_feedforward=args.dim_forwardfeed,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        use_pos_encoding=args.use_pos_encoding,
        max_seq_length=args.max_seq_length,
    )


def build_predictionHead(args):
    return PredictionHead(
        embed_dim=args.embed_dim,
        expand_factor=args.expand_factor,
        n_features=args.n_features,
    )


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super(AttentionPool, self).__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor):
        attn_logits = torch.einsum("b n d, d e -> b n e", x, self.to_attn_logits)
        attn = attn_logits.softmax(dim=-2)
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class PredictionHead(nn.Module):
    def __init__(self, embed_dim, expand_factor, n_features):
        super(PredictionHead, self).__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.expand_factor = expand_factor
        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.expand_factor * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                self.expand_factor * self.embed_dim, self.expand_factor * self.embed_dim
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.prediction_head = list()
        for i in range(self.n_genomes):
            self.prediction_head.append(
                nn.Sequential(
                    nn.Linear(self.expand_factor * self.embed_dim, self.n_features[i]),
                    nn.ReLU(),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        return output


class model(nn.Module):
    def __init__(
        self,
        backbone,
        transfomer,
        predictionHead,
        consistency_regularization: Optional[bool] = True,
    ):
        super(model, self).__init__()
        self.backbone = backbone
        self.transformer = transfomer
        num_channels = backbone.num_channels
        embed_dim = predictionHead.embed_dim
        self.attention_pool = AttentionPool(embed_dim)
        self.consistency_regularization = consistency_regularization

        self.fc = predictionHead

    def get_convlayers(self, bit: int = 0):
        for i in range(self.backbone.conv_features.__len__()):
            yield (
                self.backbone.conv_features[i][0].weight
                if i == bit
                else self.backbone.conv_features[i][0].weight.detach()
            )

    def forward(self, input: torch.Tensor, bit: int = 0):
        if input.ndim == 4:
            input = input.view(
                input.shape[0] * input.shape[1], input.shape[2], input.shape[3]
            )
        src = self.backbone(input, bit).permute(0, 2, 1)
        attention_output = self.transformer(src)
        hs = self.attention_pool(attention_output)  #

        #return hs
    
        out = self.fc(hs, bit)
        if self.consistency_regularization:
            return hs
        else:
            return hs, out


def read_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    data["n_features"] = [2]
    return data


def build_model(
    new_model: bool,
    model_path: Optional[str] = None,
    json_path: Optional[str] = "",
):
    args = DeepPlantConfig(**read_json(json_path=json_path))
    backbone = build_ConvNet(args)
    transformer = build_transformer(args)
    predictionHead = build_predictionHead(args)
    network = model(
        backbone=backbone,
        transfomer=transformer,
        predictionHead=predictionHead,
        consistency_regularization=args.consistency_regularization,
    )
    if model_path != None:
        print("Loading model state")
        model_pretrained_dict = torch.load(model_path)
        keys_pretrained = list(model_pretrained_dict.keys())
        #print(keys_pretrained[-2:])
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained[:-2])):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        if not new_model:
            model_weights[keys_net[-2]] = model_pretrained_dict[keys_pretrained[-2]]
            model_weights[keys_net[-1]] = model_pretrained_dict[keys_pretrained[-1]]
            print("Model checkpoint loaded")

        network.load_state_dict(model_weights)
    return network