from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolRegressor(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dims: list[int] | None = None,
                 out_dim: int = 6855, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        prev = in_dim
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pooled = torch.stack([f.mean(dim=0) for f in features], dim=0)
        return self.net(pooled)


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 256,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3):
        super().__init__()
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        if hidden_dims is None:
            hidden_dims = []
        prev = hidden_dim
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.classifier = nn.Sequential(*layers)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        slide_embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_w(torch.tanh(A_V) * torch.sigmoid(A_U))
            A = F.softmax(A.transpose(1, 0), dim=1)
            slide_emb = A @ h
            slide_embeddings.append(slide_emb.squeeze(0))
        return torch.stack(slide_embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


class FactorizedProteomeHead(nn.Module):
    def __init__(self, slide_encoder: nn.Module, hidden_dim: int,
                 latent_dim: int, decoder_weight: torch.Tensor,
                 decoder_bias: torch.Tensor):
        super().__init__()
        self.slide_encoder = slide_encoder
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.register_buffer("decoder_weight", decoder_weight)
        self.register_buffer("decoder_bias", decoder_bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        h = self.slide_encoder.encode(features)
        latent = self.latent_proj(h)
        return F.linear(latent, self.decoder_weight, self.decoder_bias)


def build_model(model_name: str, in_dim: int = 768, out_dim: int = 6855,
                hidden_dim: int = 256, **kwargs) -> nn.Module:
    dropout = kwargs.get("dropout", 0.3)
    if model_name == "meanpool":
        hidden = kwargs.get("hidden_dims", [512, 256, 128])
        return MeanPoolRegressor(in_dim=in_dim, hidden_dims=hidden, out_dim=out_dim, dropout=dropout)
    elif model_name == "meanpool_linear":
        return MeanPoolRegressor(in_dim=in_dim, hidden_dims=[], out_dim=out_dim, dropout=dropout)
    elif model_name == "attention":
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        return GatedAttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                                 hidden_dims=hidden, out_dim=out_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")
