from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: build classifier MLP (shared across models)
# ---------------------------------------------------------------------------

def _build_classifier(hidden_dim: int, cls_dims: list[int], out_dim: int,
                      dropout: float, norm: Literal["bn", "ln"] = "bn",
                      residual: bool = False) -> nn.Module:
    """Build a classifier head from hidden_dim -> cls_dims -> out_dim.

    Args:
        hidden_dim: Input dimension (typically attention output dim).
        cls_dims: Intermediate hidden dimensions.
        out_dim: Output dimension (proteome).
        dropout: Dropout rate.
        norm: Normalization - 'bn' (BatchNorm1d) or 'ln' (LayerNorm).
        residual: If True, add residual connections between same-sized blocks.
    """
    if not cls_dims:
        return nn.Linear(hidden_dim, out_dim)

    prev = hidden_dim
    layers = []
    for i, h in enumerate(cls_dims):
        block = []
        if norm == "bn":
            block.append(nn.Linear(prev, h))
            if h > 1:
                block.append(nn.BatchNorm1d(h))
        else:  # LayerNorm
            block.append(nn.Linear(prev, h))
            block.append(nn.LayerNorm(h))
        block.append(nn.GELU())
        block.append(nn.Dropout(dropout))

        if residual and prev == h:
            layers.append(ResidualBlock(nn.Sequential(*block)))
        else:
            layers.append(nn.Sequential(*block))
        prev = h

    layers.append(nn.Linear(prev, out_dim))

    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Residual wrapper: F(x) + x."""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x) + x


# ---------------------------------------------------------------------------
# 1. Original MeanPool + original GatedAttention (preserved)
# ---------------------------------------------------------------------------

class MeanPoolRegressor(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dims: list[int] | None = None,
                 out_dim: int = 6855, dropout: float = 0.3):
        super().__init__()
        self.net = _build_classifier(in_dim, hidden_dims or [], out_dim, dropout)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        pooled = torch.stack([f.mean(dim=0) for f in features], dim=0)
        return self.net(pooled)


class GatedAttentionMIL(nn.Module):
    """Standard single-head gated attention MIL."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 256,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "bn", residual: bool = False):
        super().__init__()
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [], out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_w(torch.tanh(A_V) * torch.sigmoid(A_U))
            A = F.softmax(A.transpose(1, 0), dim=1)
            slide_emb = A @ h
            embeddings.append(slide_emb.squeeze(0))
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 2. Multi-Head GatedAttention
# ---------------------------------------------------------------------------

class MultiHeadGatedAttentionMIL(nn.Module):
    """Multi-head gated attention MIL.

    Each head independently attends to the bag; outputs are concatenated
    and projected back to hidden_dim.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024,
                 n_heads: int = 4, hidden_dims: list[int] | None = None,
                 out_dim: int = 6855, dropout: float = 0.3,
                 norm: str = "ln", residual: bool = True):
        super().__init__()
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        head_dim = hidden_dim // n_heads

        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-head attention parameters
        self.attention_V = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])
        self.attention_U = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])
        self.attention_w = nn.ModuleList([nn.Linear(head_dim, 1, bias=False) for _ in range(n_heads)])

        # Project multi-head output back
        self.head_proj = nn.Linear(hidden_dim, hidden_dim) if n_heads > 1 else nn.Identity()

        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [], out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            h_split = h.chunk(self.n_heads, dim=-1)

            head_outputs = []
            for i in range(self.n_heads):
                hi = h_split[i]
                A_V = self.attention_V[i](hi)
                A_U = self.attention_U[i](hi)
                A = self.attention_w[i](torch.tanh(A_V) * torch.sigmoid(A_U))
                A = F.softmax(A.transpose(1, 0), dim=1)
                head_out = A @ hi
                head_outputs.append(head_out)

            combined = torch.cat(head_outputs, dim=-1)
            combined = self.head_proj(combined)
            embeddings.append(combined.squeeze(0))

        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 3. Dual-Path MIL (Attention + Mean pooling)
# ---------------------------------------------------------------------------

class DualPathMIL(nn.Module):
    """Concatenates GatedAttention output with a projected mean-pooled vector."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        self.mean_proj = nn.Linear(in_dim, hidden_dim)

        self.classifier = _build_classifier(
            hidden_dim * 2, hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_w(torch.tanh(A_V) * torch.sigmoid(A_U))
            A = F.softmax(A.transpose(1, 0), dim=1)
            attn_out = (A @ h).squeeze(0)

            mean_out = self.mean_proj(bag.mean(dim=0, keepdim=True)).squeeze(0)

            combined = torch.cat([attn_out, mean_out], dim=-1)
            embeddings.append(combined)

        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 4. Transformer MIL (CLS token + small Transformer encoder)
# ---------------------------------------------------------------------------

class TransformerMIL(nn.Module):
    """Transformer encoder over instances with learned CLS token."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 n_layers: int = 2, n_heads: int = 8,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag).unsqueeze(0)
            cls = self.cls_token.expand(1, -1, -1)
            h_cls = torch.cat([cls, h], dim=1)
            out = self.transformer(h_cls)
            cls_out = out[:, 0, :]
            embeddings.append(cls_out.squeeze(0))

        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 5. Top-K Gated Attention
# ---------------------------------------------------------------------------

class TopKGatedAttentionMIL(nn.Module):
    """GatedAttention but only aggregates the top-k patches by attention weight."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024,
                 top_k: int = 64, hidden_dims: list[int] | None = None,
                 out_dim: int = 6855, dropout: float = 0.3,
                 norm: str = "bn", residual: bool = False):
        super().__init__()
        self.top_k = top_k
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [], out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_w(torch.tanh(A_V) * torch.sigmoid(A_U))

            k = min(self.top_k, A.size(0))
            topk_vals, topk_idx = torch.topk(A.squeeze(-1), k, dim=0)
            h_topk = h[topk_idx]

            A_topk = F.softmax(topk_vals.unsqueeze(0), dim=1)
            slide_emb = A_topk @ h_topk
            embeddings.append(slide_emb.squeeze(0))

        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 6. Deep Sets (Instance MLP + Mean Pool)
# ---------------------------------------------------------------------------

class DeepSetsMIL(nn.Module):
    """Deep Sets-style: per-instance MLP then mean pooling + classifier."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.instance_phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            phi = self.instance_phi(bag)
            pooled = phi.mean(dim=0)
            embeddings.append(pooled)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 7. Low-Rank Decoder MIL (factorized output, literature-motivated)
# ---------------------------------------------------------------------------

class LowRankGatedAttentionMIL(nn.Module):
    """GatedAttention with a low-rank factorized decoder: hidden -> latent -> out.

    Uses a bottleneck (latent_dim < hidden_dim) to factorize the output
    weight matrix. Optionally adds a residual skip connection for
    high-rank residual.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 1024,
                 latent_dim: int = 64, out_dim: int = 6855,
                 dropout: float = 0.3, use_skip: bool = True):
        super().__init__()
        self.tile_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reconstruct = nn.Linear(latent_dim, out_dim)
        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Linear(hidden_dim, out_dim, bias=False)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.tile_proj(bag)
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_w(torch.tanh(A_V) * torch.sigmoid(A_U))
            A = F.softmax(A.transpose(1, 0), dim=1)
            slide_emb = A @ h
            embeddings.append(slide_emb.squeeze(0))
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        z = self.encode(features)
        z_low = self.bottleneck(z)
        out = self.reconstruct(z_low)
        if self.use_skip:
            out = out + self.skip(z)
        return out


# ---------------------------------------------------------------------------
# 8. Perceiver MIL — learned latent tokens cross-attend to patches
# ---------------------------------------------------------------------------

class PerceiverMIL(nn.Module):
    """Perceiver-style: learned latent tokens extract info from patches via
    cross-attention, then self-attend to integrate. Highly parameter-efficient
    because latents are few (k=16) and small (d=128)."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128,
                 n_latents: int = 16, n_layers: int = 2, n_heads: int = 4,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.latents = nn.Parameter(torch.randn(1, n_latents, hidden_dim) * 0.02)

        # Alternating cross-attention + self-attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict()
            block['cross_attn'] = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True)
            block['cross_norm'] = nn.LayerNorm(hidden_dim)
            block['self_attn'] = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True)
            self.blocks.append(block)

        cls_in = hidden_dim  # mean pool over latents
        self.classifier = _build_classifier(
            cls_in, hidden_dims or [cls_in, cls_in // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag).unsqueeze(0)  # [1, n, d]
            latents = self.latents.expand(h.size(0), -1, -1)  # [1, k, d]
            for block in self.blocks:
                # Cross-attend: latents(Q) query patches(K,V)
                latent_out, _ = block['cross_attn'](latents, h, h)
                latents = block['cross_norm'](latents + latent_out)
                # Self-attend among latents
                latents = block['self_attn'](latents)
            bag_emb = latents.mean(dim=1).squeeze(0)  # [d]
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 9. Prototype MIL — patches softly assigned to learned prototypes
# ---------------------------------------------------------------------------

class PrototypeMIL(nn.Module):
    """Each patch is softly assigned to K learned prototype vectors via
    RBF similarity. Bag representation = mean pooled features + prototype
    activation vector (which prototypes are activated in this bag).

    Interpretable: prototypes capture recurrent tissue morphologies."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128,
                 n_prototypes: int = 32, temperature: float = 1.0,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.temperature = temperature

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Learned prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, hidden_dim) * 0.1)
        # Per-prototype temperature (learned)
        self.log_tau = nn.Parameter(torch.zeros(n_prototypes))

        # Bag representation: mean-pooled features + prototype activations
        bag_dim = hidden_dim + n_prototypes
        self.classifier = _build_classifier(
            bag_dim, hidden_dims or [bag_dim, bag_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)  # [n, d]
            # L2 distance to each prototype: [n, K]
            diff = h.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [n, K, d]
            dists = -diff.square().sum(dim=-1)  # negative squared L2: [n, K]
            # Prototype temperature
            tau = self.log_tau.exp().unsqueeze(0)  # [1, K]
            assign = F.softmax(dists / (tau * self.temperature), dim=-1)  # [n, K]
            # Bag-level prototype activation
            proto_act = assign.mean(dim=0)  # [K]
            # Mean-pooled features
            mean_feat = h.mean(dim=0)  # [d]
            # Concatenate
            bag_emb = torch.cat([mean_feat, proto_act], dim=0)
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 10. Sinkhorn (Optimal Transport) MIL — transport patches to learned slots
# ---------------------------------------------------------------------------

class SinkhornMIL(nn.Module):
    """Optimal Transport pooling: patches are transported to M learned
    'slot' vectors via Sinkhorn normalization. Each slot captures a mode
    of the patch distribution. Bag representation = concatenated slot
    features.

    Unlike attention pooling, this captures the full distribution of
    patch features, not just a weighted centroid."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128,
                 n_slots: int = 16, n_sinkhorn_iters: int = 3,
                 sinkhorn_epsilon: float = 0.05,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.n_slots = n_slots
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Learned slot codes
        self.slots = nn.Parameter(
            torch.randn(n_slots, hidden_dim) * 0.1)

        # Bag = flattened slot features
        bag_dim = n_slots * hidden_dim
        self.classifier = _build_classifier(
            bag_dim, hidden_dims or [bag_dim, bag_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    @staticmethod
    def _sinkhorn(logits: torch.Tensor, n_iters: int = 3,
                  epsilon: float = 0.05) -> torch.Tensor:
        """Stabilized Sinkhorn-Knopp algorithm.
        logits: [n, m] negative cost matrix.
        Returns transport plan P: [n, m] doubly stochastic.
        """
        # Log-domain for stability
        log_K = logits / epsilon
        log_K = log_K - log_K.max(dim=-1, keepdim=True)[0]  # stability
        K = torch.exp(log_K)  # [n, m]

        m = logits.size(1)
        v = torch.ones(m, device=logits.device) / m

        for _ in range(n_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)

        P = torch.diag(u) @ K @ torch.diag(v)
        return P

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)  # [n, d]
            n_patches = h.size(0)

            # Cost matrix: negative squared L2 distance [n, M]
            diff = h.unsqueeze(1) - self.slots.unsqueeze(0)
            cost = -diff.square().sum(dim=-1)  # [n, M], higher = more similar

            # Sinkhorn transport plan
            P = self._sinkhorn(
                cost, self.n_sinkhorn_iters, self.sinkhorn_epsilon)  # [n, M]

            # Transport patches to slots
            slot_feats = P.T @ h  # [M, d]

            # Flatten all slot features
            bag_emb = slot_feats.reshape(-1)  # [M * d]
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 11. Mixture of Attention Experts — multiple pooling strategies + gating
# ---------------------------------------------------------------------------

class MixtureAttentionMIL(nn.Module):
    """Multiple attention experts with different pooling strategies,
    combined via learned bag-level gating. Each expert specializes in
    different tissue-proteome relationships.

    Base experts: GatedAttention, SigmoidAttention, SoftmaxAttention, MaxPool.
    When n_experts > 4, additional gated attention experts are created.
    Gate: bag-level MLP predicts expert weights."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128,
                 n_experts: int = 4,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Expert 1: Gated attention (tanh(V·h) * sigmoid(U·h))
        self.gate_V = nn.Linear(hidden_dim, hidden_dim)
        self.gate_U = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1, bias=False)

        # Expert 2: Sigmoid-only attention
        self.sigmoid_U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_w = nn.Linear(hidden_dim, 1, bias=False)

        # Expert 3: Softmax-only attention (ABMIL style)
        self.softmax_V = nn.Linear(hidden_dim, hidden_dim)
        self.softmax_w = nn.Linear(hidden_dim, 1, bias=False)

        # Expert 4: Max pooling (no extra params)

        extra_n = max(0, n_experts - 4)
        self.extra_expert_Vs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(extra_n)])
        self.extra_expert_Us = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(extra_n)])
        self.extra_expert_ws = nn.ModuleList(
            [nn.Linear(hidden_dim, 1, bias=False) for _ in range(extra_n)])

        # Gating network: patches -> bag gate -> expert weights
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_experts),
        )

        # Classifier on gated mixture
        self.classifier = _build_classifier(
            hidden_dim, hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)  # [n, d]

            # ---- Expert outputs ----
            # 1. Gated attention
            A_g = self.gate_w(
                torch.tanh(self.gate_V(h)) * torch.sigmoid(self.gate_U(h)))
            A_g = F.softmax(A_g.transpose(1, 0), dim=1)
            experts = [(A_g @ h).squeeze(0)]  # [d]

            # 2. Sigmoid attention
            A_s = self.sigmoid_w(self.sigmoid_U(h))
            A_s = F.softmax(A_s.transpose(1, 0), dim=1)
            experts.append((A_s @ h).squeeze(0))

            # 3. Softmax attention
            A_m = self.softmax_w(self.softmax_V(h))
            A_m = F.softmax(A_m.transpose(1, 0), dim=1)
            experts.append((A_m @ h).squeeze(0))

            # 4. Max pool
            experts.append(h.max(dim=0).values)

            for V, U, w in zip(self.extra_expert_Vs, self.extra_expert_Us,
                               self.extra_expert_ws):
                A = w(torch.tanh(V(h)) * torch.sigmoid(U(h)))
                A = F.softmax(A.transpose(1, 0), dim=1)
                experts.append((A @ h).squeeze(0))

            expert_stack = torch.stack(experts, dim=0)

            # ---- Gating ----
            pooled = h.mean(dim=0)
            gate_logits = self.gate_net(pooled)
            gate_w = F.softmax(gate_logits, dim=0)

            # ---- Weighted mixture ----
            bag_emb = (gate_w.unsqueeze(1) * expert_stack).sum(dim=0)
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 12. DualMoE — MoAE × Dual-Path hybrid
# ---------------------------------------------------------------------------

class DualMoE(nn.Module):
    """Each MoAE expert (gated, sigmoid, softmax, max) produces an attention
    representation. The gated mixture of expert attention outputs is then
    concatenated with a shared mean-pooled path (like Dual-Path).

    Fusion: [sum_e gate_e * attn_e || mean_pool(h)] -> 2*hidden -> classifier.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 n_experts: int = 4,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Expert 1: Gated
        self.gate_V = nn.Linear(hidden_dim, hidden_dim)
        self.gate_U = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 2: Sigmoid
        self.sigmoid_U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 3: Softmax
        self.softmax_V = nn.Linear(hidden_dim, hidden_dim)
        self.softmax_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 4: Max pooling (no extra params)

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_experts),
        )
        # Classifier on 2*hidden (attn_mixture || mean_pool)
        self.classifier = _build_classifier(
            hidden_dim * 2,
            hidden_dims or [hidden_dim * 2, hidden_dim],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)

            # Expert outputs
            A_g = F.softmax(self.gate_w(torch.tanh(self.gate_V(h)) * torch.sigmoid(self.gate_U(h))).transpose(1, 0), dim=1)
            e1 = (A_g @ h).squeeze(0)
            A_s = F.softmax(self.sigmoid_w(self.sigmoid_U(h)).transpose(1, 0), dim=1)
            e2 = (A_s @ h).squeeze(0)
            A_m = F.softmax(self.softmax_w(self.softmax_V(h)).transpose(1, 0), dim=1)
            e3 = (A_m @ h).squeeze(0)
            e4 = h.max(dim=0).values
            experts = torch.stack([e1, e2, e3, e4], dim=0)

            # Gating
            gate_w = F.softmax(self.gate_net(h.mean(dim=0)), dim=0)
            attn_mix = (gate_w.unsqueeze(1) * experts).sum(dim=0)

            # Dual path: concat attn mixture + mean pool
            mean_pool = h.mean(dim=0)
            bag_emb = torch.cat([attn_mix, mean_pool], dim=0)
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 13. CrossAttentionFusion — Dual-Path + MoAE encoders fused via cross-attn
# ---------------------------------------------------------------------------

class CrossAttentionFusionMIL(nn.Module):
    """Dual encoder: one Dual-Path encoder + one MoAE encoder.
    Bag representations are fused via cross-attention, then classified.

    Encoder 1 (Dual-Path): [attn || mean]  -> 2*hidden
    Encoder 2 (MoAE):      gated mixture    -> hidden
    Cross-attn: MoAE Q attends to Dual-Path KV -> fused -> classifier.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 n_heads: int = 4, n_latents: int = 16,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        # Shared input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Encoder 1: Dual-Path (gated attn + mean)
        self.dp_V = nn.Linear(hidden_dim, hidden_dim)
        self.dp_U = nn.Linear(hidden_dim, hidden_dim)
        self.dp_w = nn.Linear(hidden_dim, 1, bias=False)
        # Encoder 2: MoAE (4 experts)
        self.gate_V = nn.Linear(hidden_dim, hidden_dim)
        self.gate_U = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1, bias=False)
        self.sigmoid_U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_w = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax_V = nn.Linear(hidden_dim, hidden_dim)
        self.softmax_w = nn.Linear(hidden_dim, 1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4),
        )
        # Project both encoders to same dim for cross-attn
        self.proj_dp = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_moae = nn.Linear(hidden_dim, hidden_dim)
        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        # Classifier
        cls_in = hidden_dim  # fused rep
        self.classifier = _build_classifier(
            cls_in, hidden_dims or [cls_in, cls_in // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)  # [n, d]

            # Encoder 1: Dual-Path
            A_dp = F.softmax(self.dp_w(torch.tanh(self.dp_V(h)) * torch.sigmoid(self.dp_U(h))).transpose(1, 0), dim=1)
            dp_attn = (A_dp @ h).squeeze(0)
            dp_mean = h.mean(dim=0)
            bag_dp = torch.cat([dp_attn, dp_mean], dim=0)  # [2d]

            # Encoder 2: MoAE
            A_g = F.softmax(self.gate_w(torch.tanh(self.gate_V(h)) * torch.sigmoid(self.gate_U(h))).transpose(1, 0), dim=1)
            e1 = (A_g @ h).squeeze(0)
            A_s = F.softmax(self.sigmoid_w(self.sigmoid_U(h)).transpose(1, 0), dim=1)
            e2 = (A_s @ h).squeeze(0)
            A_m = F.softmax(self.softmax_w(self.softmax_V(h)).transpose(1, 0), dim=1)
            e3 = (A_m @ h).squeeze(0)
            e4 = h.max(dim=0).values
            experts = torch.stack([e1, e2, e3, e4], dim=0)
            gate_w = F.softmax(self.gate_net(h.mean(dim=0)), dim=0)
            bag_moae = (gate_w.unsqueeze(1) * experts).sum(dim=0)  # [d]

            # Project to same dim
            z_dp = self.proj_dp(bag_dp.unsqueeze(0)).unsqueeze(1)    # [1, 1, d]
            z_moae = self.proj_moae(bag_moae.unsqueeze(0)).unsqueeze(1)  # [1, 1, d]

            # Cross-attend: MoAE(Q) attends to Dual-Path(KV)
            fused, _ = self.cross_attn(z_moae, z_dp, z_dp)

            # Residual connection
            fused = self.cross_norm(fused + z_moae).squeeze(0).squeeze(0)  # [d]
            embeddings.append(fused)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 14. ProgressiveExpertMIL — residual expert stacking (boosting-style)
# ---------------------------------------------------------------------------

class ProgressiveExpertMIL(nn.Module):
    """Experts stacked as residual corrections. Each expert's bag rep is
    added as a 'boost' to the previous, with learned per-expert scaling.

    Unlike MoAE (bag-level gating), this forces each expert to contribute
    complementary information that earlier experts missed."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 n_experts: int = 4,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Expert 1: Gated
        self.gate_V = nn.Linear(hidden_dim, hidden_dim)
        self.gate_U = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 2: Sigmoid
        self.sigmoid_U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 3: Softmax
        self.softmax_V = nn.Linear(hidden_dim, hidden_dim)
        self.softmax_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 4: Mean pool
        # (no extra params)

        # Learned residual scaling weights
        self.res_weights = nn.Parameter(torch.ones(n_experts) * 0.1)
        # Per-expert LayerNorm for residual
        self.res_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_experts)])

        # Classifier on final accumulated representation
        self.classifier = _build_classifier(
            hidden_dim,
            hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)

            # Expert 1: Gated attention
            A_g = F.softmax(self.gate_w(torch.tanh(self.gate_V(h)) * torch.sigmoid(self.gate_U(h))).transpose(1, 0), dim=1)
            e1 = (A_g @ h).squeeze(0)

            # Expert 2: Sigmoid (residual)
            A_s = F.softmax(self.sigmoid_w(self.sigmoid_U(h)).transpose(1, 0), dim=1)
            e2 = (A_s @ h).squeeze(0)

            # Expert 3: Softmax (residual)
            A_m = F.softmax(self.softmax_w(self.softmax_V(h)).transpose(1, 0), dim=1)
            e3 = (A_m @ h).squeeze(0)

            # Expert 4: Mean pool (residual)
            e4 = h.mean(dim=0)

            # Residual accumulation with learned weights
            experts = [e1, e2, e3, e4]
            accum = experts[0]
            for i in range(1, len(experts)):
                accum = accum + self.res_weights[i] * self.res_norms[i](experts[i])

            # Final LayerNorm for stability
            bag_emb = self.res_norms[0](accum)
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# 15. SparseTop2MoAE — token-choice top-2 expert routing
# ---------------------------------------------------------------------------

class SparseTop2MoAE(nn.Module):
    """Each patch is routed to its top-2 experts via a learned router.
    Each expert processes only its assigned patches. Load-balancing loss
    via auxiliary loss (mean square of expert utilization).

    Key difference from MoAE: experts see DIFFERENT subsets of patches,
    forcing genuine specialization."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512,
                 n_experts: int = 4, top_k: int = 2,
                 hidden_dims: list[int] | None = None, out_dim: int = 6855,
                 dropout: float = 0.3, norm: str = "ln", residual: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        # Router: patch -> expert scores
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)

        # Expert 1: Gated
        self.gate_V = nn.Linear(hidden_dim, hidden_dim)
        self.gate_U = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 2: Sigmoid
        self.sigmoid_U = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 3: Softmax
        self.softmax_V = nn.Linear(hidden_dim, hidden_dim)
        self.softmax_w = nn.Linear(hidden_dim, 1, bias=False)
        # Expert 4: Max pool
        # (no extra params)

        # Learned gating for final expert fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_experts),
        )

        self.classifier = _build_classifier(
            hidden_dim,
            hidden_dims or [hidden_dim, hidden_dim // 2],
            out_dim, dropout, norm=norm, residual=residual)

    def _route(self, h: torch.Tensor) -> tuple:
        """Route patches to top-k experts.
        Returns: (expert_indices, expert_weights, load_balance_loss)
        """
        scores = self.router(h)  # [n, E]
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=1)

        # Softmax over selected experts per patch
        topk_weights = F.softmax(topk_vals / 0.1, dim=1)  # temperature

        # Load balancing loss (CV^2 of utilization)
        importance = scores.softmax(dim=1).mean(dim=0)  # [E]
        uniform = torch.ones_like(importance) / self.n_experts
        load_balance_loss = (importance - uniform).pow(2).mean()
        # Also encourage diversity: variance of gate usage
        self._lb_loss = load_balance_loss

        return topk_idx, topk_weights

    def _forward_one_expert(self, expert_id: int, h: torch.Tensor,
                            indices: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """Run one expert on its assigned patches."""
        # Find patches assigned to this expert
        mask = (indices == expert_id).any(dim=1)  # [n]
        if not mask.any():
            return torch.zeros(h.size(1), device=h.device)

        h_expert = h[mask]
        # Get the routing weights for these patches to this expert
        expert_weight_mask = (indices == expert_id)  # [n, k]
        # For each assigned patch, find its weight for this expert
        patch_weights = weights[mask]  # [n_assigned, k]
        weight_mask = (indices[mask] == expert_id).float()  # [n_assigned, k]
        w = (patch_weights * weight_mask).sum(dim=1)  # [n_assigned]
        w = w / (w.sum() + 1e-8)  # normalize

        if expert_id == 0:  # Gated
            A = self.gate_w(torch.tanh(self.gate_V(h_expert)) * torch.sigmoid(self.gate_U(h_expert)))
            # Weighted softmax
            A = F.softmax(A.squeeze(-1) / 0.1, dim=0)
            out = (A * w * h_expert.T).sum(dim=1)
        elif expert_id == 1:  # Sigmoid
            A = self.sigmoid_w(self.sigmoid_U(h_expert))
            A = F.softmax(A.squeeze(-1) / 0.1, dim=0)
            out = (A * w * h_expert.T).sum(dim=1)
        elif expert_id == 2:  # Softmax
            A = self.softmax_w(self.softmax_V(h_expert))
            A = F.softmax(A.squeeze(-1) / 0.1, dim=0)
            out = (A * w * h_expert.T).sum(dim=1)
        else:  # Max pool
            out = h_expert.max(dim=0).values

        return out

    def encode(self, features: list[torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for bag in features:
            h = self.input_proj(bag)
            indices, weights = self._route(h)
            expert_outputs = []
            for e in range(self.n_experts):
                out = self._forward_one_expert(e, h, indices, weights)
                expert_outputs.append(out)
            expert_stack = torch.stack(expert_outputs, dim=0)  # [E, d]

            # Fusion gate (bag-level)
            gate = F.softmax(self.fusion_gate(h.mean(dim=0)), dim=0)
            bag_emb = (gate.unsqueeze(1) * expert_stack).sum(dim=0)
            embeddings.append(bag_emb)
        return torch.stack(embeddings, dim=0)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.encode(features))


# ---------------------------------------------------------------------------
# Factorized head
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "meanpool": MeanPoolRegressor,
    "meanpool_linear": MeanPoolRegressor,
    "attention": GatedAttentionMIL,
    "multihead_attention": MultiHeadGatedAttentionMIL,
    "dualpath": DualPathMIL,
    "transformer": TransformerMIL,
    "topk_attention": TopKGatedAttentionMIL,
    "deepsets": DeepSetsMIL,
    "lowrank_attention": LowRankGatedAttentionMIL,
    "perceiver": PerceiverMIL,
    "prototype": PrototypeMIL,
    "sinkhorn": SinkhornMIL,
    "moae": MixtureAttentionMIL,
    "dualmoe": DualMoE,
    "crossattn_fusion": CrossAttentionFusionMIL,
    "progressive": ProgressiveExpertMIL,
    "sparse_top2": SparseTop2MoAE,
}


def build_model(model_name: str, in_dim: int = 768, out_dim: int = 6855,
                hidden_dim: int = 256, **kwargs) -> nn.Module:
    dropout = kwargs.get("dropout", 0.3)

    if model_name == "meanpool":
        hidden = kwargs.get("hidden_dims", [512, 256, 128])
        return MeanPoolRegressor(in_dim=in_dim, hidden_dims=hidden, out_dim=out_dim, dropout=dropout)

    if model_name == "meanpool_linear":
        return MeanPoolRegressor(in_dim=in_dim, hidden_dims=[], out_dim=out_dim, dropout=dropout)

    if model_name == "attention":
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "bn")
        residual = kwargs.get("residual", False)
        return GatedAttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                                 hidden_dims=hidden, out_dim=out_dim,
                                 dropout=dropout, norm=norm, residual=residual)

    if model_name == "multihead_attention":
        n_heads = kwargs.get("n_heads", 4)
        # Adjust hidden_dim to be divisible by n_heads
        mh_hidden = (hidden_dim // n_heads) * n_heads
        if mh_hidden < n_heads:
            mh_hidden = n_heads
        hidden = kwargs.get("hidden_dims", [mh_hidden, mh_hidden // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return MultiHeadGatedAttentionMIL(
            in_dim=in_dim, hidden_dim=mh_hidden, n_heads=n_heads,
            hidden_dims=hidden, out_dim=out_dim,
            dropout=dropout, norm=norm, residual=residual)

    if model_name == "dualpath":
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return DualPathMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                           hidden_dims=hidden, out_dim=out_dim,
                           dropout=dropout, norm=norm, residual=residual)

    if model_name == "transformer":
        n_layers = kwargs.get("n_layers", 2)
        n_heads = kwargs.get("n_heads", 8)
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return TransformerMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                              n_layers=n_layers, n_heads=n_heads,
                              hidden_dims=hidden, out_dim=out_dim,
                              dropout=dropout, norm=norm, residual=residual)

    if model_name == "topk_attention":
        top_k = kwargs.get("top_k", 64)
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "bn")
        residual = kwargs.get("residual", False)
        return TopKGatedAttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                                     top_k=top_k, hidden_dims=hidden,
                                     out_dim=out_dim, dropout=dropout,
                                     norm=norm, residual=residual)

    if model_name == "deepsets":
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return DeepSetsMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                           hidden_dims=hidden, out_dim=out_dim,
                           dropout=dropout, norm=norm, residual=residual)

    if model_name == "lowrank_attention":
        latent_dim = kwargs.get("latent_dim", 64)
        use_skip = kwargs.get("use_skip", True)
        return LowRankGatedAttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                                        latent_dim=latent_dim, out_dim=out_dim,
                                        dropout=dropout, use_skip=use_skip)

    if model_name == "perceiver":
        n_latents = kwargs.get("n_latents", 16)
        n_layers = kwargs.get("n_layers", 2)
        n_heads = kwargs.get("n_heads", 4)
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return PerceiverMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                            n_latents=n_latents, n_layers=n_layers, n_heads=n_heads,
                            hidden_dims=hidden, out_dim=out_dim,
                            dropout=dropout, norm=norm, residual=residual)

    if model_name == "prototype":
        n_prototypes = kwargs.get("n_prototypes", 32)
        temperature = kwargs.get("temperature", 1.0)
        hidden = kwargs.get("hidden_dims", [hidden_dim + n_prototypes,
                                             (hidden_dim + n_prototypes) // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return PrototypeMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                            n_prototypes=n_prototypes, temperature=temperature,
                            hidden_dims=hidden, out_dim=out_dim,
                            dropout=dropout, norm=norm, residual=residual)

    if model_name == "sinkhorn":
        n_slots = kwargs.get("n_slots", 16)
        n_sinkhorn_iters = kwargs.get("n_sinkhorn_iters", 3)
        sinkhorn_epsilon = kwargs.get("sinkhorn_epsilon", 0.05)
        bag_dim = n_slots * hidden_dim
        hidden = kwargs.get("hidden_dims", [bag_dim // 4, bag_dim // 8])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return SinkhornMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                           n_slots=n_slots, n_sinkhorn_iters=n_sinkhorn_iters,
                           sinkhorn_epsilon=sinkhorn_epsilon,
                           hidden_dims=hidden, out_dim=out_dim,
                           dropout=dropout, norm=norm, residual=residual)

    if model_name == "moae":
        n_experts = kwargs.get("n_experts", 4)
        hidden = kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2])
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return MixtureAttentionMIL(in_dim=in_dim, hidden_dim=hidden_dim,
                                   n_experts=n_experts,
                                   hidden_dims=hidden, out_dim=out_dim,
                                   dropout=dropout, norm=norm, residual=residual)

    if model_name == "dualmoe":
        n_experts = kwargs.get("n_experts", 4)
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return DualMoE(in_dim=in_dim, hidden_dim=hidden_dim,
                       n_experts=n_experts,
                       hidden_dims=kwargs.get("hidden_dims", [hidden_dim * 2, hidden_dim]),
                       out_dim=out_dim, dropout=dropout,
                       norm=norm, residual=residual)

    if model_name == "crossattn_fusion":
        n_heads = kwargs.get("n_heads", 4)
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return CrossAttentionFusionMIL(
            in_dim=in_dim, hidden_dim=hidden_dim, n_heads=n_heads,
            hidden_dims=kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2]),
            out_dim=out_dim, dropout=dropout,
            norm=norm, residual=residual)

    if model_name == "progressive":
        n_experts = kwargs.get("n_experts", 4)
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return ProgressiveExpertMIL(
            in_dim=in_dim, hidden_dim=hidden_dim, n_experts=n_experts,
            hidden_dims=kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2]),
            out_dim=out_dim, dropout=dropout,
            norm=norm, residual=residual)

    if model_name == "sparse_top2":
        n_experts = kwargs.get("n_experts", 4)
        top_k = kwargs.get("top_k", 2)
        norm = kwargs.get("norm", "ln")
        residual = kwargs.get("residual", True)
        return SparseTop2MoAE(
            in_dim=in_dim, hidden_dim=hidden_dim,
            n_experts=n_experts, top_k=top_k,
            hidden_dims=kwargs.get("hidden_dims", [hidden_dim, hidden_dim // 2]),
            out_dim=out_dim, dropout=dropout,
            norm=norm, residual=residual)

    raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
