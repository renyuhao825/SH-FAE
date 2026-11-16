"""Transformer set encoder for functional autoencoding."""

from __future__ import annotations

import torch
from torch import nn

from .embed import PointEmbed
from ..utils.config import ModelConfig


class SetEncoder(nn.Module):
    """Permutation-invariant set encoder aggregating context tokens into latent vector z."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.point_embed = PointEmbed(config.point_embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.point_embed_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.point_embed_dim * config.encoder_mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.norm = nn.LayerNorm(config.point_embed_dim)
        self.latent = nn.Linear(config.point_embed_dim, config.latent_dim)

    def forward(
        self,
        coord: torch.Tensor,
        channel: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context set into latent token z: [B, latent_dim]."""
        tokens = self.point_embed(coord, channel, value)
        tokens = self.blocks(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.latent(pooled)
