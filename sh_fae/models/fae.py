"""Soil Hydraulic Functional AutoEncoder (SH-FAE)."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import ConstrainedDecoder
from .encoder import SetEncoder
from ..utils.config import ModelConfig


class SHFAE(nn.Module):
    """Soil Hydraulic Functional AutoEncoder (SH-FAE).
    
    Encodes sparse/arbitrary constitutive context points into a compact latent vector z,
    and decodes z into continuous, strictly monotonic, bounded SWRC-HCF function pairs.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = SetEncoder(self.config)
        self.decoder = ConstrainedDecoder(self.config)

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode context set into latent tokens z: [B, latent_dim]."""
        return self.encoder(
            batch["context_coord"],
            batch["context_channel"],
            batch["context_value"],
        )

    def decode(
        self,
        z: torch.Tensor,
        query_coord: torch.Tensor,
        query_channel: torch.Tensor,
        return_curves: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode latent tokens z at query coordinates."""
        return self.decoder(
            z,
            query_coord,
            query_channel,
            return_curves=return_curves,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_curves: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode context points and decode at query points."""
        z = self.encode(batch)
        decoded = self.decode(
            z,
            batch["query_coord"],
            batch["query_channel"],
            return_curves=return_curves,
        )
        decoded["z"] = z
        return decoded
