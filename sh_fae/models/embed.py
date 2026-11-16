"""Point embedding and MLP building blocks for SH-FAE."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointEmbed(nn.Module):
    """Embeds individual context tuples: (normalized coordinate, channel id, value)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        part_dim = embed_dim
        self.coord = nn.Sequential(nn.Linear(1, part_dim), nn.SiLU())
        self.channel = nn.Embedding(2, part_dim)  # 0: SWRC, 1: HCF
        self.value = nn.Sequential(nn.Linear(1, part_dim), nn.SiLU())
        self.project = nn.Sequential(
            nn.Linear(3 * part_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        coord: torch.Tensor,
        channel: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.project(
            torch.cat(
                [
                    self.coord(coord),
                    self.channel(channel),
                    self.value(value),
                ],
                dim=-1,
            )
        )
