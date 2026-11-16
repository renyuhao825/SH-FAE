"""Physically constrained density-integration decoder for SWRC and HCF."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .embed import MLP
from ..utils.config import ModelConfig


def cumulative_trapezoid(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Cumulative trapezoid integration along the last dimension."""
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    zeros = torch.zeros(*y.shape[:-1], 1, dtype=y.dtype, device=y.device)
    return torch.cat([zeros, torch.cumsum(area, dim=-1)], dim=-1)


def interp1d_batched(
    grid: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Differentiable linear interpolation on a shared 1-D grid.
    
    Args:
        grid: [n_grid]
        values: [B, n_grid]
        query: [B, n_query]
    Returns:
        interpolated: [B, n_query]
    """
    query = query.clamp(float(grid[0]), float(grid[-1]))
    right = torch.searchsorted(grid.contiguous(), query.contiguous(), right=False)
    right = right.clamp(min=1, max=grid.numel() - 1)
    left = right - 1

    x0 = grid[left]
    x1 = grid[right]
    denom = (x1 - x0).clamp_min(torch.finfo(values.dtype).eps)
    weight = ((query - x0) / denom).to(values.dtype)

    left_values = torch.gather(values, dim=1, index=left)
    right_values = torch.gather(values, dim=1, index=right)
    return left_values + weight * (right_values - left_values)


class ConstrainedDecoder(nn.Module):
    """Decoder predicting positive densities and integrating them into monotonic curves.
    
    Guarantees:
    - Monotonically decreasing SWRC: d S_e / dx <= 0
    - Monotonically increasing HCF: d G_K / du >= 0
    - Boundedness: S_e in [0, 1], G_K in [0, 1]
    - Exact endpoint boundaries: S_e(x_min)=1, S_e(x_max)=0, G_K(0)=0, G_K(1)=1
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.y_min = float(config.y_min)
        self.density_eps = float(config.density_eps)
        self.x_min = float(config.x_min)
        self.x_max = float(config.x_max)

        # Density heads
        self.q_head = MLP(
            in_dim=config.latent_dim + 1,
            hidden_dim=config.decoder_hidden_dim,
            out_dim=1,
            num_layers=config.decoder_layers,
        )
        self.r_head = MLP(
            in_dim=config.latent_dim + 1,
            hidden_dim=config.decoder_hidden_dim,
            out_dim=1,
            num_layers=config.decoder_layers,
        )

        # Internal quadrature grids
        x_grid = torch.linspace(config.x_min, config.x_max, config.x_int_points)
        u_grid = torch.linspace(0.0, 1.0, config.u_int_points)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer(
            "x_grid_norm",
            2.0 * (x_grid - config.x_min) / (config.x_max - config.x_min) - 1.0,
        )
        self.register_buffer("u_grid", u_grid)
        self.register_buffer("u_grid_norm", 2.0 * u_grid - 1.0)

    def _density(
        self,
        z: torch.Tensor,
        coord_norm: torch.Tensor,
        head: nn.Module,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        n_points = coord_norm.numel()
        z_expanded = z[:, None, :].expand(batch_size, n_points, z.shape[-1])
        coord = coord_norm[None, :, None].expand(batch_size, n_points, 1)
        raw = head(torch.cat([z_expanded, coord], dim=-1)).squeeze(-1)
        return F.softplus(raw) + self.density_eps

    def decode_curves(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent vector z into continuous curves on internal canonical grids."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        q = self._density(z, self.x_grid_norm, self.q_head)
        r = self._density(z, self.u_grid_norm, self.r_head)

        c_x = cumulative_trapezoid(q, self.x_grid)
        c_u = cumulative_trapezoid(r, self.u_grid)
        c_x = c_x / c_x[:, -1:].clamp_min(torch.finfo(c_x.dtype).eps)
        c_u = c_u / c_u[:, -1:].clamp_min(torch.finfo(c_u.dtype).eps)

        se = (1.0 - c_x).clamp(0.0, 1.0)
        gk = c_u.clamp(0.0, 1.0)
        ykr = self.y_min + (0.0 - self.y_min) * gk
        return {
            "x": self.x_grid,
            "x_norm": self.x_grid_norm,
            "u": self.u_grid,
            "u_norm": self.u_grid_norm,
            "q_density": q,
            "r_density": r,
            "se": se,
            "gk": gk,
            "ykr": ykr,
        }

    def forward(
        self,
        z: torch.Tensor,
        query_coord: torch.Tensor,
        query_channel: torch.Tensor,
        return_curves: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Query reconstructed constitutive values at arbitrary coordinates.
        
        Args:
            z: [B, latent_dim]
            query_coord: [B, n_query, 1] normalized coordinate in [-1, 1]
            query_channel: [B, n_query] 0 for SWRC, 1 for HCF
            return_curves: whether to return full decoded curves on canonical grid
        """
        curves = self.decode_curves(z)
        query = query_coord.squeeze(-1)
        # Denormalize coordinates to canonical scale
        query_x = self.x_min + 0.5 * (query + 1.0) * (self.x_max - self.x_min)
        query_u = 0.5 * (query + 1.0)

        pred_swrc = interp1d_batched(curves["x"], curves["se"], query_x)
        pred_hcf = interp1d_batched(curves["u"], curves["gk"], query_u)
        pred = torch.where(query_channel == 0, pred_swrc, pred_hcf)

        output = {
            "pred": pred,
            "pred_swrc": pred_swrc,
            "pred_hcf": pred_hcf,
        }
        if return_curves:
            output["curves"] = curves
        return output
