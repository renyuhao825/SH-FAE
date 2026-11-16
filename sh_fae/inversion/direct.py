"""Direct curve expressivity inversion for SH-FAE.

Optimizes latent vector z against observed/target SWRC or HCF points without forward PDE.
Used for expressing unseen constitutive families (e.g., Lu 2016 SWRC, Gou 2023 HCF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..models.decoder import ConstrainedDecoder


@dataclass
class DirectInversionResult:
    z: torch.Tensor
    loss: float
    history: list[dict[str, float]]
    decoded_curves: dict[str, torch.Tensor]


def invert_direct_curve(
    decoder: ConstrainedDecoder,
    coords: np.ndarray | torch.Tensor,
    values: np.ndarray | torch.Tensor,
    channel_id: int = 0,  # 0 for SWRC, 1 for HCF
    z_init: torch.Tensor | None = None,
    z_ref: torch.Tensor | None = None,
    latent_dim: int = 64,
    n_steps: int = 3000,
    lr: float = 0.03,
    lambda_z: float = 1e-3,
    grad_clip_norm: float = 1.0,
    device: str | torch.device = "cpu",
    progress_fn: Any | None = None,
) -> DirectInversionResult:
    """Optimize latent code z to fit observed constitutive curve points."""
    device = torch.device(device)
    decoder = decoder.to(device)
    decoder.eval()

    # Prepare inputs
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=device)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
    if coords_t.ndim == 1:
        coords_t = coords_t.unsqueeze(0)
    if values_t.ndim == 1:
        values_t = values_t.unsqueeze(0)

    # Normalize coordinate to [-1, 1]
    if channel_id == 0:
        # x coordinate in [x_min, x_max]
        norm_coord = 2.0 * (coords_t - decoder.x_min) / (decoder.x_max - decoder.x_min) - 1.0
    else:
        # u coordinate in [0, 1]
        norm_coord = 2.0 * coords_t - 1.0
    norm_coord = norm_coord.unsqueeze(-1)

    channel_t = torch.full((1, coords_t.shape[1]), channel_id, dtype=torch.long, device=device)

    # Initialize latent token z
    if z_init is not None:
        z_param = torch.nn.Parameter(z_init.to(device=device, dtype=torch.float32).clone())
    else:
        z_param = torch.nn.Parameter(torch.zeros(1, latent_dim, dtype=torch.float32, device=device))

    if z_ref is None:
        z_ref = torch.zeros_like(z_param.data)
    else:
        z_ref = z_ref.to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([z_param], lr=lr)
    history: list[dict[str, float]] = []

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = decoder(z_param, norm_coord, channel_t)["pred"]

        data_loss = torch.mean((pred - values_t) ** 2)
        prior_loss = torch.mean((z_param - z_ref) ** 2)
        loss = data_loss + lambda_z * prior_loss

        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([z_param], grad_clip_norm)
        optimizer.step()

        if step % 100 == 0 or step == n_steps - 1:
            record = {
                "step": float(step),
                "loss": float(loss.item()),
                "data_loss": float(data_loss.item()),
                "prior_loss": float(prior_loss.item()),
            }
            history.append(record)
            if progress_fn:
                progress_fn(record)

    with torch.no_grad():
        curves = decoder.decode_curves(z_param)

    return DirectInversionResult(
        z=z_param.detach(),
        loss=float(history[-1]["loss"]),
        history=history,
        decoded_curves=curves,
    )
