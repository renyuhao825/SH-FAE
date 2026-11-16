"""Soil fluid and hydraulic material properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from .grid import Grid1D

SECONDS_PER_DAY = 86400.0
TensorLike = Union[float, int, torch.Tensor]


def _as_tensor(value: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(value, dtype=ref.dtype, device=ref.device)


@dataclass(frozen=True)
class Material:
    """Hydraulic material properties.
    
    Accepts intrinsic_permeability_m2 (K) or saturated_conductivity_m_per_day (Ks).
    """

    intrinsic_permeability_m2: TensorLike | None = None
    saturated_conductivity_m_per_day: TensorLike | None = None
    log10_Ks_cm_day: TensorLike | None = None
    rho: TensorLike = 0.99823e3  # water density (kg/m3)
    mu: TensorLike = 1.0005e-3   # water viscosity (Pa s)
    g: TensorLike = 9.80665      # gravity (m/s2)

    def saturated_conductivity(self, grid: Grid1D) -> torch.Tensor:
        ref = grid.z
        if self.saturated_conductivity_m_per_day is not None:
            val = _as_tensor(self.saturated_conductivity_m_per_day, ref)
            return val.expand(grid.Ns + 2) if val.ndim == 0 else val
        if self.log10_Ks_cm_day is not None:
            log10_ks = _as_tensor(self.log10_Ks_cm_day, ref)
            ks_m_day = torch.pow(_as_tensor(10.0, ref), log10_ks) / 100.0
            return ks_m_day.expand(grid.Ns + 2) if ks_m_day.ndim == 0 else ks_m_day
        if self.intrinsic_permeability_m2 is not None:
            intrinsic = _as_tensor(self.intrinsic_permeability_m2, ref)
            rho = _as_tensor(self.rho, ref)
            mu = _as_tensor(self.mu, ref)
            g = _as_tensor(self.g, ref)
            ks_m_day = intrinsic * rho * g / mu * SECONDS_PER_DAY
            return ks_m_day.expand(grid.Ns + 2) if ks_m_day.ndim == 0 else ks_m_day
        raise ValueError("Must provide intrinsic_permeability_m2 or saturated conductivity.")
