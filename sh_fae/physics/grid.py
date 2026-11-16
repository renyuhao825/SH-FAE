"""One-dimensional cell-centered finite volume grid with ghost cells."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid1D:
    """Uniform cell-centered grid with two boundary ghost cells for 1D column [-Z, 0]."""

    Z: float
    Ns: int
    z: torch.Tensor
    z_internal: torch.Tensor
    face_z: torch.Tensor
    dz: torch.Tensor
    dz_internal: torch.Tensor

    @classmethod
    def uniform(
        cls,
        Z: float,
        Ns: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "Grid1D":
        if Z <= 0.0:
            raise ValueError("Column height Z must be positive.")
        if Ns < 1:
            raise ValueError("Cell count Ns must be positive.")

        device = torch.device("cpu" if device is None else device)
        dz_value = Z / Ns
        z_internal = -Z + (torch.arange(Ns, dtype=dtype, device=device) + 0.5) * dz_value
        z = torch.empty(Ns + 2, dtype=dtype, device=device)
        z[0] = -Z - 0.5 * dz_value
        z[1:-1] = z_internal
        z[-1] = 0.5 * dz_value
        face_z = torch.linspace(-Z, 0.0, Ns + 1, dtype=dtype, device=device)
        cell_widths = torch.full((Ns + 2,), dz_value, dtype=dtype, device=device)
        return cls(
            Z=Z,
            Ns=Ns,
            z=z,
            z_internal=z_internal,
            face_z=face_z,
            dz=cell_widths,
            dz_internal=cell_widths[1:-1].clone(),
        )
