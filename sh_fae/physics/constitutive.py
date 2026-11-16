"""Callable soil hydraulic constitutive relations backed by SH-FAE."""

from __future__ import annotations

from typing import Protocol, Union

import torch

from ..models.decoder import ConstrainedDecoder, interp1d_batched

TensorLike = Union[float, int, torch.Tensor]


def _as_tensor(value: TensorLike, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(value, dtype=ref.dtype, device=ref.device)


class ConstitutiveRelation(Protocol):
    """Protocol for soil hydraulic constitutive relations."""

    def theta(self, psi: torch.Tensor) -> torch.Tensor:
        ...

    def kr(self, psi: torch.Tensor) -> torch.Tensor:
        ...


class TorchAEConstitutive:
    """Differentiable SWRC-HCF constitutive relation evaluated via frozen SH-FAE decoder."""

    def __init__(
        self,
        decoder: ConstrainedDecoder,
        z: torch.Tensor,
        theta_r: TensorLike = 0.0,
        theta_s: TensorLike = 0.45,
        eps: float = 1.0e-7,
    ) -> None:
        self.decoder = decoder
        self.z = z
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.eps = eps
        self._cached_curves: dict[str, torch.Tensor] | None = None

    def get_curves(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        z = _as_tensor(self.z, ref)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        decoder = self.decoder.to(device=ref.device, dtype=ref.dtype)
        return decoder.decode_curves(z)

    def _se_unsat(self, psi: torch.Tensor, curves: dict[str, torch.Tensor]) -> torch.Tensor:
        # Matric suction magnitude s = -psi (in meters), converted to cm: s_cm = -100 * psi
        s_cm = torch.clamp(-100.0 * psi, min=self.eps)
        x = torch.log10(s_cm)
        flat_x = x.reshape(1, -1)
        se = interp1d_batched(curves["x"], curves["se"], flat_x).reshape(psi.shape)
        return se.clamp(0.0, 1.0)

    def theta(self, psi: torch.Tensor) -> torch.Tensor:
        curves = self.get_curves(psi)
        theta_r = _as_tensor(self.theta_r, psi)
        theta_s = _as_tensor(self.theta_s, psi)
        se = self._se_unsat(psi, curves)
        theta_unsat = theta_r + (theta_s - theta_r) * se
        return torch.where(psi < 0.0, theta_unsat, theta_s.expand_as(psi))

    def kr(self, psi: torch.Tensor) -> torch.Tensor:
        curves = self.get_curves(psi)
        se = self._se_unsat(psi, curves)
        flat_se = se.reshape(1, -1)
        ykr = interp1d_batched(curves["u"], curves["ykr"], flat_se).reshape(psi.shape)
        kr_unsat = torch.pow(_as_tensor(10.0, psi), ykr).clamp(0.0, 1.0)
        return torch.where(psi < 0.0, kr_unsat, torch.ones_like(psi))


class VGMConstitutive:
    """Analytical van Genuchten - Mualem (1980) constitutive relation."""

    def __init__(
        self,
        alpha: float,
        n: float,
        theta_r: float = 0.0,
        theta_s: float = 0.33,
        tau: float = 0.5,
    ) -> None:
        self.alpha = float(alpha)
        self.n = float(n)
        self.m = 1.0 - 1.0 / self.n
        self.theta_r = float(theta_r)
        self.theta_s = float(theta_s)
        self.tau = float(tau)

    def theta(self, psi: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(-psi, min=1e-8)
        se = torch.pow(1.0 + torch.pow(self.alpha * s, self.n), -self.m)
        theta_unsat = self.theta_r + (self.theta_s - self.theta_r) * se
        return torch.where(psi < 0.0, theta_unsat, torch.full_like(psi, self.theta_s))

    def kr(self, psi: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(-psi, min=1e-8)
        se = torch.pow(1.0 + torch.pow(self.alpha * s, self.n), -self.m).clamp(0.0, 1.0)
        term = 1.0 - torch.pow(1.0 - torch.pow(se, 1.0 / self.m), self.m)
        kr_unsat = torch.pow(se, self.tau) * torch.pow(term, 2.0)
        return torch.where(psi < 0.0, kr_unsat.clamp(1e-12, 1.0), torch.ones_like(psi))
