"""Bounded parameter transformations for physical scales in inversion."""

from __future__ import annotations

import torch


def forward_tanh_transform(unconstrained: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Map unconstrained variable in (-inf, inf) to bounded range (mu - sigma, mu + sigma)."""
    return mu + sigma * torch.tanh(unconstrained)


def inverse_tanh_transform(bounded: float | torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Map bounded parameter in (mu - sigma, mu + sigma) to unconstrained space."""
    if not isinstance(bounded, torch.Tensor):
        bounded = torch.tensor(bounded, dtype=torch.float64)
    normalized = torch.clamp((bounded - mu) / sigma, -0.999999, 0.999999)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


class BoundedScaleParameter:
    """Encapsulates an unconstrained trainable parameter mapped to bounded physical space."""

    def __init__(
        self,
        name: str,
        init_val: float,
        mu: float,
        sigma: float,
        lr: float = 1e-2,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.name = name
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.lr = float(lr)

        # Initialize unconstrained parameter
        if abs(init_val - mu) < sigma:
            unconstrained_init = inverse_tanh_transform(init_val, mu, sigma)
        else:
            unconstrained_init = torch.tensor(0.0, dtype=dtype)
        self.param = torch.nn.Parameter(unconstrained_init.to(device=device, dtype=dtype))

    @property
    def value(self) -> torch.Tensor:
        """Physical bounded value."""
        return forward_tanh_transform(self.param, self.mu, self.sigma)
