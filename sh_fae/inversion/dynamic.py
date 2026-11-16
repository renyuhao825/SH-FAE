"""Differentiable Richards PDE-constrained latent inversion for SH-FAE.

Optimizes latent vector z and physical scale parameters [theta_r, theta_s, log10_K]
by backpropagating through the differentiable Richards equation solver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from .transforms import BoundedScaleParameter
from ..models.decoder import ConstrainedDecoder
from ..physics.constitutive import TorchAEConstitutive
from ..physics.grid import Grid1D
from ..physics.material import Material
from ..physics.solver import BoundaryConditions, NewtonOptions, solve_richards


@dataclass
class PDEInversionResult:
    z: torch.Tensor
    theta_r: float
    theta_s: float
    intrinsic_k: float
    loss: float
    history: list[dict[str, float]]
    simulated_theta: torch.Tensor
    decoded_curves: dict[str, torch.Tensor]


def invert_pde_constrained(
    decoder: ConstrainedDecoder,
    grid: Grid1D,
    boundary: BoundaryConditions,
    initial_psi: float | torch.Tensor,
    T: float,
    dt: float,
    obs_times_idx: list[int] | np.ndarray,
    obs_depths_idx: list[int] | np.ndarray,
    obs_theta: np.ndarray | torch.Tensor,
    obs_sigma: float = 0.01,
    z_init: torch.Tensor | None = None,
    z_ref: torch.Tensor | None = None,
    scale_configs: dict[str, dict[str, float]] | None = None,
    n_steps: int = 1500,
    lr_z: float = 1e-3,
    lambda_z: float = 1e-3,
    lambda_scale: float = 0.0,
    grad_clip_norm: float = 0.5,
    solver_options: NewtonOptions | None = None,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> PDEInversionResult:
    """Run dynamic PDE-constrained inversion."""
    device = torch.device(device)
    dtype = grid.z.dtype
    decoder = decoder.to(device)
    decoder.eval()

    obs_theta_t = torch.as_tensor(obs_theta, dtype=dtype, device=device)

    # Setup latent parameter z
    if z_init is not None:
        z_param = torch.nn.Parameter(z_init.to(device=device, dtype=dtype).clone())
    else:
        z_param = torch.nn.Parameter(torch.zeros(1, decoder.latent_dim, dtype=dtype, device=device))

    z_ref_t = torch.zeros_like(z_param.data) if z_ref is None else z_ref.to(device=device, dtype=dtype)

    # Setup physical scale parameters
    scale_configs = scale_configs or {}
    theta_r_cfg = scale_configs.get("theta_r", {"init": 0.0, "mu": 0.0, "sigma": 0.0, "lr": 1e-2})
    theta_s_cfg = scale_configs.get("theta_s", {"init": 0.35, "mu": 0.35, "sigma": 0.15, "lr": 1e-2})
    k_cfg = scale_configs.get("log10_k", {"init": -13.0, "mu": -13.0, "sigma": 2.5, "lr": 1e-2})

    opt_theta_r = theta_r_cfg.get("sigma", 0.0) > 0.0
    opt_theta_s = theta_s_cfg.get("sigma", 0.0) > 0.0
    opt_k = k_cfg.get("sigma", 0.0) > 0.0

    p_theta_r = BoundedScaleParameter("theta_r", theta_r_cfg["init"], theta_r_cfg["mu"], theta_r_cfg["sigma"], lr=theta_r_cfg["lr"], device=device, dtype=dtype) if opt_theta_r else None
    p_theta_s = BoundedScaleParameter("theta_s", theta_s_cfg["init"], theta_s_cfg["mu"], theta_s_cfg["sigma"], lr=theta_s_cfg["lr"], device=device, dtype=dtype) if opt_theta_s else None
    p_log10_k = BoundedScaleParameter("log10_k", k_cfg["init"], k_cfg["mu"], k_cfg["sigma"], lr=k_cfg["lr"], device=device, dtype=dtype) if opt_k else None

    # Optimizer param groups
    param_groups = [{"params": [z_param], "lr": lr_z}]
    if opt_theta_r:
        param_groups.append({"params": [p_theta_r.param], "lr": p_theta_r.lr})
    if opt_theta_s:
        param_groups.append({"params": [p_theta_s.param], "lr": p_theta_s.lr})
    if opt_k:
        param_groups.append({"params": [p_log10_k.param], "lr": p_log10_k.lr})

    optimizer = torch.optim.Adam(param_groups)
    history: list[dict[str, float]] = []

    pbar = tqdm(range(n_steps), desc="PDE Inversion", disable=not progress)

    best_loss = float("inf")
    best_state: dict[str, Any] = {}

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        current_theta_r = p_theta_r.value if opt_theta_r else torch.tensor(theta_r_cfg.get("init", 0.0), dtype=dtype, device=device)
        current_theta_s = p_theta_s.value if opt_theta_s else torch.tensor(theta_s_cfg.get("init", 0.35), dtype=dtype, device=device)
        current_log10_k = p_log10_k.value if opt_k else torch.tensor(k_cfg.get("init", -13.0), dtype=dtype, device=device)
        current_k = torch.pow(torch.tensor(10.0, dtype=dtype, device=device), current_log10_k)

        constitutive = TorchAEConstitutive(
            decoder=decoder,
            z=z_param,
            theta_r=current_theta_r,
            theta_s=current_theta_s,
        )
        material = Material(intrinsic_permeability_m2=current_k)

        # Run forward differentiable simulation
        sim_res = solve_richards(
            grid=grid,
            constitutive=constitutive,
            material=material,
            boundary=boundary,
            initial_psi=initial_psi,
            T=T,
            dt=dt,
            options=solver_options,
        )

        # Extract predictions at observation coordinates
        # sim_res.theta is [N_time, Ns]
        pred_theta_all = sim_res.theta
        pred_obs = pred_theta_all[obs_times_idx, :][:, obs_depths_idx]

        # Normalized data misfit
        data_loss = torch.mean(((pred_obs - obs_theta_t) / obs_sigma) ** 2)
        prior_z = torch.mean((z_param - z_ref_t) ** 2)
        loss = data_loss + lambda_z * prior_z

        if lambda_scale > 0.0:
            scale_reg = 0.0
            if opt_theta_r:
                scale_reg += p_theta_r.param.pow(2)
            if opt_theta_s:
                scale_reg += p_theta_s.param.pow(2)
            if opt_k:
                scale_reg += p_log10_k.param.pow(2)
            loss += lambda_scale * scale_reg

        if not sim_res.converged or not torch.isfinite(loss):
            # Divergence guard: reduce learning rate
            for g in optimizer.param_groups:
                g["lr"] *= 0.5
            pbar.set_postfix({"status": "diverged, lr halved", "loss": float(loss.item()) if torch.isfinite(loss) else "nan"})
            continue

        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([z_param], grad_clip_norm)

        optimizer.step()

        val_loss = float(loss.item())
        record = {
            "step": float(step),
            "loss": val_loss,
            "data_loss": float(data_loss.item()),
            "theta_r": float(current_theta_r.item()),
            "theta_s": float(current_theta_s.item()),
            "log10_k": float(current_log10_k.item()),
        }
        history.append(record)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                "z": z_param.detach().clone(),
                "theta_r": float(current_theta_r.item()),
                "theta_s": float(current_theta_s.item()),
                "intrinsic_k": float(current_k.item()),
                "simulated_theta": sim_res.theta.detach().cpu(),
            }

        pbar.set_postfix({"loss": f"{val_loss:.4e}", "best": f"{best_loss:.4e}"})

    pbar.close()

    with torch.no_grad():
        final_curves = decoder.decode_curves(best_state.get("z", z_param.detach()))

    return PDEInversionResult(
        z=best_state.get("z", z_param.detach()),
        theta_r=best_state.get("theta_r", float(current_theta_r.item())),
        theta_s=best_state.get("theta_s", float(current_theta_s.item())),
        intrinsic_k=best_state.get("intrinsic_k", float(current_k.item())),
        loss=best_loss,
        history=history,
        simulated_theta=best_state.get("simulated_theta", sim_res.theta.detach().cpu()),
        decoded_curves=final_curves,
    )
