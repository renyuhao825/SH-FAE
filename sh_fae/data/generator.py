"""Synthetic soil hydraulic constitutive library generator for SH-FAE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .families import (
    DRY_END_SE_MAX,
    FAMILY_NAMES,
    FAMILY_WEIGHTS,
    X_FLOW_MAX,
    X_MAX,
    X_MIN,
    Y_MIN_DEFAULT,
    add_pdi_like_film,
    cumtrapz,
    eval_boundary,
    eval_brooks_corey,
    eval_durner,
    eval_fredlund_xing_like,
    eval_kosugi,
    eval_nonparametric,
    eval_vgm,
    hcf_bc_closed,
    hcf_numerical_mualem,
    hcf_vgm_closed,
    log_uniform,
    make_u_grid,
    make_x_grid,
    sample_boundary_params,
    sample_brooks_corey_params,
    sample_durner_params,
    sample_fredlund_xing_params,
    sample_kosugi_params,
    sample_nonparametric_params,
    sample_pdi_like_params,
    sample_vgm_params,
    smooth_signal,
)


@dataclass
class GeneratedSample:
    family_id: int
    source_model: int
    se_x: np.ndarray
    ykr_u: np.ndarray
    gkr_u: np.ndarray
    ykr_x: np.ndarray
    theta_r: float
    theta_s: float
    log10_ks_cm_day: float
    params: dict[str, Any]


def make_loss_weight_x(x_grid: np.ndarray) -> np.ndarray:
    weights = np.ones_like(x_grid, dtype=np.float64)
    weights[(x_grid >= X_MIN) & (x_grid <= X_FLOW_MAX)] = 2.0
    weights[x_grid > X_FLOW_MAX] = 0.6
    return weights.astype(np.float32)


def make_loss_weight_u(u_grid: np.ndarray) -> np.ndarray:
    weights = np.ones_like(u_grid, dtype=np.float64)
    weights[u_grid < 0.05] = 1.5
    weights[u_grid > 0.85] = 1.5
    return weights.astype(np.float32)


def sample_global_scales(rng: np.random.Generator) -> tuple[float, float, float]:
    theta_s = float(rng.uniform(0.30, 0.60))
    theta_r_hi = min(0.18, theta_s - 0.08)
    theta_r = float(rng.uniform(0.0, max(0.01, theta_r_hi)))
    log10_ks_cm_day = float(rng.uniform(-2.0, 3.5))
    return theta_r, theta_s, log10_ks_cm_day


def sample_family_params(family_id: int, rng: np.random.Generator) -> dict[str, Any]:
    if family_id == 0:
        return sample_vgm_params(rng)
    if family_id == 1:
        return sample_brooks_corey_params(rng)
    if family_id == 2:
        return sample_kosugi_params(rng)
    if family_id == 3:
        return sample_fredlund_xing_params(rng)
    if family_id == 4:
        return sample_pdi_like_params(rng)
    if family_id == 5:
        return sample_durner_params(rng)
    if family_id == 6:
        return sample_nonparametric_params(rng)
    if family_id == 7:
        return sample_boundary_params(rng)
    raise ValueError(f"Unknown family_id={family_id}")


def evaluate_swcc(
    family_id: int,
    params: dict[str, Any],
    x_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if family_id == 0:
        return eval_vgm(x_grid, params["alpha_cm_inv"], params["n"])
    if family_id == 1:
        return eval_brooks_corey(x_grid, params["hb_cm"], params["lambda"])
    if family_id == 2:
        return eval_kosugi(x_grid, params["hm_cm"], params["sigma"])
    if family_id == 3:
        return eval_fredlund_xing_like(x_grid, params)
    if family_id == 4:
        return eval_vgm(x_grid, params["alpha_cm_inv"], params["n"])
    if family_id == 5:
        return eval_durner(x_grid, params)
    if family_id == 6:
        return eval_nonparametric(x_grid, params, rng)
    if family_id == 7:
        return eval_boundary(x_grid, params)
    raise ValueError(f"Unknown family_id={family_id}")


def accept_swcc(se_x: np.ndarray, family_id: int) -> tuple[bool, str]:
    boundary = family_id == 7
    if not np.all(np.isfinite(se_x)):
        return False, "swcc_nonfinite"
    if np.min(se_x) < -1e-7 or np.max(se_x) > 1.0 + 1e-7:
        return False, "swcc_range"
    se = np.clip(se_x, 0.0, 1.0)
    if np.any(np.diff(se) > 1e-6):
        return False, "swcc_nonmonotone"
    wet_threshold = 0.70 if boundary else 0.85
    if se[0] < wet_threshold:
        return False, "swcc_wet_end"
    if se[-1] > DRY_END_SE_MAX:
        return False, "swcc_dry_end"
    drop_threshold = 0.05 if boundary else 0.20
    if se[0] - se[-1] < drop_threshold:
        return False, "swcc_low_dynamic_range"
    if not np.any((se > 0.05) & (se < 0.95)):
        return False, "swcc_no_transition"
    max_jump = 0.45 if family_id in {1, 7} else 0.25
    if np.max(np.abs(np.diff(se))) > max_jump:
        return False, "swcc_jump"
    return True, "ok"


def derive_base_ykr(
    family_id: int,
    params: dict[str, Any],
    se_x: np.ndarray,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    y_min: float,
    nu_int: int = 512,
) -> np.ndarray:
    tau = float(params.get("tau", 0.5))
    if family_id == 0:
        return hcf_vgm_closed(u_grid, params["m"], tau, y_min)
    if family_id == 1:
        return hcf_bc_closed(u_grid, params["lambda"], tau, y_min)
    if family_id == 4:
        y_mualem = hcf_vgm_closed(u_grid, params["m"], tau, y_min)
        return add_pdi_like_film(y_mualem, u_grid, params, y_min)
    if family_id == 7 and params.get("boundary_mode") in {"sharp_vgm", "flat_vgm"}:
        return hcf_vgm_closed(u_grid, params["m"], tau, y_min)
    if family_id == 7 and params.get("boundary_mode") == "extreme_bc":
        return hcf_bc_closed(u_grid, params["lambda"], tau, y_min)
    return hcf_numerical_mualem(se_x, x_grid, u_grid, tau, y_min, nu_int)


def perturb_ykr_density(
    ykr_base: np.ndarray,
    u_grid: np.ndarray,
    y_min: float,
    rng: np.random.Generator,
    amplitude: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    g_base_norm = (np.clip(ykr_base, y_min, 0.0) - y_min) / (0.0 - y_min)
    g_base_norm = np.maximum.accumulate(g_base_norm)
    density = np.gradient(g_base_norm, u_grid, edge_order=1)
    density = np.clip(density, 0.0, np.inf)

    n_knots = int(rng.integers(6, 13))
    knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float64)
    knot_amp = rng.normal(0.0, amplitude, size=n_knots)
    field = np.interp(u_grid, knots, knot_amp)
    length_scale = float(rng.uniform(0.15, 0.35))
    sigma_points = length_scale / float(np.mean(np.diff(u_grid)))
    field = smooth_signal(field, sigma_points)

    perturbed_density = (density + 1e-4) * np.exp(field)
    cumulative = cumtrapz(perturbed_density, u_grid)
    total = float(cumulative[-1])
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(ykr_base, np.nan), np.full_like(ykr_base, np.nan)
    gkr = cumulative / total
    gkr[0] = 0.0
    gkr[-1] = 1.0
    ykr = y_min + (0.0 - y_min) * gkr
    ykr[0] = y_min
    ykr[-1] = 0.0
    return ykr, gkr


def accept_hcf(ykr_u: np.ndarray, y_min: float) -> tuple[bool, str]:
    if not np.all(np.isfinite(ykr_u)):
        return False, "hcf_nonfinite"
    if np.any(np.diff(ykr_u) < -1e-6):
        return False, "hcf_nonmonotone"
    if abs(float(ykr_u[-1])) >= 1e-5:
        return False, "hcf_endpoint"
    if float(np.min(ykr_u)) < y_min - 1e-5:
        return False, "hcf_lower_bound"
    if float(ykr_u[-1] - ykr_u[0]) < 4.0:
        return False, "hcf_low_dynamic_range"
    return True, "ok"


def generate_single_sample(
    family_id: int,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    rng: np.random.Generator,
    y_min: float = Y_MIN_DEFAULT,
) -> GeneratedSample | None:
    theta_r, theta_s, log10_ks_cm_day = sample_global_scales(rng)
    params = sample_family_params(family_id, rng)
    se_x = evaluate_swcc(family_id, params, x_grid, rng)

    ok_swcc, _ = accept_swcc(se_x, family_id)
    if not ok_swcc:
        return None

    ykr_base = derive_base_ykr(family_id, params, se_x, x_grid, u_grid, y_min)
    if not np.all(np.isfinite(ykr_base)):
        return None

    ykr_u, gkr_u = perturb_ykr_density(ykr_base, u_grid, y_min, rng)
    ok_hcf, _ = accept_hcf(ykr_u, y_min)
    if not ok_hcf:
        return None

    ykr_x = np.interp(np.clip(se_x, 0.0, 1.0), u_grid, ykr_u)
    return GeneratedSample(
        family_id=family_id,
        source_model=family_id,
        se_x=se_x.astype(np.float32),
        ykr_u=ykr_u.astype(np.float32),
        gkr_u=gkr_u.astype(np.float32),
        ykr_x=ykr_x.astype(np.float32),
        theta_r=theta_r,
        theta_s=theta_s,
        log10_ks_cm_day=log10_ks_cm_day,
        params=params,
    )


def generate_library(
    n_samples: int = 50000,
    nx: int = 384,
    nu: int = 128,
    seed: int = 42,
    y_min: float = Y_MIN_DEFAULT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Generate n_samples valid constitutive pairs and package into NPZ arrays."""
    rng = np.random.default_rng(seed)
    x_grid = make_x_grid(nx)
    u_grid = make_u_grid(nu)

    samples: list[GeneratedSample] = []
    pbar = tqdm(total=n_samples, desc="Generating synthetic library", disable=not progress)

    while len(samples) < n_samples:
        fid = int(rng.choice(np.arange(len(FAMILY_WEIGHTS)), p=FAMILY_WEIGHTS))
        s = generate_single_sample(fid, x_grid, u_grid, rng, y_min=y_min)
        if s is not None:
            samples.append(s)
            pbar.update(1)
    pbar.close()

    # Create train (0), val (1), test (2) split flags
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = np.zeros(n, dtype=np.int64)
    splits[n_train : n_train + n_val] = 1
    splits[n_train + n_val :] = 2

    arrays = {
        "curves/se_x": np.stack([s.se_x for s in samples]),
        "curves/gkr_u": np.stack([s.gkr_u for s in samples]),
        "curves/ykr_u": np.stack([s.ykr_u for s in samples]),
        "curves/ykr_x": np.stack([s.ykr_x for s in samples]),
        "grids/x_full": x_grid.astype(np.float32),
        "grids/u_grid": u_grid.astype(np.float32),
        "grids/loss_weight_x": make_loss_weight_x(x_grid),
        "grids/loss_weight_u": make_loss_weight_u(u_grid),
        "scales/theta_r": np.array([s.theta_r for s in samples], dtype=np.float32),
        "scales/theta_s": np.array([s.theta_s for s in samples], dtype=np.float32),
        "scales/log10_Ks_cm_day": np.array([s.log10_ks_cm_day for s in samples], dtype=np.float32),
        "metadata/family_id": np.array([s.family_id for s in samples], dtype=np.int64),
        "metadata/source_model": np.array([s.source_model for s in samples], dtype=np.int64),
        "metadata/split": splits,
    }
    return arrays


def save_library_npz(arrays: dict[str, np.ndarray], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
