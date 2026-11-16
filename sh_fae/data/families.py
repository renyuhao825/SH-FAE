"""Constitutive family implementations for synthetic soil hydraulic library generation.

Covers:
0: van Genuchten - Mualem (VGM)
1: Brooks - Corey (BC)
2: Kosugi (log-normal pore distribution)
3: Fredlund - Xing (FX)
4: PDI-like VG (with dry-end adsorbed film flow)
5: Durner bimodal (dual porosity)
6: Nonparametric monotonic density
7: Boundary stress (extreme test cases)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


X_MIN = -1.0
X_FLOW_MAX = 5.0
X_MAX = 8.0
S0_CM = 1.0
Y_MIN_DEFAULT = -12.0
DRY_END_SE_MAX = 0.02

FAMILY_NAMES = {
    0: "vgm",
    1: "brooks_corey",
    2: "kosugi",
    3: "fredlund_xing_like",
    4: "pdi_like_vg",
    5: "durner_bimodal",
    6: "nonparametric_density",
    7: "boundary_stress",
}

FAMILY_WEIGHTS = np.array([0.15, 0.10, 0.08, 0.07, 0.20, 0.25, 0.10, 0.05])
FAMILY_WEIGHTS = FAMILY_WEIGHTS / FAMILY_WEIGHTS.sum()


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(math.log10(low), math.log10(high)))


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float64)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def make_x_grid(nx: int) -> np.ndarray:
    return np.linspace(X_MIN, X_MAX, nx, dtype=np.float64)


def make_u_grid(nu: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, nu, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def suction_cm_from_x(x_grid: np.ndarray) -> np.ndarray:
    return S0_CM * (10.0**x_grid)


# ---------------- 0: VGM ----------------
def eval_vgm(x_grid: np.ndarray, alpha: float, n: float) -> np.ndarray:
    s_cm = suction_cm_from_x(x_grid)
    m = 1.0 - 1.0 / n
    return (1.0 + (alpha * s_cm) ** n) ** (-m)


def sample_vgm_params(rng: np.random.Generator) -> dict[str, Any]:
    alpha = log_uniform(rng, 1e-3, 3e-1)
    n = 1.0 + log_uniform(rng, 0.05, 3.0)
    return {"alpha_cm_inv": alpha, "n": n, "m": 1.0 - 1.0 / n, "tau": 0.5}


# ---------------- 1: Brooks-Corey ----------------
def eval_brooks_corey(x_grid: np.ndarray, hb_cm: float, lam: float) -> np.ndarray:
    s_cm = suction_cm_from_x(x_grid)
    se = np.ones_like(x_grid, dtype=np.float64)
    mask = s_cm > hb_cm
    se[mask] = (s_cm[mask] / hb_cm) ** (-lam)
    return se


def sample_brooks_corey_params(rng: np.random.Generator) -> dict[str, Any]:
    hb_cm = log_uniform(rng, 5.0, 500.0)
    b = float(rng.uniform(2.0, 15.0))
    return {"hb_cm": hb_cm, "b": b, "lambda": 1.0 / b, "tau": 0.5}


# ---------------- 2: Kosugi ----------------
def erfc_array(values: np.ndarray) -> np.ndarray:
    return np.fromiter((math.erfc(float(v)) for v in values.ravel()), dtype=np.float64).reshape(values.shape)


def eval_kosugi(x_grid: np.ndarray, hm_cm: float, sigma: float) -> np.ndarray:
    s_cm = suction_cm_from_x(x_grid)
    arg = np.log(s_cm / hm_cm) / (math.sqrt(2.0) * sigma)
    return 0.5 * erfc_array(arg)


def sample_kosugi_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "hm_cm": log_uniform(rng, 1e0, 1e5),
        "sigma": float(rng.uniform(0.25, 2.5)),
        "tau": 0.5,
    }


# ---------------- 3: Fredlund-Xing ----------------
def eval_fredlund_xing_like(x_grid: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    s_cm = suction_cm_from_x(x_grid)
    s_kpa = s_cm / 10.1972
    a = params["a_kpa"]
    n = params["n"]
    m = params["m"]
    hr = params["hr_kpa"]

    correction = 1.0 - np.log1p(s_kpa / hr) / np.log1p(1.0e6 / hr)
    correction = np.clip(correction, 0.0, 1.0)
    raw = correction / (np.log(math.e + (s_kpa / a) ** n) ** m)
    raw_min = float(raw[-1])
    raw_max = float(raw[0])
    if raw_max <= raw_min:
        return np.full_like(x_grid, np.nan, dtype=np.float64)
    se = (raw - raw_min) / (raw_max - raw_min)
    return np.clip(se, 0.0, 1.0)


def sample_fredlund_xing_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "a_kpa": log_uniform(rng, 1e-1, 1e5),
        "n": log_uniform(rng, 0.5, 8.0),
        "m": log_uniform(rng, 0.2, 5.0),
        "hr_kpa": log_uniform(rng, 1e3, 1e6),
        "tau": 0.5,
    }


# ---------------- 4: PDI-like VG ----------------
def sample_pdi_like_params(rng: np.random.Generator) -> dict[str, Any]:
    params = sample_vgm_params(rng)
    params.update(
        {
            "h0_cm": 10.0**6.8,
            "omega": log_uniform(rng, 1e-5, 1e-2),
            "film_A": float(rng.uniform(1.0, 5.0)),
            "film_p": float(rng.uniform(0.25, 2.0)),
            "pdi_mode": "vg_plus_synthetic_film",
        }
    )
    return params


def add_pdi_like_film(
    y_mualem: np.ndarray,
    u_grid: np.ndarray,
    params: dict[str, Any],
    y_min: float,
) -> np.ndarray:
    y_film = y_min + params["film_A"] * (np.clip(u_grid, 0.0, 1.0) ** params["film_p"])
    kr = 10.0**y_mualem + params["omega"] * (10.0**y_film)
    kr = kr / kr[-1]
    kr = np.clip(kr, 10.0**y_min, 1.0)
    y = np.log10(kr)
    y[0] = y_min
    y[-1] = 0.0
    return y


# ---------------- 5: Durner Bimodal ----------------
def sample_durner_params(rng: np.random.Generator) -> dict[str, Any]:
    for _ in range(100):
        alpha_coarse = log_uniform(rng, 1e-2, 3e-1)
        alpha_fine = log_uniform(rng, 1e-4, 3e-2)
        if math.log10(alpha_coarse) - math.log10(alpha_fine) >= 0.7:
            break
    n_coarse = 1.0 + log_uniform(rng, 0.05, 4.0)
    n_fine = 1.0 + log_uniform(rng, 0.05, 4.0)
    w_coarse = float(rng.uniform(0.05, 0.60))
    return {
        "w_coarse": w_coarse,
        "w_fine": 1.0 - w_coarse,
        "alpha_coarse_cm_inv": alpha_coarse,
        "alpha_fine_cm_inv": alpha_fine,
        "n_coarse": n_coarse,
        "n_fine": n_fine,
        "m_coarse": 1.0 - 1.0 / n_coarse,
        "m_fine": 1.0 - 1.0 / n_fine,
        "tau": 0.5,
    }


def eval_durner(x_grid: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    se_coarse = eval_vgm(x_grid, params["alpha_coarse_cm_inv"], params["n_coarse"])
    se_fine = eval_vgm(x_grid, params["alpha_fine_cm_inv"], params["n_fine"])
    return params["w_coarse"] * se_coarse + params["w_fine"] * se_fine


# ---------------- 6: Nonparametric Density ----------------
def smooth_signal(signal: np.ndarray, sigma_points: float) -> np.ndarray:
    if sigma_points <= 0.5:
        return signal
    radius = max(2, int(math.ceil(3.0 * sigma_points)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(signal, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def sample_nonparametric_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "num_density_knots": int(rng.integers(6, 15)),
        "density_amplitude_std": float(rng.uniform(0.3, 1.2)),
        "length_scale_x": float(rng.uniform(0.4, 1.5)),
        "se_tail": 0.0,
        "tau": 0.5,
    }


def eval_nonparametric(x_grid: np.ndarray, params: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    knots = np.linspace(X_MIN, X_MAX, params["num_density_knots"], dtype=np.float64)
    amps = rng.normal(0.0, params["density_amplitude_std"], size=knots.size)
    interp = np.interp(x_grid, knots, amps)
    sigma_points = params["length_scale_x"] / float(np.mean(np.diff(x_grid)))
    interp = smooth_signal(interp, sigma_points)
    q = np.exp(interp - np.max(interp))
    integral = cumtrapz(q, x_grid)
    total = float(integral[-1])
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(x_grid, np.nan, dtype=np.float64)
    cdf = integral / total
    se_tail = params["se_tail"]
    return se_tail + (1.0 - se_tail) * (1.0 - cdf)


# ---------------- 7: Boundary Stress ----------------
def sample_boundary_params(rng: np.random.Generator) -> dict[str, Any]:
    mode = str(rng.choice(["sharp_vgm", "flat_vgm", "extreme_bc", "wide_durner"]))
    if mode == "sharp_vgm":
        alpha = log_uniform(rng, 3e-2, 5e-1)
        n = 1.0 + log_uniform(rng, 1.5, 6.0)
        return {"boundary_mode": mode, "alpha_cm_inv": alpha, "n": n, "m": 1.0 - 1.0 / n, "tau": 0.5}
    if mode == "flat_vgm":
        alpha = log_uniform(rng, 1e-5, 3e-3)
        n = 1.0 + log_uniform(rng, 0.03, 0.4)
        return {"boundary_mode": mode, "alpha_cm_inv": alpha, "n": n, "m": 1.0 - 1.0 / n, "tau": 0.5}
    if mode == "extreme_bc":
        hb = log_uniform(rng, 1.0, 1e3)
        b = float(rng.uniform(1.5, 20.0))
        return {"boundary_mode": mode, "hb_cm": hb, "b": b, "lambda": 1.0 / b, "tau": 0.5}
    coarse = sample_durner_params(rng)
    coarse["boundary_mode"] = mode
    coarse["alpha_coarse_cm_inv"] = log_uniform(rng, 5e-2, 8e-1)
    coarse["alpha_fine_cm_inv"] = log_uniform(rng, 1e-5, 5e-4)
    return coarse


def eval_boundary(x_grid: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    mode = params["boundary_mode"]
    if mode in {"sharp_vgm", "flat_vgm"}:
        return eval_vgm(x_grid, params["alpha_cm_inv"], params["n"])
    if mode == "extreme_bc":
        return eval_brooks_corey(x_grid, params["hb_cm"], params["lambda"])
    return eval_durner(x_grid, params)


# ---------------- HCF & Mualem Integrals ----------------
def hcf_vgm_closed(u_grid: np.ndarray, m: float, tau: float, y_min: float) -> np.ndarray:
    u = np.clip(u_grid, 0.0, 1.0)
    term = np.zeros_like(u)
    positive = u > 0.0
    term[positive] = 1.0 - (1.0 - u[positive] ** (1.0 / m)) ** m
    kr = (u**tau) * (term**2)
    kr = np.clip(kr, 10.0**y_min, 1.0)
    y = np.log10(kr)
    y[-1] = 0.0
    y[0] = y_min
    return y


def hcf_bc_closed(u_grid: np.ndarray, lam: float, tau: float, y_min: float) -> np.ndarray:
    exponent = tau + 2.0 + 2.0 / lam
    kr = np.clip(u_grid, 0.0, 1.0) ** exponent
    kr = np.clip(kr, 10.0**y_min, 1.0)
    y = np.log10(kr)
    y[-1] = 0.0
    y[0] = y_min
    return y


def unique_monotone_points(xp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xp, kind="mergesort")
    xp_sorted = xp[order]
    fp_sorted = fp[order]
    unique_x = []
    unique_f = []
    start = 0
    n = xp_sorted.size
    while start < n:
        stop = start + 1
        while stop < n and abs(float(xp_sorted[stop] - xp_sorted[start])) <= 1e-10:
            stop += 1
        unique_x.append(float(np.mean(xp_sorted[start:stop])))
        unique_f.append(float(np.mean(fp_sorted[start:stop])))
        start = stop
    return np.asarray(unique_x), np.asarray(unique_f)


def hcf_numerical_mualem(
    se_x: np.ndarray,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    tau: float,
    y_min: float,
    nu_int: int = 512,
) -> np.ndarray:
    u_int = make_u_grid(nu_int)
    u_eval = np.clip(u_int, 1e-6, 1.0)

    xp = np.clip(se_x[::-1], 0.0, 1.0)
    fp = x_grid[::-1]
    xp_unique, fp_unique = unique_monotone_points(xp, fp)
    if xp_unique.size < 2:
        return np.full_like(u_grid, np.nan, dtype=np.float64)
    x_of_u = np.interp(u_eval, xp_unique, fp_unique, left=X_MAX, right=X_MIN)
    x_of_u = np.clip(x_of_u, X_MIN, X_MAX)
    s_of_u = S0_CM * (10.0**x_of_u)
    inv_s = 1.0 / np.clip(s_of_u, 10.0**X_MIN, 10.0**X_MAX)
    integral = cumtrapz(inv_s, u_int)
    if integral[-1] <= 0.0 or not np.isfinite(integral[-1]):
        return np.full_like(u_grid, np.nan, dtype=np.float64)
    ratio = np.interp(u_grid, u_int, integral / integral[-1])
    kr = (np.clip(u_grid, 0.0, 1.0) ** tau) * (np.clip(ratio, 0.0, 1.0) ** 2)
    kr = np.clip(kr, 10.0**y_min, 1.0)
    y = np.log10(kr)
    y[0] = y_min
    y[-1] = 0.0
    return y
