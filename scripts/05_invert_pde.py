"""Script 05: Differentiable Richards PDE-constrained latent inversion.

Supports:
1. Synthetic wetting-drying infiltration test (Case 2 in paper, Bandai et al. 2024 benchmark)
2. Laboratory upward infiltration AZ7 experiment (Case 3 in paper, Sadeghi et al. 2017)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sh_fae.inversion.dynamic import invert_pde_constrained
from sh_fae.models.fae import SHFAE
from sh_fae.physics.grid import Grid1D
from sh_fae.physics.material import Material
from sh_fae.physics.solver import BoundaryConditions, NewtonOptions, solve_richards
from sh_fae.utils.config import config_from_dict
from sh_fae.utils.metrics import normalized_rmse, rmse


def setup_synthetic_case(device: torch.device):
    """Synthetic wetting-drying column (Paper Case 2).
    
    1.5m soil column, 3 wetting-drying cycles (9 days total).
    Infiltration: 0.25 day at -0.25 m/day; Evaporation: 2.75 days at +0.005 m/day.
    Initial psi: -7.26139 m. Lower boundary Dirichlet psi: -7.26139 m.
    """
    Z = 1.5
    Ns = 150
    dt = 0.01  # days
    T = 9.0    # days
    grid = Grid1D.uniform(Z=Z, Ns=Ns, device=device, dtype=torch.float64)

    def q_ub_fn(t: float) -> float:
        cycle_t = t % 3.0
        return -0.25 if cycle_t < 0.25 else 0.005

    boundary = BoundaryConditions(psi_lb=-7.26139, q_ub=q_ub_fn)
    initial_psi = -7.26139

    # Sensor depths: ~0.1, 0.3, 0.5, 0.7, 0.9 m below surface (z = 0)
    # Cell centers are in [-Z, 0]
    target_depths = [0.10, 0.30, 0.50, 0.70, 0.90]
    z_coords = grid.z_internal.cpu().numpy()
    depths = -z_coords
    sensor_indices = [int(np.argmin(np.abs(depths - d))) for d in target_depths]

    return {
        "grid": grid,
        "boundary": boundary,
        "initial_psi": initial_psi,
        "T": T,
        "dt": dt,
        "sensor_indices": sensor_indices,
        "sensor_depths": depths[sensor_indices],
    }


def setup_laboratory_case(data_dir: Path, device: torch.device):
    """Laboratory upward infiltration AZ7 experiment (Paper Case 3).
    
    0.1m soil column, 300 minutes.
    Constant head Dirichlet bottom psi: -1.0e-4 m.
    Zero flux upper boundary: q_ub = 0.0.
    Initial oven-dry psi: -1.0e4 m.
    """
    obs_file = data_dir / "observations.npz"
    if not obs_file.exists():
        raise FileNotFoundError(f"Laboratory observation file not found: {obs_file}")

    data = np.load(obs_file)
    theta_obs = data["theta_obs"]  # [300, 8]
    obs_z_m = data["obs_z_m"]      # [8]

    Z = 0.1
    Ns = 100
    dt = 1.0  # minutes / step unit
    T = float(theta_obs.shape[0] - 1)
    grid = Grid1D.uniform(Z=Z, Ns=Ns, device=device, dtype=torch.float64)

    boundary = BoundaryConditions(psi_lb=-1.0e-4, q_ub=0.0)
    initial_psi = -1.0e4

    # Match sensors to grid cells
    z_internal = grid.z_internal.cpu().numpy()
    sensor_indices = [int(np.argmin(np.abs(z_internal - z_val))) for z_val in obs_z_m]

    return {
        "grid": grid,
        "boundary": boundary,
        "initial_psi": initial_psi,
        "T": T,
        "dt": dt,
        "theta_obs": theta_obs,
        "sensor_indices": sensor_indices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Richards PDE-constrained latent inversion.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_final.pt")
    parser.add_argument("--case", type=str, choices=["synthetic", "laboratory"], default="synthetic")
    parser.add_argument("--steps", type=int, default=1500, help="Inversion iterations")
    parser.add_argument("--lr-z", type=float, default=1e-3, help="Learning rate for latent vector z")
    parser.add_argument("--lambda-z", type=float, default=1e-3, help="Latent prior regularization")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Synthetic observation noise standard deviation")
    parser.add_argument("--stats-path", type=str, default=None, help="Latent statistics path for z0 initialization")
    parser.add_argument("--data-dir", type=str, default="data/laboratory", help="Directory containing lab observations")
    parser.add_argument("--out-dir", type=str, default="outputs/invert_pde", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = config_from_dict(ckpt["config"])

    model = SHFAE(config.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    z_init = None
    z_ref = None
    if args.stats_path and Path(args.stats_path).exists():
        stats = np.load(args.stats_path)
        z_mean = torch.as_tensor(stats["z_mean"], dtype=torch.float64, device=device).unsqueeze(0)
        z_init = z_mean
        z_ref = z_mean
        print("Using empirical training latent mean for z_init and z_ref.")

    solver_opts = NewtonOptions(
        tol=1e-8,
        max_iter=15,
        create_graph=True,
        jacobian_mode="tridiagonal",
        max_line_search_iter=8,
    )

    if args.case == "synthetic":
        print("Configuring Synthetic Infiltration-Evaporation Case (Paper Case 2)...")
        case_info = setup_synthetic_case(device)
        grid = case_info["grid"]
        boundary = case_info["boundary"]
        initial_psi = case_info["initial_psi"]
        T = case_info["T"]
        dt = case_info["dt"]
        sensor_indices = case_info["sensor_indices"]

        # Generate synthetic ground truth using VGM
        from sh_fae.physics.constitutive import VGMConstitutive
        true_vgm = VGMConstitutive(alpha=1.43, n=1.506, theta_r=0.0, theta_s=0.33, tau=0.5)
        true_mat = Material(intrinsic_permeability_m2=2.95e-13)

        print("Simulating ground truth synthetic observations...")
        true_sim = solve_richards(
            grid=grid,
            constitutive=true_vgm,
            material=true_mat,
            boundary=boundary,
            initial_psi=initial_psi,
            T=T,
            dt=dt,
            options=solver_opts,
        )

        n_times = true_sim.theta.shape[0]
        time_idx = np.arange(0, n_times, 1)
        clean_obs = true_sim.theta[time_idx, :][:, sensor_indices].detach().cpu().numpy()

        rng = np.random.default_rng(42)
        obs_theta = clean_obs + rng.normal(0.0, args.noise_std, size=clean_obs.shape)
        obs_sigma = args.noise_std

        scale_configs = {
            "theta_r": {"init": 0.0, "mu": 0.0, "sigma": 0.0, "lr": 1e-2},
            "theta_s": {"init": 0.30, "mu": 0.30, "sigma": 0.10, "lr": 1e-2},
            "log10_k": {"init": -13.0, "mu": -13.0, "sigma": 2.0, "lr": 1e-2},
        }

    else:
        print("Configuring Laboratory Upward Infiltration Case (Paper Case 3)...")
        case_info = setup_laboratory_case(Path(args.data_dir), device)
        grid = case_info["grid"]
        boundary = case_info["boundary"]
        initial_psi = case_info["initial_psi"]
        T = case_info["T"]
        dt = case_info["dt"]
        sensor_indices = case_info["sensor_indices"]
        obs_theta = case_info["theta_obs"]
        time_idx = np.arange(0, obs_theta.shape[0], 1)
        obs_sigma = 0.01

        scale_configs = {
            "theta_r": {"init": 0.05, "mu": 0.05, "sigma": 0.05, "lr": 1e-2},
            "theta_s": {"init": 0.50, "mu": 0.50, "sigma": 0.20, "lr": 1e-2},
            "log10_k": {"init": -13.0, "mu": -13.0, "sigma": 3.0, "lr": 5e-2},
        }

    print(f"Starting PDE-constrained inversion for {args.steps} steps...")
    res = invert_pde_constrained(
        decoder=model.decoder,
        grid=grid,
        boundary=boundary,
        initial_psi=initial_psi,
        T=T,
        dt=dt,
        obs_times_idx=time_idx,
        obs_depths_idx=sensor_indices,
        obs_theta=obs_theta,
        obs_sigma=obs_sigma,
        z_init=z_init,
        z_ref=z_ref,
        scale_configs=scale_configs,
        n_steps=args.steps,
        lr_z=args.lr_z,
        lambda_z=args.lambda_z,
        solver_options=solver_opts,
        device=device,
        progress=True,
    )

    print("\n--- Inversion Results ---")
    print(f"Final Data Loss: {res.loss:.4e}")
    print(f"Inferred theta_r: {res.theta_r:.4f}")
    print(f"Inferred theta_s: {res.theta_s:.4f}")
    print(f"Inferred intrinsic permeability K: {res.intrinsic_k:.4e} m^2")

    pred_obs = res.simulated_theta[time_idx, :][:, sensor_indices].numpy()
    score_rmse = rmse(obs_theta, pred_obs)
    score_nrmse = normalized_rmse(obs_theta, pred_obs)
    print(f"Water Content Misfit: RMSE = {score_rmse:.4e} | n-RMSE = {score_nrmse:.4e}")

    # Plot water content comparison
    plt.figure(figsize=(9, 6))
    times_arr = np.linspace(0.0, T, len(time_idx))
    for s_idx in range(len(sensor_indices)):
        plt.plot(times_arr, obs_theta[:, s_idx], ".", alpha=0.3, label=f"Obs Sensor {s_idx+1}" if s_idx < 2 else None)
        plt.plot(times_arr, pred_obs[:, s_idx], "-", label=f"SH-FAE Sensor {s_idx+1}" if s_idx < 2 else None)
    plt.xlabel("Time [days]" if args.case == "synthetic" else "Time [min]")
    plt.ylabel(r"Volumetric Water Content $\theta$ [$\mathrm{m}^3 \mathrm{m}^{-3}$]")
    plt.title(f"Water Content Reproduction ({args.case.capitalize()} Case)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_obs_path = out_dir / "simulated_vs_observed_theta.png"
    plt.savefig(plot_obs_path, dpi=300)
    plt.close()

    # Plot recovered constitutive curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x_c = res.decoded_curves["x"].cpu().numpy()
    se_c = res.decoded_curves["se"][0].cpu().numpy()
    u_c = res.decoded_curves["u"].cpu().numpy()
    ykr_c = res.decoded_curves["ykr"][0].cpu().numpy()

    # SWRC (Dimensional: psi in m)
    # x = log10(-100 * psi) => psi = - 10^x / 100
    psi_m = -(10.0**x_c) / 100.0
    theta_c = res.theta_r + (res.theta_s - res.theta_r) * se_c

    axes[0].plot(-psi_m, theta_c, "r-", linewidth=2.0, label="Recovered SH-FAE")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Matric Suction $-\psi$ [m]")
    axes[0].set_ylabel(r"Water Content $\theta$ [$\mathrm{m}^3 \mathrm{m}^{-3}$]")
    axes[0].set_title("Recovered SWRC")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # HCF (Unsaturated permeability K * kr)
    k_unsat = res.intrinsic_k * (10.0**ykr_c)
    axes[1].plot(u_c, k_unsat, "b-", linewidth=2.0, label="Recovered SH-FAE")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Effective Saturation $u = S_e$ [-]")
    axes[1].set_ylabel(r"Unsaturated Permeability $K \cdot k_r$ [$\mathrm{m}^2$]")
    axes[1].set_title("Recovered HCF")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_curve_path = out_dir / "recovered_constitutive_curves.png"
    plt.savefig(plot_curve_path, dpi=300)
    plt.close()

    print(f"Plots saved to {out_dir.resolve()}!")


if __name__ == "__main__":
    main()
