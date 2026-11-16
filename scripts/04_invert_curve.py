"""Script 04: Direct curve expressivity inversion for unseen constitutive models.

Evaluates whether the pretrained SH-FAE decoder can express constitutive shapes
outside its pretraining families (e.g., Lu 2016 SWRC, Gou 2023 HCF).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sh_fae.inversion.direct import invert_direct_curve
from sh_fae.models.fae import SHFAE
from sh_fae.utils.config import config_from_dict
from sh_fae.utils.metrics import r2_score, rmse


def evaluate_lu_swrc_synthetic(s_kpa: np.ndarray, alpha_inv_kpa: float = 0.05, n: float = 2.2) -> np.ndarray:
    """Analytical representative Lu (2016) generalized adsorption-capillarity SWRC."""
    # Adsorption component
    theta_a_max = 0.06
    s_c = 1500.0  # cavitation suction kPa
    s_0 = 1.0e6   # oven dry suction kPa
    f_ads = 1.0 - np.log(1.0 + s_kpa / 1.0) / np.log(1.0 + s_0 / 1.0)
    theta_ads = theta_a_max * np.clip(f_ads, 0.0, 1.0)

    # Capillary component
    m = 1.0 - 1.0 / n
    se_cap = (1.0 + (alpha_inv_kpa * s_kpa) ** n) ** (-m)
    f_cap = 1.0 - np.clip(s_kpa / s_c, 0.0, 1.0) ** 0.5
    theta_cap = (0.42 - theta_a_max) * se_cap * f_cap

    theta_total = theta_ads + theta_cap
    se_total = theta_total / 0.42
    return np.clip(se_total, 0.0, 1.0)


def evaluate_gou_hcf_synthetic(s_kpa: np.ndarray, alpha_inv_kpa: float = 0.1, n: float = 1.8) -> np.ndarray:
    """Analytical representative Gou et al. (2023) film-capillary hydraulic conductivity function."""
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha_inv_kpa * s_kpa) ** n) ** (-m)
    # Capillary component
    kr_cap = (se ** 0.5) * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2
    # Film flow component
    kr_film = 1e-4 * (s_kpa / 1.0) ** (-1.5)
    kr_total = kr_cap + kr_film
    ykr = np.log10(np.clip(kr_total, 1e-12, 1.0))
    return ykr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct curve expressivity inversion.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_final.pt")
    parser.add_argument("--mode", type=str, choices=["lu_swrc", "gou_hcf"], default="lu_swrc")
    parser.add_argument("--steps", type=int, default=3000, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--lambda-z", type=float, default=1e-3, help="Latent prior regularization weight")
    parser.add_argument("--stats-path", type=str, default=None, help="Path to trainset_latent_stats.npz for z0 and z_ref")
    parser.add_argument("--out-dir", type=str, default="outputs/invert_direct", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
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
        z_mean = torch.as_tensor(stats["z_mean"], dtype=torch.float32, device=device).unsqueeze(0)
        z_init = z_mean
        z_ref = z_mean
        print("Using empirical training latent mean for z_init and z_ref.")

    if args.mode == "lu_swrc":
        print("Running expressive test on unseen Lu (2016) SWRC model...")
        # Suction grid in kPa (10^-2 to 10^6 kPa)
        suction_kpa = np.logspace(-2.0, 6.0, 512)
        suction_cm = suction_kpa * 10.1972
        x_target = np.log10(suction_cm)
        y_target = evaluate_lu_swrc_synthetic(suction_kpa)

        res = invert_direct_curve(
            decoder=model.decoder,
            coords=x_target,
            values=y_target,
            channel_id=0,
            z_init=z_init,
            z_ref=z_ref,
            n_steps=args.steps,
            lr=args.lr,
            lambda_z=args.lambda_z,
            device=device,
        )

        pred_se = np.interp(x_target, res.decoded_curves["x"].cpu().numpy(), res.decoded_curves["se"][0].cpu().numpy())
        score_r2 = r2_score(y_target, pred_se)
        score_rmse = rmse(y_target, pred_se)

        print(f"Inversion complete! R^2: {score_r2:.4f} | RMSE: {score_rmse:.4e}")

        # Plot
        plt.figure(figsize=(6, 4.5))
        plt.plot(suction_kpa, y_target, "k-", linewidth=2.0, label="Lu (2016) Target")
        plt.plot(suction_kpa, pred_se, "r--", linewidth=2.0, label="SH-FAE Recovered")
        plt.xscale("log")
        plt.xlabel("Matric Suction [kPa]")
        plt.ylabel("Effective Saturation $S_e$ [-]")
        plt.title(f"Unseen SWRC Expressivity Test ($R^2 = {score_r2:.4f}$)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / "lu_swrc_inversion.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path.resolve()}!")

    elif args.mode == "gou_hcf":
        print("Running expressive test on unseen Gou et al. (2023) HCF model...")
        suction_kpa = np.logspace(-1.0, 5.0, 512)
        y_target = evaluate_gou_hcf_synthetic(suction_kpa)
        u_target = np.linspace(0.01, 1.0, 512)
        y_u_target = np.interp(u_target, np.linspace(0.01, 1.0, len(y_target)), np.sort(y_target))
        g_u_target = (y_u_target - (-12.0)) / (0.0 - (-12.0))

        res = invert_direct_curve(
            decoder=model.decoder,
            coords=u_target,
            values=g_u_target,
            channel_id=1,
            z_init=z_init,
            z_ref=z_ref,
            n_steps=args.steps,
            lr=args.lr,
            lambda_z=args.lambda_z,
            device=device,
        )

        pred_gk = np.interp(u_target, res.decoded_curves["u"].cpu().numpy(), res.decoded_curves["gk"][0].cpu().numpy())
        score_r2 = r2_score(g_u_target, pred_gk)
        score_rmse = rmse(g_u_target, pred_gk)

        print(f"Inversion complete! R^2: {score_r2:.4f} | RMSE: {score_rmse:.4e}")

        # Plot
        plt.figure(figsize=(6, 4.5))
        pred_ykr = -12.0 + 12.0 * pred_gk
        plt.plot(u_target, y_u_target, "k-", linewidth=2.0, label="Gou et al. (2023) Target")
        plt.plot(u_target, pred_ykr, "b--", linewidth=2.0, label="SH-FAE Recovered")
        plt.xlabel("Effective Saturation $u = S_e$ [-]")
        plt.ylabel(r"$\log_{10}(k_r)$ [-]")
        plt.title(f"Unseen HCF Expressivity Test ($R^2 = {score_r2:.4f}$)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / "gou_hcf_inversion.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path.resolve()}!")


if __name__ == "__main__":
    main()
