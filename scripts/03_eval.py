"""Script 03: Evaluate trained SH-FAE model and plot reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sh_fae.data.dataset import SWRCHCFDataset, create_raw_dataloader
from sh_fae.models.fae import SHFAE
from sh_fae.utils.config import config_from_dict
from sh_fae.utils.metrics import check_monotonicity, normalized_rmse, r2_score, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SH-FAE reconstruction quality.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_final.pt")
    parser.add_argument("--data-path", type=str, default="data/swcc_hcf_50k.npz", help="Path to test/val dataset")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate: val or test")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of sample curves to plot")
    parser.add_argument("--out-dir", type=str, default="outputs/eval", help="Directory to save evaluation plots")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


@torch.no_grad()
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

    dataset = SWRCHCFDataset(args.data_path, split=args.split)
    print(f"Evaluating on {len(dataset)} {args.split} samples...")

    x_grid = dataset.x_grid
    u_grid = dataset.u_grid
    x_norm = torch.as_tensor(dataset.x_norm, dtype=torch.float32, device=device)
    u_norm = torch.as_tensor(dataset.u_norm, dtype=torch.float32, device=device)

    swrc_rmses: list[float] = []
    swrc_nrmses: list[float] = []
    swrc_r2s: list[float] = []
    hcf_rmses: list[float] = []
    hcf_nrmses: list[float] = []
    hcf_r2s: list[float] = []
    monotone_swrc = 0
    monotone_hcf = 0

    n_eval = min(len(dataset), 500)
    for i in range(n_eval):
        sample = dataset[i]
        se_true = sample["se_x"].numpy()
        gk_true = sample["gkr_u"].numpy()

        # Context: full curve embedding
        context_coord = torch.cat([x_norm[:, None], u_norm[:, None]], dim=0)[None]
        context_channel = torch.cat(
            [
                torch.zeros(x_norm.numel(), dtype=torch.long, device=device),
                torch.ones(u_norm.numel(), dtype=torch.long, device=device),
            ],
            dim=0,
        )[None]
        context_value = torch.cat(
            [
                sample["se_x"].to(device)[None, :, None],
                sample["gkr_u"].to(device)[None, :, None],
            ],
            dim=1,
        )

        z = model.encoder(context_coord, context_channel, context_value)
        curves = model.decoder.decode_curves(z)

        se_pred = np.interp(x_grid, curves["x"].cpu().numpy(), curves["se"][0].cpu().numpy())
        gk_pred = np.interp(u_grid, curves["u"].cpu().numpy(), curves["gk"][0].cpu().numpy())

        swrc_rmses.append(rmse(se_true, se_pred))
        swrc_nrmses.append(normalized_rmse(se_true, se_pred))
        swrc_r2s.append(r2_score(se_true, se_pred))

        hcf_rmses.append(rmse(gk_true, gk_pred))
        hcf_nrmses.append(normalized_rmse(gk_true, gk_pred))
        hcf_r2s.append(r2_score(gk_true, gk_pred))

        if check_monotonicity(se_pred, decreasing=True):
            monotone_swrc += 1
        if check_monotonicity(gk_pred, decreasing=False):
            monotone_hcf += 1

    print("\n--- Quantitative Evaluation Results ---")
    print(f"Evaluated Samples: {n_eval}")
    print(f"SWRC Median RMSE: {np.median(swrc_rmses):.4e} | n-RMSE: {np.median(swrc_nrmses):.4e} | R^2: {np.median(swrc_r2s):.4f}")
    print(f"HCF  Median RMSE: {np.median(hcf_rmses):.4e} | n-RMSE: {np.median(hcf_nrmses):.4e} | R^2: {np.median(hcf_r2s):.4f}")
    print(f"Physical Monotonicity Satisfaction: SWRC {monotone_swrc/n_eval*100:.1f}%, HCF {monotone_hcf/n_eval*100:.1f}%")

    # Plot sample curves
    n_plot = min(args.num_samples, len(dataset))
    fig, axes = plt.subplots(n_plot, 2, figsize=(10, 2.5 * n_plot), squeeze=False)

    for i in range(n_plot):
        sample = dataset[i]
        context_coord = torch.cat([x_norm[:, None], u_norm[:, None]], dim=0)[None]
        context_channel = torch.cat(
            [
                torch.zeros(x_norm.numel(), dtype=torch.long, device=device),
                torch.ones(u_norm.numel(), dtype=torch.long, device=device),
            ],
            dim=0,
        )[None]
        context_value = torch.cat(
            [
                sample["se_x"].to(device)[None, :, None],
                sample["gkr_u"].to(device)[None, :, None],
            ],
            dim=1,
        )
        z = model.encoder(context_coord, context_channel, context_value)
        curves = model.decoder.decode_curves(z)

        # Plot SWRC
        ax_swrc = axes[i, 0]
        ax_swrc.plot(x_grid, sample["se_x"].numpy(), "k--", label="Ground Truth")
        ax_swrc.plot(curves["x"].cpu().numpy(), curves["se"][0].cpu().numpy(), "r-", label="SH-FAE")
        ax_swrc.set_xlabel(r"$x = \log_{10}(s/\mathrm{cm})$")
        ax_swrc.set_ylabel(r"$S_e$")
        ax_swrc.set_ylim(-0.05, 1.05)
        ax_swrc.grid(True, alpha=0.3)
        if i == 0:
            ax_swrc.set_title("SWRC: Effective Saturation")
            ax_swrc.legend()

        # Plot HCF
        ax_hcf = axes[i, 1]
        ax_hcf.plot(u_grid, sample["ykr_u"].numpy(), "k--", label="Ground Truth")
        ax_hcf.plot(curves["u"].cpu().numpy(), curves["ykr"][0].cpu().numpy(), "b-", label="SH-FAE")
        ax_hcf.set_xlabel(r"$u = S_e$")
        ax_hcf.set_ylabel(r"$y_K = \log_{10}(k_r)$")
        ax_hcf.set_ylim(-12.5, 0.5)
        ax_hcf.grid(True, alpha=0.3)
        if i == 0:
            ax_hcf.set_title("HCF: Log Relative Permeability")
            ax_hcf.legend()

    plt.tight_layout()
    plot_file = out_dir / "reconstruction_samples.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved reconstruction plot to {plot_file.resolve()}!")


if __name__ == "__main__":
    main()
