"""Script 02: Train SH-FAE on the synthetic soil hydraulic library."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from sh_fae.data.dataset import (
    CHANNEL_HCF,
    CHANNEL_SWRC,
    BatchParser,
    SWRCHCFDataset,
    create_raw_dataloader,
    cycle,
)
from sh_fae.models.fae import SHFAE
from sh_fae.utils.config import (
    AEConfig,
    config_to_dict,
    load_config_yaml,
    save_config_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SH-FAE model.")
    parser.add_argument("--config", type=str, default="configs/train_ae.yaml", help="Path to config YAML")
    parser.add_argument("--data-path", type=str, default=None, help="Override path to dataset NPZ")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, auto)")
    return parser.parse_args()


def compute_smoothness_loss(decoded: dict[str, torch.Tensor]) -> torch.Tensor:
    curves = decoded.get("curves")
    if curves is None:
        return torch.tensor(0.0, device=decoded["pred"].device)
    se = curves["se"]
    gk = curves["gk"]
    se_diff2 = se[:, 2:] - 2.0 * se[:, 1:-1] + se[:, :-2]
    gk_diff2 = gk[:, 2:] - 2.0 * gk[:, 1:-1] + gk[:, :-2]
    return se_diff2.pow(2).mean() + gk_diff2.pow(2).mean()


def train_step(
    model: SHFAE,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    smooth_weight: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    decoded = model(batch, return_curves=smooth_weight > 0.0)
    pred = decoded["pred"]
    target = batch["target_value"]
    weight = batch["query_weight"]
    channel = batch["query_channel"]

    # Channel-wise reconstruction losses
    swrc_mask = (channel == CHANNEL_SWRC)
    hcf_mask = (channel == CHANNEL_HCF)

    loss_swrc = (((pred - target) ** 2) * weight)[:, swrc_mask[0]].sum() / weight[:, swrc_mask[0]].sum().clamp_min(1e-8)
    loss_hcf = (((pred - target) ** 2) * weight)[:, hcf_mask[0]].sum() / weight[:, hcf_mask[0]].sum().clamp_min(1e-8)
    loss_recon = (((pred - target) ** 2) * weight).sum() / weight.sum().clamp_min(1e-8)

    loss_smooth = compute_smoothness_loss(decoded) if smooth_weight > 0.0 else torch.tensor(0.0)
    loss_total = loss_recon + smooth_weight * loss_smooth

    loss_total.backward()
    if grad_clip_norm > 0:
        clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    return {
        "loss_total": float(loss_total.item()),
        "loss_recon": float(loss_recon.item()),
        "loss_swrc": float(loss_swrc.item()),
        "loss_hcf": float(loss_hcf.item()),
        "loss_smooth": float(loss_smooth.item()),
    }


def compute_latent_statistics(
    model: SHFAE,
    dataset: SWRCHCFDataset,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Compute empirical latent mean and variance across training set."""
    model.eval()
    loader = create_raw_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    x_norm = torch.as_tensor(dataset.x_norm, dtype=torch.float32, device=device)
    u_norm = torch.as_tensor(dataset.u_norm, dtype=torch.float32, device=device)

    all_z: list[torch.Tensor] = []
    with torch.no_grad():
        for raw in loader:
            se = raw["se_x"].to(device)
            gk = raw["gkr_u"].to(device)
            bs = se.shape[0]

            context_coord = torch.cat([x_norm[:, None], u_norm[:, None]], dim=0)[None].expand(bs, -1, -1)
            context_channel = torch.cat(
                [
                    torch.zeros(x_norm.numel(), dtype=torch.long, device=device),
                    torch.ones(u_norm.numel(), dtype=torch.long, device=device),
                ],
                dim=0,
            )[None].expand(bs, -1)
            context_value = torch.cat([se[..., None], gk[..., None]], dim=1)

            z = model.encoder(context_coord, context_channel, context_value)
            all_z.append(z.cpu())

    z_matrix = torch.cat(all_z, dim=0).numpy()
    z_mean = np.mean(z_matrix, axis=0)
    z_std = np.std(z_matrix, axis=0)
    return {"z_mean": z_mean, "z_std": z_std, "z_matrix": z_matrix}


def main() -> None:
    args = parse_args()
    config = load_config_yaml(args.config) if Path(args.config).exists() else AEConfig()

    # CLI Overrides
    if args.data_path:
        config.training.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr_peak = args.lr
    if args.device:
        config.training.device = args.device

    out_dir = Path(config.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_yaml(config, out_dir / "config.yaml")

    device_str = config.training.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on device: {device}")

    # Datasets
    train_dataset = SWRCHCFDataset(config.training.data_path, split="train")
    val_dataset = SWRCHCFDataset(config.training.data_path, split="val")
    print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    train_loader = cycle(create_raw_dataloader(train_dataset, batch_size=config.training.batch_size, shuffle=True))
    train_parser = BatchParser(train_dataset, config.batch)

    model = SHFAE(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr_peak, weight_decay=config.training.weight_decay)

    print(f"Starting training for {config.training.max_steps} steps...")
    t0 = time.time()
    best_loss = float("inf")

    for step in range(1, config.training.max_steps + 1):
        raw_batch = next(train_loader)
        batch = train_parser(raw_batch)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        metrics = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            smooth_weight=config.loss.smooth_weight,
            grad_clip_norm=config.training.grad_clip_norm,
        )

        if step % config.training.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"[Step {step:6d}/{config.training.max_steps}] "
                f"Loss: {metrics['loss_total']:.4e} | "
                f"SWRC: {metrics['loss_swrc']:.4e} | "
                f"HCF: {metrics['loss_hcf']:.4e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % config.training.save_every == 0 or step == config.training.max_steps:
            ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config_to_dict(config),
                },
                ckpt_path,
            )

    # Save final checkpoint
    final_path = out_dir / "checkpoint_final.pt"
    torch.save(
        {
            "step": config.training.max_steps,
            "model_state": model.state_dict(),
            "config": config_to_dict(config),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}!")

    # Compute empirical latent statistics for inversion initialization
    print("Computing empirical latent statistics across training library...")
    stats = compute_latent_statistics(model, train_dataset, device=device)
    np.savez_compressed(out_dir / "trainset_latent_stats.npz", **stats)
    print(f"Saved latent statistics to {out_dir / 'trainset_latent_stats.npz'}!")


if __name__ == "__main__":
    main()
