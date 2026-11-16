"""Dataset and random context/query point sampler for SH-FAE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.config import BatchConfig


SPLIT_TO_ID = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
CHANNEL_SWRC = 0
CHANNEL_HCF = 1


def normalize_x(x: np.ndarray | torch.Tensor, x_min: float, x_max: float):
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


def normalize_u(u: np.ndarray | torch.Tensor):
    return 2.0 * u - 1.0


def _split_id(split: str | int | None) -> int | None:
    if split is None:
        return None
    if isinstance(split, int):
        return split
    key = split.lower()
    if key not in SPLIT_TO_ID:
        raise ValueError(f"Unknown split {split!r}; expected train/val/test.")
    return SPLIT_TO_ID[key]


class SWRCHCFDataset(Dataset):
    """NPZ-backed SWRC-HCF curve dataset."""

    def __init__(
        self,
        npz_path: str | Path,
        split: str | int | None = "train",
    ) -> None:
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}

        split_value = _split_id(split)
        all_indices = np.arange(arrays["curves/se_x"].shape[0], dtype=np.int64)
        if split_value is None:
            indices = all_indices
        else:
            indices = all_indices[arrays["metadata/split"] == split_value]
        if indices.size == 0:
            raise ValueError(f"No samples found for split={split!r}.")

        self.indices = indices
        self.x_grid = arrays["grids/x_full"].astype(np.float32)
        self.u_grid = arrays["grids/u_grid"].astype(np.float32)
        self.x_min = float(self.x_grid[0])
        self.x_max = float(self.x_grid[-1])
        self.x_norm = normalize_x(self.x_grid, self.x_min, self.x_max).astype(np.float32)
        self.u_norm = normalize_u(self.u_grid).astype(np.float32)
        self.loss_weight_x = arrays["grids/loss_weight_x"].astype(np.float32)
        self.loss_weight_u = arrays["grids/loss_weight_u"].astype(np.float32)

        self.se_x = arrays["curves/se_x"][indices].astype(np.float32)
        self.gkr_u = arrays["curves/gkr_u"][indices].astype(np.float32)
        self.ykr_u = arrays["curves/ykr_u"][indices].astype(np.float32)
        self.ykr_x = arrays["curves/ykr_x"][indices].astype(np.float32)
        self.theta_r = arrays["scales/theta_r"][indices].astype(np.float32)
        self.theta_s = arrays["scales/theta_s"][indices].astype(np.float32)
        self.log10_ks_cm_day = arrays["scales/log10_Ks_cm_day"][indices].astype(np.float32)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(item)
        return {
            "sample_id": torch.tensor(int(self.indices[idx]), dtype=torch.long),
            "se_x": torch.from_numpy(self.se_x[idx]),
            "gkr_u": torch.from_numpy(self.gkr_u[idx]),
            "ykr_u": torch.from_numpy(self.ykr_u[idx]),
            "ykr_x": torch.from_numpy(self.ykr_x[idx]),
            "theta_r": torch.tensor(self.theta_r[idx], dtype=torch.float32),
            "theta_s": torch.tensor(self.theta_s[idx], dtype=torch.float32),
            "log10_ks_cm_day": torch.tensor(self.log10_ks_cm_day[idx], dtype=torch.float32),
        }


class BatchParser:
    """Samples dynamic context and query point sets from curve batches."""

    def __init__(self, dataset: SWRCHCFDataset, config: BatchConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.nx = int(dataset.x_grid.size)
        self.nu = int(dataset.u_grid.size)

        self.x_norm = torch.from_numpy(dataset.x_norm)
        self.u_norm = torch.from_numpy(dataset.u_norm)
        self.weight_x = torch.from_numpy(dataset.loss_weight_x)
        self.weight_u = torch.from_numpy(dataset.loss_weight_u)

    def __call__(
        self,
        raw_batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = raw_batch["se_x"].shape[0]

        # Context indices
        c_idx_swrc = torch.randint(0, self.nx, (batch_size, self.config.nc_swrc), generator=generator)
        c_idx_hcf = torch.randint(0, self.nu, (batch_size, self.config.nc_hcf), generator=generator)

        # Query indices
        q_idx_swrc = torch.randint(0, self.nx, (batch_size, self.config.nq_swrc), generator=generator)
        q_idx_hcf = torch.randint(0, self.nu, (batch_size, self.config.nq_hcf), generator=generator)

        # Context tensors
        c_x = self.x_norm[c_idx_swrc]
        c_u = self.u_norm[c_idx_hcf]
        context_coord = torch.cat([c_x, c_u], dim=1).unsqueeze(-1)
        context_channel = torch.cat(
            [
                torch.full_like(c_x, CHANNEL_SWRC, dtype=torch.long),
                torch.full_like(c_u, CHANNEL_HCF, dtype=torch.long),
            ],
            dim=1,
        )
        context_value = torch.cat(
            [
                torch.gather(raw_batch["se_x"], dim=1, index=c_idx_swrc),
                torch.gather(raw_batch["gkr_u"], dim=1, index=c_idx_hcf),
            ],
            dim=1,
        ).unsqueeze(-1)

        # Query tensors
        q_x = self.x_norm[q_idx_swrc]
        q_u = self.u_norm[q_idx_hcf]
        query_coord = torch.cat([q_x, q_u], dim=1).unsqueeze(-1)
        query_channel = torch.cat(
            [
                torch.full_like(q_x, CHANNEL_SWRC, dtype=torch.long),
                torch.full_like(q_u, CHANNEL_HCF, dtype=torch.long),
            ],
            dim=1,
        )
        target_value = torch.cat(
            [
                torch.gather(raw_batch["se_x"], dim=1, index=q_idx_swrc),
                torch.gather(raw_batch["gkr_u"], dim=1, index=q_idx_hcf),
            ],
            dim=1,
        )
        query_weight = torch.cat(
            [
                self.weight_x[q_idx_swrc],
                self.weight_u[q_idx_hcf],
            ],
            dim=1,
        )

        return {
            "context_coord": context_coord,
            "context_channel": context_channel,
            "context_value": context_value,
            "query_coord": query_coord,
            "query_channel": query_channel,
            "target_value": target_value,
            "query_weight": query_weight,
            "raw_batch": raw_batch,
        }


def create_raw_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def cycle(iterable) -> Iterator:
    while True:
        for item in iterable:
            yield item
