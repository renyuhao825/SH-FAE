"""Evaluation metrics and statistics for SH-FAE."""

from __future__ import annotations

import numpy as np
import torch


def rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Root Mean Square Error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def normalized_rmse(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Normalized Root Mean Square Error (divided by range max - min)."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    data_range = float(np.max(y_true) - np.min(y_true))
    if data_range <= 1e-12:
        return rmse(y_true, y_pred)
    return float(rmse(y_true, y_pred) / data_range)


def mae(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Mean Absolute Error."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor) -> float:
    """Coefficient of determination R^2."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def check_monotonicity(y: np.ndarray, decreasing: bool = True, tol: float = 1e-6) -> bool:
    """Check if 1D array is strictly/weakly monotonic within tolerance."""
    diff = np.diff(y)
    if decreasing:
        return bool(np.all(diff <= tol))
    return bool(np.all(diff >= -tol))
