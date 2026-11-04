"""Posterior mean contributions of the die-flux GP covariance kernels."""

from __future__ import annotations

from typing import Any

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from linear_operator.utils.cholesky import psd_safe_cholesky


def _kernel_terms(kernel: gpytorch.kernels.Kernel) -> list[tuple[str, Any]]:
    """Expand sums and products into additive covariance terms."""
    if isinstance(kernel, gpytorch.kernels.AdditiveKernel):
        return [term for child in kernel.kernels for term in _kernel_terms(child)]
    if isinstance(kernel, gpytorch.kernels.ProductKernel):
        terms = [("", None)]
        for child in kernel.kernels:
            terms = [
                (f"{name} × {child_name}" if name else child_name,
                 child_kernel if product is None else product * child_kernel)
                for name, product in terms
                for child_name, child_kernel in _kernel_terms(child)
            ]
        return terms
    base = kernel.base_kernel if isinstance(kernel, gpytorch.kernels.ScaleKernel) else kernel
    return [(type(base).__name__, kernel)]


def plot_gp_flux_kernel_contributions(
    gp_bundle: dict[str, Any], plot_config: dict[str, Any] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Plot additive posterior mean terms in physical flux units per tool."""
    config = {**gp_bundle["config"], **(plot_config or {})}
    die = config["plot"]["plot_die"]
    datasets = gp_bundle["datasets"]
    fig, axes_array = plt.subplots(1, len(datasets), squeeze=False, figsize=(14, 5.5))
    axes = tuple(axes_array[0])
    for axis, (tool, dataset) in zip(axes, datasets.items()):
        model, likelihood = gp_bundle["models"][tool]
        rows = dataset["die_df"]
        rows = rows[rows["die_idx"] == die]
        if rows.empty:
            axis.set_title(f"{tool} (Die {die} - No Data)")
            continue
        start, end = rows["time_days"].min(), rows["time_days"].max()
        span = max(end - start, 1.0)
        times = np.linspace(start - 0.05 * span, end + 0.05 * span, 200)
        evaluation = torch.tensor(np.column_stack([times, np.full_like(times, die)]), dtype=torch.float32)
        train_x = model.train_inputs[0]
        with torch.no_grad():
            covariance = model.covar_module(train_x, train_x).to_dense()
            covariance += torch.eye(len(train_x), dtype=train_x.dtype) * likelihood.noise
            residual = model.train_targets - model.mean_module(train_x)
            factor = psd_safe_cholesky(covariance)
            weights = torch.cholesky_solve(residual.unsqueeze(-1), factor).squeeze(-1)
            scale = dataset["scaler"]["std"]
            baseline = model.mean_module(evaluation).cpu().numpy() * scale + dataset["scaler"]["mean"]
            total = baseline.copy()
            axis.plot(times, baseline, label="Baseline", linestyle="--")
            for index, (name, kernel) in enumerate(_kernel_terms(model.covar_module), 1):
                contribution = (kernel(evaluation, train_x).to_dense() @ weights).cpu().numpy() * scale
                total += contribution
                axis.plot(times, contribution, label=f"{name} {index}")
            axis.plot(times, total, color="black", linewidth=2, label="Total prediction")
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set_title(f"{tool} (Die {die})")
        axis.set_xlabel(f"Elapsed time from {config['reference_date']} (days)")
        axis.set_ylabel("Flux / contribution (counts · eV)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("GP posterior mean kernel contributions")
    fig.tight_layout()
    return fig, axes
