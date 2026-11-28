"""Posterior mean contributions of the die-flux GP covariance kernels."""

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
    n_items = len(datasets)
    n_cols = min(2, n_items) if n_items > 0 else 1
    n_rows = max(1, (n_items + n_cols - 1) // n_cols)
    fig, axes_array = plt.subplots(
        n_rows, n_cols, squeeze=False, figsize=(7.0 * n_cols, 5.0 * n_rows)
    )
    axes_flat = tuple(axes_array.ravel())
    for extra_ax in axes_flat[n_items:]:
        extra_ax.set_visible(False)
    axes = axes_flat[:n_items]
    for axis, (key, dataset) in zip(axes, datasets.items()):
        model, likelihood = gp_bundle["models"][key]
        if isinstance(key, tuple) and len(key) == 3:
            grp_parts = [str(p) for p in (key[1], key[2]) if p is not None]
            title_label = (
                f"{key[0]} | {'-'.join(grp_parts)}" if grp_parts else str(key[0])
            )
        else:
            title_label = str(key)
        rows = dataset["die_df"]
        rows = rows[rows["die_idx"] == die]
        if rows.empty:
            axis.set_title(f"{title_label} (Die {die} - No Data)")
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
        axis.set_title(f"{title_label} (Die {die})")
        axis.set_xlabel(f"Elapsed time from {config['reference_date']} (days)")
        axis.set_ylabel("Flux / contribution (counts · eV)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("GP posterior mean kernel contributions")
    fig.tight_layout()
    return fig, axes
