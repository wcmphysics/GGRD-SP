"""Exact Gaussian Process noise decomposition and denoising for regional XPS."""

from __future__ import annotations

import copy
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch
from linear_operator.operators import DiagLinearOperator

from denoise.common import (
    _SNR_DEFINITIONS,
    _validate_spectrum,
)

_KERNEL_TYPES: tuple[str, ...] = ("rbf", "matern12", "matern32", "matern52")
_MATERN_NU: dict[str, float] = {
    "matern52": 2.5,
    "matern32": 1.5,
    "matern12": 0.5,
}
_NOISE_MODELS: tuple[str, ...] = ("homoscedastic", "poisson_scaled")
_CONFIDENCE_TYPES: tuple[str, ...] = ("observation", "latent")


class _ScaledHomoskedasticNoise(
    gpytorch.likelihoods.noise_models.HomoskedasticNoise
):
    """Noise with variance ``sigma^2 * weights`` (one learned scalar sigma^2).

    Used for the ``poisson_scaled`` noise model where the noise variance grows
    proportionally to the signal intensity.
    """

    def __init__(self, weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.register_buffer("weights", weights)

    def forward(
        self,
        *params: Any,
        shape: Any = None,
        noise: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """Return the diagonal noise covariance ``sigma^2 * weights``."""
        if noise is not None:
            return DiagLinearOperator(noise)
        return DiagLinearOperator(self.noise * self.weights)


class SpectrumExactGPModel(gpytorch.models.ExactGP):
    """Exact GP (constant mean + scaled kernel) for one regional spectrum.

    After fitting, ``energy_center``, ``intensity_scale``, ``train_energy``
    and ``noise_weights`` hold the information needed to map predictions back
    to the original energy/intensity units.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        covar_module: gpytorch.kernels.Kernel,
        batch_shape: torch.Size = torch.Size(),
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = covar_module
        self.energy_center: float = 0.0
        self.intensity_scale: float = 1.0
        self.train_energy: np.ndarray = np.empty(0)
        self.noise_weights: np.ndarray = np.empty(0)

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Return the GP prior/posterior over the latent function."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def get_default_gp_denoise_config() -> dict[str, Any]:
    """Return default configuration for Gaussian Process noise analysis/denoise.

    Returns
    -------
    dict[str, Any]
        Default GP configuration dictionary.
    """
    return {
        "kernel": {
            "type": "matern52",
            "initial_lengthscale_ev": 1.0,
            "min_lengthscale_ev": None,
            "min_lengthscale_points": 3.0,
            "max_lengthscale_ev": None,
        },
        "noise_model": "homoscedastic",
        "poisson_weight_floor": 1e-2,
        "training": {"num_iterations": 100, "learning_rate": 0.1},
        "confidence_type": "observation",
        "confidence_z": 1.96,
        "snr_definition": "range",
        "device": "cpu",
        "show_progress": True,
        "batch_size": 64,
        "num_workers": 0,
        "batch_verify_rtol": None,
    }


def _resolve_gp_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user GP configuration over defaults and validate all keys."""
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_gp_denoise_config()
    user_cfg = copy.deepcopy(config or {})
    allowed_keys = set(cfg) | {"splits", "dies", "regions", "unpaired_split", "method"}
    for key in user_cfg:
        if key not in allowed_keys:
            raise ValueError(
                f"Unknown noise analysis config key '{key}'. "
                f"Valid keys: {sorted(allowed_keys)}"
            )
    for nested in ("kernel", "training"):
        if not isinstance(user_cfg.get(nested, {}), dict):
            raise ValueError(
                f"'{nested}' must be a dict, got {user_cfg[nested]!r}."
            )
        extra = set(user_cfg.get(nested, {})) - set(cfg[nested])
        if extra:
            raise ValueError(
                f"Unknown '{nested}' config keys {sorted(extra)}. "
                f"Valid keys: {sorted(cfg[nested])}"
            )
        cfg[nested] = {**cfg[nested], **user_cfg.pop(nested, {})}
    cfg.update(user_cfg)

    choices = {
        ("kernel", "type"): _KERNEL_TYPES,
        "noise_model": _NOISE_MODELS,
        "confidence_type": _CONFIDENCE_TYPES,
        "snr_definition": _SNR_DEFINITIONS,
    }
    for key, valid in choices.items():
        val = cfg[key[0]][key[1]] if isinstance(key, tuple) else cfg[key]
        if val not in valid:
            raise ValueError(
                f"Invalid {key}={val!r}; choose one of {valid}."
            )
    if cfg["confidence_z"] <= 0:
        raise ValueError(
            f"confidence_z must be > 0, got {cfg['confidence_z']}."
        )
    if (
        cfg["training"]["num_iterations"] < 1
        or cfg["training"]["learning_rate"] <= 0
    ):
        raise ValueError(f"Invalid training config: {cfg['training']}.")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        val = cfg[key]
        if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
            raise ValueError(
                f"{key} must be an integer >= {minimum}, got {val!r}."
            )
    rtol = cfg["batch_verify_rtol"]
    if rtol is not None and (not isinstance(rtol, (int, float)) or rtol <= 0):
        raise ValueError(
            f"batch_verify_rtol must be None or > 0, got {rtol!r}."
        )
    cfg["_resolved"] = True
    return cfg


def _build_kernel(
    kernel_cfg: dict[str, Any],
    spacing: float | np.ndarray,
    energy_range: float | np.ndarray,
    batch_size: int | None = None,
) -> gpytorch.kernels.Kernel:
    """Build the scaled stationary kernel with bounded lengthscale (in eV)."""
    spacing_arr = np.asarray(spacing, float)
    range_arr = np.asarray(energy_range, float)
    ls_min = kernel_cfg["min_lengthscale_ev"]
    if ls_min is None:
        ls_min = float(kernel_cfg["min_lengthscale_points"]) * spacing_arr
    ls_min = np.broadcast_to(np.asarray(ls_min, float), spacing_arr.shape)
    ls_max = kernel_cfg["max_lengthscale_ev"]
    ls_max = (
        range_arr
        if ls_max is None
        else np.full(spacing_arr.shape, float(ls_max))
    )
    if not np.all((0 < ls_min) & (ls_min < ls_max)):
        raise ValueError(
            f"Invalid lengthscale bounds: min={ls_min.min():.4g} eV, "
            f"max={ls_max.min():.4g} eV (need 0 < min < max)."
        )
    initial = np.clip(kernel_cfg["initial_lengthscale_ev"], ls_min, ls_max)
    if batch_size is None:
        batch_shape = torch.Size()
        lower, upper, init_val = float(ls_min), float(ls_max), float(initial)
    else:
        batch_shape = torch.Size([batch_size])
        lower, upper, init_val = (
            torch.tensor(np.asarray(v, float)).view(-1, 1, 1)
            for v in (ls_min, ls_max, initial)
        )
    constraint = gpytorch.constraints.Interval(lower, upper)
    kernel_type = kernel_cfg["type"]
    if kernel_type in _MATERN_NU:
        base = gpytorch.kernels.MaternKernel(
            nu=_MATERN_NU[kernel_type],
            lengthscale_constraint=constraint,
            batch_shape=batch_shape,
        )
    else:
        base = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=constraint,
            batch_shape=batch_shape,
        )
    base.lengthscale = init_val
    return gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)


def _run_single_spectrum_gp_fit(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | pd.Series | None = None,
    record_history: bool = False,
) -> tuple[SpectrumExactGPModel, dict[str, list[float]]]:
    """Fit an exact GP to one spectrum and optionally record history."""
    cfg = _resolve_gp_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    energy_arr, intensity_arr, spacing, energy_range, intensity_scale = (
        _validate_spectrum(energy, intensity, label)
    )

    energy_center = float(energy_arr.mean())
    device = torch.device(cfg["device"])
    train_x = torch.tensor(
        energy_arr - energy_center, device=device
    ).unsqueeze(-1)
    scaled_intensity = intensity_arr / intensity_scale
    train_y = torch.tensor(scaled_intensity, device=device)
    weights = np.ones_like(scaled_intensity)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    )
    if cfg["noise_model"] == "poisson_scaled":
        weights = np.clip(scaled_intensity, cfg["poisson_weight_floor"], None)
        likelihood.noise_covar = _ScaledHomoskedasticNoise(
            torch.tensor(weights, device=device),
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
        )
    likelihood.noise = 1e-3
    covar = _build_kernel(cfg["kernel"], spacing, energy_range)
    model = SpectrumExactGPModel(train_x, train_y, likelihood, covar)
    model.to(device).double()
    model.mean_module.constant.data.fill_(float(scaled_intensity.mean()))

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    history: dict[str, list[float]] = {
        "loss": [],
        "noise_sigma": [],
        "lengthscale": [],
        "outputscale": [],
    }
    mean_weight = float(np.mean(weights))
    scale_sq = float(intensity_scale**2)

    for iteration in range(cfg["training"]["num_iterations"]):
        optimizer.zero_grad()
        try:
            loss = -mll(model(train_x), train_y)
        except Exception as exc:
            raise RuntimeError(
                f"{label}: GP marginal likelihood failed at iteration "
                f"{iteration}: {exc}"
            ) from exc
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{label}: non-finite GP loss ({loss.item()}) at iteration "
                f"{iteration}."
            )
        loss.backward()
        optimizer.step()
        if record_history:
            noise_var = float(likelihood.noise.item())
            history["loss"].append(float(loss.item()))
            history["noise_sigma"].append(
                float(np.sqrt(noise_var * mean_weight) * intensity_scale)
            )
            history["lengthscale"].append(
                float(model.covar_module.base_kernel.lengthscale.item())
            )
            history["outputscale"].append(
                float(model.covar_module.outputscale.item() * scale_sq)
            )

    model.eval()
    model.energy_center = energy_center
    model.intensity_scale = intensity_scale
    model.train_energy = energy_arr
    model.noise_weights = weights
    return model, history


def fit_regional_spectrum_by_gp(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | pd.Series | None = None,
) -> SpectrumExactGPModel:
    """Fit an exact GP ``y = f(E) + eps`` to one regional spectrum.

    Parameters
    ----------
    energy, intensity : np.ndarray
        1D binding energy (eV) and intensity arrays of the spectrum.
    config : dict[str, Any] | None
        GP noise analysis config (see ``get_default_gp_denoise_config``).
    meta : dict | pd.Series | None
        Optional meta data of the spectrum, only used in error messages.

    Returns
    -------
    SpectrumExactGPModel
        The fitted GP model (in eval mode).
    """
    return _run_single_spectrum_gp_fit(energy, intensity, config, meta)[0]


def trace_gp_hyperparameter_convergence(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | pd.Series | None = None,
) -> dict[str, list[float]]:
    """Trace GP hyperparameter values across Adam optimization iterations.

    Parameters
    ----------
    energy, intensity : np.ndarray
        1D binding energy (eV) and intensity arrays of the spectrum.
    config : dict[str, Any] | None, optional
        Noise analysis configuration dictionary.
    meta : dict | pd.Series | None, optional
        Optional metadata of the spectrum for error messages.

    Returns
    -------
    dict[str, list[float]]
        Dictionary containing convergence trajectories:
        'loss', 'noise_sigma', 'lengthscale', and 'outputscale'.
    """
    return _run_single_spectrum_gp_fit(
        energy, intensity, config, meta, record_history=True
    )[1]


def create_mean_and_confidence_curves(
    gp_model: SpectrumExactGPModel,
    energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the GP mean curve and the confidence interval half-width curve.

    Parameters
    ----------
    gp_model : SpectrumExactGPModel
        Model returned by ``fit_regional_spectrum_by_gp``.
    energy : np.ndarray
        1D energy grid (eV) where the curves are evaluated.
    config : dict[str, Any] | None
        Uses ``confidence_type`` and ``confidence_z``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_curve, confidence_interval_curve)`` in intensity units.
    """
    cfg = _resolve_gp_config(config)
    energy_arr = np.asarray(energy, dtype=np.float64)
    if energy_arr.ndim != 1 or not np.isfinite(energy_arr).all():
        raise ValueError("energy must be a finite 1D array.")
    device = gp_model.train_inputs[0].device
    eval_x = torch.tensor(
        energy_arr - gp_model.energy_center, device=device
    ).unsqueeze(-1)
    gp_model.eval()
    with torch.no_grad(), gpytorch.settings.debug(False):
        latent = gp_model(eval_x)
        mean = latent.mean.cpu().numpy()
        variance = latent.variance.clamp_min(0.0).cpu().numpy()
    if cfg["confidence_type"] == "observation":
        sorted_order = np.argsort(gp_model.train_energy)
        weights = np.interp(
            energy_arr,
            gp_model.train_energy[sorted_order],
            gp_model.noise_weights[sorted_order],
        )
        variance = (
            variance + float(gp_model.likelihood.noise.item()) * weights
        )
    scale = gp_model.intensity_scale
    mean_curve = mean * scale
    confidence_curve = cfg["confidence_z"] * np.sqrt(variance) * scale
    if not (
        np.isfinite(mean_curve).all() and np.isfinite(confidence_curve).all()
    ):
        raise RuntimeError(
            "GP prediction produced non-finite mean or confidence values."
        )
    return mean_curve, confidence_curve


def _summarize_gp_fit(
    gp_model: SpectrumExactGPModel,
    mean_curve: np.ndarray,
    snr_definition: str,
) -> tuple[float, float, float]:
    """Return ``(noise_sigma, snr, lengthscale_ev)`` of a fitted GP."""
    noise_variance = float(gp_model.likelihood.noise.item())
    noise_sigma = float(
        np.sqrt(noise_variance * np.mean(gp_model.noise_weights))
        * gp_model.intensity_scale
    )
    if not np.isfinite(noise_sigma) or noise_variance <= 1.5e-8:
        raise RuntimeError(
            f"GP fit gave an invalid noise level (variance="
            f"{noise_variance:.3g}, sigma={noise_sigma:.3g}); the noise hit "
            "its lower bound, so the spectrum may be noise-free or the kernel "
            "too flexible. Consider raising kernel 'min_lengthscale_points'."
        )
    if snr_definition == "range":
        signal_strength = float(np.max(mean_curve) - np.min(mean_curve))
    else:
        signal_strength = float(np.std(mean_curve))
    lengthscale = float(gp_model.covar_module.base_kernel.lengthscale.item())
    return noise_sigma, signal_strength / noise_sigma, lengthscale


def denoise_spectrum_gp(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Denoise one 1D regional spectrum using exact Gaussian Process regression.

    Parameters
    ----------
    energy : np.ndarray
        1D array of binding energies in eV.
    intensity : np.ndarray
        1D array of measured spectral intensities.
    config : dict[str, Any] | None, optional
        GP configuration (see get_default_gp_denoise_config).
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary for error reporting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        Tuple of (mean_curve, confidence_curve,
        (noise_sigma, snr, lengthscale_ev)).
    """
    cfg = _resolve_gp_config(config)
    gp_model = fit_regional_spectrum_by_gp(energy, intensity, cfg, meta=meta)
    mean_curve, confidence_curve = create_mean_and_confidence_curves(
        gp_model, energy, cfg
    )
    summary_tuple = _summarize_gp_fit(
        gp_model, mean_curve, cfg["snr_definition"]
    )
    return mean_curve, confidence_curve, summary_tuple


def _fit_and_summarize_batch_gp(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit B spectra simultaneously with one batched exact GP."""
    batch_size = len(metas)
    labels = [f"spectrum meta={meta}" for meta in metas]
    prepared = [
        _validate_spectrum(e, i, lbl)
        for e, i, lbl in zip(energies, intensities, labels)
    ]
    energies_arr = np.stack([p[0] for p in prepared])
    intensities_arr = np.stack([p[1] for p in prepared])
    spacing, energy_range, scales = (
        np.array([p[k] for p in prepared]) for k in (2, 3, 4)
    )
    device = torch.device(cfg["device"])
    train_x = torch.tensor(
        energies_arr - energies_arr.mean(axis=1, keepdims=True), device=device
    ).unsqueeze(-1)
    scaled = intensities_arr / scales[:, None]
    train_y = torch.tensor(scaled, device=device)
    weights = np.ones_like(scaled)
    batch_shape = torch.Size([batch_size])
    noise_constraint = gpytorch.constraints.GreaterThan(1e-8)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint, batch_shape=batch_shape
    )
    if cfg["noise_model"] == "poisson_scaled":
        weights = np.clip(scaled, cfg["poisson_weight_floor"], None)
        likelihood.noise_covar = _ScaledHomoskedasticNoise(
            torch.tensor(weights, device=device),
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
    likelihood.noise = 1e-3
    covar = _build_kernel(
        cfg["kernel"], spacing, energy_range, batch_size=batch_size
    )
    model = SpectrumExactGPModel(
        train_x, train_y, likelihood, covar, batch_shape
    )
    model.to(device).double()
    constant = model.mean_module.constant
    initial_mean = torch.tensor(
        scaled.mean(axis=1), device=device
    ).reshape(constant.shape)
    constant.data.copy_(initial_mean)

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for iteration in range(cfg["training"]["num_iterations"]):
        optimizer.zero_grad()
        try:
            loss = -mll(model(train_x), train_y)
        except Exception as exc:
            raise RuntimeError(
                f"Batched GP marginal likelihood failed for {batch_size} "
                f"spectra at iteration {iteration}: {exc}"
            ) from exc
        bad = ~torch.isfinite(loss)
        if bad.any():
            first = int(bad.nonzero()[0])
            raise RuntimeError(
                f"{labels[first]}: non-finite GP loss at iteration "
                f"{iteration}."
            )
        loss.sum().backward()
        optimizer.step()

    model.eval()
    with torch.no_grad(), gpytorch.settings.debug(False):
        latent = model(train_x)
        mean = latent.mean.cpu().numpy()
        variance = latent.variance.clamp_min(0.0).cpu().numpy()
        noise_variance = (
            likelihood.noise.detach().cpu().numpy().reshape(batch_size)
        )
        lengthscale = (
            model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        )
    if cfg["confidence_type"] == "observation":
        variance = variance + noise_variance[:, None] * weights
    mean_curves = mean * scales[:, None]
    confidence_curves = (
        cfg["confidence_z"] * np.sqrt(variance) * scales[:, None]
    )
    noise_sigma = np.sqrt(noise_variance * weights.mean(axis=1)) * scales
    for idx in range(batch_size):
        if not (
            np.isfinite(mean_curves[idx]).all()
            and np.isfinite(confidence_curves[idx]).all()
        ):
            raise RuntimeError(
                f"{labels[idx]}: GP prediction produced non-finite values."
            )
        if not np.isfinite(noise_sigma[idx]) or noise_variance[idx] <= 1.5e-8:
            raise RuntimeError(
                f"{labels[idx]}: GP fit gave an invalid noise level "
                f"(variance={noise_variance[idx]:.3g}, "
                f"sigma={noise_sigma[idx]:.3g}); the noise hit its lower "
                "bound, so the spectrum may be noise-free or the kernel too "
                "flexible. Consider raising kernel 'min_lengthscale_points'."
            )
    if cfg["snr_definition"] == "range":
        strength = mean_curves.max(axis=1) - mean_curves.min(axis=1)
    else:
        strength = mean_curves.std(axis=1)
    summary = np.stack(
        [
            noise_sigma,
            strength / noise_sigma,
            lengthscale.reshape(batch_size),
        ],
        axis=1,
    )
    return mean_curves, confidence_curves, summary


def _analyse_sequential_gp(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyse spectra one by one with single-spectrum GP building blocks."""
    signal = np.empty(energies.shape, dtype=np.float64)
    noise = np.empty(energies.shape, dtype=np.float64)
    summary = np.empty((len(metas), 3), dtype=np.float64)
    for idx, meta in enumerate(metas):
        signal[idx], noise[idx], summary[idx] = denoise_spectrum_gp(
            energies[idx], intensities[idx], cfg, meta=meta
        )
    return signal, noise, summary


def denoise_batch_gp(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise a chunk/batch of spectra with exact Gaussian Process regression.

    Parameters
    ----------
    energies : np.ndarray
        2D array of binding energies of shape (B, N_points).
    intensities : np.ndarray
        2D array of measured intensities of shape (B, N_points).
    metas : list[dict[str, Any]]
        List of metadata dictionaries for each spectrum in the batch.
    config : dict[str, Any] | None, optional
        GP configuration dictionary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (signal_curves (B, N), confidence_curves (B, N),
        summary_metrics (B, 3)).
    """
    cfg = _resolve_gp_config(config)
    if cfg["batch_size"] == 1:
        return _analyse_sequential_gp(energies, intensities, metas, cfg)
    try:
        result = _fit_and_summarize_batch_gp(energies, intensities, metas, cfg)
    except RuntimeError as exc:
        _analyse_sequential_gp(energies, intensities, metas, cfg)
        raise RuntimeError(
            f"Batched GP fit failed for a chunk of {len(metas)} spectra "
            f"although each fits individually: {exc}"
        ) from exc
    rtol = cfg["batch_verify_rtol"]
    if rtol is not None:
        reference = _analyse_sequential_gp(
            energies[:1], intensities[:1], metas[:1], cfg
        )[2]
        batched_sigma, reference_sigma = result[2][0, 0], reference[0, 0]
        if abs(batched_sigma - reference_sigma) > rtol * abs(reference_sigma):
            raise RuntimeError(
                f"Batched noise sigma {batched_sigma:.6g} differs from the "
                f"sequential {reference_sigma:.6g} by more than rtol={rtol} "
                f"for spectrum meta={metas[0]}."
            )
    return result
