"""Gaussian-process noise and SNR analysis of regional spectra.

Every regional spectrum is modelled as ``y = f(E) + eps`` where ``f`` is a
GP with a configurable smooth kernel and ``eps`` is Gaussian noise whose
level is learned from the data. The learned noise standard deviation is the
noise strength of the spectrum and, together with the GP mean curve, gives a
signal-to-noise ratio (SNR).

Main entry point: ``perform_spectral_noise_analysis``. Building blocks:
``fit_regional_spectrum_by_gp`` and ``create_mean_and_confidence_curves``.
"""

from __future__ import annotations

import copy
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch
from linear_operator.operators import DiagLinearOperator
from tqdm.auto import tqdm

_VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
_KERNEL_TYPES: tuple[str, ...] = ("rbf", "matern12", "matern32", "matern52")
_MATERN_NU: dict[str, float] = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}
_NOISE_MODELS: tuple[str, ...] = ("homoscedastic", "poisson_scaled")
_CONFIDENCE_TYPES: tuple[str, ...] = ("observation", "latent")
_SNR_DEFINITIONS: tuple[str, ...] = ("range", "std")
_NEW_META_COLUMNS: tuple[str, ...] = (
    "noise_analysis_index",
    "gp_noise_sigma",
    "gp_snr",
    "gp_lengthscale",
)


class _ScaledHomoskedasticNoise(gpytorch.likelihoods.noise_models.HomoskedasticNoise):
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

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Return the GP prior/posterior over the latent function."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def get_default_noise_analysis_config() -> dict[str, Any]:
    """Return the default configuration for ``perform_spectral_noise_analysis``.

    Keys
    ----
    splits : list[str] | None
        Splits to analyse (``train``/``val``/``test``); None means all splits.
    dies : list[int] | None
        Die ids to analyse; None means all dies in the selected splits.
    regions : list[str] | None
        Region names to analyse; None means all regions.
    unpaired_split : str | None
        Unpaired spectra (split missing) are treated as this split; None
        excludes unpaired spectra.
    kernel : dict
        ``type`` (rbf/matern12/matern32/matern52), ``initial_lengthscale_ev``,
        ``min_lengthscale_ev`` (None -> ``min_lengthscale_points`` times the
        point spacing) and ``max_lengthscale_ev`` (None -> energy range).
    noise_model : str
        ``homoscedastic`` (default) or ``poisson_scaled``.
    poisson_weight_floor : float
        Lower bound (fraction of max intensity) of the Poisson noise weights.
    training : dict
        ``num_iterations`` and ``learning_rate`` of the Adam optimizer.
    confidence_type : str
        ``observation`` (latent + noise variance) or ``latent``.
    confidence_z : float
        Number of sigmas of the confidence band (1.96 -> 95 %).
    snr_definition : str
        ``range``: (max(mean)-min(mean))/sigma_noise or ``std``:
        std(mean)/sigma_noise.
    device : str
        Torch device for fitting.
    show_progress : bool
        Show a tqdm progress bar.
    batch_size : int
        Number of spectra fitted simultaneously by one batched GP (>= 1).
        1 fits spectra one by one with ``fit_regional_spectrum_by_gp``.
    num_workers : int
        Worker processes for chunks of spectra; 0 or 1 disables
        multiprocessing (default). Each worker uses a single torch thread.
    batch_verify_rtol : float | None
        If set (and ``batch_size`` > 1), the first spectrum of every chunk is
        also fitted sequentially and its noise sigma must agree with the
        batched one within this relative tolerance, else an error is raised.
        None (default) disables the check.
    """
    return {
        "splits": None,
        "dies": None,
        "regions": None,
        "unpaired_split": None,
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


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a user config over the defaults and validate every value.

    A config that was already resolved (marked internally) is returned as is,
    so inner functions do not repeat the merge and validation per spectrum.
    """
    if config is not None and config.get("_resolved"):
        return config
    cfg = get_default_noise_analysis_config()
    user = copy.deepcopy(config or {})
    for key in user:
        if key not in cfg:
            raise ValueError(
                f"Unknown noise analysis config key '{key}'. "
                f"Valid keys: {sorted(cfg)}"
            )
    for key in ("splits", "dies", "regions"):
        value = user.get(key)
        if value is not None and (isinstance(value, str) or len(value) == 0):
            raise ValueError(f"'{key}' must be None or a non-empty list, got {value!r}.")
    for nested in ("kernel", "training"):
        if not isinstance(user.get(nested, {}), dict):
            raise ValueError(f"'{nested}' must be a dict, got {user[nested]!r}.")
        extra = set(user.get(nested, {})) - set(cfg[nested])
        if extra:
            raise ValueError(
                f"Unknown '{nested}' config keys {sorted(extra)}. "
                f"Valid keys: {sorted(cfg[nested])}"
            )
        cfg[nested] = {**cfg[nested], **user.pop(nested, {})}
    cfg.update(user)
    choices = {
        ("kernel", "type"): _KERNEL_TYPES,
        "noise_model": _NOISE_MODELS,
        "confidence_type": _CONFIDENCE_TYPES,
        "snr_definition": _SNR_DEFINITIONS,
    }
    for key, valid in choices.items():
        value = cfg[key[0]][key[1]] if isinstance(key, tuple) else cfg[key]
        if value not in valid:
            raise ValueError(f"Invalid {key}={value!r}; choose one of {valid}.")
    if cfg["confidence_z"] <= 0:
        raise ValueError(f"confidence_z must be > 0, got {cfg['confidence_z']}.")
    if cfg["training"]["num_iterations"] < 1 or cfg["training"]["learning_rate"] <= 0:
        raise ValueError(f"Invalid training config: {cfg['training']}.")
    for key, minimum in (("batch_size", 1), ("num_workers", 0)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r}.")
    rtol = cfg["batch_verify_rtol"]
    if rtol is not None and (not isinstance(rtol, (int, float)) or rtol <= 0):
        raise ValueError(f"batch_verify_rtol must be None or > 0, got {rtol!r}.")
    cfg["_resolved"] = True
    return cfg


def _build_kernel(
    kernel_cfg: dict[str, Any],
    spacing: float | np.ndarray,
    energy_range: float | np.ndarray,
    batch_size: int | None = None,
) -> gpytorch.kernels.Kernel:
    """Build the scaled stationary kernel with bounded lengthscale (in eV).

    With ``batch_size`` set, ``spacing`` and ``energy_range`` are arrays of
    that length and the kernel carries one lengthscale per spectrum.
    """
    spacing, energy_range = np.asarray(spacing, float), np.asarray(energy_range, float)
    ls_min = kernel_cfg["min_lengthscale_ev"]
    if ls_min is None:
        ls_min = float(kernel_cfg["min_lengthscale_points"]) * spacing
    ls_min = np.broadcast_to(np.asarray(ls_min, float), spacing.shape)
    ls_max = kernel_cfg["max_lengthscale_ev"]
    ls_max = energy_range if ls_max is None else np.full(spacing.shape, float(ls_max))
    if not np.all((0 < ls_min) & (ls_min < ls_max)):
        raise ValueError(
            f"Invalid lengthscale bounds: min={ls_min.min():.4g} eV, "
            f"max={ls_max.min():.4g} eV (need 0 < min < max)."
        )
    initial = np.clip(kernel_cfg["initial_lengthscale_ev"], ls_min, ls_max)
    if batch_size is None:
        batch_shape = torch.Size()
        lower, upper, initial = float(ls_min), float(ls_max), float(initial)
    else:
        batch_shape = torch.Size([batch_size])
        lower, upper, initial = (
            torch.tensor(np.asarray(v, float)).view(-1, 1, 1) for v in (ls_min, ls_max, initial)
        )
    constraint = gpytorch.constraints.Interval(lower, upper)
    kernel_type = kernel_cfg["type"]
    if kernel_type in _MATERN_NU:
        base = gpytorch.kernels.MaternKernel(
            nu=_MATERN_NU[kernel_type], lengthscale_constraint=constraint, batch_shape=batch_shape
        )
    else:
        base = gpytorch.kernels.RBFKernel(lengthscale_constraint=constraint, batch_shape=batch_shape)
    base.lengthscale = initial
    return gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)


def _validate_spectrum(
    energy: np.ndarray, intensity: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Validate one spectrum; return ``(energy, intensity, spacing, range, scale)``."""
    energy = np.asarray(energy, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if energy.ndim != 1 or energy.shape != intensity.shape or energy.size < 5:
        raise ValueError(
            f"{label}: energy and intensity must be 1D arrays of equal length "
            f">= 5, got shapes {energy.shape} and {intensity.shape}."
        )
    if not (np.isfinite(energy).all() and np.isfinite(intensity).all()):
        raise ValueError(f"{label}: non-finite values in energy or intensity.")
    if np.unique(energy).size != energy.size:
        raise ValueError(f"{label}: duplicate energy values are not supported.")
    spacing = float(np.median(np.abs(np.diff(np.sort(energy)))))
    energy_range = float(energy.max() - energy.min())
    intensity_scale = float(np.max(np.abs(intensity)))
    if spacing <= 0 or energy_range <= 0 or intensity_scale <= 0:
        raise ValueError(
            f"{label}: degenerate spectrum (point spacing={spacing}, "
            f"energy range={energy_range}, max |intensity|={intensity_scale})."
        )
    return energy, intensity, spacing, energy_range, intensity_scale


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
        Noise analysis config (see ``get_default_noise_analysis_config``).
    meta : dict | pd.Series | None
        Optional meta data of the spectrum, only used in error messages.

    Returns
    -------
    SpectrumExactGPModel
        The fitted GP model (in eval mode).
    """
    cfg = _resolve_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    energy, intensity, spacing, energy_range, intensity_scale = _validate_spectrum(
        energy, intensity, label
    )

    energy_center = float(energy.mean())
    device = torch.device(cfg["device"])
    train_x = torch.tensor(energy - energy_center, device=device).unsqueeze(-1)
    scaled_intensity = intensity / intensity_scale
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for iteration in range(cfg["training"]["num_iterations"]):
        optimizer.zero_grad()
        try:
            loss = -mll(model(train_x), train_y)
        except Exception as exc:  # re-raise with spectrum context
            raise RuntimeError(
                f"{label}: GP marginal likelihood failed at iteration "
                f"{iteration}: {exc}"
            ) from exc
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{label}: non-finite GP loss ({loss.item()}) at iteration {iteration}."
            )
        loss.backward()
        optimizer.step()
    model.eval()
    model.energy_center = energy_center
    model.intensity_scale = intensity_scale
    model.train_energy = energy
    model.noise_weights = weights
    return model


def trace_gp_hyperparameter_convergence(
    energy: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | pd.Series | None = None,
) -> dict[str, list[float]]:
    """Trace GP hyperparameter values across Adam optimization iterations.

    Fits an exact GP to one spectrum while tracking Negative Marginal
    Log-Likelihood (loss), noise standard deviation in physical units,
    kernel lengthscale (eV), and signal outputscale across all iterations.

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
        - 'loss': list of negative MLL values per iteration.
        - 'noise_sigma': list of noise std in physical intensity units.
        - 'lengthscale': list of kernel lengthscales in eV.
        - 'outputscale': list of signal variances in physical intensity units.
    """
    cfg = _resolve_config(config)
    label = f"spectrum meta={dict(meta)}" if meta is not None else "spectrum"
    energy, intensity, spacing, energy_range, intensity_scale = _validate_spectrum(
        energy, intensity, label
    )

    energy_center = float(energy.mean())
    device = torch.device(cfg["device"])
    train_x = torch.tensor(energy - energy_center, device=device).unsqueeze(-1)
    scaled_intensity = intensity / intensity_scale
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
                f"{label}: non-finite GP loss ({loss.item()}) at iteration {iteration}."
            )
        loss.backward()
        optimizer.step()

        noise_var = float(likelihood.noise.item())
        noise_sigma = float(np.sqrt(noise_var * mean_weight) * intensity_scale)
        ls = float(model.covar_module.base_kernel.lengthscale.item())
        out_scale = float(model.covar_module.outputscale.item() * scale_sq)

        history["loss"].append(float(loss.item()))
        history["noise_sigma"].append(noise_sigma)
        history["lengthscale"].append(ls)
        history["outputscale"].append(out_scale)

    return history


def create_mean_and_confidence_curves(
    gp_model: SpectrumExactGPModel,
    energy: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the GP mean curve and the confidence interval half-width curve.

    The confidence band of the spectrum is ``mean +- confidence_interval``.

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
    cfg = _resolve_config(config)
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or not np.isfinite(energy).all():
        raise ValueError("energy must be a finite 1D array.")
    device = gp_model.train_inputs[0].device
    x = torch.tensor(energy - gp_model.energy_center, device=device).unsqueeze(-1)
    gp_model.eval()
    with torch.no_grad(), gpytorch.settings.debug(False):
        latent = gp_model(x)
        mean = latent.mean.cpu().numpy()
        variance = latent.variance.clamp_min(0.0).cpu().numpy()
    if cfg["confidence_type"] == "observation":
        weights = np.interp(
            energy,
            np.sort(gp_model.train_energy),
            gp_model.noise_weights[np.argsort(gp_model.train_energy)],
        )
        variance = variance + float(gp_model.likelihood.noise.item()) * weights
    scale = gp_model.intensity_scale
    mean_curve, confidence_curve = mean * scale, cfg["confidence_z"] * np.sqrt(variance) * scale
    if not (np.isfinite(mean_curve).all() and np.isfinite(confidence_curve).all()):
        raise RuntimeError("GP prediction produced non-finite mean or confidence values.")
    return mean_curve, confidence_curve


def _summarize_gp_fit(
    gp_model: SpectrumExactGPModel, mean_curve: np.ndarray, snr_definition: str
) -> tuple[float, float, float]:
    """Return ``(noise_sigma, snr, lengthscale_ev)`` of a fitted GP."""
    noise_variance = float(gp_model.likelihood.noise.item())
    noise_sigma = float(
        np.sqrt(noise_variance * np.mean(gp_model.noise_weights))
        * gp_model.intensity_scale
    )
    if not np.isfinite(noise_sigma) or noise_variance <= 1.5e-8:
        raise RuntimeError(
            f"GP fit gave an invalid noise level (variance={noise_variance:.3g}, "
            f"sigma={noise_sigma:.3g}); the noise hit its lower bound, so the "
            "spectrum may be noise-free or the kernel too flexible. Consider "
            "raising kernel 'min_lengthscale_points'."
        )
    if snr_definition == "range":
        signal_strength = float(np.max(mean_curve) - np.min(mean_curve))
    else:
        signal_strength = float(np.std(mean_curve))
    lengthscale = float(gp_model.covar_module.base_kernel.lengthscale.item())
    return noise_sigma, signal_strength / noise_sigma, lengthscale


def _fit_and_summarize_batch(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit B spectra simultaneously with one batched exact GP.

    Every spectrum keeps its own hyperparameters and noise, so the result
    equals fitting them one by one; only the Python overhead is shared.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Mean curves (B, N), confidence half-width curves (B, N) and a (B, 3)
        summary of ``(noise_sigma, snr, lengthscale_ev)``.
    """
    batch_size = len(metas)
    labels = [f"spectrum meta={meta}" for meta in metas]
    prepared = [_validate_spectrum(e, i, l) for e, i, l in zip(energies, intensities, labels)]
    energies = np.stack([p[0] for p in prepared])
    intensities = np.stack([p[1] for p in prepared])
    spacing, energy_range, scales = (np.array([p[k] for p in prepared]) for k in (2, 3, 4))
    device = torch.device(cfg["device"])
    train_x = torch.tensor(
        energies - energies.mean(axis=1, keepdims=True), device=device
    ).unsqueeze(-1)
    scaled = intensities / scales[:, None]
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
    covar = _build_kernel(cfg["kernel"], spacing, energy_range, batch_size=batch_size)
    model = SpectrumExactGPModel(train_x, train_y, likelihood, covar, batch_shape)
    model.to(device).double()
    constant = model.mean_module.constant
    initial_mean = torch.tensor(scaled.mean(axis=1), device=device).reshape(constant.shape)
    constant.data.copy_(initial_mean)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for iteration in range(cfg["training"]["num_iterations"]):
        optimizer.zero_grad()
        try:
            loss = -mll(model(train_x), train_y)
        except Exception as exc:  # re-raised; caller locates the spectrum
            raise RuntimeError(
                f"Batched GP marginal likelihood failed for {batch_size} spectra "
                f"at iteration {iteration}: {exc}"
            ) from exc
        bad = ~torch.isfinite(loss)
        if bad.any():
            first = int(bad.nonzero()[0])
            raise RuntimeError(
                f"{labels[first]}: non-finite GP loss at iteration {iteration}."
            )
        loss.sum().backward()
        optimizer.step()

    model.eval()
    with torch.no_grad(), gpytorch.settings.debug(False):
        latent = model(train_x)
        mean = latent.mean.cpu().numpy()
        variance = latent.variance.clamp_min(0.0).cpu().numpy()
        noise_variance = likelihood.noise.detach().cpu().numpy().reshape(batch_size)
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
    if cfg["confidence_type"] == "observation":
        variance = variance + noise_variance[:, None] * weights
    mean_curves = mean * scales[:, None]
    confidence_curves = cfg["confidence_z"] * np.sqrt(variance) * scales[:, None]
    noise_sigma = np.sqrt(noise_variance * weights.mean(axis=1)) * scales
    for i in range(batch_size):
        if not (np.isfinite(mean_curves[i]).all() and np.isfinite(confidence_curves[i]).all()):
            raise RuntimeError(f"{labels[i]}: GP prediction produced non-finite values.")
        if not np.isfinite(noise_sigma[i]) or noise_variance[i] <= 1.5e-8:
            raise RuntimeError(
                f"{labels[i]}: GP fit gave an invalid noise level (variance="
                f"{noise_variance[i]:.3g}, sigma={noise_sigma[i]:.3g}); the noise "
                "hit its lower bound, so the spectrum may be noise-free or the "
                "kernel too flexible. Consider raising kernel 'min_lengthscale_points'."
            )
    if cfg["snr_definition"] == "range":
        strength = mean_curves.max(axis=1) - mean_curves.min(axis=1)
    else:
        strength = mean_curves.std(axis=1)
    summary = np.stack([noise_sigma, strength / noise_sigma, lengthscale.reshape(batch_size)], axis=1)
    return mean_curves, confidence_curves, summary


def _analyse_sequential(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyse spectra one by one with the single-spectrum building blocks."""
    signal = np.empty(energies.shape, dtype=np.float64)
    noise = np.empty(energies.shape, dtype=np.float64)
    summary = np.empty((len(metas), 3), dtype=np.float64)
    for i, meta in enumerate(metas):
        gp_model = fit_regional_spectrum_by_gp(energies[i], intensities[i], cfg, meta=meta)
        signal[i], noise[i] = create_mean_and_confidence_curves(gp_model, energies[i], cfg)
        summary[i] = _summarize_gp_fit(gp_model, signal[i], cfg["snr_definition"])
    return signal, noise, summary


def _analyse_chunk(
    energies: np.ndarray,
    intensities: np.ndarray,
    metas: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analyse one chunk of spectra (batched, or sequential if batch_size is 1)."""
    if cfg["batch_size"] == 1:
        return _analyse_sequential(energies, intensities, metas, cfg)
    try:
        result = _fit_and_summarize_batch(energies, intensities, metas, cfg)
    except RuntimeError as exc:
        # Fit individually: raises an error naming the failing spectrum.
        _analyse_sequential(energies, intensities, metas, cfg)
        raise RuntimeError(
            f"Batched GP fit failed for a chunk of {len(metas)} spectra although "
            f"each fits individually: {exc}"
        ) from exc
    rtol = cfg["batch_verify_rtol"]
    if rtol is not None:
        reference = _analyse_sequential(energies[:1], intensities[:1], metas[:1], cfg)[2]
        batched_sigma, reference_sigma = result[2][0, 0], reference[0, 0]
        if abs(batched_sigma - reference_sigma) > rtol * abs(reference_sigma):
            raise RuntimeError(
                f"Batched noise sigma {batched_sigma:.6g} differs from the sequential "
                f"{reference_sigma:.6g} by more than rtol={rtol} for spectrum "
                f"meta={metas[0]}."
            )
    return result


def _init_worker() -> None:
    """Limit torch to one thread in worker processes."""
    torch.set_num_threads(1)


def _select_spectra(df_meta: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Select the meta rows to analyse by split, die and region."""
    required = {"spectrum_index", "die", "region"}
    missing = required - set(df_meta.columns)
    if missing:
        raise ValueError(f"df_meta is missing required columns: {sorted(missing)}.")
    unpaired_split = cfg["unpaired_split"]
    if unpaired_split is not None and unpaired_split not in _VALID_SPLITS:
        raise ValueError(
            f"unpaired_split must be None or one of {_VALID_SPLITS}, "
            f"got {unpaired_split!r}."
        )
    if not df_meta.index.is_unique:
        raise ValueError("df_meta.index must be unique for noise analysis.")
    if not pd.api.types.is_integer_dtype(df_meta["spectrum_index"]):
        raise ValueError(
            f"df_meta['spectrum_index'] must be integer, got dtype "
            f"{df_meta['spectrum_index'].dtype}."
        )
    if df_meta["spectrum_index"].duplicated().any():
        duplicated = df_meta.loc[df_meta["spectrum_index"].duplicated(), "spectrum_index"]
        raise ValueError(f"Duplicate spectrum_index values in df_meta: {duplicated.unique()[:5]}.")
    if "split" in df_meta.columns:
        unpaired = df_meta["split"].isna() | (df_meta["split"] == "none")
        invalid = df_meta.loc[~unpaired & ~df_meta["split"].isin(_VALID_SPLITS), "split"]
        if not invalid.empty:
            raise ValueError(
                f"Invalid split labels {sorted(invalid.unique().astype(str))}; "
                f"valid labels: {_VALID_SPLITS} or missing (unpaired)."
            )
        split = df_meta["split"].astype(object).where(~unpaired, None)
    elif unpaired_split is None:
        raise ValueError(
            "df_meta has no 'split' column and unpaired_split is None, so no "
            "spectra can be selected. Run splitting first or set unpaired_split."
        )
    else:
        split = pd.Series([None] * len(df_meta), index=df_meta.index, dtype=object)
    if unpaired_split is not None:
        split = split.where(split.notna(), unpaired_split)
    splits = cfg["splits"] or list(_VALID_SPLITS)
    bad = [s for s in splits if s not in _VALID_SPLITS]
    if bad:
        raise ValueError(f"Invalid splits {bad}; valid splits: {_VALID_SPLITS}.")
    mask = split.isin(splits)
    selected = df_meta[mask]
    for column, key in (("die", "dies"), ("region", "regions")):
        if cfg[key] is None:
            continue
        absent = set(cfg[key]) - set(selected[column].unique())
        if absent:
            raise ValueError(
                f"{key.capitalize()} {sorted(absent)} not found in splits {splits}; "
                f"available {key}: {sorted(selected[column].unique())}."
            )
        selected = selected[selected[column].isin(cfg[key])]
    if selected.empty:
        raise ValueError(
            f"No spectra selected (splits={splits}, dies={cfg['dies']}, "
            f"regions={cfg['regions']}, unpaired_split={unpaired_split})."
        )
    return selected


def perform_spectral_noise_analysis(
    ary_intensity: np.ndarray,
    ary_energy: np.ndarray,
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate noise and SNR of regional spectra with Gaussian processes.

    Parameters
    ----------
    ary_intensity, ary_energy : np.ndarray
        Canonical 2D arrays, one regional spectrum per row.
    df_meta : pd.DataFrame
        Canonical meta data (needs ``spectrum_index``, ``die``, ``region`` and
        normally ``split``). It is not modified.
    config : dict[str, Any] | None
        See ``get_default_noise_analysis_config``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        ``ary_spec_analysis_signal`` (GP mean curves),
        ``ary_spec_analysis_noise`` (confidence interval half-width curves),
        both of shape (n_analysed, N_points) with identical row order, and a
        copy of ``df_meta`` with new columns ``noise_analysis_index`` (row in
        both arrays, <NA> if not analysed), ``gp_noise_sigma``, ``gp_snr`` and
        ``gp_lengthscale`` (eV).
    """
    cfg = _resolve_config(config)
    if ary_intensity.shape != ary_energy.shape or ary_intensity.ndim != 2:
        raise ValueError(
            f"ary_intensity {ary_intensity.shape} and ary_energy "
            f"{ary_energy.shape} must be 2D arrays of the same shape."
        )
    selected = _select_spectra(df_meta, cfg)
    spectrum_indices = selected["spectrum_index"].to_numpy()
    if spectrum_indices.min() < 0 or spectrum_indices.max() >= ary_intensity.shape[0]:
        raise ValueError(
            f"spectrum_index range [{spectrum_indices.min()}, "
            f"{spectrum_indices.max()}] is outside the arrays "
            f"({ary_intensity.shape[0]} rows)."
        )

    n_rows, n_points = len(selected), ary_intensity.shape[1]
    ary_signal = np.empty((n_rows, n_points), dtype=np.float64)
    ary_noise = np.empty((n_rows, n_points), dtype=np.float64)
    summary = np.empty((n_rows, 3), dtype=np.float64)
    metas = selected.to_dict("records")
    workers = cfg["num_workers"]
    if cfg["batch_size"] > 1:
        chunk_size = cfg["batch_size"]
    elif workers > 1:
        chunk_size = math.ceil(n_rows / (workers * 4))
    else:
        chunk_size = 1
    chunks = [(start, min(start + chunk_size, n_rows)) for start in range(0, n_rows, chunk_size)]

    def chunk_args(start: int, stop: int) -> tuple[Any, ...]:
        rows = spectrum_indices[start:stop]
        return ary_energy[rows], ary_intensity[rows], metas[start:stop], cfg

    with tqdm(total=n_rows, desc="GP noise analysis", disable=not cfg["show_progress"]) as bar:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                futures = {pool.submit(_analyse_chunk, *chunk_args(*c)): c for c in chunks}
                try:
                    for future in as_completed(futures):
                        start, stop = futures[future]
                        ary_signal[start:stop], ary_noise[start:stop], summary[start:stop] = future.result()
                        bar.update(stop - start)
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        else:
            for start, stop in chunks:
                result = _analyse_chunk(*chunk_args(start, stop))
                ary_signal[start:stop], ary_noise[start:stop], summary[start:stop] = result
                bar.update(stop - start)

    df_out = df_meta.copy()
    df_out["noise_analysis_index"] = pd.array([pd.NA] * len(df_out), dtype="Int64")
    for column in _NEW_META_COLUMNS[1:]:
        df_out[column] = np.nan
    df_out.loc[selected.index, "noise_analysis_index"] = np.arange(n_rows)
    for column, values in zip(_NEW_META_COLUMNS[1:], summary.T):
        df_out.loc[selected.index, column] = values
    return ary_signal, ary_noise, df_out
