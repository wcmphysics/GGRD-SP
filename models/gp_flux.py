"""Gaussian Process module for predicting regional and die spectral fluxes.

Provides exact GP modeling with configurable composite covariance kernels,
including continuous time kernels, learnable sigmoid abrupt-jump kernels,
and categorical die index kernels using GPyTorch.
"""

from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch

_MATERN_NU_MAP: dict[str, float] = {
    "matern52": 2.5,
    "matern32": 1.5,
    "matern12": 0.5,
}


class SigmoidStepKernel(gpytorch.kernels.Kernel):
    """Sigmoid-based step jump covariance kernel for abrupt baseline shifts.

    k(t1, t2) = s(t1) * s(t2)
    where s(t) = sigmoid(steepness * (t - step_time))

    When wrapped by ScaleKernel, provides a learnable variance scale
    sigma_jump^2.
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(
        self,
        step_time: float = 10.0,
        steepness: float = 2.0,
        learn_step_time: bool = True,
        active_dims: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(active_dims=active_dims, **kwargs)
        step_tensor = torch.tensor([float(step_time)])
        if learn_step_time:
            self.register_parameter(
                name="raw_step_time",
                parameter=torch.nn.Parameter(step_tensor),
            )
        else:
            self.register_buffer("raw_step_time", step_tensor)
        self.register_parameter(
            name="raw_steepness",
            parameter=torch.nn.Parameter(torch.tensor([float(steepness)])),
        )

    @property
    def step_time(self) -> torch.Tensor:
        """Return the transition center time in days."""
        return self.raw_step_time

    @property
    def steepness(self) -> torch.Tensor:
        """Return constrained positive transition steepness."""
        return torch.nn.functional.softplus(self.raw_steepness)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        """Compute sigmoid step covariance matrix between x1 and x2."""
        del params
        s1 = torch.sigmoid(self.steepness * (x1 - self.step_time))
        if diag:
            return (s1**2).squeeze(-1)
        s2 = torch.sigmoid(self.steepness * (x2 - self.step_time))
        return torch.matmul(s1, s2.transpose(-1, -2))


def build_composite_kernel(
    kernel_config: dict[str, Any] | None = None,
) -> gpytorch.kernels.Kernel:
    """Build a configurable composite GPyTorch covariance kernel.

    Parameters
    ----------
    kernel_config : dict[str, Any] | None, optional
        Configuration dictionary defining continuous time kernel, sigmoid jump,
        die kernel, and composition rule.

    Returns
    -------
    gpytorch.kernels.Kernel
        The constructed GPyTorch covariance kernel.
    """
    cfg = kernel_config or {}
    t_cfg = cfg.get("time_continuous", {})
    t_jump_cfg = cfg.get("time_jump", {})
    die_cfg = cfg.get("die", {})
    comp = cfg.get("composition", "product_with_jump")

    # 1. Continuous time kernel
    k_type = str(t_cfg.get("type", "rbf")).lower()
    if k_type in _MATERN_NU_MAP:
        base_t: gpytorch.kernels.Kernel = gpytorch.kernels.MaternKernel(
            nu=_MATERN_NU_MAP[k_type], active_dims=[0]
        )
    else:
        base_t = gpytorch.kernels.RBFKernel(active_dims=[0])
    base_t.lengthscale = float(t_cfg.get("initial_lengthscale", 5.0))
    k_time_cont = gpytorch.kernels.ScaleKernel(base_t)

    # 2. Optional sigmoid abrupt jump kernel on time
    if t_jump_cfg.get("enabled", True):
        step_kern = SigmoidStepKernel(
            step_time=float(t_jump_cfg.get("initial_step_time_days", 10.0)),
            steepness=float(t_jump_cfg.get("initial_steepness", 2.0)),
            learn_step_time=bool(t_jump_cfg.get("learn_step_time", True)),
            active_dims=[0],
        )
        k_time = k_time_cont + gpytorch.kernels.ScaleKernel(step_kern)
    else:
        k_time = k_time_cont

    # 3. Categorical die kernel
    if str(die_cfg.get("type", "index")).lower() == "index":
        k_die: gpytorch.kernels.Kernel = gpytorch.kernels.IndexKernel(
            num_tasks=int(die_cfg.get("num_dies", 9)),
            rank=int(die_cfg.get("rank", 2)),
            active_dims=[1],
        )
    else:
        base_d = gpytorch.kernels.RBFKernel(active_dims=[1])
        base_d.lengthscale = float(die_cfg.get("initial_lengthscale", 1.0))
        k_die = gpytorch.kernels.ScaleKernel(base_d)

    # 4. Composition
    if comp == "additive":
        return k_time + k_die
    if comp == "product":
        return k_time_cont * k_die
    return k_time * k_die


class DieFluxExactGPModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process Regression model for die total flux prediction."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = build_composite_kernel(kernel_config)

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior MultivariateNormal distribution at inputs x."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def get_default_gp_flux_config() -> dict[str, Any]:
    """Return default configuration options for GP die flux prediction."""
    return {
        "reference_date": "2026-01-01",
        "unpaired_split": "train",
        "target_column": "total_flux_single_die",
        "kernel": {
            "time_continuous": {
                "type": "rbf",
                "initial_lengthscale": 5.0,
            },
            "time_jump": {
                "enabled": True,
                "initial_step_time_days": 10.0,
                "initial_steepness": 2.0,
                "learn_step_time": True,
            },
            "die": {
                "type": "index",
                "num_dies": 9,
                "rank": 2,
            },
            "composition": "product_with_jump",
        },
        "training": {
            "num_iterations": 150,
            "learning_rate": 0.05,
        },
        "plot": {
            "plot_die": 0,
        },
    }


def build_gp_flux_dataset(
    df_meta: pd.DataFrame,
    tool_name: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and prepare training, validation, and test datasets for a tool.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing measurements, dies, time, and splits.
    tool_name : str
        Target tool name (e.g. 'J4' or 'J5') to filter.
    config : dict[str, Any] | None, optional
        Configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary containing tensors, metadata subsets, and normalization
        stats.
    """
    cfg = {**get_default_gp_flux_config(), **(config or {})}
    ref_date = pd.to_datetime(cfg.get("reference_date", "2026-01-01"))
    unpaired_split = str(cfg.get("unpaired_split", "train"))
    target_col = str(cfg.get("target_column", "total_flux_single_die"))

    df_tool = df_meta[df_meta["tool"] == tool_name]
    if df_tool.empty:
        raise ValueError(f"No records found in df_meta for tool '{tool_name}'.")
    if target_col not in df_tool.columns:
        raise KeyError(f"Target column '{target_col}' not found in df_meta.")

    die_df = df_tool.drop_duplicates(subset=["measurement_id", "die"]).copy()
    if "split" not in die_df.columns:
        die_df["split"] = unpaired_split
    else:
        mask_unpaired = die_df["split"].isna() | (die_df["split"] == "none")
        die_df.loc[mask_unpaired, "split"] = unpaired_split

    die_df["time"] = pd.to_datetime(die_df["time"])
    die_df["time_days"] = (
        (die_df["time"] - ref_date).dt.total_seconds() / 86400.0
    ).astype(float)
    die_df["die_idx"] = die_df["die"].astype(int)

    df_train = die_df[die_df["split"] == "train"]
    if df_train.empty:
        raise ValueError(
            f"No training records found for tool '{tool_name}' after splitting."
        )

    y_mean = float(df_train[target_col].mean())
    y_std = float(df_train[target_col].std(ddof=0))
    if y_std < 1e-8:
        y_std = 1.0
    die_df["target_norm"] = (die_df[target_col] - y_mean) / y_std

    def _extract_tensors(
        split_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        subset = die_df[die_df["split"] == split_name]
        if subset.empty:
            return (
                torch.empty((0, 2), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )
        return (
            torch.tensor(subset[["time_days", "die_idx"]].to_numpy(dtype=np.float32)),
            torch.tensor(subset["target_norm"].to_numpy(dtype=np.float32)),
        )

    train_x, train_y = _extract_tensors("train")
    val_x, val_y = _extract_tensors("val")
    test_x, test_y = _extract_tensors("test")

    return {
        "tool": tool_name,
        "die_df": die_df,
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "scaler": {"mean": y_mean, "std": y_std},
        "target_column": target_col,
    }


def train_gp_die_flux_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> tuple[DieFluxExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    """Train exact GP model via Adam and exact marginal log-likelihood.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs of shape (N, 2), where column 0 is time and column 1
        is die.
    train_y : torch.Tensor
        Normalized training targets of shape (N,).
    config : dict[str, Any] | None, optional
        Configuration dictionary with kernel and training hyperparameters.

    Returns
    -------
    tuple[DieFluxExactGPModel, gpytorch.likelihoods.GaussianLikelihood]
        Trained model and likelihood in evaluation mode.
    """
    cfg = config or {}
    train_cfg = cfg.get("training", {})
    num_iter = int(train_cfg.get("num_iterations", 150))
    lr = float(train_cfg.get("learning_rate", 0.05))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DieFluxExactGPModel(
        train_x, train_y, likelihood, kernel_config=cfg.get("kernel", {})
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(num_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def predict_gp_die_flux(
    model: DieFluxExactGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    eval_x: torch.Tensor,
    scaler: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior mean and 2-sigma confidence intervals in physical flux.

    Parameters
    ----------
    model : DieFluxExactGPModel
        Trained GP model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Trained Gaussian likelihood.
    eval_x : torch.Tensor
        Evaluation inputs of shape (N, 2).
    scaler : dict[str, float]
        Dictionary with 'mean' and 'std' for inverting normalization.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Posterior predictive mean, lower 2-sigma bound, and upper 2-sigma bound.
    """
    if eval_x.numel() == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
        pred_dist = likelihood(model(eval_x))
        mean_norm = pred_dist.mean.cpu().numpy()
        lower_t, upper_t = pred_dist.confidence_region()
        lower_norm, upper_norm = lower_t.cpu().numpy(), upper_t.cpu().numpy()

    y_mean = scaler.get("mean", 0.0)
    y_std = scaler.get("std", 1.0)
    return (
        mean_norm * y_std + y_mean,
        lower_norm * y_std + y_mean,
        upper_norm * y_std + y_mean,
    )


def _build_split_metrics_row(
    df_subset: pd.DataFrame, tool_label: str, die_label: str
) -> dict[str, Any]:
    """Compute MAE and MAPE metrics across train, val, and test splits."""
    row: dict[str, Any] = {"tool": tool_label, "die": die_label}
    for split_name in ("train", "val", "test"):
        s_rows = df_subset[df_subset["split"] == split_name]
        if s_rows.empty:
            row[f"mae_{split_name}"] = np.nan
            row[f"mape_{split_name}%"] = "-"
        else:
            row[f"mae_{split_name}"] = round(float(s_rows["abs_err"].mean()), 2)
            row[f"mape_{split_name}%"] = f"{float(s_rows['ape'].mean()):.2f}%"
    return row


def evaluate_gp_die_flux_models(
    models_dict: dict[
        str, tuple[DieFluxExactGPModel, gpytorch.likelihoods.GaussianLikelihood]
    ],
    datasets_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate MAE and MAPE across train, val, and test per die and tool.

    Parameters
    ----------
    models_dict : dict[str, tuple]
        Dictionary mapping tool names to (model, likelihood).
    datasets_dict : dict[str, dict[str, Any]]
        Dictionary mapping tool names to prepared dataset bundles.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: 'tool', 'die', 'mae_train', 'mape_train%',
        'mae_val', 'mape_val%', 'mae_test', 'mape_test%'.
    """
    records: list[dict[str, Any]] = []

    for tool_name, (model, likelihood) in models_dict.items():
        ds = datasets_dict[tool_name]
        die_df = ds["die_df"].copy()
        target_col = ds["target_column"]

        all_x = torch.tensor(
            die_df[["time_days", "die_idx"]].to_numpy(dtype=np.float32)
        )
        pred_mean, _, _ = predict_gp_die_flux(
            model, likelihood, all_x, ds["scaler"]
        )
        die_df["pred_flux"] = pred_mean
        die_df["abs_err"] = (die_df["pred_flux"] - die_df[target_col]).abs()
        safe_true = np.maximum(die_df[target_col].to_numpy(), 1e-8)
        die_df["ape"] = (die_df["abs_err"] / safe_true) * 100.0

        for die_id in sorted(die_df["die_idx"].unique()):
            records.append(
                _build_split_metrics_row(
                    die_df[die_df["die_idx"] == die_id], tool_name, str(die_id)
                )
            )
        records.append(
            _build_split_metrics_row(die_df, f"{tool_name} (Mean)", "All")
        )

    return pd.DataFrame(records)


def run_gp_flux_pipeline(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train GP models and compute evaluation summary for tools.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing measurements, tools, dies, and splits.
    config : dict[str, Any] | None, optional
        Configuration dictionary for GP flux prediction.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        Dictionary of trained model bundles and the performance metrics
        DataFrame.
    """
    cfg = {**get_default_gp_flux_config(), **(config or {})}

    src_cfg, tgt_cfg = cfg.get("source_tool"), cfg.get("target_tool")
    if src_cfg and tgt_cfg:
        tools_to_run = [str(src_cfg), str(tgt_cfg)]
    elif "tool_target" in df_meta.columns and df_meta["tool_target"].dropna().any():
        paired_rows = df_meta[df_meta["tool_target"].notna()]
        tools_to_run = [
            str(paired_rows["tool"].iloc[0]),
            str(paired_rows["tool_target"].iloc[0]),
        ]
    else:
        tools_to_run = list(df_meta["tool"].dropna().unique())[:2]

    models_dict = {}
    datasets_dict = {}

    for tool_name in tools_to_run:
        ds = build_gp_flux_dataset(df_meta, tool_name, cfg)
        datasets_dict[tool_name] = ds
        print(
            f"  [{tool_name}] GP dataset sample counts: "
            f"train={len(ds['train_x'])}, val={len(ds['val_x'])}, "
            f"test={len(ds['test_x'])}"
        )
        models_dict[tool_name] = train_gp_die_flux_model(
            ds["train_x"], ds["train_y"], cfg
        )

    df_metrics = evaluate_gp_die_flux_models(models_dict, datasets_dict)
    return {
        "models": models_dict,
        "datasets": datasets_dict,
        "config": cfg,
    }, df_metrics
