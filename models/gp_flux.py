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


class SigmoidStepKernel(gpytorch.kernels.Kernel):
    """Sigmoid-based step jump covariance kernel for abrupt baseline shifts.

    k(t1, t2) = s(t1) * s(t2)
    where s(t) = sigmoid(steepness * (t - step_time))

    When wrapped by ScaleKernel, provides a learnable variance scale sigma_jump^2.
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
        if learn_step_time:
            self.register_parameter(
                name="raw_step_time",
                parameter=torch.nn.Parameter(torch.tensor([float(step_time)])),
            )
        else:
            self.register_buffer(
                "raw_step_time",
                torch.tensor([float(step_time)]),
            )
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
    k_type = t_cfg.get("type", "rbf").lower()
    t_len = float(t_cfg.get("initial_lengthscale", 5.0))
    if k_type == "matern52":
        base_t = gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[0])
    elif k_type == "matern32":
        base_t = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=[0])
    elif k_type == "matern12":
        base_t = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=[0])
    else:
        base_t = gpytorch.kernels.RBFKernel(active_dims=[0])
    base_t.lengthscale = t_len
    k_time_cont = gpytorch.kernels.ScaleKernel(base_t)

    # 2. Optional sigmoid abrupt jump kernel on time
    if t_jump_cfg.get("enabled", True):
        step_t = float(t_jump_cfg.get("initial_step_time_days", 10.0))
        steep = float(t_jump_cfg.get("initial_steepness", 2.0))
        learn_t = bool(t_jump_cfg.get("learn_step_time", True))
        step_kern = SigmoidStepKernel(
            step_time=step_t,
            steepness=steep,
            learn_step_time=learn_t,
            active_dims=[0],
        )
        k_jump = gpytorch.kernels.ScaleKernel(step_kern)
        k_time = k_time_cont + k_jump
    else:
        k_time = k_time_cont

    # 3. Categorical die kernel
    d_type = die_cfg.get("type", "index").lower()
    num_dies = int(die_cfg.get("num_dies", 9))
    rank = int(die_cfg.get("rank", 2))
    if d_type == "index":
        k_die = gpytorch.kernels.IndexKernel(
            num_tasks=num_dies, rank=rank, active_dims=[1]
        )
    else:
        d_len = float(die_cfg.get("initial_lengthscale", 1.0))
        base_d = gpytorch.kernels.RBFKernel(active_dims=[1])
        base_d.lengthscale = d_len
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
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_default_gp_flux_config() -> dict[str, Any]:
    """Return default configuration options for GP die flux prediction."""
    return {
        "reference_date": "2026-01-01",
        "unpaired_split": "train",
        "target_column": "die_flux",
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
        Dictionary containing tensors, metadata subsets, and normalization stats.
    """
    cfg = {**get_default_gp_flux_config(), **(config or {})}
    ref_date = pd.to_datetime(cfg.get("reference_date", "2026-01-01"))
    unpaired_split = str(cfg.get("unpaired_split", "train"))
    target_col = str(cfg.get("target_column", "die_flux"))

    # Filter to requested tool and deduplicate by measurement_id and die
    df_tool = df_meta[df_meta["tool"] == tool_name].copy()
    if df_tool.empty:
        raise ValueError(f"No records found in df_meta for tool '{tool_name}'.")

    if target_col not in df_tool.columns:
        raise KeyError(f"Target column '{target_col}' not found in df_meta.")

    die_df = df_tool.drop_duplicates(subset=["measurement_id", "die"]).copy()

    # Route unpaired measurements (split == 'none' or null) to user choice
    if "split" not in die_df.columns:
        die_df["split"] = unpaired_split
    else:
        mask_unpaired = die_df["split"].isna() | (die_df["split"] == "none")
        die_df.loc[mask_unpaired, "split"] = unpaired_split

    # Compute continuous absolute elapsed days from reference date
    die_df["time"] = pd.to_datetime(die_df["time"])
    die_df["time_days"] = (
        (die_df["time"] - ref_date).dt.total_seconds() / 86400.0
    ).astype(float)
    die_df["die_idx"] = die_df["die"].astype(int)

    # Standardize target flux strictly using training split statistics
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

    def _extract_tensors(subset: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if subset.empty:
            return torch.empty((0, 2), dtype=torch.float32), torch.empty(
                (0,), dtype=torch.float32
            )
        x_mat = subset[["time_days", "die_idx"]].to_numpy(dtype=np.float32)
        y_vec = subset["target_norm"].to_numpy(dtype=np.float32)
        return torch.tensor(x_mat), torch.tensor(y_vec)

    train_x, train_y = _extract_tensors(die_df[die_df["split"] == "train"])
    val_x, val_y = _extract_tensors(die_df[die_df["split"] == "val"])
    test_x, test_y = _extract_tensors(die_df[die_df["split"] == "test"])

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
        Training inputs of shape (N, 2), where column 0 is time and column 1 is die.
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
    kernel_cfg = cfg.get("kernel", {})
    num_iter = int(train_cfg.get("num_iterations", 150))
    lr = float(train_cfg.get("learning_rate", 0.05))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DieFluxExactGPModel(train_x, train_y, likelihood, kernel_config=kernel_cfg)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
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
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(eval_x))
        mean_norm = pred_dist.mean.cpu().numpy()
        lower_norm, upper_norm = pred_dist.confidence_region()
        lower_norm = lower_norm.cpu().numpy()
        upper_norm = upper_norm.cpu().numpy()

    y_mean = scaler.get("mean", 0.0)
    y_std = scaler.get("std", 1.0)

    pred_mean = mean_norm * y_std + y_mean
    pred_lower = lower_norm * y_std + y_mean
    pred_upper = upper_norm * y_std + y_mean

    return pred_mean, pred_lower, pred_upper


def evaluate_gp_die_flux_models(
    models_dict: dict[str, tuple[DieFluxExactGPModel, gpytorch.likelihoods.GaussianLikelihood]],
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
    records = []

    for tool_name, (model, likelihood) in models_dict.items():
        ds = datasets_dict[tool_name]
        die_df = ds["die_df"]
        scaler = ds["scaler"]
        target_col = ds["target_column"]

        # Predict across all records in die_df
        all_x = torch.tensor(
            die_df[["time_days", "die_idx"]].to_numpy(dtype=np.float32)
        )
        pred_mean, _, _ = predict_gp_die_flux(model, likelihood, all_x, scaler)
        die_df = die_df.copy()
        die_df["pred_flux"] = pred_mean
        die_df["abs_err"] = (die_df["pred_flux"] - die_df[target_col]).abs()
        safe_true = np.maximum(die_df[target_col].to_numpy(), 1e-8)
        die_df["ape"] = (die_df["abs_err"] / safe_true) * 100.0

        dies = sorted(die_df["die_idx"].unique())
        for d in dies:
            d_sub = die_df[die_df["die_idx"] == d]
            row: dict[str, Any] = {"tool": tool_name, "die": str(d)}
            for split_name in ["train", "val", "test"]:
                s_rows = d_sub[d_sub["split"] == split_name]
                if not s_rows.empty:
                    mae_val = float(s_rows["abs_err"].mean())
                    mape_val = float(s_rows["ape"].mean())
                    row[f"mae_{split_name}"] = round(mae_val, 2)
                    row[f"mape_{split_name}%"] = f"{mape_val:.2f}%"
                else:
                    row[f"mae_{split_name}"] = np.nan
                    row[f"mape_{split_name}%"] = "-"
            records.append(row)

        # Tool average row
        tool_row: dict[str, Any] = {"tool": f"{tool_name} (Mean)", "die": "All"}
        for split_name in ["train", "val", "test"]:
            s_rows = die_df[die_df["split"] == split_name]
            if not s_rows.empty:
                mae_val = float(s_rows["abs_err"].mean())
                mape_val = float(s_rows["ape"].mean())
                tool_row[f"mae_{split_name}"] = round(mae_val, 2)
                tool_row[f"mape_{split_name}%"] = f"{mape_val:.2f}%"
            else:
                tool_row[f"mae_{split_name}"] = np.nan
                tool_row[f"mape_{split_name}%"] = "-"
        records.append(tool_row)

    df_metrics = pd.DataFrame(records)
    return df_metrics


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
        Dictionary of trained model bundles and the performance metrics DataFrame.
    """
    cfg = {**get_default_gp_flux_config(), **(config or {})}

    # Identify source and target tools
    src_cfg = cfg.get("source_tool")
    tgt_cfg = cfg.get("target_tool")
    if src_cfg and tgt_cfg:
        tools_to_run = [str(src_cfg), str(tgt_cfg)]
    elif "tool_target" in df_meta.columns and df_meta["tool_target"].dropna().any():
        paired_rows = df_meta[df_meta["tool_target"].notna()]
        source_tool = str(paired_rows["tool"].iloc[0])
        target_tool = str(paired_rows["tool_target"].iloc[0])
        tools_to_run = [source_tool, target_tool]
    else:
        tools_to_run = list(df_meta["tool"].dropna().unique())[:2]

    models_dict = {}
    datasets_dict = {}

    for t in tools_to_run:
        ds = build_gp_flux_dataset(df_meta, t, cfg)
        datasets_dict[t] = ds

        # Print sample counts separately as requested
        n_tr = len(ds["train_x"])
        n_va = len(ds["val_x"])
        n_te = len(ds["test_x"])
        print(f"  [{t}] GP dataset sample counts: train={n_tr}, val={n_va}, test={n_te}")

        model, likelihood = train_gp_die_flux_model(ds["train_x"], ds["train_y"], cfg)
        models_dict[t] = (model, likelihood)

    df_metrics = evaluate_gp_die_flux_models(models_dict, datasets_dict)

    bundle = {
        "models": models_dict,
        "datasets": datasets_dict,
        "config": cfg,
    }
    return bundle, df_metrics
