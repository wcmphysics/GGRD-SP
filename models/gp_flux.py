"""Gaussian Process module for predicting regional and die spectral fluxes.

Provides exact GP modeling with configurable composite covariance kernels,
including continuous time kernels, learnable sigmoid abrupt-jump kernels,
and categorical die index kernels using GPyTorch.
"""

from typing import Any
import warnings

import gpytorch
import numpy as np
import pandas as pd
import torch

if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    torch.sparse.check_sparse_tensor_invariants.enable()

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
        s2 = torch.sigmoid(self.steepness * (x2 - self.step_time))
        if diag:
            return (s1 * s2).squeeze(-1)
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
    t_configs = cfg.get("time_continuous", [{"type": "rbf", "initial_lengthscale": 5.0}])
    jump_configs = cfg.get("time_jump", [{}])
    die_cfg = cfg.get("die", {})
    comp = cfg.get("composition", "product_with_jump")

    # 1. Continuous time kernel
    continuous = []
    for t_cfg in t_configs:
        k_type = str(t_cfg.get("type", "rbf")).lower()
        if k_type in _MATERN_NU_MAP:
            base_t = gpytorch.kernels.MaternKernel(nu=_MATERN_NU_MAP[k_type], active_dims=[0])
        else:
            base_t = gpytorch.kernels.RBFKernel(active_dims=[0])
        base_t.lengthscale = float(t_cfg.get("initial_lengthscale", 5.0))
        continuous.append(gpytorch.kernels.ScaleKernel(base_t))
    k_time_cont = gpytorch.kernels.AdditiveKernel(*continuous)

    # 2. Optional sigmoid abrupt jump kernel on time
    jumps = []
    for t_jump_cfg in jump_configs:
        step_kern = SigmoidStepKernel(
            step_time=float(t_jump_cfg.get("initial_step_time_days", 10.0)),
            steepness=float(t_jump_cfg.get("initial_steepness", 2.0)),
            learn_step_time=bool(t_jump_cfg.get("learn_step_time", True)),
            active_dims=[0],
        )
        jumps.append(gpytorch.kernels.ScaleKernel(step_kern))
    k_time = gpytorch.kernels.AdditiveKernel(*continuous, *jumps)

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
    from utility.config_schema import GPFluxConfig, resolve_subconfig

    return resolve_subconfig(GPFluxConfig)


def build_gp_flux_dataset(
    df_meta: pd.DataFrame,
    tool_name: str,
    config: dict[str, Any] | None = None,
    *,
    material: str | None = None,
    recipe: str | None = None,
) -> dict[str, Any]:
    """Extract and prepare training, validation, and test datasets for a group.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing measurements, dies, time, and splits.
    tool_name : str
        Target tool name (e.g. 'J4' or 'J5') to filter.
    config : dict[str, Any] | None, optional
        Configuration dictionary.
    material : str | None, optional
        Optional material name filter for per-(tool, material, recipe) GPs.
    recipe : str | None, optional
        Optional recipe name filter for per-(tool, material, recipe) GPs.

    Returns
    -------
    dict[str, Any]
        Dictionary containing tensors, metadata subsets, and normalization
        stats.
    """
    from utility.config_schema import GPFluxConfig, resolve_subconfig

    cfg = resolve_subconfig(GPFluxConfig, config)
    ref_date = pd.to_datetime(cfg.get("reference_date", "2026-01-01"))
    unpaired_split = str(cfg.get("unpaired_split", "train"))
    target_col = str(cfg.get("target_column", "total_flux_single_die"))

    mask = df_meta["tool"] == tool_name
    if material is not None and "material" in df_meta.columns:
        mask = mask & (df_meta["material"] == material)
    if recipe is not None and "recipe" in df_meta.columns:
        mask = mask & (df_meta["recipe"] == recipe)

    df_tool = df_meta[mask]
    group_desc = (
        f"tool '{tool_name}'"
        + (f", material '{material}'" if material is not None else "")
        + (f", recipe '{recipe}'" if recipe is not None else "")
    )
    if df_tool.empty:
        raise ValueError(f"No records found in df_meta for {group_desc}.")
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
            f"No training records found for {group_desc} after splitting."
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
        "material": material,
        "recipe": recipe,
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
    df_subset: pd.DataFrame,
    tool_label: str,
    die_label: str,
    *,
    material: str | None = None,
    recipe: str | None = None,
    include_group_cols: bool = False,
) -> dict[str, Any]:
    """Compute MAE and MAPE metrics across train, val, and test splits."""
    row: dict[str, Any] = {"tool": tool_label}
    if include_group_cols:
        row["material"] = material if material is not None else "-"
        row["recipe"] = recipe if recipe is not None else "-"
    row["die"] = die_label
    for split_name in ("train", "val", "test"):
        s_rows = df_subset[df_subset["split"] == split_name]
        if s_rows.empty:
            row[f"mae_{split_name}"] = np.nan
            row[f"mape_{split_name}_percentage"] = "-"
        else:
            row[f"mae_{split_name}"] = round(float(s_rows["abs_err"].mean()), 2)
            row[f"mape_{split_name}_percentage"] = f"{float(s_rows['ape'].mean()):.2f}%"
    return row


def _unpack_gp_key(
    key: Any,
) -> tuple[str, str | None, str | None]:
    """Unpack either a tool string key or a (tool, material, recipe) tuple."""
    if isinstance(key, tuple) and len(key) == 3:
        return str(key[0]), key[1], key[2]
    return str(key), None, None


def _format_gp_group_label(
    tool_name: str, mat_val: str | None, rcp_val: str | None
) -> str:
    """Format concise human-readable label for a GP die-flux group."""
    parts = [str(p) for p in (mat_val, rcp_val) if p is not None]
    return f"{tool_name} | {'-'.join(parts)}" if parts else tool_name


def _discover_material_recipe_groups(
    df_meta: pd.DataFrame,
) -> list[tuple[str | None, str | None]]:
    """Discover ordered (material, recipe) groups present in df_meta."""
    has_recipe = "recipe" in df_meta.columns and df_meta["recipe"].notna().any()
    has_mat = "material" in df_meta.columns and df_meta["material"].notna().any()
    if has_recipe:
        cols = ["material", "recipe"] if has_mat else ["recipe"]
        dedup = df_meta[cols].dropna().drop_duplicates()
        return [
            (str(r.material) if has_mat else None, str(r.recipe))
            for r in dedup.itertuples()
        ]
    if has_mat and df_meta["material"].dropna().nunique() > 1:
        return [(str(m), None) for m in df_meta["material"].dropna().unique()]
    return [(None, None)]


def _resolve_gp_tools(df_meta: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Determine tool identifiers to train GP die-flux models for."""
    src_cfg, tgt_cfg = cfg.get("source_tool"), cfg.get("target_tool")
    if src_cfg and tgt_cfg:
        return [str(src_cfg), str(tgt_cfg)]
    if "target_tool" in df_meta.columns and df_meta["target_tool"].dropna().any():
        paired_rows = df_meta[df_meta["target_tool"].notna()]
        return [
            str(paired_rows["tool"].iloc[0]),
            str(paired_rows["target_tool"].iloc[0]),
        ]
    return list(df_meta["tool"].dropna().unique())[:2]


def evaluate_gp_die_flux_models(
    models_dict: dict[
        Any, tuple[DieFluxExactGPModel, gpytorch.likelihoods.GaussianLikelihood]
    ],
    datasets_dict: dict[Any, dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate MAE and MAPE across train, val, and test per die and group.

    Parameters
    ----------
    models_dict : dict[Any, tuple]
        Dictionary mapping tool or (tool, material, recipe) keys to
        (model, likelihood).
    datasets_dict : dict[Any, dict[str, Any]]
        Dictionary mapping matching keys to prepared dataset bundles.

    Returns
    -------
    pd.DataFrame
        Summary table with MAE and MAPE columns across train, val, and test.
    """
    records: list[dict[str, Any]] = []
    include_group_cols = any(isinstance(k, tuple) for k in models_dict)

    for key, (model, likelihood) in models_dict.items():
        tool_name, mat_val, rcp_val = _unpack_gp_key(key)
        ds = datasets_dict[key]
        die_df = ds["die_df"].copy()
        target_col = ds["target_column"]

        all_x = torch.tensor(
            die_df[["time_days", "die_idx"]].to_numpy(dtype=np.float32)
        )
        pred_mean, _, _ = predict_gp_die_flux(
            model, likelihood, all_x, ds["scaler"]
        )
        die_df["predicted_flux"] = pred_mean
        die_df["abs_err"] = (die_df["predicted_flux"] - die_df[target_col]).abs()
        safe_true = np.maximum(die_df[target_col].to_numpy(), 1e-8)
        die_df["ape"] = (die_df["abs_err"] / safe_true) * 100.0

        for die_id in sorted(die_df["die_idx"].unique()):
            records.append(
                _build_split_metrics_row(
                    die_df[die_df["die_idx"] == die_id],
                    tool_name,
                    str(die_id),
                    material=mat_val,
                    recipe=rcp_val,
                    include_group_cols=include_group_cols,
                )
            )
        records.append(
            _build_split_metrics_row(
                die_df,
                f"{tool_name} (Mean)",
                "All",
                material=mat_val,
                recipe=rcp_val,
                include_group_cols=include_group_cols,
            )
        )

    return pd.DataFrame(records)


def run_gp_flux_pipeline(
    df_meta: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train GP models per (tool, material, recipe) group and evaluate metrics.

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
    from utility.config_schema import GPFluxConfig, resolve_subconfig

    cfg = resolve_subconfig(GPFluxConfig, config)
    tools_to_run = _resolve_gp_tools(df_meta, cfg)
    groups = _discover_material_recipe_groups(df_meta)
    use_tuple_keys = any(m is not None or r is not None for m, r in groups)

    models_dict: dict[Any, Any] = {}
    datasets_dict: dict[Any, dict[str, Any]] = {}

    for mat_val, rcp_val in groups:
        for tool_name in tools_to_run:
            key: Any = (
                (tool_name, mat_val, rcp_val) if use_tuple_keys else tool_name
            )
            try:
                ds = build_gp_flux_dataset(
                    df_meta,
                    tool_name,
                    cfg,
                    material=mat_val,
                    recipe=rcp_val,
                )
            except ValueError as err:
                if use_tuple_keys:
                    warnings.warn(
                        f"Skipping GP die-flux model for {key}: {err}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                raise
            datasets_dict[key] = ds
            label = _format_gp_group_label(tool_name, mat_val, rcp_val)
            print(
                f"  [{label}] GP dataset sample counts: "
                f"train={len(ds['train_x'])}, val={len(ds['val_x'])}, "
                f"test={len(ds['test_x'])}"
            )
            models_dict[key] = train_gp_die_flux_model(
                ds["train_x"], ds["train_y"], cfg
            )

    if not models_dict:
        raise ValueError(
            "No GP die-flux models could be trained for any "
            "(tool, material, recipe) group."
        )

    df_metrics = evaluate_gp_die_flux_models(models_dict, datasets_dict)
    return {
        "models": models_dict,
        "datasets": datasets_dict,
        "config": cfg,
    }, df_metrics
