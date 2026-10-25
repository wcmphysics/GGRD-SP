"""Pydantic configuration schema and validation for GGRD-SP.

Provides strict static and runtime validation of pipeline configurations,
ensuring physical consistency, type safety, and domain constraints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


class MaterialSpec(BaseModel):
    """Specification of chemical elements in a specimen material."""

    elements: list[str] = Field(..., min_length=1)
    model_config = ConfigDict(extra="allow")


class GlobalConfig(BaseModel):
    """Global experiment and pipeline execution parameters."""

    material: str = "NMG"
    model_type: str = "resnet"
    use_sliding_window: bool = False
    use_film: bool = False
    apply_denoise: bool = False
    unified_model: bool = True
    source_tool: str = "J4"
    target_tool: str = "J5"
    example_region: str = "Ti2p"
    sample_measurement_id: str = "NMG_M_J4_00000"
    model_config = ConfigDict(extra="allow")


class ToolOffsetConfig(BaseModel):
    """Synthetic spectral drift and scale offsets per measurement tool."""

    shift_ev: float = 0.0
    scale: float = Field(10000.0, gt=0.0)
    model_config = ConfigDict(extra="allow")


class DataGenerationConfig(BaseModel):
    """Synthetic spectral data generation parameters."""

    regions: list[str] = Field(..., min_length=1)
    region_relative_strengths: dict[str, float] = Field(default_factory=dict)
    n_points: int = Field(100, gt=1)
    measurements_per_tool: dict[str, int] = Field(default_factory=dict)
    measurements_per_t7_code: int = Field(10, gt=0)
    interval_hours_range: tuple[float, float] | list[float] = (4.0, 24.0)
    die_variation_std: float = Field(0.03, ge=0.0)
    noise_relative_std: float = Field(0.015, ge=0.0)
    seed: int | None = None
    tool_offsets: dict[str, ToolOffsetConfig] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class PairingConfig(BaseModel):
    """Source-to-target temporal bipartite matching parameters."""

    time_threshold_hours: float = Field(8.0, gt=0.0)
    method: Literal["optimal", "greedy"] = "optimal"
    source_tool: str | None = None
    target_tool: str | None = None
    material: str | None = None
    model_config = ConfigDict(extra="allow")


class NormalizationConfig(BaseModel):
    """Physical spectral flux and intensity normalization options."""

    normalization_mode: Literal[
        "by_total_flux_single_die",
        "by_regional_max_intensity",
        "by_regional_max_intensity_of_source",
        "by_mean_intensity_single_die",
        "no_normalization",
    ] = "by_regional_max_intensity_of_source"
    eps: float = Field(1e-6, gt=0.0)
    model_config = ConfigDict(extra="allow")


class SplittingConfig(BaseModel):
    """Session-level train-val-test partitioning parameters."""

    method: Literal["random", "direct"] = "random"
    train_ratio: float = Field(0.5, ge=0.0, le=1.0)
    val_ratio: float = Field(0.2, ge=0.0, le=1.0)
    test_ratio: float = Field(0.3, ge=0.0, le=1.0)
    seed: int | None = None
    source_tool: str | None = None
    target_tool: str | None = None
    assignment: dict[str, list[str]] | None = None
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_ratios_sum_to_one(self) -> SplittingConfig:
        """Verify partition proportions sum to 1.0 for random splitting."""
        if self.method == "random":
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"Partition ratios must sum to 1.0, got {total:.4f} "
                    f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})."
                )
        return self


class GPFluxTimeContinuous(BaseModel):
    """Continuous temporal covariance component."""

    type: str = "rbf"
    initial_lengthscale: float = Field(5.0, gt=0.0)
    model_config = ConfigDict(extra="allow")


class GPFluxTimeJump(BaseModel):
    """Learnable sigmoid maintenance baseline shift kernel."""

    enabled: bool = True
    initial_step_time_days: float = 10.0
    initial_steepness: float = Field(2.0, gt=0.0)
    learn_step_time: bool = True
    model_config = ConfigDict(extra="allow")


class GPFluxDie(BaseModel):
    """Die spatial index covariance component."""

    type: str = "index"
    num_dies: int = Field(9, gt=0)
    rank: int = Field(2, gt=0)
    model_config = ConfigDict(extra="allow")


class GPFluxKernel(BaseModel):
    """Composite covariance kernel specification for die flux."""

    time_continuous: GPFluxTimeContinuous = Field(default_factory=GPFluxTimeContinuous)
    time_jump: GPFluxTimeJump = Field(default_factory=GPFluxTimeJump)
    die: GPFluxDie = Field(default_factory=GPFluxDie)
    composition: str = "product_with_jump"
    model_config = ConfigDict(extra="allow")


class GPFluxTraining(BaseModel):
    """Optimization hyperparameters for exact GP flux regression."""

    num_iterations: int = Field(150, gt=0)
    learning_rate: float = Field(0.05, gt=0.0)
    model_config = ConfigDict(extra="allow")


class GPFluxPlot(BaseModel):
    """Diagnostic visualization options for die flux drift."""

    plot_die: int = Field(0, ge=0)
    model_config = ConfigDict(extra="allow")


class GPFluxConfig(BaseModel):
    """Gaussian Process die flux drift and aging modeling parameters."""

    reference_date: str = "2026-01-01"
    unpaired_split: str = "train"
    target_column: str = "total_flux_single_die"
    kernel: GPFluxKernel = Field(default_factory=GPFluxKernel)
    training: GPFluxTraining = Field(default_factory=GPFluxTraining)
    plot: GPFluxPlot = Field(default_factory=GPFluxPlot)
    model_config = ConfigDict(extra="allow")


class DatasetConfig(BaseModel):
    """Dataset assembly and sliding window patch slicing options."""

    window_size_ev: float = Field(3.0, gt=0.0)
    sliding_stride_ev: float = Field(1.0, gt=0.0)
    region_configs: dict[str, Any] = Field(default_factory=dict)
    source_tool: str | None = None
    target_tool: str | None = None
    use_sliding_window: bool | None = None
    regions: list[str] | None = None
    model_config = ConfigDict(extra="allow")


class CostConfig(BaseModel):
    """Multi-component spectral transfer loss function weights."""

    w_mse: float = Field(1.0, ge=0.0)
    w_shape: float = Field(0.0, ge=0.0)
    w_gradient: float = Field(0.0, ge=0.0)
    l2_weight: float = Field(1e-6, ge=0.0)
    normalized_by_target_max: bool = True
    model_config = ConfigDict(extra="allow")


class NoiseKernelConfig(BaseModel):
    """Kernel covariance options for spectral noise decomposition."""

    type: str = "matern52"
    initial_lengthscale_ev: float = Field(1.0, gt=0.0)
    model_config = ConfigDict(extra="allow")


class NoiseTrainingConfig(BaseModel):
    """Training iterations for Gaussian Process noise estimation."""

    num_iterations: int = Field(50, gt=0)
    learning_rate: float = Field(0.2, gt=0.0)
    model_config = ConfigDict(extra="allow")


class NoiseAnalysisConfig(BaseModel):
    """Spectral noise estimation and SNR quantification configuration."""

    method: Literal["gp", "savgol", "wavelet", "fourier"] = "gp"
    splits: list[str] | None = None
    dies: list[int] | None = None
    batch_size: int = Field(64, gt=0)
    unpaired_split: str = "train"
    kernel: NoiseKernelConfig = Field(default_factory=NoiseKernelConfig)
    noise_model: str = "homoscedastic"
    confidence_type: str = "observation"
    training: NoiseTrainingConfig = Field(default_factory=NoiseTrainingConfig)
    model_config = ConfigDict(extra="allow")


class DenoiseSavgolConfig(BaseModel):
    """Savitzky-Golay polynomial filtering hyperparameters."""

    window_length: int = Field(11, gt=2)
    polynomial_order: int = Field(3, ge=1)
    boundary_mode: str = "interp"
    model_config = ConfigDict(extra="allow")

    @field_validator("window_length")
    @classmethod
    def validate_window_length_odd(cls, val: int) -> int:
        """Ensure window_length is a positive odd integer."""
        if val % 2 == 0:
            raise ValueError(f"window_length must be an odd integer, got {val}")
        return val


class DenoiseWaveletConfig(BaseModel):
    """Wavelet shrinkage decomposition hyperparameters."""

    transform_type: Literal["stationary", "discrete"] = "stationary"
    wavelet_name: str = "sym4"
    decomposition_level: int = Field(2, ge=1)
    threshold_mode: Literal["soft", "hard", "garrote"] = "soft"
    threshold_rule: Literal["universal", "sure", "bayes"] = "universal"
    threshold_scale: float = Field(1.0, gt=0.0)
    model_config = ConfigDict(extra="allow")


class DenoiseFourierConfig(BaseModel):
    """Frequency-domain filtering hyperparameters."""

    filter_type: Literal["dct", "butterworth"] = "dct"
    cutoff_ratio: float = Field(0.25, gt=0.0, le=1.0)
    filter_order: int = Field(4, ge=1)
    rolloff_type: str = "butterworth"
    padding_mode: str = "even"
    model_config = ConfigDict(extra="allow")


class DenoiseGPConfig(BaseModel):
    """Gaussian process filtering hyperparameters."""

    kernel: NoiseKernelConfig = Field(default_factory=NoiseKernelConfig)
    noise_model: str = "homoscedastic"
    confidence_type: str = "observation"
    training: NoiseTrainingConfig = Field(default_factory=NoiseTrainingConfig)
    model_config = ConfigDict(extra="allow")


class DenoiseConfig(BaseModel):
    """Regional de-noising execution configuration."""

    method: Literal["savgol", "wavelet", "fourier", "gp"] = "savgol"
    regions: list[str] = Field(default_factory=lambda: ["Cl2p", "Ti2p"])
    apply_to: Literal["both", "source", "target"] = "both"
    savgol: DenoiseSavgolConfig = Field(default_factory=DenoiseSavgolConfig)
    wavelet: DenoiseWaveletConfig = Field(default_factory=DenoiseWaveletConfig)
    fourier: DenoiseFourierConfig = Field(default_factory=DenoiseFourierConfig)
    gp: DenoiseGPConfig = Field(default_factory=DenoiseGPConfig)
    region_configs: dict[str, Any] = Field(default_factory=dict)
    rsf_table: dict[str, float] | None = None
    model_config = ConfigDict(extra="allow")


class NoisePlotConfig(BaseModel):
    """Diagnostic noise visualization configuration."""

    plots: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="allow")


class FiLMConfig(BaseModel):
    """Feature-wise Linear Modulation conditioning parameters."""

    encode_die: bool = True
    encode_tool: bool = True
    encode_flux: bool = True
    encode_time: bool = True
    encode_energy: bool = True
    encode_region: bool = True
    encode_relative_intensity: bool = True
    encode_elements: bool = True
    region_embed_dim: int = Field(4, gt=0)
    energy_scale_ev: float = Field(1000.0, gt=0.0)
    die_embed_dim: int = Field(4, gt=0)
    tool_embed_dim: int = Field(4, gt=0)
    hidden_dim: int = Field(32, gt=0)
    relative_learning_rate: float = Field(0.1, gt=0.0)
    material_elements: dict[str, list[str]] | None = None
    element_vocab: list[str] | None = None
    model_config = ConfigDict(extra="allow")


class BayesianOptConfig(BaseModel):
    """Ax Bayesian optimization search space and trial budget."""

    use_bayesian_opt: bool = False
    num_trials: int = Field(5, gt=0)
    epochs_per_trial: int = Field(15, gt=0)
    verbose: bool = True
    learning_rate_bounds: list[float] = Field(
        default_factory=lambda: [0.0001, 0.01]
    )
    l2_bounds: list[float] = Field(default_factory=lambda: [0.000001, 0.01])
    batch_sizes: list[int] = Field(default_factory=lambda: [16, 32, 64])
    window_size_choices: list[int] = Field(default_factory=lambda: [15])
    kernel_sizes: list[int] = Field(default_factory=lambda: [3, 5, 7])
    resnet_channels: list[int] = Field(default_factory=lambda: [16, 32, 64])
    residual_unet_channels: list[int] = Field(default_factory=lambda: [16, 32])
    unet_channels: list[int] = Field(default_factory=lambda: [16, 32])
    deeplabv3_backbone_channels: list[int] = Field(
        default_factory=lambda: [16, 32, 64]
    )
    deeplabv3_aspp_channels: list[int] = Field(
        default_factory=lambda: [16, 32, 64]
    )
    depths: list[int] = Field(default_factory=lambda: [2, 3])
    model_config = ConfigDict(extra="allow")


class TrainConfig(BaseModel):
    """Deep neural network training hyperparameters."""

    epochs: int = Field(100, gt=0)
    batch_size: int = Field(16, gt=0)
    learning_rate: float = Field(0.0001, gt=0.0)
    depth: int = Field(3, gt=0)
    kernel_size: int = Field(9, gt=0)
    use_learning_rate_scheduler: bool = True
    learning_rate_reduce_factor: float = Field(0.9, gt=0.0, lt=1.0)
    learning_rate_scheduler_patience: int = Field(10, ge=1)
    early_stopping_patience: int = Field(20, ge=1)
    verbose: bool = True
    resnet_channel: int = Field(32, gt=0)
    residual_unet_channel: int = Field(16, gt=0)
    unet_channel: int = Field(16, gt=0)
    deeplabv3_backbone_channel: int = Field(32, gt=0)
    deeplabv3_aspp_channel: int = Field(32, gt=0)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    use_batch_norm: bool = True
    padding_mode: str = "reflect"
    use_film: bool = False
    energy_scale_ev: float = Field(1000.0, gt=0.0)
    model_config = ConfigDict(extra="allow")

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel_size_odd(cls, val: int) -> int:
        """Ensure convolutional kernel size is a positive odd integer."""
        if val % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer, got {val}")
        return val


class TrainingSectionConfig(BaseModel):
    """Training stage configuration section."""

    train_config: TrainConfig = Field(default_factory=TrainConfig)
    region_configs: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class PredictionConfig(BaseModel):
    """Post-training spectral inference configuration."""

    clamp_non_negative: bool = True
    model_config = ConfigDict(extra="allow")


class MetricsReportConfig(BaseModel):
    """Quantitative performance reporting configuration."""

    splits: list[str] = Field(default_factory=lambda: ["train", "test"])
    metrics: list[str] = Field(
        default_factory=lambda: [
            "peak_error_percentage",
            "max_error_percentage",
        ]
    )
    model_config = ConfigDict(extra="allow")


class QuantificationConfig(BaseModel):
    """Shirley background subtraction and atomic percentage calculation."""

    rsf_table: dict[str, float] = Field(..., min_length=1)
    default_endpoint_strategy: str = "minima"
    default_endpoint_average_ev: float = Field(1.0, gt=0.0)
    max_iter: int = Field(50, gt=0)
    tol: float = Field(1e-5, gt=0.0)
    model_config = ConfigDict(extra="allow")


class VisualizationConfig(BaseModel):
    """Diagnostic plotting switches and global display options."""

    show_plots: bool = True
    plots: dict[str, bool] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class PipelineConfig(BaseModel):
    """Root configuration schema for the GGRD-SP spectral transfer pipeline."""

    materials: dict[str, MaterialSpec] = Field(..., min_length=1)
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    data_generation: DataGenerationConfig
    pairing: PairingConfig = Field(default_factory=PairingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    gp_flux: GPFluxConfig = Field(default_factory=GPFluxConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    noise_analysis: NoiseAnalysisConfig = Field(default_factory=NoiseAnalysisConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    noise_plot: NoisePlotConfig = Field(default_factory=NoisePlotConfig)
    film: FiLMConfig = Field(default_factory=FiLMConfig)
    bayesian_opt: BayesianOptConfig = Field(default_factory=BayesianOptConfig)
    training: TrainingSectionConfig = Field(default_factory=TrainingSectionConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    metrics_report: MetricsReportConfig = Field(default_factory=MetricsReportConfig)
    quantification: QuantificationConfig
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> PipelineConfig:
        """Load and validate pipeline configuration directly from YAML."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Convert validated schema back into a native Python dictionary.

        Uses by_alias=True to restore 'global' as the dictionary key and
        exclude_none=True so unresolved optional keys fallback cleanly.
        """
        return self.model_dump(by_alias=True, exclude_none=True)

    @classmethod
    def export_json_schema(
        cls, output_path: str | Path = "config/config.schema.json"
    ) -> Path:
        """Export the Pydantic model JSON Schema for editor autocomplete."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema_dict = cls.model_json_schema()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)
        return path


__all__ = [
    "BayesianOptConfig",
    "CostConfig",
    "DataGenerationConfig",
    "DenoiseConfig",
    "DatasetConfig",
    "FiLMConfig",
    "GlobalConfig",
    "GPFluxConfig",
    "MaterialSpec",
    "MetricsReportConfig",
    "NoiseAnalysisConfig",
    "NormalizationConfig",
    "PairingConfig",
    "PipelineConfig",
    "PredictionConfig",
    "QuantificationConfig",
    "SplittingConfig",
    "TrainingSectionConfig",
    "VisualizationConfig",
]
