# Multi-Field-of-View 1D Models (`inception_time`, `tcn`, `fpn`) & Per-Model Config Files — Implementation Plan

## 1. Objective & Scope

Cross-tool XPS spectra exhibit region-dependent and chemical-state-dependent peak widths (FWHM) alongside broad inelastic backgrounds. To capture narrow core-level spin-orbit doublets, medium-width satellite peaks, and wide background slopes simultaneously without relying on sliding-window patches, this plan introduces **three multi-field-of-view (multi-FOV) 1D architectures**:

1. **`inception_time`** (`InceptionTime1D` in `models/inception.py`): Multi-branch parallel 1D convolutions with heterogeneous odd kernel sizes (`[3, 7, 15, 31]`) plus a parallel max-pooling branch inside residual Inception modules. `kernel_sizes` is fully configurable.
2. **`tcn`** (`TCN1D` in `models/tcn.py`): Dilated Temporal Convolutional Network with exponentially increasing dilation factors (`[1, 2, 4, 8, 16]`) across stacked residual blocks. Crucially, because XPS binding energy is a static bidirectional physical axis where peaks must not be shifted by causal padding, `TCN1D` uses **non-causal symmetric reflection padding**; `kernel_size` is fully configurable.
3. **`fpn`** (`FPN1D` in `models/fpn.py`): 1D Feature Pyramid Network combining a bottom-up hierarchical encoder, top-down lateral pathway, and multi-level feature fusion head that aggregates predictions from all pyramid scales simultaneously.

In addition:
- **Dedicated Per-Model Config Files (Full Migration - Option B)**: Every model has its own standalone YAML file under `config/`:
  - `config/inception_time.yaml`
  - `config/tcn.yaml`
  - `config/fpn.yaml`
  - `config/resnet.yaml` (also covers `resnet_scaled_short`)
  - `config/unet.yaml` (also covers `residual_unet`)
  - `config/deeplabv3.yaml`
  Each model config contains its own dedicated `train_config`, `region_configs`, `bayesian_opt`, and `film` sections.
- **Model Hyperparameter Defaults**: For `inception_time.yaml`, `tcn.yaml`, and `fpn.yaml`, default settings are `epochs: 50`, `learning_rate: 0.001`, and `early_stopping_patience: 20`.
- **FiLM Setting Migration**: FiLM configuration belongs inside each model-dedicated config so each model has its own FiLM setting, and FiLM conditioning is supported across all models including `inception_time`, `tcn`, and `fpn`.
- **Bayesian Optimization Migration**: Bayesian optimization configuration belongs inside each model-dedicated config so each model has its own optimization scheme and parameter search bounds.
- **Strict Multi-Layer Sliding-Window Prohibition**: Configuring `use_sliding_window: true` with `model_type` set to `'inception_time'`, `'tcn'`, or `'fpn'` raises an explicit, descriptive `ValueError` at four levels:
  1. Config schema validation (`utility/config_schema.py`) and config loader resolution (`utility/config_loader.py`).
  2. Orchestration (`models/orchestration.py`).
  3. Trainer (`models/trainer.py`) and Bayesian optimization (`models/bayesian_opt.py`).
  4. Model constructors (`InceptionTime1D`, `TCN1D`, `FPN1D`).

---

## 2. Architecture Specifications

All three models adhere to the repository unified model interface in `models/factory.py`:
- `__init__(self, n_points: int, config: dict[str, Any] | None = None) -> None`
- `forward(self, x: torch.Tensor, cond: dict[str, torch.Tensor] | torch.Tensor | None = None) -> torch.Tensor` mapping `(batch_size, 1, n_points)` -> `(batch_size, 1, n_points)`
- `get_l2_regularization(self) -> torch.Tensor` via `compute_conv1d_l2_reg(self)`
- Global residual learning `y = x + delta(x)` when `residual: true` (default `True`), learning the systematic inter-tool transfer delta.

### 2.1 InceptionTime1D (`models/inception.py`)
- **Bottleneck**: 1x1 Conv1d when `in_channels > 1` and `use_bottleneck=True`.
- **Parallel Multi-Scale Conv Branches**: One convolution per kernel size in `kernel_sizes` (default `[3, 7, 15, 31]`, fully configurable), each outputting `inception_channel` channels with `SafeReflectionPad1d((k - 1) // 2)` preserving sequence length.
- **Parallel Max-Pool Branch**: `SafeReflectionPad1d(1)` -> `nn.MaxPool1d(kernel_size=3, stride=1)` -> `nn.Conv1d(in_channels, inception_channel, kernel_size=1, bias=False)`.
- **Residual Blocks**: Stacks `num_blocks` blocks (default 3), each containing `modules_per_block` (default 2) Inception modules with a 1x1 + `SafeBatchNorm1d` shortcut.
- **FiLM Modulation**: Feature-wise linear modulation modulating the final Inception representation before the 1x1 projection head.
- **Output Head**: 1x1 Conv1d projecting to 1 channel + global residual connection `x + delta`.

### 2.2 TCN1D (`models/tcn.py`)
- **Physical Motivation for Non-Causal Padding**: In standard time-series modeling, causal (unidirectional) padding is mandatory because future points are unknown. In XPS, binding energy is a static, bidirectional physical axis where peaks (core levels, spin-orbit split doublets, Auger and shake-up satellites) possess bidirectional symmetry and local dependencies in both directions along binding energy. Using causal padding would introduce artificial peak shifts and asymmetric edge distortion. Therefore, `TCN1D` uses **non-causal symmetric reflection padding** (`SafeReflectionPad1d`).
- **Residual Dilated Blocks**: Stacks residual blocks with exponential dilation factors (`dilations: [1, 2, 4, 8, 16]`). Each block contains two dilated 1D convolutions with configurable odd `kernel_size` (default 5), `SafeBatchNorm1d`, activation (`relu`), and dropout.
- **FiLM Modulation**: Feature-wise linear modulation on the final dilated block representation before the 1x1 projection head.
- **Output Head**: 1x1 Conv1d projecting to 1 channel + global residual connection `x + delta`.

### 2.3 FPN1D (`models/fpn.py`)
- **Bottom-Up Pathway (Encoder)**: `depth` stages (default 3). Stage $i$ has channels $C_i = \text{fpn\_base\_channel} \cdot 2^i$ (default `fpn_base_channel: 16`). Stage 0 operates at full resolution $L$; stages $1 \dots \text{depth}-1$ are preceded by `nn.MaxPool1d(kernel_size=2, stride=2)`.
- **Top-Down Pathway & Lateral Connections**: Each encoder level is mapped to `fpn_pyramid_channel` (default 32) via a 1x1 lateral conv. Higher pyramid levels are upsampled via linear interpolation, added element-wise to lateral maps, and smoothed with a 3x1 convolution.
- **Multi-Scale Fusion Head**: All pyramid levels ($P_0 \dots P_{\text{depth}-1}$) are upsampled to full length $L$, concatenated along channels (`depth * fpn_pyramid_channel`), passed through a 3x1 fusion conv + `SafeBatchNorm1d` + activation, optionally modulated with FiLM, and projected to 1 channel via a 1x1 conv + global residual shortcut `x + delta`.

---

## 3. Dedicated Per-Model Configuration Files & Schema

Migrate all model configurations into dedicated YAML files in `config/`:
- `config/resnet.yaml`: ResNet and ResNetScaledShort configurations.
- `config/unet.yaml`: Conventional U-Net and Residual U-Net configurations.
- `config/deeplabv3.yaml`: DeepLabV3 1D ASPP configuration.
- `config/inception_time.yaml`: InceptionTime1D configuration.
- `config/tcn.yaml`: Dilated TCN1D configuration.
- `config/fpn.yaml`: Feature Pyramid Network 1D configuration.

Each YAML file defines:
- `train_config`: Model architecture hyperparameters and training loop settings.
- `region_configs`: Optional per-region overrides.
- `bayesian_opt`: Bayesian optimization settings and model-specific parameter search space.
- `film`: Model-specific FiLM conditioning switches and embedding dimensions.

---

## 4. Multi-Layer Enforcement of Sliding-Window Prohibition

`use_sliding_window: true` is strictly prohibited for `inception_time`, `tcn`, and `fpn` at four independent barriers:
1. `utility/config_schema.py` (`PipelineConfig` validator) & `utility/config_loader.py` (`resolve_pipeline_config`).
2. `models/orchestration.py` (`perform_model_training_and_bayesian_optimization`).
3. `models/trainer.py` (`_setup_training_dataloaders`) & `models/bayesian_opt.py` (`_build_architecture_parameters`).
4. `InceptionTime1D.__init__`, `TCN1D.__init__`, and `FPN1D.__init__`.

Each layer raises a clear `ValueError` stating that multi-field-of-view models process full regional spectra directly.
