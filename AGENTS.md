# GGRD-SP — Agent Guide

## Project Overview
GGRD-SP is a sequence-to-sequence machine-learning project for **cross-tool XPS spectrum transfer**: given a spectrum measured on a *source* tool, predict the spectrum the same specimen would produce on a *target* tool. The model learns the systematic difference between tools so as to transfer spectra between tools.

## Repository Layout
| Path | Contents |
|------|----------|
| `main.py` | End-to-end pipeline orchestration (stages 1–13 below) |
| `main_notebook.ipynb` | Exploration |
| `config/config.yaml` | Single pipeline config (`# yaml-language-server: $schema=config.schema.json`) |
| `config/config.schema.json` | JSON schema for editor autocomplete/hover docs |
| `models/` | Datasets, loss, networks, FiLM, training, Bayesian optimization, GP flux, inference |
| `denoise/` | Noise analysis and de-noising (`savgol`, `wavelet`, `fourier`, `gp`) |
| `utility/` | Config loading/schema, pairing, splitting, preprocessing, normalization, patching, quantification, evaluation, reporting |
| `visualization/` | Plotting modules and the config-driven dispatcher |
| `tests/` | `unittest` test suite (`test_*.py`) |
| `docs/` | Documentation |

## Tech Stack
Python 3 (CPU-only) · PyTorch · GPyTorch · Ax/BoTorch · Pydantic · NumPy · Pandas · SciPy · scikit-learn · PyWavelets · Matplotlib/Seaborn

## Core Data Model
- **Measurement (session)**: one XPS run of one material on one tool at one time. It produces `N_die` full spectra, each from a die (default 9 dies) on a wafer.
- **Full spectrum**: the signal from one die and it contains `N_region` regional spectra (`N_region` depends on the material).
- **Regional spectrum**: the spectrum of a region (certain part in binding energy axis, generally designated by elemental orbital like Ti2p), containing `N_points` sampled intensity over binding energy.
- `ary_intensity`, `ary_energy`: 2D NumPy arrays with shape (`N_rows`, `N_points`). Row *n* of `ary_intensity` is one regional spectrum from a die of a measurement of a material. The corresponding binding energy is located also at row *n* in `ary_energy` (in eV, ascending). `N_rows` = sum over measurements of `N_die × N_region`.
- `df_meta`: a dataframe served as informatio carrier for spectra. One row per regional spectrum. It holds material, tool, measurement time, die, region, `N_points`, and `spectrum_index` (row in the intensity and energy arrays). Later stages add columns to it to enrich more information.

## Pipeline Stages
Each stage has its own config section in `config/config.yaml`.

### 1. Data reading / generation 
Build `ary_intensity`, `ary_energy`, and `df_meta` as defined above.

### 2. Source–target pairing
- `source_tool`, `target_tool` come from `global`. `pairing.material` is `'all'` (pair within each material separately) or one material name.
- Measurements are matched **1-to-1** with `|Δt| <= time_threshold_hours` (default 12). `method`: `optimal` (Hungarian) or `greedy` (nearest-first). Source measurements without a match stay unpaired.
- Dies and regions inherit the pairing: source (die A, region B) <-> target (die A, region B).
- Adds to `df_meta`: `target_tool`, `target_measurement_id`, `target_spectrum_index`, `target_time`, `time_diff_hours`, `target_t7_code`.

### 3. Train/val/test splitting 
- Splits at the **measurement level**, so all dies and regions of a measurement share a split. This prevents spatial and temporal data leakage.
- Adds a `split` column: `train` | `val` | `test` | `none` (unpaired).
- Modes:
  - `random`: uses `train_ratio`, `val_ratio`, `test_ratio`, and `seed` to do splitting (default `None`).
  - Direct assignment: a dict `{train, val, test}` → lists of measurement IDs. Each ID must appear in exactly one group, and together the groups must cover all paired measurements.
- Returns `session_splits` and `measurement_mapping` (source measurement ID -> target measurement ID).

### 4. Noise analysis & de-noising
- Methods:
  - `savgol`: Savitzky–Golay.
  - `wavelet`: SWT/DWT shrinkage.
  - `fourier`: DCT or zero-phase Butterworth `filtfilt`.
  - `gp`: exact GP with Matérn/RBF kernel and a homoscedastic or Poisson-scaled noise model.
- Adds to `df_meta`: `noise_analysis_index`, `gp_noise_sigma`, `gp_snr`, `gp_lengthscale`.
- Allow for applying de-noise to designated regions (default `Cl2p`, `Ti2p`) by `global.apply_denoise: true`. The de-noise can be limited to measurements in source, target, or both tools by `apply_to` (`both` | `source` | `target`). The un-denoised `ary_intensity` is kept by a deep copy.
- Reports atomic-percentage differences and regional MAE between raw and de-noised spectra.

### 5. Pre-processing
- **Physical metrics**:
  - All integration done by numpy trapezoidal integration over energy.
  - Per region: `total_flux_single_region`, `mean_intensity_single_region`.
  - Per die (added to `df_meta`): `total_flux_single_die` (sum of regional fluxes) and `mean_intensity_single_die` (mean over all regional points).
- **Categorical indices** (zero-based integers): `source_tool_index`, `target_tool_index`, `region_index`.
- **Standardization**: adds `time_standardized` and `total_flux_single_die_standardized`. The z-score mean and standard deviation are fitted **only on `train`** to avoid data leakage.
- **Normalization**: input and output: `ary_intensity`, `ary_energy`, `df_meta`; defensive clamping against near-zero scales using configurable `eps`. Normalization modes available:

  | Option | Each regional spectrum is divided by |
  |--------|--------------------------------------|
  | `by_total_flux_single_die` (default) | `total_flux_single_die` of its die/measurement |
  | `by_regional_max_intensity` | its own regional maximum intensity|
  | `by_regional_max_intensity_of_source` | the maximum intensity of the **paired source** regional spectrum (applied to both source and target; unpaired spectra are left unscaled) |
  | `by_mean_intensity_single_die` | `mean_intensity_single_die` of its die/measurement |
  | `no_normalization` | none (unscaled) |

### 6. Die-flux GP modeling
- Exact GP (GPyTorch) of die flux over time, capturing drift, aging, and maintenance jumps.
- The kernel combines three parts:
  - **Temporal**: RBF or Matérn.
  - **`SigmoidStepKernel`:** $k(t_1,t_2)=\sigma_{jump}^2\,\sigma(s(t_1-t_0))\,\sigma(s(t_2-t_0))$, with learnable `t0` and `s`.
  - **Die index**: a categorical covariance across dies.
- Reports MAE and `mape_{train,val,test}_percentage` per tool.

### 7. Datasets & sliding window
- Two first-class layouts:
  - **Regional**: one `dataset_train`/`dataset_val`/`dataset_test` bundle per region. Train a model for each region indepedently.
  - **Unified**: all regions in one dataset. Train a model for all regions together.
- **Sliding window** (`global.use_sliding_window`, available to every model):
  - Window and stride can be given in eV (defaults 6.0 / 2.0) or in points.
  - Right-edge anchoring gives 100% coverage with no padding.
  - Each source patch maps to a target patch of the same length W.
  - Full spectra are rebuilt by averaging the overlapping patch predictions.
  - Validation cost and early stopping use the **reconstructed full spectrum**.
  - Window and stride can be tuned by Bayesian optimization.
  - When using the unified layout, sliding window size is forced to be fixed over all regions for ease of maintenance (it's not necessary as CNN layers are indifferent to sequence size in general).
  - [Need to check] FiLM encoding like region for unifeid layout all fall back to first region? 

### 8. Cost function
- The total cost is a weighted mean over regions (equal weights by default).
- Per-region cost is the weighted sum of:

  | Term | Default weight |
  |------|----------------|
  | MSE of `I_pred/max(I_target)` vs `I_target/max(I_target)` (turn the division off with `normalized_by_target_max`) | — |
  | Cosine distance `1 − cos(I_pred, I_target)` | 0 |
  | MSE of central-difference gradients | 0 |
  | L2 on model parameters (`cost.l2_weight`) | 0 |

- Works on both patch tensors and full regional spectra.
- Possible devision by tiny number when `normalized_by_target_max` is on is guarded and will raise warning of hitting the guard.

### 9. Models, FiLM & training
- Orchestration: build the model → define the Ax/BoTorch search → train with an LR scheduler (`ReduceLROnPlateau` or `CosineAnnealingLR`) and early stopping. Returns the best model.
- Models (`global.model_type`):

  | Key | Architecture |
  |-----|--------------|
  | `resnet` | `num_layers` conv layers (default 3) + identity shortcut; reflection padding keeps the length; kernel size and stride are tunable |
  | `resnet_scaled_short` | `resnet` with a learnable scale on the shortcut (`ResNetScaledShort1D`) |
  | `residual_unet` | `y = x + UNet(x)`, depth 3 by default |
  | `unet` | `y = UNet(x)` |
  | `deeplabv3` | 1-D ASPP (parallel dilated convolutions) |

- **FiLM** (`global.use_film`): `h' = γ(c)·h + β(c)`.
  - The condition `c` contains: die index, source/target tool index, `time_standardized`, standardized source die flux, region index, relative intensity, energy bounds / `energy_scale_ev`, and a material element multi-hot vector. None of these may leak target information.
  - Energy bounds: conv blocks encode the window bounds when sliding window is on and the regional bounds when it's off. The scaled-shortcut FiLM always uses `encode_energy: false`, so the tool scaling is the same for every window.
- **Search space:** learning rate, L2 weight, batch size, kernel size, channels, depth, and (if sliding window is on) window and stride.

### 10. Prediction
- Predicts target spectra for train, val, and test. Sliding-window models predict per patch (conditioned on patch energy bounds) and then reconstruct the full spectrum.
- `clamp_non_negative: true` clamps predictions at 0 to ensure non-negativity of a spectrum.
- Returns `(ary_intensity_pred, ary_energy_pred, df_meta_pred)`, keeping the energy grids and source metadata.

### 11. Evaluation
- Computed per sample and aggregated per region:
  - normalized MSE;
  - raw RMSE;
  - `peak_error_percentage`: error in peak intensity of the predicted spectrum relative to the peak intensity of the target spectrum.  
  - `chebyshev_error_peak_normalized_percentage`: largest point-wise relative error (Chebyshev / L-infinity norm normalized by target maximum intensity).
- Prints side-by-side train/test tables.

### 12. Quantification
- Steps for calculate atomic percentage:
  1. Iterative Shirley background removal. Endpoint options are `minima`, `edge`, or `direct`, averaged over an eV window with mirror padding; iteration stops at `max_iter` or `tol`. The option `direct` is a special '2-side directional search' for some XPS tool types.
  2. Trapezoidal net area.
  3. Divide by the RSF, taken **only** from `quantification.rsf_table`.
  4. Normalize to 100% per measurement and die.
- Reports MAE and base MAE across `source_atomic_percentage`, `target_atomic_percentage`, and `predicted_atomic_percentage`, per split and element.

### 13. Visualization
Plots should go here and controlled by the dispatcher with one boolean toggle per plot:
- GP flux timeline with maintenance steps
- Pairing timeline
- Flux and mean intensity vs. time
- Shirley overlays
- Noise decomposition and SNR
- GP convergence
- Raw vs. de-noised overlay (solid black vs. dashed dark pink)
- De-noising atomic-% MAE box plots
- Loss histories
- Prediction vs. target overlays
- Intensity distributions
- Atomic-% distributions and MAE plots
- Sliding-window decomposition and reconstruction

---

## Agent Instructions

### 1. Environment
- Execution is strictly CPU-only (CUDA/GPU is disabled and should not be used).
- Run Python only through `.\.venv\Scripts\python.exe`, from the repository root.
- Run tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests`.
- Run main.py in headless mode (by passing the argument `--headless`) to suppress plotting window during test, otherwise the program never ends 

### 2. Workflow
1. **Clarify:** if anything is ambiguous, ask before acting.
2. **Propose:** before changing existing code, state what will change and why. Wait for approval.
3. **Implement:** once approved, implement, test, and commit without further confirmation rounds.
4. **Review loop:** for new or significant code, launch a reviewer subagent. It critiques the code, probes edge cases, checks error handling, and runs tests. Fix what it finds. Repeat up to 3 iterations.
5. **Commit & push:**
   - Use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).
   - Split any change over 50 lines into readable, logically grouped commits.
   - List every commit hash and message in the final summary.

### 3. Editing Existing Code (strict)
- **Minimal diff:** change only the lines the request needs. Never rewrite whole files or functions without permission. Use targeted search-and-replace edits, not full-file overwrites.
- **No unsolicited changes:** keep unrelated names, formatting, comments, docstrings, import order, defensive checks, and error handling as they are, even if they could be shorter.
- **No regressions:** make sure existing workflows still work, and run the relevant tests.

### 4. Code Style
- PEP 8, except: code lines ≤ 120 characters; comment and docstring lines ≤ 80 characters.
- Every new function gets type hints and a concise docstring.
- For a function with coding line > 30, add 1~2 lines comment on code block to describe the purpose of the block. 
- Cyclomatic complexity ≤ 10 per function. Above that, split into single-responsibility helpers.

### 5. Modularity
- New features go in dedicated modules under `models/`, `denoise/`, `utility/`, or `visualization/`. Expose them as functions or classes that `main.py` can import. Don't edit `main.py` or other modules unless asked.
- Keep coupling between modules low. Prefix internal helpers with `_`.
- When merging or splitting modules, don't leave facade modules or re-exports behind for backward compatibility.
- **Parameter packaging:** a public function with ≥ 5 parameters takes a config `dict` instead. `_`-prefixed helpers are exempt.

### 6. Configuration
- **Single source of truth:** these shared settings are defined once under `global:` in `config/config.yaml` and passed to the stages via `resolve_pipeline_config`:
  - `material`, `model_type`, `use_sliding_window`, `use_film`, `apply_denoise`, `unified_model`
  - `source_tool`, `target_tool`, `example_region`, `sample_measurement_id`
- **Validation:**
  - Pydantic (`utility/config_schema.py`) validates the config when it's loaded.
  - Missing required keys raise an exception that names the missing keys.
  - Defaults live in the YAML or the schema, not in fallback blocks in the code.
- **No aliases:** never write alias maps or support legacy keys. Unknown keys are tolerated by the schema (`extra="allow"`), and no code may read them as substitutes for canonical keys.
- **No redundant checks:** don't re-cast schema-typed values (`int()`, `float()`, `bool()`, `str()`) or repeat static checks (positive, odd, …) in code. Runtime checks are only for what the schema can't know, e.g. `kernel_size <= n_points_generalized`.

### 7. Naming
- Use full, explicit names: `learning_rate` (not `lr`), `energy_endpoints` (not `points`), `epochs` (not `num_epochs`).
- Paired context uses prefixes: `source_*` and `target_*` (`target_measurement_id`, not `measurement_id_target`).
- Z-scored metadata uses `*_standardized`. `normalize_*` is only for scaling spectral intensity.
- Predictions use `predicted_*`, never `pred_*` (e.g. `predicted_flux`, `predicted_atomic_percentage`).
- Percentages use `percentage`, never `%` or `_pct` (e.g. `peak_error_percentage`, `abs_diff_atomic_percentage`).

### 8. Errors & Unexpected Cases
- Error messages must say what failed, which value caused it, and what was expected.
- Never fail or fall back in unexpected cases silently.
- Never quietly work around an unexpected case or guess at a fix. Ask the user first.

### 9. Line-Count Requests
Only code lines count; comments and docstrings don't.
