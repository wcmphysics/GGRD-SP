# GGRD-SP Project Context
This is a sequence-to-sequence Machine Learning project that try to find how to transform a spectrum meassured at a tool to another spectrum as if the specimen were measured at another tool. Namely, we want to use machine learning model to capture tool difference in spectra so we can transfer spectra between tools. This project relies heavily on PyTorch for neural networks and Pandas/SciPy for data processing.

## Program Structure
The codes will be composed of the following key stages
1. Spectrum data reading or generation
   - there are many kinds of materials for the specimen under XPS measurement (defined in config)
   - 1 XPS measurement session gives many (specified by `N_die` and defaulted to `9`) full spectra
   - 1 full spectrum contains many (specified by `N_region` depending on material) regions (or called regional spectra)
   - one region has `N_points` data points over binding energy
   - store the regional spectrum data (arrays of intensity and binding energy) and meta data separately
   - store regional spectrum intensity as 2d numpy array `ary_intensity` where the nth row is the nth intensity array
   - store regional binding energy as 2d numpy array `ary_energy` where the nth row is the nth binding energy array in eV (ascending order)
   - Note that one array in `ary_intensity` or `ary_energy` is one regional spectrum for one specific material, one specific measurement, one specific die, and one specific region. The shape of `ary_intensity` or `ary_energy` is (`N_measurement`*`N_die`*`N_region`, `N_points`).
   - meta data (`df_meta`) is a pandas dataframe containing all meta data of the spectra recorded
   - meta data includes: material, tool and time of measurement, die, region, number of points in one spectrum, and the corresponding index in the intensity and energy array (`spectrum_index`)
   - the meta data dataframe `df_meta` serves as a body to carry information for all spectra and will be enriched by other information later
2. Pairing source and target spectra
   - allow users to specify source tool (`source_tool`) and target tool (`target_tool`) in config
   - allow filtering by material (`pairing.material`): `'all'` pairs within each material separately, or a specific material name pairs only that material
   - for every measurement in the source tool, we find/map a measurement from the target tool
   - this source to target mapping is 1-to-1 (no multiple source measurement is linked to the same target measurement)
   - the mapping maps measurements from the source tool to the target tool, the die and region in that measurement will also be paired accordingly (spectrum from die A region B of the source measurement will map to the same die A and the same region B in the target measurement)
   - the time difference between source and target measurement should be lower than a user-specified threshold (`time_threshold_hours`, default to 12 hours), utilizing optimal bipartite matching or greedy temporal matching
   - pairing information is stored in the meta data dataframe (add columns like `target_tool`, `target_measurement_id`, `target_spectrum_index`, `target_time`, `time_diff_hours`, `target_t7_code`)
3. Train-val-test data splitting
   - need a dedicated config for this part (`splitting`)
   - the splitting is based on measurement (session level) and its corresponding dies and regions follow that splitting to prevent spatial and temporal data leakage
   - need to add a `split` column into the meta data `df_meta` to indicate which split a row belongs to
   - the split column can take `train`, `test`, `val`, or `unpaired` (or none) if the measurement is not paired
   - allow for setting up splitting ratio for train, val, and test (`train_ratio`, `val_ratio`, `test_ratio`)
   - 2 options for splitting: (1) random and (2) direct assignment
   - configurable seed for random splitting (default to None)
   - split for direct assignment takes in a dictionary with keys `train`, `val`, and `test` and their values are a list of measurement ids. Check must be made to verify one measurement only appears in one group and the 3 groups take all paired measurements
   - output a session partition dictionary `session_splits` and dictionary `measurement_mapping` with keys being the measurement id from the paired source tool and values being the corresponding paired measurement id from target tool
4. Spectral noise analysis and de-noising
   - encapsulated in the `denoise/` package with dedicated configs (`noise_analysis`, `denoise`)
   - supports four signal-noise decomposition and de-noising techniques:
      1. Savitzky–Golay polynomial filtering (`savgol`)
      2. Wavelet coefficient shrinkage (`wavelet`, supporting Stationary and Discrete Wavelet Transforms via PyWavelets)
      3. Fourier / zero-phase low-pass filtering (`fourier`, supporting Discrete Cosine Transform and zero-phase Butterworth `filtfilt`)
      4. Exact Gaussian Process regression (`gp`, with smooth Matérn/RBF kernels and homoscedastic or Poisson-scaled noise models)
   - quantifies regional noise standard deviation sigma and signal-to-noise ratio (SNR), appending `noise_analysis_index`, `gp_noise_sigma`, `gp_snr`, and `gp_lengthscale` to `df_meta`
   - deep-copies `ary_intensity` to preserve non-denoised spectra, applies de-noising to configured regions (default `Cl2p` and `Ti2p`) and tool scope (`apply_to`: `'both'`, `'source'`, or `'target'`), and updates `ary_intensity` for downstream stages when `global.apply_denoise` is `true`
   - evaluates atomic percentage differences and regional Mean Absolute Error (MAE) between non-denoised and denoised spectra
5. Pre-processing
   - encapsulated pre-processing module (`preprocess`) and spectral normalization (`normalize_spectral_data`)
   - physical spectral flux and mean intensity metrics are extracted via numerical trapezoidal integration over binding energy:
      - regional flux (`total_flux_single_region`) and regional mean intensity (`mean_intensity_single_region`)
      - die-level total flux (`total_flux_single_die`): summation of integrated flux among all regions for that specific die and measurement session
      - die-level mean intensity (`mean_intensity_single_die`): average intensity across all regional points for that specific die and measurement session
      - both die-level metrics are added directly into `df_meta`
   - establish canonical zero-based integer indexing for neural network categorical conditioning:
      - `source_tool_index`: integer index mapping for source tool
      - `target_tool_index`: integer index mapping for target tool
      - `region_index`: integer index mapping for each regional spectrum
   - metadata feature standardization (`standardize_metadata_features`): standardizes continuous conditioning variables (`time_standardized`, `total_flux_single_die_standardized`) with mean and standard deviation fitted strictly on the `train` split to prevent information leakage
   - normalization of spectral data:
      - inputs are `ary_intensity`, `ary_energy`, and `df_meta`
      - 5 options for performing normalization: 'by_total_flux_single_die' (default), 'by_regional_max_intensity', 'by_regional_max_intensity_of_source', 'by_mean_intensity_single_die', or 'no_normalization'
      - for normalization by total flux (`by_total_flux_single_die`), intensity of every regional spectrum from a die in a measurement was divided by the total flux of that die in that measurement (`total_flux_single_die`, summation of flux among all regions for that specific die and that specific measurement)
      - for normalization by regional maximum intensity (`by_regional_max_intensity`), intensity of a regional spectrum is divided by the maximum intensity of that specific regional spectrum
      - for normalization by source regional maximum intensity (`by_regional_max_intensity_of_source`), this normalization is applied to each region individually. For each paired record, both the source regional spectrum and its paired target regional spectrum are divided by the maximum intensity of that source regional spectrum. Unpaired spectra are not scaled (left unscaled)
      - for normalization by mean intensity (`by_mean_intensity_single_die`), intensity of a regional spectrum is divided by the mean intensity across all regional points for that die in that measurement (`mean_intensity_single_die`)
      - for no normalization (`no_normalization`), intensity array is left unscaled
      - defensive scale clamping against near-zero scales using configurable `eps`
      - output are `ary_intensity` (normalization applied), `ary_energy`, and `df_meta`
6. Die flux modeling and prediction
   - need a dedicated config for this part (`gp_flux`)
   - models tool intensity drift, aging, and baseline shifts in die flux over time using exact Gaussian Process regression (GPyTorch)
   - features a composite covariance kernel:
      1. continuous temporal covariance (RBF or Matérn kernel) capturing smooth drift
      2. learnable sigmoid step kernel (`SigmoidStepKernel`): k(t1, t2) = sigma_jump^2 * sigmoid(s * (t1 - t0)) * sigmoid(s * (t2 - t0)), with learnable transition time t0 and steepness s capturing abrupt tool maintenance baseline shifts
      3. categorical die index covariance modeling inter-die spatial correlations across the sample holder
   - evaluates Mean Absolute Error (MAE) and Mean Absolute Percentage Error (`mape_train_percentage`, `mape_val_percentage`, `mape_test_percentage`) across train, val, and test partitions for source and target tools
7. Data set and data loader definition (with sliding window patch augmentation)
   - define dataset and data loader in this part
   - supports two equal first-class paradigms for dataset organization:
      1. Regional Datasets: separate dataset bundles (`dataset_train`, `dataset_val`, `dataset_test`) assembled independently for each core-level spectral region
      2. Unified Dataset: merges all spectral regions into a single unified dataset (`create_unified_dataset`), enabling a single neural network to process all regions conditioned on categorical region embeddings
   - data augmentation by sliding window, or sometimes called patch, is utilized if users turn this option on in the config file (`use_sliding_window`)
   - sliding window should be available to each model and user can specify whether to use it or not
   - when sliding window is on, prepare the dataset and data loader accordingly
   - details about sliding window:
      - local sequence transformation using sliding windows (patches) from the source spectrum to the target spectrum
      - window size and sliding stride can be specified in eV (default window size: 6.0 eV, default stride: 2.0 eV) or in data points
      - sliding window slices a regional spectrum into overlapping patches to capture local spectral features and augment training samples
      - right-edge anchoring guarantees 100% spectral coverage using only real physical measurements without artificial boundary padding
      - patch-to-patch mapping: predicts a target patch of length W for each input window (`SpectrumPatchDataset`)
      - the predicted regional spectrum is reconstructed by accumulating and averaging all overlapping predicted slices across that region
      - validation loss and early stopping are evaluated directly on the reconstructed full spectrum
      - when sliding window is used, Bayesian optimization should allow for hyper-parameter search for window size and stride
8. Cost function definition and preparation
   - define cost functions for model to use in this part (`SpectralLoss`)
   - need a dedicated config for this part (`cost`)
   - the cost function is mainly composed of a weighted average error among all regions (default is equal weight for all regions but tunable by user via the config)
   - the error in each region is defined by several weighted components (each weight is configurable):
      1. mean-squared error (MSE) from I_pred/max(I_target) and I_target/max(I_target), with an option (`normalized_by_target_max`) to turn off the max(I_target) normalization
      2. cosine similarity (1 - cos(I_pred, I_target)) penalizing spectral shape divergence (default weight=0)
      3. mean-squared error (MSE) between gradient of predicted and true intensity (central difference, default weight=0) to preserve peak inflections and peak shapes
      4. L2 regularization term (default weight=0) penalizing model parameter magnitudes
   - the cost function handles both sliding window patch tensors and full regional spectra
9. Neural network definition, training, and hyperparameter search (by Bayesian optimization)
   - need a dedicated config for this part (`training`, `film`, `bayesian_opt`)
   - we want to train a model such that it can find the mapping of spectra from source tool to the target tool, that is, the transformation such that the measurement performed on source tool can be transformed into the measurement performed on the target tool
   - the transformation info lies in the paired source and target measurement compiled in the "pairing source and target" step
   - we have many models and each of them can be specified in the config feeding to a main orchestration function
   - output should contain the trained model (best model from model training and Bayesian optimization of hyper-parameters)
   - the general architecture and workflow inside a main orchestration function will be:
      1. model definition and architecture instantiation
      2. definition of hyper-parameter search (Bayesian optimization via Ax / BoTorch)
      3. model training and validation with learning rate scheduling and early stopping
   - here is a list of models implemented:
      - ResNet (`resnet`):
         - convolutional residual network composed of configurable CNN layers (default 3 layers, tunable via `num_layers`) and one shortcut (kernel size and step size are hyper-parameters)
         - padding to maintain the sequence length (by mirror reflection padding)
      - ResNet-ScaledShort (`resnet_scaled_short`):
         - residual network featuring a learnable scaling factor on the identity shortcut path
      - Residual U-Net (`residual_unet`):
         - this model combines the idea of shortcut and conventional u-net so the target y and input x has the relation "y = x + UNet(x)"
         - default depth 3 (tunable by user)
      - U-Net (`unet`):
         - conventional encoder-decoder U-Net without shortcut: from "y = x + UNet(x)" to just "y = UNet(x)"
      - DeepLabV3 (`deeplabv3`):
         - 1D Atrous Spatial Pyramid Pooling (ASPP) network capturing multi-scale context through parallel dilated convolutions
   - Feature-wise Linear Modulation (FiLM):
      - dynamically modulates intermediate convolutional feature maps: h' = gamma(c) * h + beta(c)
      - multi-attribute condition vector c encodes: die index, source tool index, target tool index, standardized measurement time, standardized source die flux, region index, relative intensity, energy bounds normalized by `energy_scale_ev`, and material element multi-hot vector without target leakage
      - FiLM energy encoding roles: CNN convolutional blocks encode window bounds when sliding window is active and regional bounds when inactive; scaled shortcut FiLM (`ResNetScaledShort1D`) unconditionally disables energy bounds (`encode_energy: false`) to ensure tool scaling is uniform and invariant across spectral windows
   - Hyperparameter search via Ax Bayesian Optimization:
      - tunes learning rate, L2 regularization weight, batch size, convolutional kernel size, channel capacities, network depth, and sliding window parameters
      - employs ReduceLROnPlateau or CosineAnnealingLR schedulers and validation early stopping
10. Prediction from model
   - dedicated configuration section (`prediction`)
   - use the trained model from the training part to predict target spectra on train/validation/test data for model performance observation later
   - for sliding window models, slices input spectra into patches, conditions on patch energy bounds, and reconstructs full continuous spectra via overlap accumulation and averaging
   - supports non-negative clamping (`clamp_non_negative: true`) to preserve physical intensity bounds
   - preserves energy grids and source metadata
   - output contains predicted intensity, energy, and meta data: `(ary_intensity_pred, ary_energy_pred, df_meta_pred)`
11. Quantitative model evaluation
   - dedicated configuration section (`metrics_report`)
   - computes numerical metrics comparing predicted spectra against ground truth target spectra per sample and aggregated per region:
      - Normalized MSE and Raw RMSE
      - Peak Relative Error Percentage (`peak_error_percentage`): relative error percentage at the ground truth peak maximum location
      - Maximum Absolute Error Percentage (`max_error_percentage`): maximum point-wise relative percentage deviation across the regional spectrum
   - formats structured side-by-side comparison tables across train and test partitions
12. Calculation of atomic percentage for elements by area integration (Quantification)
   - dedicated configuration section (`quantification`)
   - to get atomic percentage for elements you need to remove background, integrate area, and correct with sensitivity factors
   - you do not need to deconvolute peaks, you only need to remove background signal using iterative Shirley background subtraction:
      - flexible endpoint determination strategies: local minima search (`minima`), boundary edge points (`edge`), or explicit energy points (`direct`), with mirror-padded boundary averaging over an eV window
      - iterates to numerical convergence bounded by `max_iter` and tolerance `tol`
   - integrate net peak areas via numerical trapezoidal integration after background removal
   - relative sensitivity factors (RSF) are read strictly from `quantification.rsf_table` in `config/config.yaml`
   - normalize scaled areas row-wise to 100% atomic percentage per measurement session and die
   - evaluate Mean Absolute Error (MAE) and base MAE between source, target, and predicted atomic percentages (`source_atomic_percentage`, `target_atomic_percentage`, `predicted_atomic_percentage`) across splits and elements
13. Diagnostic visualization dispatching
   - dedicated configuration section (`visualization`)
   - centralized dispatching module driven by boolean toggles in YAML config
   - supported diagnostic plots:
      - GP die flux timelines and maintenance step tracking
      - temporal pairing timeline between source and target tools
      - flux and averaged intensity vs. measurement time
      - Shirley background subtraction and endpoint overlays
      - spectral noise decomposition and SNR curves
      - GP hyperparameter optimization convergence curves
      - non-denoised vs. denoised regional spectrum comparison overlays (solid black vs. dashed dark pink)
      - denoised vs. non-denoised atomic percentage Mean Absolute Error (MAE) regional box plots
      - model training and validation loss progression histories
      - transformed prediction vs. target spectral comparison overlays
      - regional spectrum intensity distributions
      - atomic percentage distributions and Mean Absolute Error (MAE) box/bar plots across splits
      - sliding window patch decomposition and reconstruction slices

## Tech Stack
- Python 3
- Scikit-learn, PyTorch, GPyTorch, Pandas, NumPy, SciPy, PyWavelets, Ax (for Bayesian optimization)
- Jupyter Notebooks for exploration, Python scripts for utilities.

## Agent Instructions
1. **Virtual Environment:** Always execute Python commands using the local environment at `\.venv\Scripts\python.exe`.
2. **File Structure:** 
   - Main python body: `main.py`
   - Core ML models, denoising/noise analysis, utilities, and plotting modules belong in the `models/`, `denoise/`, `utility/`, and `visualization/` folders.
3. **Code Style:** Use strict PEP-8 formatting with two exception: one coding line cannot exceed 120 characters and one comment or docstring line cannot exceed 80 characters. Include type hints and concise docstrings for all new functions. 
3. Ensure cyclomatic complexity <= 10 in a function. Decompose the function with single-responsibility helper functions if the complexity > 10. 
4. **Autonomous Commits and Pushing:**
   - Once a feature request, plan, or review iteration is approved, autonomously commit and push the changes without asking for separate micro-permissions on commit messages.
   - Follow standard Conventional Commits format (e.g. `feat:`, `fix:`, `test:`, `docs:`) and always display the created commit hashes and messages in your final summary for transparency.
5. **Commit Size:** When the change is more than 50 lines, try to separate the commits and so one commit is readable and will not be flooded with changes.
6. **Modular Functionality:** 
   - Implement new features in a strictly modular fashion inside dedicated modules within `models/`, `denoise/`, `utility/`, or `visualization/`.
   - New functionality must be encapsulated as callable functions or classes ready for import by `main.py`, avoiding unsolicited edits or side effects to `main.py` or other existing modules.
   - Try to minimize coupling between modules so editing one is less likely to break the others.
   - Encapsulate internal functions that will not or should not be used by other modules.
   - Do not leave dead backward-compatibility facade modules or intra-module re-exports when consolidating or decomposing modules.
7. **Clean Function Interfaces (Parameter Packaging):**
   - If a public function requires 5 or more parameters (>= 5), bundle them into a readable configuration dictionary. Internal helper functions (whose names start with `_`) are exempt and may accept 5 or more parameters.
   - Dictionaries are strongly preferred by default for readability and simplicity. A dedicated dataclass should only be used when there is a clear, tangible advantage over a dictionary (such as complex nested hierarchies or strict validation requirements).
8. **Existing Code Modifications & Safeguards:**
   - When modifying existing code is required, briefly state what will change in your explanation. Once given the go-ahead, proceed through implementation, testing, and commits without redundant confirmation rounds.
   - Verify and guarantee that the modifications do not break existing functionality or workflows.
9. **Multi-Agent Review and Test Loop:**
   - When writing new code or significant components, launch a separate reviewer subagent to critique the code, probe for edge cases, inspect error handling, and run verification tests.
   - Address the reviewer agent's feedback and refine the implementation.
   - Repeat this write-test-rewrite loop up to 3 iterations until the code is robust and passes all checks.
10. ** Error Handling and Warning*:*
   - When error occurs provide informative details about the error.
   - Avoid silent error.
   - Always discuss with the user first about how to handle unexpected cases or error. Do not handle unexpected case silently. Do not guess what the fix should be without asking or at least notifying the user. 
11. ** Code Line Count Reduction**
   - When asked to reduce the number of code lines (comment and docstrings are not counted) for one specific file or many specified files:
      1. DO NOT EVER modify anything outside the specified files. 
      2. DO NOT package coding lines into another module as it is merely moving lines to somewhere else rather than cutting down lines. 
      3. DO NOT rename variables to shorten the line counts. Variable names should be easy to understand for reader. 
      4. DO NOT just use `;` to put different lines into one line. This is not cutting down lines.
      5. Note that comment and docstring lines are not counted. Exploit this to ensure code clarity and readability when reducing code lines.
      6. Comment lines and docstring should still be complete and comprehensive. No need to remove them unless explicitly requested.
   - Ask me anything if there is something unclear to you
   - Always Propose your plans first for me to review. Only implement the plan if I approve
12. **Strict Canonical Configurations & Comprehensive Variable Naming:**
   - Always use strict, explicit, and comprehensive canonical names for
     configuration keys and variables (e.g., use `learning_rate` instead
     of `lr`, `energy_endpoints` instead of `points`, `epochs` instead
     of `num_epochs`).
   - DO NOT construct bloated alias-mapping dictionaries or maintain lists
     of legacy/deprecated keys.
   - Treat any non-canonical or unrecognized configuration key simply as an
     invalid key and raise an informative `KeyError` or `ValueError` stating
     the invalid key and what canonical keys are expected.
   - Validate required keys directly: missing required keys must raise an
     informative exception specifying which required keys are missing.
   - Single Source of Truth in `config/config.yaml`: Shared pipeline settings
     (`material`, `model_type`, `use_sliding_window`, `use_film`,
     `apply_denoise`, `unified_model`, `source_tool`, `target_tool`,
     `example_region`, `sample_measurement_id`) must be defined once under
     `global:` (and `l2_weight` once under `cost:`) and propagated to stage
     sub-configs via `resolve_pipeline_config`, never duplicated with
     conflicting defaults in stage sections.
   - Canonical Variable & Column Naming Rules:
      1. Prefix Style for Pairing & Tool Context: Always use `source_*` and
         `target_*` prefixes in paired context (e.g., `source_tool`, `target_tool`,
         `source_tool_index`, `target_tool_index`, `target_measurement_id`,
         `target_spectrum_index`, `target_time`, `target_t7_code`). Do not use
         ambiguous suffix styles like `tool_target` or `measurement_id_target`.
      2. Feature Standardization: Continuous metadata features standardized via
         z-score must be named with `*_standardized` (e.g., `time_standardized`,
         `total_flux_single_die_standardized`), reserving `normalize_*`
         strictly for spectral intensity scaling.
      3. Prediction Variables: Use full `predicted_*` instead of abbreviated
         `pred_` (e.g., `predicted_flux`, `predicted_atomic_percentage`,
         `predicted_measurement_id`, `predicted_mean`, `predicted_std`).
      4. Percentage Metrics: Always use full `percentage` instead of `%` or
         `_pct` in column names and metric dictionary keys (e.g.,
         `peak_error_percentage`, `max_error_percentage`, `mape_train_percentage`,
         `source_atomic_percentage`, `target_atomic_percentage`,
         `predicted_atomic_percentage`, `diff_atomic_percentage`,
         `abs_diff_atomic_percentage`, `base_abs_diff_atomic_percentage`).
13. **Configuration Schema Validation & Fallback Elimination:**
   - Pipeline configuration in `config/config.yaml` is validated at load boundaries via Pydantic schema (`utility/config_schema.py`).
   - Maintain editor autocomplete and hover docs via `# yaml-language-server: $schema=config.schema.json`.
   - Avoid lengthy in-code default fallback blocks: put default values in YAML/schema. Check only for missing required keys (raise informative exceptions); allow extra keys cleanly without manual disallowed-key sets.
   - Avoid Redundant In-Code Type Casting and Static Property Checks:
     When configuration hyperparameters are validated and typed by Pydantic
     schemas (e.g. `TrainConfig`), downstream model constructors and layer
     initializers must NOT apply redundant type casts (`int()`, `float()`,
     `bool()`, `str()`) or duplicate static checks (positive or odd integers).
     In-code model validation must strictly focus on dynamic runtime invariants
     that Pydantic cannot evaluate at load time
     (e.g., `kernel_size <= n_points_generalized`).

# Code Modification Rules (CRITICAL - Strictly Enforced)
1. **Surgical Edits Only (Minimal Diff)**:
   - NEVER rewrite entire files or entire functions without explicit permission.
   - Modify ONLY the minimal lines of code directly required to fulfill the user's request.

2. **No Unsolicited Refactoring**:
   - Do NOT alter unrelated variable names, indentation, formatting, comments, or import orders.
   - Preserve all existing defensive checks, edge-case logic, error handling, and comments—even if you think they could be written more concisely.

3. **Strict Tool Usage & No Lazy Placeholders**:
   - When modifying existing files, you MUST prioritize targeted patch/diff tools (Search and Replace) over full-file overwrite tools.
   - NEVER output lazy truncation placeholders (e.g., `// ... rest of code ...` or `// ... existing implementation ...`) when writing to files.