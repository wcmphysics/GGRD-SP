# Inference Guide: Cross-Tool Spectral Transfer on New Datasets

This guide details how to use a trained model (after pipeline Stage 12) to transfer a newly collected dataset (`df_meta_new`, `ary_intensity_new`, `ary_energy_new`) from a source tool to a target tool.

---

## 1. Prerequisites and Setup

Ensure that:
1. The model training pipeline has completed through Stage 9 (or Stage 12), yielding the trained model dictionary: `training_results["models"]`.
2. The pipeline configuration dictionary `config` (as loaded by `utility.load_config("config/config.yaml")`) is available.
3. The new dataset matches the spectral structure expected by the pipeline:
   - `ary_intensity_new`: 2D array of shape `(N_rows, N_points)`.
   - `ary_energy_new`: 2D array of shape `(N_rows, N_points)` in eV ascending.
   - `df_meta_new`: DataFrame with one row per regional spectrum, containing `tool`, `measurement_id`, `die`, and `region`.

---

## 2. Preprocessing and Normalization (Stage 5 Consistency)

Inference inputs must undergo preprocessing and normalization consistent with the training stage.

```python
import numpy as np
import utility
import models

# 1. Align spectrum indexing and assign split
df_meta_new = df_meta_new.copy()
df_meta_new["spectrum_index"] = np.arange(len(df_meta_new))

# When evaluating standalone new data, assign split to 'test'
df_meta_new["split"] = "test"
preprocess_cfg = {**config, "fit_split": "test"}

# 2. Compute regional and die-level physical metrics (mean intensity, total flux)
df_meta_new, _, _ = utility.preprocess(
    ary_intensity_new,
    ary_energy_new,
    df_meta_new,
    config=preprocess_cfg,
)

# 3. Apply normalization
norm_mode = config.get("normalization", {}).get("normalization_mode")
eps = float(config.get("normalization", {}).get("eps", 1e-6))

if norm_mode == "by_regional_max_intensity_of_source":
    # For standalone source spectra without paired target spectra, scale each spectrum by its own max
    raw_scales = np.max(np.abs(ary_intensity_new), axis=1, keepdims=True)
    scales = np.where(raw_scales < eps, eps, raw_scales)
    ary_intensity_norm = ary_intensity_new / scales
    ary_energy_norm = ary_energy_new.copy()
    df_meta_norm = df_meta_new.copy()
    df_meta_norm["normalization_scale"] = scales.ravel()
else:
    ary_intensity_norm, ary_energy_norm, df_meta_norm = utility.normalize_spectral_data(
        ary_intensity_new,
        ary_energy_new,
        df_meta_new,
        config["normalization"],
    )
```

---

## 3. Running Prediction (`models.predict_spectra`)

Configure and invoke `predict_spectra` from `models.inference`:

```python
# Assemble prediction configuration dictionary
prediction_config = {
    **config.get("prediction", {}),
    "source_tool": config["global"]["source_tool"],
    "target_tool": config["global"]["target_tool"],
    "use_sliding_window": config["global"].get("use_sliding_window", False),
    "window_size": config["dataset"].get("window_size", 15),
    "stride": config["dataset"].get("sliding_stride", 7),
    "clamp_non_negative": True,
}

# Run inference using the trained models dictionary
# (training_results["models"] from Stage 9)
ary_intensity_pred, ary_energy_pred, df_meta_pred = models.predict_spectra(
    models=training_results["models"],
    data=(ary_intensity_norm, ary_energy_norm, df_meta_norm),
    config=prediction_config,
)
```

---

## 4. Recovering Physical Intensities (Inversion)

`ary_intensity_pred` is produced in the normalized scale. To restore physical intensity scales:

### Mode: `by_regional_max_intensity_of_source`
```python
source_max = np.max(ary_intensity_new, axis=1, keepdims=True)
ary_intensity_pred_raw = ary_intensity_pred * source_max
```

### Mode: `by_total_flux_single_die`
```python
die_flux = df_meta_pred["total_flux_single_die"].to_numpy()[:, None]
ary_intensity_pred_raw = ary_intensity_pred * die_flux
```

---

## 5. Downstream Quantification (Stage 12)

Quantify the transferred spectra into per-die atomic concentration percentages using Shirley background subtraction and RSF sensitivity factors:

```python
df_per_die_pred, df_summary_pred = utility.calculate_atomic_percentages(
    ary_energy_pred,
    ary_intensity_pred,
    df_meta_pred,
    config=config["quantification"],
)
```
