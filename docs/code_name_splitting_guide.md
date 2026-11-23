# Code-Name Splitting Method Guide

GGRD-SP supports assigning groups of measurement sessions to `train`, `val`, and `test` splits using the `code_name` splitting method. The code name is extracted as the leading `_`-delimited token of each `measurement_id` (e.g., `PMGA_M_J4_00001` → `PMGA`).

## Configuration

In `config/config.yaml`, set `splitting.method` to `code_name` and provide `code_name_assignment`:

```yaml
splitting:
  method: code_name
  train_ratio: 0.5
  val_ratio: 0.2
  test_ratio: 0.3
  seed: 42
  code_name_assignment:
    train:
      - PMGA
    val:
      - PMGA
    test:
      - PMGB
```

## Rules and Behavior

1. **Exact Leading Token**: Code names are matched by exact leading `_` token (`measurement_id.split("_", 1)[0]`), so `PMG` and `PMGA` never collide.
2. **Single-Split Assignment**: If a code name appears in only one split (e.g., `PMGB` only in `test`), 100% of its paired sessions are assigned to that split. Its split ratio may be `0.0` if no multi-split code uses that split.
3. **Multi-Split Stratified Partitioning**: If a code name is assigned to 2 or 3 splits (e.g., `PMGA` in `train` and `val`), its paired sessions are randomly partitioned across those splits using the renormalized ratios (`train_ratio / (train_ratio + val_ratio)` vs `val_ratio / (train_ratio + val_ratio)`). Each code name is split independently (stratified).
4. **Strict Coverage**: Every code name found in paired source measurements must be assigned in `code_name_assignment`, and every configured code name must exist in the paired measurements; otherwise a `ValueError` is raised.
5. **Paired Target & Unpaired Sessions**: Paired target measurements, dies, and regions inherit their source session's split; unpaired sessions receive `split = "none"`.
