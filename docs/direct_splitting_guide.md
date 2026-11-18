# Direct Splitting Method Guide

GGRD-SP supports explicit assignment of paired measurement sessions into `train`, `val`, and `test` splits using the `direct` splitting method.

## Configuration

In `config/config.yaml`, set `splitting.method` to `direct` and provide a `direct_assignment` mapping of paired source `measurement_id`s:

```yaml
splitting:
  method: direct
  direct_assignment:
    train:
      - NMG_M_J4_00000
      - NMG_M_J4_00001
    val:
      - NMG_M_J4_00002
    test:
      - NMG_M_J4_00003
```

## Rules and Behavior

1. **Source Measurement IDs**: List the paired source tool `measurement_id`s under `train`, `val`, and `test`. Their paired target measurements (`target_measurement_id`) and all dies and regions within those sessions automatically receive the same split label.
2. **Mutually Exclusive**: Each measurement ID can appear in at most one split. Overlapping IDs across `train`, `val`, and `test` raise a `ValueError`.
3. **Exhaustive Over Paired Measurements**: The union of `train`, `val`, and `test` must exactly match the set of all paired source measurement IDs. Missing paired sessions or unknown/unpaired IDs raise a `ValueError`.
4. **Unpaired Measurements**: Any unpaired measurements are automatically assigned `split = "none"`.
