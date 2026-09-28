"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt

from utility import (
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_pairing_timeline,
    plot_regional_spectra,
    plot_tool_comparison,
)


def main() -> None:
    """Execute pseudo-measurement generation, pairing, and visualization workflow."""
    print("=== Part 1: Generating Pseudo-Measurements ===")
    ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

    print(f"Intensity array shape: {ary_intensity.shape}")
    print(f"Energy array shape:    {ary_energy.shape}")
    print(f"Total spectra:         {len(meta_df)}")
    print(
        f"Measurements per tool: {meta_df.groupby('tool')['measurement_id'].nunique().to_dict()}"
    )

    print("\n=== Part 2: Pairing Source (J4) and Target (J5) Spectra ===")
    meta_df = pair_source_target_spectra(meta_df)

    paired_src = meta_df[
        (meta_df["tool"] == "J4") & meta_df["measurement_id_target"].notna()
    ]
    total_src_sessions = meta_df[meta_df["tool"] == "J4"]["measurement_id"].nunique()
    paired_src_sessions = paired_src["measurement_id"].nunique()

    print(
        f"Paired measurement sessions (within 12h): {paired_src_sessions}/{total_src_sessions}"
    )
    print("\nUpdated metadata summary with pairing columns:")
    print(meta_df.info())
    print("\nSample paired source rows:")
    cols_to_show = [
        "spectrum_index",
        "tool",
        "measurement_id",
        "die",
        "region",
        "tool_target",
        "measurement_id_target",
        "spectrum_index_target",
        "time_diff_hours",
    ]
    print(paired_src[cols_to_show].head(5))

    print("\nGenerating demonstration plots...")
    # 1. Regional spectra for the first measurement session
    plot_regional_spectra(
        ary_energy,
        ary_intensity,
        meta_df,
        plot_config={"title": "Sample Regional Spectra (First Measurement Session)"},
    )

    # 2. Direct tool comparison between J4 and J5 for Al2p (Die 0)
    plot_tool_comparison(
        ary_energy,
        ary_intensity,
        meta_df,
        compare_config={"region": "Al2p", "die": 0},
    )

    # 3. 1-to-1 Measurement pairing timeline
    plot_pairing_timeline(meta_df)

    print("Showing plots (close plot windows to finish execution)...")
    plt.show()


if __name__ == "__main__":
    main()