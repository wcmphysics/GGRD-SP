"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt

from utility import (
    generate_pseudo_measurements,
    plot_regional_spectra,
    plot_tool_comparison,
)


def main() -> None:
    """Execute pseudo-measurement generation and demonstration visualization."""
    print("=== Generating GGRD-SP Pseudo-Measurements ===")
    ary_intensity, ary_energy, meta_df = generate_pseudo_measurements()

    print(f"Intensity array shape: {ary_intensity.shape}")
    print(f"Energy array shape:    {ary_energy.shape}")
    print(f"Total spectra:         {len(meta_df)}")
    print(
        f"Measurements per tool: {meta_df.groupby('tool')['measurement_id'].nunique().to_dict()}"
    )
    print("\nMetadata summary:")
    print(meta_df.info())
    print("\nFirst 5 metadata entries:")
    print(meta_df.head())

    print("\nGenerating demonstration plots...")
    # 1. Regional spectra for the first measurement session
    plot_regional_spectra(
        ary_energy,
        ary_intensity,
        meta_df,
        plot_config={"title": "Sample Regional Spectra (First Measurement Session)"},
    )

    # 2. Direct comparison between J4 and J5 for Al2p (Die 0)
    plot_tool_comparison(
        ary_energy,
        ary_intensity,
        meta_df,
        compare_config={"region": "Al2p", "die": 0},
    )

    print("Showing plots (close plot windows to finish execution)...")
    plt.show()


if __name__ == "__main__":
    main()