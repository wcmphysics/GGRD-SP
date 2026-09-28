"""Main entry point for GGRD-SP project."""

from __future__ import annotations

import matplotlib.pyplot as plt

from utility import (
    calculate_atomic_percentages,
    generate_pseudo_measurements,
    pair_source_target_spectra,
    plot_pairing_timeline,
    plot_regional_spectra,
    plot_shirley_background,
    plot_tool_comparison,
)


def main() -> None:
    """Execute pseudo-measurement generation, pairing, quantification, and visualization."""
    
    # global setup
    source_tool = 'J4'
    target_tool = 'H1'
    example_region = 'Ti2p'
    sample_meas_id = "M_J4_003" # for at% calculation
    
    
    
    #
    # generate psuedo measurement
    #    
    print("=== Part 1: Generating Pseudo-Measurements ===")
    ary_intensity, ary_energy, meta_df = generate_pseudo_measurements(
        {
        "material": "NMG",
        "regions": ["Al2p", "Ti2p", "O1s", "C1s", "Cl2p"],
        "n_points": 100,
        "measurements_per_tool": {source_tool: 15, target_tool: 20},
        "interval_hours_range": (4.0, 24.0),
        "tool_offsets": {
            source_tool: {"shift_ev": 0.0, "scale": 1.0},
            target_tool: {"shift_ev": 0.2, "scale": 1.1},
        },
        "die_variation_std": 0.03,
        "noise_relative_std": 0.015,
        "seed": None,
        }
    )

    print(f"Intensity array shape: {ary_intensity.shape}")
    print(f"Energy array shape:    {ary_energy.shape}")
    print(f"Total spectra:         {len(meta_df)}")
    print(
        f"Measurements per tool: {meta_df.groupby('tool')['measurement_id'].nunique().to_dict()}"
    )





    #
    # Find pairing measurement
    #
    print("\n=== Part 2: Pairing Source and Target Spectra ===")
    meta_df = pair_source_target_spectra(meta_df, 
        {
        "source_tool": source_tool,
        "target_tool": target_tool,
        "time_threshold_hours": 8.0,
        "method": "optimal",
        "material": None,
        }
    )

    paired_src = meta_df[
        (meta_df["tool"] == source_tool) & meta_df["measurement_id_target"].notna()
    ]
    total_src_sessions = meta_df[meta_df["tool"] == source_tool]["measurement_id"].nunique()
    paired_src_sessions = paired_src["measurement_id"].nunique()

    print(
        f"Paired measurement sessions / All measurement sessions: {paired_src_sessions}/{total_src_sessions}"
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
    
    
    
    
    
    

    #
    # Calculate atomic percentage
    #
    print("\n=== Part 5: Calculating Atomic Percentage (Shirley Integration) ===")

    df_per_die, df_summary = calculate_atomic_percentages(
        ary_energy,
        ary_intensity,
        meta_df,
        config={"measurement_id": sample_meas_id},
    )

    print(f"Atomic percentage per die for session '{sample_meas_id}' (first 3 dies):")
    print(df_per_die.head(3).to_string(index=False))
    print(f"\nSummary statistics across all 9 dies for session '{sample_meas_id}':")
    print(df_summary.to_string(index=False))





    
    #
    # Create plots
    # 
    

    print("\nGenerating demonstration plots...")
    # 1. Regional spectra for the first measurement session
    # plot_regional_spectra(
    #     ary_energy,
    #     ary_intensity,
    #     meta_df,
    #     plot_config={ 'filters':{'tool': source_tool, 'region': example_region}, "title": "Sample Regional Spectra (First Measurement Session)"},
    # )

    # 2. Direct tool comparison between tools for a region (Die 0)
    plot_tool_comparison(
        ary_energy,
        ary_intensity,
        meta_df,
        compare_config={"tools": (source_tool, target_tool), "region": example_region, "die": 0},
    )

    # 3. 1-to-1 Measurement pairing timeline
    plot_pairing_timeline(meta_df, plot_config={'source_tool':source_tool, 'target_tool':target_tool})

    # 4. Shirley background subtraction in multi-region grid mode (all 5 regions for Die 0)
    plot_shirley_background(
        ary_energy,
        ary_intensity,
        meta_df,
        plot_config={"measurement_id": sample_meas_id, "die": 0},
    )

    print("Showing plots (close plot windows to finish execution)...")
    plt.show()


if __name__ == "__main__":
    main()