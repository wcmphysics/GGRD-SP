"""Matrix test runner for all 32 combinations of main.py parameters."""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any

import main


def run_all_combinations() -> list[dict[str, Any]]:
    models = ["resnet", "residual_unet", "unet", "deeplabv3"]
    sw_options = [False, True]
    film_options = [False, True]
    unified_options = [False, True]

    results: list[dict[str, Any]] = []
    combo_idx = 0
    total_combos = (
        len(models) * len(sw_options) * len(film_options) * len(unified_options)
    )

    print(f"Executing {total_combos} combinations of main.py...")
    print("-" * 80)
    header_fmt = (
        f"{'#':<3} | {'Model':<14} | {'SlidingWin':<10} | {'FiLM':<6} | "
        f"{'Unified':<8} | {'Status':<8} | {'Time (s)':<8}"
    )
    print(header_fmt)
    print("-" * 80)

    for model in models:
        for sw in sw_options:
            for film in film_options:
                for unified in unified_options:
                    combo_idx += 1
                    cfg = {
                        "model_type": model,
                        "use_sliding_window": sw,
                        "use_film": film,
                        "unified_model": unified,
                        "epochs": 1,
                        "measurements_per_tool": {"J4": 6, "J5": 8},
                        "kernel_size": 3,
                        "depth": 2,
                        "gp_num_iterations": 5,
                        "verbose": False,
                        "show_plots": False,
                    }

                    start_t = time.perf_counter()
                    status = "PASS"
                    err_msg = ""

                    try:
                        main.main(config_override=cfg)
                    except Exception as exc:
                        status = "FAIL"
                        tb_str = traceback.format_exc()
                        err_msg = f"{type(exc).__name__}: {exc}\n{tb_str}"

                    dur = time.perf_counter() - start_t
                    sw_str = "ON" if sw else "OFF"
                    film_str = "ON" if film else "OFF"
                    unified_str = "ON" if unified else "OFF"

                    row_str = (
                        f"{combo_idx:<3} | {model:<14} | {sw_str:<10} | "
                        f"{film_str:<6} | {unified_str:<8} | {status:<8} | "
                        f"{dur:<8.2f}"
                    )
                    print(row_str)
                    sys.stdout.flush()

                    results.append({
                        "idx": combo_idx,
                        "model": model,
                        "sliding_window": sw_str,
                        "film": film_str,
                        "unified": unified_str,
                        "status": status,
                        "duration_s": dur,
                        "error": err_msg,
                    })

    print("-" * 80)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"Summary: {passed}/{total_combos} passed, {failed} failed.")

    if failed > 0:
        print("\nErrors encountered:")
        for r in results:
            if r["status"] == "FAIL":
                title = (
                    f"--- Combo #{r['idx']} ({r['model']}, "
                    f"SW={r['sliding_window']}, FiLM={r['film']}, "
                    f"Unified={r['unified']}) ---"
                )
                print(title)
                print(r["error"])

    return results


if __name__ == "__main__":
    run_all_combinations()
