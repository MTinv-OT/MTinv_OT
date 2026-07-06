"""Cross-run comparison plots (OT vs MSE, etc.)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._style import apply_plot_style


def _history_to_arrays(
    history: Union[Sequence[Dict[str, Any]], pd.DataFrame],
) -> Dict[str, np.ndarray]:
    if isinstance(history, pd.DataFrame):
        df = history
    else:
        df = pd.DataFrame(list(history))
    if df.empty:
        raise ValueError("history is empty")

    def col(name: str, default=np.nan) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(df), default, dtype=float)
        return np.asarray(df[name], dtype=float)

    return {
        "epoch": np.asarray(df.get("epoch", np.arange(len(df))), dtype=float),
        "misfit": col("misfit"),
        "data_loss": col("data_loss"),
        "ot_distance": col("ot_distance"),
    }


def plot_ot_mse_convergence(
    history_ot: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    history_mse: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    *,
    target_misfit: float = 1.05,
    log_y: bool = True,
    show: bool = True,
) -> plt.Axes:
    """Plot RMS χ² and monitored OT distance for OT/MSE runs on **one** figure.

    RMS χ² uses the left y-axis; OT distance (view-only monitor, logged in both runs)
    uses the right y-axis. MSE inversion often lowers RMS while OT distance rises —
    both curves are shown so that trade-off is visible.
    """
    apply_plot_style()
    ot = _history_to_arrays(history_ot)
    mse = _history_to_arrays(history_mse)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(ot["epoch"], ot["misfit"], "C0-", lw=2, label="OT run — RMS χ²")
    ax.plot(mse["epoch"], mse["misfit"], "C1-", lw=2, label="MSE run — RMS χ²")
    ax.axhline(target_misfit, color="k", ls="--", lw=1, label=f"Target ({target_misfit})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMS χ²")
    ax.set_title("RMS χ² & OT distance (monitor)")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    has_ot_dist = np.any(np.isfinite(ot["ot_distance"])) or np.any(np.isfinite(mse["ot_distance"]))
    if has_ot_dist:
        ax_ot = ax.twinx()
        if np.any(np.isfinite(ot["ot_distance"])):
            ax_ot.plot(
                ot["epoch"],
                ot["ot_distance"],
                "C0--",
                lw=1.8,
                alpha=0.85,
                label="OT run — OT distance",
            )
        if np.any(np.isfinite(mse["ot_distance"])):
            ax_ot.plot(
                mse["epoch"],
                mse["ot_distance"],
                "C1--",
                lw=1.8,
                alpha=0.85,
                label="MSE run — OT distance",
            )
        ax_ot.set_ylabel("OT distance (6D, scaled)", color="0.35")
        if log_y:
            ax_ot.set_yscale("log")
        lines_l, lab_l = ax.get_legend_handles_labels()
        lines_r, lab_r = ax_ot.get_legend_handles_labels()
        ax.legend(lines_l + lines_r, lab_l + lab_r, loc="best", fontsize=9)
    else:
        ax.legend(loc="best")

    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_ot_mse_convergence_from_dirs(
    run_dir_ot: Union[str, Path],
    run_dir_mse: Union[str, Path],
    **kwargs,
) -> plt.Axes:
    """Load ``history.csv`` from two experiment folders and plot convergence."""
    ot_csv = Path(run_dir_ot) / "history.csv"
    mse_csv = Path(run_dir_mse) / "history.csv"
    return plot_ot_mse_convergence(
        pd.read_csv(ot_csv),
        pd.read_csv(mse_csv),
        **kwargs,
    )
