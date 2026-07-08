"""Apparent-resistivity fitting curves (obs / true / OT / MSE)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_plot_style


def _component_keys(mode: str) -> tuple[str, str, str]:
    m = mode.strip().lower()
    if m in ("xy", "te", "rhoxy"):
        return "rhoxy", "true_rhoxy", "obs_rhoxy"
    if m in ("yx", "tm", "rhoyx"):
        return "rhoyx", "true_rhoyx", "obs_rhoyx"
    raise ValueError(f"mode must be 'xy' or 'yx', got {mode!r}")


def plot_rho_fitting_comparison(
    freqs: np.ndarray,
    rho_true: np.ndarray,
    rho_obs: np.ndarray,
    rho_pred_ot: np.ndarray,
    rho_pred_mse: np.ndarray,
    *,
    station_idx: int = 0,
    station_label: Optional[str] = None,
    mode: str = "xy",
    log_x: bool = True,
    log_y: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    rho_obs_no_shift: Optional[np.ndarray] = None,
) -> plt.Axes:
    """Four-line ρ_a comparison at one station: true / obs / OT pred / MSE pred."""
    apply_plot_style()
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    def _slice(rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 2:
            return rho[:, station_idx]
        return rho.reshape(-1)

    rho_t = _slice(rho_true)
    rho_o = _slice(rho_obs)
    rho_ot = _slice(rho_pred_ot)
    rho_mse = _slice(rho_pred_mse)
    rho_o_no_shift = _slice(rho_obs_no_shift) if rho_obs_no_shift is not None else None

    ax.plot(freqs, rho_t, "k-", lw=2, label="True")
    if rho_o_no_shift is not None:
        ax.plot(freqs, rho_o_no_shift, "C3--", lw=1.5, label="Observed (before shift)")
    ax.plot(freqs, rho_o, "C2o", lw=1.5, label="Observed")
    ax.plot(freqs, rho_ot, "C0-", lw=2, label="OT inverted")
    ax.plot(freqs, rho_mse, "C1-", lw=2, label="MSE inverted")
    if log_x:
        ax.set_xscale("log")
        f_min, f_max = float(freqs.min()), float(freqs.max())
        ax.set_xlim(f_max, f_min)  # 左高频、右低频，与 plot_data_fitting 一致
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Apparent resistivity (Ω·m)")
    comp, _, _ = _component_keys(mode)
    # 台站编号统一为 1-based：station_idx=0 → S1（与 inv.plot_data_fitting 一致）
    st = station_label if station_label is not None else f"S{station_idx + 1}"
    ax.set_title(f"ρ_a fitting ({comp}) @ {st}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if created:
        plt.tight_layout()
        if show:
            plt.show()
    return ax


def plot_rho_fitting_from_npz(
    npz_ot: Union[str, Path],
    npz_mse: Union[str, Path],
    *,
    station_idx: int = 0,
    mode: str = "xy",
    **kwargs,
) -> plt.Axes:
    """Load two ``apparent_resistivity.npz`` files and plot four-line comparison."""
    d_ot = np.load(npz_ot)
    d_mse = np.load(npz_mse)
    _, true_key, obs_key = _component_keys(mode)
    pred_key = f"pred_{_component_keys(mode)[0]}"
    obs_no_shift_key = f"obs_no_shift_{obs_key[4:]}"
    station_label = kwargs.pop("station_label", None)
    if station_label is None and "stations" in d_ot:
        st_km = float(np.asarray(d_ot["stations"]).reshape(-1)[station_idx]) / 1000.0
        station_label = f"S{station_idx + 1} ({st_km:.1f} km)"
    rho_obs_no_shift = d_ot[obs_no_shift_key] if obs_no_shift_key in d_ot else None
    return plot_rho_fitting_comparison(
        freqs=d_ot["freqs"],
        rho_true=d_ot[true_key],
        rho_obs=d_ot[obs_key],
        rho_pred_ot=d_ot[pred_key],
        rho_pred_mse=d_mse[pred_key],
        station_idx=station_idx,
        station_label=station_label,
        mode=mode,
        rho_obs_no_shift=rho_obs_no_shift,
        **kwargs,
    )


def plot_rho_fitting_from_single_npz(
    npz_path: Union[str, Path],
    *,
    station_idx: int = 0,
    mode: str = "xy",
    include_pred: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot true / obs / pred from one ``apparent_resistivity.npz`` (single run)."""
    d = np.load(npz_path)
    comp, true_key, obs_key = _component_keys(mode)
    pred_key = f"pred_{comp}"
    apply_plot_style()
    freqs = d["freqs"]

    def _col(key: str) -> np.ndarray:
        if key not in d:
            raise KeyError(f"{key} not in {npz_path}")
        return d[key]

    rho_true = _col(true_key)
    rho_obs = _col(obs_key)
    obs_no_shift_key = f"obs_no_shift_{obs_key[4:]}"
    created_ax = kwargs.pop("ax", None)
    if include_pred and pred_key in d:
        rho_pred = _col(pred_key)
        rho_ot = rho_pred
        rho_mse = rho_pred
    else:
        rho_ot = rho_obs
        rho_mse = rho_obs

    rho_obs_no_shift = _col(obs_no_shift_key) if obs_no_shift_key in d else None
    return plot_rho_fitting_comparison(
        freqs=freqs,
        rho_true=rho_true,
        rho_obs=rho_obs,
        rho_pred_ot=rho_ot,
        rho_pred_mse=rho_mse,
        station_idx=station_idx,
        mode=mode,
        ax=created_ax,
        rho_obs_no_shift=rho_obs_no_shift,
        **kwargs,
    )
