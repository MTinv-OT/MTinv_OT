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


def _normalize_station_indices(station_idx: Union[int, Sequence[int], np.ndarray]) -> list[int]:
    if isinstance(station_idx, (int, np.integer)):
        return [int(station_idx)]
    if isinstance(station_idx, Sequence) and not isinstance(station_idx, (str, bytes)):
        indices = [int(v) for v in station_idx]
        if not indices:
            raise ValueError("station_idx cannot be empty")
        return indices
    raise TypeError(f"station_idx must be an int or a sequence of ints, got {type(station_idx)!r}")


def plot_rho_fitting_comparison(
    freqs: np.ndarray,
    rho_true: np.ndarray,
    rho_obs: np.ndarray,
    rho_pred_ot: np.ndarray,
    rho_pred_mse: np.ndarray,
    *,
    station_idx: Union[int, Sequence[int], np.ndarray] = 0,
    station_label: Optional[Union[str, Sequence[str]]] = None,
    mode: str = "xy",
    log_x: bool = True,
    log_y: bool = True,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    rho_obs_no_shift: Optional[np.ndarray] = None,
) -> Union[plt.Axes, list[plt.Axes]]:
    """Four-line ρ_a comparison at one or more stations: true / obs / OT pred / MSE pred."""
    apply_plot_style()
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    station_indices = _normalize_station_indices(station_idx)

    def _slice(rho: np.ndarray, idx: int) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 2:
            return rho[:, idx]
        return rho.reshape(-1)

    created = ax is None
    if created:
        if len(station_indices) == 1:
            _, single_ax = plt.subplots(figsize=(7, 5))
            axes = [single_ax]
        else:
            fig, axes = plt.subplots(1, len(station_indices), figsize=(4.5 * len(station_indices), 5), squeeze=False)
            axes = list(axes.reshape(-1))
    else:
        if len(station_indices) != 1:
            raise ValueError("When ax is provided, station_idx must contain exactly one station")
        axes = [ax]

    comp, _, _ = _component_keys(mode)
    for i, st_idx in enumerate(station_indices):
        ax_i = axes[i]
        rho_t = _slice(rho_true, st_idx)
        rho_o = _slice(rho_obs, st_idx)
        rho_ot = _slice(rho_pred_ot, st_idx)
        rho_mse = _slice(rho_pred_mse, st_idx)
        rho_o_no_shift = _slice(rho_obs_no_shift, st_idx) if rho_obs_no_shift is not None else None

        ax_i.plot(freqs, rho_t, "k-", lw=2, label="True")
        if rho_o_no_shift is not None:
            ax_i.plot(freqs, rho_o_no_shift, "C3--", lw=1.5, label="Observed (before shift)")
        ax_i.plot(freqs, rho_o, "C2o", lw=1.5, label="Observed")
        ax_i.plot(freqs, rho_ot, "C0-", lw=2, label="OT inverted")
        ax_i.plot(freqs, rho_mse, "C1-", lw=2, label="MSE inverted")
        if log_x:
            ax_i.set_xscale("log")
            f_min, f_max = float(freqs.min()), float(freqs.max())
            ax_i.set_xlim(f_max, f_min)  # high frequency on the left, low on the right (same as plot_data_fitting)
        if log_y:
            ax_i.set_yscale("log")
        ax_i.set_xlabel("Frequency (Hz)")
        ax_i.set_ylabel("Apparent resistivity (Ω·m)")

        if station_label is None:
            st = f"S{st_idx + 1}"
        elif isinstance(station_label, Sequence) and not isinstance(station_label, (str, bytes)):
            st = station_label[i]
        else:
            st = str(station_label)
        ax_i.set_title(f"ρ_a fitting ({comp}) @ {st}")
        ax_i.grid(True, which="both", alpha=0.3)
        ax_i.legend(loc="upper right")

    if created:
        plt.tight_layout()
        if show:
            plt.show()
    return axes[0] if len(axes) == 1 else axes


def plot_rho_fitting_from_npz(
    npz_ot: Union[str, Path],
    npz_mse: Union[str, Path],
    *,
    station_idx: Union[int, Sequence[int], np.ndarray] = 0,
    mode: str = "xy",
    **kwargs,
) -> Union[plt.Axes, list[plt.Axes]]:
    """Load two ``apparent_resistivity.npz`` files and plot four-line comparison."""
    d_ot = np.load(npz_ot)
    d_mse = np.load(npz_mse)
    _, true_key, obs_key = _component_keys(mode)
    pred_key = f"pred_{_component_keys(mode)[0]}"
    station_label = kwargs.pop("station_label", None)
    station_indices = _normalize_station_indices(station_idx)
    if station_label is None and "stations" in d_ot:
        station_labels = []
        stations = np.asarray(d_ot["stations"]).reshape(-1)
        for st_idx in station_indices:
            st_km = float(stations[st_idx]) / 1000.0
            station_labels.append(f"S{st_idx + 1} ({st_km:.1f} km)")
        station_label = station_labels if len(station_labels) > 1 else station_labels[0]
    # rho_obs_no_shift plotting was removed
    return plot_rho_fitting_comparison(
        freqs=d_ot["freqs"],
        rho_true=d_ot[true_key],
        rho_obs=d_ot[obs_key],
        rho_pred_ot=d_ot[pred_key],
        rho_pred_mse=d_mse[pred_key],
        station_idx=station_indices,
        station_label=station_label,
        mode=mode,
        # do not pass rho_obs_no_shift
        **kwargs,
    )

def plot_rho_fitting_from_single_npz(
    npz_path: Union[str, Path],
    *,
    station_idx: Union[int, Sequence[int], np.ndarray] = 0,
    mode: str = "xy",
    include_pred: bool = True,
    **kwargs,
) -> Union[plt.Axes, list[plt.Axes]]:
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

    station_indices = _normalize_station_indices(station_idx)
    rho_obs_no_shift = _col(obs_no_shift_key) if obs_no_shift_key in d else None
    return plot_rho_fitting_comparison(
        freqs=freqs,
        rho_true=rho_true,
        rho_obs=rho_obs,
        rho_pred_ot=rho_ot,
        rho_pred_mse=rho_mse,
        station_idx=station_indices,
        mode=mode,
        ax=created_ax,
        rho_obs_no_shift=rho_obs_no_shift,
        **kwargs,
    )
