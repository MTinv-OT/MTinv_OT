"""Apparent-resistivity pseudosection plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_plot_style


def _rho_component(mode: str) -> str:
    m = mode.strip().lower()
    if m in ("xy", "te", "rhoxy"):
        return "rhoxy"
    if m in ("yx", "tm", "rhoyx"):
        return "rhoyx"
    raise ValueError(f"mode must be 'xy' or 'yx', got {mode!r}")


def _valid_rho_values(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    return rho[np.isfinite(rho) & (rho > 0)]


def _auto_vlim(
    rho: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentiles: Tuple[float, float] = (2.0, 98.0),
) -> Tuple[float, float]:
    valid = _valid_rho_values(rho)
    if valid.size == 0:
        return 1.0, 100.0
    lo, hi = float(percentiles[0]), float(percentiles[1])
    vmin_use = float(vmin) if vmin is not None else float(np.nanpercentile(valid, lo))
    vmax_use = float(vmax) if vmax is not None else float(np.nanpercentile(valid, hi))
    if vmax_use <= vmin_use:
        vmax_use = vmin_use * 1.01
    return vmin_use, vmax_use

def plot_apparent_resistivity_pseudosection(
    freqs: np.ndarray,
    stations: np.ndarray,
    rho: np.ndarray,
    *,
    cmap: str = "jet_r",
    log_y: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vlim_percentiles: Tuple[float, float] = (2.0, 98.0),
    ax: Optional[plt.Axes] = None,
    add_colorbar: bool = True,
    show: bool = True,
    title: Optional[str] = None,
    label_fontsize: float = 18,
    tick_fontsize: float = 20,
    title_fontsize: float = 18,
    colorbar_label_fontsize: float = 16,
    colorbar_tick_fontsize: float = 14,
) -> plt.Axes:
    """Contour pseudosection: stations (x) vs frequency/period (y)."""
    apply_plot_style()

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    stations = np.asarray(stations, dtype=float).reshape(-1)
    rho = np.asarray(rho, dtype=float)

    if rho.ndim != 2:
        raise ValueError(
            f"rho must be 2D (n_freq, n_station), got shape {rho.shape}"
        )

    expected_shape = (freqs.size, stations.size)
    if rho.shape != expected_shape:
        raise ValueError(
            f"rho shape must be {expected_shape}, got {rho.shape}"
        )

    vmin_use, vmax_use = _auto_vlim(
        rho,
        vmin=vmin,
        vmax=vmax,
        percentiles=vlim_percentiles,
    )

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    st_km = stations / 1000.0
    periods = 1.0 / np.clip(freqs, 1e-30, None)
    X, Y = np.meshgrid(st_km, periods)

    data = np.ma.masked_invalid(rho)

    pcm = ax.pcolormesh(
        X,
        Y,
        data,
        shading="auto",
        cmap=cmap,
        vmin=vmin_use,
        vmax=vmax_use,
    )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(
        "Distance (km)",
        fontsize=label_fontsize,
    )
    ax.set_ylabel(
        "Period (s)",
        fontsize=label_fontsize,
    )
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=tick_fontsize,
    )

    if title:
        ax.set_title(
            title,
            fontsize=title_fontsize,
        )

    ax.invert_yaxis()

    if add_colorbar:
        cb = ax.figure.colorbar(pcm, ax=ax)
        cb.set_label(
            "ρₐ (Ω·m)",
            fontsize=colorbar_label_fontsize,
        )
        cb.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
        )

    if created:
        ax.figure.tight_layout()

        if show:
            plt.show()

    return ax

def plot_pseudosection_from_npz(
    npz_path: Union[str, Path],
    *,
    field: str = "obs_rhoxy",
    mode: str = "xy",
    **kwargs,
) -> plt.Axes:
    """Plot one field from ``apparent_resistivity.npz``."""
    d = np.load(npz_path)
    comp = _rho_component(mode)
    key = field if field in d else f"{field.split('_')[0]}_{comp}"
    if key not in d:
        key = f"obs_{comp}"
    title = kwargs.pop("title", key)
    return plot_apparent_resistivity_pseudosection(
        d["freqs"], d["stations"], d[key], title=title, **kwargs
    )
    
def plot_inversion_pseudosection_comparison(
    freqs: np.ndarray,
    stations: np.ndarray,
    rho_dict: Dict[str, np.ndarray],
    *,
    mode: str = "xy",
    cmap: str = "jet_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vlim_percentiles: Tuple[float, float] = (2.0, 98.0),
    suptitle: Optional[str] = None,
    show: bool = True,
    label_fontsize: float = 18,
    tick_fontsize: float = 20,
    title_fontsize: float = 18,
    suptitle_fontsize: float = 18,
    colorbar_label_fontsize: float = 16,
    colorbar_tick_fontsize: float = 14,
) -> np.ndarray:
    """Plot multiple apparent-resistivity panels with one color scale."""
    apply_plot_style()

    if not rho_dict:
        raise ValueError("rho_dict must contain at least one panel")

    valid_arrays = [
        _valid_rho_values(rho)
        for rho in rho_dict.values()
    ]
    valid_arrays = [
        values
        for values in valid_arrays
        if values.size > 0
    ]

    if vmin is None or vmax is None:
        if valid_arrays:
            stacked = np.concatenate(valid_arrays)
            vmin_u, vmax_u = _auto_vlim(
                stacked,
                vmin=vmin,
                vmax=vmax,
                percentiles=vlim_percentiles,
            )
        else:
            vmin_u, vmax_u = 1.0, 100.0
    else:
        vmin_u, vmax_u = float(vmin), float(vmax)

    freqs_a = np.asarray(freqs, dtype=float).reshape(-1)
    stations_a = np.asarray(stations, dtype=float).reshape(-1)

    st_km = stations_a / 1000.0
    periods = 1.0 / np.clip(freqs_a, 1e-30, None)
    X, Y = np.meshgrid(st_km, periods)

    n = len(rho_dict)

    fig = plt.figure(
        figsize=(5 * n + 1, 5),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        n + 1,
        width_ratios=[1.0] * n + [0.05],
        wspace=0.15,
    )

    axes = [
        fig.add_subplot(gs[0, i])
        for i in range(n)
    ]

    last_pcm = None

    for ax, (label, rho) in zip(axes, rho_dict.items()):
        rho_a = np.asarray(rho, dtype=float)

        expected_shape = (freqs_a.size, stations_a.size)
        if rho_a.shape != expected_shape:
            raise ValueError(
                f"Panel {label!r}: expected shape "
                f"{expected_shape}, got {rho_a.shape}"
            )

        data = np.ma.masked_invalid(rho_a)

        last_pcm = ax.pcolormesh(
            X,
            Y,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin_u,
            vmax=vmax_u,
        )

        ax.set_yscale("log")
        ax.set_xlabel(
            "Distance (km)",
            fontsize=label_fontsize,
        )
        ax.set_ylabel(
            "Period (s)",
            fontsize=label_fontsize,
        )
        ax.set_title(
            label,
            fontsize=title_fontsize,
        )
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=tick_fontsize,
        )
        ax.invert_yaxis()

    if last_pcm is not None:
        cax = fig.add_subplot(gs[0, -1])
        cb = fig.colorbar(last_pcm, cax=cax)
        cb.set_label(
            "ρₐ (Ω·m)",
            fontsize=colorbar_label_fontsize,
        )
        cb.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
        )

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=suptitle_fontsize,
        )

    if show:
        plt.show()

    return np.asarray(axes, dtype=object)

def plot_ot_mse_pseudosection_from_npz(
    npz_ot: Union[str, Path],
    npz_mse: Union[str, Path],
    *,
    mode: str = "xy",
    cmap: str = "jet_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vlim_percentiles: Tuple[float, float] = (2.0, 98.0),
    observed_vlim_percentiles: Tuple[float, float] = (2.0, 98.0),
    show: bool = True,
) -> dict:
    """OT/MSE comparison pseudosections.

    - Figure 1: True / OT inverted / MSE inverted — **shared** color scale (from True by default).
    - Figure 2: Observed — **separate** figure with its own auto color scale.
    """
    d0 = np.load(npz_ot)
    d1 = np.load(npz_mse)
    comp = _rho_component(mode)
    freqs = d0["freqs"]
    stations = d0["stations"]
    rho_true = d0[f"true_{comp}"]
    rho_obs = d0[f"obs_{comp}"]
    rho_ot = d0[f"pred_{comp}"]
    rho_mse = d1[f"pred_{comp}"]

    vmin_u, vmax_u = _auto_vlim(
        rho_true, vmin=vmin, vmax=vmax, percentiles=vlim_percentiles
    )
    inv_dict = {
        "True": rho_true,
        "OT inverted": rho_ot,
        "MSE inverted": rho_mse,
    }
    axes_inv = plot_inversion_pseudosection_comparison(
        freqs,
        stations,
        inv_dict,
        mode=mode,
        cmap=cmap,
        vmin=vmin_u,
        vmax=vmax_u,
        suptitle=f"ρ_a pseudosection ({comp}) — True / OT / MSE (shared scale)",
        show=False,
    )

    ax_obs = plot_apparent_resistivity_pseudosection(
        freqs,
        stations,
        rho_obs,
        cmap=cmap,
        vlim_percentiles=observed_vlim_percentiles,
        show=False,
    )
    ax_obs.figure.suptitle(f"ρ_a pseudosection ({comp}) — Observed (own scale)", fontsize=18, y=1.05)

    if show:
        plt.show()
    return {"inversion_axes": axes_inv, "observed_axes": ax_obs}
