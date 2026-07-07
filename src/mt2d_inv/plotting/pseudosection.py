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
) -> plt.Axes:
    """Contour pseudosection: stations (x) vs frequency/period (y)."""
    apply_plot_style()
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    stations = np.asarray(stations, dtype=float).reshape(-1)
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2D (n_freq, n_station), got shape {rho.shape}")

    vmin_use, vmax_use = _auto_vlim(
        rho, vmin=vmin, vmax=vmax, percentiles=vlim_percentiles
    )

    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    st_km = stations / 1000.0
    periods = 1.0 / np.clip(freqs, 1e-30, None)
    X, Y = np.meshgrid(st_km, periods)
    data = np.ma.masked_invalid(rho)
    pcm = ax.pcolormesh(
        X, Y, data, shading="auto", cmap=cmap, vmin=vmin_use, vmax=vmax_use
    )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Period (s)")
    ax.invert_yaxis()
    if add_colorbar:
        plt.colorbar(pcm, ax=ax, label="ρ_a (Ω·m)")
    if created:
        plt.tight_layout()
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
) -> np.ndarray:
    """Plot multiple ρ_a panels side by side with a shared color scale."""
    apply_plot_style()
    if vmin is None or vmax is None:
        stacked = np.concatenate([_valid_rho_values(r) for r in rho_dict.values()])
        if stacked.size == 0:
            vmin_u, vmax_u = 1.0, 100.0
        else:
            vmin_u, vmax_u = _auto_vlim(
                stacked, vmin=vmin, vmax=vmax, percentiles=vlim_percentiles
            )
    else:
        vmin_u, vmax_u = float(vmin), float(vmax)

    n = len(rho_dict)
    
    # ✅ 使用 GridSpec 明确分配空间：n 个子图 + 1 个 colorbar 位置
    fig = plt.figure(figsize=(5 * n + 1, 5))  # 额外加宽 1 英寸给 colorbar
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.05], wspace=0.3)
    
    axes = []
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    
    last_pcm = None
    for ax, (label, rho) in zip(axes, rho_dict.items()):
        freqs_a = np.asarray(freqs, dtype=float).reshape(-1)
        stations_a = np.asarray(stations, dtype=float).reshape(-1)
        st_km = stations_a / 1000.0
        periods = 1.0 / np.clip(freqs_a, 1e-30, None)
        X, Y = np.meshgrid(st_km, periods)
        data = np.ma.masked_invalid(np.asarray(rho, dtype=float))
        last_pcm = ax.pcolormesh(
            X, Y, data, shading="auto", cmap=cmap, vmin=vmin_u, vmax=vmax_u
        )
        ax.set_yscale("log")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Period (s)")
        ax.set_title(label)
        ax.invert_yaxis()
    
    if last_pcm is not None:
        # ✅ colorbar 放在单独预留的 GridSpec 位置
        cax = fig.add_subplot(gs[0, -1])
        fig.colorbar(last_pcm, cax=cax, label="ρ_a (Ω·m)")
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, y=1.02) 
    
    plt.tight_layout()
    
    if show:
        plt.show()
    return np.array(axes)


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
