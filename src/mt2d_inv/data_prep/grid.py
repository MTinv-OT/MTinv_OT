"""2D MT data preparation (EDI → tensors, profiles, plotting, etc.).

Revision notes (2026-04-11)
    GMT prior-grid integration: ``PrepareData`` adds ``_sorted_mts_for_export``,
    ``get_station_lon_lat_sorted``, ``build_prior_lon_lat_at_y_centers``, and
    ``build_prior_options``. Station lon/lat are linearly interpolated at ``yn``
    horizontal cell centers (``extrapolate=clamp``). Station ordering matches
    ``export_data_dict_for_2d_inversion(..., sort_by=...)``, producing
    ``prior_options['lon']/['lat']`` for ``prior_grids.build_prior_sigma_earth``
    and ``MT2DInverter.initialize_model(..., use_prior_model=True)``.

    A slab ``.grd`` supplies only an interface-depth scalar field; conductivities
    above/below the slab are set by the caller via ``sigma_above_slab`` and
    ``sigma_below_slab`` (S/m). Optional ``slab_plate_thickness_m`` +
    ``sigma_mantle_deep`` limits the resistive slab to finite thickness and
    restores mantle conductivity below it (avoiding a semi-infinite oceanic-crust column).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def compute_grid_horizontal_extent(
    stations_m: np.ndarray,
    freqs_hz: np.ndarray,
    rho_typical_ohm_m: float = 100.0,
    skin_depth_multiple: float = 2.0,
    min_extend_km: float = 5.0,
) -> Tuple[float, float]:
    """Compute recommended left/right boundaries (m) for 2D MT inversion grid.

    The grid should extend at least `skin_depth_multiple` × max skin depth from the
    outermost stations, and at least `min_extend_km` km on each side.

    Parameters
    ----------
    stations_m : np.ndarray
        Station positions along profile (m).
    freqs_hz : np.ndarray
        Frequencies (Hz). Lowest freq gives max skin depth.
    rho_typical_ohm_m : float
        Typical resistivity (Ω·m) for skin depth estimate. Default 100.
    skin_depth_multiple : float
        Grid must extend this many × max skin depth from edge stations. Default 2.
    min_extend_km : float
        Minimum extension (km) beyond stations on each side. Default 5.

    Returns
    -------
    y_left_m, y_right_m : float
        Left and right boundaries in meters.
    """
    st_min = float(np.min(stations_m))
    st_max = float(np.max(stations_m))
    f_min = float(np.min(freqs_hz))
    if f_min <= 0:
        raise ValueError("Frequencies must be positive")
    # Skin depth: δ = sqrt(2/(σ ω μ)) = sqrt(2 ρ / (ω μ)), σ=1/ρ, ω=2πf, μ=4πe-7
    mu0 = 4e-7 * np.pi
    omega = 2.0 * np.pi * f_min
    sigma = 1.0 / (rho_typical_ohm_m)
    skin_depth_m = np.sqrt(2.0 / (sigma * omega * mu0))
    extend_skin_m = skin_depth_multiple * skin_depth_m
    extend_min_m = min_extend_km * 1000.0
    half_extend_m = max(extend_skin_m, extend_min_m)
    return st_min - half_extend_m, st_max + half_extend_m


def build_yn_from_stations(
    stations_m: np.ndarray,
    freqs_hz: np.ndarray,
    n_center_cells: int = 31,
    n_side_cells: int = 15,
    rho_typical_ohm_m: float = 100.0,
    skin_depth_multiple: float = 2.0,
    min_extend_km: float = 5.0,
) -> np.ndarray:
    """Build horizontal grid (yn, node positions in m) for 2D MT inversion.

    Center region covers stations; sides extend at least max(skin_depth_multiple × max
    skin depth, min_extend_km) from the outermost station.

    Parameters
    ----------
    stations_m, freqs_hz, rho_typical_ohm_m, skin_depth_multiple, min_extend_km
        See compute_grid_horizontal_extent().
    n_center_cells : int
        Number of cells in the center (uniform) region.
    n_side_cells : int
        Number of cells in each side (log-spaced) extension.

    Returns
    -------
    yn : np.ndarray
        Node positions (m) for the horizontal grid.
    """
    y_left_m, y_right_m = compute_grid_horizontal_extent(
        stations_m, freqs_hz,
        rho_typical_ohm_m=rho_typical_ohm_m,
        skin_depth_multiple=skin_depth_multiple,
        min_extend_km=min_extend_km,
    )
    st_min, st_max = float(np.min(stations_m)), float(np.max(stations_m))
    # Center: uniform from st_min to st_max
    y_center = np.linspace(st_min, st_max, n_center_cells + 1)
    # Left extension: log-spaced from y_left_m to st_min (exclusive)
    d_left = st_min - y_left_m
    if d_left > 1:
        y_left = st_min - np.logspace(np.log10(d_left), np.log10(1.0), n_side_cells + 1)
    else:
        y_left = np.array([y_left_m])
    # Right extension: log-spaced from st_max to y_right_m (exclusive of st_max)
    d_right = y_right_m - st_max
    if d_right > 1:
        y_right = st_max + np.logspace(np.log10(1.0), np.log10(d_right), n_side_cells + 1)
    else:
        y_right = np.array([y_right_m])
    yn = np.concatenate([y_left, y_center, y_right])
    return yn


