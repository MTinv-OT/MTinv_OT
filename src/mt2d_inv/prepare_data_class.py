"""2D MT 数据准备（EDI → 张量、剖面、绘图等）。

修订说明（2026-04-11）
    与 GMT 先验网格衔接：``PrepareData`` 新增 ``_sorted_mts_for_export``、
    ``get_station_lon_lat_sorted``、``build_prior_lon_lat_at_y_centers``、
    ``build_prior_options``。在 ``yn`` 水平元胞中心对台站经纬度线性插值（``extrapolate=clamp``），
    台站排序与 ``export_data_dict_for_2d_inversion(..., sort_by=...)`` 一致，生成
    ``prior_options['lon']/['lat']``，供 ``prior_grids.build_prior_sigma_earth`` 与
    ``MT2DInverter.initialize_model(..., use_prior_model=True)`` 使用。

    Slab 的 ``.grd`` 只提供界面深度标量场；板片上/下方电导率在先验里由调用方通过
    ``sigma_above_slab``、``sigma_below_slab``（S/m）传入。可选 ``slab_plate_thickness_m`` +
    ``sigma_mantle_deep`` 将高阻板片限制为有限厚度，其下恢复地幔电导率（避免半无限洋壳柱）。
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


class PrepareData:
    """
    - 读取 EDI -> `CustomMT` 列表
    - 相位张量/走向/偏角修正
    - 台站投影到剖面坐标，并写回 `profile_pos_m`
    - 绘图工具（走向、剖面、相位张量椭圆）
    - 可选：整体旋转阻抗张量
    """

    # -------------------------- Data container --------------------------

    @dataclass
    class CustomMT:
        """MT data container"""

        lat: float
        lon: float
        frequency: np.ndarray
        Z: np.ndarray
        Z_err: Optional[np.ndarray] = None
        
        # Derived quantities (apparent resistivity / phase) and their errors
        rho: Optional[np.ndarray] = None
        phs: Optional[np.ndarray] = None
        rho_err: Optional[np.ndarray] = None
        phs_err: Optional[np.ndarray] = None

        # Optional: dimensionless noise std-dev used for chi^2 / RMS (log10(rho), phs/90)
        rho_noise_std_log10: Optional[np.ndarray] = None
        phs_noise_std_norm: Optional[np.ndarray] = None

        station_id: Optional[str] = None
        profile_pos_m: Optional[float] = None

        # Tipper T (n_freq, 2) complex: T[:,0]=Tzx, T[:,1]=Tzy; Hz = Tzx*Hx + Tzy*Hy
        T: Optional[np.ndarray] = None

        def rotate(self, angle_deg_clockwise: float) -> None:
            """Rotate impedance tensor (clockwise degrees)."""

            rad = np.deg2rad(float(angle_deg_clockwise))
            c = np.cos(rad)
            s = np.sin(rad)
            R = np.array([[c, s], [-s, c]])

            new_Z = np.zeros_like(self.Z, dtype=complex)
            new_Err = None
            if self.Z_err is not None:
                new_Err = np.zeros_like(self.Z_err)

            for i in range(len(self.frequency)):
                z = self.Z[i]
                new_Z[i] = R @ z @ R.T
                if new_Err is not None:
                    sigma = self.Z_err[i]
                    var = sigma * sigma
                    R_sq = R**2
                    new_var = R_sq @ var @ R_sq.T
                    new_Err[i] = np.sqrt(np.maximum(new_var, 0))

            self.Z = new_Z
            if new_Err is not None:
                self.Z_err = new_Err

            # Rotate Tipper: T' = R @ T (T is (n_freq, 2), each row [Tzx, Tzy])
            if getattr(self, "T", None) is not None and self.T.shape[0] == len(self.frequency):
                new_T = np.zeros_like(self.T, dtype=complex)
                for i in range(len(self.frequency)):
                    new_T[i] = R @ self.T[i]
                self.T = new_T

    # -------------------------- ctor / config --------------------------

    def __init__(
        self,
        edi_dir: str = "AKBST-AMT-L08",
        mag_declination_deg: float = 4.0,
        edi_impedance_unit: str = "mv/km/nt",
        mag_declination_date=None,
        freq_min_hz: Optional[float] = None,
        freq_max_hz: Optional[float] = None,
        n_freq_target: Optional[int] = None,
        station_id_mode: str = "auto",
        user_strike_true_deg: Optional[float] = None,
        user_strike_magnetic_deg: Optional[float] = None,
        clean_data: bool = True,
        clean_rel_err_max: float = 0.5,
        clean_skew_threshold: Optional[float] = 8.0,
        clean_neighbor_z_thresh: float = 4.0,
        clean_neighbor_rho_log10_floor: float = 0.25,
        clean_neighbor_phs_deg_floor: float = 8.0,
        harmonize_freqs: bool = False,
        strike_skew_threshold: float = 5.0,
    ) -> None:
        self.edi_dir = str(edi_dir)
        self.mag_declination_deg = float(mag_declination_deg)
        # EDI impedance unit handling.
        # Our rho formula assumes Z is in ohms.
        # Default: many EDI datasets store impedance in (mV/km)/nT, so we default to that
        # and convert to ohms when loading.
        # - "mv/km/nt": treat as (mV/km)/nT and convert to ohms.
        # - "ohm": assume already in ohms (no scaling).
        # - "auto": detect by magnitude; if |Z| is extremely large (e.g. > 1e3), treat as (mV/km)/nT.
        self.edi_impedance_unit = str(edi_impedance_unit).strip().lower()
        self.mag_declination_date = mag_declination_date

        # Optional: frequency band selection applied at EDI read time.
        # This trims mt.frequency/Z/Z_err early so all downstream diagnostics/plots
        # (strike, phase tensor, etc.) operate on the selected band.
        self.freq_min_hz = float(freq_min_hz) if freq_min_hz is not None else None
        self.freq_max_hz = float(freq_max_hz) if freq_max_hz is not None else None
        self.n_freq_target = int(n_freq_target) if n_freq_target is not None else None
        # station_id_mode: "auto" = DATAID -> filename stem -> S1,S2,...; "index" = always S1,S2,...
        self.station_id_mode = str(station_id_mode).strip().lower()

        # Data cleaning (OOQ + rel_err + skew + neighbor-spike); used by load_mt_objects unless overridden.
        self.harmonize_freqs = bool(harmonize_freqs)
        self.clean_data = bool(clean_data)
        self.clean_rel_err_max = float(clean_rel_err_max)
        self.clean_skew_threshold = float(clean_skew_threshold)
        self.clean_neighbor_z_thresh = float(clean_neighbor_z_thresh)
        self.clean_neighbor_rho_log10_floor = float(clean_neighbor_rho_log10_floor)
        self.clean_neighbor_phs_deg_floor = float(clean_neighbor_phs_deg_floor)
        # Phase-tensor strike: keep strike estimate only when |β_skew| < this (deg).
        # Larger => more (period,station) strikes kept; unrelated to clean_skew_threshold (data NaN-ing).
        self.strike_skew_threshold = float(strike_skew_threshold)

        # Optional: user-defined strike (preferred over estimated strike for rotation/profile/plots).
        # - *_true: true-north reference (deg)
        # - *_magnetic: magnetic-north reference (deg)
        # If only one is provided, the other is derived using mag_declination_deg.
        self.user_strike_true: Optional[float] = None
        self.user_strike_magnetic: Optional[float] = None
        self.set_user_strike(
            strike_true_deg=user_strike_true_deg,
            strike_magnetic_deg=user_strike_magnetic_deg,
            verbose=False,
        )

        self.mt_objects: list[PrepareData.CustomMT] = []
        self.files: list[str] = []

        self.regional_strike_magnetic: Optional[float] = None
        self.all_strikes_magnetic: Optional[np.ndarray] = None
        self.regional_strike_true: Optional[float] = None
        self.all_strikes_true: Optional[np.ndarray] = None

    # -------------------------- user strike override --------------------------

    def clear_user_strike(self) -> None:
        """Clear user-defined strike override."""

        self.user_strike_true = None
        self.user_strike_magnetic = None

    def set_user_strike(
        self,
        *,
        strike_true_deg: Optional[float] = None,
        strike_magnetic_deg: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Set a user-defined strike angle.

        Parameters
        ----------
        strike_true_deg
            Strike in degrees referenced to true north.
        strike_magnetic_deg
            Strike in degrees referenced to magnetic north.
        verbose
            If True, prints the stored strike values.

        Notes
        -----
        If only one of (true/magnetic) is provided, the other is derived using
        ``mag_declination_deg`` with the convention:
        ``strike_true = strike_magnetic + declination`` (east-positive declination).
        """

        if strike_true_deg is None and strike_magnetic_deg is None:
            self.clear_user_strike()
            return None, None

        decl = float(self.mag_declination_deg)

        if strike_true_deg is not None and strike_magnetic_deg is not None:
            st_true = float(self.wrap_deg_180(float(strike_true_deg)))
            st_mag = float(self.wrap_deg_180(float(strike_magnetic_deg)))
        elif strike_true_deg is not None:
            st_true = float(self.wrap_deg_180(float(strike_true_deg)))
            st_mag = float(self.wrap_deg_180(float(st_true) - decl))
        else:
            st_mag = float(self.wrap_deg_180(float(strike_magnetic_deg)))
            st_true = float(self.wrap_deg_180(float(st_mag) + decl))

        self.user_strike_true = st_true
        self.user_strike_magnetic = st_mag

        if verbose:
            print(
                f"[PrepareData] Using user strike override: "
                f"true={float(st_true):.3f} deg, magnetic={float(st_mag):.3f} deg "
                f"(declination={decl:.3f} deg)"
            )

        return self.user_strike_true, self.user_strike_magnetic

    def _select_strike_true_deg(self, strike_true_deg: Optional[float] = None) -> Optional[float]:
        """Select strike (true-north) with priority: explicit > user override > estimated."""

        if strike_true_deg is not None and np.isfinite(float(strike_true_deg)):
            return float(self.wrap_deg_180(float(strike_true_deg)))
        if self.user_strike_true is not None and np.isfinite(float(self.user_strike_true)):
            return float(self.user_strike_true)
        if self.regional_strike_true is not None and np.isfinite(float(self.regional_strike_true)):
            return float(self.regional_strike_true)
        return None

    def _select_strike_magnetic_deg(self, strike_magnetic_deg: Optional[float] = None) -> Optional[float]:
        """Select strike (magnetic-north) with priority: explicit > user override > estimated."""

        if strike_magnetic_deg is not None and np.isfinite(float(strike_magnetic_deg)):
            return float(self.wrap_deg_180(float(strike_magnetic_deg)))
        if self.user_strike_magnetic is not None and np.isfinite(float(self.user_strike_magnetic)):
            return float(self.user_strike_magnetic)
        if self.regional_strike_magnetic is not None and np.isfinite(float(self.regional_strike_magnetic)):
            return float(self.regional_strike_magnetic)
        return None

    # -------------------------- basic utils --------------------------

    @staticmethod
    def parse_dms(dms_str: str) -> float:
        """DMS to decimal."""

        dms_str = str(dms_str).strip()
        parts = dms_str.split(":")
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        sign = 1
        if d < 0:
            sign = -1
            d = abs(d)
        return sign * (d + m / 60 + s / 3600)

    @staticmethod
    def _module_dir() -> Path:
        return Path(__file__).resolve().parent

    # -------------------------- EDI parser --------------------------

    @classmethod
    def find_edi_files(cls, edi_dir: str) -> Tuple[str, list[str]]:
        """Resolve EDI directory robustly (works even if cwd isn't the notebook folder)."""

        base = cls._module_dir()
        candidates = [
            Path(edi_dir),
            base / edi_dir,
            Path("AKBST-AMT-L08"),
            base / "AKBST-AMT-L08",
            Path("./AKBST-AMT-L08"),
            Path("edi"),
            Path("./edi"),
            Path("MTinv_OT") / "Data-preprocess" / "1" / "AKBST-AMT-L08",
            Path("Data-preprocess") / "1" / "AKBST-AMT-L08",
        ]
        for c in candidates:
            if c.is_dir():
                fs = sorted(c.glob("*.edi"))
                if fs:
                    return str(c), [str(f) for f in fs]
        return str(edi_dir), []

    @classmethod
    def read_custom_edi(
        cls,
        file_path: str,
        *,
        freq_min_hz: Optional[float] = None,
        freq_max_hz: Optional[float] = None,
    ) -> "PrepareData.CustomMT":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        data_map: dict[str, np.ndarray] = {}
        lat, lon = 0.0, 0.0
        station_id: Optional[str] = None
        current_key: Optional[str] = None
        current_vals: list[float] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("LAT="):
                lat = cls.parse_dms(line.split("=")[1])
                continue
            if line.startswith("LONG="):
                lon = cls.parse_dms(line.split("=")[1])
                continue
            if line.startswith("DATAID="):
                raw = line.split("=", 1)[1].strip().strip('"\'')
                if raw:
                    station_id = raw
                continue

            if line.startswith(">"):
                if current_key:
                    data_map[current_key] = np.array(current_vals)
                key = line[1:].split("//")[0].strip()
                key = key.split()[0]
                key = key.replace(".", "").upper()
                current_key = key
                current_vals = []
            else:
                if current_key:
                    try:
                        vals = [float(x) for x in line.split()]
                        current_vals.extend(vals)
                    except Exception:
                        pass

        if current_key:
            data_map[current_key] = np.array(current_vals)

        # Fallback 1: filename stem (e.g. CAF054.edi -> CAF054, B01.edi -> B01)
        # Fallback 2: done in load_mt_objects using index (S1, S2, ...) if still None
        if station_id is None:
            station_id = Path(file_path).stem

        # CAFE land EDI (DATAID / filename ``CAF*``): ``Z..VAR`` blocks store **std-dev** of Re/Im.
        # CAFE offshore ``B*`` and other datasets (e.g. China): ``Z..VAR`` stores **variance** → sqrt.
        sid_up = str(station_id).strip().upper()
        z_var_blocks_are_std = sid_up.startswith("CAF")

        freqs = data_map["FREQ"]
        n = len(freqs)
        Z = np.zeros((n, 2, 2), dtype=complex)
        Z_err = np.zeros((n, 2, 2), dtype=float)
        has_err = False

        for i, c1 in enumerate(["X", "Y"]):
            for j, c2 in enumerate(["X", "Y"]):
                comp = c1 + c2
                rkey = "Z" + comp + "R"
                ikey = "Z" + comp + "I"
                if rkey in data_map and ikey in data_map:
                    Z[:, i, j] = data_map[rkey][:n] + 1j * data_map[ikey][:n]
                vkey = "Z" + comp + "VAR"
                if vkey in data_map:
                    v = np.abs(np.asarray(data_map[vkey][:n], dtype=float))
                    if z_var_blocks_are_std:
                        Z_err[:, i, j] = v
                    else:
                        Z_err[:, i, j] = np.sqrt(v)
                    has_err = True

        Z_err_out = Z_err if has_err else None

        # Optional: Tipper T (n_freq, 2) complex: Tzx, Tzy (Hz = Tzx*Hx + Tzy*Hy)
        # EDI block names: TXR.EXP/TXI.EXP/TYR.EXP/TYI.EXP or TZXR/TZXI/TZYR/TZYI
        T_out = None
        for txr, txi, tyr, tyi in [
            ("TXREXP", "TXIEXP", "TYREXP", "TYIEXP"),
            ("TZXR", "TZXI", "TZYR", "TZYI"),
        ]:
            if txr in data_map and txi in data_map and tyr in data_map and tyi in data_map:
                tzx = np.asarray(data_map[txr][:n], dtype=float) + 1j * np.asarray(data_map[txi][:n], dtype=float)
                tzy = np.asarray(data_map[tyr][:n], dtype=float) + 1j * np.asarray(data_map[tyi][:n], dtype=float)
                T_out = np.stack([tzx, tzy], axis=1)
                break

        # Optional: trim frequency band early (applies to frequency/Z/Z_err/T).
        if freq_min_hz is not None or freq_max_hz is not None:
            fmin = float(freq_min_hz) if freq_min_hz is not None else None
            fmax = float(freq_max_hz) if freq_max_hz is not None else None
            mask = np.ones(freqs.shape[0], dtype=bool)
            if fmin is not None:
                mask &= freqs >= fmin
            if fmax is not None:
                mask &= freqs <= fmax
            if not np.any(mask):
                raise ValueError(
                    f"No frequencies left after band selection while reading EDI: {file_path}. "
                    f"freq_min_hz={freq_min_hz}, freq_max_hz={freq_max_hz}."
                )
            freqs = freqs[mask]
            Z = Z[mask]
            if Z_err_out is not None:
                Z_err_out = Z_err_out[mask]
            if T_out is not None:
                T_out = T_out[mask]
           
        return cls.CustomMT(
            lat=float(lat),
            lon=float(lon),
            frequency=np.array(freqs),
            Z=Z,
            Z_err=Z_err_out,
            station_id=station_id,
            T=T_out,
        )

    def load_mt_objects(self, *, harmonize_freqs: Optional[bool] = None) -> list["PrepareData.CustomMT"]:
        """Load EDI stations into ``self.mt_objects``.

        harmonize_freqs
            If None (default), use ``self.harmonize_freqs`` from ``PrepareData(...)``.
            If True/False, override for this call only (advanced; notebooks normally set
            ``harmonize_freqs`` on ``PrepareData`` instead).
        """
        use_harmonize = bool(self.harmonize_freqs) if harmonize_freqs is None else bool(harmonize_freqs)

        edi_dir, files = self.find_edi_files(self.edi_dir)
        self.edi_dir = edi_dir
        self.files = files

        print("Using EDI_DIR =", self.edi_dir)
        print("Found", len(self.files), "EDI files")

        self.mt_objects = [
            self.read_custom_edi(
                f,
                freq_min_hz=self.freq_min_hz,
                freq_max_hz=self.freq_max_hz,
            )
            for f in self.files
        ]

        # station_id: "index" = always S1,S2,...; "auto" = fallback to index if still None
        for i, mt in enumerate(self.mt_objects):
            if self.station_id_mode == "index":
                mt.station_id = f"S{i + 1}"
            elif getattr(mt, "station_id", None) is None:
                mt.station_id = f"S{i + 1}"

        # Convert impedance units to ohms if needed (before computing derived rho/phs).
        n_scaled = 0
        for mt in self.mt_objects:
            if self._maybe_scale_impedance_units_inplace(mt):
                n_scaled += 1
        if n_scaled:
            print(
                f"[PrepareData] Converted impedance units to ohms for {n_scaled}/{len(self.mt_objects)} stations "
                f"(edi_impedance_unit={self.edi_impedance_unit!r})."
            )
        n_tipper = sum(1 for mt in self.mt_objects if getattr(mt, "T", None) is not None)
        if n_tipper:
            print(f"[PrepareData] Loaded Tipper for {n_tipper}/{len(self.mt_objects)} stations (90° strike check enabled).")

        # [Priority 1+2+3] Data cleaning: OOQ + rel_err + skew (before phase fold / rho-phs compute)
        if self.clean_data:
            self._clean_data_ooq_rel_err_inplace(
                self.mt_objects,
                rel_err_max=self.clean_rel_err_max,
                skew_threshold=self.clean_skew_threshold,
                neighbor_z_thresh=self.clean_neighbor_z_thresh,
                neighbor_rho_log10_floor=self.clean_neighbor_rho_log10_floor,
                neighbor_phs_deg_floor=self.clean_neighbor_phs_deg_floor,
            )

        # Pre-compute derived rho/phase from raw tensors (useful even before rotation)
        self.compute_rho_phase_all(self.mt_objects)

        # =====================================================================
        # [核心修改]：根据开关决定是否进行强制频率对齐与降采样
        # =====================================================================
        if use_harmonize:
            # 传统的老路径：强制所有台站取交集对齐 (削足适履)
            self._harmonize_frequencies_inplace()

            # Optional: downsample frequencies to n_freq_target (log-uniform)
            # 注意：全局降采样依赖于所有台站频率已经对齐，因此只能在 harmonize 后执行
            if self.n_freq_target is not None and self.mt_objects:
                n_available = len(self.mt_objects[0].frequency)
                if self.n_freq_target > n_available:
                    raise ValueError(
                        f"n_freq_target={int(self.n_freq_target)} is larger than the minimum available "
                        f"frequency count per station after band selection/harmonization ({int(n_available)}). "
                        f"Reduce n_freq_target or widen FREQ_MIN_HZ/FREQ_MAX_HZ."
                    )
                if n_available > self.n_freq_target:
                    self._decimate_frequencies_inplace()
                    n_after = len(self.mt_objects[0].frequency)
                    print(
                        f"[PrepareData] Downsampled frequencies (after band mask): {n_available} -> {n_after}"
                    )
        else:
            # 现代的新路径：保留数据的异频原貌，供 export_data_dict_for_2d_inversion 提取全局并集并做 NaN 掩码
            print("[PrepareData] Skipped frequency harmonization. Stations retain their original frequencies.")
            
            # 如果此时依然设置了降采样，给出警告（因为旧版的 _decimate 是基于强制对齐逻辑写的）
            if self.n_freq_target is not None:
                print(
                    f"[PrepareData] WARNING: n_freq_target={self.n_freq_target} is ignored because "
                    "harmonize_freqs=False. Downsampling requires an aligned frequency grid in the current implementation."
                )

        return self.mt_objects

    def _clean_data_ooq_rel_err_inplace(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        zxy_phase_range: Tuple[float, float] = (-5.0, 95.0),
        zyx_phase_range: Tuple[float, float] = (175.0, 275.0), # 推荐直接使用 0~360 的角度定义
        rel_err_max: float = 0.5,
        skew_threshold: Optional[float] = 15.0,
        neighbor_z_thresh: float = 4.0,
        neighbor_rho_log10_floor: float = 0.25,
        neighbor_phs_deg_floor: float = 8.0,
    ) -> None:
        """
        数据清洗：OOQ + rel_err + Phase Tensor Skew + 邻频(ρa/φ+误差棒)飞点。

        P1 (OOQ)：能量耗散关系反了则一票否决，在相位折叠到 0~90° 之前拦截。
        - Zxy 理论在第一象限，检查 arctan2(Im, Re) 是否在 zxy_phase_range (默认 [-5°, 95°])
        - Zyx 理论在第三象限，检查 arctan2(Im, Re) % 360 是否在 zyx_phase_range (默认 [175°, 275°])
        越界者：将该频点该模式的 Z, Z_err 置 NaN

        P2 (rel_err)：相对误差超 rel_err_max (默认 50%) 则剔除该频点的有效反演分量。
        - 仅对反演使用的离对角分量 Zxy/Zyx 计算相对误差（避免 Zxx/Zyy 小量噪声导致误杀）
        - 若 max(rel_err(Zxy), rel_err(Zyx)) > rel_err_max，则将该频点的 Zxy/Zyx (及其 Z_err) 置 NaN

        P3 (skew)：强三维畸变剔除，保护 2D 正演不被 3D 现象“欺骗”。
        - 计算相位张量偏斜度 β，若 |β| > skew_threshold (默认 8°)，将该点双模式(TE/TM)全位置 NaN
        - skew_threshold=None 则跳过此检查

        P4 (neighbor spike, per channel)：
        - 在每个模式分量（Zxy/Zyx）上，使用 q=log10(ρa) 与 q=φ(°) 分别做邻频比较
        - 当前点与邻居中位数的偏差若同时满足：
            |Δq| > abs_floor 且 |Δq| / sqrt(σ_i^2 + σ_nei^2) > z_thresh
          则判为飞点（仅该分量该频点置 NaN）
        - 阈值默认较宽松：z_thresh=4，ρa floor=0.25 log10，φ floor=8°
        - 需要 Z_err 才能使用误差棒；若无 Z_err 则跳过 P4
        """
        if mt_objects is None:
            mt_objects = self.mt_objects
        if not mt_objects:
            return

        n_ooq_xy = 0
        n_ooq_yx = 0
        n_rel_err = 0
        n_skew = 0
        n_spike_xy = 0
        n_spike_yx = 0

        for mt in mt_objects:
            Z = np.asarray(mt.Z, dtype=np.complex128)
            Z_err = mt.Z_err
            if Z.ndim != 3 or Z.shape[1:] != (2, 2):
                continue
            n_f = Z.shape[0]

            # -------- Priority 1: OOQ (raw phase, before fold) --------
            phs_xy = np.degrees(np.arctan2(Z[:, 0, 1].imag, Z[:, 0, 1].real))
            # 对于 Zyx 采用模 360 运算，彻底解决 -180/180 跃变问题
            phs_yx_360 = np.degrees(np.arctan2(Z[:, 1, 0].imag, Z[:, 1, 0].real)) % 360.0

            bad_xy = (phs_xy < zxy_phase_range[0]) | (phs_xy > zxy_phase_range[1])
            bad_yx = (phs_yx_360 < zyx_phase_range[0]) | (phs_yx_360 > zyx_phase_range[1])

            # Mask also invalid (NaN/Inf) as bad
            bad_xy = bad_xy | ~np.isfinite(Z[:, 0, 1])
            bad_yx = bad_yx | ~np.isfinite(Z[:, 1, 0])

            n_ooq_xy += int(np.sum(bad_xy))
            n_ooq_yx += int(np.sum(bad_yx))

            # 使用直接切片赋值，避免 Numpy 链式视图潜在警告
            Z[bad_xy, 0, 1] = np.nan + 1j * np.nan
            Z[bad_yx, 1, 0] = np.nan + 1j * np.nan
            
            if Z_err is not None:
                Z_err = np.asarray(Z_err, dtype=float)
                Z_err[bad_xy, 0, 1] = np.nan
                Z_err[bad_yx, 1, 0] = np.nan
                mt.Z_err = Z_err
            mt.Z = Z

            # -------- Priority 2: rel_err (per-freq, off-diagonal only) --------
            if Z_err is not None:
                # Only use Zxy/Zyx since 2D inversion uses these components.
                # Diagonals (Zxx/Zyy) are often near-zero and can have large relative error,
                # which should not invalidate the off-diagonal data.
                zxy_abs = np.abs(Z[:, 0, 1])
                zyx_abs = np.abs(Z[:, 1, 0])
                zxy_abs = np.maximum(zxy_abs, 1e-20)
                zyx_abs = np.maximum(zyx_abs, 1e-20)
                rel_xy = np.abs(Z_err[:, 0, 1]) / zxy_abs
                rel_yx = np.abs(Z_err[:, 1, 0]) / zyx_abs
                # fmax ignores NaN (treats NaN as missing), and won't warn on all-NaN slices
                rel_err_max_per_freq = np.fmax(rel_xy, rel_yx)
                bad_rel = rel_err_max_per_freq > rel_err_max
                n_rel_err += int(np.sum(bad_rel))

                # Mask only the off-diagonal components used by inversion
                Z[bad_rel, 0, 1] = np.nan + 1j * np.nan
                Z[bad_rel, 1, 0] = np.nan + 1j * np.nan
                Z_err[bad_rel, 0, 1] = np.nan
                Z_err[bad_rel, 1, 0] = np.nan

                mt.Z = Z
                mt.Z_err = Z_err

            # -------- Priority 3: skew (per-freq, whole 2x2 if |β| > threshold) --------
            if skew_threshold is not None and skew_threshold > 0:
                for i in range(n_f):
                    if not np.all(np.isfinite(Z[i])):
                        continue
                    skew = self.phase_tensor_skew(Z[i])
                    if np.isfinite(skew) and abs(skew) > float(skew_threshold):
                        n_skew += 1
                        Z[i, :, :] = np.nan + 1j * np.nan
                        if Z_err is not None:
                            Z_err[i, :, :] = np.nan
                mt.Z = Z
                if Z_err is not None:
                    mt.Z_err = Z_err

            # -------- Priority 4: neighbor spike by (rho/phi + error bars), per channel --------
            if Z_err is not None and n_f >= 2:
                _, phs, _, phs_err, rho_noise_std_log10, _ = self.impedance_to_rho_phase(
                    freqs_hz=np.asarray(mt.frequency, dtype=float),
                    Z=Z,
                    Z_err=Z_err,
                )
                rho = (np.abs(Z) ** 2) / ((2.0 * np.pi * np.asarray(mt.frequency, dtype=float))[:, None, None] * self._mu0())
                log_rho = np.log10(np.maximum(rho, 1e-20))
                phs = np.asarray(phs, dtype=float)
                phs_err = np.asarray(phs_err, dtype=float) if phs_err is not None else None
                rho_noise_std_log10 = np.asarray(rho_noise_std_log10, dtype=float) if rho_noise_std_log10 is not None else None

                def _neighbor_spike_mask(q: np.ndarray, sigma_q: np.ndarray, z_thr: float, abs_floor: float) -> np.ndarray:
                    q = np.asarray(q, dtype=float)
                    sigma_q = np.asarray(sigma_q, dtype=float)
                    bad = np.zeros(q.shape[0], dtype=bool)
                    for ii in range(q.shape[0]):
                        if not np.isfinite(q[ii]) or not np.isfinite(sigma_q[ii]):
                            continue
                        qn = []
                        sn = []
                        if ii - 1 >= 0 and np.isfinite(q[ii - 1]) and np.isfinite(sigma_q[ii - 1]):
                            qn.append(q[ii - 1]); sn.append(sigma_q[ii - 1])
                        if ii + 1 < q.shape[0] and np.isfinite(q[ii + 1]) and np.isfinite(sigma_q[ii + 1]):
                            qn.append(q[ii + 1]); sn.append(sigma_q[ii + 1])
                        if len(qn) == 0:
                            continue
                        q_ref = float(np.median(np.asarray(qn, dtype=float)))
                        s_ref = float(np.median(np.asarray(sn, dtype=float)))
                        dq = abs(float(q[ii]) - q_ref)
                        s_eff = float(np.sqrt(float(sigma_q[ii]) ** 2 + s_ref ** 2 + 1e-20))
                        zscore = dq / s_eff if s_eff > 0 else 0.0
                        if dq > float(abs_floor) and zscore > float(z_thr):
                            bad[ii] = True
                    return bad

                # XY channel
                bad_xy_rho = _neighbor_spike_mask(
                    q=log_rho[:, 0, 1],
                    sigma_q=np.maximum(rho_noise_std_log10[:, 0, 1], 1e-6),
                    z_thr=float(neighbor_z_thresh),
                    abs_floor=float(neighbor_rho_log10_floor),
                )
                bad_xy_phs = _neighbor_spike_mask(
                    q=phs[:, 0, 1],
                    sigma_q=np.maximum(phs_err[:, 0, 1], 1e-6),
                    z_thr=float(neighbor_z_thresh),
                    abs_floor=float(neighbor_phs_deg_floor),
                )
                bad_xy_spike = bad_xy_rho | bad_xy_phs

                # YX channel
                bad_yx_rho = _neighbor_spike_mask(
                    q=log_rho[:, 1, 0],
                    sigma_q=np.maximum(rho_noise_std_log10[:, 1, 0], 1e-6),
                    z_thr=float(neighbor_z_thresh),
                    abs_floor=float(neighbor_rho_log10_floor),
                )
                bad_yx_phs = _neighbor_spike_mask(
                    q=phs[:, 1, 0],
                    sigma_q=np.maximum(phs_err[:, 1, 0], 1e-6),
                    z_thr=float(neighbor_z_thresh),
                    abs_floor=float(neighbor_phs_deg_floor),
                )
                bad_yx_spike = bad_yx_rho | bad_yx_phs

                n_spike_xy += int(np.sum(bad_xy_spike))
                n_spike_yx += int(np.sum(bad_yx_spike))

                if np.any(bad_xy_spike):
                    Z[bad_xy_spike, 0, 1] = np.nan + 1j * np.nan
                    Z_err[bad_xy_spike, 0, 1] = np.nan
                if np.any(bad_yx_spike):
                    Z[bad_yx_spike, 1, 0] = np.nan + 1j * np.nan
                    Z_err[bad_yx_spike, 1, 0] = np.nan
                mt.Z = Z
                mt.Z_err = Z_err

        if n_ooq_xy or n_ooq_yx or n_rel_err or n_skew or n_spike_xy or n_spike_yx:
            msg = (
                f"[PrepareData] Data cleaning (OOQ + rel_err + skew + neighbor-spike): "
                f"OOQ Zxy={n_ooq_xy}, OOQ Zyx={n_ooq_yx}, rel_err>{rel_err_max:.0%}={n_rel_err}"
            )
            if skew_threshold is not None:
                msg += f", |skew|>{skew_threshold}°={n_skew}"
            msg += (
                f", neighbor-spike(z>{float(neighbor_z_thresh):.1f}, "
                f"rho_log10>{float(neighbor_rho_log10_floor):.2f}, "
                f"phs>{float(neighbor_phs_deg_floor):.1f}°)"
                f"=(Zxy:{n_spike_xy}, Zyx:{n_spike_yx})"
            )
            msg += " pts set to NaN"
            print(msg)

    def _harmonize_frequencies_inplace(self, freq_rtol: float = 1e-4) -> None:
        """Make all stations share the same frequency grid (intersection via nearest match)."""
        if not self.mt_objects:
            return
        n_per = [len(mt.frequency) for mt in self.mt_objects]
        if len(set(n_per)) <= 1:
            return
        # Use station with fewest frequencies as reference (most conservative)
        ref_idx = int(np.argmin(n_per))
        f_ref = np.asarray(self.mt_objects[ref_idx].frequency, dtype=float)
        f_ref = np.sort(f_ref)
        for mt in self.mt_objects:
            if mt is self.mt_objects[ref_idx]:
                continue
            f = np.asarray(mt.frequency, dtype=float)
            idx = np.array([np.abs(f - ft).argmin() for ft in f_ref], dtype=np.intp)
            # Keep only where match is close enough
            ok = np.abs(f[idx] - f_ref) <= (freq_rtol * f_ref + 1e-20)
            if not np.all(ok):
                idx = idx[ok]
                f_ref = f_ref[ok]
        # Re-apply with final f_ref (in case we trimmed)
        f_ref = np.asarray(f_ref, dtype=float)
        for i, mt in enumerate(self.mt_objects):
            f = np.asarray(mt.frequency, dtype=float)
            idx = np.array([np.abs(f - ft).argmin() for ft in f_ref], dtype=np.intp)
            mt.frequency = np.array(f[idx])
            mt.Z = np.asarray(mt.Z, dtype=np.complex128)[idx]
            if getattr(mt, "Z_err", None) is not None:
                mt.Z_err = np.asarray(mt.Z_err, dtype=float)[idx]
            if getattr(mt, "rho", None) is not None:
                mt.rho = np.asarray(mt.rho, dtype=float)[idx]
            if getattr(mt, "phs", None) is not None:
                mt.phs = np.asarray(mt.phs, dtype=float)[idx]
            if getattr(mt, "rho_err", None) is not None:
                mt.rho_err = np.asarray(mt.rho_err, dtype=float)[idx]
            if getattr(mt, "phs_err", None) is not None:
                mt.phs_err = np.asarray(mt.phs_err, dtype=float)[idx]
            if getattr(mt, "rho_noise_std_log10", None) is not None:
                mt.rho_noise_std_log10 = np.asarray(mt.rho_noise_std_log10, dtype=float)[idx]
            if getattr(mt, "phs_noise_std_norm", None) is not None:
                mt.phs_noise_std_norm = np.asarray(mt.phs_noise_std_norm, dtype=float)[idx]
            if getattr(mt, "T", None) is not None:
                mt.T = np.asarray(mt.T, dtype=np.complex128)[idx]
        n_after = len(self.mt_objects[0].frequency)
        print(f"[PrepareData] Harmonized frequencies: all stations now have {n_after} frequencies")


    def export_data_dict_for_2d_inversion(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        sort_by: str = "profile_pos_m",
        freq_rtol: float = 1e-6,  # 相对容差
        freq_atol: float = 1e-8,  # 绝对容差（新增/保留，专门对付低频浮点误差）
        device: Optional[str] = None,
        dtype=None,
        save_to_self: bool = True,
    ):
        """
        提取全局频率并集，使用 NaN 填充缺失数据，生成绝对规则的稠密张量。
        废弃了强制频率对齐，尊重数据的物理真实缺失。
        使用 rtol 和 atol 联合控制相近频点的合并。
        """
        try:
            import torch
        except Exception as e:
            raise ImportError("Requires torch") from e

        if mt_objects is None:
            mt_objects = self.mt_objects
        if not mt_objects:
            raise RuntimeError("mt_objects is empty; call load_mt_objects() first")

        need_derived = any(getattr(mt, "rho", None) is None or getattr(mt, "phs", None) is None for mt in mt_objects)
        if need_derived:
            self.compute_rho_phase_all(mt_objects)

        # 1. 排序台站
        mts = list(mt_objects)
        sort_key = sort_by.strip().lower() if sort_by else "none"
        if sort_key in {"profile_pos_m", "profile", "pos", "position"}:
            mts = sorted(mts, key=lambda m: float(getattr(m, "profile_pos_m", float("inf"))))
        elif sort_key in {"station_id", "id"}:
            mts = sorted(mts, key=lambda m: int(str(getattr(m, "station_id", 10**9))) if str(getattr(m, "station_id", "")).isdigit() else 10**9)

        stations = np.array([getattr(mt, "profile_pos_m", np.nan) for mt in mts], dtype=float)
        if not np.isfinite(stations).all():
            raise ValueError("Some stations have invalid profile_pos_m. Call assign_profile_pos_m() first.")

        # Station IDs aligned with `stations` ordering (after sorting).
        station_ids = []
        for j, mt in enumerate(mts):
            sid = getattr(mt, "station_id", None)
            if sid is None or str(sid).strip() == "":
                sid = f"S{j + 1}"
            station_ids.append(str(sid))

        # 2. 构建全局频率轴 (Union) 并使用 atol + rtol 联合去重
        all_freqs = np.concatenate([np.asarray(getattr(mt, "frequency"), dtype=float) for mt in mts])
        all_freqs = np.sort(all_freqs)
        
        # 核心合并逻辑：差值 > (atol + rtol * 当前频率) 才认为是不同的频点
        is_unique = np.append([True], np.diff(all_freqs) > (freq_atol + freq_rtol * all_freqs[:-1]))
        freqs_global = all_freqs[is_unique]
        
        n_freq = len(freqs_global)
        n_stn = len(mts)

        # 3. 初始化全 NaN 矩阵
        rhoxy = np.full((n_freq, n_stn), np.nan, dtype=float)
        phsxy = np.full((n_freq, n_stn), np.nan, dtype=float)
        rhoyx = np.full((n_freq, n_stn), np.nan, dtype=float)
        phsyx = np.full((n_freq, n_stn), np.nan, dtype=float)
        zxy = np.full((n_freq, n_stn), np.nan + 1j * np.nan, dtype=np.complex128)
        zyx = np.full((n_freq, n_stn), np.nan + 1j * np.nan, dtype=np.complex128)
        zxy_err = np.full((n_freq, n_stn), np.nan, dtype=float)
        zyx_err = np.full((n_freq, n_stn), np.nan, dtype=float)
        any_has_err = False

        # 4. 对号入座
        for j, mt in enumerate(mts):
            f_stn = np.asarray(getattr(mt, "frequency"), dtype=float)
            idx_global = np.array([np.abs(freqs_global - f).argmin() for f in f_stn])
            
            rho = np.asarray(getattr(mt, "rho", None), dtype=float)
            phs = np.asarray(getattr(mt, "phs", None), dtype=float)
            z_arr = np.asarray(getattr(mt, "Z", None), dtype=np.complex128)
            
            if rho is not None and rho.size > 0:
                rhoxy[idx_global, j] = rho[:, 0, 1]
                rhoyx[idx_global, j] = rho[:, 1, 0]
            if phs is not None and phs.size > 0:
                phsxy[idx_global, j] = phs[:, 0, 1]
                phsyx[idx_global, j] = phs[:, 1, 0]
            if z_arr is not None and z_arr.size > 0:
                zxy[idx_global, j] = z_arr[:, 0, 1]
                zyx[idx_global, j] = z_arr[:, 1, 0]
                
            z_err = getattr(mt, "Z_err", None)
            if z_err is not None:
                any_has_err = True
                z_err = np.asarray(z_err, dtype=float)
                zxy_err[idx_global, j] = z_err[:, 0, 1]
                zyx_err[idx_global, j] = z_err[:, 1, 0]

        # 5. 转换为 Torch Tensor
        if dtype is None:
            dtype = torch.float64
        freqs_t = torch.as_tensor(freqs_global, dtype=dtype, device=device)
        stations_t = torch.as_tensor(stations, dtype=dtype, device=device)
        
        obs_data = {
            "rhoxy": torch.as_tensor(rhoxy, dtype=dtype, device=device),
            "phsxy": torch.as_tensor(phsxy, dtype=dtype, device=device),
            "rhoyx": torch.as_tensor(rhoyx, dtype=dtype, device=device),
            "phsyx": torch.as_tensor(phsyx, dtype=dtype, device=device),
        }

        # Impedance & std-dev for error propagation on inverter side (require Zxy, Zyx + delta for calculate_data_errors_2d)
        data_std = {}
        torch_complex_dtype = self._torch_complex_dtype_from_float(dtype)
        data_std["Zxy"] = torch.as_tensor(zxy, dtype=torch_complex_dtype, device=device)
        data_std["Zyx"] = torch.as_tensor(zyx, dtype=torch_complex_dtype, device=device)
        if any_has_err:
            # Use EDI Z_err where available; in union+NaN mode may contain NaN at missing points (inverter masks)
            data_std["delta_zxy_real"] = torch.as_tensor(zxy_err, dtype=dtype, device=device)
            data_std["delta_zxy_imag"] = torch.as_tensor(zxy_err, dtype=dtype, device=device)
            data_std["delta_zyx_real"] = torch.as_tensor(zyx_err, dtype=dtype, device=device)
            data_std["delta_zyx_imag"] = torch.as_tensor(zyx_err, dtype=dtype, device=device)
        else:
            # No Z_err in EDI: use default 1% relative error for propagation (inverter noise_floor can further clip)
            zxy_abs = np.abs(zxy)
            zyx_abs = np.abs(zyx)
            delta_default = 0.01
            data_std["delta_zxy_real"] = torch.as_tensor(
                np.where(np.isfinite(zxy_abs), delta_default * zxy_abs, np.nan), dtype=dtype, device=device
            )
            data_std["delta_zxy_imag"] = data_std["delta_zxy_real"]
            data_std["delta_zyx_real"] = torch.as_tensor(
                np.where(np.isfinite(zyx_abs), delta_default * zyx_abs, np.nan), dtype=dtype, device=device
            )
            data_std["delta_zyx_imag"] = data_std["delta_zyx_real"]

        data_dict = {"obs_data": obs_data, "data_std": data_std, "station_ids": station_ids}

        if save_to_self:
            self.inv_freqs_torch = freqs_t
            self.inv_stations_torch = stations_t
            self.inv_data_dict = data_dict
            self.inv_station_ids = station_ids

        print(f"[PrepareData] Exported Tensor shape: Frequencies={n_freq}, Stations={n_stn}")
        return freqs_t, stations_t, data_dict

    def _sorted_mts_for_export(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        sort_by: str = "profile_pos_m",
    ) -> List["PrepareData.CustomMT"]:
        """Same station ordering as :meth:`export_data_dict_for_2d_inversion`."""
        if mt_objects is None:
            mt_objects = self.mt_objects
        mts = list(mt_objects)
        sort_key = sort_by.strip().lower() if sort_by else "none"
        if sort_key in {"profile_pos_m", "profile", "pos", "position"}:
            mts.sort(key=lambda m: float(getattr(m, "profile_pos_m", float("inf"))))
        elif sort_key in {"station_id", "id"}:
            mts.sort(
                key=lambda m: int(str(getattr(m, "station_id", 10**9)))
                if str(getattr(m, "station_id", "")).isdigit()
                else 10**9
            )
        return mts

    def get_station_lon_lat_sorted(
        self,
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Profile positions and geographic coordinates per station (after ``sort_by``).

        Returns
        -------
        stations_m, lon, lat : ndarray
            Shapes (n_stn,). Same order as columns from ``export_data_dict_for_2d_inversion``.
        """
        mts = self._sorted_mts_for_export(mt_objects=mt_objects, sort_by=sort_by)
        if not mts:
            raise RuntimeError("mt_objects is empty")
        stations = np.array([float(getattr(mt, "profile_pos_m", np.nan)) for mt in mts], dtype=float)
        if not np.isfinite(stations).all():
            raise ValueError("Invalid profile_pos_m; call assign_profile_pos_m() first.")
        lon = np.array([float(getattr(mt, "lon", np.nan)) for mt in mts], dtype=float)
        lat = np.array([float(getattr(mt, "lat", np.nan)) for mt in mts], dtype=float)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise ValueError("Some stations have invalid lat/lon (check EDI headers).")
        return stations, lon, lat

    @staticmethod
    def _yn_edges_to_numpy(yn: Union[np.ndarray, Any]) -> np.ndarray:
        try:
            import torch

            if torch.is_tensor(yn):
                return yn.detach().cpu().numpy().astype(np.float64).reshape(-1)
        except ImportError:
            pass
        return np.asarray(yn, dtype=np.float64).reshape(-1)

    def build_prior_lon_lat_at_y_centers(
        self,
        yn: Union[np.ndarray, Any],
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        extrapolate: str = "clamp",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate station lon/lat onto horizontal **cell centers** of ``yn`` (meters along profile).

        Use the same ``sort_by`` as ``export_data_dict_for_2d_inversion`` so columns match
        :class:`mt2d_inv.MTinv_2d.MT2DInverter` ``yn`` / ``stations``.

        Parameters
        ----------
        yn
            Horizontal grid **node** positions (m), length ``ny+1``; centers are ``(yn[:-1]+yn[1:])/2``.
        extrapolate
            ``clamp`` (default): outside the station span, hold edge station lon/lat.
            ``linear``: use ``numpy.interp`` default (constant edge value in numpy).

        Returns
        -------
        lon_centers, lat_centers : ndarray
            Length ``len(yn) - 1``, ready for ``prior_options['lon']`` / ``['lat']``.
        """
        yn_np = self._yn_edges_to_numpy(yn)
        if yn_np.size < 2:
            raise ValueError("yn must have at least 2 nodes")
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        st, lon_s, lat_s = self.get_station_lon_lat_sorted(sort_by=sort_by, mt_objects=mt_objects)
        if st.size < 2:
            lon_c = np.full(y_centers.shape, float(lon_s[0]), dtype=np.float64)
            lat_c = np.full(y_centers.shape, float(lat_s[0]), dtype=np.float64)
            return lon_c, lat_c
        if extrapolate == "clamp":
            lon_c = np.interp(y_centers, st, lon_s, left=float(lon_s[0]), right=float(lon_s[-1]))
            lat_c = np.interp(y_centers, st, lat_s, left=float(lat_s[0]), right=float(lat_s[-1]))
        elif extrapolate == "linear":
            lon_c = np.interp(y_centers, st, lon_s)
            lat_c = np.interp(y_centers, st, lat_s)
        else:
            raise ValueError("extrapolate must be 'clamp' or 'linear'")
        return lon_c.astype(np.float64), lat_c.astype(np.float64)

    def build_prior_options(
        self,
        yn: Union[np.ndarray, Any],
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        extrapolate: str = "clamp",
        **prior_fields: Any,
    ) -> Dict[str, Any]:
        """
        Build a ``prior_options`` dict for :meth:`mt2d_inv.MTinv_2d.MT2DInverter.initialize_model`
        (``use_prior_model=True``).

        Inserts ``lon`` and ``lat`` at each horizontal cell center (see ``build_prior_lon_lat_at_y_centers``).
        Pass GMT paths and conductivities as keywords, e.g.::

            prior = prep.build_prior_options(
                inv.yn,
                sediment_grd=\".../sedthick_world_v2.grd\",
                sigma_sediment=1.0/5.0,
                sigma_background=1e-2,
            )

        Notes
        -----
        Standard ``.grd`` files are **2D** (lon × lat) with one scalar per node (e.g. thickness or
        interface depth in m), not (lon, lat, depth, conductivity).
        """
        lon_c, lat_c = self.build_prior_lon_lat_at_y_centers(
            yn, sort_by=sort_by, mt_objects=mt_objects, extrapolate=extrapolate
        )
        out: Dict[str, Any] = {"lon": lon_c, "lat": lat_c}
        for k, v in prior_fields.items():
            if v is not None:
                out[k] = v
        return out

    def _decimate_frequencies_inplace(self) -> None:
        """Reduce each mt's frequency/Z/Z_err to n_freq_target log-uniformly spaced frequencies."""
        if not self.mt_objects or self.n_freq_target is None:
            return
        mt0 = self.mt_objects[0]
        freqs = np.asarray(mt0.frequency, dtype=float)
        if freqs.size <= self.n_freq_target:
            return
        # Log-uniform target frequencies
        fmin, fmax = float(np.nanmin(freqs)), float(np.nanmax(freqs))
        if fmin <= 0 or fmax <= 0:
            return
        f_target = np.logspace(np.log10(fmin), np.log10(fmax), self.n_freq_target)
        # For each target, pick closest index in original freqs
        idx = np.unique(
            np.array(
                [np.abs(freqs - ft).argmin() for ft in f_target],
                dtype=np.intp,
            )
        )
        idx = np.sort(idx)
        for mt in self.mt_objects:
            f = np.asarray(mt.frequency, dtype=float)
            mt.frequency = np.array(f[idx])
            mt.Z = np.asarray(mt.Z, dtype=np.complex128)[idx]
            if getattr(mt, "Z_err", None) is not None:
                mt.Z_err = np.asarray(mt.Z_err, dtype=float)[idx]
            if getattr(mt, "rho", None) is not None:
                mt.rho = np.asarray(mt.rho, dtype=float)[idx]
            if getattr(mt, "phs", None) is not None:
                mt.phs = np.asarray(mt.phs, dtype=float)[idx]
            if getattr(mt, "rho_err", None) is not None:
                mt.rho_err = np.asarray(mt.rho_err, dtype=float)[idx]
            if getattr(mt, "phs_err", None) is not None:
                mt.phs_err = np.asarray(mt.phs_err, dtype=float)[idx]
            if getattr(mt, "rho_noise_std_log10", None) is not None:
                mt.rho_noise_std_log10 = np.asarray(mt.rho_noise_std_log10, dtype=float)[idx]
            if getattr(mt, "phs_noise_std_norm", None) is not None:
                mt.phs_noise_std_norm = np.asarray(mt.phs_noise_std_norm, dtype=float)[idx]
            if getattr(mt, "T", None) is not None:
                mt.T = np.asarray(mt.T, dtype=np.complex128)[idx]

    @classmethod
    def _mv_per_km_per_nt_to_ohm_scale(cls) -> float:
        """Convert impedance from (mV/km)/nT to ohm.

        EDI commonly uses E in mV/km and B in nT:
        - E[V/m] = E[mV/km] * 1e-6
        - H[A/m] = B[T]/mu0 = (B[nT]*1e-9)/mu0
        So Z[ohm] = (E/H) = Z[(mV/km)/nT] * (1e-6) / (1e-9/mu0) = Z * (mu0 * 1e3)
        """

        return cls._mu0() * 1e3

    @classmethod
    def _normalize_edi_impedance_unit(cls, unit: str) -> str:
        u = str(unit).strip().lower()
        u = u.replace(" ", "")
        u = u.replace("\\", "/")
        u = u.replace("per", "/")
        # common aliases
        if u in {"omega", "ohms", "ohm", "si"}:
            return "ohm"
        if u in {"mv/km/nt", "mvkmnt", "mv/km/nT".lower(), "mvperkmpernt", "mv/km/nanoTesla".lower()}:
            return "mv/km/nt"
        if u in {"auto", "detect"}:
            return "auto"
        return u

    def _maybe_scale_impedance_units_inplace(self, mt: "PrepareData.CustomMT") -> bool:
        """Scale mt.Z and mt.Z_err so that impedance is in ohms.

        Returns True if scaling was applied.
        """

        unit = self._normalize_edi_impedance_unit(self.edi_impedance_unit)
        if unit not in {"auto", "ohm", "mv/km/nt"}:
            raise ValueError(
                f"Unknown edi_impedance_unit={self.edi_impedance_unit!r}. "
                "Use 'auto', 'ohm', or 'mv/km/nt'."
            )

        z = np.asarray(getattr(mt, "Z", None))
        if z is None or z.size == 0:
            return False

        scale = 1.0
        applied_unit = "ohm"

        if unit == "ohm":
            scale = 1.0
        elif unit == "mv/km/nt":
            scale = float(self._mv_per_km_per_nt_to_ohm_scale())
            applied_unit = "mv/km/nt->ohm"
        else:
            # Heuristic detection: if |Z| is extremely large, it's almost surely (mV/km)/nT.
            zabs_max = float(np.nanmax(np.abs(z)))
            if np.isfinite(zabs_max) and zabs_max > 1e3:
                scale = float(self._mv_per_km_per_nt_to_ohm_scale())
                applied_unit = "auto(mv/km/nt->ohm)"

        if scale == 1.0:
            setattr(mt, "impedance_unit", "ohm")
            return False

        mt.Z = np.asarray(mt.Z, dtype=np.complex128) * scale
        if getattr(mt, "Z_err", None) is not None:
            mt.Z_err = np.asarray(mt.Z_err, dtype=float) * abs(scale)
        setattr(mt, "impedance_unit", applied_unit)
        return True

    # -------------------------- impedance -> rho/phase (+ errors) --------------------------

    @staticmethod
    def _mu0() -> float:
        return float(4e-7 * np.pi)

    @classmethod
    def impedance_to_rho_phase(
        cls,
        freqs_hz: np.ndarray,
        Z: np.ndarray,
        Z_err: Optional[np.ndarray] = None,
        *,
        eps: float = 1e-8,
        eps_rho: float = 1e-6,
        eps_z: float = 1e-12,
        max_noise_std: float = 1.0,
        err_component_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """把阻抗张量 Z 转成视电阻率 rho 与相位 phs，并可选传播误差。

        公式（与用户给出的 torch 版本一致）：
        - rho = |Z|^2 / (omega * mu0)
        - phs_raw = atan2(Im(Z), Re(Z)) [deg]  (range: (-180, 180])
        - phs = fold(phs_raw) into principal phase [0, 90]

        误差传播（假设 Re/Im 独立，且两者 std-dev 相同）：
        - d(rho) = sqrt( (∂rho/∂Zr * σr)^2 + (∂rho/∂Zi * σi)^2 )
        - d(phi) = sqrt( (∂phi/∂Zr * σr)^2 + (∂phi/∂Zi * σi)^2 )

        返回：rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm

        Notes
        -----
        - `Z_err` 来自 EDI 的 `Z..VAR`：在 ``read_custom_edi`` 中，``CAF*`` 台站按块内为 **std-dev**
          直接使用；其余台站按 **方差** 读入并已 ``sqrt`` 为 std-dev。此处一律按“每个复阻抗
          分量的 std-dev”做误差传播。若仍需缩放，可用 ``err_component_scale``。
        """

        freqs_hz = np.asarray(freqs_hz, dtype=float)
        Z = np.asarray(Z)
        if Z.ndim != 3 or Z.shape[1:] != (2, 2):
            raise ValueError(f"Z must have shape (n_freq,2,2), got {Z.shape}")
        if freqs_hz.ndim != 1 or freqs_hz.shape[0] != Z.shape[0]:
            raise ValueError("freqs_hz must be 1D and match Z.shape[0]")

        omega = 2.0 * np.pi * freqs_hz  # (n,)
        denom = omega * cls._mu0()      # (n,)
        denom_3d = denom[:, None, None]

        rho = (np.abs(Z) ** 2) / denom_3d
        phs_raw = np.degrees(np.arctan2(Z.imag, Z.real))
        # Fold to principal phase in [0, 90] to match inversion normalization (phase/90)
        phs_0_180 = np.mod(phs_raw, 180.0)
        phs = np.minimum(phs_0_180, 180.0 - phs_0_180)

        if Z_err is None:
            return rho, phs, None, None, None, None

        Z_err = np.asarray(Z_err, dtype=float)
        if Z_err.shape != Z.shape:
            raise ValueError(f"Z_err must have same shape as Z, got {Z_err.shape} vs {Z.shape}")

        sigma = np.maximum(Z_err, 0.0) * float(err_component_scale)
        sigma_r = sigma
        sigma_i = sigma

        Zr = Z.real
        Zi = Z.imag

        # ---------- rho std-dev ----------
        dRho_dZr = 2.0 * Zr / denom_3d
        dRho_dZi = 2.0 * Zi / denom_3d
        rho_err = np.sqrt((dRho_dZr * sigma_r) ** 2 + (dRho_dZi * sigma_i) ** 2)

        # ---------- phi std-dev (avoid 1/|Z|^2 explosion) ----------
        Zabs = np.abs(Z)
        Zabs_safe = np.maximum(Zabs, float(eps_z))
        dPhi_dZr = -Zi / (Zabs_safe**2)
        dPhi_dZi = Zr / (Zabs_safe**2)
        phi_err_rad = np.sqrt((dPhi_dZr * sigma_r) ** 2 + (dPhi_dZi * sigma_i) ** 2)
        phs_err = phi_err_rad * (180.0 / np.pi)

        # ---------- Dimensionless noise std-dev (for chi^2 / RMS) ----------
        rho_obs_safe = np.maximum(rho, float(eps_rho))
        rho_noise_std_log10 = np.clip(
            rho_err / (rho_obs_safe * np.log(10.0)),
            float(eps),
            float(max_noise_std),
        )
        phs_noise_std_norm = np.clip(
            phs_err / 90.0,
            float(eps),
            float(max_noise_std),
        )

        return rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm

    def compute_rho_phase(
        self,
        mt: "PrepareData.CustomMT",
        *,
        err_component_scale: float = 1.0,
    ) -> None:
        rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm = self.impedance_to_rho_phase(
            freqs_hz=mt.frequency,
            Z=mt.Z,
            Z_err=mt.Z_err,
            err_component_scale=err_component_scale,
        )
        mt.rho = rho
        mt.phs = phs
        mt.rho_err = rho_err
        mt.phs_err = phs_err
        mt.rho_noise_std_log10 = rho_noise_std_log10
        mt.phs_noise_std_norm = phs_noise_std_norm

    def compute_rho_phase_all(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        err_component_scale: float = 1.0,
    ) -> None:
        if mt_objects is None:
            mt_objects = self.mt_objects
        for mt in mt_objects:
            self.compute_rho_phase(mt, err_component_scale=err_component_scale)

    # -------------------------- export for inversion (torch) --------------------------

    @staticmethod
    def _torch_complex_dtype_from_float(dtype):
        """Map float dtype -> complex dtype (torch)."""

        try:
            import torch  # type: ignore[import-untyped]
        except Exception as e:
            raise ImportError("export_data_dict_for_2d_inversion() requires torch") from e

        if dtype is None:
            return torch.complex128
        if dtype == torch.float32:
            return torch.complex64
        return torch.complex128



    def plot_pseudosection(
        self,
        components: Optional[List[str]] = None,
        cmap_rho: str = "jet_r",
        cmap_phs: str = "jet",
        profile_extend_km: float = 5.0,
        y_axis: str = "period",
    ):
        """Plot apparent resistivity and phase pseudosections (observation data).

        Call after run_all_simple() or export_data_dict_for_2d_inversion().
        Uses inv_freqs_torch, inv_stations_torch, inv_data_dict if available;
        otherwise calls export_data_dict_for_2d_inversion() first.

        X-axis: distance along profile (km), 0-based.
        Y-axis: period (s) or frequency (Hz), log scale, increasing period downward.

        Parameters
        ----------
        components : list of str, optional
            Which components to plot: "rhoxy", "phsxy", "rhoyx", "phsyx".
            Default: ["rhoxy", "phsxy", "rhoyx", "phsyx"] (2x2: TE + TM).
        cmap_rho : str
            Colormap for resistivity. Default "jet_r".
        cmap_phs : str
            Colormap for phase. Default "jet".
        profile_extend_km : float
            Horizontal extent beyond stations (km). Default 5.
        y_axis : str
            "period" (default) or "freq" for Y-axis.
        """
        if not hasattr(self, "inv_data_dict") or self.inv_data_dict is None:
            self.export_data_dict_for_2d_inversion()
        obs_data = self.inv_data_dict["obs_data"]
        freqs_t = self.inv_freqs_torch
        stations_t = self.inv_stations_torch

        def _to_np(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            return np.asarray(x)

        st_km = _to_np(stations_t) / 1000.0
        freqs_hz = _to_np(freqs_t)
        st_min, st_max = float(st_km.min()), float(st_km.max())
        offset_km = st_min - profile_extend_km
        st_x = st_km - offset_km

        if components is None:
            components = ["rhoxy", "phsxy", "rhoyx", "phsyx"]
        available = [k for k in components if k in obs_data]
        if not available:
            print(f"No requested components. Available: {list(obs_data.keys())}")
            return

        if y_axis == "period":
            y_vals = 1.0 / freqs_hz
            y_label = "Period (s)"
        else:
            y_vals = freqs_hz
            y_label = "Frequency (Hz)"

        n_f, n_s = _to_np(obs_data[available[0]]).shape
        if n_s > 1:
            x_mid = 0.5 * (st_x[:-1] + st_x[1:])
            x_edges = np.concatenate([
                [2 * st_x[0] - x_mid[0]], x_mid, [2 * st_x[-1] - x_mid[-1]],
            ])
        else:
            x_edges = np.array([st_x[0] - 0.5, st_x[0] + 0.5])
        if n_f > 1:
            log_y = np.log10(y_vals)
            y_mid = 0.5 * (log_y[:-1] + log_y[1:])
            y_edges = 10 ** np.concatenate([
                [2 * log_y[0] - y_mid[0]], y_mid, [2 * log_y[-1] - y_mid[-1]],
            ])
        else:
            y_edges = np.array([y_vals[0] * 0.5, y_vals[0] * 1.5])
        X, Y = np.meshgrid(x_edges, y_edges)

        n_plot = len(available)
        n_col = min(2, n_plot)
        n_row = (n_plot + n_col - 1) // n_col
        fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row), sharex=True, sharey=True)
        if n_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, key in enumerate(available):
            ax = axes[idx]
            data = _to_np(obs_data[key])
            masked_data = np.ma.masked_invalid(data)
            if "rho" in key.lower():
                data_plot = np.ma.log10(np.ma.clip(masked_data, 1e-2, 1e6))
                im = ax.pcolormesh(X, Y, data_plot, cmap=cmap_rho, shading="flat")
                plt.colorbar(im, ax=ax, label=r"log$_{10}$ $\rho_a$ (Ω·m)")
            else:
                data_plot = np.ma.clip(masked_data, -90, 90)
                im = ax.pcolormesh(X, Y, data_plot, cmap=cmap_phs, shading="flat")
                plt.colorbar(im, ax=ax, label="Phase (°)")
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_xlabel("Distance along profile (km)")
            ax.set_ylabel(y_label)
            ax.set_title(key)
            ax.set_xlim(0, (st_max - st_min) + 2 * profile_extend_km)

        for j in range(n_plot, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    # -------------------------- export: mt_object -> txt --------------------------

    @staticmethod
    def export_mt_object_to_txt(
        mt_object: Any,
        txt_path: Union[str, Path],
        *,
        sort_by_frequency: bool = True,
        include_derived: bool = True,
        include_errors: bool = True,
        float_format: str = ".6e",
        encoding: str = "utf-8",
        overwrite: bool = True,
    ) -> Path:
        """把单个 MT 对象导出为 txt（便于检查/复现）。

        兼容两类对象：
        - 本模块的 `PrepareData.CustomMT`（推荐）
        - 外部库对象（如 mtpy 风格）：支持 `obj.Z.z` / `obj.Z.z_err` 取值

        输出为“带注释头 + 表格数据”的纯文本。表格列会尽量覆盖：
        - freq_hz, period_s
        - Z(2x2) 的实部/虚部（按 xx,xy,yx,yy 顺序展开）
        - 可选：Z_err(2x2)
        - 可选：rho/phs 及其误差、噪声归一化 std-dev
        """

        def _maybe_get(obj: Any, names: Sequence[str]) -> Any:
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return None

        def _as_array(x: Any, *, dtype=None) -> Optional[np.ndarray]:
            if x is None:
                return None
            try:
                return np.asarray(x, dtype=dtype)
            except Exception:
                return None

        txt_path = Path(txt_path)
        if txt_path.exists() and (not overwrite):
            raise FileExistsError(f"Target txt already exists: {txt_path}")
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # -------- meta --------
        station_id = _maybe_get(mt_object, ["station_id", "station", "name", "id"])
        lat = _maybe_get(mt_object, ["lat", "latitude"])
        lon = _maybe_get(mt_object, ["lon", "long", "longitude"])
        profile_pos_m = _maybe_get(mt_object, ["profile_pos_m", "profile_pos", "profile_position_m"])

        # -------- frequency --------
        freqs = _maybe_get(mt_object, ["frequency", "freq", "freqs", "frequencies"])
        if freqs is None:
            # mtpy sometimes stores period; try to invert
            period = _maybe_get(mt_object, ["period", "periods"])
            period = _as_array(period, dtype=float)
            if period is not None and period.ndim == 1 and period.size > 0:
                freqs = 1.0 / period
        freqs = _as_array(freqs, dtype=float)

        # -------- impedance tensor + err --------
        Z_raw = _maybe_get(mt_object, ["Z", "z"])
        Z_arr: Optional[np.ndarray] = None
        Z_err_arr: Optional[np.ndarray] = None
        if isinstance(Z_raw, np.ndarray) or np.isscalar(Z_raw):
            Z_arr = _as_array(Z_raw)
        else:
            # mtpy: mt.Z.z, mt.Z.z_err
            Z_arr = _as_array(_maybe_get(Z_raw, ["z", "Z", "impedance"]))
            Z_err_arr = _as_array(_maybe_get(Z_raw, ["z_err", "Z_err", "zerr", "error"]))

        if Z_arr is None:
            Z_arr = _as_array(_maybe_get(mt_object, ["Z", "z_array", "z_tensor"]))
        if Z_err_arr is None:
            Z_err_arr = _as_array(_maybe_get(mt_object, ["Z_err", "z_err", "ZERR"]))

        if freqs is None or freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("mt_object has no valid frequency axis (expected 1D non-empty `frequency`)")
        if Z_arr is None:
            raise ValueError("mt_object has no valid impedance tensor (expected `Z` with shape (n,2,2))")
        Z_arr = np.asarray(Z_arr)
        if Z_arr.ndim != 3 or Z_arr.shape[1:] != (2, 2):
            raise ValueError(f"Z must have shape (n,2,2), got {Z_arr.shape}")
        if Z_arr.shape[0] != freqs.shape[0]:
            raise ValueError(f"frequency length {freqs.shape[0]} doesn't match Z length {Z_arr.shape[0]}")

        if Z_err_arr is not None:
            Z_err_arr = np.asarray(Z_err_arr, dtype=float)
            if Z_err_arr.shape != Z_arr.shape:
                Z_err_arr = None

        # -------- derived (rho/phs + errors/noise std) --------
        rho = _as_array(_maybe_get(mt_object, ["rho", "apparent_resistivity"]))
        phs = _as_array(_maybe_get(mt_object, ["phs", "phase"]))
        rho_err = _as_array(_maybe_get(mt_object, ["rho_err", "rho_error"]))
        phs_err = _as_array(_maybe_get(mt_object, ["phs_err", "phs_error"]))
        rho_noise_std_log10 = _as_array(_maybe_get(mt_object, ["rho_noise_std_log10", "rho_noise_std"]))
        phs_noise_std_norm = _as_array(_maybe_get(mt_object, ["phs_noise_std_norm", "phs_noise_std"]))

        rho_ok = isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape
        phs_ok = isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape
        rho_err_ok = isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape
        phs_err_ok = isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape
        rn_ok = isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape
        pn_ok = isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape

        if include_derived and ((not rho_ok) or (not phs_ok) or (include_errors and ((not rho_err_ok) or (not phs_err_ok) or (not rn_ok) or (not pn_ok)))):
            rho_calc, phs_calc, rho_err_calc, phs_err_calc, rn_calc, pn_calc = PrepareData.impedance_to_rho_phase(
                freqs_hz=freqs,
                Z=Z_arr,
                Z_err=Z_err_arr if include_errors else None,
            )
            if (not rho_ok) and rho_calc is not None:
                rho = rho_calc
            if (not phs_ok) and phs_calc is not None:
                phs = phs_calc
            if include_errors and (not rho_err_ok) and (rho_err_calc is not None):
                rho_err = rho_err_calc
            if include_errors and (not phs_err_ok) and (phs_err_calc is not None):
                phs_err = phs_err_calc
            if include_errors and (not rn_ok) and (rn_calc is not None):
                rho_noise_std_log10 = rn_calc
            if include_errors and (not pn_ok) and (pn_calc is not None):
                phs_noise_std_norm = pn_calc

        # -------- optional sort by frequency --------
        order = np.arange(freqs.size)
        if sort_by_frequency:
            order = np.argsort(freqs)
            freqs = freqs[order]
            Z_arr = Z_arr[order]
            if Z_err_arr is not None:
                Z_err_arr = Z_err_arr[order]
            if isinstance(rho, np.ndarray) and rho.shape[:1] == (order.size,):
                rho = rho[order]
            if isinstance(phs, np.ndarray) and phs.shape[:1] == (order.size,):
                phs = phs[order]
            if isinstance(rho_err, np.ndarray) and rho_err.shape[:1] == (order.size,):
                rho_err = rho_err[order]
            if isinstance(phs_err, np.ndarray) and phs_err.shape[:1] == (order.size,):
                phs_err = phs_err[order]
            if isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape[:1] == (order.size,):
                rho_noise_std_log10 = rho_noise_std_log10[order]
            if isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape[:1] == (order.size,):
                phs_noise_std_norm = phs_noise_std_norm[order]

        # -------- write --------
        def _fmt(x: float) -> str:
            if x is None:
                return "nan"
            try:
                xf = float(x)
            except Exception:
                return "nan"
            if not np.isfinite(xf):
                return "nan"
            return format(xf, float_format)

        def _flatten_2x2(M: np.ndarray) -> tuple[float, float, float, float]:
            return (float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]))

        cols: list[str] = [
            "freq_hz",
            "period_s",
            "Zxx_real",
            "Zxx_imag",
            "Zxy_real",
            "Zxy_imag",
            "Zyx_real",
            "Zyx_imag",
            "Zyy_real",
            "Zyy_imag",
        ]
        if include_errors and (Z_err_arr is not None):
            cols += ["Zxx_err", "Zxy_err", "Zyx_err", "Zyy_err"]
        if include_derived and isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape:
            cols += ["rho_xx", "rho_xy", "rho_yx", "rho_yy"]
        if include_derived and isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape:
            cols += ["phs_xx_deg", "phs_xy_deg", "phs_yx_deg", "phs_yy_deg"]
        if include_errors and isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape:
            cols += ["rho_err_xx", "rho_err_xy", "rho_err_yx", "rho_err_yy"]
        if include_errors and isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape:
            cols += ["phs_err_xx_deg", "phs_err_xy_deg", "phs_err_yx_deg", "phs_err_yy_deg"]
        if include_errors and isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape:
            cols += [
                "rho_noise_std_log10_xx",
                "rho_noise_std_log10_xy",
                "rho_noise_std_log10_yx",
                "rho_noise_std_log10_yy",
            ]
        if include_errors and isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape:
            cols += [
                "phs_noise_std_norm_xx",
                "phs_noise_std_norm_xy",
                "phs_noise_std_norm_yx",
                "phs_noise_std_norm_yy",
            ]

        with open(txt_path, "w", encoding=encoding, newline="\n") as f:
            f.write("# export_mt_object_to_txt\n")
            if station_id is not None:
                f.write(f"# station_id: {station_id}\n")
            if lat is not None:
                f.write(f"# lat: {lat}\n")
            if lon is not None:
                f.write(f"# lon: {lon}\n")
            if profile_pos_m is not None:
                f.write(f"# profile_pos_m: {profile_pos_m}\n")
            f.write(f"# n_freq: {int(freqs.size)}\n")
            f.write("# columns: " + "\t".join(cols) + "\n")
            f.write("\t".join(cols) + "\n")

            for i in range(freqs.size):
                freq = float(freqs[i])
                period_s = 1.0 / freq if np.isfinite(freq) and freq != 0 else float("nan")
                z = Z_arr[i]
                zxx, zxy, zyx, zyy = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
                row: list[str] = [
                    _fmt(freq),
                    _fmt(period_s),
                    _fmt(zxx.real),
                    _fmt(zxx.imag),
                    _fmt(zxy.real),
                    _fmt(zxy.imag),
                    _fmt(zyx.real),
                    _fmt(zyx.imag),
                    _fmt(zyy.real),
                    _fmt(zyy.imag),
                ]

                if include_errors and (Z_err_arr is not None):
                    ze = Z_err_arr[i]
                    exx, exy, eyx, eyy = _flatten_2x2(ze)
                    row += [_fmt(exx), _fmt(exy), _fmt(eyx), _fmt(eyy)]

                if include_derived and isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape:
                    r = rho[i]
                    rxx, rxy, ryx, ryy = _flatten_2x2(r)
                    row += [_fmt(rxx), _fmt(rxy), _fmt(ryx), _fmt(ryy)]

                if include_derived and isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape:
                    p = phs[i]
                    pxx, pxy, pyx, pyy = _flatten_2x2(p)
                    row += [_fmt(pxx), _fmt(pxy), _fmt(pyx), _fmt(pyy)]

                if include_errors and isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape:
                    re = rho_err[i]
                    rexx, rexy, reyx, reyy = _flatten_2x2(re)
                    row += [_fmt(rexx), _fmt(rexy), _fmt(reyx), _fmt(reyy)]

                if include_errors and isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape:
                    pe = phs_err[i]
                    pexx, pexy, peyx, peyy = _flatten_2x2(pe)
                    row += [_fmt(pexx), _fmt(pexy), _fmt(peyx), _fmt(peyy)]

                if include_errors and isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape:
                    rn = rho_noise_std_log10[i]
                    rnxx, rnxy, rnyx, rnyy = _flatten_2x2(rn)
                    row += [_fmt(rnxx), _fmt(rnxy), _fmt(rnyx), _fmt(rnyy)]

                if include_errors and isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape:
                    pn = phs_noise_std_norm[i]
                    pnxx, pnxy, pnyx, pnyy = _flatten_2x2(pn)
                    row += [_fmt(pnxx), _fmt(pnxy), _fmt(pnyx), _fmt(pnyy)]

                f.write("\t".join(row) + "\n")

        return txt_path

    # -------------------------- phase tensor --------------------------

    @staticmethod
    def phase_tensor(Z: np.ndarray) -> Optional[np.ndarray]:
        X = np.asarray(Z).real
        Y = np.asarray(Z).imag
        try:
            Phi = np.linalg.inv(X) @ Y
        except Exception:
            return None
        return Phi

    @classmethod
    def phase_tensor_skew(cls, Z: np.ndarray) -> float:
        Phi = cls.phase_tensor(Z)
        if Phi is None:
            return float("nan")

        pxx, pxy = Phi[0]
        pyx, pyy = Phi[1]

        beta = 0.5 * np.degrees(np.arctan2((pxy - pyx), (pxx + pyy)))
        return float(beta)

    @classmethod
    def strike_from_phase_tensor(cls, Z: np.ndarray) -> float:
        Phi = cls.phase_tensor(Z)
        if Phi is None:
            return float("nan")

        pxx = Phi[0, 0]
        pxy = Phi[0, 1]
        pyx = Phi[1, 0]
        pyy = Phi[1, 1]
        strike = 0.5 * np.degrees(np.arctan2(2 * (pxy + pyx), pxx - pyy))
        return float(strike)

    @classmethod
    def station_strike(
        cls,
        mt: "PrepareData.CustomMT",
        skew_threshold: float = 5.0,
    ) -> np.ndarray:
        """返回所有频率的走向，但不符合2D条件的设为 NaN。"""

        strikes: list[float] = []
        for i in range(len(mt.frequency)):
            s = cls.strike_from_phase_tensor(mt.Z[i])
            skew = cls.phase_tensor_skew(mt.Z[i])
            if np.isfinite(s) and np.isfinite(skew) and abs(skew) < float(skew_threshold):
                strikes.append(float(s))
            else:
                strikes.append(float("nan"))
        return np.array(strikes, dtype=float)

    def estimate_regional_strike(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        *,
        skew_threshold: float = 5.0,
    ) -> Tuple[float, np.ndarray]:
        """Pool phase-tensor strikes across stations and frequencies.

        For each station/frequency, :meth:`station_strike` returns a strike (deg,
        magnetic reference) when the phase tensor skew is below ``skew_threshold``;
        otherwise NaN.

        **90° ambiguity handling**:
        - **With Tipper**: unwrap along period (continuity), then median; Tipper cross-check in :meth:`compute_strike`.
        - **Without Tipper**: fold to [0, 90) (WPS-style % 90 with negative fix), then median.

        Returns
        -------
        regional_strike_magnetic : float
            Central estimate in degrees (magnetic north).
        all_strikes_magnetic : np.ndarray
            1D array of per-frequency strikes in station order.
        """
        if not mt_objects:
            return float("nan"), np.array([], dtype=float)

        has_tipper = any(getattr(mt, "T", None) is not None for mt in mt_objects)

        blocks: list[np.ndarray] = []
        for mt in mt_objects:
            s = self.station_strike(mt, skew_threshold=skew_threshold)
            if has_tipper:
                period = 1.0 / np.asarray(mt.frequency, dtype=float)
                s_proc = self._unwrap_strike_90(period, s)
            else:
                # WPS-style: fold to [0, 90) via % 90
                s_proc = np.where(np.isfinite(s), np.asarray(s, dtype=float) % 90.0, np.nan)
            blocks.append(np.asarray(s_proc, dtype=float))

        all_strikes_magnetic = np.concatenate(blocks) if blocks else np.array([], dtype=float)
        valid = all_strikes_magnetic[np.isfinite(all_strikes_magnetic)]
        if valid.size == 0:
            return float("nan"), all_strikes_magnetic

        regional = float(np.median(valid))
        return regional, all_strikes_magnetic

    @staticmethod
    def parkinson_arrow_azimuth(T: np.ndarray) -> np.ndarray:
        """Compute Parkinson real induction arrow azimuth (deg), 0–360 from North.

        T: (n_freq, 2) complex, T[:,0]=Tzx, T[:,1]=Tzy. Arrow points toward conductor
        = -Real(T). X=North, Y=East; azimuth = atan2(-Ty_real, -Tx_real).
        """
        Tx_real = np.asarray(T[:, 0]).real
        Ty_real = np.asarray(T[:, 1]).real
        arrow_x = -Tx_real
        arrow_y = -Ty_real
        azimuth_rad = np.arctan2(arrow_y, arrow_x)
        azimuth_deg = np.degrees(azimuth_rad)
        return azimuth_deg % 360.0

    @staticmethod
    def wrap_deg_180(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map angle(s) to (-180, 180] degrees (azimuth / strike display).

        Use this after ``strike_magnetic + declination`` so true-north strikes stay
        consistent with map conventions. **Do not** use ``% 90``: that mixes TE/TM
        ambiguity with declination and breaks negative angles (e.g. in Python
        ``(-40) % 90 == 50``).
        """
        d = np.asarray(deg, dtype=float)
        w = (d + 180.0) % 360.0 - 180.0
        if w.ndim == 0:
            return float(w)
        return w

    def compute_strike(self) -> Tuple[float, float]:
        if not self.mt_objects:
            raise RuntimeError("mt_objects is empty; call load_mt_objects() first")

        regional_strike_magnetic, all_strikes_magnetic = self.estimate_regional_strike(
            self.mt_objects,
            skew_threshold=self.strike_skew_threshold,
        )
        self.regional_strike_magnetic = float(regional_strike_magnetic)
        self.all_strikes_magnetic = np.asarray(all_strikes_magnetic, dtype=float)
        print(
            "Estimated regional strike (magnetic reference) =",
            self.regional_strike_magnetic,
        )

        decl = float(self.mag_declination_deg)
        print(f"Magnetic declination (manual, east positive) = {decl:.3f} deg")

        # True north: add declination, then wrap to (-180, 180] (not % 90).
        self.regional_strike_true = float(
            self.wrap_deg_180(float(self.regional_strike_magnetic) + float(decl))
        )
        self.all_strikes_true = self.wrap_deg_180(
            np.asarray(self.all_strikes_magnetic, dtype=float) + float(decl)
        )
        print("Estimated regional strike (true north reference) =", self.regional_strike_true)

        # Tipper 90° cross-check: if arrow direction suggests wrong strike, flip by 90°
        arrow_azimuths: list[float] = []
        for mt in self.mt_objects:
            if getattr(mt, "T", None) is not None:
                az = self.parkinson_arrow_azimuth(mt.T)
                arrow_azimuths.extend(az[np.isfinite(az)].tolist())

        if arrow_azimuths:
            rads = np.radians(arrow_azimuths)
            mean_u = np.mean(np.sin(rads))
            mean_v = np.mean(np.cos(rads))
            mean_arrow_azimuth = np.degrees(np.arctan2(mean_u, mean_v)) % 360.0
            inferred_strike_1 = (mean_arrow_azimuth + 90.0) % 180.0
            inferred_strike_2 = (mean_arrow_azimuth - 90.0) % 180.0
            current_strike = float(self.regional_strike_true) % 180.0
            diff = min(
                abs(current_strike - inferred_strike_1),
                abs(current_strike - inferred_strike_2),
            )
            if diff > 45.0:
                print("⚠ Tipper suggests 90° ambiguity flip; correcting strike.")
                self.regional_strike_true = float(
                    self.wrap_deg_180(float(self.regional_strike_true) + 90.0)
                )
                self.regional_strike_magnetic = float(
                    self.wrap_deg_180(float(self.regional_strike_magnetic) + 90.0)
                )
                print("  Corrected regional strike (true north) =", self.regional_strike_true)

        return self.regional_strike_magnetic, self.regional_strike_true

    # -------------------------- strike plotting --------------------------

    def plot_strike_period(self, mt: "PrepareData.CustomMT", unwrap_90: bool = False) -> None:
        """Plot strike vs period for a single station.

        unwrap_90: If True, unwrap by 90° for continuity. Can hide real variation.
        """
        s = self.station_strike(mt)
        period = 1 / mt.frequency
        if unwrap_90:
            s = self._unwrap_strike_90(period, s)
        valid_mask = np.isfinite(s)

        plt.figure(figsize=(10, 6))
        plt.semilogx(period[valid_mask], s[valid_mask], "b.", markersize=8, alpha=0.7)
        plt.xlabel("Period (s)", fontsize=12)
        plt.ylabel("Strike (deg)", fontsize=12)
        plt.title("Strike vs Period", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rose(strikes: np.ndarray) -> None:
        theta = np.deg2rad(np.asarray(strikes, dtype=float))
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.hist(theta, bins=36)
        ax.set_title("Strike Rose Diagram")
        plt.show()

    def plot_skew_period(self, mt: "PrepareData.CustomMT") -> None:
        skews = [self.phase_tensor_skew(mt.Z[i]) for i in range(len(mt.frequency))]
        period = 1 / mt.frequency
        plt.figure()
        plt.semilogx(period, skews, ".")
        plt.xlabel("Period (s)")
        plt.ylabel("Phase Tensor Skew (deg)")
        plt.grid()
        plt.show()

    @staticmethod
    def _unwrap_strike_90(period: np.ndarray, strike: np.ndarray) -> np.ndarray:
        """Unwrap strike by 90° to minimize jumps (MT strike has 90° ambiguity).
        Sorts by period, then for each point picks θ or θ±90° to be closest to previous.
        """
        valid = np.isfinite(strike) & np.isfinite(period) & (period > 0)
        if not np.any(valid):
            return strike.copy()
        idx = np.argsort(period[valid])
        s = strike[valid][idx].astype(float)
        unwrapped = np.full(len(s), np.nan)
        unwrapped[0] = s[0]
        for i in range(1, len(s)):
            candidates = [s[i], s[i] + 90, s[i] - 90]
            best = min(candidates, key=lambda c: abs(c - unwrapped[i - 1]))
            unwrapped[i] = best
        out = strike.copy()
        orig_valid_idx = np.where(valid)[0][idx]
        out[orig_valid_idx] = unwrapped
        return out

    def plot_all_strikes_subplots(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        n_cols: int = 4,
        unwrap_90: bool = False,
    ) -> None:
        """Plot strike vs period for each station.

        unwrap_90: If True, unwrap strike by 90° to reduce discontinuity from ambiguity.
            Use only when jumps are known to be from 90° ambiguity; it can hide real
            depth-dependent strike variation (e.g., 3D structure).
        """
        n_stations = len(mt_objects)
        n_rows = (n_stations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        fig.suptitle("Strike vs Period - All Stations", fontsize=16, y=1.02)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, mt in enumerate(mt_objects):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_mask = np.isfinite(s)

            ax.semilogx(period[valid_mask], s[valid_mask], "b.", markersize=5, alpha=0.6)
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Strike (deg)")
            ax.set_title(f"Station {idx + 1}")
            ax.grid(True, alpha=0.3)
            if unwrap_90 and np.any(valid_mask):
                ymin, ymax = float(np.nanmin(s[valid_mask])), float(np.nanmax(s[valid_mask]))
                margin = max(10, (ymax - ymin) * 0.1)
                ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.set_ylim(-90, 90)

        for idx in range(n_stations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_comprehensive_strike_analysis(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        regional_strike: Optional[float] = None,
        all_strikes: Optional[np.ndarray] = None,
        unwrap_90: bool = False,
    ) -> None:
        """Comprehensive strike analysis (Strike vs Period, Rose, histogram, boxplot).

        unwrap_90: If True, unwrap strike by 90° for continuity. Can hide real variation.
        """
        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, (1, 2))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(mt_objects)))

        all_strikes_from_curves: list[float] = []
        for i, mt in enumerate(mt_objects):
            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_mask = np.isfinite(s)

            ax1.semilogx(
                period[valid_mask],
                s[valid_mask],
                ".",
                color=colors[i],
                markersize=4,
                alpha=0.5,
                label=f"St{i + 1}" if i < 10 else None,
            )

            if all_strikes is None:
                all_strikes_from_curves.extend(s[valid_mask].tolist())

        ax1.set_xlabel("Period (s)", fontsize=12)
        ax1.set_ylabel("Strike (deg)", fontsize=12)
        ax1.set_title("All Stations: Strike vs Period", fontsize=14)
        ax1.grid(True, alpha=0.3)
        if unwrap_90 and all_strikes_from_curves:
            arr = np.asarray(all_strikes_from_curves, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                ymin, ymax = float(np.nanmin(finite)), float(np.nanmax(finite))
                margin = max(10, (ymax - ymin) * 0.1)
                ax1.set_ylim(ymin - margin, ymax + margin)
            else:
                ax1.set_ylim(-90, 90)
        else:
            ax1.set_ylim(-90, 90)
        if len(mt_objects) <= 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if all_strikes is None:
            all_strikes_arr = np.asarray(all_strikes_from_curves, dtype=float)
        else:
            all_strikes_arr = np.asarray(all_strikes, dtype=float)

        ax2 = plt.subplot(2, 3, 3, polar=True)
        theta = np.deg2rad(all_strikes_arr[np.isfinite(all_strikes_arr)])
        ax2.hist(theta, bins=36, alpha=0.7)
        ax2.set_title("Rose Diagram (All Stations)", fontsize=12)

        ax3 = plt.subplot(2, 3, 4)
        finite_strikes = all_strikes_arr[np.isfinite(all_strikes_arr)]
        ax3.hist(finite_strikes, bins=30, edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Strike (deg)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Strike Distribution", fontsize=12)
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 3, (5, 6))

        station_data = []
        station_labels = []
        for i, mt in enumerate(mt_objects):
            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_s = s[np.isfinite(s)]
            if len(valid_s) > 0:
                station_data.append(valid_s)
                station_labels.append(f"S{i + 1}")

        bp = ax4.boxplot(station_data, tick_labels=station_labels, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor("lightblue")
            box.set_alpha(0.7)

        ax4.set_xlabel("Station")
        ax4.set_ylabel("Strike (deg)")
        ax4.set_title("Strike Statistics by Station", fontsize=12)
        ax4.grid(True, alpha=0.3)
        if not unwrap_90:
            ax4.set_ylim(-90, 90)

        if regional_strike is None:
            regional_strike = self._select_strike_true_deg()
        if regional_strike is None or not np.isfinite(float(regional_strike)):
            regional_strike, _ = self.estimate_regional_strike(mt_objects)

        ax4.axhline(
            y=float(regional_strike),
            color="r",
            linestyle="--",
            label=f"Regional: {float(regional_strike):.1f}°",
        )
        ax4.axhline(y=float(regional_strike) + 90, color="r", linestyle=":", alpha=0.5)
        ax4.axhline(y=float(regional_strike) - 90, color="r", linestyle=":", alpha=0.5)
        ax4.legend()

        plt.tight_layout()
        plt.show()

    # -------------------------- station projection / profile --------------------------

    @staticmethod
    def project_stations_perpendicular(lats: np.ndarray, lons: np.ndarray, strike: float) -> np.ndarray:
        lat0, lon0 = float(lats[0]), float(lons[0])
        d_lat = (lats - lat0) * 111.132
        d_lon = (lons - lon0) * 111.32 * np.cos(np.deg2rad(lat0))

        perp_azimuth = float(strike) + 90.0
        rad = np.deg2rad(perp_azimuth)
        distances = d_lon * np.sin(rad) + d_lat * np.cos(rad)
        return distances

    @staticmethod
    def compute_centered_profile_pos_m(profile_dist_km: np.ndarray) -> np.ndarray:
        profile_dist_km = np.asarray(profile_dist_km, dtype=float)
        if profile_dist_km.size == 0:
            return profile_dist_km

        dmin = float(np.nanmin(profile_dist_km))
        dmax = float(np.nanmax(profile_dist_km))
        center_km = 0.5 * (dmin + dmax)
        profile_pos_m = (profile_dist_km - center_km) * 1000.0
        return profile_pos_m

    def assign_profile_pos_m(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        strike: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lats = np.array([mt.lat for mt in mt_objects], dtype=float)
        lons = np.array([mt.lon for mt in mt_objects], dtype=float)

        strike_use = self._select_strike_true_deg(strike)
        if strike_use is None or not np.isfinite(float(strike_use)):
            raise ValueError(
                "Strike angle is not available. Provide strike (true-north degrees) explicitly, "
                "or call set_user_strike(), or call compute_strike() first."
            )

        profile_dist_km = self.project_stations_perpendicular(lats, lons, float(strike_use))
        profile_pos_m = self.compute_centered_profile_pos_m(profile_dist_km)

        for mt, pos_m in zip(mt_objects, profile_pos_m):
            mt.profile_pos_m = float(pos_m)

        return profile_pos_m, profile_dist_km

    @staticmethod
    def _coerce_station_ids_for_plot(mt_objects: Sequence["PrepareData.CustomMT"]) -> Tuple[np.ndarray, np.ndarray]:
        raw_ids = [getattr(mt, "station_id", None) for mt in mt_objects]
        numeric_ids: list[int] = []
        labels: list[str] = []

        for i, sid in enumerate(raw_ids):
            if sid is None:
                sid_int = i + 1
                numeric_ids.append(int(sid_int))
                labels.append(f"{sid_int:02d}")
                continue

            try:
                sid_int = int(str(sid))
                numeric_ids.append(int(sid_int))
                labels.append(f"{sid_int:02d}")
            except Exception:
                sid_int = i + 1
                numeric_ids.append(int(sid_int))
                labels.append(str(sid))

        return np.asarray(numeric_ids, dtype=int), np.asarray(labels, dtype=object)

    def plot_station_profile(self, mt_objects: Sequence["PrepareData.CustomMT"], strike: Optional[float] = None):
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        profile_pos_m, profile_dist = self.assign_profile_pos_m(mt_objects, strike_use)

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(
            profile_dist,
            [0] * len(profile_dist),
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )
        plt.colorbar(scatter, label="Station ID")

        for i, (pd, lab) in enumerate(zip(profile_dist, station_id_labels)):
            plt.annotate(
                f"{lab}",
                (pd, 0),
                xytext=(0, 15 if i % 2 == 0 else -25),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        plt.xlabel(
            f"Distance perpendicular to strike (km) - Strike: {float(strike_use):.1f}°",
            fontsize=12,
        )
        plt.ylabel("Profile line", fontsize=12)
        plt.title("Station Projection onto Profile Perpendicular to Strike", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)

        margin = (profile_dist.max() - profile_dist.min()) * 0.1
        plt.xlim(profile_dist.min() - margin, profile_dist.max() + margin)

        plt.text(
            0.02,
            0.98,
            f"Profile direction: {float(strike_use) + 90:.1f}°\n(⊥ to strike)",
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        sorted_idx = np.argsort(station_ids_num)
        print("\n=== Profile Information (Perpendicular to Strike) ===")
        print(f"Strike: {float(strike_use):.1f}°, Profile direction: {float(strike_use) + 90:.1f}°")
        print(f"Number of stations: {len(profile_dist)}")
        print("\nStation positions (sorted by ID):")
        for idx in sorted_idx:
            print(
                f"  Station {station_id_labels[idx]}: {profile_dist[idx]:8.2f} km  "
                f"(Lat: {lats[idx]:.4f}°, Lon: {lons[idx]:.4f}°)  "
                f"profile_pos_m: {profile_pos_m[idx]:9.1f} m"
            )

        return profile_dist, station_ids_num

    @staticmethod
    def add_scale_bar(ax, lon_lim, lat_lim, bar_length_km: float = 10.0) -> None:
        lon_bar = lon_lim[0] + 0.05 * (lon_lim[1] - lon_lim[0])
        lat_bar = lat_lim[0] + 0.05 * (lat_lim[1] - lat_lim[0])

        lat_center = np.mean(lat_lim)
        lon_per_km = 1 / (111.32 * np.cos(np.deg2rad(lat_center)))
        bar_length_deg = float(bar_length_km) * lon_per_km

        ax.plot([lon_bar, lon_bar + bar_length_deg], [lat_bar, lat_bar], "k-", linewidth=3)
        ax.plot([lon_bar, lon_bar], [lat_bar - 0.001, lat_bar + 0.001], "k-", linewidth=2)
        ax.plot(
            [lon_bar + bar_length_deg, lon_bar + bar_length_deg],
            [lat_bar - 0.001, lat_bar + 0.001],
            "k-",
            linewidth=2,
        )

        ax.text(
            lon_bar + bar_length_deg / 2,
            lat_bar - 0.002,
            f"{bar_length_km:g} km",
            ha="center",
            va="top",
            fontsize=8,
        )

    def plot_station_map_with_profile(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        strike: Optional[float] = None,
        basemap=None,
    ) -> None:
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        if strike_use is None or not np.isfinite(float(strike_use)):
            raise ValueError(
                "Strike angle is not available for plotting. Provide strike explicitly, "
                "or call set_user_strike(), or call compute_strike() first."
            )

        plt.figure(figsize=(12, 10))

        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        lat_margin = lat_range * 0.1
        lon_margin = lon_range * 0.1
        lat_lim = [lats.min() - lat_margin, lats.max() + lat_margin]
        lon_lim = [lons.min() - lon_margin, lons.max() + lon_margin]

        try:
            if basemap == "google":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                import cartopy.feature as cfeature  # type: ignore[import-untyped]

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], crs=ccrs.PlateCarree())

                ax.add_feature(cfeature.LAND, facecolor="lightgray")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                ax.add_feature(cfeature.RIVERS, linewidth=0.5)

                gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
                gl.top_labels = False
                gl.right_labels = False

            elif basemap == "terrain":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                import cartopy.feature as cfeature  # type: ignore[import-untyped]

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], crs=ccrs.PlateCarree())

                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.add_feature(cfeature.LAND, facecolor="lightgreen", alpha=0.3)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                ax.add_feature(cfeature.RIVERS, linewidth=0.5)

                gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
                gl.top_labels = False
                gl.right_labels = False

            elif basemap == "osm":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                from cartopy.io.img_tiles import OSM  # type: ignore[import-untyped]

                imagery = OSM()
                ax = plt.axes(projection=imagery.crs)
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
                ax.add_image(imagery, 10)

            else:
                ax = plt.gca()
                ax.set_xlabel("Longitude (°)", fontsize=12)
                ax.set_ylabel("Latitude (°)", fontsize=12)

        except ImportError:
            print("Note: Install cartopy for better basemaps. Using simple plot.")
            ax = plt.gca()
            ax.set_xlabel("Longitude (°)", fontsize=12)
            ax.set_ylabel("Latitude (°)", fontsize=12)

        scatter = ax.scatter(
            lons,
            lats,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )
        plt.colorbar(scatter, ax=ax, label="Station ID", shrink=0.8)

        for lat, lon, lab in zip(lats, lons, station_id_labels):
            ax.annotate(
                f"{lab}",
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        lon_center = np.mean(lons)
        lat_center = np.mean(lats)

        profile_length = 1.2 * max(
            lat_range * 111.132,
            lon_range * 111.32 * np.cos(np.deg2rad(lat_center)),
        )

        rad = np.deg2rad(float(strike_use) + 90)
        dx_deg = profile_length * np.sin(rad) / (111.32 * np.cos(np.deg2rad(lat_center)))
        dy_deg = profile_length * np.cos(rad) / 111.132

        ax.plot(
            [lon_center - dx_deg, lon_center + dx_deg],
            [lat_center - dy_deg, lat_center + dy_deg],
            "r-",
            linewidth=3,
            alpha=0.7,
            zorder=4,
            label="Profile (⊥ strike)",
        )

        rad_strike = np.deg2rad(float(strike_use))
        dx_strike_deg = (
            profile_length
            * 0.5
            * np.sin(rad_strike)
            / (111.32 * np.cos(np.deg2rad(lat_center)))
        )
        dy_strike_deg = profile_length * 0.5 * np.cos(rad_strike) / 111.132

        ax.plot(
            [lon_center - dx_strike_deg, lon_center + dx_strike_deg],
            [lat_center - dy_strike_deg, lat_center + dy_strike_deg],
            "b--",
            linewidth=2,
            alpha=0.5,
            zorder=3,
            label=f"Strike: {float(strike_use):.1f}°",
        )

        ax.annotate(
            "SW",
            (lon_center - dx_deg, lat_center - dy_deg),
            xytext=(-10, -10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red",
        )
        ax.annotate(
            "NE",
            (lon_center + dx_deg, lat_center + dy_deg),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

        if not hasattr(ax, "projection"):
            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)
            ax.set_xlabel("Longitude (°)", fontsize=12)
            ax.set_ylabel("Latitude (°)", fontsize=12)

        ax.set_title(
            f"Station Location Map with Profile\n"
            f"Strike: {float(strike_use):.1f}°, Profile: {float(strike_use) + 90:.1f}°",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=10)

        self.add_scale_bar(ax, lon_lim, lat_lim)

        plt.tight_layout()
        plt.show()

        print("\n=== Station Map Information ===")
        print(f"Number of stations: {len(mt_objects)}")
        print(f"Latitude range: {lats.min():.4f}° - {lats.max():.4f}°")
        print(f"Longitude range: {lons.min():.4f}° - {lons.max():.4f}°")
        print(f"Strike direction: {float(strike_use):.1f}°")
        print(f"Profile direction (⊥ strike): {float(strike_use) + 90:.1f}°")
        print(f"Profile length: {profile_length:.1f} km")

    def plot_2d_profile_coordinates(self, mt_objects: Sequence["PrepareData.CustomMT"], strike: Optional[float] = None):
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        if strike_use is None or not np.isfinite(float(strike_use)):
            raise ValueError(
                "Strike angle is not available for plotting. Provide strike explicitly, "
                "or call set_user_strike(), or call compute_strike() first."
            )

        lat0, lon0 = lats[0], lons[0]
        d_lat = (lats - lat0) * 111.132
        d_lon = (lons - lon0) * 111.32 * np.cos(np.deg2rad(lat0))

        rad_perp = np.deg2rad(float(strike_use) + 90)
        profile_dist = d_lon * np.sin(rad_perp) + d_lat * np.cos(rad_perp)

        rad_para = np.deg2rad(float(strike_use))
        parallel_dist = d_lon * np.sin(rad_para) + d_lat * np.cos(rad_para)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter1 = ax1.scatter(
            lons,
            lats,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.colorbar(scatter1, ax=ax1, label="Station ID")

        for lat, lon, lab in zip(lats, lons, station_id_labels):
            ax1.annotate(
                f"{lab}",
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        ax1.set_xlabel("Longitude (°)", fontsize=12)
        ax1.set_ylabel("Latitude (°)", fontsize=12)
        ax1.set_title("Original Coordinates", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        scatter2 = ax2.scatter(
            profile_dist,
            parallel_dist,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.colorbar(scatter2, ax=ax2, label="Station ID")

        for pd, pl, lab in zip(profile_dist, parallel_dist, station_id_labels):
            ax2.annotate(
                f"{lab}",
                (pd, pl),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Profile line")
        ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

        ax2.set_xlabel(
            f"Distance ⊥ to strike (km) - Profile direction: {float(strike_use) + 90:.1f}°",
            fontsize=12,
        )
        ax2.set_ylabel(
            f"Distance ∥ to strike (km) - Strike: {float(strike_use):.1f}°",
            fontsize=12,
        )
        ax2.set_title("Profile Coordinates", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis("equal")

        plt.suptitle("Station Distribution: Original vs Profile Coordinates", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        return profile_dist, parallel_dist

    # -------------------------- phase tensor ellipse glyph map --------------------------

    @staticmethod
    def skew_cmap_bgr():
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "skew_bgr",
            [(0.0, "#2b6cb0"), (0.5, "#a3d9a5"), (1.0, "#c92a2a")],
            N=256,
        )

    @staticmethod
    def _phase_tensor_ellipse_params(Phi: np.ndarray, normalize: bool = True):
        Phi = np.asarray(Phi, dtype=float)
        if Phi.shape != (2, 2) or not np.isfinite(Phi).all():
            return None
        try:
            U, s, _ = np.linalg.svd(Phi)
        except Exception:
            return None
        if s.size != 2 or not np.isfinite(s).all():
            return None
        if normalize and s[0] > 1e-12:
            s = s / s[0]
        a = float(s[0])
        b = float(s[1])
        angle = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
        return a, b, angle

    def plot_phase_tensor_ellipses_station_freq(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        normalize: bool = True,
        x_scale: float = 0.8,
        y_scale: float = 0.8,
        x_step: float = 1.5,
        skew_threshold: Optional[float] = None,
        skew_clip: Optional[float] = 5.0,
        cmap=None,
        alpha: float = 0.85,
        linewidth: float = 0.3,
        use_station_id: bool = True,
        sort_by_station_id: bool = True,
        figsize=(16, 12),
    ):
        from matplotlib.colors import Normalize
        from matplotlib.patches import Ellipse

        if mt_objects is None or len(mt_objects) == 0:
            raise ValueError("mt_objects is empty")

        try:
            x_step = float(x_step)
        except Exception:
            x_step = 1.0
        if (not np.isfinite(x_step)) or x_step <= 0:
            x_step = 1.0

        if skew_clip is not None:
            skew_clip = float(skew_clip)
            if (not np.isfinite(skew_clip)) or skew_clip <= 0:
                skew_clip = None

        if cmap is None:
            cmap = self.skew_cmap_bgr()

        station_ids = [getattr(mt, "station_id", None) for mt in mt_objects]
        has_ids = all(sid is not None for sid in station_ids)

        if use_station_id and has_ids and sort_by_station_id:
            # Robust sort: numeric IDs sort numerically; non-numeric keep stable order after numeric ones.
            def _try_int(x):
                try:
                    return int(str(x))
                except Exception:
                    return None

            numeric = [_try_int(sid) for sid in station_ids]
            order = sorted(
                range(len(station_ids)),
                key=lambda i: (numeric[i] is None, numeric[i] if numeric[i] is not None else i),
            )
            mts = [mt_objects[i] for i in order]
            station_ids_sorted = [station_ids[i] for i in order]
        else:
            mts = list(mt_objects)
            station_ids_sorted = station_ids

        if use_station_id and has_ids:
            # Use numeric label when possible; otherwise use original string ID.
            x_tick_labels = []
            for sid in station_ids_sorted:
                try:
                    x_tick_labels.append(f"{int(str(sid))}")
                except Exception:
                    x_tick_labels.append(str(sid))
            x_label = "Station ID"
        else:
            x_tick_labels = [f"{i + 1}" for i in range(len(mts))]
            x_label = "Station #"

        n_stations = len(mts)
        x_positions = 1.0 + np.arange(n_stations, dtype=float) * x_step

        records = []
        skew_vals: list[float] = []
        y_logs: list[float] = []
        a_vals: list[float] = []
        b_vals: list[float] = []

        for si, mt in enumerate(mts):
            freqs = np.asarray(mt.frequency)
            if freqs.ndim != 1:
                continue
            x = float(x_positions[si])
            for fi, f in enumerate(freqs):
                if not np.isfinite(f) or float(f) <= 0:
                    continue
                Phi = self.phase_tensor(mt.Z[fi])
                if Phi is None:
                    continue
                skew = self.phase_tensor_skew(mt.Z[fi])
                if skew_threshold is not None and np.isfinite(skew) and abs(skew) >= float(skew_threshold):
                    continue
                params = self._phase_tensor_ellipse_params(Phi, normalize=normalize)
                if params is None:
                    continue
                a, b, angle = params
                if not (np.isfinite(a) and np.isfinite(b) and a >= 0 and b >= 0):
                    continue
                y = float(np.log10(float(f)))
                records.append((x, y, float(skew) if np.isfinite(skew) else np.nan, a, b, angle))
                y_logs.append(y)
                a_vals.append(float(a))
                b_vals.append(float(b))
                if np.isfinite(skew):
                    skew_vals.append(float(skew))

        if len(records) == 0:
            raise RuntimeError("No valid phase tensor ellipses to plot.")

        if skew_clip is not None:
            norm = Normalize(vmin=-skew_clip, vmax=skew_clip, clip=True)
            skew_max = skew_clip
        else:
            if len(skew_vals) == 0:
                skew_max = 1.0
            else:
                skew_max = float(np.nanpercentile(np.abs(skew_vals), 95))
                if not np.isfinite(skew_max) or skew_max < 1e-6:
                    skew_max = float(np.nanmax(np.abs(skew_vals)))
                if not np.isfinite(skew_max) or skew_max < 1e-6:
                    skew_max = 1.0
            norm = Normalize(vmin=-skew_max, vmax=skew_max)

        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        fig, ax = plt.subplots(figsize=figsize)
        for x, y, skew, a, b, angle in records:
            if np.isfinite(skew):
                skew_for_color = float(skew)
                if skew_clip is not None:
                    skew_for_color = float(np.clip(skew_for_color, -skew_clip, skew_clip))
                color = cmap_obj(norm(skew_for_color))
            else:
                color = (0.6, 0.6, 0.6, 1.0)

            e = Ellipse(
                (x, y),
                width=2.0 * float(x_scale) * float(a),
                height=2.0 * float(y_scale) * float(b),
                angle=float(angle),
                facecolor=color,
                edgecolor="black",
                alpha=float(alpha),
                linewidth=float(linewidth),
            )
            ax.add_patch(e)

        max_a = float(np.nanmax(a_vals)) if len(a_vals) else 1.0
        max_b = float(np.nanmax(b_vals)) if len(b_vals) else 1.0
        x_pad = max(0.6 * x_step, 1.3 * float(x_scale) * max_a)
        y_pad = max(0.15, 1.3 * float(y_scale) * max_b)

        ax.set_xlim(float(x_positions[0] - x_pad), float(x_positions[-1] + x_pad))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        ax.set_xlabel(x_label)

        y_min = float(np.nanmin(y_logs))
        y_max = float(np.nanmax(y_logs))
        y0 = np.floor(y_min)
        y1 = np.ceil(y_max)
        decade_ticks = np.arange(y0, y1 + 1)
        ax.set_yticks(decade_ticks)
        ax.set_yticklabels([f"{(10 ** t):g}" for t in decade_ticks])
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel("Frequency (Hz)")

        title = "Phase Tensor Ellipses (x=station, y=frequency, color=skew)"
        if skew_threshold is not None:
            title += f"  (|skew| < {float(skew_threshold):g}°)"
        if skew_clip is not None:
            title += f"  (skew clipped to ±{float(skew_max):g}°)"
        ax.set_title(title)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Phase tensor skew (deg)")

        plt.tight_layout()
        plt.show()

        return records

    # -------------------------- inspection / rotate-all --------------------------

    def inspect_mt_objects(self) -> None:
        if not self.mt_objects:
            print("mt_objects is not defined / empty.")
            return

        print("=== mt_objects summary ===")
        print("type(mt_objects):", type(self.mt_objects))
        print("len(mt_objects):", len(self.mt_objects))

        first = self.mt_objects[0]
        print("type(mt_objects[0]):", type(first))

        candidate_attrs = ["station_id", "lat", "lon", "frequency", "Z", "Z_err"]
        present_attrs = [a for a in candidate_attrs if hasattr(first, a)]
        print("attributes on CustomMT[0] (subset):", present_attrs)

        station_ids_raw = [getattr(mt, "station_id", None) for mt in self.mt_objects]
        sid_types = sorted({type(s).__name__ for s in station_ids_raw if s is not None})
        n_missing_sid = sum(s is None for s in station_ids_raw)
        print("station_id types:", sid_types if sid_types else ["(all None)"])
        print("missing station_id:", n_missing_sid)
        print("station_id sample (first 12):", station_ids_raw[:12])

        lats = np.array([getattr(mt, "lat", np.nan) for mt in self.mt_objects], dtype=float)
        lons = np.array([getattr(mt, "lon", np.nan) for mt in self.mt_objects], dtype=float)
        print(f"lat range: {np.nanmin(lats):.6f} .. {np.nanmax(lats):.6f}")
        print(f"lon range: {np.nanmin(lons):.6f} .. {np.nanmax(lons):.6f}")

        freq_lens = []
        fmins = []
        fmaxs = []
        for mt in self.mt_objects:
            freqs = getattr(mt, "frequency", None)
            if freqs is None:
                continue
            freqs = np.asarray(freqs)
            if freqs.ndim != 1 or freqs.size == 0:
                continue
            freq_lens.append(int(freqs.size))
            fmins.append(float(np.nanmin(freqs)))
            fmaxs.append(float(np.nanmax(freqs)))

        if freq_lens:
            print(
                "frequency length per station: min/median/max =",
                int(np.min(freq_lens)),
                "/",
                int(np.median(freq_lens)),
                "/",
                int(np.max(freq_lens)),
            )
            print(f"frequency range (Hz): {np.nanmin(fmins):.6g} .. {np.nanmax(fmaxs):.6g}")
        else:
            print("frequency: not found or empty")

        for name in ["Z", "Z_err"]:
            arr = getattr(first, name, None)
            if arr is None:
                print(f"{name}: None")
                continue
            arr = np.asarray(arr)
            print(f"{name}: dtype={arr.dtype}, shape={arr.shape}")
            if arr.size:
                sample = arr.reshape(-1)[0]
                print(f"  {name} sample[0]:", sample)

        freqs0 = np.asarray(getattr(first, "frequency", []), dtype=float)
        if freqs0.size:
            print("first station: freq has NaN?", bool(np.isnan(freqs0).any()))

        print("\n=== profile_pos_m check (meters, centered; left negative / right positive) ===")

        _, station_labels = self._coerce_station_ids_for_plot(self.mt_objects)

        profile_pos_vals = []
        for i, mt in enumerate(self.mt_objects):
            lab = station_labels[i]
            pos = getattr(mt, "profile_pos_m", None)
            if pos is None:
                profile_pos_vals.append(np.nan)
                print(f"  Station {lab}: profile_pos_m = None")
            else:
                try:
                    pos_f = float(np.asarray(pos).reshape(-1)[0])
                    profile_pos_vals.append(pos_f)
                    print(f"  Station {lab}: {pos_f:9.1f} m")
                except Exception:
                    profile_pos_vals.append(np.nan)
                    print(f"  Station {lab}: profile_pos_m = {pos}  (non-numeric!)")

        profile_pos_vals = np.asarray(profile_pos_vals, dtype=float)
        if np.isfinite(profile_pos_vals).any():
            pmin = float(np.nanmin(profile_pos_vals))
            pmax = float(np.nanmax(profile_pos_vals))
            prange = pmax - pmin
            sym_err = abs(pmin + pmax) / prange if prange > 0 else np.nan
            print(f"\nprofile_pos_m min/max (m): {pmin:.1f} / {pmax:.1f}")
            print(f"centering check |min+max|/range: {sym_err:.3g} (closer to 0 => better centered)")

    def rotate_all(self, angle_deg_clockwise: float) -> None:
        if not self.mt_objects:
            raise RuntimeError("mt_objects is empty")

        angle = float(angle_deg_clockwise)
        for mt in self.mt_objects:
            mt.rotate(angle)

        # After rotation, refresh derived rho/phase so mt_objects is self-consistent.
        self.compute_rho_phase_all(self.mt_objects)
        print(f"All stations rotated by {angle:.2f} degrees")

    # -------------------------- one-shot runner (mirrors notebook) --------------------------

    def run_all(self, *, unwrap_90: bool = False) -> None:
        # Frequency alignment follows ``PrepareData(..., harmonize_freqs=...)`` only.
        self.load_mt_objects()
        self.compute_strike()

        # ---- plots (cell 2) ----
        if self.all_strikes_true is not None:
            self.plot_rose(self.all_strikes_true)
        self.plot_all_strikes_subplots(self.mt_objects, unwrap_90=unwrap_90)

        rs_true_use = self._select_strike_true_deg()
        self.plot_comprehensive_strike_analysis(
            self.mt_objects,
            regional_strike=rs_true_use,
            all_strikes=self.all_strikes_true,
            unwrap_90=unwrap_90,
        )

        # ---- station profile analysis (cell 3) ----
        print("\n" + "=" * 50)
        print("STATION PROFILE ANALYSIS")
        print("=" * 50)
        rs = self._select_strike_true_deg()
        if rs is None or not np.isfinite(float(rs)):
            rs, _ = self.estimate_regional_strike(self.mt_objects)
            print("Estimated regional strike (computed here) =", rs)
        else:
            if self.user_strike_true is not None and np.isfinite(float(self.user_strike_true)):
                print("Using user-defined strike (true north) =", rs)
            else:
                print("Using precomputed regional strike =", rs)

        has_ids = all(getattr(mt, "station_id", None) is not None for mt in self.mt_objects)
        if has_ids:
            print(f"Station IDs: {[mt.station_id for mt in self.mt_objects]}")

            print("\n1. Plotting station map with profile...")
            self.plot_station_map_with_profile(self.mt_objects, rs, basemap=None)

            print("\n2. Plotting station profile...")
            self.plot_station_profile(self.mt_objects, rs)

            print("\n3. Plotting 2D profile coordinates...")
            self.plot_2d_profile_coordinates(self.mt_objects, rs)

            print("\nprofile_pos_m written to each CustomMT (meters, centered):")
            print(
                "  min/max =",
                float(np.min([mt.profile_pos_m for mt in self.mt_objects])),
                "/",
                float(np.max([mt.profile_pos_m for mt in self.mt_objects])),
            )
        else:
            print("Warning: Some stations missing IDs. Cannot plot profile.")

        # ---- phase tensor ellipse map (cell 4) ----
        self.plot_phase_tensor_ellipses_station_freq(
            self.mt_objects,
            normalize=True,
            x_scale=0.8,
            y_scale=0.8,
            x_step=1.5,
            skew_threshold=None,
            skew_clip=5.0,
        )

        # ---- inspect (cell 5) ----
        self.inspect_mt_objects()

        # ---- simple rotation (cell 6) ----
        rot_mag = self._select_strike_magnetic_deg()
        if rot_mag is not None and np.isfinite(float(rot_mag)):
            self.rotate_all(float(rot_mag))

        # ---- export mt_objects to txt ----
        base_dir = Path(self.edi_dir) if Path(self.edi_dir).is_dir() else self._module_dir()
        out_dir = base_dir / "mt_objects_txt"
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, mt in enumerate(self.mt_objects):
            sid = getattr(mt, "station_id", None)
            if sid is None:
                fname = f"station_{i + 1:02d}.txt"
            else:
                sid_s = str(sid)
                sid_s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sid_s)
                fname = f"station_{sid_s}.txt"
            self.export_mt_object_to_txt(mt, out_dir / fname)

        print(f"Exported {len(self.mt_objects)} mt_objects to: {out_dir}")


    def run_all_simple(
        self,
        *,
        rotate: bool = True,
        strike_true_deg: Optional[float] = None,
        strike_magnetic_deg: Optional[float] = None,
        sort_by: str = "profile_pos_m",
        freq_rtol: float = 1e-5,        # 并集合并容差
        freq_atol: float = 1e-7,        # 并集合并容差
        device: Optional[str] = None,
        dtype=None,
        save_to_self: bool = True,
    ):
        """最简一键流程：只组织反演需要的数据，不做任何绘图。"""

        # 1. 读取数据，由 harmonize_freqs 控制是否削足适履，clean_data 控制 OOQ+rel_err+skew 清洗
        self.load_mt_objects()
        self.compute_strike()

        # ---- strike selection (for profile projection) ----
        strike_true_use = self._select_strike_true_deg(strike_true_deg)
        if strike_true_use is None or (not np.isfinite(float(strike_true_use))):
            raise ValueError(
                "regional_strike_true is NaN/None; please pass strike_true_deg explicitly "
                "(in degrees, true-north reference) to compute profile_pos_m."
            )

        # ---- write profile_pos_m ----
        self.assign_profile_pos_m(self.mt_objects, float(strike_true_use))

        # ---- optional rotation ----
        if rotate:
            strike_mag_use = self._select_strike_magnetic_deg(strike_magnetic_deg)
            if strike_mag_use is not None and np.isfinite(float(strike_mag_use)):
                self.rotate_all(float(strike_mag_use))
            else:
                print("[run_all_simple] Warning: regional_strike_magnetic is NaN/None; skip rotation")

        # ---- export tensors for inversion ----
        # 干净利落地透传物理控制参数
        return self.export_data_dict_for_2d_inversion(
            self.mt_objects,
            sort_by=sort_by,
            freq_rtol=freq_rtol,
            freq_atol=freq_atol,
            device=device,
            dtype=dtype,
            save_to_self=save_to_self,
        )
