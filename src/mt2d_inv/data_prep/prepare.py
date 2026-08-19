"""MT data preparation orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .grid import compute_grid_horizontal_extent, build_yn_from_stations
from .edi import EdiMixin
from .cleaning import CleaningMixin
from .export import ExportMixin
from .prior import PriorMixin
from .strike import StrikeMixin
from ..plotting.prepare_data import PrepareDataPlotMixin


class PrepareData(
    EdiMixin,
    CleaningMixin,
    ExportMixin,
    PriorMixin,
    StrikeMixin,
    PrepareDataPlotMixin,
):
    """
    - Read EDI files into a list of `CustomMT`
    - Phase-tensor / strike / declination correction
    - Project stations onto the profile and write `profile_pos_m`
    - Plotting helpers (strike, profile, phase-tensor ellipses)
    - Optional: rotate the impedance tensor as a whole
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
        # Choose forced frequency alignment and downsampling from the switch
        # =====================================================================
        if use_harmonize:
            # Legacy path: force all stations onto the frequency intersection (truncate to the common set)
            self._harmonize_frequencies_inplace()

            # Optional: downsample frequencies to n_freq_target (log-uniform)
            # Note: global downsampling requires aligned frequencies, so it must run after harmonize
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
            # New path: keep each station's native frequencies; export_data_dict_for_2d_inversion builds the global union and NaN-masks missing data
            print("[PrepareData] Skipped frequency harmonization. Stations retain their original frequencies.")
            
            # Warn if downsampling is still requested (the legacy _decimate assumes a forced-aligned grid)
            if self.n_freq_target is not None:
                print(
                    f"[PrepareData] WARNING: n_freq_target={self.n_freq_target} is ignored because "
                    "harmonize_freqs=False. Downsampling requires an aligned frequency grid in the current implementation."
                )

        return self.mt_objects

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
        freq_rtol: float = 1e-5,        # union merge tolerance
        freq_atol: float = 1e-7,        # union merge tolerance
        device: Optional[str] = None,
        dtype=None,
        save_to_self: bool = True,
    ):
        """Minimal one-shot pipeline: assemble inversion data only, with no plotting."""

        # 1. Load data; harmonize_freqs controls intersection alignment, clean_data controls OOQ+rel_err+skew cleaning
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
        # Pass physical control parameters through unchanged
        return self.export_data_dict_for_2d_inversion(
            self.mt_objects,
            sort_by=sort_by,
            freq_rtol=freq_rtol,
            freq_atol=freq_atol,
            device=device,
            dtype=dtype,
            save_to_self=save_to_self,
        )
