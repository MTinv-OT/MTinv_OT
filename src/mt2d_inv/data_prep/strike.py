"""Strike estimation and profile projection."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np



class StrikeMixin:
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
        """Return strike at all frequencies; set NaN where the 2-D condition is not met."""

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

