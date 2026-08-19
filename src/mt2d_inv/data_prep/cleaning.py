"""MT data quality control."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np



class CleaningMixin:
    def _clean_data_ooq_rel_err_inplace(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        zxy_phase_range: Tuple[float, float] = (-5.0, 95.0),
        zyx_phase_range: Tuple[float, float] = (175.0, 275.0), # prefer a 0–360° angle convention
        rel_err_max: float = 0.5,
        skew_threshold: Optional[float] = 15.0,
        neighbor_z_thresh: float = 4.0,
        neighbor_rho_log10_floor: float = 0.25,
        neighbor_phs_deg_floor: float = 8.0,
    ) -> None:
        """
        Data cleaning: OOQ + rel_err + phase-tensor skew + neighbor-frequency (ρa/φ + error bars) spikes.

        P1 (OOQ): reject frequencies whose energy-dissipation relation is reversed, before folding phase into 0–90°.
        - Zxy is theoretically in the first quadrant; check whether arctan2(Im, Re) lies in zxy_phase_range (default [-5°, 95°])
        - Zyx is theoretically in the third quadrant; check whether arctan2(Im, Re) % 360 lies in zyx_phase_range (default [175°, 275°])
        Out-of-range: set that frequency's Z and Z_err for the mode to NaN

        P2 (rel_err): drop inversion-used components at a frequency if relative error exceeds rel_err_max (default 50%).
        - Relative error is computed only on off-diagonal Zxy/Zyx (avoid false rejects from noisy near-zero Zxx/Zyy)
        - If max(rel_err(Zxy), rel_err(Zyx)) > rel_err_max, set that frequency's Zxy/Zyx (and Z_err) to NaN

        P3 (skew): reject strong 3-D distortion so 2-D forward modeling is not misled by 3-D effects.
        - Compute phase-tensor skew β; if |β| > skew_threshold (default 8°), set both TE/TM modes at that frequency to NaN
        - skew_threshold=None skips this check

        P4 (neighbor spike, per channel):
        - On each mode (Zxy/Zyx), compare neighboring frequencies using q=log10(ρa) and q=φ(°) separately
        - Flag a spike if the deviation from the neighbor median satisfies both:
            |Δq| > abs_floor and |Δq| / sqrt(σ_i^2 + σ_nei^2) > z_thresh
          (only that component at that frequency is set to NaN)
        - Defaults are relatively loose: z_thresh=4, ρa floor=0.25 log10, φ floor=8°
        - Error bars require Z_err; skip P4 if Z_err is missing
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
            # Use modulo-360 for Zyx to avoid the -180/180 wraparound
            phs_yx_360 = np.degrees(np.arctan2(Z[:, 1, 0].imag, Z[:, 1, 0].real)) % 360.0

            bad_xy = (phs_xy < zxy_phase_range[0]) | (phs_xy > zxy_phase_range[1])
            bad_yx = (phs_yx_360 < zyx_phase_range[0]) | (phs_yx_360 > zyx_phase_range[1])

            # Mask also invalid (NaN/Inf) as bad
            bad_xy = bad_xy | ~np.isfinite(Z[:, 0, 1])
            bad_yx = bad_yx | ~np.isfinite(Z[:, 1, 0])

            n_ooq_xy += int(np.sum(bad_xy))
            n_ooq_yx += int(np.sum(bad_yx))

            # Assign via direct slicing to avoid NumPy chained-view warnings
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
