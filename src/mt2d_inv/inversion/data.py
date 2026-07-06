"""Synthetic/observed data, errors, and weights."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch



class InversionDataMixin:
    def create_synthetic_data(
        self,
        noise_level: float = 0.01,
        noise_type: str = "gaussian",
        outlier_frac: float = 0.05,
        outlier_strength: float = 4.0,
        student_t_df: float = 3.0,
        static_shift_std: float = 0.0,      # 【新增】静位移的标准差(对数域)
        shift_modes: tuple = ("xy", "yx"),  # 【新增】注入静位移的模式
        shift_stations: str = "middle",     # 【新增】注入静位移的台站 ('all', 'middle', 'random')
        shift_ratio: float = 0.1,
    ):
        """
        Generate 2D MT synthetic data: add noise at the impedance level, then
        compute apparent resistivity (rho) and phase (phi) consistently.

        Args:
            noise_level: Relative noise level (relative to |Z|)
            noise_type: "gaussian" | "nongaussian" | "student_t"
                - gaussian: Gaussian noise only
                - nongaussian: Gaussian + random outliers on a subset of points
                - student_t: Student-t distribution (heavy tails, direct non-Gaussian)
            outlier_frac: Outlier fraction in [0, 1], only for noise_type=="nongaussian"
            outlier_strength: Outlier strength (multiple of baseline delta), only for "nongaussian"
            student_t_df: Degrees of freedom for Student-t (default 3). Smaller = heavier tails.
        """
        if noise_type not in ("gaussian", "nongaussian", "student_t"):
            raise ValueError(
                f'noise_type must be "gaussian", "nongaussian", or "student_t", got "{noise_type}".'
            )

        if noise_type == "nongaussian":
            if not (0 <= outlier_frac <= 1):
                raise ValueError(f"noise_type='nongaussian' requires outlier_frac in [0, 1], got {outlier_frac}")
            if outlier_strength <= 0:
                raise ValueError(f"noise_type='nongaussian' requires outlier_strength > 0, got {outlier_strength}")
        if noise_type == "student_t" and student_t_df <= 2:
            raise ValueError(f"noise_type='student_t' requires student_t_df > 2 for finite variance, got {student_t_df}")
        self.noise_level = noise_level
        if self.sig_true is None:
            raise ValueError("sig_true not set; please set inverter.sig_true = sig_true first")
        if isinstance(self.sig_true, (list, tuple)):
            raise ValueError(
                "sig_true cannot be tuple/list. If using create_commemi_2d0, unpack:\n"
                "  zn, yn, freq, ry, sig = MT2DTrueModels.create_commemi_2d0(nza, device)\n"
                "  inverter.sig_true = torch.tensor(sig, dtype=torch.float64, device=device)\n"
                "If using create_geological_models, assign directly:\n"
                "  sig_true = MT2DTrueModels.create_geological_models(zn, yn, model_type='magma_chamber', device=device)\n"
                "  inverter.sig_true = sig_true"
            )
        if not isinstance(self.sig_true, torch.Tensor):
            raise TypeError(f"sig_true must be torch.Tensor, got {type(self.sig_true)}")
        print("Generating 2D MT synthetic data...")

        with torch.no_grad():
            pred_true = self.forward_operator(self.sig_true)

        omega = 2 * np.pi * self.freqs[:, None]
        MU = 4e-7 * np.pi
        n_station = len(self.stations)

        obs_data = {}
        self.data_std = {}   # Store impedance std-dev for error propagation

        shift_mask = torch.zeros(n_station, dtype=torch.bool, device=self.device)
        if shift_stations == "all":
            shift_mask[:] = True
        elif shift_stations == "middle":
            n_shift = int(round(n_station * shift_ratio))

            # 防止超过范围
            n_shift = max(0, min(n_shift, n_station))

            if n_shift > 0:
                start_idx = (n_station - n_shift) // 2
                end_idx = start_idx + n_shift
                shift_mask[start_idx:end_idx] = True
        elif shift_stations == "random":
            n_shift = int(round(n_station * shift_ratio))
            n_shift = max(1,min(n_shift,n_station)) if shift_ratio >0 else 0
            if n_shift >0:
                idx = torch.randperm(n_station, device = self.device)[:n_shift]
                shift_mask[idx] = True
                
        else:
            raise ValueError(f"shift_stations 参数不合法: '{shift_stations}'。")
        
        self.shift_mask = shift_mask.clone()
        self.shift_station_ids = torch.where(shift_mask)[0].cpu().tolist()
        n_shift = int(shift_mask.sum().item())
        self.n_shift_stations = n_shift
        self.shift_fraction_actual = float(n_shift / n_station) if n_station > 0 else 0.0

        self.static_shift_std = static_shift_std
        self.shift_ratio = shift_ratio
        self.shift_modes = shift_modes
        self.shift_stations = shift_stations
        self.true_data_no_shift = {}
        self.static_shift_factors = {}
        self.static_shift_log = {}
        for mode in ["xy", "yx"]:
            Z = pred_true[f"Z{mode}"]      # (nf, nstation)

            # 静位移前：真实模型正演视电阻率/相位（无静位移、无噪声）
            rho_true_clean = torch.abs(Z) ** 2 / (omega * MU)
            phs_true_clean = -torch.atan2(Z.imag, Z.real) * 180.0 / np.pi
            self.true_data_no_shift[f"rho{mode}"] = rho_true_clean.clone()
            self.true_data_no_shift[f"phs{mode}"] = phs_true_clean.clone()

            # 静位移注入：在加噪声之前进行，使噪声随着被放大的阻抗成比例增大
            # ===================================================
            if static_shift_std > 0.0 and mode in shift_modes:
                # 生成高斯分布的对数乘子 (标准差为 static_shift_std)
                shift_log = torch.randn(1, n_station, device=self.device, dtype=torch.float64) * static_shift_std
                # 转换回线性域，得到实数常数乘子
                shift_factor = 10.0 ** shift_log

                # 利用 Mask 决定哪些台站被篡改
                effective_shift_log = torch.where(
                    shift_mask.unsqueeze(0),
                    shift_log,
                    torch.zeros_like(shift_log),
                )
                shift_factor = torch.where(shift_mask.unsqueeze(0), shift_factor, torch.ones_like(shift_factor))

                # 将阻抗乘上实数常数：这会导致模被放大/缩小，而相位(实部和虚部的比例)绝对保持不变！
                Z = Z * shift_factor
            else:
                effective_shift_log = torch.zeros(1, n_station, device=self.device, dtype=torch.float64)
                shift_factor = torch.ones(1, n_station, device=self.device, dtype=torch.float64)

            self.static_shift_log[mode] = effective_shift_log.squeeze(0).clone()
            self.static_shift_factors[mode] = shift_factor.squeeze(0).clone()

            Zabs = torch.abs(Z)

            # -------- Impedance noise (relative) --------
            delta = noise_level * Zabs

            if noise_type == "student_t":
                # Student-t: heavy tails, direct non-Gaussian. Scale to match variance ~ delta^2.
                # Var(StudentT(df, scale=s)) = df/(df-2) * s^2, so s = delta * sqrt((df-2)/df)
                scale = delta * np.sqrt((student_t_df - 2) / student_t_df)
                dist = torch.distributions.StudentT(df=student_t_df, loc=0.0, scale=scale)
                noise_real = dist.sample(Z.real.shape).to(self.device, dtype=torch.float64)
                noise_imag = dist.sample(Z.imag.shape).to(self.device, dtype=torch.float64)
            else:
                # Gaussian baseline
                noise_real = torch.randn_like(Z.real) * delta
                noise_imag = torch.randn_like(Z.imag) * delta

            Z_obs = torch.complex(
                Z.real + noise_real,
                Z.imag + noise_imag
            )

            # -------- Non-Gaussian: add outliers on a subset of points (gaussian + outliers) --------
            if noise_type == "nongaussian":
                n_tot = Z_obs.numel()
                n_out = int(round(outlier_frac * n_tot))
                idx = torch.randperm(n_tot, device=self.device)[:n_out]
                mask = torch.zeros(n_tot, dtype=torch.bool, device=self.device)
                mask[idx] = True
                mask = mask.reshape(Z_obs.shape)
                out_real = torch.randn_like(Z.real) * (outlier_strength * delta)
                out_imag = torch.randn_like(Z.imag) * (outlier_strength * delta)
                Z_obs = torch.complex(
                    Z_obs.real + torch.where(mask, out_real, torch.zeros_like(Z_obs.real)),
                    Z_obs.imag + torch.where(mask, out_imag, torch.zeros_like(Z_obs.imag))
                )

            # -------- Compute rho / phi from impedance --------
            rho_obs = torch.abs(Z_obs) ** 2 / (omega * MU)
            phs_obs = -torch.atan2(Z_obs.imag, Z_obs.real) * 180.0 / np.pi

            obs_data[f"rho{mode}"] = rho_obs
            obs_data[f"phs{mode}"] = phs_obs

            # Store impedance errors for calculate_data_errors_2d
            self.data_std[f"delta_z{mode}_real"] = delta
            self.data_std[f"delta_z{mode}_imag"] = delta
            self.data_std[f"Z{mode}"] = Z_obs

        self.obs_data = obs_data
        # First, propagate errors to obtain per-point noise std
        self.calculate_data_errors_2d()
        # Then, build data weights from the std
        self._compute_data_weights(noise_floor=self.noise_level)
        # Persist noise floor so all diagnostics/OT weighting use the same floor.
        self.noise_floor = float(self.noise_level)

        # Cache RMS chi^2 targets/sigmas (constant during inversion).
        self._build_rms_chi2_cache()

        print("✓ Synthetic data generated")
        print(f"  -> Impedance noise level: {noise_level*100:.1f}% ({noise_type})")
        if noise_type == "nongaussian":
            print(f"  -> Non-Gaussian: outlier_frac={outlier_frac}, outlier_strength={outlier_strength}")
        elif noise_type == "student_t":
            print(f"  -> Student-t: df={student_t_df}")

    def load_obs_data(self, data_dict: dict, noise_floor: float = 0.1):
        """
            data_dict: 须包含 obs_data, data_std
            noise_floor: 噪声下限（相对误差）。用于：
                1) _compute_data_weights 的误差下限/兜底
                2) compute_rms_chi2 的 sigma 下限/兜底（避免由于过小的 sigma 导致 RMS χ² 虚高）
        """
        self.obs_data = {k: v.to(self.device, dtype=torch.float64) for k, v in data_dict["obs_data"].items()}
        station_ids = data_dict.get("station_ids", None)
        if station_ids is None:
            self.station_ids = None
        else:
            try:
                self.station_ids = [str(s) for s in list(station_ids)]
            except Exception:
                self.station_ids = [str(station_ids)]
        self.data_std = {}
        for k, v in data_dict["data_std"].items():
            if torch.is_tensor(v):
                self.data_std[k] = v.to(self.device)
            else:
                self.data_std[k] = v
        self.calculate_data_errors_2d()
        # Persist for later RMS/diagnostics.
        self.noise_floor = noise_floor
        self._compute_data_weights(noise_floor=self.noise_floor)

        # Cache RMS chi^2 targets/sigmas (constant during inversion).
        self._build_rms_chi2_cache()
        print("✓ Observed data loaded from file")

    def calculate_data_errors_2d(self):
        """
        Propagate impedance errors to obtain std-dev for rho and phi (2D), and
        construct dimensionless noise std-dev used for chi^2 / RMS.
        When |Z| or rho is very small (e.g. dead band), the propagation can explode;
        we clamp denominators and apply a fixed numeric ceiling on the *propagated*
        dimensionless std (explosion / NaN guard only). Do not lower this ceiling to
        ``shrink'' error bars for inversion: that would inflate weights (w ∝ 1/σ²)
        on high-uncertainty points. For display-only caps use ``plot_noise_cap`` in
        :meth:`plot_data_fitting`.
        """
        eps = 1e-8
        eps_rho = 1e-6   # min rho in denominator to avoid explosion
        eps_z = 1e-12    # min |Z| in phi derivative
        # Fixed safety cap (same units as rho_noise_std_log / phs_noise_std_norm).
        max_noise_std = 1.0

        omega = 2 * np.pi * self.freqs[:, None]
        MU = 4e-7 * np.pi  # Vacuum magnetic permeability (H/m)

        # Noise std-dev for all modes (used in inversion)
        self.data_noise_std = {}

        for mode in ["xy", "yx"]:
            key_z = f"Z{mode}"
            key_rho = f"rho{mode}"
            key_phs = f"phs{mode}"

            # Only propagate errors if impedance has been saved into data_std
            if key_z not in self.data_std:
                continue  # Allow using a single mode (e.g., one polarization)

            Z = self.data_std[key_z]
            Zr = Z.real
            Zi = Z.imag
            Zabs = torch.abs(Z)
            Zabs_safe = torch.clamp(Zabs, min=eps_z)

            delta_z_real = self.data_std[f"delta_z{mode}_real"]
            delta_z_imag = self.data_std[f"delta_z{mode}_imag"]

            # ---------- rho std-dev ----------
            dRho_dZr = 2.0 * Zr / (omega * MU)
            dRho_dZi = 2.0 * Zi / (omega * MU)

            delta_rho = torch.sqrt(
                (dRho_dZr * delta_z_real) ** 2 +
                (dRho_dZi * delta_z_imag) ** 2
            )

            # ---------- phi std-dev (use Zabs_safe to avoid 1/|Z|^2 explosion) ----------
            dPhi_dZr = -Zi / (Zabs_safe ** 2)
            dPhi_dZi =  Zr / (Zabs_safe ** 2)

            delta_phi_rad = torch.sqrt(
                (dPhi_dZr * delta_z_real) ** 2 +
                (dPhi_dZi * delta_z_imag) ** 2
            )

            delta_phs = delta_phi_rad * 180.0 / np.pi

            # ---------- Dimensionless noise std-dev (for chi^2 / RMS) ----------
            # log10(rho): avoid division by near-zero rho, then cap
            rho_obs = torch.clamp(self.obs_data[key_rho], min=eps_rho)
            rho_noise_std_log = torch.clamp(
                delta_rho / (rho_obs * np.log(10)),
                min=eps,
                max=max_noise_std
            )
            rho_noise_std_log = torch.nan_to_num(
                rho_noise_std_log,
                nan=max_noise_std,
                posinf=max_noise_std,
                neginf=max_noise_std,
            )

            # phi / 90°
            phs_noise_std_norm = torch.clamp(
                delta_phs / 90.0,
                min=eps,
                max=max_noise_std
            )
            phs_noise_std_norm = torch.nan_to_num(
                phs_noise_std_norm,
                nan=max_noise_std,
                posinf=max_noise_std,
                neginf=max_noise_std,
            )

            # ---------- Save ----------
            self.data_noise_std[f"rho{mode}"] = rho_noise_std_log
            self.data_noise_std[f"phs{mode}"] = phs_noise_std_norm

            print(f"✓ {mode.upper()} mode error propagation completed")
            print(f"   rho(log10) noise mean: {rho_noise_std_log.mean():.4f}")
            print(f"   phi(normalized) noise mean: {phs_noise_std_norm.mean():.4f}")

        print("✓ 2D data error propagation completed")

    def _get_noise_std_floors(self, noise_floor: Optional[float] = None) -> Dict[str, float]:
        """Return per-type noise std floors in the same units as data_noise_std.

        - rho*: std of log10(rho)
        - phs*: std of (phi/90)
        """
        nf = float(getattr(self, "noise_floor", 0.01) if noise_floor is None else noise_floor)
        nf = float(nf or 0.01)

        sigma_rho_floor = nf / float(np.log(10.0))
        phase_error_deg = max(nf * 28.6, 0.5)
        sigma_phs_norm_floor = float(phase_error_deg) / 90.0
        return {"rho": float(sigma_rho_floor), "phs": float(sigma_phs_norm_floor)}

    def get_effective_data_noise_std(
        self,
        key: str,
        *,
        noise_floor: Optional[float] = None,
        eps: float = 1e-12,
    ) -> Optional[torch.Tensor]:
        """Noise std used consistently across weighting/plots (noise_floor-clipped).

        Returns a tensor in the SAME units as self.data_noise_std[key].
        """
        if not hasattr(self, "data_noise_std"):
            return None
        t = self.data_noise_std.get(key, None)
        if t is None:
            return None

        # Robustness: avoid NaN/Inf propagating into weights/RMS/plots.
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        floors = self._get_noise_std_floors(noise_floor=noise_floor)
        if "rho" in key.lower():
            floor = floors["rho"]
        elif "phs" in key.lower():
            floor = floors["phs"]
        else:
            floor = 0.0
        return torch.clamp(t.to(self.device, dtype=torch.float64), min=float(max(floor, eps)))

    def get_effective_cloud_noise_std(
        self,
        key: str,
        *,
        noise_floor: Optional[float] = None,
        sigma_min: Optional[float] = None,
        eps: float = 1e-12,
    ) -> Optional[torch.Tensor]:
        """Effective noise std in *point-cloud normalized units* (noise_floor-clipped).

        Conventions:
        - rho*: data_noise_std is std of log10(rho); cloud uses (log10(rho)+2)/8, so sigma_cloud = sigma_log/8.
        - phs*: data_noise_std is std of (phi/90), already in cloud units.
        """
        t = self.get_effective_data_noise_std(key, noise_floor=noise_floor, eps=eps)
        if t is None:
            return None
        if "rho" in key.lower():
            t = t / 8.0
        if sigma_min is not None:
            t = torch.clamp(t, min=float(max(sigma_min, eps)))
        return t

    def _build_rms_chi2_cache(self, *, eps: float = 1e-12):
        """Precompute flat targets and sigma for compute_rms_chi2.

        Why: obs_data and noise std (data_noise_std) are constant during inversion.
        Building these once avoids per-epoch nan_to_num/clamp/full_like overhead.

        Cache format:
            self._rms_chi2_cache[key] = {
                'idx':   LongTensor of indices into flattened (nf*nstation),
                'obs':   FloatTensor of transformed obs values (1D, masked),
                'sigma': FloatTensor of effective std (1D, masked)
            }
        """
        if not isinstance(getattr(self, "obs_data", None), dict) or len(self.obs_data) == 0:
            self._rms_chi2_cache = None
            self._rms_chi2_n_data = 0
            return

        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        n_total = 0

        # Use the same noise-floor conventions as weighting.
        noise_floor = float(getattr(self, "noise_floor", 0.01) or 0.01)
        floors = self._get_noise_std_floors(noise_floor=noise_floor)

        for key in ("rhoxy", "phsxy", "rhoyx", "phsyx"):
            obs_raw = self.obs_data.get(key, None)
            if obs_raw is None:
                continue
            if not torch.is_tensor(obs_raw):
                continue
            obs_raw = obs_raw.to(self.device, dtype=torch.float64)

            valid_mask_flat = ~torch.isnan(obs_raw.flatten())
            if not torch.any(valid_mask_flat):
                continue
            idx = torch.nonzero(valid_mask_flat, as_tuple=False).flatten().to(device=self.device)

            if "rho" in key.lower():
                obs_trans = torch.log10(torch.clamp(obs_raw, min=eps)).flatten().index_select(0, idx)
                floor_val = float(max(floors["rho"], eps))
            else:
                obs_trans = (obs_raw / 90.0).flatten().index_select(0, idx)
                floor_val = float(max(floors["phs"], eps))

            sigma_eff = self.get_effective_data_noise_std(key, noise_floor=noise_floor, eps=eps)
            if sigma_eff is None:
                sigma_flat = torch.full(
                    (idx.numel(),),
                    floor_val,
                    device=self.device,
                    dtype=torch.float64,
                )
            else:
                sigma_flat = sigma_eff.flatten().index_select(0, idx).to(self.device, dtype=torch.float64)

            cache[key] = {"idx": idx, "obs": obs_trans, "sigma": sigma_flat}
            n_total += int(idx.numel())

        self._rms_chi2_cache = cache
        self._rms_chi2_n_data = int(n_total)

    def compute_rms_chi2(self, pred_dict):
        """
        Compute statistically meaningful RMS chi^2.
        RMS ~ 1 means the fit is at the noise level.
        """
        if self._rms_chi2_cache is None:
            self._build_rms_chi2_cache()

        if not self._rms_chi2_cache:
            return float("nan")

        n_data = int(getattr(self, "_rms_chi2_n_data", 0) or 0)
        if n_data <= 0:
            return float("nan")

        chi2_sum = 0.0
        for key, item in self._rms_chi2_cache.items():
            if key not in pred_dict:
                continue

            idx = item["idx"]
            obs_flat = item["obs"]
            sigma_flat = item["sigma"]

            pred_raw = pred_dict[key].to(self.device, dtype=torch.float64)
            if "rho" in key.lower():
                pred_trans_flat = torch.log10(torch.clamp(pred_raw, min=1e-12)).flatten().index_select(0, idx)
            else:
                pred_trans_flat = (pred_raw / 90.0).flatten().index_select(0, idx)

            res = (pred_trans_flat - obs_flat) / sigma_flat
            chi2_sum += torch.sum(res ** 2)

        rms = torch.sqrt(chi2_sum / float(n_data))
        return float(rms.item())

    def _compute_data_weights(self, noise_floor=0.01, error_floor=1e-3,
                              normalize: bool = True, normalize_by: str = "mean"):
        """
        Compute data weights (W_d).

        Notes:
            - These weights are used by mode='mse' as ((obs-pred)*w)^2.
            - By default we normalize the overall scale of weights to keep the MSE
              data term numerically stable across datasets/noise levels, while
              preserving relative weights between components and points.
        """
        self.data_weights = {}
        
        # Rule of thumb: 1% relative error corresponds to ~0.286 degrees phase error
        # Phase_Err_Deg ≈ Noise_Level * (180 / pi / sqrt(2)) (or empirical)
        # Empirical conversion: 10% ~ 2.86 deg -> 1% ~ 0.286 deg
        phase_error_deg = noise_floor * 28.6 
        
        # Set a physical lower bound to avoid exploding weights.
        # In practice, instrument phase error is rarely below ~0.5 degrees.
        if phase_error_deg < 0.5:
            phase_error_deg = 0.5

        # Apply noise_floor consistently as a lower bound when propagated std-dev exists.
        sigma_rho_floor = float(noise_floor) / float(np.log(10.0))
        sigma_phs_deg_floor = float(phase_error_deg)
            
        print(f"Computing data weights (Target Noise: {noise_floor*100:.1f}%)")
        print(f"  - Resistivity Error Floor: {noise_floor*100:.1f}%")
        print(f"  - Phase Error Floor:       {phase_error_deg:.3f} deg")
        
        for key, data in self.obs_data.items():
            # Default weight is 1.0; if noise std-dev is available, use 1/sigma for weighted chi^2
            if 'rho' in key.lower():
                # Std-dev corresponds to log10(rho) in calculate_data_errors_2d
                sigma_log = self.data_noise_std.get(key, None)
                if sigma_log is not None:
                    sigma_min = float(max(error_floor, sigma_rho_floor))
                    sigma_clamped = torch.clamp(sigma_log, min=sigma_min)
                    w_tensor = 1.0 / sigma_clamped
                else:
                    # Fallback: constant relative-error approximation
                    sigma = noise_floor / 2.3026
                    w_tensor = torch.full_like(data, 1.0/(sigma + 1e-8), device=self.device)

            elif 'phs' in key.lower():
                # data_noise_std stores std-dev of (phi/90)
                sigma_norm = self.data_noise_std.get(key, None)
                if sigma_norm is not None:
                    # In MSE we use residuals in degrees,
                    # so choose weight = 1/(90 * sigma_norm)
                    # such that (Δphi * weight)^2 ≈ ((Δphi/90)/sigma_norm)^2
                    sigma_eff = 90.0 * sigma_norm
                    sigma_min = float(max(error_floor, sigma_phs_deg_floor))
                    sigma_clamped = torch.clamp(sigma_eff, min=sigma_min)
                    w_tensor = 1.0 / sigma_clamped
                else:
                    # Fallback: constant phase error
                    sigma = phase_error_deg
                    w_tensor = torch.full_like(data, 1.0/(sigma + 1e-8), device=self.device)
            else:
                w_tensor = torch.ones_like(data, device=self.device)
            self.data_weights[key] = w_tensor

        if normalize:
            keys = [k for k in self.data_weights.keys() if ("rho" in k.lower()) or ("phs" in k.lower())]
            if len(keys) > 0:
                w_all = torch.cat([self.data_weights[k].reshape(-1) for k in keys], dim=0)
                w_valid = w_all[~torch.isnan(w_all)] 
                
                if normalize_by == "mean":
                    scale = w_valid.mean().clamp(min=1e-12)
                elif normalize_by == "rms":
                    scale = torch.sqrt((w_valid ** 2).mean()).clamp(min=1e-12)
                
                for k in keys:
                    self.data_weights[k] = self.data_weights[k] / scale
                self.data_weights_scale = float(scale.item())
                print(f"✓ MSE data weights normalized ({normalize_by}): scale={self.data_weights_scale:.6e}")    
 
