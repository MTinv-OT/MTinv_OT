"""
修订说明（2026-04-14）：
1、参考模型约束支持先验模型
"""
import torch
import torch.nn as nn
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skimage.metrics import structural_similarity as ssim

# Import sibling modules (relative imports)
from .constraints import ConstraintCalculator
from .optimizer import OptimizerConfig
from .MT2D import MT2DFD_Torch
torch.set_default_dtype(torch.float64)

class MT2DInverter:
    """
    2D magnetotelluric (MT) inverter.
    """
    
    def __init__(self, 
                 yn: torch.Tensor = None,
                 zn: torch.Tensor = None,
                 nza: int = 10,
                 freqs: torch.Tensor = None, 
                 stations: torch.Tensor = None,
                 device: str = "cuda", 
                 random_seed: int = 42,
                 ot_options: Dict = None,
                 te_weight: float = 1.0,
                 tm_weight: float = 1.0,
                 data_loss_scale: float = 100.0,
                 ):


        # TE/TM mode weights (used in 6d/3d OT and MSE mode)
        self.te_weight = float(te_weight)
        self.tm_weight = float(tm_weight)
        self.data_loss_scale = float(data_loss_scale)

        # Default OT hyper-parameters
        default_ot = {
            "p": 2,
            "blur": 0.01,
            "scaling": 0.9,
            "reach": None,
            "backend": "tensorized",
            "sigma_min": 0.03,
            "sigma_6d": None,
        }

        self.ot_config = default_ot.copy()
        if ot_options is not None:
            self.ot_config.update(ot_options)
        self.set_random_seed(random_seed)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.yn = yn.to(self.device)
        self.zn = zn.to(self.device)
        self.freqs = freqs.to(self.device, dtype=torch.float64)
        self.stations = stations.to(self.device, dtype=torch.float64)
        self.opt_config = OptimizerConfig(self.device)
        self.nza: int = nza
        self.air_sigma_value: float = 1e-10
        self._air_sigma_cache: torch.Tensor = None
        self.model_log_sigma = None
        self.initial_model_sigma = None
        self.obs_data = {}
        self.forward_operator = None
        self.loss_history = []
        self.sig_true = None
        self.noise_level = None
        self.sig_ref = None
        self.model_log_sigma_ref = None
        self.grad_norm_d_history = []
        self.grad_norm_m_history = []
        self.ratio_history = []
        self.time_stats = {
            'total_inversion_time': 0,
            'avg_epoch_time': 0,
            'epoch_times': [],
            'start_time': 0,
            'end_time': 0
        }

        # Cached targets for RMS chi^2 (obs + sigma as flat tensors). Built once after obs data is ready.
        self._rms_chi2_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._rms_chi2_n_data: int = 0

        self.sinkhorn_loss = None
        self._init_sinkhorn(**self.ot_config)

    def set_random_seed(self, seed: int = 42):
        """Set random seeds."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"✓ Random seed set: {seed}")

    def set_forward_operator(self):
        """Set the forward operator.

        Convention: the air layer participates in forward modeling only and is
        not updated by inversion.

        - full sigma shape: (nz-1, ny-1)      (includes air)
        - earth sigma shape: (nz-1-nza, ny-1) (subsurface only)
        """
        # Fixed air-layer conductivity (adjust if needed)
        self.air_sigma_value = float(getattr(self, "air_sigma_value", 1e-10))
        self._air_sigma_cache = None

        expected_full = (len(self.zn) - 1, len(self.yn) - 1)
        expected_earth = (expected_full[0] - self.nza, expected_full[1])
        if expected_earth[0] <= 0:
            raise ValueError(f"nza={self.nza} too large: subsurface layers <= 0 (full={expected_full})")

        def _forward(sigma: torch.Tensor):
            sigma = sigma.to(self.device, dtype=torch.float64)

            # Compatibility: accept both full sigma and earth-only sigma
            if tuple(sigma.shape) == expected_earth:
                sigma_full = self._assemble_sigma_full(sigma)
            elif tuple(sigma.shape) == expected_full:
                sigma_full = sigma
            else:
                raise ValueError(
                    f"Sigma size mismatch. Expected earth={expected_earth} or full={expected_full}, got {tuple(sigma.shape)}"
                )

            # Avoid 0/NaN/Inf causing singularities in the forward linear system
            sigma_safe = torch.clamp(sigma_full, min=1e-10, max=1e2)
            fwd = MT2DFD_Torch(
                nza=self.nza,
                zn=self.zn,
                yn=self.yn,
                freq=self.freqs,
                ry=self.stations,
                sig=sigma_safe,
                device=self.device,
            )
            return fwd(mode="TETM")

        self.forward_operator = _forward

        # Read grid
        dummy = torch.ones(expected_full, device=self.device, dtype=torch.float64)
        fwd = MT2DFD_Torch(self.nza, self.zn, self.yn, self.freqs, self.stations, dummy, self.device)

        self.nz, self.ny = fwd.nz, fwd.ny
        self.dz, self.dy = fwd.dz, fwd.dy
        self.dz_earth = self.dz[self.nza:]

        # Constraints are computed on the earth-parameter domain only
        self.constraint_calc = ConstraintCalculator(
            self.ny - 1,
            (self.nz - 1) - self.nza,
            self.dy,
            self.dz_earth,
            device=self.device,
        )

    def _get_air_sigma(self, ny_model: int) -> torch.Tensor:
        """Return the fixed air-layer conductivity tensor with shape (nza, ny_model)."""
        if self.nza <= 0:
            return torch.empty((0, ny_model), device=self.device, dtype=torch.float64)
        if self._air_sigma_cache is not None and tuple(self._air_sigma_cache.shape) == (self.nza, ny_model):
            return self._air_sigma_cache
        self._air_sigma_cache = torch.full(
            (self.nza, ny_model),
            self.air_sigma_value,
            device=self.device,
            dtype=torch.float64,
        )
        return self._air_sigma_cache

    def _assemble_sigma_full(self, sigma_earth: torch.Tensor) -> torch.Tensor:
        """Assemble full sigma by concatenating fixed air (top nza layers) + earth sigma."""
        sigma_earth = sigma_earth.to(self.device, dtype=torch.float64)
        if self.nza <= 0:
            return sigma_earth
        air = self._get_air_sigma(sigma_earth.shape[1])
        return torch.cat([air, sigma_earth], dim=0)

    def get_sigma_full(self) -> torch.Tensor:
        """Return current full sigma (including the fixed air layer)."""
        if self.model_log_sigma is None:
            raise RuntimeError("Please call initialize_model first")
        sigma_earth = torch.exp(self.model_log_sigma)
        return self._assemble_sigma_full(sigma_earth)

    def create_synthetic_data(
        self,
        noise_level: float = 0.01,
        noise_type: str = "gaussian",
        outlier_frac: float = 0.05,
        outlier_strength: float = 4.0,
        student_t_df: float = 3.0,
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

        obs_data = {}
        self.data_std = {}   # Store impedance std-dev for error propagation

        for mode in ["xy", "yx"]:
            Z = pred_true[f"Z{mode}"]      # (nf, nstation)
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
                n_out = max(1, int(round(outlier_frac * n_tot)))
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

    def _get_6d_valid_mask(self) -> torch.Tensor:
        """
        获取 6D OT 的全局有效数据掩码。
        在 6D 点云中，一个 (freq, station) 点必须四个分量全部存在才视为有效点。
        """
        m1 = ~torch.isnan(self.obs_data.get('rhoxy', torch.tensor(float('nan'))))
        m2 = ~torch.isnan(self.obs_data.get('phsxy', torch.tensor(float('nan'))))
        m3 = ~torch.isnan(self.obs_data.get('rhoyx', torch.tensor(float('nan'))))
        m4 = ~torch.isnan(self.obs_data.get('phsyx', torch.tensor(float('nan'))))
        return (m1 & m2 & m3 & m4).flatten()

    def _prepare_3d_ot_cloud(self, 
                             data_tensor: torch.Tensor, 
                             key: str) -> torch.Tensor:
        """
        Build a normalized (N_valid, 3) point cloud: [Freq, Station, Value].
        使用动态掩码剔除 NaN 数据，确保 OT 计算只在真实数据点上进行。
        """
        obs_raw = self.obs_data[key]
        valid_mask = ~torch.isnan(obs_raw.flatten())
        
        n_freq = len(self.freqs)
        n_stations = len(self.stations)
        
        # 1) Normalize frequency (log domain) -> [0, 1] -> 剔除 NaN
        log_freq = torch.log10(self.freqs)
        norm_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-8)
        grid_freq = norm_freq.view(-1, 1).expand(n_freq, n_stations).flatten()[valid_mask]
        
        # 2) Normalize stations -> [0, 1] -> 剔除 NaN
        norm_stn = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_stn = norm_stn.view(1, -1).expand(n_freq, n_stations).flatten()[valid_mask]
        
        # 3) Normalize values -> [0, 1] -> 剔除 NaN
        data_flat = data_tensor.flatten()[valid_mask]
        if 'rho' in key.lower():
            val_log = torch.log10(data_flat + 1e-12)
            norm_val = (val_log - (-2.0)) / (6.0 - (-2.0))
        else:
            norm_val = data_flat / 90.0
            
        # 4) Stack: (Batch, N_valid_points, Dim)
        points = torch.stack([grid_freq, grid_stn, norm_val], dim=1)
        return points.unsqueeze(0)

    def _build_3d_ot_weights(self, key: str):
        """Build OT weights (alpha, beta) for a 3D point cloud."""
        eps = 1e-8
        
        # [关键修改] 同步提取 Mask
        obs_raw = self.obs_data[key]
        valid_mask = ~torch.isnan(obs_raw.flatten())
        n_valid = valid_mask.sum().item()
        
        noise_std = self.get_effective_cloud_noise_std(key, eps=eps)
        
        if noise_std is None:
            alpha = torch.full((1, n_valid), 1.0 / n_valid, device=self.device, dtype=torch.float64)
            beta = alpha.clone()
            return alpha, beta

        # [关键修改] 权重只保留有效数据位置
        noise_std_valid = noise_std.flatten()[valid_mask]

        w_obs = 1.0 / (noise_std_valid ** 2 + eps)
        w_obs = torch.nan_to_num(w_obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(w_obs <= 0):
            w_obs = torch.ones_like(w_obs)
        w_obs = w_obs / (w_obs.sum() + eps)
        alpha = torch.full_like(w_obs, 1.0 / n_valid)
        return alpha.unsqueeze(0), w_obs.unsqueeze(0)

    def _get_6d_cost_sigma(self) -> torch.Tensor:
        """
        Per-dimension σ for 6D OT: observation/prediction clouds are multiplied by (1/σ)
        before Sinkhorn (defines cost geometry only).

        **Does not** use ``data_noise_std`` — avoid duplicating noise information already used
        elsewhere (e.g. 3D marginal weights). Set explicitly via ``ot_options['sigma_6d']``:

        - ``None`` (default): use six ``1.0`` (then clamped ≥ ``sigma_min``); tune with ``sigma_6d``.
        - Sequence of 6 positive floats: per-dimension scales (each clamped ≥ ``sigma_min``).
        """
        sigma_min = float(self.ot_config.get("sigma_min", 0.03))
        cfg = self.ot_config.get("sigma_6d", None)
        _DEFAULT_SIGMA_6D = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        if cfg is not None:
            arr = np.asarray(cfg, dtype=float).reshape(-1)
            if arr.size != 6:
                raise ValueError(f"ot_config['sigma_6d'] must have 6 elements, got {arr.size}")
            vals = [max(float(x), sigma_min) for x in arr]
        else:
            vals = [max(float(x), sigma_min) for x in _DEFAULT_SIGMA_6D]

        return torch.tensor(vals, device=self.device, dtype=torch.float64)

    def _prepare_6d_ot_cloud_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build 6D observation point cloud, dropping points where ANY component is NaN."""
        valid_mask = self._get_6d_valid_mask()
        
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        
        log_f = torch.log10(self.freqs)
        norm_f = (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-8)
        grid_f = norm_f.view(-1, 1).expand(n_freq, n_stn).flatten()[valid_mask]
        
        norm_s = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_s = norm_s.view(1, -1).expand(n_freq, n_stn).flatten()[valid_mask]

        def _norm_obs(key: str, data: torch.Tensor):
            data_flat = data.flatten()[valid_mask]
            if 'rho' in key.lower():
                val_log = torch.log10(data_flat + 1e-12)
                return (val_log - (-2.0)) / (6.0 - (-2.0))
            return data_flat / 90.0

        obs_rhoxy = _norm_obs('rhoxy', obs_dict['rhoxy'])
        obs_phsxy = _norm_obs('phsxy', obs_dict['phsxy'])
        obs_rhoyx = _norm_obs('rhoyx', obs_dict['rhoyx'])
        obs_phsyx = _norm_obs('phsyx', obs_dict['phsyx'])
        
        obs_points = torch.stack([grid_f, grid_s, obs_rhoxy, obs_phsxy, obs_rhoyx, obs_phsyx], dim=1)
        sigma_6d = self._get_6d_cost_sigma()
        obs_points = (obs_points * (1.0 / sigma_6d).unsqueeze(0)).unsqueeze(0)
        return obs_points

    def _prepare_6d_ot_cloud_pred(self, pred_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build 6D prediction point cloud (changes every iteration), using same mask as obs."""
        valid_mask = self._get_6d_valid_mask()
        
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        log_f = torch.log10(self.freqs)
        norm_f = (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-8)
        grid_f = norm_f.view(-1, 1).expand(n_freq, n_stn).flatten()[valid_mask]
        
        norm_s = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_s = norm_s.view(1, -1).expand(n_freq, n_stn).flatten()[valid_mask]

        def _norm_pred(key: str, data: torch.Tensor):
            data_flat = data.flatten()[valid_mask] # [关键修改] 过滤预测数据
            if 'rho' in key.lower():
                val_log = torch.log10(data_flat + 1e-12)
                return (val_log - (-2.0)) / (6.0 - (-2.0))
            return data_flat / 90.0

        pred_rhoxy = _norm_pred('rhoxy', pred_dict['rhoxy'])
        pred_phsxy = _norm_pred('phsxy', pred_dict['phsxy'])
        pred_rhoyx = _norm_pred('rhoyx', pred_dict['rhoyx'])
        pred_phsyx = _norm_pred('phsyx', pred_dict['phsyx'])
        
        pred_points = torch.stack([grid_f, grid_s, pred_rhoxy, pred_phsxy, pred_rhoyx, pred_phsyx], dim=1)
        sigma_6d = self._get_6d_cost_sigma()
        pred_points = (pred_points * (1.0 / sigma_6d).unsqueeze(0)).unsqueeze(0)
        return pred_points

    # ----- Potential OT improvements (see readme/docs) -----
    # e.g., unbalanced OT (reach>0), multiscale backend, p=1 for robustness,
    # annealing blur, staged OT+MSE, etc.

    def _init_sinkhorn(self, p: int, blur: float, scaling: float, reach: float, backend: str, **_ignored):
        """
        Initialize the Sinkhorn OT loss (fully controlled via external parameters).
        """
        # debias=True is important for matching high-dimensional features
        self.sinkhorn_loss = self.opt_config.create_sinkhorn_loss(
            p=p,
            blur=blur,
            scaling=scaling,
            reach=reach,
            debias=True,
            backend=backend
        )
        print(
            f"✓ Sinkhorn OT Loss initialized: "
            f"p={p}, blur={blur:.4f}, scale={scaling}, reach={reach}, backend={backend}"
        )

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
 
    def initialize_model(
                        self,
                        initial_sigma: float = 1e-2,
                        random_init: bool = False,
                        sigma_min: float = 1e-3, sigma_max: float = 1,
                        init_type: str = "uniform",
                        offset_y_km: tuple = (-10, -5), offset_z_km: tuple = (10, 20),
                        offset_rho: float = 1000.0,
                        use_prior_model: bool = False,
                        prior_options: Optional[Dict[str, Any]] = None,
                        initial_model_sigma: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Initialize the inversion model.
        
        Args:
            initial_sigma: Initial conductivity for uniform background (S/m).
            random_init: Whether to use random initialization (log-uniform + Gaussian smoothing).
            sigma_min: Lower bound for random initialization (S/m).
            sigma_max: Upper bound for random initialization (S/m).
            init_type: "uniform" | "random" | "offset". offset=place block at wrong location for testing.
            offset_y_km: (y_min, y_max) km for offset block, when init_type="offset".
            offset_z_km: (z_min, z_max) km depth for offset block, when init_type="offset".
            offset_rho: Resistivity (Ω·m) of offset block when init_type="offset".
            use_prior_model: If True, build ``sigma_init`` from ``prior_options`` (GMT .grd priors);
                see :mod:`mt2d_inv.prior_grids`. Ignores ``init_type``, ``random_init``, and ``offset_*``.
            prior_options: Dict for ``build_prior_sigma_earth`` (lon, lat, sediment_grd, slab_grd, conductivities, ...).
                        initial_model_sigma: Optional 2D conductivity model (S/m) on cell centers.
                                Accepts either:
                                - full model including air: shape (len(zn)-1, len(yn)-1)
                                - earth-only model: shape (len(zn)-1-nza, len(yn)-1)
                                Notes:
                                - The air layer is NOT inverted; forward modeling uses a fixed air conductivity
                                    given by ``self.air_sigma_value``.
                                - If you pass a full model, the top ``nza`` rows are ignored; inversion parameters
                                    come from the earth-only part.
        """
        # 1) Cell-center coordinates
        # self.zn / self.yn are edges; model parameters live at cell centers
        # Ensure tensors are on the correct device
        zn_tensor = self.zn.clone().detach().to(device=self.device, dtype=torch.float64)
        yn_tensor = self.yn.clone().detach().to(device=self.device, dtype=torch.float64)
        # Compute centers
        z_centers = (zn_tensor[:-1] + zn_tensor[1:]) / 2.0
        y_centers = (yn_tensor[:-1] + yn_tensor[1:]) / 2.0
        
        # Model shape (for consistency checks)
        nz_model = len(z_centers)  # full: len(zn_tensor) - 1 (includes air)
        ny_model = len(y_centers)  # full: len(yn_tensor) - 1

        if not hasattr(self, "nza"):
            self.nza = 0
        nz_earth = nz_model - int(self.nza)

        expected_full = (nz_model, ny_model)
        expected_earth = (nz_earth, ny_model)

        # User-provided initial model overrides all other init modes.
        if initial_model_sigma is not None:
            if use_prior_model or random_init or (init_type and init_type.strip().lower() not in {"uniform", ""}):
                raise ValueError(
                    "initial_model_sigma is mutually exclusive with use_prior_model/random_init/init_type. "
                    "Provide only one initialization mode."
                )

            sigma_in = torch.as_tensor(initial_model_sigma, device=self.device, dtype=torch.float64)
            if sigma_in.ndim != 2:
                raise ValueError(f"initial_model_sigma must be 2D, got shape {tuple(sigma_in.shape)}")

            passed_full = False
            if tuple(sigma_in.shape) == expected_full:
                passed_full = True
                # Air layers in the provided full model are ignored for forward modeling;
                # use earth-only part for inversion parameters.
                sigma_init = sigma_in[int(self.nza):, :]
            elif tuple(sigma_in.shape) == expected_earth:
                sigma_init = sigma_in
            else:
                raise ValueError(
                    f"initial_model_sigma shape mismatch: got {tuple(sigma_in.shape)}, "
                    f"expected full {expected_full} or earth-only {expected_earth}."
                )

            # If a full model was passed, ensure it's finite (avoid confusing NaNs in diagnostics).
            if passed_full and (not torch.isfinite(sigma_in).all()):
                raise ValueError("initial_model_sigma (full) contains non-finite values (NaN/Inf)")

            if not torch.isfinite(sigma_init).all():
                raise ValueError("initial_model_sigma contains non-finite values (NaN/Inf)")
            if (sigma_init <= 0).any():
                raise ValueError("initial_model_sigma must be strictly positive everywhere in earth layers")

            # Save the initial model for plotting/diagnostics (full shape).
            # Always assemble with the fixed air conductivity so plots match the forward convention.
            sigma_init_full = self._assemble_sigma_full(sigma_init)
            self.initial_model_sigma = sigma_init_full.detach().clone()

            # Parameterization: earth-only
            self.model_log_sigma = nn.Parameter(torch.log(sigma_init))
            self.model_log_sigma.requires_grad = True

            print("✓ Model initialization complete: User-provided initial_model_sigma.")
            if passed_full and self.nza > 0:
                print("  - Note: provided air-layer values are ignored; fixed air_sigma_value is used.")
            if self.nza > 0:
                print(f"  - Air layer fixed: nza={self.nza}, air_sigma={self.air_sigma_value:.2e} S/m")
            return
        
        # If self.nz and self.ny are defined, check consistency
        if hasattr(self, 'nz') and hasattr(self, 'ny'):
            if nz_model != self.nz - 1 or ny_model != self.ny - 1:
                print(f"Warning: model size ({nz_model}, {ny_model}) != expected ({self.nz-1}, {self.ny-1})")
        
        if use_prior_model:
            from .prior_grids import build_prior_sigma_earth

            z_earth = z_centers[self.nza:]
            sigma_np = build_prior_sigma_earth(
                y_centers_m=y_centers.detach().cpu().numpy(),
                z_earth_centers_m=z_earth.detach().cpu().numpy(),
                nz_earth=nz_earth,
                ny_model=ny_model,
                initial_sigma=float(initial_sigma),
                prior_options=dict(prior_options or {}),
            )
            sigma_init = torch.as_tensor(sigma_np, device=self.device, dtype=torch.float64)
        elif init_type == "offset":
            sigma_init = torch.ones((nz_earth, ny_model), device=self.device, dtype=torch.float64) * initial_sigma
            z_earth = z_centers[self.nza:]
            y_min_m, y_max_m = offset_y_km[0] * 1e3, offset_y_km[1] * 1e3
            z_min_m, z_max_m = offset_z_km[0] * 1e3, offset_z_km[1] * 1e3
            sigma_block = 1.0 / offset_rho
            for i in range(nz_earth):
                for j in range(ny_model):
                    if (z_min_m <= z_earth[i].item() < z_max_m and
                            y_min_m <= y_centers[j].item() < y_max_m):
                        sigma_init[i, j] = sigma_block
        elif random_init or init_type == "random":
            # Log-uniform sampling
            # Standard practice: sample in log10 space so each order of magnitude is equally likely
            log_min = np.log10(sigma_min)
            log_max = np.log10(sigma_max)
            
            # Random samples in log10 space
            random_exponents = torch.rand((nz_earth, ny_model), device=self.device, dtype=torch.float64)
            random_exponents = log_min + (log_max - log_min) * random_exponents
            
            # Convert back to linear conductivity
            sigma_init = 10 ** random_exponents
    
            # Spatial smoothing (Gaussian filter)
            # Standard practice: use correlated noise rather than pure white noise
            # This stabilizes gradients and mimics blocky geology
            sigma_np = sigma_init.cpu().numpy()
            # sigma=2 means ~2-cell smoothing radius, reducing checkerboard artifacts
            sigma_smooth = gaussian_filter(sigma_np, sigma=2.0) 
            sigma_init = torch.tensor(sigma_smooth, device=self.device, dtype=torch.float64)
        else:
            # Default: uniform initialization
            sigma_init = torch.ones((nz_earth, ny_model), device=self.device, dtype=torch.float64) * initial_sigma
    
        # Save the initial model for plotting/diagnostics (full shape)
        sigma_init_full = self._assemble_sigma_full(sigma_init)
        self.initial_model_sigma = sigma_init_full.detach().clone()
    
        # Parameterization: use natural log conductivity as inversion parameters
        # This guarantees positivity of conductivity
        # Only subsurface (earth) layers are inverted
        self.model_log_sigma = nn.Parameter(torch.log(sigma_init)) 
        self.model_log_sigma.requires_grad = True
        
        # Initialization info
        if use_prior_model:
            init_desc = "Prior grids (sediment/slab .grd via prior_grids)"
        elif init_type == "offset":
            init_desc = f"Offset (block {offset_y_km} km × {offset_z_km} km, {offset_rho} Ω·m)"
        elif random_init or init_type == "random":
            init_desc = "Random (Log-Uniform + Smooth)"
        else:
            init_desc = "Uniform"
        print(f"✓ Model initialization complete: {init_desc}.")
        if self.nza > 0:
            print(f"  - Air layer fixed: nza={self.nza}, air_sigma={self.air_sigma_value:.2e} S/m")
    
    def set_reference_model(self, sig_ref: Union[np.ndarray, torch.Tensor]):
        """
        Set a reference model (for regularization).
        
        Args:
            sig_ref: Reference conductivity model [nz-1, ny-1], consistent with the model grid
        """
        sig_ref_t = torch.as_tensor(sig_ref, device=self.device, dtype=torch.float64)
        if sig_ref_t.ndim != 2:
            raise ValueError(f"sig_ref must be 2D, got shape {tuple(sig_ref_t.shape)}")

        # Expected shapes are defined by the current grid (edges) and nza.
        nz_full = int(self.zn.numel()) - 1
        ny_full = int(self.yn.numel()) - 1
        expected_full = (nz_full, ny_full)
        expected_earth = (nz_full - int(self.nza), ny_full)

        if tuple(sig_ref_t.shape) == expected_full:
            sig_ref_earth = sig_ref_t[int(self.nza):, :]
        elif tuple(sig_ref_t.shape) == expected_earth:
            sig_ref_earth = sig_ref_t
        else:
            raise ValueError(
                f"sig_ref shape mismatch: got {tuple(sig_ref_t.shape)}, expected full {expected_full} "
                f"or earth-only {expected_earth}."
            )

        if not torch.isfinite(sig_ref_earth).all():
            raise ValueError("sig_ref contains non-finite values (NaN/Inf) in earth layers")
        if (sig_ref_earth <= 0).any():
            raise ValueError("sig_ref must be strictly positive everywhere in earth layers")

        # Save full reference model for plotting/diagnostics using the fixed-air convention.
        self.sig_ref = self._assemble_sigma_full(sig_ref_earth).detach().clone()
        # Constraints use earth-only parameters (must match self.model_log_sigma shape)
        self.model_log_sigma_ref = torch.log(sig_ref_earth).to(self.device, dtype=torch.float64)
        print("✓ Reference model set")
    
    def update_lambda_by_gradient_balance(
        self,
        loss_data: torch.Tensor,
        loss_model: torch.Tensor,
        current_lambda: float,
        alpha: float = 0.8,
        lambda_min: float = 1e-6,
        bl: float = 2.0,
        min_ratio_for_update: float = 0.01
    ):
        """
        Adaptively update lambda based on gradient magnitudes.
        Lambda is only allowed to decrease (gradually relax regularization).
        Target (soft): ||∇Phi_d|| ≲ bl * lambda * ||∇Phi_m||.

        Args:
            loss_data: Data loss tensor.
            loss_model: Model/regularization loss tensor.
            current_lambda: Current regularization weight.
            alpha: Exponent for exponential decrease when relaxing lambda (default: 0.8).
            lambda_min: Lower bound for lambda.
            bl: Balance factor scaling the gradient-norm target. Larger bl = more tolerant,
            min_ratio_for_update: Skip lambda update if ratio < this (avoid overly
                aggressive decrease when ratio is tiny). Default: 0.01.

        Returns:
            Tuple of (new_lambda, norm_d, norm_m) where norm_m is **raw** ||∇Φ_m|| (not ×λ).
            Callers that log/plot the model term should multiply by λ: λ·||∇Φ_m||.
        """

        # 1) Current gradient norms
        grad_d = torch.autograd.grad(
            loss_data,
            self.model_log_sigma,
            retain_graph=True,
            create_graph=False
        )[0]

        grad_m = torch.autograd.grad(
            loss_model,
            self.model_log_sigma,
            retain_graph=True,
            create_graph=False
        )[0]
        
        norm_d_raw = torch.sqrt(torch.mean(grad_d**2))
        norm_m_raw = torch.sqrt(torch.mean(grad_m**2)) + 1e-12
        
        # Record raw values
        norm_d_item = norm_d_raw.item()
        norm_m_item = norm_m_raw.item()
        
        # 2) Update histories
        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)
        # Keep histories bounded (avoid unbounded memory growth)
        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        # 3) Moving average (smooth gradient norms)
        # Fixed smoothing window (per request): only keep the last 3 samples.
        window_size = 3
        if len(self.grad_norm_d_history) >= window_size:
            norm_d_smooth = np.mean(self.grad_norm_d_history[-window_size:])
            norm_m_smooth = np.mean(self.grad_norm_m_history[-window_size:])
        else:
            # If history is short, fall back to current values
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        # 4) Ratio (using smoothed gradient norms)
        # Use smoothed norms directly to avoid single-step noise.
        # Note: do not use historical ratio statistics; gradients typically decay rapidly.
        ratio = norm_d_smooth / (bl * current_lambda * norm_m_smooth + 1e-12)
        ratio = float(ratio)
        # Store ratio history (monitoring only)
        self.ratio_history.append(ratio)

        # 5) Lambda can only decrease (exponential decrease)
        if ratio < 1.0:
            # If ratio is extremely small, decreasing lambda using the raw ratio can be too aggressive.
            # Instead of skipping updates (which can freeze lambda at a too-large value),
            # clamp ratio from below to allow a gradual monotonic decrease.
            ratio_eff = max(ratio, float(min_ratio_for_update))
            new_lambda = current_lambda * (ratio_eff ** alpha)
        else:
            new_lambda = current_lambda

        # 6) Safety constraints
        new_lambda = float(max(new_lambda, lambda_min))
        new_lambda = min(current_lambda, new_lambda)  # monotonic non-increase

        return new_lambda, norm_d_item, norm_m_item

    def run_inversion(self, 
                    n_epochs: int = 100, 
                    mode: str = "6dot",
                    progress_interval: int = 10,
                    current_lambda: float = 0.01,
                    use_adaptive_lambda: bool = True,
                    compute_lambda_grads_every_epoch: bool = False,
                    lr: float = 0.05,
                    bl: float = 2.0,
                    norm_type = "L2",
                    use_reference_model: bool = False,
                    reference_weight: float = 0.1,
                    alpha_x: float = 1.0,
                    alpha_z: float = 1.0,
                    update_interval: int = 10,   # Update interval
                    warmup_epochs: int = 5,
                    alpha: float = 0.5,        
                    use_ot_weights: bool = True,  # 3dot only: noise-based marginal weights; 6dot ignores (uniform)
                    use_depth_weights: bool = True,  # Whether roughness uses depth weighting
                    depth_beta: float = 0.3,
                    rms_chi2_stop: float = 1.05,
                    profile_timing: bool = False,
                    # --- Blur (epsilon) annealing (opt-in, default disabled) ---
                    enable_blur_anneal: bool = False,
                    blur_anneal_window: int = 5,
                    blur_anneal_rel_change_thresh: float = 0.03, 
                    blur_anneal_factor: float = 0.9,
                    blur_anneal_min: float = 1e-4,
                    # Smooth data_loss to avoid small oscillations triggering anneal.
                    blur_anneal_smooth_window: int = 3,
                    # After each anneal, wait this many epochs before next anneal.
                    blur_anneal_cooldown_epochs: int = 20,
                    ):  
        """
        Run inversion.
        
        Args:
            n_epochs: Number of epochs
            mode: Inversion mode ('3dot' / '6dot' / 'mse')
            progress_interval: Logging interval
            current_lambda: Initial regularization weight lambda
            alpha_x: Weight for horizontal (x) roughness term in model regularization (default 1.0)
            alpha_z: Weight for vertical (z) roughness term in model regularization (default 1.0)
            use_ot_weights: If True, **3dot** uses (alpha, beta) from data_noise_std. **6dot** always
                uses uniform marginals; cost geometry is ``sigma_6d`` in ``ot_config`` only.
            use_depth_weights: If True, roughness uses depth weighting (z/z0)^beta; otherwise uniform
            depth_beta: Exponent beta in depth weighting w(z) = (z/z0)^beta (only used when use_depth_weights=True)

        Extension hook: Subclasses may override _on_epoch_end(epoch, n_epochs) for per-epoch logic
        (e.g. GPU cache cleanup, checkpointing, OT blur annealing, real-time plotting).

        profile_timing: If True, accumulate and print per-epoch time breakdown (forward, sinkhorn, backward, step).
        """
        if self.forward_operator is None:
            raise RuntimeError("Please set the forward operator first")

        self._last_inversion_mode = mode  # 供 print_ot_dimension_contributions 等检查

        # Timing
        total_start_time = time.time()
        self.time_stats['start_time'] = total_start_time
        self.time_stats['epoch_times'] = []
        
        # Total number of VALID data points (exclude NaN from data cleaning / missing data)
        num_data = sum((~torch.isnan(v)).sum().item() for v in self.obs_data.values())
        num_data = max(int(num_data), 1)  # avoid division by zero

        optimizer = self.opt_config.create_optimizer(
            [self.model_log_sigma], lr=lr, optimizer_type="AdamW"
        )

        # If reference-model constraint is requested but no reference has been set,
        # use the initial model by default (common and avoids silent no-op).
        if use_reference_model and self.model_log_sigma_ref is None:
            if getattr(self, "initial_model_sigma", None) is not None:
                self.set_reference_model(self.initial_model_sigma)
                print("✓ Reference model not provided; using initial_model_sigma as reference.")
            else:
                raise RuntimeError(
                    "use_reference_model=True but no reference model is set. "
                    "Call set_reference_model(...) after initialize_model(), or ensure initial_model_sigma exists."
                )
        
        # Depth weights are computed for earth layers only (air is not inverted/regularized)
        depth_weights = (
            self.constraint_calc.compute_depth_weights_from_zn(
                zn=self.zn,
                nza=self.nza,
                beta=depth_beta,
            )
            if use_depth_weights
            else None
        )

        # Precompute observation point clouds once (obs_data does not change during inversion)
        cloud_obs_6d = None
        cloud_obs_3d = None
        if mode == '6dot':
            cloud_obs_6d = self._prepare_6d_ot_cloud_obs(self.obs_data)
        elif mode == '3dot':
            cloud_obs_3d = {key: self._prepare_3d_ot_cloud(self.obs_data[key], key) for key in self.obs_data.keys()}

        # -----------------------------
        # Cache fixed masks/coordinates
        # -----------------------------
        # obs_data does not change during inversion, so masks and (freq, station) coordinates
        # after masking are constant. Cache them to avoid repeated expand/flatten/mask each epoch.
        n_freq = len(self.freqs)
        n_stations = len(self.stations)

        log_freq = torch.log10(self.freqs)
        norm_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-8)
        norm_stn = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)

        # 6D cache: valid_mask + masked (f, s)
        cache_6d = None
        if mode == "6dot":
            valid_mask_6d = self._get_6d_valid_mask()
            grid_f_6d = norm_freq.view(-1, 1).expand(n_freq, n_stations).flatten()[valid_mask_6d]
            grid_s_6d = norm_stn.view(1, -1).expand(n_freq, n_stations).flatten()[valid_mask_6d]
            cache_6d = {
                "valid_mask": valid_mask_6d,
                "grid_f": grid_f_6d,
                "grid_s": grid_s_6d,
            }

        # 3D cache: per-key valid_mask + masked (f, s)
        cache_3d = None
        if mode == "3dot":
            cache_3d = {}
            base_grid_f = norm_freq.view(-1, 1).expand(n_freq, n_stations).flatten()
            base_grid_s = norm_stn.view(1, -1).expand(n_freq, n_stations).flatten()
            for key in self.obs_data.keys():
                obs_raw = self.obs_data[key]
                valid_mask = ~torch.isnan(obs_raw.flatten())
                cache_3d[key] = {
                    "valid_mask": valid_mask,
                    "grid_f": base_grid_f[valid_mask],
                    "grid_s": base_grid_s[valid_mask],
                }

        # MSE cache: per-key valid indices + transformed obs + masked weights.
        # Avoid repeated isnan/log10/masking every epoch.
        mse_cache = None
        if mode == "mse":
            if not hasattr(self, "data_weights") or not isinstance(getattr(self, "data_weights", None), dict) or len(self.data_weights) == 0:
                # Best-effort: build weights if missing (obs/data_noise_std must already exist).
                nf = float(getattr(self, "noise_floor", 0.01) or 0.01)
                self._compute_data_weights(noise_floor=nf)

            mse_cache = {}
            eps = 1e-12
            te_w = float(self.te_weight)
            tm_w = float(self.tm_weight)
            for key, obs_raw in self.obs_data.items():
                if obs_raw is None or (not torch.is_tensor(obs_raw)):
                    continue
                if key not in self.data_weights:
                    continue

                obs_raw = obs_raw.to(self.device, dtype=torch.float64)
                valid_mask_flat = ~torch.isnan(obs_raw.flatten())
                if not torch.any(valid_mask_flat):
                    continue
                idx = torch.nonzero(valid_mask_flat, as_tuple=False).flatten().to(device=self.device)

                obs_flat = obs_raw.flatten().index_select(0, idx)
                if "rho" in key.lower():
                    obs_val = torch.log10(torch.clamp(obs_flat, min=eps))
                    is_rho = True
                else:
                    obs_val = obs_flat
                    is_rho = False

                w_raw = self.data_weights[key].to(self.device, dtype=torch.float64)
                w_flat = w_raw.flatten().index_select(0, idx)

                mode_weight = te_w if key in ("rhoxy", "phsxy") else tm_w
                mse_cache[key] = {
                    "idx": idx,
                    "obs": obs_val,
                    "w": w_flat,
                    "is_rho": is_rho,
                    "mode_weight": float(mode_weight),
                }

        def _prepare_6d_cloud_pred_cached(pred_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            if cache_6d is None:
                return self._prepare_6d_ot_cloud_pred(pred_dict)
            valid_mask = cache_6d["valid_mask"]
            grid_f = cache_6d["grid_f"]
            grid_s = cache_6d["grid_s"]

            def _norm_pred(key: str, data: torch.Tensor):
                data_flat = data.flatten()[valid_mask]
                if 'rho' in key.lower():
                    val_log = torch.log10(data_flat + 1e-12)
                    return (val_log - (-2.0)) / (6.0 - (-2.0))
                return data_flat / 90.0

            pred_rhoxy = _norm_pred('rhoxy', pred_dict['rhoxy'])
            pred_phsxy = _norm_pred('phsxy', pred_dict['phsxy'])
            pred_rhoyx = _norm_pred('rhoyx', pred_dict['rhoyx'])
            pred_phsyx = _norm_pred('phsyx', pred_dict['phsyx'])

            pred_points = torch.stack([grid_f, grid_s, pred_rhoxy, pred_phsxy, pred_rhoyx, pred_phsyx], dim=1)
            sigma_6d = self._get_6d_cost_sigma()
            pred_points = (pred_points * (1.0 / sigma_6d).unsqueeze(0)).unsqueeze(0)
            return pred_points

        def _prepare_3d_cloud_pred_cached(data_tensor: torch.Tensor, key: str) -> torch.Tensor:
            if cache_3d is None or key not in cache_3d:
                return self._prepare_3d_ot_cloud(data_tensor, key)
            valid_mask = cache_3d[key]["valid_mask"]
            grid_f = cache_3d[key]["grid_f"]
            grid_s = cache_3d[key]["grid_s"]

            data_flat = data_tensor.flatten()[valid_mask]
            if 'rho' in key.lower():
                val_log = torch.log10(data_flat + 1e-12)
                norm_val = (val_log - (-2.0)) / (6.0 - (-2.0))
            else:
                norm_val = data_flat / 90.0
            points = torch.stack([grid_f, grid_s, norm_val], dim=1)
            return points.unsqueeze(0)

        def _sync():
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()

        profile_times = {
            "forward": [],
            "data_prep": [],
            "data_term": [],
            "backward": [],
            "backward_data_probe": [],
            "backward_model_probe": [],
            "step": [],
            "regularization": [],
        } if profile_timing else None

        last_blur_anneal_epoch = None

        # Cache last computed gradient norms for logging; only computed on selected epochs.
        last_g_d_norm = float("nan")
        last_g_m_norm = float("nan")
        last_ratio = float("nan")
       
        # AdamW optimization
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            
            # 1) Forward
            if profile_timing:
                _sync()
                t0 = time.time()
            sigma_earth = torch.exp(self.model_log_sigma)
            sigma_full = self._assemble_sigma_full(sigma_earth)
            pred_dict = self.forward_operator(sigma_full)
            if profile_timing:
                _sync()
                profile_times["forward"].append(time.time() - t0)
            
            # 2) Data loss (single global scale: self.data_loss_scale)
            loss_data = torch.tensor(0.0, device=self.device)
            data_loss_scale = self.data_loss_scale
            with torch.no_grad():
                rms_chi2 = self.compute_rms_chi2(pred_dict)
            
            if profile_timing:
                _sync()
                t0 = time.time()
            if mode == '6dot':
                if profile_timing:
                    _sync()
                    t_prep = time.time()
                cloud_pred = _prepare_6d_cloud_pred_cached(pred_dict)
                if profile_timing:
                    _sync()
                    profile_times["data_prep"].append(time.time() - t_prep)
                loss_data = data_loss_scale * self.sinkhorn_loss(cloud_pred, cloud_obs_6d)

            elif mode == '3dot':
                te_w = float(self.te_weight)
                tm_w = float(self.tm_weight)
                prep_acc = 0.0
                for key in self.obs_data.keys():
                    pred = pred_dict[key]
                    if profile_timing:
                        _sync()
                        t_prep = time.time()
                    cloud_pred = _prepare_3d_cloud_pred_cached(pred, key)
                    if profile_timing:
                        _sync()
                        prep_acc += (time.time() - t_prep)
                    mode_weight = te_w if key in ("rhoxy", "phsxy") else tm_w
                    if use_ot_weights:
                        alpha_w, beta_w = self._build_3d_ot_weights(key)
                        loss_data += mode_weight * self.sinkhorn_loss(alpha_w, cloud_pred, beta_w, cloud_obs_3d[key]).sum()
                    else:
                        loss_data += mode_weight * self.sinkhorn_loss(cloud_pred, cloud_obs_3d[key]).sum()
                loss_data = loss_data * data_loss_scale
                if profile_timing:
                    profile_times["data_prep"].append(prep_acc)

            elif mode == 'mse':
                if profile_timing:
                    profile_times["data_prep"].append(0.0)
                if not mse_cache:
                    raise RuntimeError("MSE cache not built; check obs_data/data_weights setup")

                eps = 1e-12
                for key, item in mse_cache.items():
                    if key not in pred_dict:
                        continue
                    idx = item["idx"]
                    obs_val = item["obs"]
                    w_flat = item["w"]
                    mode_weight = float(item["mode_weight"])

                    pred_raw = pred_dict[key].to(self.device, dtype=torch.float64)
                    pred_flat = pred_raw.flatten().index_select(0, idx)
                    if item["is_rho"]:
                        p_val = torch.log10(torch.clamp(pred_flat, min=eps))
                    else:
                        p_val = pred_flat

                    # 梯度只会流向有效位置 (pred_flat 从有效位置 index_select 而来)
                    loss_component = torch.sum(((obs_val - p_val) * w_flat) ** 2)
                    loss_data += mode_weight * loss_component
                loss_data = data_loss_scale * loss_data / num_data
            else:
                raise ValueError(f"Unknown inversion mode: {mode}")
            if profile_timing:
                _sync()
                profile_times["data_term"].append(time.time() - t0)

            # 3) Regularization term (supports reference-model constraint)
            if profile_timing:
                _sync()
                t0 = time.time()
            if use_reference_model and self.model_log_sigma_ref is not None:
                loss_model = self.constraint_calc.calculate_combined_constraint(
                    model_log_sigma=self.model_log_sigma,
                    reference_model_log_sigma=self.model_log_sigma_ref,
                    roughness_weights=depth_weights,
                    roughness_norm=norm_type,
                    reference_norm=norm_type,
                    reference_weight=reference_weight,
                    alpha_x=alpha_x,
                    alpha_z=alpha_z,
                )
            else:
                loss_model = self.constraint_calc.calculate_weighted_roughness(
                    self.model_log_sigma,
                    depth_weights,
                    norm_type,
                    alpha_x=alpha_x,
                    alpha_z=alpha_z,
                )
            if profile_timing:
                _sync()
                profile_times["regularization"].append(time.time() - t0)
    
            # 4) Backprop
            if use_adaptive_lambda:
                # Only compute/record lambda-update gradients on the update epoch and its
                # two preceding epochs (fixed window=3).
                is_update_tick = (epoch >= warmup_epochs) and ((epoch - warmup_epochs) % update_interval == 0)
                phases = {0}
                if int(update_interval) >= 2:
                    phases.add(int(update_interval) - 1)
                if int(update_interval) >= 3:
                    phases.add(int(update_interval) - 2)

                if compute_lambda_grads_every_epoch:
                    compute_lambda_grads = True
                else:
                    compute_lambda_grads = False
                    if epoch >= warmup_epochs:
                        phase = int((epoch - warmup_epochs) % update_interval)
                        compute_lambda_grads = phase in phases
                    else:
                        # For the first update tick at epoch==warmup_epochs, also compute
                        # gradients at epochs warmup_epochs-2 and warmup_epochs-1.
                        compute_lambda_grads = epoch >= max(0, int(warmup_epochs) - 2)

                if compute_lambda_grads:
                    proposed_lambda, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                        loss_data,
                        loss_model,
                        current_lambda,
                        alpha=alpha,
                        lambda_min=1e-6,
                        bl=bl,
                    )
                    last_g_d_norm = float(g_d_norm)
                    last_g_m_norm = float(g_m_norm)
                    last_ratio = float(self.ratio_history[-1]) if getattr(self, "ratio_history", None) else float("nan")
                else:
                    proposed_lambda = current_lambda
                    g_d_norm = float("nan")
                    g_m_norm = float("nan")
                # 2) Decide whether to apply the update (using the passed-in schedule)
                is_warmup = epoch < warmup_epochs
                is_update_tick = (epoch >= warmup_epochs) and ((epoch - warmup_epochs) % update_interval == 0)
                
                if not is_warmup and is_update_tick:
                # Only consider updates after warmup and on scheduled ticks
                    if abs(proposed_lambda - current_lambda) / current_lambda > 0.05:
                        ratio_last = last_ratio
                        print(
                            f" [Auto-Lambda] Epoch {epoch}: Adjusted {current_lambda:.2e} -> {proposed_lambda:.2e} "
                            f"(ratio={ratio_last:.3e})"
                        )
                        current_lambda = proposed_lambda
            else:
                # If adaptive lambda is off, only compute gradient norms for monitoring
                if compute_lambda_grads_every_epoch:
                    _, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                        loss_data, loss_model, current_lambda, bl=bl
                    )
                else:
                    g_d_norm = float("nan")
                    g_m_norm = float("nan")
            # Monitoring / loss_history: model-term contribution matches total_loss gradient
            # (∇(λ Φ_m) = λ ∇Φ_m for fixed λ), so report λ·||∇Φ_m|| (same RMS scale as ||·|| on g_m).
            if not np.isfinite(float(g_m_norm)):
                g_m_norm_scaled = float("nan")
            else:
                g_m_norm_scaled = float(current_lambda) * float(g_m_norm)

            total_loss = loss_data + current_lambda * loss_model
            if profile_timing:
                # Probe-only gradients for fair OT/L2 timing comparison (do not update .grad).
                _sync()
                t0 = time.time()
                _ = torch.autograd.grad(
                    loss_data,
                    self.model_log_sigma,
                    retain_graph=True,
                    allow_unused=True,
                )
                _sync()
                profile_times["backward_data_probe"].append(time.time() - t0)

                _sync()
                t0 = time.time()
                _ = torch.autograd.grad(
                    current_lambda * loss_model,
                    self.model_log_sigma,
                    retain_graph=True,
                    allow_unused=True,
                )
                _sync()
                profile_times["backward_model_probe"].append(time.time() - t0)
            if profile_timing:
                _sync()
                t0 = time.time()
            total_loss.backward()
            if profile_timing:
                _sync()
                profile_times["backward"].append(time.time() - t0)

            torch.nn.utils.clip_grad_norm_([self.model_log_sigma], 1.0)
            if profile_timing:
                _sync()
                t0 = time.time()
            optimizer.step()
            if profile_timing:
                _sync()
                profile_times["step"].append(time.time() - t0)
            
            with torch.no_grad():
                self.model_log_sigma.clamp_(min=-11.5, max=4.6)
            
            # Record epoch runtime
            epoch_time = time.time() - epoch_start_time
            self.time_stats['epoch_times'].append(epoch_time)
            
            # Progress logging: print every progress_interval epochs
            if epoch % progress_interval == 0 or epoch == n_epochs - 1:
                if profile_timing and profile_times["forward"]:
                    fw = np.mean(profile_times["forward"]) * 1000
                    dp = np.mean(profile_times["data_prep"]) * 1000
                    dt = np.mean(profile_times["data_term"]) * 1000
                    bwd_d = np.mean(profile_times["backward_data_probe"]) * 1000
                    bwd_m = np.mean(profile_times["backward_model_probe"]) * 1000
                    rg = np.mean(profile_times["regularization"]) * 1000
                    bw = np.mean(profile_times["backward"]) * 1000
                    st = np.mean(profile_times["step"]) * 1000

                    # 针对3dot/6dot vs MSE的对比逻辑
                    if mode == "mse":
                        main = dt + bw
                        main_label = "data_term+backward"
                    else:
                        main = dt + bw + dp
                        main_label = "data_term+backward+prep"

                    main_pct = 100 * main / (main + 1e-8)

                    print(
                        f"  [Timing ms] forward={fw:.0f} | data_term={dt:.0f} ({100*dt/main:.1f}%) | "
                        f"backward={bw:.0f} ({100*bw/main:.1f}%) | data_prep={dp:.0f} | "
                        f"{main_label}={main:.0f}ms ({main_pct:.1f}% of main) | "
                        f"probe_d={bwd_d:.0f} probe_m={bwd_m:.0f} step={st:.0f} reg={rg:.0f}"
                    )
                elapsed_time = time.time() - total_start_time
                avg_epoch_time = np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else epoch_time
                remaining_epochs = n_epochs - epoch - 1
                remaining_time = avg_epoch_time * remaining_epochs
                progress_percent = (epoch + 1) / n_epochs * 100
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                remaining_str = str(timedelta(seconds=int(remaining_time)))
                eta_time = datetime.now() + timedelta(seconds=remaining_time)
                eta_str = eta_time.strftime("%H:%M:%S")
                print(f"Epoch {epoch+1}/{n_epochs} [ {progress_percent:5.1f}%]")
                print(f"  Elapsed: {elapsed_str} | Remaining: ~{remaining_str} | ETA: {eta_str}")
                print(f"  Epoch time: {epoch_time:.2f}s | Avg: {avg_epoch_time:.2f}s")
                print(f"  Total: {total_loss.item():.4e} | Data({mode}): {loss_data.item():.4e}")
                # Print cost weights when using MT2DInverterWeightedCost
                if hasattr(self, "_cost_weights") and self._cost_weights is not None and epoch == 0:
                    cw = self._cost_weights
                    print(f"  [Cost weights] w_s={cw['w_s']:.3f}, w_f={cw['w_f']:.3f}, w_d={cw['w_d']}")
                print(f"  Misfit(RMS χ²): {rms_chi2:.3f} | Rough: {loss_model.item():.2e} | Lam: {current_lambda:.7f}")
                print(f"  GradNorms: |g_d|={g_d_norm:.3e} | |λ·g_m|={g_m_norm_scaled:.3e}")
            
            # Store loss history
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'data_loss': loss_data.item(),
                'model_loss': loss_model.item(),
                'misfit': rms_chi2,
                'lambda': current_lambda,
                'epoch_time': epoch_time,
                # grad norms are only computed on selected epochs (update epoch and its
                # two preceding epochs); otherwise NaN.
                'grad_data_norm': float(g_d_norm) if np.isfinite(float(g_d_norm)) else float("nan"),
                # λ·||∇Φ_m|| (scaled), comparable to ||∇Φ_d|| for total gradient magnitude
                'grad_model_norm': g_m_norm_scaled,
            })

            # Optional: blur (epsilon) annealing driven by Data plateau.
            # Now supports both 6dot and 3dot (3dot uses total data_loss).
            if (
                enable_blur_anneal
                and mode in ("6dot", "3dot")
                and blur_anneal_window is not None
                and blur_anneal_window >= 1
                and epoch >= blur_anneal_window
            ):
                # Cooldown: do not adjust blur too frequently.
                if (
                    last_blur_anneal_epoch is not None
                    and blur_anneal_cooldown_epochs is not None
                    and blur_anneal_cooldown_epochs >= 0
                    and (epoch - last_blur_anneal_epoch) < int(blur_anneal_cooldown_epochs)
                ):
                    pass
                else:
                    # Moving-average smoothing on data_loss to reduce oscillation sensitivity.
                    smooth_window = max(1, int(blur_anneal_smooth_window))
                    data_losses = [float(x["data_loss"]) for x in self.loss_history]
                    end_idx = len(data_losses) - 1  # should equal current epoch
                    start_idx = end_idx - int(blur_anneal_window)
                    if start_idx >= 0:
                        smoothed_losses = []
                        for j in range(start_idx, end_idx + 1):
                            s = max(0, j - smooth_window + 1)
                            smoothed_losses.append(float(np.mean(data_losses[s:j + 1])))
                        eps = 1e-12
                        rel_changes = [
                            abs(smoothed_losses[i + 1] - smoothed_losses[i]) / (abs(smoothed_losses[i]) + eps)
                            for i in range(int(blur_anneal_window))
                        ]

                        if max(rel_changes) < float(blur_anneal_rel_change_thresh):
                            old_blur = float(self.ot_config.get("blur", 0.01))
                            new_blur = old_blur * float(blur_anneal_factor)
                            # Hard lower bound to avoid numerical underflow / precision issues.
                            new_blur = max(new_blur, float(blur_anneal_min))
                            if new_blur < old_blur:
                                print(
                                    f"  [Blur anneal] Epoch {epoch} ({mode}): "
                                    f"SMA plateau (max|Δ|/prev < {blur_anneal_rel_change_thresh:.3g}, "
                                    f"SMA_window={smooth_window}, cooldown={blur_anneal_cooldown_epochs}). "
                                    f"blur {old_blur:.4f} -> {new_blur:.4f}"
                                )
                                self.ot_config["blur"] = new_blur
                                self._init_sinkhorn(**self.ot_config)
                                last_blur_anneal_epoch = epoch

            # Subclass hook (e.g. for adaptive OT weights every N epochs)
            if hasattr(self, '_on_epoch_end') and callable(getattr(self, '_on_epoch_end')):
                self._on_epoch_end(epoch, n_epochs)

            if float(rms_chi2) < rms_chi2_stop:
                print(f"  [Early stop] RMS χ² = {float(rms_chi2):.3f} < {rms_chi2_stop}, epoch = {epoch}")
                break

        # End timing
        total_end_time = time.time()
        total_inversion_time = total_end_time - total_start_time
        
        # Update timing stats
        stats_update = {
            'end_time': total_end_time,
            'total_inversion_time': total_inversion_time,
            'avg_epoch_time': np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'min_epoch_time': np.min(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'max_epoch_time': np.max(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'std_epoch_time': np.std(self.time_stats['epoch_times']) if len(self.time_stats['epoch_times']) > 1 else 0,
        }
        if profile_timing and profile_times and profile_times["forward"]:
            pts = profile_times
            stats_update['profile'] = {
                'forward_ms': float(np.mean(pts["forward"]) * 1000),
                'data_prep_ms': float(np.mean(pts["data_prep"]) * 1000),
                'data_term_ms': float(np.mean(pts["data_term"]) * 1000),
                'backward_ms': float(np.mean(pts["backward"]) * 1000),
                'backward_data_probe_ms': float(np.mean(pts["backward_data_probe"]) * 1000),
                'backward_model_probe_ms': float(np.mean(pts["backward_model_probe"]) * 1000),
                'step_ms': float(np.mean(pts["step"]) * 1000),
                'regularization_ms': float(np.mean(pts["regularization"]) * 1000),
            }
            p = stats_update['profile']
            if mode == "mse":
                main = p['data_term_ms'] + p['backward_ms']
                main_label = "data_term+backward"
            else:
                main = p['data_term_ms'] + p['backward_ms'] + p['data_prep_ms']
                main_label = "data_term+backward+prep"

            main_pct = 100 * main / (main + 1e-8)
            print(f"[Profile Summary] {main_label} = {main:.0f}ms ({main_pct:.1f}% of main) | "
                  f"data_term={p['data_term_ms']:.0f}ms ({100*p['data_term_ms']/main:.1f}%) | "
                  f"backward={p['backward_ms']:.0f}ms ({100*p['backward_ms']/main:.1f}%) | "
                  f"data_prep={p['data_prep_ms']:.0f}ms | forward={p['forward_ms']:.0f}ms | "
                  f"probe_d={p['backward_data_probe_ms']:.0f} probe_m={p['backward_model_probe_ms']:.0f} "
                  f"step={p['step_ms']:.0f} reg={p['regularization_ms']:.0f}ms")
        self.time_stats.update(stats_update)
        
        print("Inversion completed.")
        return self.get_sigma_full().detach()

    def _apply_plot_style(self):
        """Apply a paper-friendly Matplotlib style (font-related only)."""
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                "mathtext.fontset": "stix",
                "axes.unicode_minus": False,
            }
        )

    def plot_model_comparison(
            self,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],     # X-axis bounds (ignored when clip_to_stations=True)
            ylim: list = [50, 0],      # Y-axis bounds
            clip_to_stations: bool = False,
            profile_extend_km: float = 5.0,
            profile_axis_width_km: float = None,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            synthetic_figwidth_scale: float = 1.0,
            axes_aspect: Optional[Union[str, float]] = None,
            figsize: Optional[Tuple[float, float]] = None,
        ):
        """
        Plot inverted model (log10 domain). If a true model (self.sig_true) exists,
        also plot a side-by-side comparison (true vs inverted). Air layer is masked,
        and a view mask is applied according to the given bounds.

        clip_to_stations: If True, horizontal extent follows stations. For real data,
            use True to get profile-based display (stations + profile_extend_km each side).
        profile_extend_km: For real data, extend display this many km beyond stations
            on each side. Default 5 km.
        profile_axis_width_km: If set (e.g. 50) with real-data profile mode, force matplotlib
            x-axis to [0, width] km (left = st_min - profile_extend_km in physical coords).
            Default None uses (st_max - st_min) + 2 * profile_extend_km.

        vmin/vmax: Optional color scale limits for the plotted quantity (log10 resistivity).
            - If provided, both true/inverted panels (if present) use the same limits.
            - If omitted (None), auto-scale using masked min/max ± 0.5 (current behavior).
        synthetic_figwidth_scale: Synthetic (true model) plots only. >1 widens x vs depth on screen
            and expands figure width; 1.0 is 1:1 km scaling. Ignored for real-data profile plots.
        axes_aspect: Override the x–z axes aspect (Matplotlib "aspect").
            - None: keep default behavior (synthetic: 1/synthetic_figwidth_scale; real data: 'auto')
            - 1.0: 1 km (x) equals 1 km (z)
            - 0.5: x is stretched 2× relative to z
        figsize: If provided, overrides the computed figure size in inches, e.g. (10, 6).
        """
        self._apply_plot_style()
        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = self.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true_model = hasattr(self, "sig_true") and isinstance(self.sig_true, torch.Tensor)
        # Real data: profile extent = stations + extend_km each side; 0-based display, triangles above
        use_profile = not has_true_model
        if use_profile and clip_to_stations:
            extend_km = profile_extend_km
            offset_km = st_min - extend_km
            if profile_axis_width_km is not None:
                w = float(profile_axis_width_km)
                xlim_use = [0.0, max(w, 1e-6)]
                xlim_orig = [offset_km, offset_km + w]
            else:
                xlim_orig = [st_min - extend_km, st_max + extend_km]
                xlim_use = [0.0, (st_max - st_min) + 2 * extend_km]
            xlabel_str = "Distance along profile (km)"
        else:
            xlim_use = [st_min, st_max] if clip_to_stations else xlim
            xlim_orig = xlim_use
            xlabel_str = "Distance (km)"
        # -------- Model values --------
        sigma_inv = self.get_sigma_full().detach().cpu().numpy()
        sigma_true = None
        if has_true_model:
            sigma_true = self.sig_true.detach().cpu().numpy()
            try:
                if sigma_true.shape == sigma_inv.shape:
                    data_range = np.log10(sigma_true.max()) - np.log10(sigma_true.min())
                    score = ssim(np.log10(sigma_true), np.log10(sigma_inv), data_range=data_range, win_size=3)
                    print(f"Model structural similarity (SSIM): {score:.4f}")
            except Exception:
                pass
        eps = 1e-12
        model_inv = np.log10(1.0 / (sigma_inv + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_true = "True log10 Resistivity"
        title_inv = "Inverted log10 Resistivity" if has_true_model else "Inverted log10 Resistivity"
        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        if use_profile and clip_to_stations:
            offset_km = st_min - profile_extend_km
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
        else:
            YY_plot = YY
            st_x_plot = st_km
        mask_air = ZZ < 0
        mask_ground = ZZ >= 0
        # -------- Build view mask (based on xlim_orig/ylim; YY in original coords) --------
        mask_x_min, mask_x_max = min(xlim_orig), max(xlim_orig)
        y_bottom, y_top = max(ylim), min(ylim)
        mask_view = (YY >= mask_x_min) & (YY <= mask_x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
        # -------- Figure size / aspect --------
        x_range = max(float(abs(max(xlim_use) - min(xlim_use))), 1e-6)
        y_range = max(float(abs(y_bottom - y_top)), 1e-6)

        # Station marker: keep surface border at z=0 (ylim top), and draw markers ABOVE the axes box.
        # Use axes-fraction y (>1) so it is always visible and does not change data limits.
        # Keep it close to the border to avoid colliding with the title.
        station_y_axes = 1.03
        panel_h = 5.0

        sx = max(float(synthetic_figwidth_scale), 1e-9) if has_true_model else 1.0
        axes_w_data = panel_h * (x_range / y_range) * sx
        fig_w = axes_w_data + 1.2
        fig_h = panel_h * (2.0 if has_true_model else 1.0)
        if figsize is not None:
            fig_w, fig_h = float(figsize[0]), float(figsize[1])
        aspect_default = (1.0 / sx) if has_true_model else "auto"
        aspect_use = aspect_default if axes_aspect is None else axes_aspect
        if has_true_model:
            model_true = np.log10(1.0 / (sigma_true + eps))
            model_true_masked = np.ma.masked_where(~mask_view, model_true)
        model_inv_masked = np.ma.masked_where(~mask_view, model_inv)
        # Min/max inside the mask for colorbar scaling (auto); allow user override.
        if has_true_model:
            auto_vmin = model_true_masked.min() - 0.5
            auto_vmax = model_true_masked.max() + 0.5
        else:
            auto_vmin = model_inv_masked.min() - 0.5
            auto_vmax = model_inv_masked.max() + 0.5

        vmin_use = float(auto_vmin) if vmin is None else float(vmin)
        vmax_use = float(auto_vmax) if vmax is None else float(vmax)
        int_ticks = np.arange(int(np.ceil(vmin_use)), int(np.floor(vmax_use)) + 1, dtype=int)

        # Font sizes (tuned for readability; per-plot, no global settings)
        title_fs = 30
        label_fs = 26
        tick_fs = 24
        cbar_label_fs = 25
        cbar_tick_fs = 24
        title_pad = 24
        # Mark station locations (km)
        st_x_km = self.stations.cpu().numpy() / 1000.0
        if has_true_model:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
            im1 = ax1.pcolormesh(YY, ZZ, model_true_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax1.invert_yaxis()
            ax1.set_title(title_true, fontsize=title_fs, pad=title_pad)
            ax1.set_ylabel('Depth (km)', fontsize=label_fs)
            ax1.set_xlim(xlim_use)
            ax1.set_ylim(ylim)
            ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax1.set_aspect(aspect_use, adjustable='box')
            ax1.tick_params(axis='both', labelsize=tick_fs)
            cb1 = plt.colorbar(im1, ax=ax1)
            cb1.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb1.set_ticks(int_ticks)
                cb1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb1.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb1.ax.tick_params(labelsize=cbar_tick_fs)
            ax1.scatter(
                st_x_km,
                np.full(st_x_km.shape, station_y_axes, dtype=float),
                transform=ax1.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
            im2 = ax2.pcolormesh(YY, ZZ, model_inv_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax2.invert_yaxis()
            ax2.set_title(title_inv, fontsize=title_fs, pad=title_pad)
            ax2.set_ylabel('Depth (km)', fontsize=label_fs)
            ax2.set_xlabel('Distance (km)', fontsize=label_fs)
            ax2.set_xlim(xlim_use)
            ax2.set_ylim(ylim)
            ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax2.set_aspect(aspect_use, adjustable='box')
            ax2.tick_params(axis='both', labelsize=tick_fs)
            cb2 = plt.colorbar(im2, ax=ax2)
            cb2.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb2.set_ticks(int_ticks)
                cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb2.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb2.ax.tick_params(labelsize=cbar_tick_fs)
            ax2.scatter(
                st_x_km,
                np.full(st_x_km.shape, station_y_axes, dtype=float),
                transform=ax2.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
        else:
            # Inverted only (real data, no true model): default aspect='auto' (fill); set axes_aspect=1.0 for 1:1 km
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), sharex=False)
            im = ax.pcolormesh(YY_plot, ZZ, model_inv_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax.invert_yaxis()
            ax.set_title(title_inv, fontsize=title_fs, pad=title_pad)
            ax.set_ylabel('Depth (km)', fontsize=label_fs)
            ax.set_xlabel(xlabel_str, fontsize=label_fs)
            ax.set_xlim(xlim_use)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax.set_aspect(aspect_use, adjustable='box')
            ax.tick_params(axis='both', labelsize=tick_fs)
            cb = plt.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb.set_ticks(int_ticks)
                cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb.ax.tick_params(labelsize=cbar_tick_fs)
            ax.scatter(
                st_x_plot,
                np.full(st_x_plot.shape, station_y_axes, dtype=float),
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
        # More padding to avoid title overlapping with axes/markers.
        if has_true_model:
            fig.subplots_adjust(hspace=0.28)
        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_initial_model(
            self,
            cmap: str = "jet_r",
            xlim: list = None,
            ylim: list = None,
            clip_to_stations: bool = False,
            profile_extend_km: float = 5.0,
            profile_axis_width_km: float = None,
            ylim_auto: bool = True,
            synthetic_figwidth_scale: float = 1.0,
        ):
        """Plot the initial model (log10 domain), masking air and applying a view mask.

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim as provided.
        profile_extend_km: For real data (no sig_true), extend display this many km beyond
            stations on each side. Default 5 km.
        profile_axis_width_km: If set with profile mode, force x-axis to [0, width] km.
        ylim_auto: If True, ylim is auto-set to grid depth extent to avoid empty space below.
        synthetic_figwidth_scale: Ignored in real-data profile mode. Otherwise same as plot_model_comparison.
        """
        self._apply_plot_style()
        if xlim is None:
            xlim = [-20, 20]
        if ylim is None:
            ylim = [50, 0]
        if self.initial_model_sigma is None:
            print("Initial model is not saved; call initialize_model first")
            return
        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = self.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true = hasattr(self, "sig_true") and self.sig_true is not None
        use_profile = not has_true and clip_to_stations
        if use_profile:
            offset_km = st_min - profile_extend_km
            if profile_axis_width_km is not None:
                w = float(profile_axis_width_km)
                xlim_use = [0.0, max(w, 1e-6)]
                xlim_orig = [offset_km, offset_km + w]
            else:
                xlim_orig = [st_min - profile_extend_km, st_max + profile_extend_km]
                xlim_use = [0.0, (st_max - st_min) + 2 * profile_extend_km]
            xlabel_str = "Distance along profile (km)"
            st_y_plot = 0.0  # triangle tip at z=0 km (surface)
        else:
            xlim_use = [st_min, st_max] if clip_to_stations else xlim
            xlim_orig = xlim_use
            xlabel_str = "Distance (km)"
            st_y_plot = 0.0
        # -------- Model values --------
        sigma_init = self.initial_model_sigma.detach().cpu().numpy()
        eps = 1e-12

        model_init = np.log10(1.0 / (sigma_init + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_init = "Initial log10 Resistivity"
        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        z_max_km = float(ZZ[ZZ >= 0].max()) if np.any(ZZ >= 0) else 10.0
        if ylim_auto and max(ylim) > z_max_km:
            ylim = [min(ylim[0], z_max_km * 1.05), ylim[1]]
        if use_profile:
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
        else:
            YY_plot = YY
            st_x_plot = st_km
        mask_air = ZZ < 0
        mask_ground = ZZ >= 0
        # -------- Build view mask (based on xlim_orig/ylim; YY in original coords) --------
        mask_x_min, mask_x_max = min(xlim_orig), max(xlim_orig)
        y_bottom, y_top = max(ylim), min(ylim)
        mask_view = (YY >= mask_x_min) & (YY <= mask_x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
        # -------- Figure size / aspect --------
        x_range = max(float(abs(max(xlim_use) - min(xlim_use))), 1e-6)
        y_range = max(float(abs(y_bottom - y_top)), 1e-6)
        panel_h = 5.0
        sx = 1.0 if use_profile else max(float(synthetic_figwidth_scale), 1e-9)
        axes_w = panel_h * (x_range / y_range) * sx
        fig_w = axes_w + 1.2  # colorbar
        fig_h = panel_h
        aspect_xz = (1.0 / sx) if not use_profile else 1.0
        model_init_masked = np.ma.masked_where(~mask_view, model_init)
        vmin = model_init_masked.min() - 0.5
        vmax = model_init_masked.max() + 0.5
        int_ticks = np.arange(int(np.ceil(vmin)), int(np.floor(vmax)) + 1, dtype=int)

        # Match font sizes with plot_model_comparison() for paper-ready figures
        title_fs = 30
        label_fs = 26
        tick_fs = 24
        cbar_label_fs = 25
        cbar_tick_fs = 24
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax.pcolormesh(YY_plot, ZZ, model_init_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.invert_yaxis()
        ax.set_title(title_init, fontsize=title_fs, pad=24)
        ax.set_ylabel('Depth (km)', fontsize=label_fs)
        ax.set_xlabel(xlabel_str, fontsize=label_fs)
        ax.set_xlim(xlim_use)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.set_aspect(aspect_xz, adjustable='box')
        ax.tick_params(axis='both', labelsize=tick_fs)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(label, fontsize=cbar_label_fs)
        if int_ticks.size > 0:
            cb.set_ticks(int_ticks)
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            cb.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
        cb.ax.tick_params(labelsize=cbar_tick_fs)
        plt.tight_layout()
        plt.show()

    def plot_loss_history(self, target_misfit: float = 1.0, plot_roughness_vs_misfit: bool = False):
        """
        Plot loss terms and parameter evolution during inversion.

        plot_roughness_vs_misfit
            If True, also open a second figure: model roughness (x) vs data misfit (y),
            see ``plot_roughness_misfit_curve``.
        """
        self._apply_plot_style()
        # Extract series
        epochs = [log['epoch'] for log in self.loss_history]
        misfit = [log['misfit'] for log in self.loss_history]
        lambdas = [log['lambda'] for log in self.loss_history]
        data_loss = [log['data_loss'] for log in self.loss_history]
        model_loss = [log['model_loss'] for log in self.loss_history]

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # --- Panel 1: Misfit (RMS) ---
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='Current RMS')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit Convergence", fontsize=title_fs)
        axes[0].set_xlabel("Epoch", fontsize=label_fs)
        axes[0].set_ylabel("RMS Error", fontsize=label_fs)
        axes[0].set_yscale('log')  # RMS often spans orders of magnitude
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].tick_params(axis='both', labelsize=tick_fs)
        axes[0].legend(fontsize=legend_fs)
        # --- Panel 2: Data Loss vs Model Loss (Roughness) ---
        ax2_twin = axes[1].twinx()
        p1, = axes[1].plot(epochs, data_loss, 'c-', label='Data Loss')
        p2, = ax2_twin.plot(epochs, model_loss, 'm-', label='Model Roughness')
        axes[1].set_title("Loss Components Trade-off", fontsize=title_fs)
        axes[1].set_xlabel("Epoch", fontsize=label_fs)
        axes[1].set_ylabel("Data Loss", color='c', fontsize=label_fs)
        axes[1].set_yscale('log')
        ax2_twin.set_ylabel("Roughness (Model Loss)", color='m', fontsize=label_fs)
        # Merge legends
        axes[1].tick_params(axis='both', labelsize=tick_fs)
        ax2_twin.tick_params(axis='both', labelsize=tick_fs)
        axes[1].legend(handles=[p1, p2], fontsize=legend_fs)
        axes[1].grid(True, alpha=0.3)
        # --- Panel 3: Lambda evolution ---
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization Parameter (Lambda)", fontsize=title_fs)
        axes[2].set_xlabel("Epoch", fontsize=label_fs)
        axes[2].set_ylabel("Lambda Value", fontsize=label_fs)
        axes[2].set_yscale('log')  # Lambda can span orders of magnitude
        axes[2].grid(True, which="both", ls="-", alpha=0.5)
        axes[2].tick_params(axis='both', labelsize=tick_fs)
        plt.tight_layout()
        plt.show()

        if plot_roughness_vs_misfit:
            self.plot_roughness_misfit_curve()

    def plot_roughness_misfit_curve(
        self,
        *,
        y_metric: str = "rms",
        log_x: bool = True,
        log_y: bool = True,
        mark_epochs: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        """
        L-curve style plot: model roughness (regularization term Φ_m) vs data misfit.

        Uses ``loss_history`` recorded by ``run_inversion`` (one point per logged epoch).

        Parameters
        ----------
        y_metric:
            ``"rms"`` — normalized RMS χ² from ``compute_rms_chi2`` (same as log ``Misfit(RMS χ²)``).
            ``"data_loss"`` — raw data objective ``loss_data`` used in the total loss.
        log_x, log_y:
            If True, use log scale on roughness / misfit axis (typical when both span orders of magnitude).
        mark_epochs:
            If True, scatter points colored by epoch index (line still connects in epoch order).
        ax:
            Optional matplotlib Axes; if None, a new figure is created.
        """
        if not getattr(self, "loss_history", None) or len(self.loss_history) == 0:
            raise RuntimeError("loss_history is empty; run inversion first.")

        self._apply_plot_style()
        rough = np.array([float(log["model_loss"]) for log in self.loss_history], dtype=float)
        if y_metric.lower() in ("rms", "chi2", "misfit"):
            yvals = np.array([float(log["misfit"]) for log in self.loss_history], dtype=float)
            y_label = r"Data misfit (RMS $\chi^2$)"
        elif y_metric.lower() in ("data_loss", "data", "loss_data"):
            yvals = np.array([float(log["data_loss"]) for log in self.loss_history], dtype=float)
            y_label = "Data loss (optimization objective)"
        else:
            raise ValueError(f"y_metric must be 'rms' or 'data_loss', got {y_metric!r}")

        epochs = np.array([int(log["epoch"]) for log in self.loss_history], dtype=int)
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5.5))

        valid = np.isfinite(rough) & np.isfinite(yvals) & (rough > 0) & (yvals > 0)
        if not np.any(valid):
            raise RuntimeError("No finite positive (roughness, misfit) pairs to plot.")

        r_p, y_p, e_p = rough[valid], yvals[valid], epochs[valid]
        ax.plot(r_p, y_p, "b-", linewidth=1.5, alpha=0.85, zorder=1, label="Trajectory (epoch order)")
        if mark_epochs:
            sc = ax.scatter(r_p, y_p, c=e_p, cmap="viridis", s=28, zorder=2, edgecolors="k", linewidths=0.3)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Epoch", fontsize=11)

        ax.set_xlabel(r"Model roughness $\Phi_m$ (regularization loss)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title("Roughness vs data misfit (L-curve trace)", fontsize=14)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.tick_params(axis="both", labelsize=10)
        if not mark_epochs:
            ax.legend(fontsize=10, loc="best")
        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_gradient_history(self):
        """Plot gradient-norm histories: data ||∇Φ_d|| and model λ·||∇Φ_m|| (scaled)."""
        self._apply_plot_style()
        logs = self.loss_history
        if len(logs) > 1:
            logs = logs[1:]

        epochs = [log['epoch'] for log in logs]
        g_d = [log['grad_data_norm'] for log in logs]
        g_m = [log['grad_model_norm'] for log in logs]

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, g_d, 'b-', label='||∇Φ_d|| (Data)', linewidth=2)
        plt.plot(epochs, g_m, 'r-', label='λ·||∇Φ_m|| (Model, scaled)', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=label_fs)
        plt.ylabel('Gradient norm', fontsize=label_fs)
        plt.title('Data vs model gradient norms (model: λ·||∇Φ_m||)', fontsize=title_fs)
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.grid(True, which='both', ls='-', alpha=0.5)
        plt.legend(fontsize=legend_fs)
        plt.tight_layout()
        plt.show()

    def compute_sensitivity_matrix(self, num_probes: int = 5):
        """
        Estimate per-cell sensitivity to observations using Hutchinson's stochastic estimator
        (approximately diag(J^T J)^{0.5}).
        
        Returns:
            sens_2d: (nz-1, ny-1) sensitivity magnitude
            YY, ZZ: grid coordinates (km) for pcolormesh
        """
        if self.forward_operator is None or self.model_log_sigma is None:
            raise RuntimeError("Please call set_forward_operator and initialize_model first")
        sigma_params = self.model_log_sigma.detach().requires_grad_(True)
        sigma_full = self._assemble_sigma_full(torch.exp(sigma_params))
        pred_dict = self.forward_operator(sigma_full)
        all_preds_list = []
        for key in self.obs_data.keys():
            pred_raw = pred_dict[key]
            if "rho" in key.lower():
                pred_val = torch.log10(pred_raw + 1e-12)
            else:
                pred_val = pred_raw
            all_preds_list.append(pred_val.flatten())
        all_preds = torch.cat(all_preds_list)
        n_data = all_preds.numel()
        sensitivity_sq_sum = torch.zeros_like(sigma_params.flatten())
        for _ in range(num_probes):
            v = torch.randint(0, 2, (n_data,), device=self.device).double() * 2 - 1
            v_dot_pred = torch.sum(v * all_preds)
            grad = torch.autograd.grad(v_dot_pred, sigma_params, retain_graph=True)[0].flatten()
            sensitivity_sq_sum += grad ** 2
        sensitivity = torch.sqrt(sensitivity_sq_sum / num_probes)
        sens_earth = sensitivity.view((self.nz - 1) - self.nza, self.ny - 1).detach().cpu().numpy()

        # Pad back to full size (air layers set to 0; they will be masked in plots)
        sens_2d = np.zeros((self.nz - 1, self.ny - 1), dtype=np.float64)
        sens_2d[self.nza:, :] = sens_earth
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:]).cpu().numpy()
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:]).cpu().numpy()
        YY, ZZ = np.meshgrid(yc, zc)
        return sens_2d, YY, ZZ

    def plot_sensitivity(self, xlim=None, ylim=None, cmap: str = "viridis", clip_to_stations: bool = True,
                         profile_extend_km: float = 5.0):
        """Plot sensitivity matrix heatmap (per-cell sensitivity to observations).

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim when provided.
        profile_extend_km: For real data (no sig_true), extend display this many km beyond
            stations on each side. Default 5 km.
        """
        self._apply_plot_style()
        sens_2d, YY, ZZ = self.compute_sensitivity_matrix()
        eps = 1e-16
        sens_log = np.log10(sens_2d + eps)
        mask_air = ZZ < 0
        sens_masked = np.ma.masked_where(mask_air, sens_log)
        st_km = self.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true = hasattr(self, "sig_true") and self.sig_true is not None
        use_profile = not has_true and clip_to_stations
        if clip_to_stations and not use_profile:
            x_min, x_max = st_min, st_max
        elif not clip_to_stations:
            x_min, x_max = float(YY.min()), float(YY.max())
        else:
            x_min, x_max = st_min - profile_extend_km, st_max + profile_extend_km
        if xlim is not None:
            x_min, x_max = xlim[0], xlim[1]
        if use_profile:
            offset_km = st_min - profile_extend_km
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
            x_min, x_max = 0.0, (st_max - st_min) + 2 * profile_extend_km
            xlabel_str = "Distance along profile (km)"
            st_y_plot = 0.0  # triangle tip at z=0 km (surface)
        else:
            YY_plot = YY
            st_x_plot = st_km
            xlabel_str = "Distance (km)"
            st_y_plot = 0.0
        y_min, y_max = (float(ZZ.min()), float(ZZ.max())) if ylim is None else (ylim[0], ylim[1])
        x_range = max(abs(x_max - x_min), 1e-6)
        y_range = max(abs(y_max - y_min), 1e-6)
        panel_h = 5.0
        axes_w = panel_h * (x_range / y_range) if use_profile else 2.0 * panel_h * (x_range / y_range)
        fig_w = axes_w + 1.2
        fig, ax = plt.subplots(figsize=(fig_w, panel_h))

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11
        cbar_label_fs = 12
        cbar_tick_fs = 10
        im = ax.pcolormesh(YY_plot, ZZ, sens_masked, cmap=cmap, shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel_str, fontsize=label_fs)
        ax.set_ylabel("Depth (km)", fontsize=label_fs)
        ax.set_title("Sensitivity Matrix (∂log pred / ∂log σ)", fontsize=title_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(r"log$_{10}$ Sensitivity", fontsize=cbar_label_fs)
        cb.ax.tick_params(labelsize=cbar_tick_fs)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.set_aspect(1.0 if use_profile else 0.5, adjustable='box')  # Real data: 1:1; synthetic: 2x
        ax.scatter(st_x_plot, np.full_like(st_x_plot, st_y_plot), c="k", s=10, marker="v", label="Stations")
        ax.legend(loc="upper right", fontsize=legend_fs)
        plt.tight_layout()
        plt.show()

    def plot_data_fitting(self, station_indices=None, *, plot_noise_cap: Optional[float] = None):
        """
        Plot data-fit comparisons for selected stations.

        plot_noise_cap: Optional upper limit on **displayed** error-bar sigmas only, in the same
            units as ``data_noise_std`` (log10(ρ) std for rho*; (φ/90) std for phs*). Does not
            affect ``_compute_data_weights``, RMS χ², or OT — lowering σ in the inversion path
            would inflate weights (∝ 1/σ²) on formally noisy points. If None, uses
            ``self.plot_noise_cap`` when set.
        """
        self._apply_plot_style()
        with torch.no_grad():
            sigma_full = self.get_sigma_full()
            pred_dict = self.forward_operator(sigma_full)
        freqs = self.freqs.cpu().numpy()
        n_stations = len(self.stations)
        if station_indices is None:
            station_indices = [0, n_stations // 2, n_stations - 1]

        title_fs = 13
        label_fs = 12
        tick_fs = 10
        legend_fs = 10
        n_plots = len(station_indices)
        fig, axes = plt.subplots(
            2, n_plots,
            figsize=(5 * n_plots, 9),
            sharex=True
        )
        if n_plots == 1:
            axes = axes.reshape(2, 1)
        # Noise floors match weighting/RMS; optional plot_noise_cap only trims the bars.
        noise_floor = float(getattr(self, "noise_floor", 0.01) or 0.01)
        sigma_rho_floor = noise_floor / float(np.log(10.0))
        phase_error_deg_floor = max(noise_floor * 28.6, 0.5)
        bar_cap = plot_noise_cap if plot_noise_cap is not None else getattr(self, "plot_noise_cap", None)
        if bar_cap is not None:
            bar_cap = float(bar_cap)
        rho_obs_all = []
        for i, st_idx in enumerate(station_indices):
            st_pos = self.stations[st_idx].item() / 1000.0
            st_id = None
            if getattr(self, "station_ids", None) is not None and int(st_idx) < len(self.station_ids):
                st_id = str(self.station_ids[int(st_idx)])
            if st_id is None or st_id.strip() == "":
                st_id = f"S{int(st_idx) + 1}"
            ax_rho = axes[0, i]
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_rho = f"rho{mode}"
                if key_rho not in self.obs_data:
                    continue
                rho_obs = self.obs_data[key_rho][:, st_idx].cpu().numpy()
                rho_pred = pred_dict[key_rho][:, st_idx].cpu().numpy()
                # rho is plotted on a log y-axis; ignore non-finite and non-positive values.
                valid = np.isfinite(rho_obs) & (rho_obs > 0)
                if np.any(valid):
                    rho_obs_valid = rho_obs[valid]
                    freqs_valid = freqs[valid]
                    rho_obs_all.append(rho_obs_valid)
                    sigma_log_eff_t = self.get_effective_data_noise_std(key_rho)
                    if sigma_log_eff_t is None:
                        sigma_log_eff = np.full_like(rho_obs_valid, sigma_rho_floor, dtype=float)
                    else:
                        sigma_log_eff = sigma_log_eff_t[:, st_idx].detach().cpu().numpy()[valid]
                    if bar_cap is not None:
                        sigma_log_eff = np.minimum(sigma_log_eff, bar_cap)
                    rho_up = rho_obs_valid * 10.0 ** sigma_log_eff
                    rho_dn = rho_obs_valid * 10.0 ** (-sigma_log_eff)
                    yerr = [rho_obs_valid - rho_dn, rho_up - rho_obs_valid]
                    ax_rho.errorbar(
                        freqs_valid, rho_obs_valid, yerr=yerr,
                        fmt='o', ms=4, alpha=0.6,
                        color=color, ecolor=color,
                        elinewidth=1, capsize=2,
                        label=f"Obs {mode.upper()}"
                    )
                rho_pred_safe = np.clip(np.nan_to_num(rho_pred, nan=1e-2, posinf=1e6, neginf=1e-6), 1e-6, 1e10)
                ax_rho.plot(
                    freqs, rho_pred_safe,
                    f'{color}-', lw=1.5,
                    label=f"Pred {mode.upper()}"
                )
            ax_rho.set_xscale("log")
            ax_rho.set_yscale("log")
            ax_rho.set_box_aspect(1.0)
            ax_rho.set_title(f"{st_id}\nApp. Resistivity", fontsize=title_fs)
            ax_rho.set_xlabel("Frequency (Hz)", fontsize=label_fs)
            ax_rho.tick_params(labelbottom=True)
            if i == 0:
                ax_rho.set_ylabel(r"$\rho_a$ ($\Omega\cdot$m)", fontsize=label_fs)
            ax_rho.grid(True, which="both", alpha=0.3)
            ax_rho.tick_params(axis='both', labelsize=tick_fs)
            ax_rho.legend(fontsize=legend_fs)
            ax_phs = axes[1, i]
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_phs = f"phs{mode}"
                if key_phs not in self.obs_data:
                    continue
                phs_obs = self.obs_data[key_phs][:, st_idx].cpu().numpy()
                phs_pred = pred_dict[key_phs][:, st_idx].cpu().numpy()
                valid = np.isfinite(phs_obs)
                if np.any(valid):
                    phs_obs_valid = phs_obs[valid]
                    freqs_valid = freqs[valid]

                    sigma_norm_eff_t = self.get_effective_data_noise_std(key_phs)
                    if sigma_norm_eff_t is None:
                        sigma_norm_eff = np.full_like(
                            phs_obs_valid, phase_error_deg_floor / 90.0, dtype=float
                        )
                    else:
                        sigma_norm_eff = (
                            sigma_norm_eff_t[:, st_idx].detach().cpu().numpy()[valid]
                        )
                    if bar_cap is not None:
                        sigma_norm_eff = np.minimum(sigma_norm_eff, bar_cap)
                    yerr = sigma_norm_eff * 90.0

                    ax_phs.errorbar(
                        freqs_valid, phs_obs_valid, yerr=yerr,
                        fmt='o', ms=4, alpha=0.6,
                        color=color, ecolor=color,
                        elinewidth=1, capsize=2
                    )

                phs_pred_safe = np.clip(np.nan_to_num(phs_pred, nan=45.0, posinf=90, neginf=0), 0, 90)
                ax_phs.plot(freqs, phs_pred_safe, f'{color}-', lw=1.5)
            ax_phs.set_xscale("log")
            ax_phs.set_ylim(0, 90)
            ax_phs.set_box_aspect(1.0)
            ax_phs.set_xlabel("Frequency (Hz)", fontsize=label_fs)
            if i == 0:
                ax_phs.set_ylabel("Phase (deg)", fontsize=label_fs)
            ax_phs.grid(True, which="both", alpha=0.3)
            ax_phs.set_title(f"{st_id}\nPhase", fontsize=title_fs)
            ax_phs.tick_params(axis='both', labelsize=tick_fs)
        if len(rho_obs_all) > 0:
            rho_obs_all = np.concatenate(rho_obs_all)
            rho_valid = rho_obs_all[(np.isfinite(rho_obs_all) & (rho_obs_all > 0))]
            if rho_valid.size > 0:
                ymin = float(rho_valid.min()) / 2.0
                ymax = float(rho_valid.max()) * 2.0
            else:
                ymin, ymax = 1.0, 1e6
            for ax in axes[0, :]:
                ax.set_ylim(ymin, ymax)
        axes[0, 0].invert_xaxis()
        f_min, f_max = freqs.min(), freqs.max()
        axes[0, 0].set_xlim(f_max, f_min) 
        plt.tight_layout()
        plt.show()

        
    def plot_1d_profiles(
        self,
        station_indices: List[int] = None,
        depth_limit_km: float = None
    ):
        """
        Plot 1D vertical profiles at selected station locations.

        If a true model (self.sig_true) is available, plot both true and inverted.
        Otherwise (real-data case), plot inverted only.
        """
        self._apply_plot_style()
        # 1) Model values
        sigma_inv = self.get_sigma_full().detach().cpu().numpy()
        has_true_model = hasattr(self, "sig_true") and isinstance(getattr(self, "sig_true"), torch.Tensor)
        sigma_true = self.sig_true.detach().cpu().numpy() if has_true_model else None

        # 2) Depth coordinates
        zn_km = 0.001 * self.zn.cpu().numpy()
        zc_km = 0.5 * (zn_km[:-1] + zn_km[1:])
        cell_mask = zc_km >= 0
        edge_mask = zn_km >= 0
        zc_ground = zc_km[cell_mask]
        zn_ground = zn_km[edge_mask]

        # 3) Station selection
        if station_indices is None:
            n_stations = len(self.stations)
            station_indices = [0, n_stations // 2, n_stations - 1]

        # 4) Shared x-axis limits
        all_vals = []
        y_centers = 0.5 * (self.yn[:-1] + self.yn[1:]).cpu().numpy()
        for st_idx in station_indices:
            col_idx = np.abs(y_centers - self.stations[st_idx].item()).argmin()

            all_vals.append(1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12))
            if has_true_model and sigma_true is not None:
                all_vals.append(1.0 / (sigma_true[cell_mask, col_idx] + 1e-12))
        all_vals = np.hstack(all_vals)
        valid = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        if valid.size == 0:
            xmin, xmax = 1.0, 1.0
        else:
            xmin = valid.min() / 2
            xmax = valid.max() * 2

        # 5) Plot
        title_fs = 13
        label_fs = 12
        tick_fs = 10
        legend_fs = 10
        n_plots = len(station_indices)
        fig, axes = plt.subplots(
            1, n_plots, figsize=(4 * n_plots, 6), sharey=True
        )
        if n_plots == 1:
            axes = [axes]
        for i, st_idx in enumerate(station_indices):
            ax = axes[i]
            col_idx = np.abs(
                y_centers - self.stations[st_idx].item()
            ).argmin()
            # -------- Inverted (cell-centered) --------
            val_inv = 1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12)
            xlabel = r"Resistivity ($\Omega \cdot m$)"
            if has_true_model and sigma_true is not None:
                # -------- True (edge-based, perfect blocks) --------
                val_true_cell = 1.0 / (sigma_true[cell_mask, col_idx] + 1e-12)
                # Expand cell values into an edge-based step function
                val_true_step = np.repeat(val_true_cell, 2)
                z_true_step = np.repeat(zn_ground, 2)[1:-1]
                ax.plot(
                    val_true_step,
                    z_true_step,
                    'k--',
                    linewidth=1.5,
                    label='True'
                )
            ax.step(
                val_inv,
                zc_ground,
                where='mid',
                color='r',
                linewidth=2,
                label='Inverted')
            ax.set_xscale('log')
            ax.set_xlim(xmin, xmax)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=label_fs)
            st_y_km = self.stations[st_idx].item() / 1000.0
            ax.set_title(f"Profile {st_y_km:.1f} km", fontsize=title_fs)
            ax.grid(True, which='both', alpha=0.3)
            if i == 0:
                ax.set_ylabel("Depth (km)", fontsize=label_fs)
            if depth_limit_km is not None:
                ax.set_ylim([depth_limit_km, 0])
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax.tick_params(axis='both', labelsize=tick_fs)
            ax.legend(fontsize=legend_fs)
        plt.tight_layout()
        plt.show()