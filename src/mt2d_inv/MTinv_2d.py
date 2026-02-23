import torch
import torch.nn as nn
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
from typing import Dict, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
                 ):

        # Default OT hyper-parameters
        default_ot = {
            "p": 2,              # Consider trying 1 for high-noise data
            "blur": 0.01,        # Consider a larger value, e.g., 0.05
            "scaling": 0.9,      # 0.5 can be more stable than 0.9
            "reach": None,       # None=balanced OT (mass-preserving, default); >0=unbalanced OT
            "backend": "tensorized"    # Or force "multiscale"
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

        # Air layer policy: forward-only (fixed), not inverted.
        # Specify the number of air layers via set_forward_operator(nza=...)
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
        self.sig_ref = 0.01  # Reference model (conductivity)
        self.model_log_sigma_ref = None  # Reference model (log conductivity)
        
        # Gradient histories (for moving-average smoothing and monitoring)
        self.grad_norm_d_history = []  # Data-term gradient norm history
        self.grad_norm_m_history = []  # Model-term gradient norm history
        self.ratio_history = []  # Ratio history

        self.time_stats = {
            'total_inversion_time': 0,
            'avg_epoch_time': 0,
            'epoch_times': [],
            'start_time': 0,
            'end_time': 0
        }
        
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
            raise ValueError(f"nza={self.nza} 过大，地下层数 <= 0（full={expected_full}）")

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
            raise RuntimeError("请先 initialize_model")
        sigma_earth = torch.exp(self.model_log_sigma)
        return self._assemble_sigma_full(sigma_earth)

    def create_synthetic_data(
        self,
        noise_level: float = 0.01,
        noise_type: str = "gaussian",
        outlier_frac: float = 0.05,
        outlier_strength: float = 4.0,
    ):
        """
        Generate 2D MT synthetic data: add noise at the impedance level, then
        compute apparent resistivity (rho) and phase (phi) consistently.

        Args:
            noise_level: Relative noise level (relative to |Z|)
            noise_type: "gaussian" for Gaussian noise only; "nongaussian" adds random outliers on top
            outlier_frac: Outlier fraction in [0, 1], only used when noise_type=="nongaussian"
            outlier_strength: Outlier strength (multiple of the baseline delta), only for "nongaussian"
        """
        
        
        if noise_type not in ("gaussian", "nongaussian"):
            raise ValueError(
                f'noise_type 必须为 "gaussian" 或 "nongaussian"，当前为 "{noise_type}"。'
                '请检查拼写（如 nonguassin -> nongaussian）。'
            )

        if noise_type == "nongaussian":
            if not (0 <= outlier_frac <= 1):
                raise ValueError(f"noise_type='nongaussian' 时 outlier_frac 须在 [0, 1]，当前为 {outlier_frac}")
            if outlier_strength <= 0:
                raise ValueError(f"noise_type='nongaussian' 时 outlier_strength 须 > 0，当前为 {outlier_strength}")
        self.noise_level = noise_level
        if self.sig_true is None:
            raise ValueError("sig_true 未设置，请先执行 inverter.sig_true = sig_true")
        if isinstance(self.sig_true, (list, tuple)):
            raise ValueError(
                "sig_true 不能为 tuple/list。若使用 create_commemi_2d0，应解包：\n"
                "  zn, yn, freq, ry, sig = MT2DTrueModels.create_commemi_2d0(nza, device)\n"
                "  inverter.sig_true = torch.tensor(sig, dtype=torch.float64, device=device)\n"
                "若使用 create_geological_models，直接赋值：\n"
                "  sig_true = MT2DTrueModels.create_geological_models(zn, yn, model_type='magma_chamber', device=device)\n"
                "  inverter.sig_true = sig_true"
            )
        if not isinstance(self.sig_true, torch.Tensor):
            raise TypeError(f"sig_true 须为 torch.Tensor，当前为 {type(self.sig_true)}")
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

            # -------- Impedance noise (relative, Gaussian) --------
            delta_real = noise_level * Zabs
            delta_imag = noise_level * Zabs

            noise_real = torch.randn_like(Z.real) * delta_real
            noise_imag = torch.randn_like(Z.imag) * delta_imag

            Z_obs = torch.complex(
                Z.real + noise_real,
                Z.imag + noise_imag
            )

            # -------- Non-Gaussian: add outliers on a subset of points --------
            if noise_type == "nongaussian":
                n_tot = Z_obs.numel()
                n_out = max(1, int(round(outlier_frac * n_tot)))
                idx = torch.randperm(n_tot, device=self.device)[:n_out]
                mask = torch.zeros(n_tot, dtype=torch.bool, device=self.device)
                mask[idx] = True
                mask = mask.reshape(Z_obs.shape)
                out_real = torch.randn_like(Z.real) * (outlier_strength * delta_real)
                out_imag = torch.randn_like(Z.imag) * (outlier_strength * delta_imag)
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
            self.data_std[f"delta_z{mode}_real"] = delta_real
            self.data_std[f"delta_z{mode}_imag"] = delta_imag
            self.data_std[f"Z{mode}"] = Z_obs

        self.obs_data = obs_data
        # First, propagate errors to obtain per-point noise std
        self.calculate_data_errors_2d()
        # Then, build data weights from the std
        self._compute_data_weights(noise_floor=self.noise_level)

        print("✓ Synthetic data generated")
        print(f"  -> Impedance noise level: {noise_level*100:.1f}% ({noise_type})")
        if noise_type == "nongaussian":
            print(f"  -> Non-Gaussian: outlier_frac={outlier_frac}, outlier_strength={outlier_strength}")

    def calculate_data_errors_2d(self):
        """
        Propagate impedance errors to obtain std-dev for rho and phi (2D), and
        construct dimensionless noise std-dev used for chi^2 / RMS.
        """
        eps = 1e-8

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

            delta_z_real = self.data_std[f"delta_z{mode}_real"]
            delta_z_imag = self.data_std[f"delta_z{mode}_imag"]

            # ---------- rho std-dev ----------
            dRho_dZr = 2.0 * Zr / (omega * MU)
            dRho_dZi = 2.0 * Zi / (omega * MU)

            delta_rho = torch.sqrt(
                (dRho_dZr * delta_z_real) ** 2 +
                (dRho_dZi * delta_z_imag) ** 2
            )

            # ---------- phi std-dev ----------
            dPhi_dZr = -Zi / (Zabs ** 2)
            dPhi_dZi =  Zr / (Zabs ** 2)

            delta_phi_rad = torch.sqrt(
                (dPhi_dZr * delta_z_real) ** 2 +
                (dPhi_dZi * delta_z_imag) ** 2
            )

            delta_phs = delta_phi_rad * 180.0 / np.pi

            # ---------- Dimensionless noise std-dev (for chi^2 / RMS) ----------
            # log10(rho)
            rho_noise_std_log = torch.clamp(
                delta_rho / (self.obs_data[key_rho] * np.log(10)),
                min=eps
            )

            # phi / 90°
            phs_noise_std_norm = torch.clamp(
                delta_phs / 90.0,
                min=eps
            )

            # ---------- Save ----------
            self.data_noise_std[f"rho{mode}"] = rho_noise_std_log
            self.data_noise_std[f"phs{mode}"] = phs_noise_std_norm

            print(f"✓ {mode.upper()} mode error propagation completed")
            print(f"   rho(log10) noise mean: {rho_noise_std_log.mean():.4f}")
            print(f"   phi(normalized) noise mean: {phs_noise_std_norm.mean():.4f}")

        print("✓ 2D data error propagation completed")

    def compute_rms_chi2(self, pred_dict):
        """
        Compute statistically meaningful RMS chi^2.
        RMS ~ 1 means the fit is at the noise level.
        """
        chi2_sum = 0.0
        n_data = 0

        for mode in ["xy", "yx"]:
            # ---------- rho ----------
            key_rho = f"rho{mode}"
            if key_rho in self.obs_data:
                rho_pred_log = torch.log10(pred_dict[key_rho] + 1e-12)
                rho_obs_log  = torch.log10(self.obs_data[key_rho] + 1e-12)

                res_rho = (rho_pred_log - rho_obs_log) / self.data_noise_std[key_rho]
                chi2_sum += torch.sum(res_rho ** 2)
                n_data += res_rho.numel()

            # ---------- phi ----------
            key_phs = f"phs{mode}"
            if key_phs in self.obs_data:
                phs_pred_norm = pred_dict[key_phs] / 90.0
                phs_obs_norm  = self.obs_data[key_phs] / 90.0

                res_phs = (phs_pred_norm - phs_obs_norm) / self.data_noise_std[key_phs]
                chi2_sum += torch.sum(res_phs ** 2)
                n_data += res_phs.numel()

        rms = torch.sqrt(chi2_sum / n_data)
        return rms.item()


    def _prepare_3d_ot_cloud(self, 
                             data_tensor: torch.Tensor, 
                             key: str) -> torch.Tensor:
        """
        Build a normalized (N, 3) point cloud: [Freq, Station, Value].

        All dimensions are normalized to [0, 1]; this is key for stable OT.
        Convention: input rho is linear-scale, but we normalize it in log10(rho).
        """
        n_freq = len(self.freqs)
        n_stations = len(self.stations)
        
        # 1) Normalize frequency (log domain) -> [0, 1]
        log_freq = torch.log10(self.freqs)
        norm_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-8)
        grid_freq = norm_freq.view(-1, 1).expand(n_freq, n_stations)
        
        # 2) Normalize stations -> [0, 1]
        norm_stn = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_stn = norm_stn.view(1, -1).expand(n_freq, n_stations)
        
        # 3) Normalize values -> [0, 1] (critical)
        if 'rho' in key.lower():
            # Fixed: take log10 in linear domain, then normalize by a physical range [-2, 6]
            val_log = torch.log10(data_tensor + 1e-12)
            norm_val = (val_log - (-2.0)) / (6.0 - (-2.0))
        else:
            # Phase: 0 ~ 90 degrees
            norm_val = data_tensor / 90.0
            
        # 4) Stack: (Batch, Points, Dim)
        points = torch.stack([grid_freq.flatten(), grid_stn.flatten(), norm_val.flatten()], dim=1)
        return points.unsqueeze(0)
    
    def _prepare_6d_ot_cloud(self, 
                              pred_dict: Dict[str, torch.Tensor], 
                              obs_dict: Dict[str, torch.Tensor]) -> tuple:
        """
        Multi-variable OT: consider all components at once.
        Returns a point cloud where each point is [freq, station, rhoxy, phsxy, rhoyx, phsyx].
        """
        # 1) Base dimensions
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        
        # 2) Normalize frequency and stations
        log_f = torch.log10(self.freqs)
        norm_f = (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-8)
        grid_f = norm_f.view(-1, 1).expand(n_freq, n_stn).flatten()
        # 3) Normalize all components
        # Convention: input rho is linear-scale, but we normalize it in log10(rho)
        
        norm_s = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_s = norm_s.view(1, -1).expand(n_freq, n_stn).flatten()

        def normalize_component(key: str,
                        pred_data: torch.Tensor,
                        obs_data: torch.Tensor):
            if 'rho' in key.lower():
                val_log_pred = torch.log10(pred_data + 1e-12)
                val_log_obs = torch.log10(obs_data + 1e-12)
                norm_pred = (val_log_pred.flatten() - (-2.0)) / (6.0 - (-2.0))
                norm_obs = (val_log_obs.flatten() - (-2.0)) / (6.0 - (-2.0))
            else:
                norm_pred = pred_data.flatten() / 90.0
                norm_obs = obs_data.flatten() / 90.0

            return norm_pred, norm_obs

        # 4) Build multi-variable point clouds
        pred_rhoxy, obs_rhoxy = normalize_component('rhoxy', pred_dict['rhoxy'], obs_dict['rhoxy'])
        pred_phsxy, obs_phsxy = normalize_component('phsxy', pred_dict['phsxy'], obs_dict['phsxy'])
        pred_rhoyx, obs_rhoyx = normalize_component('rhoyx', pred_dict['rhoyx'], obs_dict['rhoyx'])
        pred_phsyx, obs_phsyx = normalize_component('phsyx', pred_dict['phsyx'], obs_dict['phsyx'])

        pred_points = torch.stack([
            grid_f,
            grid_s,
            pred_rhoxy,
            pred_phsxy,
            pred_rhoyx,
            pred_phsyx
        ], dim=1)
        
        obs_points = torch.stack([
            grid_f,
            grid_s,
            obs_rhoxy,
            obs_phsxy,
            obs_rhoyx,
            obs_phsyx
        ], dim=1)
        
        return pred_points.unsqueeze(0), obs_points.unsqueeze(0)

    def _compute_frequency_weights(self) -> torch.Tensor:
        """
        Construct normalized frequency weights based on log10(freq) spacing (Delta log(f)).

        Used for OT weights to give equal mass to equally spaced frequencies in log space.
        freqs can be in arbitrary order; output has shape (n_freq,) and sums to 1.
        """
        freqs = self.freqs.to(self.device, dtype=torch.float64)
        log_f = torch.log10(freqs)
        diffs = torch.abs(log_f[1:] - log_f[:-1])
        N = len(freqs)
        weights = torch.zeros(N, device=self.device, dtype=torch.float64)
        if N == 1:
            weights[0] = 1.0
            return weights
        weights[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
        weights[0] = 0.5 * diffs[0]
        weights[-1] = 0.5 * diffs[-1]
        mass_distribution = weights / (torch.sum(weights) + 1e-12)
        return mass_distribution

    def _build_3d_ot_weights(self, key: str):
        """Build OT weights (alpha, beta) for a 3D point cloud.

        Using noise std-dev sigma_ij from data_noise_std:
        - Observation-side weights: beta_ij ∝ 1 / (sigma_ij^2 + eps) (larger sigma -> smaller weight)
        - Prediction-side weights: uniform alpha
        - Normalize alpha and beta so both sides sum to 1 (balanced OT mass constraint)
        """
        eps = 1e-8
        noise_std = self.data_noise_std.get(key, None)
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        n_pts = n_freq * n_stn
        freq_w = self._compute_frequency_weights()
        freq_w_expand = freq_w.view(-1, 1).expand(n_freq, n_stn).flatten()
        if noise_std is None:
            alpha = torch.full((1, n_pts), 1.0 / n_pts, device=self.device, dtype=torch.float64)
            alpha = (alpha * freq_w_expand) / (alpha * freq_w_expand).sum().clamp(min=eps)
            beta = alpha.clone()
            return alpha, beta

        w_obs = 1.0 / (noise_std.reshape(-1) ** 2 + eps)
        w_obs = torch.nan_to_num(w_obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(w_obs <= 0):
            w_obs = torch.ones_like(w_obs)
        w_obs = w_obs / (w_obs.sum() + eps)
        alpha = torch.full_like(w_obs, 1.0 / n_pts)
        alpha = (alpha * freq_w_expand) / (alpha * freq_w_expand).sum().clamp(min=eps)
        w_obs = (w_obs * freq_w_expand) / (w_obs * freq_w_expand).sum().clamp(min=eps)
        return alpha.unsqueeze(0), w_obs.unsqueeze(0)

    def _build_6d_ot_weights(self):
        """Build OT weights (alpha, beta) for a 6D point cloud.

        For each (freq, station), compute component-wise precision 1/sigma_k^2 and
        sum them into a total precision.
        beta_ij ∝ precision_ij; after normalization, alpha is uniform, satisfying
        the balanced-OT mass constraint.
        """
        eps = 1e-8
        noise_rhoxy = self.data_noise_std.get("rhoxy", None)
        noise_phsxy = self.data_noise_std.get("phsxy", None)
        noise_rhoyx = self.data_noise_std.get("rhoyx", None)
        noise_phsyx = self.data_noise_std.get("phsyx", None)
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        precision = torch.zeros((n_freq, n_stn), device=self.device, dtype=torch.float64)

        def add_precision(noise_std_tensor):
            if noise_std_tensor is None:
                return
            inv_var = 1.0 / (noise_std_tensor ** 2 + eps)
            precision.add_(inv_var)

        add_precision(noise_rhoxy)
        add_precision(noise_phsxy)
        add_precision(noise_rhoyx)
        add_precision(noise_phsyx)

        w_obs = precision.reshape(-1)
        w_obs = torch.nan_to_num(w_obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(w_obs <= 0):
            w_obs = torch.ones_like(w_obs)
        w_obs = w_obs / (w_obs.sum() + eps)
        n_pts = w_obs.numel()
        alpha = torch.full_like(w_obs, 1.0 / n_pts)
        # Multiply OT weights by Delta log(freq) frequency weights, then renormalize
        freq_w = self._compute_frequency_weights()
        freq_w_expand = freq_w.view(-1, 1).expand(n_freq, n_stn).flatten()
        alpha = (alpha * freq_w_expand) / (alpha * freq_w_expand).sum().clamp(min=eps)
        w_obs = (w_obs * freq_w_expand) / (w_obs * freq_w_expand).sum().clamp(min=eps)
        return alpha.unsqueeze(0), w_obs.unsqueeze(0)

    # ----- Potential OT improvements (see readme/docs) -----
    # e.g., unbalanced OT (reach>0), multiscale backend, p=1 for robustness,
    # annealing blur, staged OT+MSE, etc.

    def _init_sinkhorn(self, p: int, blur: float, scaling: float, reach: float, backend: str):
        """
        Initialize the Sinkhorn OT loss (fully controlled via external parameters).
        """
        try:
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
                f"p={p}, blur={blur}, scale={scaling}, reach={reach}, backend={backend}"
            )
        except Exception as e:
            print(f"[Warning] Sinkhorn init failed: {e}")
            self.sinkhorn_loss = None

    def _compute_data_weights(self, noise_floor=0.01, error_floor=1e-3):
        """
        Compute data weights (W_d).
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
            
        print(f"Computing data weights (Target Noise: {noise_floor*100:.1f}%)")
        print(f"  - Resistivity Error Floor: {noise_floor*100:.1f}%")
        print(f"  - Phase Error Floor:       {phase_error_deg:.3f} deg")
        
        for key, data in self.obs_data.items():
            # Default weight is 1.0; if noise std-dev is available, use 1/sigma for weighted chi^2
            if 'rho' in key.lower():
                # Std-dev corresponds to log10(rho) in calculate_data_errors_2d
                sigma_log = self.data_noise_std.get(key, None)
                if sigma_log is not None:
                    sigma_clamped = torch.clamp(sigma_log, min=error_floor)
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
                    sigma_clamped = torch.clamp(sigma_eff, min=error_floor)
                    w_tensor = 1.0 / sigma_clamped
                else:
                    # Fallback: constant phase error
                    sigma = phase_error_deg
                    w_tensor = torch.full_like(data, 1.0/(sigma + 1e-8), device=self.device)

            else:
                w_tensor = torch.ones_like(data, device=self.device)

            self.data_weights[key] = w_tensor
    
    def initialize_model(self, initial_sigma: float = 1e-2, random_init: bool = False,
                        sigma_min: float = 1e-3, sigma_max: float = 1,
                        init_type: str = "uniform",
                        offset_y_km: tuple = (-10, -5), offset_z_km: tuple = (10, 20),
                        offset_rho: float = 1000.0):
        """
        Initialize the inversion model.
        
        Args:
            initial_sigma: Initial conductivity for uniform background (S/m).
            random_init: Whether to use random initialization (log-uniform + Gaussian smoothing).
            sigma_min: Lower bound for random initialization (S/m).
            sigma_max: Upper bound for random initialization (S/m).
            init_type: "uniform" | "random" | "offset". offset=在错误位置放置块体。
            offset_y_km: (y_min, y_max) km for offset block, when init_type="offset".
            offset_z_km: (z_min, z_max) km depth for offset block, when init_type="offset".
            offset_rho: Resistivity (Ω·m) of offset block when init_type="offset".
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
        if nz_earth <= 0:
            raise RuntimeError(
                f"地下层数 <= 0：nz_model={nz_model}, nza={self.nza}。请先 set_forward_operator(nza=...) 并确保 nza 合理。"
            )
        
        # If self.nz and self.ny are defined, check consistency
        if hasattr(self, 'nz') and hasattr(self, 'ny'):
            if nz_model != self.nz - 1 or ny_model != self.ny - 1:
                print(f"Warning: model size ({nz_model}, {ny_model}) != expected ({self.nz-1}, {self.ny-1})")
        
        if init_type == "offset":
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
            # ------------------------------------------------------------------
            # [CORRECTION 1] Log-uniform sampling
            # Standard practice: sample in log10 space so each order of magnitude is equally likely
            # ------------------------------------------------------------------
            log_min = np.log10(sigma_min)
            log_max = np.log10(sigma_max)
            
            # Random samples in log10 space
            random_exponents = torch.rand((nz_earth, ny_model), device=self.device, dtype=torch.float64)
            random_exponents = log_min + (log_max - log_min) * random_exponents
            
            # Convert back to linear conductivity
            sigma_init = 10 ** random_exponents
    
            # ------------------------------------------------------------------
            # [CORRECTION 2] Spatial smoothing (Gaussian filter)
            # Standard practice: use correlated noise rather than pure white noise
            # This stabilizes gradients and mimics blocky geology
            # ------------------------------------------------------------------
            # Note: SciPy runs on CPU (or implement a torch conv2d alternative)
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
        if init_type == "offset":
            init_desc = f"Offset (block {offset_y_km} km × {offset_z_km} km, {offset_rho} Ω·m)"
        elif random_init or init_type == "random":
            init_desc = "Random (Log-Uniform + Smooth)"
        else:
            init_desc = "Uniform"
        print(f"✓ Model initialization complete: {init_desc}.")
        if self.nza > 0:
            print(f"  - Air layer fixed: nza={self.nza}, air_sigma={self.air_sigma_value:.2e} S/m")
    
    def set_reference_model(self, sig_ref: torch.Tensor):
        """
        Set a reference model (for regularization).
        
        Args:
            sig_ref: Reference conductivity model [nz-1, ny-1], consistent with the model grid
        """
        expected_full = (self.nz - 1, self.ny - 1)
        expected_earth = (expected_full[0] - self.nza, expected_full[1])

        # Accept either a full or earth-only reference model
        if tuple(sig_ref.shape) == expected_full:
            sig_ref_full = sig_ref
            sig_ref_earth = sig_ref[self.nza:, :]
        elif tuple(sig_ref.shape) == expected_earth:
            sig_ref_earth = sig_ref
            sig_ref_full = self._assemble_sigma_full(sig_ref_earth)
        else:
            raise ValueError(
                f"参考模型尺寸 {tuple(sig_ref.shape)} 不匹配：期望 earth={expected_earth} 或 full={expected_full}"
            )

        # Save full reference model for plotting/diagnostics
        self.sig_ref = sig_ref_full.to(self.device, dtype=torch.float64)
        # Constraints use earth-only parameters (must match self.model_log_sigma shape)
        self.model_log_sigma_ref = torch.log(sig_ref_earth.to(self.device, dtype=torch.float64))
        
        print("✓ Reference model set")
        print(
            f"   - (Earth) Conductivity range: {sig_ref_earth.min().item():.6e} - {sig_ref_earth.max().item():.6e} S/m"
        )
        print(
            f"   - (Earth) Resistivity range: {1/sig_ref_earth.max().item():.2f} - {1/sig_ref_earth.min().item():.2f} Ω·m"
        )
    
    def update_lambda_by_gradient_balance(
        self,
        loss_data: torch.Tensor,
        loss_model: torch.Tensor,
        current_lambda: float,
        alpha: float = 0.5,
        lambda_min: float = 1e-8,
        lambda_max: float = 1e3,
        bl: float = 2.0,
        window_size: int = 5,
        min_ratio_for_update: float = 0.1
    ):
        """
        Adaptively update lambda based on gradient magnitudes (improved version).

        Improvements:
        1) Moving-average smoothing of gradient norms to avoid single-step noise
        2) Keep the exponential decrease mechanism (empirically effective)
        3) Safety guard to avoid overly aggressive decreases when ratio is tiny

        Constraint: lambda is only allowed to decrease (gradually relax regularization).
        Target (soft): ||∇Phi_d|| ≲ lambda ||∇Phi_m||

        Args:
            window_size: Moving-average window size
            min_ratio_for_update: Do not update lambda if ratio is below this threshold
        """

        # ----------------------------
        # 1) Current gradient norms
        # ----------------------------
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
        
        # ----------------------------
        # 2) Update histories
        # ----------------------------
        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)
        
        # Keep histories bounded (avoid unbounded memory growth)
        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        # ----------------------------
        # 3) Moving average (smooth gradient norms)
        # ----------------------------
        if len(self.grad_norm_d_history) >= window_size:
            # Moving average over the last window_size samples
            norm_d_smooth = np.mean(self.grad_norm_d_history[-window_size:])
            norm_m_smooth = np.mean(self.grad_norm_m_history[-window_size:])
        else:
            # If history is short, fall back to current values
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        # ----------------------------
        # 4) Ratio (using smoothed gradient norms)
        # ----------------------------
        # Use smoothed norms directly to avoid single-step noise.
        # Note: do not use historical ratio statistics; gradients typically decay rapidly.
        ratio = norm_d_smooth / (bl * current_lambda * norm_m_smooth + 1e-12)
        ratio = float(ratio)
        
        # Store ratio history (monitoring only)
        self.ratio_history.append(ratio)

        # ----------------------------
        # 5) Lambda can only decrease (exponential decrease)
        # ----------------------------
        if ratio < 1.0:
            # Need to relax regularization
            if ratio < min_ratio_for_update:
                # Ratio is too small; skip update to avoid overly aggressive decrease
                new_lambda = current_lambda
            else:
                # Exponential decrease (empirically effective)
                # Use the smoothed ratio to reduce noise sensitivity
                proposed_lambda = current_lambda * (ratio ** alpha)
                new_lambda = proposed_lambda
        else:
            # Not imbalanced yet; keep lambda
            new_lambda = current_lambda

        # ----------------------------
        # 6) Safety constraints
        # ----------------------------
        new_lambda = float(np.clip(new_lambda, lambda_min, lambda_max))
        
        # Ensure monotonic non-increase (extra safety)
        new_lambda = min(current_lambda, new_lambda)

        return new_lambda, norm_d_item, norm_m_item

    def run_inversion(self, 
                    n_epochs: int = 100, 
                    mode: str = "6dot",
                    progress_interval: int = 10,
                    current_lambda: float = 0.01,
                    use_adaptive_lambda: bool = True,
                    lr: float = 0.05,
                    bl: float = 2.0,
                    norm_type = "L2",
                    use_reference_model: bool = False,
                    reference_weight: float = 0.1,
                    update_interval: int = 10,   # Update interval
                    warmup_epochs: int = 10,
                    alpha: float = 0.5,        
                    use_ot_weights: bool = True,  # Whether OT uses noise-weighted (alpha, beta)
                    use_depth_weights: bool = True,  # Whether roughness uses depth weighting
                    rms_chi2_stop: float = 1.05
                    ):  
        """
        Run inversion.
        
        Args:
            n_epochs: Number of epochs
            mode: Inversion mode ('3dot' / '6dot' / 'mse')
            progress_interval: Logging interval
            current_lambda: Initial regularization weight lambda
            use_ot_weights: If True, 3dot/6dot uses (alpha, beta) weights built from data_noise_std
            use_depth_weights: If True, roughness uses depth weighting (z/z0)^beta; otherwise uniform
        """
        if self.forward_operator is None:
            raise RuntimeError("请先设置正演算子")

        # Timing
        total_start_time = time.time()
        self.time_stats['start_time'] = total_start_time
        self.time_stats['epoch_times'] = []
        
        # Total number of data points
        num_data = sum(v.numel() for v in self.obs_data.values())

        optimizer = self.opt_config.create_optimizer(
            [self.model_log_sigma], lr=lr, optimizer_type="AdamW"
        )
        
        # Depth weights are computed for earth layers only (air is not inverted/regularized)
        zn_tensor = self.zn.clone().detach().to(self.device, dtype=torch.float64)
        z_centers_full = (zn_tensor[:-1] + zn_tensor[1:]) / 2.0
        z_centers_earth = z_centers_full[self.nza:]

        z_pos = torch.clamp(z_centers_earth, min=1.0)
        z0 = z_pos.min().clamp(min=1.0)
        beta = 0.3
        w_z = (z_pos / z0).pow(beta).unsqueeze(1).expand(-1, self.ny - 1)
        w_z = torch.clamp(w_z, max=500.0)
        w_z = w_z / w_z.max().clamp(min=1e-12)
        depth_weights = w_z.to(dtype=torch.float64) if use_depth_weights else None
       
        # AdamW optimization
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            
            # 1) Forward
            sigma_earth = torch.exp(self.model_log_sigma)
            sigma_full = self._assemble_sigma_full(sigma_earth)
            pred_dict = self.forward_operator(sigma_full)
            
            # 2) Data loss
            loss_data = torch.tensor(0.0, device=self.device)
            with torch.no_grad():
                rms_chi2 = self.compute_rms_chi2(pred_dict)
            
            if mode == '6dot':
                cloud_pred, cloud_obs = self._prepare_6d_ot_cloud(
                    pred_dict, self.obs_data
                )
                if use_ot_weights:
                    alpha_w, beta_w = self._build_6d_ot_weights()
                    loss_data = 600 * self.sinkhorn_loss(alpha_w, cloud_pred, beta_w, cloud_obs)
                else:
                    loss_data = 600 * self.sinkhorn_loss(cloud_pred, cloud_obs)

            elif mode == '3dot':
                for key in self.obs_data.keys():
                    obs = self.obs_data[key]
                    pred = pred_dict[key]
                    cloud_obs = self._prepare_3d_ot_cloud(obs, key)
                    cloud_pred = self._prepare_3d_ot_cloud(pred, key)
                    if use_ot_weights:
                        alpha_w, beta_w = self._build_3d_ot_weights(key)
                        loss_data += self.sinkhorn_loss(alpha_w, cloud_pred, beta_w, cloud_obs).sum()
                    else:
                        loss_data += self.sinkhorn_loss(cloud_pred, cloud_obs).sum()
                    loss_data = loss_data * 600

            elif mode == 'mse':
                for key in self.obs_data.keys():
                    if 'rho' in key.lower():
                        # Fixed: compute MSE in log10(rho) domain
                        obs_val = torch.log10(self.obs_data[key] + 1e-12)
                        p_val = torch.log10(pred_dict[key] + 1e-12)
                    else:
                        obs_val = self.obs_data[key]
                        p_val = pred_dict[key]

                    loss_data += torch.sum(
                        ((obs_val - p_val) * self.data_weights[key]) ** 2
                    )
                loss_data = loss_data / num_data
            
            else:
                raise ValueError(f"Unknown inversion mode: {mode}")
            # 3) Regularization term (supports reference-model constraint)
            if use_reference_model and self.model_log_sigma_ref is not None:
                loss_model = self.constraint_calc.calculate_combined_constraint(
                    model_log_sigma=self.model_log_sigma,
                    reference_model_log_sigma=self.model_log_sigma_ref,
                    roughness_weights=depth_weights,
                    roughness_norm=norm_type,
                    reference_norm=norm_type,
                    reference_weight=reference_weight
                )
            else:
                loss_model = self.constraint_calc.calculate_weighted_roughness(
                    self.model_log_sigma, depth_weights, norm_type
                )            
    
            # 4) Backprop
            if use_adaptive_lambda:
                # 1) Compute a proposed value every epoch
                proposed_lambda, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                loss_data, loss_model, current_lambda, 
                alpha=alpha, 
                lambda_min=1e-8,
                bl=bl
            )
                # 2) Decide whether to apply the update (using the passed-in schedule)
                is_warmup = epoch < warmup_epochs
                is_update_tick = (epoch - warmup_epochs) % update_interval == 0
                
                if not is_warmup and is_update_tick:
                # Only consider updates after warmup and on scheduled ticks
                    if abs(proposed_lambda - current_lambda) / current_lambda > 0.05:
                        print(f" [Auto-Lambda] Epoch {epoch}: Adjusted {current_lambda:.2e} -> {proposed_lambda:.2e}")
                        current_lambda = proposed_lambda
            else:
                # If adaptive lambda is off, only compute gradient norms for monitoring
                _, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                    loss_data, loss_model, current_lambda, bl=bl
                )
            total_loss = loss_data + current_lambda * loss_model
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([self.model_log_sigma], 1.0)
            optimizer.step()
            
            with torch.no_grad():
                self.model_log_sigma.clamp_(min=-11.5, max=4.6)
            
            # Record epoch runtime
            epoch_time = time.time() - epoch_start_time
            self.time_stats['epoch_times'].append(epoch_time)
            
            # Progress logging: print every progress_interval epochs
            if epoch % progress_interval == 0 or epoch == n_epochs - 1:
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
                print(f"  Misfit(RMS χ²): {rms_chi2:.3f} | Rough: {loss_model.item():.2e} | Lam: {current_lambda:.7f}")
                print(f"  GradNorms: |g_d|={g_d_norm:.3e} | |g_m|={g_m_norm:.3e}")
            
            # Store loss history
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'data_loss': loss_data.item(),
                'model_loss': loss_model.item(),
                'misfit': rms_chi2,
                'lambda': current_lambda,
                'epoch_time': epoch_time,
                'grad_data_norm': g_d_norm,
                'grad_model_norm': g_m_norm
            })

            if float(rms_chi2) < rms_chi2_stop:
                print(f"  [Early stop] RMS χ² = {float(rms_chi2):.3f} < {rms_chi2_stop}, epoch = {epoch}")
                break

        # End timing
        total_end_time = time.time()
        total_inversion_time = total_end_time - total_start_time
        
        # Update timing stats
        self.time_stats.update({
            'end_time': total_end_time,
            'total_inversion_time': total_inversion_time,
            'avg_epoch_time': np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'min_epoch_time': np.min(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'max_epoch_time': np.max(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'std_epoch_time': np.std(self.time_stats['epoch_times']) if len(self.time_stats['epoch_times']) > 1 else 0,
        })
        
        print("Inversion completed.")
        return self.get_sigma_full().detach()

    def plot_model_comparison(
            self,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],     # X-axis bounds (ignored when clip_to_stations=True)
            ylim: list = [50, 0],      # Y-axis bounds
            clip_to_stations: bool = False
        ):
        """
        Plot true model vs inverted model (log10 domain), masking the air layer and
        applying a view mask according to the given bounds.

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim as provided.
        """
        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = self.stations.cpu().numpy() / 1000.0
        xlim_use = [float(st_km.min()), float(st_km.max())] if clip_to_stations else xlim

        # -------- Model values --------
        sigma_true = self.sig_true.detach().cpu().numpy()
        sigma_inv = self.get_sigma_full().detach().cpu().numpy()
    
        try:
            data_range = np.log10(sigma_true.max()) - np.log10(sigma_true.min())
            score = ssim(np.log10(sigma_true), np.log10(sigma_inv), data_range=data_range, win_size=3)
            print(f"Model structural similarity (SSIM): {score:.4f}")
        except:
            pass
    
        eps = 1e-12
    
        model_true = np.log10(1.0 / (sigma_true + eps))
        model_inv = np.log10(1.0 / (sigma_inv + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_true = "True log10 Resistivity"
        title_inv = "Inverted log10 Resistivity"
    
        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        mask_air = ZZ < 0
        mask_ground = ZZ >= 0
    
        # -------- Build view mask (based on xlim_use/ylim) --------
        x_min, x_max = min(xlim_use), max(xlim_use)
        y_bottom, y_top = max(ylim), min(ylim)
        mask_view = (YY >= x_min) & (YY <= x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
        
        model_true_masked = np.ma.masked_where(~mask_view, model_true)
        model_inv_masked = np.ma.masked_where(~mask_view, model_inv)
    
        # Min/max inside the mask for colorbar scaling
        vmin = model_true_masked.min() - 0.5
        vmax = model_true_masked.max() + 0.5
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
        # --- True model ---
        im1 = ax1.pcolormesh(YY, ZZ, model_true_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax1.invert_yaxis()
        ax1.set_title(title_true)
        ax1.set_ylabel('Depth (km)')
        ax1.set_xlim(xlim_use)
        ax1.set_ylim(ylim)
        plt.colorbar(im1, ax=ax1, label=label)
    
        # --- Inverted model ---
        im2 = ax2.pcolormesh(YY, ZZ, model_inv_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax2.invert_yaxis()
        ax2.set_title(title_inv)
        ax2.set_ylabel('Depth (km)')
        ax2.set_xlabel('Distance (km)')
        ax2.set_xlim(xlim_use)
        ax2.set_ylim(ylim)
        plt.colorbar(im2, ax=ax2, label=label)
    
        # Mark station locations
        st_y = self.stations.cpu().numpy() / 1000.0
        ax2.scatter(st_y, np.zeros_like(st_y), c='k', s=10, marker='v', label='Stations')
    
        plt.tight_layout()
        plt.show()

    def plot_initial_model(
            self,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],
            ylim: list = [50, 0],
            clip_to_stations: bool = True
        ):
        """Plot the initial model (log10 domain), masking air and applying a view mask.

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim as provided.
        """
        if self.initial_model_sigma is None:
            print("Initial model is not saved; call initialize_model first")
            return

        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = self.stations.cpu().numpy() / 1000.0
        xlim_use = [float(st_km.min()), float(st_km.max())] if clip_to_stations else xlim

        # -------- Model values --------
        sigma_init = self.initial_model_sigma.detach().cpu().numpy()
        eps = 1e-12

        model_init = np.log10(1.0 / (sigma_init + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_init = "Initial log10 Resistivity"

        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])  # (nz,)
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        mask_air = ZZ < 0

        # -------- Build view mask (based on xlim_use/ylim) --------
        x_min, x_max = min(xlim_use), max(xlim_use)
        y_bottom, y_top = max(ylim), min(ylim)

        mask_view = (YY >= x_min) & (YY <= x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air

        model_init_masked = np.ma.masked_where(~mask_view, model_init)

        vmin = model_init_masked.min() - 0.5
        vmax = model_init_masked.max() + 0.5

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        im = ax.pcolormesh(YY, ZZ, model_init_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.invert_yaxis()
        ax.set_title(title_init)
        ax.set_ylabel('Depth (km)')
        ax.set_xlabel('Distance (km)')
        ax.set_xlim(xlim_use)
        ax.set_ylim(ylim)
        plt.colorbar(im, ax=ax, label=label)

        # Mark station locations
        st_y = self.stations.cpu().numpy() / 1000.0
        ax.scatter(st_y, np.zeros_like(st_y), c='k', s=10, marker='v', label='Stations')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_loss_history(self, target_misfit: float = 1.0):
        """
        Plot loss terms and parameter evolution during inversion.
        """
        # Extract series
        epochs = [log['epoch'] for log in self.loss_history]
        misfit = [log['misfit'] for log in self.loss_history]
        lambdas = [log['lambda'] for log in self.loss_history]
        data_loss = [log['data_loss'] for log in self.loss_history]
        model_loss = [log['model_loss'] for log in self.loss_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Panel 1: Misfit (RMS) ---
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='Current RMS')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_yscale('log')  # RMS often spans orders of magnitude
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].legend()

        # --- Panel 2: Data Loss vs Model Loss (Roughness) ---
        ax2_twin = axes[1].twinx()
        p1, = axes[1].plot(epochs, data_loss, 'c-', label='Data Loss')
        p2, = ax2_twin.plot(epochs, model_loss, 'm-', label='Model Roughness')
        
        axes[1].set_title("Loss Components Trade-off")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Data Loss", color='c')
        axes[1].set_yscale('log')
        ax2_twin.set_ylabel("Roughness (Model Loss)", color='m')
        
        # Merge legends
        axes[1].legend(handles=[p1, p2])
        axes[1].grid(True, alpha=0.3)

        # --- Panel 3: Lambda evolution ---
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization Parameter (Lambda)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lambda Value")
        axes[2].set_yscale('log')  # Lambda can span orders of magnitude
        axes[2].grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_gradient_history(self):
        """Plot gradient-norm histories for the data and model terms."""
        logs = self.loss_history
        if len(logs) > 1:
            logs = logs[1:]

        epochs = [log['epoch'] for log in logs]
        g_d = [log['grad_data_norm'] for log in logs]
        g_m = [log['grad_model_norm'] for log in logs]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, g_d, 'b-', label='||∇Φ_d|| (Data)', linewidth=2)
        plt.plot(epochs, g_m, 'r-', label='||∇Φ_m|| (Model)', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms of Data and Model Terms')
        plt.grid(True, which='both', ls='-', alpha=0.5)
        plt.legend()
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
            raise RuntimeError("请先 set_forward_operator 并 initialize_model")
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

    def plot_sensitivity(self, xlim=None, ylim=None, cmap: str = "viridis", clip_to_stations: bool = True):
        """Plot sensitivity matrix heatmap (per-cell sensitivity to observations).

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim when provided.
        """
        sens_2d, YY, ZZ = self.compute_sensitivity_matrix()
        eps = 1e-16
        sens_log = np.log10(sens_2d + eps)
        mask_air = ZZ < 0
        sens_masked = np.ma.masked_where(mask_air, sens_log)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.pcolormesh(YY, ZZ, sens_masked, cmap=cmap, shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Depth (km)")
        ax.set_title("Sensitivity Matrix (∂log pred / ∂log σ)")
        plt.colorbar(im, ax=ax, label=r"log$_{10}$ Sensitivity")
        if clip_to_stations:
            st_km = self.stations.cpu().numpy() / 1000.0
            ax.set_xlim([float(st_km.min()), float(st_km.max())])
        elif xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        st_y = self.stations.cpu().numpy() / 1000.0
        ax.scatter(st_y, np.zeros_like(st_y), c="k", s=10, marker="v", label="Stations")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    
    def plot_data_fitting(self, station_indices=None):
        """
        Plot data-fit comparisons for selected stations.
        """
        with torch.no_grad():
            sigma_full = self.get_sigma_full()
            pred_dict = self.forward_operator(sigma_full)
    
        freqs = self.freqs.cpu().numpy()
    
        n_stations = len(self.stations)
        if station_indices is None:
            station_indices = [0, n_stations // 2, n_stations - 1]
    
        n_plots = len(station_indices)
    
        fig, axes = plt.subplots(
            2, n_plots,
            figsize=(5 * n_plots, 8),
            sharex=True
        )
    
        if n_plots == 1:
            axes = axes.reshape(2, 1)
    
        rho_obs_all = []
    
        for i, st_idx in enumerate(station_indices):
            st_pos = self.stations[st_idx].item() / 1000.0
            ax_rho = axes[0, i]
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_rho = f"rho{mode}"
                if key_rho not in self.obs_data:
                    continue
    
                rho_obs = self.obs_data[key_rho][:, st_idx].cpu().numpy()
                rho_pred = pred_dict[key_rho][:, st_idx].cpu().numpy()
    
                rho_obs_all.append(rho_obs)
    
                # Asymmetric error bars (log-domain error)
                yerr = None
                if key_rho in self.data_noise_std:
                    sigma_log = self.data_noise_std[key_rho][:, st_idx].cpu().numpy()
                    rho_up = rho_obs * 10.0 ** sigma_log
                    rho_dn = rho_obs * 10.0 ** (-sigma_log)
                    yerr = [rho_obs - rho_dn, rho_up - rho_obs]
    
                if yerr is not None:
                    ax_rho.errorbar(
                        freqs, rho_obs, yerr=yerr,
                        fmt='o', ms=4, alpha=0.6,
                        color=color, ecolor=color,
                        elinewidth=1, capsize=2,
                        label=f"Obs {mode.upper()}"
                    )
                else:
                    ax_rho.plot(freqs, rho_obs, f'{color}o', ms=4, alpha=0.6)
    
                ax_rho.plot(
                    freqs, rho_pred,
                    f'{color}-', lw=1.5,
                    label=f"Pred {mode.upper()}"
                )
    
            ax_rho.set_xscale("log")
            ax_rho.set_yscale("log")
            ax_rho.set_title(f"Station {st_pos:.1f} km\nApp. Resistivity")
            ax_rho.set_xlabel("Frequency (Hz)")
            ax_rho.tick_params(labelbottom=True)  # Force frequency tick labels
            if i == 0:
                ax_rho.set_ylabel(r"$\rho_a$ ($\Omega\cdot$m)")
            ax_rho.grid(True, which="both", alpha=0.3)
            ax_rho.legend(fontsize="small")
    
            ax_phs = axes[1, i]
    
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_phs = f"phs{mode}"
                if key_phs not in self.obs_data:
                    continue
    
                phs_obs = self.obs_data[key_phs][:, st_idx].cpu().numpy()
                phs_pred = pred_dict[key_phs][:, st_idx].cpu().numpy()
    
                yerr = None
                if key_phs in self.data_noise_std:
                    yerr = (self.data_noise_std[key_phs][:, st_idx] * 90.0).cpu().numpy()
    
                if yerr is not None:
                    ax_phs.errorbar(
                        freqs, phs_obs, yerr=yerr,
                        fmt='o', ms=4, alpha=0.6,
                        color=color, ecolor=color,
                        elinewidth=1, capsize=2
                    )
                else:
                    ax_phs.plot(freqs, phs_obs, f'{color}o', ms=4, alpha=0.6)
    
                ax_phs.plot(freqs, phs_pred, f'{color}-', lw=1.5)
    
            ax_phs.set_xscale("log")
            ax_phs.set_ylim(0, 90)
            ax_phs.set_xlabel("Frequency (Hz)")
            if i == 0:
                ax_phs.set_ylabel("Phase (deg)")
            ax_phs.grid(True, which="both", alpha=0.3)
            ax_phs.set_title(f"Station {st_pos:.1f} km\nPhase")
    
        if len(rho_obs_all) > 0:
            rho_obs_all = np.concatenate(rho_obs_all)
            ymin = rho_obs_all.min() / 2.0
            ymax = rho_obs_all.max() * 2.0
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
        """
        # 1) Model values
        sigma_inv = self.get_sigma_full().detach().cpu().numpy()
        sigma_true = (self.sig_true.detach().cpu().numpy())

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
            all_vals.append(1.0 / (sigma_true[cell_mask, col_idx] + 1e-12))

        all_vals = np.hstack(all_vals)
        valid = all_vals[all_vals > 0]

        xmin = valid.min() / 2
        xmax = valid.max() * 2

        # 5) Plot
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
            xlabel = "Resistivity ($\Omega \cdot m$)"

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
            ax.set_xlabel(xlabel)

            st_y_km = self.stations[st_idx].item() / 1000.0
            ax.set_title(f"Profile {st_y_km:.1f} km")

            ax.grid(True, which='both', alpha=0.3)

            if i == 0:
                ax.set_ylabel("Depth (km)")

            if depth_limit_km is not None:
                ax.set_ylim([depth_limit_km, 0])

            ax.legend(fontsize='small')

        plt.tight_layout()
        plt.show()