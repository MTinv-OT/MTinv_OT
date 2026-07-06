import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path 
import time
from scipy.ndimage import gaussian_filter
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import subprocess
torch.set_default_dtype(torch.float64)

# Import sibling modules (relative imports)
from ..constraints import ConstraintCalculator
from ..optimizer import OptimizerConfig
from ..forward.solver import MT2DFD_Torch

def log_gpu_usage() -> float:
        """零依赖获取 GPU 算力占用率，直接调用底层 nvidia-smi"""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            return float(result.strip().split('\n')[0])
        except Exception:
            return 0.0

from .data import InversionDataMixin
from .ot import InversionOTMixin
from .regularization import InversionRegularizationMixin
from .metrics import InversionMetricsMixin

class MT2DInverter(
    InversionDataMixin,
    InversionOTMixin,
    InversionRegularizationMixin,
    InversionMetricsMixin,
):
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
        self.sig_ref = 0.01
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
                    warmup_epochs: int = 10,
                    alpha: float = 0.5,        
                    use_ot_weights: bool = True,  # 3dot only: noise-based marginal weights; 6dot ignores (uniform)
                    use_depth_weights: bool = True,  # Whether roughness uses depth weighting
                    depth_beta: float = 0.3,
                    rms_chi2_stop: float = 1.05,
                    monitor_ot_distance: bool = True,
                    ot_distance_interval: Optional[int] = None,
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
                    resume_from: Optional[str] = None,      # checkpoint路径
                    checkpoint_interval: Optional[int] = None,
                    checkpoint_dir: Optional[str] = "./checkpoints",
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

        # Optional display-only OT distance monitor (never used by optimization/early-stop).
        if ot_distance_interval is None:
            ot_interval_eff = max(int(progress_interval), 1)
        else:
            ot_interval_eff = max(int(ot_distance_interval), 1)

        cloud_obs_ot_monitor = None
        if monitor_ot_distance:
            if self.sinkhorn_loss is None:
                print("[OT-distance monitor] sinkhorn_loss is None; OT distance will be NaN.")
            else:
                cloud_obs_ot_monitor = self._prepare_6d_ot_cloud_obs(self.obs_data)

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
        if resume_from is not None:
            # 从checkpoint恢复
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model_log_sigma.data = checkpoint['model_state']
            
            # 恢复优化器
            optimizer = self.opt_config.create_optimizer(
                [self.model_log_sigma], 
                lr=lr,
                optimizer_type="AdamW",
               )
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # 恢复其他状态
            start_epoch = checkpoint['epoch'] + 1
            self.loss_history = checkpoint.get('loss_history', [])
            # === 利用你的思路：直接从历史记录中找回 Lambda ===
            if len(self.loss_history) > 0:
                # 注意：你需要确认一下你的字典里存 lambda 的键叫什么（比如 'lam', 'lambda_val' 等）
                # 这里假设键名是 'lam'
                last_lam = self.loss_history[-1].get('lambda', current_lambda)
                
                # 覆盖掉主程序传入的初始大 lambda
                current_lambda = last_lam
            
            # 恢复随机种子
            torch.set_rng_state(checkpoint['rng_state'].cpu().byte())
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu().byte())
        else:
            # 全新开始
            optimizer = self.opt_config.create_optimizer(
                [self.model_log_sigma], 
                lr=lr,
                optimizer_type="AdamW"
            )
            self.loss_history = []
            start_epoch=0
            
        # 2) 确保checkpoint目录存在
        if checkpoint_interval is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            # AdamW optimization
        for epoch in range(start_epoch, n_epochs):
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
            ot_distance_metric = float("nan")
            with torch.no_grad():
                rms_chi2 = self.compute_rms_chi2(pred_dict)
                if monitor_ot_distance and self.sinkhorn_loss is not None:
                    if cloud_obs_ot_monitor is None:
                        cloud_obs_ot_monitor = self._prepare_6d_ot_cloud_obs(self.obs_data)
                    cloud_pred_ot = _prepare_6d_cloud_pred_cached(pred_dict)
                    ot_distance_metric = float(self.sinkhorn_loss(cloud_pred_ot, cloud_obs_ot_monitor).item()) * data_loss_scale
            
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

            current_gpu_util = log_gpu_usage()
            
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
                if monitor_ot_distance and np.isfinite(ot_distance_metric):
                    print(f"  OT-distance(6D, view-only): {ot_distance_metric:.6e}")
                print(f"  GradNorms: |g_d|={g_d_norm:.3e} | |λ·g_m|={g_m_norm_scaled:.3e}")
            
            # Store loss history
            self.loss_history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'data_loss': loss_data.item(),
                'model_loss': loss_model.item(),
                'misfit': rms_chi2,
                'ot_distance': ot_distance_metric,
                'lambda': current_lambda,
                'epoch_time': epoch_time,
                # grad norms are only computed on selected epochs (update epoch and its
                # two preceding epochs); otherwise NaN.
                'grad_data_norm': float(g_d_norm) if np.isfinite(float(g_d_norm)) else float("nan"),
                # λ·||∇Φ_m|| (scaled), comparable to ||∇Φ_d|| for total gradient magnitude
                'grad_model_norm': g_m_norm_scaled,
                'gpu_util': current_gpu_util,  
            })
            if checkpoint_interval and (epoch % checkpoint_interval == 0 or epoch == n_epochs - 1):
                checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model_log_sigma.data,
                    'optimizer_state': optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'config': {
                        'n_epochs': n_epochs,
                        'lr': lr,
                        'mode': mode,
                        # ... 其他重要参数 ...
                    }
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
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
        self._last_run_config = {
        "n_epochs": n_epochs,
        "mode": mode,
        "progress_interval": progress_interval,
        "current_lambda": current_lambda,
        "use_adaptive_lambda": use_adaptive_lambda,
        "compute_lambda_grads_every_epoch": compute_lambda_grads_every_epoch,
        "lr": lr,
        "bl": bl,
        "norm_type": norm_type,
        "use_reference_model": use_reference_model,
        "reference_weight": reference_weight,
        "alpha_x": alpha_x,
        "alpha_z": alpha_z,
        "update_interval": update_interval,
        "warmup_epochs": warmup_epochs,
        "alpha": alpha,
        "use_ot_weights": use_ot_weights,
        "use_depth_weights": use_depth_weights,
        "depth_beta": depth_beta,
        "rms_chi2_stop": rms_chi2_stop,
        "monitor_ot_distance": monitor_ot_distance,
        "ot_distance_interval": ot_distance_interval,
        "profile_timing": profile_timing,
        "enable_blur_anneal": enable_blur_anneal,
        "blur_anneal_window": blur_anneal_window,
        "blur_anneal_rel_change_thresh": blur_anneal_rel_change_thresh,
        "blur_anneal_factor": blur_anneal_factor,
        "blur_anneal_min": blur_anneal_min,
        "blur_anneal_smooth_window": blur_anneal_smooth_window,
        "blur_anneal_cooldown_epochs": blur_anneal_cooldown_epochs,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_dir": checkpoint_dir
    }
        return self.get_sigma_full().detach()

    def _apply_plot_style(self):
        from ..plotting._style import apply_plot_style
        apply_plot_style()

    def plot_model_comparison(self, **kwargs):
        from ..plotting.inversion import plot_model_comparison as _plot
        return _plot(self, **kwargs)

    def plot_initial_model(self, **kwargs):
        from ..plotting.inversion import plot_initial_model as _plot
        return _plot(self, **kwargs)

    def plot_loss_history(self, **kwargs):
        from ..plotting.inversion import plot_loss_history as _plot
        return _plot(self, **kwargs)

    def plot_roughness_misfit_curve(self, **kwargs):
        from ..plotting.inversion import plot_roughness_misfit_curve as _plot
        return _plot(self, **kwargs)

    def plot_gradient_history(self, **kwargs):
        from ..plotting.inversion import plot_gradient_history as _plot
        return _plot(self, **kwargs)

    def plot_sensitivity(self, **kwargs):
        from ..plotting.inversion import plot_sensitivity as _plot
        return _plot(self, **kwargs)

    def plot_data_fitting(self, **kwargs):
        from ..plotting.inversion import plot_data_fitting as _plot
        return _plot(self, **kwargs)

    def plot_1d_profiles(self, **kwargs):
        from ..plotting.inversion import plot_1d_profiles as _plot
        return _plot(self, **kwargs)
