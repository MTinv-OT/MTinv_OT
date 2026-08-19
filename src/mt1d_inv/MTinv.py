import time
from datetime import datetime, timedelta
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
try:
    # First try relative import (when used as part of a package)
    from .constraints import ConstraintCalculator
    from .optimizer import OptimizerConfig
except ImportError:
    try:
        # Then try absolute import (from src)
        from src.constraints import ConstraintCalculator
        from src.optimizer import OptimizerConfig
    except ImportError:
        # Finally try a direct import (when run from the src directory)
        from constraints import ConstraintCalculator
        from optimizer import OptimizerConfig


class MT1DInverter:
    """
    MT 1D inversion class (fused: selective log parameterization + Occam constraint)
    """
    MU = 4e-7 * math.pi
    PI = math.pi

    def __init__(self, device: str = None, mu: float = None, 
                 use_sinkhorn: bool = True, sinkhorn_dim: int = 3,use_data_weighting: bool = True, constraint_type: str = "roughness"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = mu or self.MU
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_dim = sinkhorn_dim

        # Model parameters
        self.true_dz = None
        self.true_sig = None
        self.dz_inv = None
        self.log_sig_inv = None
        self.sig_inv = None

        # Data
        self.freq = None
        self.zxy_obs = None
        self.rho_obs = None
        self.phs_obs = None
        self.noise_level = None

        # Error estimates
        self.delta_rho = None
        self.delta_phs = None

        # Optimizer and constraint calculator
        self.optimizer_config = None
        self.optimizer_config = OptimizerConfig(device=self.device)
        self.constraint_calc = ConstraintCalculator(device=self.device)
        self.loss_history = []

        # Occam parameters
        self.use_occam_constraint = False
        self.occam_mu = 0.0001
        self.occam_target_misfit = 1.0
        self.constraint_type = constraint_type
        self.use_adaptive_regularization = True
        self.adaptation_factor = 1.0+1e-5

        self.data_misfit_history = []
        self.model_norm_history = []
        self.regularization_history = []
        self.chi2_history = []

        # Adaptive OT blur
        self.blur_init = 0.1
        self.blur_min = 0.002
        self.blur_decay = 0.93
        self.current_blur = self.blur_init

        # Style data weights
        self.rho_weights = None
        self.phs_weights = None
        self.use_data_weighting = use_data_weighting
        self.gradient_clip_value = 1.0  # gradient clipping parameter
        # Reference-model correction: pull the inversion toward a reference model
        self.reference_sig = None   # reference conductivity (n_layers,), matching inversion layers
        self.ref_weight = 0.0      # reference-model penalty weight; 0 means unused
        param_mode = "log parameter space"
        print(f"Using device: {self.device}")
        print(f"Parameterization: {param_mode}")
        print(f"Using {sinkhorn_dim}D Sinkhorn loss function" if use_sinkhorn else "Using MSE loss function")

    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MT 1D forward modeling"""
        nf = len(freq)
        zxy = torch.zeros(nf, dtype=torch.complex64, device=self.device)
        rho = torch.zeros(nf, dtype=torch.float32, device=self.device)
        phs = torch.zeros(nf, dtype=torch.float32, device=self.device)

        n_layers = sig.shape[0]

        for kf in range(nf):
            omega = 2.0 * self.PI * freq[kf]
            sqrt_arg = torch.complex(torch.tensor(0.0, device=self.device), -omega * self.mu) / sig[-1]
            Z = torch.sqrt(sqrt_arg)

            for m in range(n_layers-2, -1, -1):
                km_arg = torch.complex(torch.tensor(0.0, device=self.device), omega * self.mu * sig[m])
                km = torch.sqrt(km_arg)
                Z0 = -1j * omega * self.mu / km
                R = torch.exp(-2.0 * km * dz[m]) * (Z - Z0) / (Z + Z0)
                Z = Z0 * (1.0 + R) / (1. - R)

            zxy[kf] = Z
            rho[kf] = torch.abs(Z)**2 / (omega * self.mu)
            phs[kf] = torch.atan2(Z.imag, Z.real) * 180.0 / self.PI

        self.zxy = zxy
        self.rho = rho
        self.phs = phs
        return zxy, rho, phs

    def calculate_data_errors(self):
        """
        Compute apparent-resistivity and phase errors from impedance errors
        using the law of error propagation
        """
        # Impedance real/imaginary errors (known)
        sigma_Z_real = self.delta_zxy_real
        sigma_Z_imag = self.delta_zxy_imag
        
        Z = self.zxy_obs
        Z_abs = torch.abs(Z)
        
        # Apparent-resistivity error propagation
        # ρ_a = |Z|² / (ωμ) => σ_ρ ≈ 2ρ × (σ_Z/|Z|)
        omega = 2.0 * self.PI * self.freq
        rho_apparent = torch.abs(Z)**2 / (omega * self.MU)
        
        # Relative error: σ_ρ/ρ ≈ 2 × σ_Z/|Z|
        relative_error_rho = 2.0 * self.noise_level  # because σ_Z/|Z| = noise_level
        self.delta_rho = relative_error_rho * rho_apparent
        
        # Phase error propagation  
        # φ = atan2(Z_imag, Z_real) => σ_φ ≈ σ_Z/|Z| (radians)
        sigma_phi_rad = self.noise_level  # σ_Z/|Z| = noise_level
        self.delta_phs = torch.full_like(self.freq, sigma_phi_rad * 180.0 / self.PI)  # convert to degrees
        
        print(f"Apparent resistivity error range: {torch.min(self.delta_rho):.4f} - {torch.max(self.delta_rho):.4f} Ω·m")
        print(f"Phase error: {torch.mean(self.delta_phs):.2f}°")
        
        # Normalize weights (so the mean is 1)
        eps = 1e-10
        self.rho_weights = 1.0 / (self.delta_rho + eps)
        self.phs_weights = 1.0 / (self.delta_phs + eps)
        self.rho_weights = self.rho_weights / torch.mean(self.rho_weights)
        self.phs_weights = self.phs_weights / torch.mean(self.phs_weights)
        if self.use_data_weighting:
        # Compute weights from exact error propagation
            self.rho_weights = 1.0 / (self.delta_rho + 1e-10)
            self.phs_weights = 1.0 / (self.delta_phs + 1e-10)
            
            # Normalize weights
            self.rho_weights = self.rho_weights / torch.mean(self.rho_weights)
            self.phs_weights = self.phs_weights / torch.mean(self.phs_weights)
        else:
            self.rho_weights = None
            self.phs_weights = None
            return self.delta_rho, self.delta_phs

    def generate_synthetic_data(self, true_dz: torch.Tensor, true_sig: torch.Tensor,
                              freq_range: Tuple[float, float] = (-1, 4),
                              n_freq: int = 60,
                              noise_level: float = 0.05,
                              noise_type: str = "gaussian",
                              outlier_frac: float = 0.05,
                              outlier_strength: float = 4.0,
                              seed: Optional[int] = None) -> None:
        """
        Generate synthetic observations (add noise at the impedance level, then derive ρ/φ).

        Args:
            true_dz: true layer thicknesses (m)
            true_sig: true conductivity (S/m)
            freq_range: frequency range (log10 Hz)
            n_freq: number of frequencies
            noise_level: relative noise level (relative to |Z|)
            noise_type: "gaussian" Gaussian noise only; "nongaussian" overlays random outliers on Gaussian noise.
                Other common non-Gaussian options (extensible later): Laplace/double-exponential, Student-t heavy tails, uniform outliers, etc.
            outlier_frac: outlier fraction (0~1), used only when noise_type=="nongaussian"
            outlier_strength: outlier strength (multiple of the baseline delta), used only for nongaussian
            seed: random seed
        """
        if noise_type not in ("gaussian", "nongaussian"):
            raise ValueError(
                f'noise_type must be "gaussian" or "nongaussian", got "{noise_type}".'
                'Please check the spelling (e.g. nonguassin -> nongaussian).'
            )
        if noise_type == "nongaussian":
            if not (0 <= outlier_frac <= 1):
                raise ValueError(f"When noise_type='nongaussian', outlier_frac must be in [0, 1], got {outlier_frac}")
            if outlier_strength <= 0:
                raise ValueError(f"When noise_type='nongaussian', outlier_strength must be > 0, got {outlier_strength}")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.true_dz = true_dz.to(self.device)
        self.true_sig = true_sig.to(self.device)
        self.noise_level = noise_level

        self.freq = torch.logspace(freq_range[0], freq_range[1], n_freq, dtype=torch.float32, device=self.device)

        self.true_zxy, self.true_rho, self.true_phs = self.mt1d_forward(self.freq, self.true_dz, true_sig)

        mod_zxy_true = torch.abs(self.true_zxy)
        self.delta_zxy_real = noise_level * mod_zxy_true
        self.delta_zxy_imag = noise_level * mod_zxy_true

        noise_real = torch.randn_like(self.true_zxy.real) * self.delta_zxy_real
        noise_imag = torch.randn_like(self.true_zxy.imag) * self.delta_zxy_imag
        self.zxy_obs = torch.complex(self.true_zxy.real + noise_real, self.true_zxy.imag + noise_imag)

        # Non-Gaussian: add outlier noise on a subset of points (simulate wild values)
        if noise_type == "nongaussian":
            n_tot = self.zxy_obs.numel()
            n_out = max(1, int(round(outlier_frac * n_tot)))
            idx = torch.randperm(n_tot, device=self.device)[:n_out]
            mask = torch.zeros(n_tot, dtype=torch.bool, device=self.device)
            mask[idx] = True
            mask = mask.reshape(self.zxy_obs.shape)
            out_real = torch.randn_like(self.true_zxy.real) * (outlier_strength * self.delta_zxy_real)
            out_imag = torch.randn_like(self.true_zxy.imag) * (outlier_strength * self.delta_zxy_imag)
            self.zxy_obs = torch.complex(
                self.zxy_obs.real + torch.where(mask, out_real, torch.zeros_like(self.zxy_obs.real)),
                self.zxy_obs.imag + torch.where(mask, out_imag, torch.zeros_like(self.zxy_obs.imag))
            )

        omega = 2.0 * self.PI * self.freq
        self.rho_obs = torch.abs(self.zxy_obs)**2 / (omega * self.mu)
        self.phs_obs = torch.atan2(self.zxy_obs.imag, self.zxy_obs.real) * 180.0 / self.PI

        # Compute apparent-resistivity and phase errors
        self.calculate_data_errors()

        # Save noise std in log domain (still based on baseline Gaussian level; outliers appear in residuals)
        eps = 1e-8
        self.rho_noise_std_log = 2 * max(eps, noise_level) / math.log(10)
        self.phs_noise_std_norm = max(eps, noise_level)

        print(f"Generated synthetic data with {noise_level*100}% noise ({noise_type})")
        print(f"  → Noise std in log10(rho): {self.rho_noise_std_log:.4f}")
        print(f"  → Noise std in normalized phase: {self.phs_noise_std_norm:.4f}")
        if noise_type == "nongaussian":
            print(f"  → Non-Gaussian: outlier_frac={outlier_frac}, outlier_strength={outlier_strength}")

    def initialize_model(self, n_layers: int, total_depth: float, initial_sig: float = 0.01,
                     thickness_mode: str = "equal",
                     increasing_exponent: float = 1.0) -> None:
        """
        Initialize the inversion model (supports different thickness-allocation strategies)
        """
        # Validate
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")

        n_dz = n_layers - 1  # number of thickness blocks
        device = self.device

        if thickness_mode == "equal":
            dz_value = total_depth / n_dz
            self.dz_inv = torch.full((n_dz,), dz_value, dtype=torch.float32,
                                    device=device, requires_grad=False)
            mode_name = "equal thickness"

        elif thickness_mode in ("increasing_linear", "increasing_geometric"):
            # Build a baseline weight sequence, then normalize and scale by total_depth
            if thickness_mode == "increasing_linear":
                # Linear or power-law growth: weights i^p (i from 1 to n_dz)
                p = float(increasing_exponent) if increasing_exponent > 0 else 1.0
                indices = np.arange(1, n_dz + 1, dtype=np.float64)
                weights = indices ** p
                mode_name = f"linear/power growth (exponent={p})"
            else:  # increasing_geometric
                # Geometric growth: weights = r^(i-1); choose r from the given exponent so the sum is reasonable.
                r = float(increasing_exponent) ** (1.0 / max(1, n_dz - 1))
                indices = np.arange(0, n_dz, dtype=np.float64)
                weights = r ** indices
                mode_name = f"geometric growth (approx r={r:.3f})"

            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                raise ValueError("Generated weights sum to 0; check the parameters")
            
            dz_np = (weights / weights_sum) * total_depth
            self.dz_inv = torch.tensor(dz_np, dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            raise ValueError(f"Unsupported thickness_mode: {thickness_mode}")
        
        # Always use log parameterization: both Sinkhorn and non-Sinkhorn optimize log(sig)
        self.log_sig_inv = torch.full(
            (n_layers,),
            torch.log(torch.tensor(initial_sig, dtype=torch.float32, device=self.device)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        if self.use_sinkhorn:
            print(f"Initialized log-conductivity parameters (Sinkhorn mode)")
        else:
            print(f"Initialized log-conductivity parameters (non-Sinkhorn mode)")
        print(f"Initial conductivity: {torch.exp(self.log_sig_inv).tolist()}")

        print(f"Using {mode_name} mode")
        cum_depth = np.cumsum(self.dz_inv.detach().cpu().numpy())
        print(f"Initialized model with {n_layers} layers")
        print(f"Cumulative depth: {cum_depth.tolist()}")

    
    def set_reference_model(self, ref_sig, weight: float = 0.01) -> None:
        """
        Set a reference-model correction: add weight * ||log10(sig) - log10(ref)||^2 to the loss
        so the inversion is pulled toward the reference model.
        ref_sig: reference conductivity, shape (n_layers,), must match n_layers from initialize_model;
                 may be a list, numpy array, or tensor.
        weight: reference-model penalty weight; 0 disables it. Typical values 0.001~0.1.
        """
        if weight <= 0:
            self.reference_sig = None
            self.ref_weight = 0.0
            return
        t = torch.as_tensor(ref_sig, dtype=torch.float32, device=self.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        self.reference_sig = t
        self.ref_weight = float(weight)
        n = t.numel()
        if self.log_sig_inv is not None and self.log_sig_inv.numel() != n:
            raise ValueError(f"Reference model has {n} layers but inversion has {self.log_sig_inv.numel()}; call initialize_model before set_reference_model")
        print(f"Reference-model correction enabled: ref_weight={self.ref_weight}, n_layers={n}")

    def setup_optimizer(self, lr: float = 0.01, reg_weight_sig: float = 0.0001, phs_weight: float = 0.5,
                    p: int = 2, scaling: float = 0.9,
                    reach: Optional[float] = None,
                    optimizer_type: str = "AdamW", weight_decay: float = 0.0,
                    betas: Tuple[float, float] = (0.9, 0.999),
                    eps: float = 1e-8, momentum: float = 0.9) -> None:
        """
        Set up the optimizer (via the optimizer-config module).
        Currently log-parameterized as log(σ); the gradient scale differs from raw σ: ∂L/∂(log σ)=σ·∂L/∂σ,
        so updates are slower at small conductivity. If the fit is insufficient within 100 epochs, increase lr (e.g. 0.01–0.02) or num_epochs.
        """
        self.reg_weight_sig = reg_weight_sig
        self.phs_weight = phs_weight
        self.p_norm = p

        # Ensure parameters are initialized (both Sinkhorn and MSE use log parameter log_sig_inv)
        if self.log_sig_inv is None:
            raise ValueError("log_sig_inv is not initialized; call initialize_model first")
        params = [self.log_sig_inv]

        # Create optimizer via the optimizer-config module
        self.optimizer = self.optimizer_config.create_optimizer(
            params=params,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            momentum=momentum
        )

        # Create loss functions
        if self.use_sinkhorn:
            self.current_blur = self.blur_init
            self.sinkhorn_loss = self.optimizer_config.create_sinkhorn_loss(
                p=p,
                blur=self.current_blur,
                scaling=scaling,
                reach=reach,
                debias=True,
                backend="tensorized"
            )
            print(f"Using {self.sinkhorn_dim}D Sinkhorn loss with p={p}, blur={self.current_blur}, reach={reach} ({'unbalanced' if reach else 'balanced'})")
        else:
            self.data_loss_fn = self.optimizer_config.create_data_loss(p=p)
            print(f"Using {'L1' if p == 1 else 'MSE'} loss (p={p} equivalent)")

        print(f"Optimizer setup: {optimizer_type} with lr={lr}, weight_decay={weight_decay}")
        print(f"Regularization: sig_reg={reg_weight_sig}, phs_weight={phs_weight}")

    def update_lambda_by_gradient_balance(
        self,
        loss_data: torch.Tensor,
        loss_model: torch.Tensor,
        current_lambda: float,
        alpha: float = 0.5,
        lambda_min: float = 1e-5,
        lambda_max: float = 1e3,
        dr: float = 2.0,
        window_size: int = 5,
        min_ratio_for_update: float = 0.1
    ):
        """
        Adaptively update lambda from the relative magnitude of data-term and model-term gradients
        (improved version, consistent with the 2D implementation).

        Improvements:
        1. Smooth gradient norms with a moving average to avoid misjudging from single-step noise
        2. Keep the exponential-decay mechanism (works well)
        3. Add a safety check so lambda does not drop too far when the ratio is very small

        Constraint: lambda is only allowed to decrease (gradually relax regularization)
        Target (soft constraint): ||∇Φ_d|| ≲ λ ||∇Φ_m||

        Note: in 1D, Occam regularization may be off (loss_model=0); lambda is then left unchanged.
        """

        eps = 1e-12
        params = self.log_sig_inv

        # ----------------------------
        # 1. Current gradient norms
        # ----------------------------
        grad_d = torch.autograd.grad(
            loss_data,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad_d is None:
            grad_d = torch.zeros_like(params)

        # Note: when Occam is off, loss_model is often the constant 0 and cannot be differentiated; handle that here.
        grad_m = None
        can_grad_model = (
            isinstance(loss_model, torch.Tensor)
            and loss_model.requires_grad
            and (loss_model.grad_fn is not None)
        )
        if can_grad_model:
            grad_m = torch.autograd.grad(
                loss_model,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
        if grad_m is None:
            grad_m = torch.zeros_like(params)

        norm_d_raw = torch.sqrt(torch.mean(grad_d ** 2))
        norm_m_raw = torch.sqrt(torch.mean(grad_m ** 2)) + eps

        norm_d_item = float(norm_d_raw.item())
        norm_m_item = float(norm_m_raw.item())

        # ----------------------------
        # 2. Update history
        # ----------------------------
        if not hasattr(self, "grad_norm_d_history"):
            self.grad_norm_d_history = []
        if not hasattr(self, "grad_norm_m_history"):
            self.grad_norm_m_history = []
        if not hasattr(self, "ratio_history"):
            self.ratio_history = []

        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)

        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        # ----------------------------
        # 3. Moving average (smooth gradient norms)
        # ----------------------------
        if len(self.grad_norm_d_history) >= window_size:
            norm_d_smooth = float(np.mean(self.grad_norm_d_history[-window_size:]))
            norm_m_smooth = float(np.mean(self.grad_norm_m_history[-window_size:]))
        else:
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        # ----------------------------
        # 4. Compute ratio (already smoothed)
        # ----------------------------
        ratio = norm_d_smooth / (dr * float(current_lambda) * norm_m_smooth + eps)
        ratio = float(ratio)
        self.ratio_history.append(ratio)

        # ----------------------------
        # 5. Only allow lambda to decrease (exponential decay)
        # ----------------------------
        if ratio < 1.0:
            if ratio < min_ratio_for_update:
                new_lambda = float(current_lambda)
            else:
                proposed_lambda = float(current_lambda) * (ratio ** float(alpha))
                new_lambda = float(proposed_lambda)
        else:
            new_lambda = float(current_lambda)

        # ----------------------------
        # 6. Safety constraints
        # ----------------------------
        new_lambda = float(np.clip(new_lambda, lambda_min, lambda_max))
        new_lambda = min(float(current_lambda), new_lambda)

        return new_lambda, norm_d_item, norm_m_item

    def _compute_rms_chi2_from_pred(self, rho_pred: torch.Tensor, phs_pred: torch.Tensor) -> float:
        """Compute total χ² RMS from the current-step predictions (rho_pred, phs_pred), without an extra forward run."""
        rho_obs_np = self.rho_obs.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        rho_pred_np = rho_pred.detach().cpu().numpy()
        phs_pred_np = phs_pred.detach().cpu().numpy()
        delta_rho_np = self.delta_rho.cpu().numpy()
        delta_phs_np = self.delta_phs.cpu().numpy()
        rho_chi2_rms = np.sqrt(np.mean(((rho_obs_np - rho_pred_np) / (delta_rho_np + 1e-12)) ** 2))
        phs_chi2_rms = np.sqrt(np.mean(((phs_obs_np - phs_pred_np) / (delta_phs_np + 1e-12)) ** 2))
        total_chi2_rms = np.sqrt(0.5 * rho_chi2_rms ** 2 + 0.5 * phs_chi2_rms ** 2)
        return float(total_chi2_rms)

    def calculate_chi2_rms(self) -> Dict[str, float]:
        """
        Compute RMS based on χ² statistics
        """
        with torch.no_grad():
            sig_raw = torch.exp(self.log_sig_inv)
            zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                self.freq, self.dz_inv, sig_raw)
        
        # Convert to numpy for computation
        rho_obs_np = self.rho_obs.cpu().numpy()
        rho_pred_np = rho_pred.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        phs_pred_np = phs_pred.cpu().numpy()
        delta_rho_np = self.delta_rho.cpu().numpy()
        delta_phs_np = self.delta_phs.cpu().numpy()
        
        results = {}
        
        # 1. Apparent-resistivity χ²
        rho_chi = (rho_obs_np - rho_pred_np) / delta_rho_np
        rho_chi_squared = rho_chi**2
        rho_chi2_rms = np.sqrt(np.mean(rho_chi_squared))
        results['rho_chi2_rms'] = float(rho_chi2_rms)
        results['rho_chi2_mean'] = float(np.mean(rho_chi_squared))
        results['rho_chi2_max'] = float(np.max(rho_chi_squared))
        
        # 2. Phase χ²
        phs_chi = (phs_obs_np - phs_pred_np) / delta_phs_np
        phs_chi_squared = phs_chi**2
        phs_chi2_rms = np.sqrt(np.mean(phs_chi_squared))
        results['phs_chi2_rms'] = float(phs_chi2_rms)
        results['phs_chi2_mean'] = float(np.mean(phs_chi_squared))
        results['phs_chi2_max'] = float(np.max(phs_chi_squared))
        
        # 3. Total χ² RMS (equal weights)
        total_chi2_rms = np.sqrt(0.5 * rho_chi2_rms**2 + 0.5 * phs_chi2_rms**2)
        results['total_chi2_rms'] = float(total_chi2_rms)
        
        # 4. Traditional RMS (absolute error)
        rho_obs_log = np.log10(rho_obs_np)
        rho_pred_log = np.log10(rho_pred_np)
        results['rho_rms_log'] = float(np.sqrt(np.mean((rho_obs_log - rho_pred_log)**2)))
        results['phs_rms_deg'] = float(np.sqrt(np.mean((phs_obs_np - phs_pred_np)**2)))
        
        # 5. Relative error
        rho_relative_error = np.abs(rho_obs_np - rho_pred_np) / (rho_obs_np + 1e-10)
        results['rho_rms_relative'] = float(np.sqrt(np.mean(rho_relative_error**2)))
        
        phs_relative_error = np.abs(phs_obs_np - phs_pred_np) / 90.0
        results['phs_rms_relative'] = float(np.sqrt(np.mean(phs_relative_error**2)))
        
        # 6. Added: diagnostic info
        results['n_outliers_rho'] = int(np.sum(rho_chi_squared > 9))  # number of points with χ² > 9 (3σ)
        results['n_outliers_phs'] = int(np.sum(phs_chi_squared > 9))
        
        # Quality of error bars (whether δ is reasonable)
        results['error_scale_rho'] = float(np.sqrt(np.mean(rho_chi_squared)))  # should be ≈1
        results['error_scale_phs'] = float(np.sqrt(np.mean(phs_chi_squared)))  # should be ≈1
        
        return results
    
    # Params: between calculate_chi2_rms and run_inversion
    def _check_convergence(self, chi2_results: Dict[str, float], 
                         target_rms: float, 
                         tol: float = 1e-4) -> bool:
        """
        Internal convergence check: both an absolute RMS target and relative loss change
        """
        if target_rms is None: target_rms = 1.05
        # 1. Extract current metrics
        rho_rms = chi2_results['rho_chi2_rms']
        phs_rms = chi2_results['phs_chi2_rms']
        total_rms = chi2_results['total_chi2_rms']

        # 2. Absolute convergence (RMS target met)
        # Strategy: total RMS is on target, and neither component is extremely off (avoid one-sided poor fit)
        if total_rms < target_rms:
            if rho_rms < target_rms * 1.5 and phs_rms < target_rms * 1.5:
                print(f"✅ [STOP] Target RMS reached: Total={total_rms:.3f} (Rho={rho_rms:.3f}, Phs={phs_rms:.3f})")
                return True
        
        # 3. Relative convergence (loss stalled)
        # Check whether loss has barely changed over the last N iterations
        window = 20
        if len(self.loss_history) > window:
            prev_loss = self.loss_history[-window]['total_loss']
            curr_loss = self.loss_history[-1]['total_loss']
            rel_change = abs(prev_loss - curr_loss) / (prev_loss + 1e-10)
            
           
            if rel_change < tol and total_rms < 1.5:
                print(f" [STOP] Loss stalled: relative change {rel_change:.2e} < {tol} over {window} epochs")
                print(f"   Current RMS: Total={total_rms:.3f}")
                return True
                
        return False

    def run_inversion(self, 
                 num_epochs: int = 1000, 
                 print_interval: int = 20, 
                 seed: int = 42,
                 track_chi2: bool = True, 
                 enable_auto_stop: bool = True,
                 use_adaptive_lambda: bool = False,
                 current_lambda: float = 0.01,
                 warmup_epochs: int = 5,
                 update_interval: int = 1,
                 alpha: float = 0.5,
                 target_rms: float = 1.05) -> List[float]:
        """
        Run the inversion (supports adaptive regularization)
        
        Preconditions:
        - setup_constraints() must be called before this method
        - setup_optimizer() must be called before this method
        
        Args:
            num_epochs: number of iterations. With log parameterization, ≥200–500 is recommended; if the fit is slow, increase further or raise setup_optimizer lr.
            print_interval: print interval
            use_adaptive_lambda: whether to enable adaptive lambda updates
            current_lambda: initial regularization parameter
            warmup_epochs: number of warmup epochs
            update_interval: lambda update interval
            alpha: exponential decay factor for gradient balancing
            lambda_min: minimum lambda
        """
        
        # ===== Check required initialization =====
        if not hasattr(self, 'constraint_calc'):
            raise RuntimeError("Must call setup_constraints() first")
        
        if not hasattr(self, 'optimizer'):
            raise RuntimeError("Must call setup_optimizer() first")
        
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f"[Info] Random seed = {self.seed}")
        
        # ===== Initialize history =====
        self.loss_history = []
        self.data_misfit_history = []
        self.model_norm_history = []
        self.regularization_history = []
        self.time_history = []
        self.chi2_history = [] if track_chi2 else None
        self.lambda_history = [current_lambda] if use_adaptive_lambda else []
        self.grad_norm_d_history = []
        self.grad_norm_m_history = []
        self.ratio_history = [] if use_adaptive_lambda else []

        print(f"\nStarting inversion for {num_epochs} epochs...")
        print(f"Adaptive Lambda: {use_adaptive_lambda}")
        print(f"Occam Constraint: {self.use_occam_constraint}")
        print(f"Constraint Type: {self.constraint_type}")
        start_total = time.time()
        g_d, g_m = 0.0, 0.0
        for epoch in range(num_epochs):
            start_epoch = time.time()
            
            self.optimizer.zero_grad()
            
            # ===== Get conductivity (log parameterization: exp first, then forward and loss) =====
            sig_raw = torch.exp(self.log_sig_inv)
            
            # ===== Forward computation =====
            if self.use_sinkhorn:
                zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                    self.freq, self.dz_inv, sig_raw
                )

                # Data standardization
                if hasattr(self, 'rho_weights') and self.rho_weights is not None:
                    rho_pred_weighted = torch.log10(rho_pred) * self.rho_weights
                    rho_obs_weighted = torch.log10(self.rho_obs) * self.rho_weights
                    phs_pred_weighted = (phs_pred / 90.0) * self.phs_weights
                    phs_obs_weighted = (self.phs_obs / 90.0) * self.phs_weights
                else:
                    rho_pred_weighted = torch.log10(rho_pred)
                    rho_obs_weighted = torch.log10(self.rho_obs)
                    phs_pred_weighted = phs_pred / 90.0
                    phs_obs_weighted = self.phs_obs / 90.0

                # ===== Sinkhorn data term =====
                pred_points = torch.stack([
                    rho_pred_weighted,
                    phs_pred_weighted,
                    torch.log10(self.freq)
                ], dim=1)
                obs_points = torch.stack([
                    rho_obs_weighted,
                    phs_obs_weighted,
                    torch.log10(self.freq)
                ], dim=1)
                loss_data = self.sinkhorn_loss(pred_points, obs_points)
            
                # ===== Occam constraint (log space) =====
                loss_model = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model = self.constraint_calc.calculate_1d_model_norm(
                        model=model_for_occam,
                        constraint_type=self.constraint_type,
                        dz=self.dz_inv
                    )

            else:
                # ===== Non-Sinkhorn mode =====
                zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                    self.freq, self.dz_inv, sig_raw
                )

                rho_pred_log = torch.log10(rho_pred)
                rho_obs_log = torch.log10(self.rho_obs)
                residual_rho = rho_pred_log - rho_obs_log

                phs_pred_norm = phs_pred / 90.0
                phs_obs_norm = self.phs_obs / 90.0
                residual_phs = phs_pred_norm - phs_obs_norm

                if hasattr(self, 'rho_weights') and self.rho_weights is not None:
                    weighted_residual_rho = residual_rho * self.rho_weights
                    weighted_residual_phs = residual_phs * self.phs_weights
                else:
                    weighted_residual_rho = residual_rho / self.rho_noise_std_log
                    weighted_residual_phs = residual_phs / self.phs_noise_std_norm
                
                loss_rho = torch.mean(weighted_residual_rho**2)
                loss_phs = torch.mean(weighted_residual_phs**2)
                loss_data = loss_rho + self.phs_weight * loss_phs

                # ===== Constraint term =====
                loss_model = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model = self.constraint_calc.calculate_1d_model_norm(
                        model=model_for_occam,
                        constraint_type=self.constraint_type,
                        dz=self.dz_inv
                    )
            
            # ===== Compute gradient norms every epoch (for logging and plots); update lambda on interval when adaptive =====
            proposed_lambda, g_d, g_m = self.update_lambda_by_gradient_balance(
                loss_data=loss_data,
                loss_model=loss_model,
                current_lambda=current_lambda,
                alpha=alpha,
            )
            if use_adaptive_lambda and epoch > warmup_epochs and epoch % update_interval == 0:
                current_lambda = proposed_lambda
                self.lambda_history.append(current_lambda)
            
            # ===== Total loss =====
            total_loss = loss_data + current_lambda * loss_model
            
            # ===== Reference-model correction =====
            if self.reference_sig is not None and self.ref_weight > 0:
                ref = self.reference_sig.to(self.device).detach().clamp(min=1e-6)
                log10_ref = torch.log10(ref)
                log10_sig = self.log_sig_inv / math.log(10)
                loss_ref = self.ref_weight * torch.mean(
                    (log10_sig - log10_ref) ** 2
                )
                total_loss = total_loss + loss_ref
            
            # ===== Backprop and optimization =====
            total_loss.backward()
            
            # Gradient clipping
            self.optimizer_config.clip_gradients(
                [self.log_sig_inv], 
                max_norm=self.gradient_clip_value
            )
            
            self.optimizer.step()
            
            # Parameter bounds (log space: conductivity about 1e-4 ~ 10)
            self.optimizer_config.clamp_parameters(
                self.log_sig_inv, 
                min_val=-9.2, 
                max_val=2.3, 
                use_log_space=True
            )
            
            epoch_time = time.time() - start_epoch
            rms_chi2 = self._compute_rms_chi2_from_pred(rho_pred, phs_pred)
            # Per-epoch log: epoch as x-axis; chi-squared, lambda, and gradient norms for plotting
            self.loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss.item(),
                'data_loss': loss_data.item(),
                'model_loss': loss_model.item(),
                'misfit': rms_chi2,
                'lambda': current_lambda,
                'epoch_time': epoch_time,
                'grad_data_norm': g_d,
                'grad_model_norm': g_m,
            })
            self.data_misfit_history.append(loss_data.item())
            self.model_norm_history.append(loss_model.item())
            self.regularization_history.append((current_lambda * loss_model).item())
            self.time_history.append(epoch_time)

            # ===== Adaptive Sinkhorn blur decay =====
            if self.use_sinkhorn:
                prev_blur = self.current_blur
                self.current_blur = max(self.blur_min, self.current_blur * self.blur_decay)
                if abs(self.current_blur - prev_blur) > 1e-8:
                    self.sinkhorn_loss.blur = self.current_blur

            # ===== Periodically compute χ² RMS and check convergence (detailed chi2 is written with epoch in the print_interval block below) =====
            if track_chi2 and (epoch + 1) % print_interval == 0:
                chi2_results = self.calculate_chi2_rms()
                # Call the encapsulated convergence-check function
                if enable_auto_stop:
                    should_stop = self._check_convergence(
                        chi2_results, 
                        target_rms=target_rms, 
                        tol=1e-4
                    )
                    
                    if should_stop:
                        print(f"Inversion converged at epoch {epoch + 1}.")
                        break

            # --- 7. Print progress every print_interval epochs (format aligned with 2D) ---
            if (epoch + 1) % print_interval == 0:
                elapsed_sec = time.time() - start_total
                epoch_sec = time.time() - start_epoch
                percent = (epoch + 1) / num_epochs * 100
                avg_time = elapsed_sec / (epoch + 1)
                remaining_sec = avg_time * (num_epochs - epoch - 1)
                def _fmt(s):
                    m, s = divmod(int(s), 60)
                    h, m = divmod(m, 60)
                    return f"{h:d}:{m:02d}:{s:02d}"
                eta_str = (datetime.now() + timedelta(seconds=remaining_sec)).strftime("%H:%M:%S")
                elapsed_str = _fmt(elapsed_sec)
                remaining_str = _fmt(remaining_sec)
                avg_epoch_time = elapsed_sec / (epoch + 1)

                chi2_stats = self.calculate_chi2_rms()
                if track_chi2 and self.chi2_history is not None:
                    self.chi2_history.append({**chi2_stats, 'epoch': epoch + 1})
                total_rms = chi2_stats['total_chi2_rms']
                data_label = "Sinkhorn" if self.use_sinkhorn else "MSE"

                print(f"Epoch {epoch+1}/{num_epochs} [ {percent:5.1f}%]")
                print(f"  Elapsed: {elapsed_str} | Remaining: ~{remaining_str} | ETA: {eta_str}")
                print(f"  Epoch time: {epoch_sec:.2f}s | Average time: {avg_epoch_time:.2f}s")
                print(f"  Total: {total_loss.item():.4e} | Data({data_label}): {loss_data.item():.4e}")
                print(f"  Misfit(RMS χ²): {total_rms:.3f} | Rough: {loss_model.item():.2e} | Lam: {current_lambda:.7f}")
                print(f"  GradNorms: |g_d|={g_d:.3e} | |g_m|={g_m:.3e}")

                if enable_auto_stop and track_chi2:
                    should_stop = self._check_convergence(
                        chi2_stats,
                        target_rms=target_rms,
                        tol=1e-4
                    )
                    if should_stop:
                        print(f"\n>>> Converged at epoch {epoch + 1} <<<")
                        break

        # ===== Inversion finished =====
        total_time = time.time() - start_total
        avg_epoch_time = total_time / (epoch + 1) if epoch > 0 else total_time
        
        # Final χ² evaluation
        if track_chi2:
            final_chi2 = self.calculate_chi2_rms()
            print(f"\n=== Final χ² statistics ===")
            print(f"Apparent resistivity χ² RMS: {final_chi2['rho_chi2_rms']:.3f}")
            print(f"Phase χ² RMS: {final_chi2['phs_chi2_rms']:.3f}") 
            print(f"Total χ² RMS: {final_chi2['total_chi2_rms']:.3f}")
            
            total_chi2 = final_chi2['total_chi2_rms']
            if 0.8 <= total_chi2 <= 1.2:
                print("Excellent fit: χ² ≈ 1.0, the model fits the data within the error bars")
            elif total_chi2 > 1.5:
                print("Underfitting: χ² > 1.5, the model does not fit the data well enough")
            elif total_chi2 < 0.5:
                print("Overfitting: χ² < 0.5, the model may be fitting noise")
        
        print(f"\nInversion finished. Total time: {total_time:.2f}s ({total_time/60:.2f}min), "
            f"Average epoch time: {avg_epoch_time:.3f}s")
        
        final_sig = torch.exp(self.log_sig_inv).detach().cpu().numpy()

        print(f"\n=== Final results ===")
        print(f"True dz: {self.true_dz.cpu().numpy().tolist()}")
        print(f"True sig: {self.true_sig.cpu().numpy().tolist()}")
        print(f"Inverted sig: {final_sig.tolist()}")

        return self.loss_history
    # File: src/mt1d_inv/MTinv.py
    def setup_constraints(self, 
                     constraint_type: str = 'roughness',
                     use_occam_constraint: bool = True,
                     ref_weight: float = 0.0,
                     reference_sig: Optional[torch.Tensor] = None):
        """
        Set inversion constraints
        """
        self.use_occam_constraint = use_occam_constraint
        self.constraint_type = constraint_type
        self.ref_weight = ref_weight
        
        # Handle reference_sig
        if reference_sig is None:
            self.reference_sig = torch.exp(self.log_sig_inv).clone().detach() if self.log_sig_inv is not None else None
        else:
            self.reference_sig = reference_sig.to(self.device)
        
        print(f"[Constraint Setup]")
        print(f"  - Occam Constraint: {use_occam_constraint}")
        print(f"  - Constraint Type: {constraint_type}")
        print(f"  - Reference Model Weight: {ref_weight}")

    def calculate_sensitivity_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the fully normalized sensitivity matrix (Jacobian)
        
        Normalization: J_ij = ∂(log10(ρ_i)) / ∂(log10(σ_j))
        Physical meaning: how many orders of magnitude apparent resistivity changes when resistivity changes by 1 order of magnitude.
        
        Returns:
            J (np.ndarray): sensitivity matrix [n_freq, n_layer]
            z_grid (np.ndarray): depth-grid nodes (x-axis for plotting)
        """
        # Constant ln(10) for the chain-rule conversion
        LN_10 = math.log(10.0)
        
        # 1. Prepare parameters for which gradients are needed (log parameterization, always use log_sig_inv)
        sig_inv = self.log_sig_inv.detach().clone().requires_grad_(True)
        # 2. Forward: conductivity = exp(log_sig_inv)
        _, rho_pred, _ = self.mt1d_forward(self.freq, self.dz_inv, torch.exp(sig_inv))
        
        # 3. Differentiate frequency by frequency
        target = torch.log10(rho_pred)
        n_data = len(target)
        n_param = len(sig_inv)
        
        J = torch.zeros((n_data, n_param), device=self.device)
        
        for i in range(n_data):
            if i == n_data - 1:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=False)[0]
            else:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=True)[0]
            
            # --- Normalize: d(log10_rho)/d(log10_sig) = d(log10_rho)/d(log_sig) * ln(10) ---
            grad = grad * LN_10
            J[i, :] = grad
            
        return J.detach().cpu().numpy(), np.concatenate(([0], np.cumsum(self.dz_inv.detach().cpu().numpy())))
        
    # Other plotting methods remain unchanged...
    def plot_synthetic_data(self) -> None:
        """Plot synthetic data"""
        freq_np = self.freq.cpu().numpy()
        rho_np = self.rho.cpu().numpy()
        phs_np = self.phs.cpu().numpy()
        rho_obs_np = self.rho_obs.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0.1)
        
        ax1.loglog(freq_np, rho_np, 'b-', label='True Rho', linewidth=2)
        ax1.loglog(freq_np, rho_obs_np, 'r--', linewidth=2, 
                label=f'Noisy Rho ({self.noise_level*100:.0f}% Gaussian)', alpha=0.7)
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", linestyle='--', alpha=0.5)
        
        ax2.semilogx(freq_np, phs_np, 'b-', label='True Phs', linewidth=2)
        ax2.semilogx(freq_np, phs_obs_np, 'r--', linewidth=2, 
                    label=f'Noisy Phs ({self.noise_level*100:.0f}% of 90°)', alpha=0.7)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, which="both", linestyle='--', alpha=0.5)
        
        plt.suptitle(f'MT Synthetic Data with {self.noise_level*100:.0f}% Gaussian Noise', fontsize=14)
        plt.show()

    def plot_data_fit(self) -> plt.Figure:
        """Plot data fit"""
        with torch.no_grad():
            sig_raw = torch.exp(self.log_sig_inv)
            zxy_final_pred, rho_final_pred, phs_final_pred = self.mt1d_forward(
                self.freq, self.dz_inv, sig_raw)
        
        # Compute comprehensive RMS metrics
        rms_results = self.calculate_chi2_rms()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        freq_np = self.freq.cpu().numpy()
        rho_obs_np = self.rho_obs.cpu().numpy()
        rho_pred_np = rho_final_pred.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        phs_pred_np = phs_final_pred.cpu().numpy()
        
        # Apparent-resistivity subplot
        ax1.loglog(freq_np, rho_obs_np, 'ro', markersize=4, label='Observed', alpha=0.7)
        ax1.loglog(freq_np, rho_pred_np, 'b-', linewidth=2, label='Predicted')
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, which="both", linestyle='--', alpha=0.5)
        ax1.set_title(f'Apparent Resistivity Fit\nχ² RMS = {rms_results["rho_chi2_rms"]:.3f}', fontsize=13)
        
        # Phase subplot
        ax2.semilogx(freq_np, phs_obs_np, 'ro', markersize=4, label='Observed', alpha=0.7)
        ax2.semilogx(freq_np, phs_pred_np, 'b-', linewidth=2, label='Predicted')
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, which="both", linestyle='--', alpha=0.5)
        ax2.set_title(f'Phase Fit\nχ² RMS = {rms_results["phs_chi2_rms"]:.3f}', fontsize=13)
        
        plt.tight_layout()
        return fig

    def plot_loss_history(self, target_misfit: float = 1.0) -> plt.Figure:
        """
        Plot loss, chi-squared, and lambda evolution during inversion.
        The x-axis is the total iteration count (epoch).
        """
        if not self.loss_history:
            print("No loss history found; run run_inversion first.")
            return None
        # Backward compatible: a list of scalars has no misfit/lambda
        first = self.loss_history[0]
        if isinstance(first, (int, float)):
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(self.loss_history) + 1)
            ax.semilogy(epochs, self.loss_history, 'b-', linewidth=2, label='Total Loss')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, which="both", linestyle='--', alpha=0.5)
            ax.set_title('Inversion Loss History', fontsize=14)
            plt.tight_layout()
            return fig
        epochs = [log['epoch'] for log in self.loss_history]
        misfit = [log['misfit'] for log in self.loss_history]
        lambdas = [log['lambda'] for log in self.loss_history]
        data_loss = [log['data_loss'] for log in self.loss_history]
        model_loss = [log['model_loss'] for log in self.loss_history]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Panel 1: data-fit RMS (χ²)
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='RMS χ²')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit (χ² RMS)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_yscale('log')
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].legend()
        # Panel 2: Data Loss vs Model Loss
        ax2_twin = axes[1].twinx()
        p1, = axes[1].plot(epochs, data_loss, 'c-', label='Data Loss')
        p2, = ax2_twin.plot(epochs, model_loss, 'm-', label='Model (Roughness)')
        axes[1].set_title("Loss Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Data Loss", color='c')
        axes[1].set_yscale('log')
        ax2_twin.set_ylabel("Model Loss", color='m')
        axes[1].legend(handles=[p1, p2])
        axes[1].grid(True, alpha=0.3)
        # Panel 3: Lambda
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization (Lambda)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lambda")
        axes[2].set_yscale('log')
        axes[2].grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_gradient_history(self) -> plt.Figure:
        """Plot data-term and model-term gradient norms vs total iteration count."""
        if not self.loss_history:
            print("No loss history found; run run_inversion first.")
            return None
        first = self.loss_history[0]
        if isinstance(first, (int, float)) or 'grad_data_norm' not in first or 'grad_model_norm' not in first:
            print("Current loss_history has no gradient norms; re-run inversion with the latest run_inversion.")
            return None
        epochs = [log['epoch'] for log in self.loss_history]
        g_d = [log['grad_data_norm'] for log in self.loss_history]
        g_m = [log['grad_model_norm'] for log in self.loss_history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, g_d, 'b-', label='||∇Φ_d|| (Data)', linewidth=2)
        ax.plot(epochs, g_m, 'r-', label='||∇Φ_m|| (Model)', linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms of Data and Model Terms')
        ax.grid(True, which='both', ls='-', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_model_comparison(self) -> plt.Figure:
        """Plot model comparison"""
        inv_sig = torch.exp(self.log_sig_inv).detach().cpu().numpy()
        
        true_sig_np = self.true_sig.cpu().numpy()
        true_dz_np = self.true_dz.cpu().numpy()
        inv_dz_np = self.dz_inv.detach().cpu().numpy()
        
        def create_model_profile(dz, sig):
            """Create a depth–resistivity profile of the model"""
            if len(dz) == 0 or len(sig) == 0:
                return [], []
                
            depths = [0.0]
            resistivities = [1.0 / sig[0]]
            
            current_depth = 0.0
            for i in range(len(dz)):
                current_depth += dz[i]
                depths.extend([current_depth, current_depth])
                
                if i < len(sig) - 1:
                    resistivities.extend([1.0 / sig[i], 1.0 / sig[i+1]])
                else:
                    resistivities.extend([1.0 / sig[i], 1.0 / sig[i]])
            
            extension_depth = current_depth * 1.5
            depths.append(extension_depth)
            resistivities.append(resistivities[-1])
            
            return depths, resistivities
        
        true_depths, true_resistivities = create_model_profile(true_dz_np, true_sig_np)
        inv_depths, inv_resistivities = create_model_profile(inv_dz_np, inv_sig)
        
        plt.figure(figsize=(6, 8))
        plt.step(true_resistivities, true_depths, where='post',
                label='True Model', color='red', linewidth=2)
        plt.step(inv_resistivities, inv_depths, where='post',
                label='Inverted Model', color='blue', linewidth=2)
        
        plt.xscale('log')
        plt.xlabel("Resistivity ($\Omega \cdot$m)", fontsize=20)
        plt.ylabel("Depth (m)", fontsize=20)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc='upper right', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.gca().invert_yaxis()
        
        # Drop the main title; do not call plt.title()
        
        plt.tight_layout()
        return plt.gcf()


    def plot_chi2_history(self) -> plt.Figure:
        """Plot χ² history; the x-axis is the total iteration count."""
        if not self.chi2_history:
            print("Warning: No χ² history available")
            return None
        # If entries are dicts with epoch (sampled at print_interval)
        first = self.chi2_history[0]
        if isinstance(first, dict) and 'epoch' in first:
            epochs = [h['epoch'] for h in self.chi2_history]
        else:
            epochs = range(1, len(self.chi2_history) + 1)
        rho_chi2 = [h['rho_chi2_rms'] for h in self.chi2_history]
        phs_chi2 = [h['phs_chi2_rms'] for h in self.chi2_history]
        total_chi2 = [h['total_chi2_rms'] for h in self.chi2_history]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, rho_chi2, 'r-', linewidth=2, label='Resistivity χ² RMS')
        ax.plot(epochs, phs_chi2, 'g-', linewidth=2, label='Phase χ² RMS')
        ax.plot(epochs, total_chi2, 'b-', linewidth=3, label='Total χ² RMS')
        
        # Ideal-fit line
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal Fit (χ²=1)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('χ² RMS', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.set_title('χ² RMS History', fontsize=14)
        
        plt.tight_layout()
        return fig

    def plot_sensitivity(self) -> None:
        """Plot a sensitivity-matrix heatmap (show once; do not return fig, to avoid duplicate Jupyter output)"""
        J, z_grid = self.calculate_sensitivity_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        # X-axis: frequency index (high frequency to low frequency)
        # Y-axis: layer depth
        
        # For plotting convenience, typically transpose: rows are depth, columns are frequency
        im = ax.imshow(J.T, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest', origin='upper',
                       extent=[np.log10(self.freq[-1].item()), np.log10(self.freq[0].item()), z_grid[-1], 0])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Sensitivity $\partial \log \\rho / \partial \log \sigma$")
        
        ax.set_xlabel("Log10 Frequency (Hz)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Sensitivity Matrix (Jacobian)")
        
        plt.tight_layout()
        plt.show()
