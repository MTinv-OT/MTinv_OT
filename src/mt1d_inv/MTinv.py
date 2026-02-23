import time
from datetime import datetime, timedelta
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
try:
    # First try relative import (when used as package)
    from .constraints import ConstraintCalculator
    from .optimizer import OptimizerConfig
except ImportError:
    try:
        # Then try absolute import (from src)
        from src.constraints import ConstraintCalculator
        from src.optimizer import OptimizerConfig
    except ImportError:
        # Finally try direct import (if running inside src)
        from constraints import ConstraintCalculator
        from optimizer import OptimizerConfig


class MT1DInverter:
    """
    MT 1D inverter class (log parameterization + Occam constraint).
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

        # Error estimation
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

        # OT blur自适应
        self.blur_init = 0.1
        self.blur_min = 0.002
        self.blur_decay = 0.93
        self.current_blur = self.blur_init

        # Data weighting
        self.rho_weights = None
        self.phs_weights = None
        self.use_data_weighting = use_data_weighting
        self.gradient_clip_value = 1.0  # Gradient clipping
        # Reference model: pull inversion toward reference
        self.reference_sig = None   # Reference conductivity (n_layers)
        self.ref_weight = 0.0      # Reference model penalty weight, 0 = disabled
        param_mode = "对数参数空间"
        print(f"Using device: {self.device}")
        print(f"Parameterization: {param_mode}")
        print(f"Using {sinkhorn_dim}D Sinkhorn loss function" if use_sinkhorn else "Using MSE loss function")

    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MT 1D forward modelling"""
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
        Compute apparent resistivity and phase errors from impedance errors.
        Uses error propagation.
        """
        # Impedance real/imag error (known)
        sigma_Z_real = self.delta_zxy_real
        sigma_Z_imag = self.delta_zxy_imag
        
        Z = self.zxy_obs
        Z_abs = torch.abs(Z)
        
        # Apparent resistivity error propagation
        # ρ_a = |Z|² / (ωμ) => σ_ρ ≈ 2ρ × (σ_Z/|Z|)
        omega = 2.0 * self.PI * self.freq
        rho_apparent = torch.abs(Z)**2 / (omega * self.MU)
        
        # Relative error: σ_ρ/ρ ≈ 2 × σ_Z/|Z|
        relative_error_rho = 2.0 * self.noise_level  # σ_Z/|Z| = noise_level
        self.delta_rho = relative_error_rho * rho_apparent
        
        # Phase error propagation
        # φ = atan2(Z_imag, Z_real) => σ_φ ≈ σ_Z/|Z| (radians)
        sigma_phi_rad = self.noise_level  # σ_Z/|Z| = noise_level
        self.delta_phs = torch.full_like(self.freq, sigma_phi_rad * 180.0 / self.PI)  # Convert to degrees
        
        print(f"视电阻率误差范围: {torch.min(self.delta_rho):.4f} - {torch.max(self.delta_rho):.4f} Ω·m")
        print(f"相位误差: {torch.mean(self.delta_phs):.2f}°")
        
        # Normalize weights (mean=1)
        eps = 1e-10
        self.rho_weights = 1.0 / (self.delta_rho + eps)
        self.phs_weights = 1.0 / (self.delta_phs + eps)
        self.rho_weights = self.rho_weights / torch.mean(self.rho_weights)
        self.phs_weights = self.phs_weights / torch.mean(self.phs_weights)
        if self.use_data_weighting:
        # Use exact error propagation for weights
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
        Generate synthetic observed data (impedance with noise, then export ρ/φ).

        Args:
            true_dz: True layer thickness (m)
            true_sig: True conductivity (S/m)
            freq_range: Frequency range (log10 Hz)
            n_freq: Number of frequencies
            noise_level: Relative noise level (relative to |Z|)
            noise_type: "gaussian" for Gaussian only; "nongaussian" adds random outliers.
            outlier_frac: Outlier fraction (0~1), only when noise_type=="nongaussian"
            outlier_strength: Outlier strength (multiple of baseline delta)
            seed: Random seed
        """
        if noise_type not in ("gaussian", "nongaussian"):
            raise ValueError(
                f'noise_type must be "gaussian" or "nongaussian", got "{noise_type}".'
                'Check spelling (e.g. nonguassin -> nongaussian).'
            )
        if noise_type == "nongaussian":
            if not (0 <= outlier_frac <= 1):
                raise ValueError(f"noise_type='nongaussian' requires outlier_frac in [0, 1], got {outlier_frac}")
            if outlier_strength <= 0:
                raise ValueError(f"noise_type='nongaussian' requires outlier_strength > 0, got {outlier_strength}")
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

        # Non-Gaussian: add outlier noise on some points
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

        # Compute apparent resistivity and phase errors
        self.calculate_data_errors()

        # Save noise std in log domain (baseline Gaussian level; outliers appear in residuals)
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
        Initialize inversion model (supports different thickness allocation).
        """
        # Validation
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")

        n_dz = n_layers - 1  # Thickness blocks
        device = self.device

        if thickness_mode == "equal":
            dz_value = total_depth / n_dz
            self.dz_inv = torch.full((n_dz,), dz_value, dtype=torch.float32,
                                    device=device, requires_grad=False)
            mode_name = "Equal thickness"

        elif thickness_mode in ("increasing_linear", "increasing_geometric"):
            # Generate weight sequence, normalize and multiply by total_depth
            if thickness_mode == "increasing_linear":
                # Linear/power growth: weight i^p (i from 1 to n_dz)
                p = float(increasing_exponent) if increasing_exponent > 0 else 1.0
                indices = np.arange(1, n_dz + 1, dtype=np.float64)
                weights = indices ** p
                mode_name = f"Linear/power growth (exponent={p})"
            else:  # increasing_geometric
                # Geometric growth: weight = r^(i-1)
                r = float(increasing_exponent) ** (1.0 / max(1, n_dz - 1))
                indices = np.arange(0, n_dz, dtype=np.float64)
                weights = r ** indices
                mode_name = f"几何增长 (approx r={r:.3f})"

            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                raise ValueError("Generated weights sum to 0, check parameters")
            
            dz_np = (weights / weights_sum) * total_depth
            self.dz_inv = torch.tensor(dz_np, dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            raise ValueError(f"Unsupported thickness_mode: {thickness_mode}")
        
        # Unified log parameterization: both Sinkhorn and non-Sinkhorn optimize log(sig)
        self.log_sig_inv = torch.full(
            (n_layers,),
            torch.log(torch.tensor(initial_sig, dtype=torch.float32, device=self.device)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        if self.use_sinkhorn:
            print(f"Initialized log conductivity parameters (Sinkhorn mode)")
        else:
            print(f"Initialized log conductivity parameters (non-Sinkhorn mode)")
        print(f"初始电导率: {torch.exp(self.log_sig_inv).tolist()}")

        print(f"Using {mode_name} mode")
        cum_depth = np.cumsum(self.dz_inv.detach().cpu().numpy())
        print(f"Initialized model with {n_layers} layers")
        print(f"Cumulative depth: {cum_depth.tolist()}")

    
    def set_reference_model(self, ref_sig, weight: float = 0.01) -> None:
        """
        Set reference model: add weight * ||log10(sig) - log10(ref)||^2 to loss.
        ref_sig: Reference conductivity (n_layers); list/numpy/tensor.
        weight: Reference model penalty weight; 0 to disable. Typical 0.001~0.1.
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
            raise ValueError(f"Reference model layers {n} mismatch inversion layers {self.log_sig_inv.numel()}; call initialize_model before set_reference_model")
        print(f"Reference model enabled: ref_weight={self.ref_weight}, ref_layers={n}")

    def setup_optimizer(self, lr: float = 0.01, reg_weight_sig: float = 0.0001, phs_weight: float = 0.5,
                    p: int = 2, scaling: float = 0.9,
                    reach: Optional[float] = None,
                    optimizer_type: str = "AdamW", weight_decay: float = 0.0,
                    betas: Tuple[float, float] = (0.9, 0.999),
                    eps: float = 1e-8, momentum: float = 0.9) -> None:
        """
        Setup optimizer (using optimizer config module).
        Log parameterization log(σ): gradient scale differs from σ; small conductivity updates slower.
        If fit insufficient in 100 epochs, increase lr (e.g. 0.01~0.02) or num_epochs.
        """
        self.reg_weight_sig = reg_weight_sig
        self.phs_weight = phs_weight
        self.p_norm = p

        # Ensure params initialized (both Sinkhorn and MSE use log_sig_inv)
        if self.log_sig_inv is None:
            raise ValueError("log_sig_inv not initialized; call initialize_model first")
        params = [self.log_sig_inv]

        # Create optimizer via config module
        self.optimizer = self.optimizer_config.create_optimizer(
            params=params,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            momentum=momentum
        )

        # Create loss function
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
        lambda_min: float = 1e-8,
        lambda_max: float = 1e5,
        smoothing_window: int = 5,
        force_monotonic_decrease: bool = True  # Force monotonic decrease (Occam-style)
    ):
        """
        [Automatic gradient balance - monotonic decrease version]
        """
        
        # 1. Compute current gradient norms
        params = self.log_sig_inv
        
        grad_d = torch.autograd.grad(
            loss_data, params, retain_graph=True, create_graph=False, allow_unused=True
        )[0]
        if grad_d is None: grad_d = torch.zeros_like(params)
        norm_d_curr = torch.norm(grad_d).item()

        if isinstance(loss_model, torch.Tensor) and loss_model.item() != 0:
            grad_m = torch.autograd.grad(
                loss_model, params, retain_graph=True, create_graph=False, allow_unused=True
            )[0]
            if grad_m is None: grad_m = torch.zeros_like(params)
            norm_m_curr = torch.norm(grad_m).item()
        else:
            norm_m_curr = 0.0
            
        # 2. History smoothing
        if not hasattr(self, 'grad_norm_d_history'):
            self.grad_norm_d_history = []
            self.grad_norm_m_history = []
        
        self.grad_norm_d_history.append(norm_d_curr)
        self.grad_norm_m_history.append(norm_m_curr)
        
        if len(self.grad_norm_d_history) > smoothing_window:
            self.grad_norm_d_history.pop(0)
            self.grad_norm_m_history.pop(0)
            
        avg_norm_d = np.mean(self.grad_norm_d_history)
        avg_norm_m = np.mean(self.grad_norm_m_history)

        # 3. Auto-compute target Lambda
        if avg_norm_m < 1e-20:
            target_lambda = current_lambda
        else:
            # Target: balance the two gradient norms
            target_lambda = avg_norm_d / (avg_norm_m + 1e-20)

        # 4. Log-domain smooth update
        target_lambda = max(lambda_min, min(lambda_max, target_lambda))
        current_lambda = max(lambda_min, current_lambda)
        
        log_curr = math.log10(current_lambda)
        log_target = math.log10(target_lambda)
        
        log_new = (1.0 - alpha) * log_curr + alpha * log_target
        proposed_lambda = 10.0 ** log_new

        # ---------------------------------------------------------
        # 5. Force monotonic decrease logic
        # ---------------------------------------------------------
        if force_monotonic_decrease:
            # If new lambda > current, keep current (ignore increase)
            if proposed_lambda > current_lambda:
                new_lambda = current_lambda
            else:
                new_lambda = proposed_lambda
        else:
            # Allow bidirectional change
            new_lambda = proposed_lambda

        return new_lambda, norm_d_curr, norm_m_curr

    def _compute_rms_chi2_from_pred(self, rho_pred: torch.Tensor, phs_pred: torch.Tensor) -> float:
        """Compute total χ² RMS from current predictions (rho_pred, phs_pred); no extra forward."""

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
        Compute RMS based on χ² statistics (standard error assessment).
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
        
        # 1. Apparent resistivity χ²
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
        
        # 3. Total χ² RMS (equal weight)
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
        
        # 6. Diagnostics
        results['n_outliers_rho'] = int(np.sum(rho_chi_squared > 9))  # χ² > 9 (3σ) count
        results['n_outliers_phs'] = int(np.sum(phs_chi_squared > 9))
        
        # Error bar quality (δ reasonable)
        results['error_scale_rho'] = float(np.sqrt(np.mean(rho_chi_squared)))  # Should ≈1
        results['error_scale_phs'] = float(np.sqrt(np.mean(phs_chi_squared)))  # Should ≈1
        
        return results
    
    #Params: 放在 calculate_chi2_rms 和 run_inversion 之间
    def _check_convergence(self, chi2_results: Dict[str, float], 
                         target_rms: float, 
                         tol: float = 1e-4) -> bool:
        """
        Internal convergence check: absolute RMS target and relative loss change.
        """
        if target_rms is None: target_rms = 1.05
        # 1. Extract current metrics
        rho_rms = chi2_results['rho_chi2_rms']
        phs_rms = chi2_results['phs_chi2_rms']
        total_rms = chi2_results['total_chi2_rms']

        # 2. Absolute convergence (RMS target met)
        # Strategy: total RMS met, neither side too extreme
        if total_rms < target_rms:
            if rho_rms < target_rms * 1.5 and phs_rms < target_rms * 1.5:
                print(f"✅ [Stop] Target RMS reached: Total={total_rms:.3f} (Rho={rho_rms:.3f}, Phs={phs_rms:.3f})")
                return True
        
        # 3. Relative convergence (Loss stagnant)
        # Check if loss changed little over past N iterations
        window = 20
        if len(self.loss_history) > window:
            prev_loss = self.loss_history[-window]['total_loss']
            curr_loss = self.loss_history[-1]['total_loss']
            rel_change = abs(prev_loss - curr_loss) / (prev_loss + 1e-10)
            
           
            if rel_change < tol and total_rms < 1.5:
                print(f"🛑 [Stop] Loss converged (stagnant), {window}-epoch rel change {rel_change:.2e} < {tol}")
                print(f"   Current RMS: Total={total_rms:.3f}")
                return True
                
        return False

    def run_inversion(self, 
                 num_epochs: int = 1000, 
                 print_interval: int = 10, 
                 seed: int = None,
                 track_chi2: bool = True, 
                 enable_auto_stop: bool = True,
                 use_adaptive_lambda: bool = False,
                 current_lambda: float = 0.0001,
                 warmup_epochs: int = 30,
                 update_interval: int = 20,
                 alpha: float = 0.1,
                 target_rms: float = 1.05) -> List[float]:
        """
        Run inversion (supports adaptive regularization).
        
        Prerequisites:
        - setup_constraints() must be called before
        - setup_optimizer() must be called before
        
        Args:
            num_epochs: Number of iterations; suggest ≥200~500 for log param.
            print_interval: Print interval
            use_adaptive_lambda: Enable adaptive lambda update
            current_lambda: Initial regularization parameter
            warmup_epochs: Warmup epochs
            update_interval: Lambda update interval
            alpha: Gradient balance decay factor
        """
        
        # ===== Check prerequisites =====
        if not hasattr(self, 'constraint_calc'):
            raise RuntimeError("Must call setup_constraints() first")
        
        if not hasattr(self, 'optimizer'):
            raise RuntimeError("Must call setup_optimizer() first")
        
        self.seed = seed or int(time.time())
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
            
            # ===== Get conductivity (log param: exp then forward & loss) =====
            sig_raw = torch.exp(self.log_sig_inv)
            
            # ===== Forward pass =====
            if self.use_sinkhorn:
                zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                    self.freq, self.dz_inv, sig_raw
                )

                # Data normalization
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
                loss_model_occam = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model_occam = self.constraint_calc.calculate_1d_model_norm(
                        model_for_occam,
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
                loss_model_occam = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model_occam = self.constraint_calc.calculate_1d_model_norm(
                        model_for_occam,
                        constraint_type=self.constraint_type,
                        dz=self.dz_inv
                    )
            
            # ===== Compute gradient norms (for logging/plot); update lambda at interval =====
            proposed_lambda, g_d, g_m = self.update_lambda_by_gradient_balance(
                loss_data,
                loss_model_occam,
                current_lambda,
                alpha=alpha,
                smoothing_window=10
            )
            if use_adaptive_lambda and epoch > warmup_epochs and epoch % update_interval == 0:
                current_lambda = proposed_lambda
                self.lambda_history.append(current_lambda)
            
            # ===== Total loss =====
            total_loss = loss_data + current_lambda * loss_model_occam
            
            # ===== Reference model correction =====
            if self.reference_sig is not None and self.ref_weight > 0:
                ref = self.reference_sig.to(self.device).detach().clamp(min=1e-6)
                log10_ref = torch.log10(ref)
                log10_sig = self.log_sig_inv / math.log(10)
                loss_ref = self.ref_weight * torch.mean(
                    (log10_sig - log10_ref) ** 2
                )
                total_loss = total_loss + loss_ref
            
            # ===== Backward and optimize =====
            total_loss.backward()
            
            # Gradient clipping
            self.optimizer_config.clip_gradients(
                [self.log_sig_inv], 
                max_norm=self.gradient_clip_value
            )
            
            self.optimizer.step()
            
            # Parameter clamp (log space: conductivity ~ 1e-4 ~ 10)
            self.optimizer_config.clamp_parameters(
                self.log_sig_inv, 
                min_val=-9.2, 
                max_val=2.3, 
                use_log_space=True
            )
            
            epoch_time = time.time() - start_epoch
            rms_chi2 = self._compute_rms_chi2_from_pred(rho_pred, phs_pred)
            # Epoch logging: chi2, lambda, gradient norms for plotting
            self.loss_history.append({
                'epoch': epoch + 1,
                'total_loss': total_loss.item(),
                'data_loss': loss_data.item(),
                'model_loss': loss_model_occam.item(),
                'misfit': rms_chi2,
                'lambda': current_lambda,
                'epoch_time': epoch_time,
                'grad_data_norm': g_d,
                'grad_model_norm': g_m,
            })
            self.data_misfit_history.append(loss_data.item())
            self.model_norm_history.append(loss_model_occam.item())
            self.regularization_history.append((current_lambda * loss_model_occam).item())
            self.time_history.append(epoch_time)

            # ===== Sinkhorn blur adaptive decay =====
            if self.use_sinkhorn:
                prev_blur = self.current_blur
                self.current_blur = max(self.blur_min, self.current_blur * self.blur_decay)
                if abs(self.current_blur - prev_blur) > 1e-8:
                    self.sinkhorn_loss.blur = self.current_blur

            # ===== Periodically compute χ² RMS and check convergence =====
            if track_chi2 and (epoch + 1) % print_interval == 0:
                chi2_results = self.calculate_chi2_rms()
                # Call convergence check
                if enable_auto_stop:
                    should_stop = self._check_convergence(
                        chi2_results, 
                        target_rms=target_rms, 
                        tol=1e-4
                    )
                    
                    if should_stop:
                        print(f"Inversion converged at epoch {epoch + 1}.")
                        break

            # --- Print progress every print_interval ---
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
                print(f"  Epoch time: {epoch_sec:.2f}s | Avg: {avg_epoch_time:.2f}s")
                print(f"  Total: {total_loss.item():.4e} | Data({data_label}): {loss_data.item():.4e}")
                print(f"  Misfit(RMS χ²): {total_rms:.3f} | Rough: {loss_model_occam.item():.2e} | Lam: {current_lambda:.7f}")
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

        # ===== Inversion complete =====
        total_time = time.time() - start_total
        avg_epoch_time = total_time / (epoch + 1) if epoch > 0 else total_time
        
        # Final χ² evaluation
        if track_chi2:
            final_chi2 = self.calculate_chi2_rms()
            print(f"\n=== Final χ² Statistics ===")
            print(f"Apparent resistivity χ² RMS: {final_chi2['rho_chi2_rms']:.3f}")
            print(f"Phase χ² RMS: {final_chi2['phs_chi2_rms']:.3f}") 
            print(f"Total χ² RMS: {final_chi2['total_chi2_rms']:.3f}")
            
            total_chi2 = final_chi2['total_chi2_rms']
            if 0.8 <= total_chi2 <= 1.2:
                print("Good fit: χ² ≈ 1.0, model fits data within error")
            elif total_chi2 > 1.5:
                print("Under-fit: χ² > 1.5, model insufficient")
            elif total_chi2 < 0.5:
                print("Over-fit: χ² < 0.5, model may fit noise")
        
        print(f"\nInversion finished. Total time: {total_time:.2f}s ({total_time/60:.2f}min), "
            f"Average epoch time: {avg_epoch_time:.3f}s")
        
        final_sig = torch.exp(self.log_sig_inv).detach().cpu().numpy()

        print(f"\n=== Final Results ===")
        print(f"True dz: {self.true_dz.cpu().numpy().tolist()}")
        print(f"True sig: {self.true_sig.cpu().numpy().tolist()}")
        print(f"Inverted sig: {final_sig.tolist()}")

        return self.loss_history
    def setup_constraints(self, 
                     constraint_type: str = 'roughness',
                     use_occam_constraint: bool = True,
                     ref_weight: float = 0.0,
                     reference_sig: Optional[torch.Tensor] = None):
        """
        Setup inversion constraints.
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
        Compute normalized sensitivity matrix (Jacobian).
        
        Formula: J_ij = ∂(log10(ρ_i)) / ∂(log10(σ_j))
        
        Returns:
            J (np.ndarray): Sensitivity matrix [n_freq, n_layer]
            z_grid (np.ndarray): Depth grid nodes (for plot x-axis)
        """
        # LN_10 for chain rule
        LN_10 = math.log(10.0)
        
        # 1. Params for gradient (log param: log_sig_inv)
        sig_inv = self.log_sig_inv.detach().clone().requires_grad_(True)
        # 2. Forward: conductivity = exp(log_sig_inv)
        _, rho_pred, _ = self.mt1d_forward(self.freq, self.dz_inv, torch.exp(sig_inv))
        
        # 3. Per-frequency derivative
        target = torch.log10(rho_pred)
        n_data = len(target)
        n_param = len(sig_inv)
        
        J = torch.zeros((n_data, n_param), device=self.device)
        
        for i in range(n_data):
            if i == n_data - 1:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=False)[0]
            else:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=True)[0]
            
            # Normalize: d(log10_rho)/d(log10_sig) = d(log10_rho)/d(log_sig) * ln(10)
            grad = grad * LN_10
            J[i, :] = grad
            
        return J.detach().cpu().numpy(), np.concatenate(([0], np.cumsum(self.dz_inv.detach().cpu().numpy())))
        
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
        
        # Apparent resistivity subplot
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
        Plot loss, chi2 and lambda evolution during inversion.
        X-axis: total epoch count.
        """
        if not self.loss_history:
            print("No loss history found; run run_inversion first.")
            return None
        # Legacy: scalar list has no misfit/lambda
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
        # Plot 1: Data misfit RMS (χ²)
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='RMS χ²')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit (χ² RMS)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_yscale('log')
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].legend()
        # Plot 2: Data Loss vs Model Loss
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
        # Plot 3: Lambda
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization (Lambda)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lambda")
        axes[2].set_yscale('log')
        axes[2].grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_gradient_history(self) -> plt.Figure:
        """Plot gradient norms of data and model terms vs epoch."""
        if not self.loss_history:
            print("No loss history found; run run_inversion first.")
            return None
        first = self.loss_history[0]
        if isinstance(first, (int, float)) or 'grad_data_norm' not in first or 'grad_model_norm' not in first:
            print("No gradient norms in loss_history; re-run inversion with latest run_inversion.")
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
            """Create depth-resistivity profile from model"""
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
        
        # No main title
        
        plt.tight_layout()
        return plt.gcf()


    def plot_chi2_history(self) -> plt.Figure:
        """Plot χ² history; x-axis is epoch."""
        if not self.chi2_history:
            print("Warning: No χ² history available")
            return None
        # If dict with epoch (print_interval sampling)
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
        
        # Add ideal fit line
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal Fit (χ²=1)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('χ² RMS', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.set_title('χ² RMS History', fontsize=14)
        
        plt.tight_layout()
        return fig

    def plot_sensitivity(self) -> None:
        """Plot sensitivity matrix heatmap (show once, no return to avoid Jupyter duplicate)"""
        J, z_grid = self.calculate_sensitivity_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        # X-axis: frequency index (high to low)
        # Y-axis: layer depth
        # Transpose: rows=depth, cols=frequency
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