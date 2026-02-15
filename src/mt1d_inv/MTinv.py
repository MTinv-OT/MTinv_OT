import time
from datetime import datetime, timedelta
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
try:
    # 首先尝试相对导入（当作为包的一部分时）
    from .constraints import ConstraintCalculator
    from .optimizer import OptimizerConfig
except ImportError:
    try:
        # 再尝试绝对导入（从src开始）
        from src.constraints import ConstraintCalculator
        from src.optimizer import OptimizerConfig
    except ImportError:
        # 最后尝试直接导入（如果在src目录中运行）
        from constraints import ConstraintCalculator
        from optimizer import OptimizerConfig


class MT1DInverter:
    """
    MT 1D 反演类 (融合: 选择性对数参数化 + Occam约束)
    """
    MU = 4e-7 * math.pi
    PI = math.pi

    def __init__(self, device: str = None, mu: float = None, 
                 use_sinkhorn: bool = True, sinkhorn_dim: int = 3,use_data_weighting: bool = True, constraint_type: str = "roughness"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = mu or self.MU
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_dim = sinkhorn_dim

        # 模型参数
        self.true_dz = None
        self.true_sig = None
        self.dz_inv = None
        self.log_sig_inv = None
        self.sig_inv = None

        # 数据
        self.freq = None
        self.zxy_obs = None
        self.rho_obs = None
        self.phs_obs = None
        self.noise_level = None

        # 误差估计
        self.delta_rho = None
        self.delta_phs = None

        # 优化器和约束计算器
        self.optimizer_config = None
        self.optimizer_config = OptimizerConfig(device=self.device)
        self.constraint_calc = ConstraintCalculator(device=self.device)
        self.loss_history = []

        # Occam参数
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

        # 风格数据权重
        self.rho_weights = None
        self.phs_weights = None
        self.use_data_weighting = use_data_weighting
        self.gradient_clip_value = 1.0  # 添加梯度裁剪参数
        # 参考模型修正：使反演结果向参考模型靠拢
        self.reference_sig = None   # 参考电导率 (n_layers,)，与反演层数一致
        self.ref_weight = 0.0      # 参考模型惩罚权重，0 表示不使用
        param_mode = "对数参数空间"
        print(f"Using device: {self.device}")
        print(f"Parameterization: {param_mode}")
        print(f"Using {sinkhorn_dim}D Sinkhorn loss function" if use_sinkhorn else "Using MSE loss function")

    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MT 1D 正演"""
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
        根据阻抗误差计算视电阻率和相位的误差
        使用误差传播定律
        """
        # 阻抗实部虚部误差（已知）
        sigma_Z_real = self.delta_zxy_real
        sigma_Z_imag = self.delta_zxy_imag
        
        Z = self.zxy_obs
        Z_abs = torch.abs(Z)
        
        # 视电阻率误差传播
        # ρ_a = |Z|² / (ωμ) => σ_ρ ≈ 2ρ × (σ_Z/|Z|)
        omega = 2.0 * self.PI * self.freq
        rho_apparent = torch.abs(Z)**2 / (omega * self.MU)
        
        # 相对误差：σ_ρ/ρ ≈ 2 × σ_Z/|Z|
        relative_error_rho = 2.0 * self.noise_level  # 因为 σ_Z/|Z| = noise_level
        self.delta_rho = relative_error_rho * rho_apparent
        
        # 相位误差传播  
        # φ = atan2(Z_imag, Z_real) => σ_φ ≈ σ_Z/|Z| (弧度)
        sigma_phi_rad = self.noise_level  # σ_Z/|Z| = noise_level
        self.delta_phs = torch.full_like(self.freq, sigma_phi_rad * 180.0 / self.PI)  # 转换为度
        
        print(f"视电阻率误差范围: {torch.min(self.delta_rho):.4f} - {torch.max(self.delta_rho):.4f} Ω·m")
        print(f"相位误差: {torch.mean(self.delta_phs):.2f}°")
        
        # 归一化权重(使均值=1)
        eps = 1e-10
        self.rho_weights = 1.0 / (self.delta_rho + eps)
        self.phs_weights = 1.0 / (self.delta_phs + eps)
        self.rho_weights = self.rho_weights / torch.mean(self.rho_weights)
        self.phs_weights = self.phs_weights / torch.mean(self.phs_weights)
        if self.use_data_weighting:
        # 使用精确的误差传播计算权重
            self.rho_weights = 1.0 / (self.delta_rho + 1e-10)
            self.phs_weights = 1.0 / (self.delta_phs + 1e-10)
            
            # 归一化权重
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
        生成合成观测数据（阻抗层面加噪声后导出 ρ/φ）。

        Args:
            true_dz: 真实层厚 (m)
            true_sig: 真实电导率 (S/m)
            freq_range: 频率范围 (log10 Hz)
            n_freq: 频点数
            noise_level: 相对噪声水平（相对 |Z|）
            noise_type: "gaussian" 仅高斯噪声；"nongaussian" 在高斯基础上叠加随机离群值。
                其他常见非高斯方式（可后续扩展）：拉普拉斯/双指数、Student-t 重尾、均匀野值等。
            outlier_frac: 离群值比例 (0~1)，仅当 noise_type=="nongaussian" 时有效
            outlier_strength: 离群值强度（相对基线 delta 的倍数），仅 nongaussian 时有效
            seed: 随机种子
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

        # 非高斯：在部分点上叠加离群噪声（模拟野值）
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

        # 计算视电阻率和相位的误差
        self.calculate_data_errors()

        # 保存对数域的噪声标准差（仍按基线高斯水平，离群值会体现在残差中）
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
        初始化反演模型（支持不同的厚度分配策略）
        """
        # 验证
        if n_layers < 2:
            raise ValueError("n_layers 必须 >= 2")

        n_dz = n_layers - 1  # 厚度块数
        device = self.device

        if thickness_mode == "equal":
            dz_value = total_depth / n_dz
            self.dz_inv = torch.full((n_dz,), dz_value, dtype=torch.float32,
                                    device=device, requires_grad=False)
            mode_name = "等厚度"

        elif thickness_mode in ("increasing_linear", "increasing_geometric"):
            # 生成一个基准权重序列，然后归一化乘以 total_depth
            if thickness_mode == "increasing_linear":
                # 线性或幂律增长：权重 i^p （i 从 1 到 n_dz）
                p = float(increasing_exponent) if increasing_exponent > 0 else 1.0
                indices = np.arange(1, n_dz + 1, dtype=np.float64)
                weights = indices ** p
                mode_name = f"线性/幂次增长 (exponent={p})"
            else:  # increasing_geometric
                # 几何增长：权重 = r^(i-1)，通过给定 exponent 找到 r 使得总和合理。
                r = float(increasing_exponent) ** (1.0 / max(1, n_dz - 1))
                indices = np.arange(0, n_dz, dtype=np.float64)
                weights = r ** indices
                mode_name = f"几何增长 (approx r={r:.3f})"

            weights_sum = np.sum(weights)
            if weights_sum <= 0:
                raise ValueError("生成的权重和为0，检查参数")
            
            dz_np = (weights / weights_sum) * total_depth
            self.dz_inv = torch.tensor(dz_np, dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            raise ValueError(f"不支持的 thickness_mode: {thickness_mode}")
        
        # 统一使用对数参数化：Sinkhorn 与非 Sinkhorn 均优化 log(sig)
        self.log_sig_inv = torch.full(
            (n_layers,),
            torch.log(torch.tensor(initial_sig, dtype=torch.float32, device=self.device)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        if self.use_sinkhorn:
            print(f"初始化对数电导率参数（Sinkhorn模式）")
        else:
            print(f"初始化对数电导率参数（非Sinkhorn模式）")
        print(f"初始电导率: {torch.exp(self.log_sig_inv).tolist()}")

        print(f"使用{mode_name}模式")
        cum_depth = np.cumsum(self.dz_inv.detach().cpu().numpy())
        print(f"初始化模型，共 {n_layers} 层")
        print(f"累计深度: {cum_depth.tolist()}")

    
    def set_reference_model(self, ref_sig, weight: float = 0.01) -> None:
        """
        设置参考模型修正：在损失中加入 weight * ||log10(sig) - log10(ref)||^2，
        使反演结果向参考模型靠拢。
        ref_sig: 参考电导率，形状 (n_layers,) 需与 initialize_model 的 n_layers 一致；
                 可为 list、numpy 或 tensor。
        weight: 参考模型惩罚权重，0 表示关闭。典型值 0.001~0.1。
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
            raise ValueError(f"参考模型层数 {n} 与反演层数 {self.log_sig_inv.numel()} 不一致，请先 initialize_model 再 set_reference_model")
        print(f"参考模型修正已开启: ref_weight={self.ref_weight}, 参考层数={n}")

    def setup_optimizer(self, lr: float = 0.01, reg_weight_sig: float = 0.0001, phs_weight: float = 0.5,
                    p: int = 2, scaling: float = 0.9,
                    reach: Optional[float] = None,
                    optimizer_type: str = "AdamW", weight_decay: float = 0.0,
                    betas: Tuple[float, float] = (0.9, 0.999),
                    eps: float = 1e-8, momentum: float = 0.9) -> None:
        """
        设置优化器（使用优化器配置模块）。
        当前为对数参数化 log(σ)，梯度尺度与原始 σ 不同：∂L/∂(log σ)=σ·∂L/∂σ，
        小电导率时更新较慢。若 100 轮内拟合不足，可增大 lr（如 0.01～0.02）或增加 num_epochs。
        """
        self.reg_weight_sig = reg_weight_sig
        self.phs_weight = phs_weight
        self.p_norm = p

        # 确保参数已经初始化（Sinkhorn 与 MSE 均使用对数参数 log_sig_inv）
        if self.log_sig_inv is None:
            raise ValueError("log_sig_inv 未初始化，请先调用 initialize_model")
        params = [self.log_sig_inv]

        # 使用优化器配置模块创建优化器
        self.optimizer = self.optimizer_config.create_optimizer(
            params=params,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            momentum=momentum
        )

        # 创建损失函数
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
        force_monotonic_decrease: bool = True  # <--- 【新增开关】是否强制单调递减/occam or not
    ):
        """
        [自动梯度平衡 - 单调递减版]
        """
        
        # 1. 计算当前梯度范数
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
            
        # 2. 历史平滑
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

        # 3. 自动计算目标 Lambda
        if avg_norm_m < 1e-20:
            target_lambda = current_lambda
        else:
            # 这里的 target 就是为了让两边梯度平衡
            target_lambda = avg_norm_d / (avg_norm_m + 1e-20)

        # 4. 对数域平滑更新
        target_lambda = max(lambda_min, min(lambda_max, target_lambda))
        current_lambda = max(lambda_min, current_lambda)
        
        log_curr = math.log10(current_lambda)
        log_target = math.log10(target_lambda)
        
        log_new = (1.0 - alpha) * log_curr + alpha * log_target
        proposed_lambda = 10.0 ** log_new

        # ---------------------------------------------------------
        # 5. 【关键修改】强制单调递减逻辑 (复现你想要的功能)
        # ---------------------------------------------------------
        if force_monotonic_decrease:
            # 如果计算出的新 lambda 比当前大，则忽略，保持当前值（或轻微衰减）
            # 这对应你旧代码里的 else: new_lambda = current_lambda
            if proposed_lambda > current_lambda:
                new_lambda = current_lambda
            else:
                new_lambda = proposed_lambda
        else:
            # 允许双向波动
            new_lambda = proposed_lambda

        return new_lambda, norm_d_curr, norm_m_curr

    def _compute_rms_chi2_from_pred(self, rho_pred: torch.Tensor, phs_pred: torch.Tensor) -> float:
        """从当前步的预测 (rho_pred, phs_pred) 计算总 χ² RMS，无需额外正演。"""
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
        计算基于χ²统计量的RMS，这是最专业的误差评估方法
        """
        with torch.no_grad():
            sig_raw = torch.exp(self.log_sig_inv)
            zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                self.freq, self.dz_inv, sig_raw)
        
        # 转换为numpy便于计算
        rho_obs_np = self.rho_obs.cpu().numpy()
        rho_pred_np = rho_pred.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        phs_pred_np = phs_pred.cpu().numpy()
        delta_rho_np = self.delta_rho.cpu().numpy()
        delta_phs_np = self.delta_phs.cpu().numpy()
        
        results = {}
        
        # 1. 视电阻率χ²
        rho_chi = (rho_obs_np - rho_pred_np) / delta_rho_np
        rho_chi_squared = rho_chi**2
        rho_chi2_rms = np.sqrt(np.mean(rho_chi_squared))
        results['rho_chi2_rms'] = float(rho_chi2_rms)
        results['rho_chi2_mean'] = float(np.mean(rho_chi_squared))
        results['rho_chi2_max'] = float(np.max(rho_chi_squared))
        
        # 2. 相位χ²
        phs_chi = (phs_obs_np - phs_pred_np) / delta_phs_np
        phs_chi_squared = phs_chi**2
        phs_chi2_rms = np.sqrt(np.mean(phs_chi_squared))
        results['phs_chi2_rms'] = float(phs_chi2_rms)
        results['phs_chi2_mean'] = float(np.mean(phs_chi_squared))
        results['phs_chi2_max'] = float(np.max(phs_chi_squared))
        
        # 3. 总χ² RMS（等权重）
        total_chi2_rms = np.sqrt(0.5 * rho_chi2_rms**2 + 0.5 * phs_chi2_rms**2)
        results['total_chi2_rms'] = float(total_chi2_rms)
        
        # 4. 传统RMS（绝对误差）
        rho_obs_log = np.log10(rho_obs_np)
        rho_pred_log = np.log10(rho_pred_np)
        results['rho_rms_log'] = float(np.sqrt(np.mean((rho_obs_log - rho_pred_log)**2)))
        results['phs_rms_deg'] = float(np.sqrt(np.mean((phs_obs_np - phs_pred_np)**2)))
        
        # 5. 相对误差
        rho_relative_error = np.abs(rho_obs_np - rho_pred_np) / (rho_obs_np + 1e-10)
        results['rho_rms_relative'] = float(np.sqrt(np.mean(rho_relative_error**2)))
        
        phs_relative_error = np.abs(phs_obs_np - phs_pred_np) / 90.0
        results['phs_rms_relative'] = float(np.sqrt(np.mean(phs_relative_error**2)))
        
        # 6. 新增：诊断信息
        results['n_outliers_rho'] = int(np.sum(rho_chi_squared > 9))  # χ² > 9 (3σ) 的点数
        results['n_outliers_phs'] = int(np.sum(phs_chi_squared > 9))
        
        # 计算误差条的质量（δ是否合理）
        results['error_scale_rho'] = float(np.sqrt(np.mean(rho_chi_squared)))  # 应该≈1
        results['error_scale_phs'] = float(np.sqrt(np.mean(phs_chi_squared)))  # 应该≈1
        
        return results
    
    #Params: 放在 calculate_chi2_rms 和 run_inversion 之间
    def _check_convergence(self, chi2_results: Dict[str, float], 
                         target_rms: float, 
                         tol: float = 1e-4) -> bool:
        """
        内部收敛检查逻辑：同时检查绝对RMS目标和相对Loss变化
        """
        if target_rms is None: target_rms = 1.05
        # 1. 提取当前指标
        rho_rms = chi2_results['rho_chi2_rms']
        phs_rms = chi2_results['phs_chi2_rms']
        total_rms = chi2_results['total_chi2_rms']

        # 2. 绝对收敛标准 (RMS达标)
        # 策略：总RMS达标，且两者偏差不能太极端（防止单一方拟合极差）
        if total_rms < target_rms:
            if rho_rms < target_rms * 1.5 and phs_rms < target_rms * 1.5:
                print(f"✅ [停止] 达到目标 RMS: Total={total_rms:.3f} (Rho={rho_rms:.3f}, Phs={phs_rms:.3f})")
                return True
        
        # 3. 相对收敛标准 (Loss 停滞)
        # 检查过去 N 次迭代 Loss 是否几乎没变
        window = 20
        if len(self.loss_history) > window:
            prev_loss = self.loss_history[-window]['total_loss']
            curr_loss = self.loss_history[-1]['total_loss']
            rel_change = abs(prev_loss - curr_loss) / (prev_loss + 1e-10)
            
           
            if rel_change < tol and total_rms < 1.5:
                print(f"🛑 [停止] Loss 收敛停滞，{window}轮内相对变化率 {rel_change:.2e} < {tol}")
                print(f"   当前 RMS: Total={total_rms:.3f}")
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
        执行反演过程（支持自适应正则化）
        
        前置条件：
        - setup_constraints() 必须在此方法之前调用
        - setup_optimizer() 必须在此方法之前调用
        
        Args:
            num_epochs: 迭代次数。对数参数化下建议 ≥200～500，若拟合慢可再增大或适当提高 setup_optimizer 的 lr。
            print_interval: 打印间隔
            use_adaptive_lambda: 是否启用自适应 lambda 更新
            current_lambda: 初始正则化参数
            warmup_epochs: 预热阶段轮数
            update_interval: lambda 更新间隔
            alpha: 梯度平衡的指数下降因子
            lambda_min: lambda 最小值
            bl: 平衡系数
        """
        
        # ===== 检查必要的初始化 =====
        if not hasattr(self, 'constraint_calc'):
            raise RuntimeError("必须先调用 setup_constraints() 方法")
        
        if not hasattr(self, 'optimizer'):
            raise RuntimeError("必须先调用 setup_optimizer() 方法")
        
        self.seed = seed or int(time.time())
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f"[Info] Random seed = {self.seed}")
        
        # ===== 初始化历史记录 =====
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
            
            # ===== 获取电导率（对数参数化：先 exp 再参与正演与损失） =====
            sig_raw = torch.exp(self.log_sig_inv)
            
            # ===== 正向计算 =====
            if self.use_sinkhorn:
                zxy_pred, rho_pred, phs_pred = self.mt1d_forward(
                    self.freq, self.dz_inv, sig_raw
                )

                # 数据标准化
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

                # ===== Sinkhorn 数据项 =====
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
            
                # ===== Occam 约束项（对数空间） =====
                loss_model_occam = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model_occam = self.constraint_calc.calculate_1d_model_norm(
                        model_for_occam,
                        constraint_type=self.constraint_type,
                        dz=self.dz_inv
                    )

            else:
                # ===== 非 Sinkhorn 模式 =====
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

                # ===== 约束项 =====
                loss_model_occam = torch.tensor(0.0, device=self.device)
                if self.use_occam_constraint:
                    model_for_occam = self.log_sig_inv / math.log(10)
                    loss_model_occam = self.constraint_calc.calculate_1d_model_norm(
                        model_for_occam,
                        constraint_type=self.constraint_type,
                        dz=self.dz_inv
                    )
            
            # ===== 每轮计算梯度范数（用于记录与绘图）；自适应时按间隔更新 lambda =====
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
            
            # ===== 计算总损失 =====
            total_loss = loss_data + current_lambda * loss_model_occam
            
            # ===== 参考模型修正 =====
            if self.reference_sig is not None and self.ref_weight > 0:
                ref = self.reference_sig.to(self.device).detach().clamp(min=1e-6)
                log10_ref = torch.log10(ref)
                log10_sig = self.log_sig_inv / math.log(10)
                loss_ref = self.ref_weight * torch.mean(
                    (log10_sig - log10_ref) ** 2
                )
                total_loss = total_loss + loss_ref
            
            # ===== 反向传播和优化 =====
            total_loss.backward()
            
            # 梯度裁剪
            self.optimizer_config.clip_gradients(
                [self.log_sig_inv], 
                max_norm=self.gradient_clip_value
            )
            
            self.optimizer.step()
            
            # 参数约束（对数空间：对应电导率约 1e-4 ~ 10）
            self.optimizer_config.clamp_parameters(
                self.log_sig_inv, 
                min_val=-9.2, 
                max_val=2.3, 
                use_log_space=True
            )
            
            epoch_time = time.time() - start_epoch
            rms_chi2 = self._compute_rms_chi2_from_pred(rho_pred, phs_pred)
            # 每轮记录：总迭代次数为横坐标，卡方、lambda、梯度范数供绘图
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

            # ===== Sinkhorn blur 自适应衰减 =====
            if self.use_sinkhorn:
                prev_blur = self.current_blur
                self.current_blur = max(self.blur_min, self.current_blur * self.blur_decay)
                if abs(self.current_blur - prev_blur) > 1e-8:
                    self.sinkhorn_loss.blur = self.current_blur

            # ===== 定期计算χ² RMS 并检查收敛（chi2 详细记录在下方 print_interval 块中带 epoch 写入） =====
            if track_chi2 and (epoch + 1) % print_interval == 0:
                chi2_results = self.calculate_chi2_rms()
                # 【新增】调用封装好的收敛检查函数
                if enable_auto_stop:
                    should_stop = self._check_convergence(
                        chi2_results, 
                        target_rms=target_rms, 
                        tol=1e-4
                    )
                    
                    if should_stop:
                        print(f"反演在第 {epoch + 1} 轮收敛。")
                        break

            # --- 7. 每 print_interval 轮输出一次进度（格式与 2D 统一）---
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
                print(f"  已用时间: {elapsed_str} | 剩余时间: ~{remaining_str} | ETA: {eta_str}")
                print(f"  Epoch耗时: {epoch_sec:.2f}s | 平均耗时: {avg_epoch_time:.2f}s")
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

        # ===== 反演完成 =====
        total_time = time.time() - start_total
        avg_epoch_time = total_time / (epoch + 1) if epoch > 0 else total_time
        
        # 最终χ²评估
        if track_chi2:
            final_chi2 = self.calculate_chi2_rms()
            print(f"\n=== 最终χ²统计结果 ===")
            print(f"视电阻率 χ² RMS: {final_chi2['rho_chi2_rms']:.3f}")
            print(f"相位 χ² RMS: {final_chi2['phs_chi2_rms']:.3f}") 
            print(f"总 χ² RMS: {final_chi2['total_chi2_rms']:.3f}")
            
            total_chi2 = final_chi2['total_chi2_rms']
            if 0.8 <= total_chi2 <= 1.2:
                print("优秀拟合：χ² ≈ 1.0，模型在误差范围内完美拟合数据")
            elif total_chi2 > 1.5:
                print("拟合不足：χ² > 1.5，模型未能充分拟合数据")
            elif total_chi2 < 0.5:
                print("过度拟合：χ² < 0.5，模型可能拟合了噪声")
        
        print(f"\nInversion finished. Total time: {total_time:.2f}s ({total_time/60:.2f}min), "
            f"Average epoch time: {avg_epoch_time:.3f}s")
        
        final_sig = torch.exp(self.log_sig_inv).detach().cpu().numpy()

        print(f"\n=== 最终结果 ===")
        print(f"True dz: {self.true_dz.cpu().numpy().tolist()}")
        print(f"True sig: {self.true_sig.cpu().numpy().tolist()}")
        print(f"Inverted sig: {final_sig.tolist()}")

        return self.loss_history
    # 文件：src/mt1d_inv/MTinv.py
    def setup_constraints(self, 
                     constraint_type: str = 'roughness',
                     use_occam_constraint: bool = True,
                     ref_weight: float = 0.0,
                     reference_sig: Optional[torch.Tensor] = None):
        """
        设置反演约束条件
        """
        self.use_occam_constraint = use_occam_constraint
        self.constraint_type = constraint_type
        self.ref_weight = ref_weight
        
        # reference_sig 的处理
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
        计算全归一化灵敏度矩阵 (Jacobian)
        
        标准化公式: J_ij = ∂(log10(ρ_i)) / ∂(log10(σ_j))
        物理含义：电阻率变化 1 个数量级会导致视电阻率变化多少个数量级。
        
        Returns:
            J (np.ndarray): 灵敏度矩阵 [n_freq, n_layer]
            z_grid (np.ndarray): 深度网格节点 (用于绘图x轴)
        """
        # 常数 ln(10) 用于链式法则转换
        LN_10 = math.log(10.0)
        
        # 1. 准备需计算梯度的参数（对数参数化，统一用 log_sig_inv）
        sig_inv = self.log_sig_inv.detach().clone().requires_grad_(True)
        # 2. 前向计算：电导率 = exp(log_sig_inv)
        _, rho_pred, _ = self.mt1d_forward(self.freq, self.dz_inv, torch.exp(sig_inv))
        
        # 3. 逐频点求导
        target = torch.log10(rho_pred)
        n_data = len(target)
        n_param = len(sig_inv)
        
        J = torch.zeros((n_data, n_param), device=self.device)
        
        for i in range(n_data):
            if i == n_data - 1:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=False)[0]
            else:
                grad = torch.autograd.grad(target[i], sig_inv, retain_graph=True)[0]
            
            # --- 归一化：d(log10_rho)/d(log10_sig) = d(log10_rho)/d(log_sig) * ln(10) ---
            grad = grad * LN_10
            J[i, :] = grad
            
        return J.detach().cpu().numpy(), np.concatenate(([0], np.cumsum(self.dz_inv.detach().cpu().numpy())))
        
    # 其他绘图方法保持不变...
    def plot_synthetic_data(self) -> None:
        """绘制合成数据图"""
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
        """绘制数据拟合图"""
        with torch.no_grad():
            sig_raw = torch.exp(self.log_sig_inv)
            zxy_final_pred, rho_final_pred, phs_final_pred = self.mt1d_forward(
                self.freq, self.dz_inv, sig_raw)
        
        # 计算全面的RMS指标
        rms_results = self.calculate_chi2_rms()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        freq_np = self.freq.cpu().numpy()
        rho_obs_np = self.rho_obs.cpu().numpy()
        rho_pred_np = rho_final_pred.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        phs_pred_np = phs_final_pred.cpu().numpy()
        
        # 视电阻率子图
        ax1.loglog(freq_np, rho_obs_np, 'ro', markersize=4, label='Observed', alpha=0.7)
        ax1.loglog(freq_np, rho_pred_np, 'b-', linewidth=2, label='Predicted')
        ax1.set_xlabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, which="both", linestyle='--', alpha=0.5)
        ax1.set_title(f'Apparent Resistivity Fit\nχ² RMS = {rms_results["rho_chi2_rms"]:.3f}', fontsize=13)
        
        # 相位子图
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
        绘制反演过程中的损失、卡方与 lambda 演化曲线。
        横坐标为总迭代次数 (epoch)。
        """
        if not self.loss_history:
            print("没有找到损失历史记录，请先运行 run_inversion。")
            return None
        # 兼容旧版：若为标量列表则无 misfit/lambda
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
        # 图1: 数据拟合 RMS (χ²)
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='RMS χ²')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit (χ² RMS)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_yscale('log')
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].legend()
        # 图2: Data Loss vs Model Loss
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
        # 图3: Lambda
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization (Lambda)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lambda")
        axes[2].set_yscale('log')
        axes[2].grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_gradient_history(self) -> plt.Figure:
        """绘制数据项和模型项梯度范数随总迭代次数的变化。"""
        if not self.loss_history:
            print("没有找到损失历史记录，请先运行 run_inversion。")
            return None
        first = self.loss_history[0]
        if isinstance(first, (int, float)) or 'grad_data_norm' not in first or 'grad_model_norm' not in first:
            print("当前 loss_history 中无梯度范数，请用最新版 run_inversion 重新反演。")
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
        """绘制模型对比图"""
        inv_sig = torch.exp(self.log_sig_inv).detach().cpu().numpy()
        
        true_sig_np = self.true_sig.cpu().numpy()
        true_dz_np = self.true_dz.cpu().numpy()
        inv_dz_np = self.dz_inv.detach().cpu().numpy()
        
        def create_model_profile(dz, sig):
            """创建模型的深度-电阻率剖面"""
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
        
        # 去掉大标题，不使用 plt.title()
        
        plt.tight_layout()
        return plt.gcf()


    def plot_chi2_history(self) -> plt.Figure:
        """绘制χ²历史图，横坐标为总迭代次数。"""
        if not self.chi2_history:
            print("Warning: No χ² history available")
            return None
        # 若为带 epoch 的 dict（print_interval 采样）
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
        
        # 添加理想拟合线
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Ideal Fit (χ²=1)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('χ² RMS', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.set_title('χ² RMS History', fontsize=14)
        
        plt.tight_layout()
        return fig

    def plot_sensitivity(self) -> None:
        """绘制灵敏度矩阵热力图（仅显示一次，不返回 fig 避免 Jupyter 重复输出）"""
        J, z_grid = self.calculate_sensitivity_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制热力图
        # X轴: 频率索引 (从高频到低频)
        # Y轴: 层深度
        
        # 为了绘图方便，通常转置一下：行是深度，列是频率
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