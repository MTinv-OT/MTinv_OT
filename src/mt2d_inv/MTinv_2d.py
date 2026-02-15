import torch
import torch.nn as nn
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
from typing import Dict, Callable, Optional, Union, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from src.constraints import ConstraintCalculator
from src.optimizer import OptimizerConfig
from src.MT2D import MT2DFD_Torch
torch.set_default_dtype(torch.float64)

class MT2DInverter:
    """
    2D 大地电磁(MT)反演器
    """
    
    def __init__(self, 
                 yn: torch.Tensor = None,
                 zn: torch.Tensor = None,
                 freqs: torch.Tensor = None, 
                 stations: torch.Tensor = None,
                 device: str = "cuda", 
                 random_seed: int = 42,
                 sinkhorn_reach: Optional[float] = None,
                 ot_options: Optional[Dict] = None):

        self.set_random_seed(random_seed)
        self.device = device if torch.cuda.is_available() else "cpu"

        self.yn = yn.to(self.device)
        self.zn = zn.to(self.device)
        self.freqs = freqs.to(self.device, dtype=torch.float64)
        self.stations = stations.to(self.device, dtype=torch.float64)
        self.opt_config = OptimizerConfig(self.device)
        
        self.model_log_sigma = None
        self.initial_model_sigma = None
        self.obs_data = {}
        self.forward_operator = None
        self.loss_history = []
        self.sig_true = None
        self.noise_level = None
        self.sig_ref = 0.01
        self.model_log_sigma_ref = None

        self.grad_norm_d_history = []  # 数据项梯度范数历史
        self.grad_norm_m_history = []  # 模型项梯度范数历史
        self.ratio_history = []

        self.time_stats = {
            'total_inversion_time': 0,
            'avg_epoch_time': 0,
            'epoch_times': [],
            'start_time': 0,
            'end_time': 0
        }
        
        self.sinkhorn_loss = None
        # 默认 OT 参数配置，可通过 ot_options 覆盖
        default_ot = {
            "p": 2,
            "blur": 0.01,
            "scaling": 0.9,
            "reach": sinkhorn_reach,
            "backend": "tensorized",
        }

        self.ot_config = default_ot.copy()
        if ot_options is not None:
            self.ot_config.update(ot_options)

        self._init_sinkhorn(**self.ot_config)

    def set_random_seed(self, seed: int = 42):
        """ 设置随机种子 """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"✓ 随机种子已设置: {seed}")

    def set_forward_operator(self,  nza=10):
        def _forward(sigma):
            fwd = MT2DFD_Torch(
                nza=nza,
                zn=self.zn,
                yn=self.yn,
                freq=self.freqs,
                ry=self.stations,
                sig=sigma,
                device=self.device
            )
            return fwd(mode="TETM")

        self.forward_operator = _forward

        # 读取网格
        dummy = torch.ones((len(self.zn)-1, len(self.yn)-1), device=self.device, dtype=torch.float64)
        fwd = MT2DFD_Torch(nza, self.zn, self.yn, self.freqs, self.stations, dummy, self.device)

        self.nz, self.ny = fwd.nz, fwd.ny
        self.dz, self.dy = fwd.dz, fwd.dy

        self.constraint_calc = ConstraintCalculator(
            self.ny-1, self.nz-1, self.dy, self.dz, device=self.device
        )

    def create_synthetic_data(
        self,
        noise_level: float = 0.01,
        noise_type: str = "gaussian",
        outlier_frac: float = 0.05,
        outlier_strength: float = 4.0,
    ):
        """
        二维 MT 合成数据生成：在阻抗层面加噪声，然后统一计算 ρ / φ。

        Args:
            noise_level: 相对噪声水平（相对 |Z|）
            noise_type: "gaussian" 仅高斯噪声；"nongaussian" 在高斯基础上叠加随机离群值
            outlier_frac: 离群值比例 (0~1)，仅当 noise_type=="nongaussian" 时有效
            outlier_strength: 离群值强度（相对基线 delta 的倍数），仅 nongaussian 时有效
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
        print("正在生成二维 MT 合成数据...")

        with torch.no_grad():
            pred_true = self.forward_operator(self.sig_true)

        omega = 2 * np.pi * self.freqs[:, None]
        MU = 4e-7 * np.pi

        obs_data = {}
        self.data_std = {}   # 保存阻抗标准差，供误差传播用

        for mode in ["xy", "yx"]:
            Z = pred_true[f"Z{mode}"]      # (nf, nstation)
            Zabs = torch.abs(Z)

            # -------- 阻抗噪声（相对，高斯）--------
            delta_real = noise_level * Zabs
            delta_imag = noise_level * Zabs

            noise_real = torch.randn_like(Z.real) * delta_real
            noise_imag = torch.randn_like(Z.imag) * delta_imag

            Z_obs = torch.complex(
                Z.real + noise_real,
                Z.imag + noise_imag
            )

            # -------- 非高斯：在部分点上叠加离群噪声 --------
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

            # -------- 由阻抗计算 ρ / φ --------
            rho_obs = torch.abs(Z_obs) ** 2 / (omega * MU)
            phs_obs = -torch.atan2(Z_obs.imag, Z_obs.real) * 180.0 / np.pi

            obs_data[f"rho{mode}"] = rho_obs
            obs_data[f"phs{mode}"] = phs_obs

            # 保存阻抗误差进入数组（供 calculate_data_errors_2d传播误差）
            self.data_std[f"delta_z{mode}_real"] = delta_real
            self.data_std[f"delta_z{mode}_imag"] = delta_imag
            self.data_std[f"Z{mode}"] = Z_obs

        self.obs_data = obs_data
        # 先进行误差传播，得到逐点噪声标准差
        self.calculate_data_errors_2d()
        # 再根据标准差构造数据权重
        self._compute_data_weights(noise_floor=self.noise_level)

        print("✓ 合成数据生成完成")
        print(f"  → 阻抗噪声水平: {noise_level*100:.1f}% ({noise_type})")
        if noise_type == "nongaussian":
            print(f"  → 非高斯: outlier_frac={outlier_frac}, outlier_strength={outlier_strength}")

    def calculate_data_errors_2d(self):
        """
        从阻抗误差传播得到 ρ 和 φ 的标准差（二维）
        并构造用于 χ² / RMS 的无量纲噪声标准差
        """
        eps = 1e-8

        omega = 2 * np.pi * self.freqs[:, None]
        MU = 4e-7 * np.pi  # 真空磁导率 (H/m)  

        # 存放所有模式的噪声标准差（用于反演）
        self.data_noise_std = {}

        for mode in ["xy", "yx"]:
            key_z = f"Z{mode}"
            key_rho = f"rho{mode}"
            key_phs = f"phs{mode}"

            # 只在对应阻抗已保存到 data_std 时进行误差传播
            if key_z not in self.data_std:
                continue  # 允许只用单一模式（如只算某一极化）

            Z = self.data_std[key_z]
            Zr = Z.real
            Zi = Z.imag
            Zabs = torch.abs(Z)

            delta_z_real = self.data_std[f"delta_z{mode}_real"]
            delta_z_imag = self.data_std[f"delta_z{mode}_imag"]

            # ---------- ρ 的标准差 ----------
            dRho_dZr = 2.0 * Zr / (omega * MU)
            dRho_dZi = 2.0 * Zi / (omega * MU)

            delta_rho = torch.sqrt(
                (dRho_dZr * delta_z_real) ** 2 +
                (dRho_dZi * delta_z_imag) ** 2
            )

            # ---------- φ 的标准差 ----------
            dPhi_dZr = -Zi / (Zabs ** 2)
            dPhi_dZi =  Zr / (Zabs ** 2)

            delta_phi_rad = torch.sqrt(
                (dPhi_dZr * delta_z_real) ** 2 +
                (dPhi_dZi * delta_z_imag) ** 2
            )

            delta_phs = delta_phi_rad * 180.0 / np.pi

            # ---------- 无量纲噪声标准差（用于 χ² / RMS） ----------
            # log10(ρ)
            rho_noise_std_log = torch.clamp(
                delta_rho / (self.obs_data[key_rho] * np.log(10)),
                min=eps
            )

            # φ / 90°
            phs_noise_std_norm = torch.clamp(
                delta_phs / 90.0,
                min=eps
            )

            # ---------- 保存 ----------
            self.data_noise_std[f"rho{mode}"] = rho_noise_std_log
            self.data_noise_std[f"phs{mode}"] = phs_noise_std_norm

            print(f"✓ {mode.upper()} 模式误差传播完成")
            print(f"   ρ(log10) 噪声均值: {rho_noise_std_log.mean():.4f}")
            print(f"   φ(归一化) 噪声均值: {phs_noise_std_norm.mean():.4f}")

        print("✓ 二维数据误差传播全部完成")

    def compute_rms_chi2(self, pred_dict):
        """
        计算统计意义上的 χ² RMS
        RMS ≈ 1 → 拟合到噪声水平
        """
        chi2_sum = 0.0
        n_data = 0

        for mode in ["xy", "yx"]:
            # ---------- ρ ----------
            key_rho = f"rho{mode}"
            if key_rho in self.obs_data:
                rho_pred_log = torch.log10(pred_dict[key_rho] + 1e-12)
                rho_obs_log  = torch.log10(self.obs_data[key_rho] + 1e-12)

                res_rho = (rho_pred_log - rho_obs_log) / self.data_noise_std[key_rho]
                chi2_sum += torch.sum(res_rho ** 2)
                n_data += res_rho.numel()

            # ---------- φ ----------
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
        构建归一化的 (N, 3) 点云: [Freq, Station, Value]
        所有维度归一化到 [0, 1]，这是纯 OT 稳定的物理基础。
        约定：传入的 ρ 为线性尺度，这里固定在 log10(ρ) 域归一化。
        """
        n_freq = len(self.freqs)
        n_stations = len(self.stations)
        
        # 1. 归一化频率 (Log域) -> [0, 1]
        log_freq = torch.log10(self.freqs)
        norm_freq = (log_freq - log_freq.min()) / (log_freq.max() - log_freq.min() + 1e-8)
        grid_freq = norm_freq.view(-1, 1).expand(n_freq, n_stations)
        
        # 2. 归一化测点 -> [0, 1]
        norm_stn = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_stn = norm_stn.view(1, -1).expand(n_freq, n_stations)
        
        # 3. 归一化数值 -> [0, 1] (至关重要)
        if 'rho' in key.lower():
            # 固定：在线性尺度上先取 log10，再按物理范围 [-2, 6] 归一化
            val_log = torch.log10(data_tensor + 1e-12)
            norm_val = (val_log - (-2.0)) / (6.0 - (-2.0))
        else:
            # 相位 0 ~ 90 度
            norm_val = data_tensor / 90.0
            
        # 4. 堆叠: (Batch, Points, Dim)
        points = torch.stack([grid_freq.flatten(), grid_stn.flatten(), norm_val.flatten()], dim=1)
        return points.unsqueeze(0)
    
    def _prepare_6d_ot_cloud(self, 
                              pred_dict: Dict[str, torch.Tensor], 
                              obs_dict: Dict[str, torch.Tensor]) -> tuple:
        """
        多变量OT：一次性考虑所有分量
        返回：一个点云，每个点是 [freq, station, rhoxy, phsxy, rhoyx, phsyx]
        """
        # 1. 基础维度
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        
        # 2. 归一化频率和站点
        log_f = torch.log10(self.freqs)
        norm_f = (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-8)
        grid_f = norm_f.view(-1, 1).expand(n_freq, n_stn).flatten()
        # 3. 归一化所有分量
        # 约定：传入的 ρ 为线性尺度，这里固定在 log10(ρ) 域归一化
        
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

        # 4. 构建多变量点云
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
    
    def _build_3d_ot_weights(self, key: str):
        """为 3D OT 点云构造权重 (α, β)。

        思路：
        - 使用 data_noise_std 中对应分量的噪声标准差 σ_ij；
        - 观测侧权重 β_ij ∝ 1 / (σ_ij^2 + eps)，σ 越大，权重越小；
        - 预测侧暂时使用均匀权重 α；
        - 再对 α, β 进行归一化，使两侧总质量均为 1（满足平衡 OT 的质量约束）。
        """
        eps = 1e-8

        noise_std = self.data_noise_std.get(key, None)
        if noise_std is None:
            # 若没有噪声信息，退化为均匀权重（源/目标一致）
            n_freq = len(self.freqs)
            n_stn = len(self.stations)
            n_pts = n_freq * n_stn
            w = torch.full((1, n_pts), 1.0 / n_pts, device=self.device, dtype=torch.float64)
            return w, w.clone()

        # (nf, ns) -> (N,)
        w = 1.0 / (noise_std.reshape(-1) ** 2 + eps)
        # 防止全 0 或 NaN
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(w <= 0):
            w = torch.ones_like(w)

        # 源/目标两侧质量：都按标准差加权并各自归一化到 1
        w = w / (w.sum() + eps)

        alpha = w.unsqueeze(0)
        beta = w.unsqueeze(0).clone()

        return alpha, beta

    def _build_6d_ot_weights(self):
        """为 6D OT 点云构造权重 (α, β)。

        使用四个分量的噪声：rhoxy, phsxy, rhoyx, phsyx。
        对每个 (freq, station) 组合：
        - 先对每个存在的分量计算精度 1/σ_k^2；
        - 将这些精度求和得到总精度 precision_ij；
        - 设 β_ij ∝ precision_ij，precision 越大（噪声越小）权重越大。
        """
        eps = 1e-8

        # 取出各分量的噪声标准差，注意有的模式可能缺失
        noise_rhoxy = self.data_noise_std.get("rhoxy", None)
        noise_phsxy = self.data_noise_std.get("phsxy", None)
        noise_rhoyx = self.data_noise_std.get("rhoyx", None)
        noise_phsyx = self.data_noise_std.get("phsyx", None)

        n_freq = len(self.freqs)
        n_stn = len(self.stations)

        precision = torch.zeros((n_freq, n_stn), device=self.device, dtype=torch.float64)

        def add_precision(noise_std: torch.Tensor):
            if noise_std is None:
                return
            inv_var = 1.0 / (noise_std ** 2 + eps)
            precision.add_(inv_var)

        add_precision(noise_rhoxy)
        add_precision(noise_phsxy)
        add_precision(noise_rhoyx)
        add_precision(noise_phsyx)

        w = precision.reshape(-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(w <= 0):
            w = torch.ones_like(w)

        # 源/目标两侧质量：都按综合精度加权并各自归一化到 1
        w = w / (w.sum() + eps)

        alpha = w.unsqueeze(0)
        beta = w.unsqueeze(0).clone()

        return alpha, beta
    
    def _init_sinkhorn(
        self,
        p: int = 2,
        blur: float = 0.01,
        scaling: float = 0.9,
        reach: Optional[float] = None,
        backend: str = "tensorized",
    ):
        """初始化 Sinkhorn OT 损失（geomloss 参数可配置）。

        参数通过 __init__ 中的 ot_options 传入，未提供时使用默认值。
        """
        try:
            # debias=True 对于高维特征匹配至关重要
            self.sinkhorn_loss = self.opt_config.create_sinkhorn_loss(
                p=p,
                blur=blur,
                scaling=scaling,
                reach=reach,
                debias=True,
                backend=backend,
            )
            print(
                f"✓ Sinkhorn OT Loss 初始化成功: "
                f"p={p}, blur={blur}, scale={scaling}, reach={reach}, backend={backend}"
            )
        except Exception as e:
            print(f"[Warning] Sinkhorn init failed: {e}")
            self.sinkhorn_loss = None

    def _compute_data_weights(self, noise_floor=0.01, error_floor=1e-3):
        """
        计算数据权重 (W_d)
        """
        self.data_weights = {}
        
        # 理论公式：相对误差 1% (0.01) 对应相位误差约为 0.286 度
        phase_error_deg = noise_floor * 28.6 
        
        # 给相位误差设置一个物理底限，防止过小导致权重爆炸
        # 真实的仪器误差通常很难小于 0.5 度
        if phase_error_deg < 0.5:
            phase_error_deg = 0.5
            
        print(f"正在计算数据权重 (Target Noise: {noise_floor*100:.1f}%)")
        print(f"  - Resistivity Error Floor: {noise_floor*100:.1f}%")
        print(f"  - Phase Error Floor:       {phase_error_deg:.3f} deg")
        
        for key, data in self.obs_data.items():
            # 默认权重为 1.0，若有误差标准差则按 1/sigma 构造加权 χ²
            if 'rho' in key.lower():
                # 对应 calculate_data_errors_2d 中 log10(ρ) 的标准差
                sigma_log = self.data_noise_std.get(key, None)
                if sigma_log is not None:
                    sigma_clamped = torch.clamp(sigma_log, min=error_floor)
                    w_tensor = 1.0 / sigma_clamped
                else:
                    # 回退到常数相对误差近似
                    sigma = noise_floor / 2.3026
                    w_tensor = torch.full_like(data, 1.0/(sigma + 1e-8), device=self.device)

            elif 'phs' in key.lower():
                # data_noise_std 中存的是 (φ/90) 的标准差
                sigma_norm = self.data_noise_std.get(key, None)
                if sigma_norm is not None:
                    # 在 MSE 中我们用的是度量残差 (deg)，
                    # 因此权重选用 1/(90 * sigma_norm)，
                    # 使得 (Δφ * weight)^2 ≈ ((Δφ/90)/sigma_norm)^2
                    sigma_eff = 90.0 * sigma_norm
                    sigma_clamped = torch.clamp(sigma_eff, min=error_floor)
                    w_tensor = 1.0 / sigma_clamped
                else:
                    # 回退到经验相位误差常数
                    sigma = phase_error_deg
                    w_tensor = torch.full_like(data, 1.0/(sigma + 1e-8), device=self.device)

            else:
                w_tensor = torch.ones_like(data, device=self.device)

            self.data_weights[key] = w_tensor
    
    def initialize_model(self, initial_sigma: float = 1e-2, random_init: bool = False, sigma_min: float = 1e-3, sigma_max: float = 1):
        """
        初始化反演模型
        
        Args:
            initial_sigma: 初始电导率值（均匀背景，单位：S/m）。
            random_init: 是否使用随机生成模型（对数均匀分布 + 高斯平滑）。
            sigma_min: 随机初始化时的电导率下限 (S/m)。
            sigma_max: 随机初始化时的电导率上限 (S/m)。
        """
        # 1. 计算网格中心物理坐标 (Physical Coordinates of Cell Centers)
        zn_tensor = self.zn.clone().detach().to(device=self.device, dtype=torch.float64)
        yn_tensor = self.yn.clone().detach().to(device=self.device, dtype=torch.float64)
        z_centers = (zn_tensor[:-1] + zn_tensor[1:]) / 2.0
        y_centers = (yn_tensor[:-1] + yn_tensor[1:]) / 2.0
        
        # 计算模型参数的实际尺寸（用于尺寸检查）
        nz_model = len(z_centers)
        ny_model = len(y_centers)
        
        # 如果 self.nz 和 self.ny 已定义，检查一致性
        if hasattr(self, 'nz') and hasattr(self, 'ny'):
            if nz_model != self.nz - 1 or ny_model != self.ny - 1:
                print(f"警告: 模型尺寸 ({nz_model}, {ny_model}) 与预期尺寸 ({self.nz-1}, {self.ny-1}) 不一致")
        
        if random_init:
            # 1) 在 log10(σ) 空间做均匀采样
            log_min = np.log10(sigma_min)
            log_max = np.log10(sigma_max)
            log_sigma = torch.rand((nz_model, ny_model), device=self.device, dtype=torch.float64)
            log_sigma = log_min + (log_max - log_min) * log_sigma

            # 2) 在 log10(σ) 空间做空间高斯平滑
            #    这样既保持“对数域随机”，又引入一定空间相关性
            log_sigma_np = log_sigma.cpu().numpy()
            log_sigma_smooth = gaussian_filter(log_sigma_np, sigma=1.0)

            log_sigma_smooth = np.clip(log_sigma_smooth, log_min, log_max)

            log_sigma_smooth = torch.tensor(log_sigma_smooth, device=self.device, dtype=torch.float64)
            sigma_init = 10 ** log_sigma_smooth
        else:
            # 默认的均匀初始化
            sigma_init = torch.ones((nz_model, ny_model), device=self.device, dtype=torch.float64) * initial_sigma
    
        # 保存初始化时的模型（用于后续绘图对比）
        self.initial_model_sigma = sigma_init.detach().clone()
    
        # 初始化参数：使用自然对数 (Natural Log) 作为反演参数是 PyTorch 里的标准操作
        # 它可以保证电导率永远为正数
        self.model_log_sigma = nn.Parameter(torch.log(sigma_init)) 
        self.model_log_sigma.requires_grad = True
        
        # 打印初始化信息
        if random_init:
            init_type = "Random (Log-Uniform + Smooth)"
        else:
            init_type = "Uniform"
        print(f"✓ 模型初始化完成: {init_type} initialization.")
    
    def set_reference_model(self, sig_ref: torch.Tensor = None):
        """设置参考模型（用于约束反演）

        Args:
            sig_ref: 参考模型电导率 [nz-1, ny-1]，应与模型网格尺寸一致。
                     若为 None，则使用均匀电导率 0.01 S/m 作为默认参考模型。
        """

        # 若未提供参考模型，则使用默认均匀模型 sigma = 0.01 S/m
        if sig_ref is None:
            sig_ref = torch.full(
                (self.nz - 1, self.ny - 1),
                0.01,
                device=self.device,
                dtype=torch.float64,
            )
        else:
            if sig_ref.shape != (self.nz - 1, self.ny - 1):
                raise ValueError(
                    f"参考模型尺寸 {sig_ref.shape} 与模型网格尺寸 {(self.nz-1, self.ny-1)} 不匹配"
                )

            # 确保参考模型在正确的设备上
            sig_ref = sig_ref.to(self.device, dtype=torch.float64)

        self.sig_ref = sig_ref

        # 计算参考模型的对数电导率
        self.model_log_sigma_ref = torch.log(self.sig_ref)

        print(f"✓ 参考模型已设置")
        print(f"   - 电导率范围: {self.sig_ref.min().item():.6e} - {self.sig_ref.max().item():.6e} S/m")
        print(f"   - 电阻率范围: {1/self.sig_ref.max().item():.2f} - {1/self.sig_ref.min().item():.2f} Ω·m")
    
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
        根据数据项与模型项的梯度量级，自适应更新 lambda（改进版）
        
        改进点：
        1. 使用移动平均平滑梯度范数，避免单次噪声误判
        2. 保留指数下降机制（效果良好）
        3. 添加安全机制，避免ratio过小时过度下降
        
        约束：lambda 只允许下降（逐步放松正则）
        目标（软约束）：||∇Φ_d|| ≲ λ ||∇Φ_m||
        
        Args:
            window_size: 移动平均窗口大小（用于平滑单次梯度噪声）
            min_ratio_for_update: ratio小于此值时不更新lambda（避免过度下降）
        """

        # ----------------------------
        # 1. 计算当前梯度范数
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
        
        # 记录原始值
        norm_d_item = norm_d_raw.item()
        norm_m_item = norm_m_raw.item()
        
        # ----------------------------
        # 2. 更新历史记录
        # ----------------------------
        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)
        
        # 保持历史记录在合理大小（避免内存无限增长）
        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        # ----------------------------
        # 3. 计算移动平均（平滑梯度范数）
        # ----------------------------
        if len(self.grad_norm_d_history) >= window_size:
            # 使用最近window_size个值的移动平均
            norm_d_smooth = np.mean(self.grad_norm_d_history[-window_size:])
            norm_m_smooth = np.mean(self.grad_norm_m_history[-window_size:])
        else:
            # 历史不足时，使用当前值
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        # ----------------------------
        # 4. 计算ratio（使用移动平均的梯度范数，已平滑）
        # ----------------------------
        # 直接使用移动平均的梯度范数计算ratio（已平滑，避免单次噪声）
        # 注意：不使用历史ratio统计，因为梯度是快速下降的曲线，历史统计不适用
        ratio = norm_d_smooth / (dr * current_lambda * norm_m_smooth + 1e-12)
        ratio = float(ratio)
        
        # 记录ratio历史（仅用于监控，不用于计算）
        self.ratio_history.append(ratio)

        # ----------------------------
        # 5. 只允许 lambda 下降（使用指数下降，效果良好）
        # ----------------------------
        if ratio < 1.0:
            # 需要放松正则
            if ratio < min_ratio_for_update:
                # ratio过小，不更新（避免过度下降）
                new_lambda = current_lambda
            else:
                # 使用指数下降（原始方法，效果良好）
                # 使用移动平均后的ratio，已平滑，避免单次噪声
                proposed_lambda = current_lambda * (ratio ** alpha)
                new_lambda = proposed_lambda
        else:
            # 梯度尚未失衡，保持 lambda
            new_lambda = current_lambda

        # ----------------------------
        # 6. 安全约束
        # ----------------------------
        new_lambda = float(np.clip(new_lambda, lambda_min, lambda_max))
        
        # 确保不会上升（双重保险）
        new_lambda = min(current_lambda, new_lambda)

        return new_lambda, norm_d_item, norm_m_item

    def _compute_losses(
        self,
        pred_dict: Dict[str, torch.Tensor],
        mode: str,
        norm_type: str,
        use_reference_model: bool,
        reference_weight: float,
        loss_data_ratio: float,
        num_data: int,
    ):
        """统一计算数据项与模型正则项损失，复用新版 MTinv_2d_cxz.py 的逻辑。"""

        # 数据项
        loss_data = torch.zeros((), device=self.device)
        with torch.no_grad():
            rms_chi2 = self.compute_rms_chi2(pred_dict)

        if mode == "6dot":
            cloud_pred, cloud_obs = self._prepare_6d_ot_cloud(pred_dict, self.obs_data)
            alpha, beta = self._build_6d_ot_weights()
            loss_data = loss_data_ratio * self.sinkhorn_loss(alpha, cloud_pred, beta, cloud_obs)
        elif mode == "3dot":
            for key in self.obs_data.keys():
                obs = self.obs_data[key]
                pred = pred_dict[key]

                cloud_obs = self._prepare_3d_ot_cloud(obs, key)
                cloud_pred = self._prepare_3d_ot_cloud(pred, key)
                alpha, beta = self._build_3d_ot_weights(key)
                loss_data += self.sinkhorn_loss(alpha, cloud_pred, beta, cloud_obs).sum()

            loss_data = loss_data_ratio * loss_data
        elif mode == "mse":
            for key in self.obs_data.keys():
                if "rho" in key.lower():
                    obs_val = torch.log10(self.obs_data[key] + 1e-12)
                    p_val = torch.log10(pred_dict[key] + 1e-12)
                else:
                    obs_val = self.obs_data[key]
                    p_val = pred_dict[key]

                loss_data += torch.sum(
                    ((obs_val - p_val) * self.data_weights[key]) ** 2
                )
            loss_data = loss_data_ratio * loss_data / num_data
        else:
            raise ValueError(f"Unknown inversion mode: {mode}")

        # 模型正则项（支持参考模型约束）
        if use_reference_model and self.model_log_sigma_ref is not None:
            loss_model = self.constraint_calc.calculate_combined_constraint(
                model_log_sigma=self.model_log_sigma,
                reference_model_log_sigma=self.model_log_sigma_ref,
                roughness_norm=norm_type,
                reference_norm=norm_type,
                reference_weight=reference_weight,
            )
        else:
            loss_model = self.constraint_calc.calculate_weighted_roughness(
                self.model_log_sigma, norm_type=norm_type
            )

        return loss_data, loss_model, rms_chi2

    def run_inversion(self, 
                    n_epochs: int = 100, 
                    mode: str = "6dot",
                    progress_interval: int = 10,
                    current_lambda: float = 0.01,
                    use_adaptive_lambda: bool = True,
                    loss_data_ratio: float = 50.0,
                    lr: float = 0.05,
                    use_lbfgs_tail: bool = False,
                    lbfgs_epochs: int = 10,
                    lbfgs_lr: float = 0.5,
                    dr: float = 2.0,
                    norm_type = "L2",
                    warmup_epochs: int = 5,
                    use_reference_model: bool = False,
                    reference_weight: float = 0.1 ): 
        """
        运行反演
        
        Args:
            n_epochs: 迭代次数
            mode: 反演模式 ('6dot' / '3dot' / 'mse')
            progress_interval: 进度显示间隔
            current_lambda: 正则化参数 λ
            use_lbfgs_tail: 是否在最后若干轮改用 LBFGS 优化
            lbfgs_epochs: 使用 LBFGS 的轮数（从最后往前数）
            lbfgs_lr: LBFGS 的学习率（步长因子）
        """
        if self.forward_operator is None:
            raise RuntimeError("请先设置正演算子")
        
        # 开始计时
        total_start_time = time.time()
        self.time_stats['start_time'] = total_start_time
        self.time_stats['epoch_times'] = []
        
        # 预先计算数据点数
        num_data = sum(v.numel() for v in self.obs_data.values())

        # 总迭代轮数拆分为 AdamW 阶段 + LBFGS 尾阶段
        if use_lbfgs_tail and lbfgs_epochs > 0:
            lbfgs_epochs_eff = min(lbfgs_epochs, n_epochs)
        else:
            lbfgs_epochs_eff = 0
        adam_epochs = n_epochs - lbfgs_epochs_eff

        optimizer = self.opt_config.create_optimizer(
            [self.model_log_sigma], lr=lr, optimizer_type="AdamW"
        )

        # 前若干个 epoch 固定正则参数，不做自适应更新
        warmup_epochs = warmup_epochs if use_adaptive_lambda else 0

        # =========================
        # 阶段一：AdamW 优化
        # =========================
        for epoch in range(adam_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            
            # 1. 正演
            sigma = torch.exp(self.model_log_sigma)
            pred_dict = self.forward_operator(sigma)
            
            # 2. Data loss & Model loss（统一函数计算）
            loss_data, loss_model, rms_chi2 = self._compute_losses(
                pred_dict=pred_dict,
                mode=mode,
                norm_type=norm_type,
                use_reference_model=use_reference_model,
                reference_weight=reference_weight,
                loss_data_ratio=loss_data_ratio,
                num_data=num_data,
            )
            
            # 4. 反传
            if use_adaptive_lambda:
                if epoch < warmup_epochs:
                    # 预热阶段：只计算梯度范数，保持 current_lambda 不变
                    _, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                        loss_data, loss_model, current_lambda, dr=dr
                    )
                else:
                    # 之后才真正自适应更新 lambda
                    current_lambda, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                        loss_data, loss_model, current_lambda, dr=dr
                    )
            else:
                # 关闭自适应时，仅计算梯度范数用于监控，lambda 保持初始值
                _, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                    loss_data, loss_model, current_lambda, dr=dr
                )
            total_loss = loss_data + current_lambda * loss_model
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([self.model_log_sigma], 1.0)
            optimizer.step()
            
            with torch.no_grad():
                self.model_log_sigma.clamp_(min=-11.5, max=4.6)
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            self.time_stats['epoch_times'].append(epoch_time)
            
            # 进度显示
            if epoch % progress_interval == 0 or epoch == n_epochs - 1:
                # 计算统计信息
                elapsed_time = time.time() - total_start_time
                avg_epoch_time = np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else epoch_time
                remaining_epochs = n_epochs - epoch - 1
                remaining_time = avg_epoch_time * remaining_epochs
                progress_percent = (epoch + 1) / n_epochs * 100
                
                # 格式化时间
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                remaining_str = str(timedelta(seconds=int(remaining_time)))
                eta_time = datetime.now() + timedelta(seconds=remaining_time)
                eta_str = eta_time.strftime("%H:%M:%S")
                
                print(f"Epoch {epoch:03d}/{n_epochs} [{progress_percent:5.1f}%]")
                print(f"  已用时间: {elapsed_str} | 剩余时间: ~{remaining_str} | ETA: {eta_str}")
                print(f"  Epoch耗时: {epoch_time:.2f}s | 平均耗时: {avg_epoch_time:.2f}s")
                print(f"  Total: {total_loss.item():.4e} | Data({mode}): {loss_data.item():.4e}")
                print(f"  Misfit(RMS χ²): {rms_chi2:.3f} | Rough: {loss_model.item():.2e} | Lam: {current_lambda:.7f}")
                print(f"  GradNorms: |g_d|={g_d_norm:.3e} | |g_m|={g_m_norm:.3e}")
                print("-" * 80)
            
            # 记录损失历史
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

        # =========================
        # 阶段二：LBFGS 尾阶段（可选）
        # =========================
        if lbfgs_epochs_eff > 0:
            print(f"\n切换到 LBFGS 进行最后 {lbfgs_epochs_eff} 轮优化...")

            lbfgs_optimizer = torch.optim.LBFGS(
                [self.model_log_sigma],
                lr=lbfgs_lr,
                max_iter=5,
                history_size=10,
                line_search_fn='strong_wolfe'
            )

            for k in range(lbfgs_epochs_eff):
                epoch = adam_epochs + k
                epoch_start_time = time.time()

                lbfgs_cache = {}

                def closure():
                    lbfgs_optimizer.zero_grad()

                    sigma = torch.exp(self.model_log_sigma)
                    pred_dict = self.forward_operator(sigma)
                    
                    # Data & Model loss（与 Adam 阶段共用同一实现）
                    loss_data, loss_model, rms_chi2 = self._compute_losses(
                        pred_dict=pred_dict,
                        mode=mode,
                        norm_type=norm_type,
                        use_reference_model=use_reference_model,
                        reference_weight=reference_weight,
                        loss_data_ratio=loss_data_ratio,
                        num_data=num_data,
                    )

                    # 仅用于记录梯度范数，不更新 lambda
                    _, g_d_norm, g_m_norm = self.update_lambda_by_gradient_balance(
                        loss_data, loss_model, current_lambda, dr=dr
                    )

                    total_loss = loss_data + current_lambda * loss_model
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_([self.model_log_sigma], 1.0)

                    with torch.no_grad():
                        self.model_log_sigma.clamp_(min=-11.5, max=4.6)

                    # 将当前轮次的信息缓存出来，供外层记录和打印
                    lbfgs_cache['loss_data'] = loss_data.item()
                    lbfgs_cache['loss_model'] = loss_model.item()
                    lbfgs_cache['total_loss'] = total_loss.item()
                    lbfgs_cache['rms_chi2'] = rms_chi2
                    lbfgs_cache['g_d_norm'] = g_d_norm
                    lbfgs_cache['g_m_norm'] = g_m_norm

                    return total_loss

                # 执行一次 LBFGS 更新（内部会多次调用 closure）
                lbfgs_optimizer.step(closure)

                # 记录时间
                epoch_time = time.time() - epoch_start_time
                self.time_stats['epoch_times'].append(epoch_time)

                # 进度显示
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

                    print(f"Epoch {epoch:03d}/{n_epochs} [{progress_percent:5.1f}%] (LBFGS)")
                    print(f"  已用时间: {elapsed_str} | 剩余时间: ~{remaining_str} | ETA: {eta_str}")
                    print(f"  Epoch耗时: {epoch_time:.2f}s | 平均耗时: {avg_epoch_time:.2f}s")
                    print(f"  Total: {lbfgs_cache['total_loss']:.4e} | Data({mode}): {lbfgs_cache['loss_data']:.4e}")
                    print(f"  Misfit(RMS χ²): {lbfgs_cache['rms_chi2']:.3f} | Rough: {lbfgs_cache['loss_model']:.2e} | Lam: {current_lambda:.7f}")
                    print(f"  GradNorms: |g_d|={lbfgs_cache['g_d_norm']:.3e} | |g_m|={lbfgs_cache['g_m_norm']:.3e}")
                    print("-" * 80)

                # 记录损失历史
                self.loss_history.append({
                    'epoch': epoch,
                    'total_loss': lbfgs_cache['total_loss'],
                    'data_loss': lbfgs_cache['loss_data'],
                    'model_loss': lbfgs_cache['loss_model'],
                    'misfit': lbfgs_cache['rms_chi2'],
                    'lambda': current_lambda,
                    'epoch_time': epoch_time,
                    'grad_data_norm': lbfgs_cache['g_d_norm'],
                    'grad_model_norm': lbfgs_cache['g_m_norm']
                })

        # 结束计时
        total_end_time = time.time()
        total_inversion_time = total_end_time - total_start_time
        
        # 更新计时统计
        self.time_stats.update({
            'end_time': total_end_time,
            'total_inversion_time': total_inversion_time,
            'avg_epoch_time': np.mean(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'min_epoch_time': np.min(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'max_epoch_time': np.max(self.time_stats['epoch_times']) if self.time_stats['epoch_times'] else 0,
            'std_epoch_time': np.std(self.time_stats['epoch_times']) if len(self.time_stats['epoch_times']) > 1 else 0,
        })
        
        print("反演完成.")
        return torch.exp(self.model_log_sigma).detach()

    def plot_true_vs_inverted_model(
            self,
            use_resistivity: bool = True,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],     # x轴的左右边界
            ylim: list = [50, 0]       # y轴的上下边界
        ):
        """
        绘制真实模型 vs 反演模型对比图（log10 坐标，屏蔽空气层，并根据指定边界应用掩膜）
        """
        # -------- 模型取值 --------
        sigma_true = self.sig_true.detach().cpu().numpy()
        sigma_inv = torch.exp(self.model_log_sigma).detach().cpu().numpy()
    
        try:
            data_range = np.log10(sigma_true.max()) - np.log10(sigma_true.min())
            score = ssim(np.log10(sigma_true), np.log10(sigma_inv), data_range=data_range, win_size=3)
            print(f"模型结构相似度 (SSIM): {score:.4f}")
        except:
            pass
    
        eps = 1e-12
    
        if use_resistivity:
            model_true = np.log10(1.0 / (sigma_true + eps))
            model_inv = np.log10(1.0 / (sigma_inv + eps))
            label = r"log$_{10}$ Resistivity (Ω·m)"
            title_true = "True log10 Resistivity"
            title_inv = "Inverted log10 Resistivity"
        else:
            model_true = np.log10(sigma_true + eps)
            model_inv = np.log10(sigma_inv + eps)
            label = r"log$_{10}$ Conductivity (S/m)"
            title_true = "True log10 Conductivity"
            title_inv = "Inverted log10 Conductivity"
    
        # -------- 屏蔽空气层 (z < 0) --------
        # zn 是节点，取单元中心判断
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])  # (nz,)
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        mask_air = ZZ < 0  # 屏蔽空气层
        mask_ground = ZZ >= 0  # 只保留地面以上部分
    
        # -------- 创建掩膜 (根据 xlim 和 ylim 参数) --------
        x_min, x_max = min(xlim), max(xlim)
        y_bottom, y_top = max(ylim), min(ylim)  # 上下边界：深度小于等于深界，大于等于浅界
    
        mask_view = (YY >= x_min) & (YY <= x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
    
        # 应用掩膜
        model_true_masked = np.ma.masked_where(~mask_view, model_true)
        model_inv_masked = np.ma.masked_where(~mask_view, model_inv)
    
        # 获取掩膜区域内的最小和最大值，用于设置 colorbar 范围
        vmin = model_true_masked.min() - 0.5
        vmax = model_true_masked.max() + 0.5
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
        # --- 真实模型 ---
        im1 = ax1.pcolormesh(YY, ZZ, model_true_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax1.invert_yaxis()
        ax1.set_title(title_true)
        ax1.set_ylabel('Depth (km)')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        plt.colorbar(im1, ax=ax1, label=label)
    
        # --- 反演模型 ---
        im2 = ax2.pcolormesh(YY, ZZ, model_inv_masked, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax2.invert_yaxis()
        ax2.set_title(title_inv)
        ax2.set_ylabel('Depth (km)')
        ax2.set_xlabel('Distance (km)')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        plt.colorbar(im2, ax=ax2, label=label)
    
        # 标出测点位置
        st_y = self.stations.cpu().numpy() / 1000.0
        ax2.scatter(st_y, np.zeros_like(st_y), c='k', s=10, marker='v', label='Stations')
    
        plt.tight_layout()
        plt.show()

    def plot_initial_model(
            self,
            use_resistivity: bool = True,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],
            ylim: list = [50, 0]
        ):
        """绘制初始模型（log10 坐标，屏蔽空气层，并根据指定边界应用掩膜）"""
        if self.initial_model_sigma is None:
            print("初始模型尚未保存，请先调用 initialize_model 或 run_inversion。")
            return

        # -------- 模型取值 --------
        sigma_init = self.initial_model_sigma.detach().cpu().numpy()

        eps = 1e-12

        if use_resistivity:
            model_init = np.log10(1.0 / (sigma_init + eps))
            label = r"log$_{10}$ Resistivity (Ω·m)"
            title_init = "Initial log10 Resistivity"
        else:
            model_init = np.log10(sigma_init + eps)
            label = r"log$_{10}$ Conductivity (S/m)"
            title_init = "Initial log10 Conductivity"

        # -------- 屏蔽空气层 (z < 0) --------
        zc = 0.001 * 0.5 * (self.zn[:-1] + self.zn[1:])  # (nz,)
        yc = 0.001 * 0.5 * (self.yn[:-1] + self.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        mask_air = ZZ < 0

        # -------- 创建掩膜 (根据 xlim 和 ylim 参数) --------
        x_min, x_max = min(xlim), max(xlim)
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
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.colorbar(im, ax=ax, label=label)

        # 标出测点位置
        st_y = self.stations.cpu().numpy() / 1000.0
        ax.scatter(st_y, np.zeros_like(st_y), c='k', s=10, marker='v', label='Stations')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_loss_history(self, target_misfit: float = 1.0):
        """
        绘制反演过程中的各项损失和参数演化曲线
        """
        if not self.loss_history:
            print("没有找到损失历史记录，请先运行 run_inversion。")
            return

        # 提取数据
        epochs = [log['epoch'] for log in self.loss_history]
        misfit = [log['misfit'] for log in self.loss_history]
        lambdas = [log['lambda'] for log in self.loss_history]
        data_loss = [log['data_loss'] for log in self.loss_history]
        model_loss = [log['model_loss'] for log in self.loss_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- 图 1: Misfit (RMS) ---
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='Current RMS')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMS Error")
        axes[0].set_yscale('log')  # 通常 RMS 下降跨度大，建议用 log
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].legend()

        # --- 图 2: Data Loss vs Model Loss (Roughness) ---
        ax2_twin = axes[1].twinx()
        p1, = axes[1].plot(epochs, data_loss, 'c-', label='Data Loss')
        p2, = ax2_twin.plot(epochs, model_loss, 'm-', label='Model Roughness')
        
        axes[1].set_title("Loss Components Trade-off")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Data Loss", color='c')
        axes[1].set_yscale('log')
        ax2_twin.set_ylabel("Roughness (Model Loss)", color='m')
        
        # 合并图例
        axes[1].legend(handles=[p1, p2])
        axes[1].grid(True, alpha=0.3)

        # --- 图 3: Lambda  ---
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization Parameter (Lambda)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lambda Value")
        axes[2].set_yscale('log')  # Lambda 变化范围极大
        axes[2].grid(True, which="both", ls="-", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def plot_gradient_history(self):
        """绘制数据项和模型项梯度范数随迭代的变化曲线"""

        # 从第二轮开始画
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
    
    def plot_data_fitting(self, station_indices=None):
        """
        绘制指定台站的数据拟合对比图
        """
        with torch.no_grad():
            sigma_final = torch.exp(self.model_log_sigma)
            pred_dict = self.forward_operator(sigma_final)
    
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
    
                # 不等长误差棒（log 误差）
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
            ax_rho.tick_params(labelbottom=True)  # 强制显示频率刻度
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
        use_resistivity: bool = True,
        depth_limit_km: float = None
    ):
        """
        在指定位置绘制 1D 垂直剖面图
        """
        # 1. 模型数据
        sigma_inv = torch.exp(self.model_log_sigma).detach().cpu().numpy()
        sigma_true = (
            self.sig_true.detach().cpu().numpy()
            if self.sig_true is not None else None
        )

        # 2. 深度坐标
        zn_km = 0.001 * self.zn.cpu().numpy()
        zc_km = 0.5 * (zn_km[:-1] + zn_km[1:])
        cell_mask = zc_km >= 0
        edge_mask = zn_km >= 0
        zc_ground = zc_km[cell_mask]
        zn_ground = zn_km[edge_mask]

        # 3. 台站选择
        if station_indices is None:
            n_stations = len(self.stations)
            station_indices = [0, n_stations // 2, n_stations - 1]

        # 4. 统一横坐标范围
        all_vals = []

        y_centers = 0.5 * (self.yn[:-1] + self.yn[1:]).cpu().numpy()

        for st_idx in station_indices:
            col_idx = np.abs(
                y_centers - self.stations[st_idx].item()
            ).argmin()

            if use_resistivity:
                all_vals.append(1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12))
                if sigma_true is not None:
                    all_vals.append(1.0 / (sigma_true[cell_mask, col_idx] + 1e-12))
            else:
                all_vals.append(sigma_inv[cell_mask, col_idx])
                if sigma_true is not None:
                    all_vals.append(sigma_true[cell_mask, col_idx])

        all_vals = np.hstack(all_vals)
        valid = all_vals[all_vals > 0]

        xmin = valid.min() / 2
        xmax = valid.max() * 2

        # 5. 画图
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
            if use_resistivity:
                val_inv = 1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12)
                xlabel = "Resistivity ($\Omega \cdot m$)"
            else:
                val_inv = sigma_inv[cell_mask, col_idx]
                xlabel = "Conductivity (S/m)"

            # -------- True (edge-based, perfect blocks) --------
            if sigma_true is not None:
                if use_resistivity:
                    val_true_cell = 1.0 / (sigma_true[cell_mask, col_idx] + 1e-12)
                else:
                    val_true_cell = sigma_true[cell_mask, col_idx]

                # 将 cell 值扩展为边界阶梯
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
                label='Inverted'
            )

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
