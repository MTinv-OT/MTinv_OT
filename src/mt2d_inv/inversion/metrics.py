"""Inversion quality metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch



class InversionMetricsMixin:
    def compute_recovery_rate(self, 
                              sigma_inv: Optional[torch.Tensor] = None,
                              sig_true: Optional[torch.Tensor] = None,
                              yn: Optional[np.ndarray] = None,
                              zn: Optional[np.ndarray] = None,
                              nza: Optional[int] = None,
                              anomaly_y_range: Optional[tuple] = None,
                              anomaly_z_range: Optional[tuple] = None,
                              eps: float = 1e-12) -> Dict[str, float]:
        """
        计算模型的恢复率指标。

        Parameters
        ----------
        sigma_inv : torch.Tensor, optional
            反演得到的 conductivity 模型。若为 None，则使用当前模型 (self.get_sigma_full())。
        sig_true : torch.Tensor, optional
            真实 conductivity 模型。若为 None，则使用 self.sig_true。
        yn : np.ndarray, optional
            y 方向网格节点（米）。若为 None，则使用 self.yn。
        zn : np.ndarray, optional
            z 方向网格节点（米）。若为 None，则使用 self.zn。
        nza : int, optional
            空气层数。若为 None，则使用 self.nza。
        anomaly_y_range : tuple, optional
            异常体的 y 范围 (y_min, y_max)，单位：米。若提供，则计算异常体区域的恢复率。
        anomaly_z_range : tuple, optional
            异常体的 z 范围 (z_min, z_max)，单位：米。若提供，则计算异常体区域的恢复率。
        eps : float, default=1e-12
            防止除零的小量。

        Returns
        -------
        dict
            包含以下键值：
            - rmse: 均方根误差 (log10 电阻率)
            - mape: 平均绝对百分比误差 (%)
            - correlation: 皮尔逊相关系数
            - ssim: 结构相似性指数 (若 skimage 未安装则为 NaN)
            - anomaly_rmse: 异常体区域的 RMSE (若未指定 anomaly_y_range/anomaly_z_range 则为 NaN)
            - anomaly_mape: 异常体区域的 MAPE (若未指定则为 NaN)
        """
        # 1) 使用传入参数或对象属性
        if sigma_inv is None:
            sigma_inv = self.get_sigma_full()
        if sig_true is None:
            sig_true = getattr(self, 'sig_true', None)
        if sig_true is None:
            raise ValueError("sig_true not set in inverter. Please set inv.sig_true first.")
        if yn is None:
            yn = self.yn.cpu().numpy() if torch.is_tensor(self.yn) else self.yn
        if zn is None:
            zn = self.zn.cpu().numpy() if torch.is_tensor(self.zn) else self.zn
        if nza is None:
            nza = self.nza

        # 2) 提取地下部分（去掉空气层）
        sigma_inv_earth = sigma_inv[nza:, :].detach().cpu().numpy()
        sigma_true_earth = sig_true[nza:, :].detach().cpu().numpy()

        # 3) 转换为电阻率（对数域）
        rho_inv = 1.0 / (sigma_inv_earth + eps)
        rho_true = 1.0 / (sigma_true_earth + eps)
        log_rho_inv = np.log10(rho_inv)
        log_rho_true = np.log10(rho_true)

        # 4) 全局恢复率指标
        # RMSE
        rmse = np.sqrt(np.mean((log_rho_inv - log_rho_true) ** 2))

        # MAPE (%)
        mape = np.mean(np.abs(log_rho_inv - log_rho_true) / (np.abs(log_rho_true) + eps)) * 100

        # 皮尔逊相关系数
        corr = np.corrcoef(log_rho_inv.flatten(), log_rho_true.flatten())[0, 1]

        # SSIM (可选，需要 skimage)
        try:
            from skimage.metrics import structural_similarity as ssim
            data_range = log_rho_true.max() - log_rho_true.min()
            if data_range > eps:
                ssim_val = ssim(log_rho_inv, log_rho_true, data_range=data_range)
            else:
                ssim_val = np.nan
        except ImportError:
            ssim_val = np.nan

        # 5) 异常体区域恢复率（如果指定了范围）
        anomaly_rmse, anomaly_mape = np.nan, np.nan
        if anomaly_y_range is not None and anomaly_z_range is not None:
            # 计算网格中心坐标
            y_centers = 0.5 * (yn[:-1] + yn[1:])
            z_centers = 0.5 * (zn[nza:][:-1] + zn[nza:][1:]) if len(zn) > nza + 1 else None

            if z_centers is not None and len(z_centers) > 0:
                y_grid, z_grid = np.meshgrid(y_centers, z_centers)
                y_min, y_max = anomaly_y_range
                z_min, z_max = anomaly_z_range
                anomaly_mask = (y_grid >= y_min) & (y_grid <= y_max) & \
                               (z_grid >= z_min) & (z_grid <= z_max)

                if np.any(anomaly_mask):
                    inv_anomaly = log_rho_inv[anomaly_mask]
                    true_anomaly = log_rho_true[anomaly_mask]
                    anomaly_rmse = np.sqrt(np.mean((inv_anomaly - true_anomaly) ** 2))
                    anomaly_mape = np.mean(np.abs(inv_anomaly - true_anomaly) / (np.abs(true_anomaly) + eps)) * 100

        return {
            "rmse": rmse,
            "mape": mape,
            "correlation": corr,
            "ssim": ssim_val,
            "anomaly_rmse": anomaly_rmse,
            "anomaly_mape": anomaly_mape,
        }
