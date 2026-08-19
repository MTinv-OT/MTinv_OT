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
        Compute model recovery metrics.

        Parameters
        ----------
        sigma_inv : torch.Tensor, optional
            Inverted conductivity model. If None, use the current model (self.get_sigma_full()).
        sig_true : torch.Tensor, optional
            True conductivity model. If None, use self.sig_true.
        yn : np.ndarray, optional
            y-direction grid nodes (m). If None, use self.yn.
        zn : np.ndarray, optional
            z-direction grid nodes (m). If None, use self.zn.
        nza : int, optional
            Number of air layers. If None, use self.nza.
        anomaly_y_range : tuple, optional
            Anomaly y-range (y_min, y_max) in meters. If given, also compute recovery inside the anomaly.
        anomaly_z_range : tuple, optional
            Anomaly z-range (z_min, z_max) in meters. If given, also compute recovery inside the anomaly.
        eps : float, default=1e-12
            Small value to avoid division by zero.

        Returns
        -------
        dict
            Keys:
            - rmse: root-mean-square error (log10 resistivity)
            - mape: mean absolute percentage error (%)
            - correlation: Pearson correlation coefficient
            - ssim: structural similarity index (NaN if skimage is not installed)
            - anomaly_rmse: RMSE inside the anomaly (NaN if anomaly_y_range/anomaly_z_range are not set)
            - anomaly_mape: MAPE inside the anomaly (NaN if not specified)
        """
        # 1) Use passed arguments or object attributes
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

        # 2) Extract the subsurface (drop air layers)
        sigma_inv_earth = sigma_inv[nza:, :].detach().cpu().numpy()
        sigma_true_earth = sig_true[nza:, :].detach().cpu().numpy()

        # 3) Convert to resistivity (log domain)
        rho_inv = 1.0 / (sigma_inv_earth + eps)
        rho_true = 1.0 / (sigma_true_earth + eps)
        log_rho_inv = np.log10(rho_inv)
        log_rho_true = np.log10(rho_true)

        # 4) Global recovery metrics
        # RMSE
        rmse = np.sqrt(np.mean((log_rho_inv - log_rho_true) ** 2))

        # MAPE (%)
        mape = np.mean(np.abs(log_rho_inv - log_rho_true) / (np.abs(log_rho_true) + eps)) * 100

        # Pearson correlation
        corr = np.corrcoef(log_rho_inv.flatten(), log_rho_true.flatten())[0, 1]

        # SSIM (optional; requires skimage)
        try:
            from skimage.metrics import structural_similarity as ssim
            data_range = log_rho_true.max() - log_rho_true.min()
            if data_range > eps:
                ssim_val = ssim(log_rho_inv, log_rho_true, data_range=data_range)
            else:
                ssim_val = np.nan
        except ImportError:
            ssim_val = np.nan

        # 5) Anomaly-region recovery (if ranges are given)
        anomaly_rmse, anomaly_mape = np.nan, np.nan
        if anomaly_y_range is not None and anomaly_z_range is not None:
            # Cell-center coordinates
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
