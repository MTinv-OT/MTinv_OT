"""OT point clouds and Sinkhorn setup."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch



class InversionOTMixin:
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
        
        obs_raw = self.obs_data[key]
        valid_mask = ~torch.isnan(obs_raw.flatten())
        n_valid = valid_mask.sum().item()
        
        noise_std = self.get_effective_cloud_noise_std(key, eps=eps)
        
        if noise_std is None:
            alpha = torch.full((1, n_valid), 1.0 / n_valid, device=self.device, dtype=torch.float64)
            beta = alpha.clone()
            return alpha, beta

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

