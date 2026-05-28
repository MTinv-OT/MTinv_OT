"""
26/3/15 by cxz
加了逐点权重w_d的支持，构建了make_weighted_cost_fn函数来生成加权的OT成本函数，并在MT2DInverterWeightedCost类中使用这个加权成本函数初始化Sinkhorn OT Loss。同时提供了compute_w_d_per_point_from_noise方法根据观测数据的噪声水平计算逐点权重。
"""
import torch
import warnings
from typing import Optional, Union, List, Tuple

from .MTinv_2d import MT2DInverter


def make_weighted_cost_fn(
    w_s: float = 1.0,
    w_f: float = 1.0,
    w_d: Union[float, List[float], Tuple[float, ...], torch.Tensor] = 1.0,
):
    """
    Build weighted cost C_ij = w_s*(Δs)² + w_f*(Δlog f)² + sum_k w_d_k*(Δdata_k)².

    w_d supports:
        - scalar: same weight for all 4 data dims
        - sequence of 4: (rhoxy, phsxy, rhoyx, phsyx)
        - torch.Tensor for per-point weights:
            * shape (4,)                 -> same as sequence of 4
            * shape (M, 4) or (4, M)     -> per-observation-point weights for each dim
              where M = number of points (= n_freq * n_station)
    """

    w_d_tensor: Optional[torch.Tensor] = None
    if torch.is_tensor(w_d):
        w_d_tensor = w_d
    elif isinstance(w_d, (list, tuple)):
        w_d_arr = tuple(float(x) for x in w_d)
        if len(w_d_arr) != 4:
            raise ValueError("w_d must be scalar, tensor, or sequence of length 4")
        w_d_tensor = torch.tensor(w_d_arr, dtype=torch.float64)
    else:
        w_d_arr = (float(w_d),) * 4
        w_d_tensor = torch.tensor(w_d_arr, dtype=torch.float64)

    # Detach to ensure no autograd tracking for weights.
    # IMPORTANT: do not force weights onto CPU, otherwise Sinkhorn iterations would
    # repeatedly transfer a potentially large (M,4) tensor to GPU.
    w_d_tensor = w_d_tensor.detach()

    _w_cache = {}

    def _coerce_w_d_for_y(w: torch.Tensor, M: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return weights with shape (1, 1, M, K) for broadcasting (cached)."""
        cache_key = (str(device), str(dtype), int(M), tuple(w.shape), int(w.ndim))
        cached = _w_cache.get(cache_key, None)
        if cached is not None:
            return cached

        w = w.to(device=device, dtype=dtype)
        if w.ndim == 1:
            if w.numel() != 4:
                raise ValueError(f"w_d tensor must have 4 elements, got {w.numel()}")
            # (K,) -> (1,1,1,K)
            out = w.view(1, 1, 1, -1)
            _w_cache[cache_key] = out
            return out
        if w.ndim == 2:
            if w.shape == (M, 4):
                out = w.view(1, 1, M, 4)
                _w_cache[cache_key] = out
                return out
            if w.shape == (4, M):
                out = w.transpose(0, 1).contiguous().view(1, 1, M, 4)
                _w_cache[cache_key] = out
                return out
            raise ValueError(f"w_d tensor must be shaped (M,4) or (4,M); got {tuple(w.shape)} (M={M})")
        raise ValueError(f"w_d tensor must be 1D or 2D, got ndim={w.ndim}")

    def cost_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), y: (B, M, D)
        B, N, D = x.shape
        M = y.shape[1]

        # (B, N, M)
        d_freq = (x[:, :, 0].unsqueeze(2) - y[:, :, 0].unsqueeze(1)).pow(2)
        d_stn = (x[:, :, 1].unsqueeze(2) - y[:, :, 1].unsqueeze(1)).pow(2)

        K = min(4, max(0, D - 2))
        if K == 0:
            d_data = torch.zeros((B, N, M), device=x.device, dtype=x.dtype)
        else:
            x_data = x[:, :, 2:2 + K].unsqueeze(2)      # (B, N, 1, K)
            y_data = y[:, :, 2:2 + K].unsqueeze(1)      # (B, 1, M, K)
            d = (x_data - y_data).pow(2)                # (B, N, M, K)
            w_y = _coerce_w_d_for_y(w_d_tensor, M=M, dtype=x.dtype, device=x.device)[..., :K]
            d_data = (d * w_y).sum(dim=-1)

        return (w_f * d_freq) + (w_s * d_stn) + d_data
    return cost_fn


class MT2DInverterWeightedCost(MT2DInverter):
    """
    MT2D inverter with weighted cost C_ij = w_s*(Δs)² + w_f*(Δlog f)² + sum_k w_d_k*(Δdata_k)².

    ot_options:
        w_s, w_f, w_d: weights (float or for w_d: length-4 list [w_rhoxy, w_phsxy, w_rhoyx, w_phsyx]).
    """

    def __init__(self, **kwargs):
        ot_options = kwargs.get("ot_options") or {}
        ot_options.setdefault("w_s", 1.0)
        ot_options.setdefault("w_f", 1.0)
        ot_options.setdefault("w_d", 1.0)
        kwargs["ot_options"] = ot_options
        super().__init__(**kwargs)

    def _init_sinkhorn(
        self,
        p: int,
        blur: float,
        scaling: float,
        reach: Optional[float],
        backend: str,
        w_s: float = 1.0,
        w_f: float = 1.0,
        w_d: Union[float, List[float], Tuple[float, ...], torch.Tensor] = 1.0,
        **kwargs,
    ):
        """Init Sinkhorn with custom weighted cost."""
        if w_s == "auto" or w_f == "auto" or w_d == "auto":
            raise ValueError(
                'w_s, w_f, w_d="auto" is deprecated. Provide explicit values, e.g. ot_options={"w_s": 1.0, "w_f": 1.0, "w_d": 1.0}'
            )
        w_d_norm = self._normalize_w_d(w_d)
        self._cost_weights = {"w_s": float(w_s), "w_f": float(w_f), "w_d": w_d_norm}
        cost_fn = make_weighted_cost_fn(w_s=float(w_s), w_f=float(w_f), w_d=w_d_norm)
        try:
            self.sinkhorn_loss = self.opt_config.create_sinkhorn_loss_with_cost(
                cost=cost_fn,
                blur=blur,
                scaling=scaling,
                reach=reach,
                debias=True,
                backend=backend
            )
            if torch.is_tensor(w_d_norm):
                w_d_desc = f"tensor{tuple(w_d_norm.shape)}@{w_d_norm.device.type}"
            else:
                w_d_desc = str(w_d_norm)
            print(
                f"✓ Sinkhorn OT Loss (weighted): w_s={w_s}, w_f={w_f}, w_d={w_d_desc}, "
                f"blur={blur:.4f}, scale={scaling}, reach={reach}"
            )
        except Exception as e:
            print(f"[Warning] Sinkhorn init failed: {e}")
            self.sinkhorn_loss = None

    def _normalize_w_d(
        self,
        w_d: Union[float, List[float], Tuple[float, ...], torch.Tensor],
    ) -> Union[Tuple[float, ...], torch.Tensor]:
        """Normalize w_d input.

        Returns:
            - tuple(4,) of floats for scalar/list/tuple
            - torch.Tensor unchanged for per-point weights
        """
        if torch.is_tensor(w_d):
            return w_d
        if isinstance(w_d, (list, tuple)):
            w = tuple(float(x) for x in w_d)
            if len(w) != 4:
                raise ValueError("w_d must be scalar, tensor, or sequence of length 4")
            return w
        return (float(w_d),) * 4

    def _prepare_6d_ot_cloud_obs(self, obs_dict):
        """6D observation point cloud; mask out NaN so OT receives only valid points."""
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
            if "rho" in key.lower():
                val_log = torch.log10(data_flat + 1e-12)
                return (val_log - (-2.0)) / (6.0 - (-2.0))
            return data_flat / 90.0

        obs_rhoxy = _norm_obs("rhoxy", obs_dict["rhoxy"])
        obs_phsxy = _norm_obs("phsxy", obs_dict["phsxy"])
        obs_rhoyx = _norm_obs("rhoyx", obs_dict["rhoyx"])
        obs_phsyx = _norm_obs("phsyx", obs_dict["phsyx"])
        obs_points = torch.stack([grid_f, grid_s, obs_rhoxy, obs_phsxy, obs_rhoyx, obs_phsyx], dim=1)
        return obs_points.unsqueeze(0)

    def _prepare_6d_ot_cloud_pred(self, pred_dict):
        """6D prediction point cloud; use same mask as obs for matching."""
        valid_mask = self._get_6d_valid_mask()
        n_freq = len(self.freqs)
        n_stn = len(self.stations)
        log_f = torch.log10(self.freqs)
        norm_f = (log_f - log_f.min()) / (log_f.max() - log_f.min() + 1e-8)
        grid_f = norm_f.view(-1, 1).expand(n_freq, n_stn).flatten()[valid_mask]
        norm_s = (self.stations - self.stations.min()) / (self.stations.max() - self.stations.min() + 1e-8)
        grid_s = norm_s.view(1, -1).expand(n_freq, n_stn).flatten()[valid_mask]

        def _norm_pred(key: str, data: torch.Tensor):
            data_flat = data.flatten()[valid_mask]
            if "rho" in key.lower():
                val_log = torch.log10(data_flat + 1e-12)
                return (val_log - (-2.0)) / (6.0 - (-2.0))
            return data_flat / 90.0

        pred_rhoxy = _norm_pred("rhoxy", pred_dict["rhoxy"])
        pred_phsxy = _norm_pred("phsxy", pred_dict["phsxy"])
        pred_rhoyx = _norm_pred("rhoyx", pred_dict["rhoyx"])
        pred_phsyx = _norm_pred("phsyx", pred_dict["phsyx"])
        pred_points = torch.stack([grid_f, grid_s, pred_rhoxy, pred_phsxy, pred_rhoyx, pred_phsyx], dim=1)
        return pred_points.unsqueeze(0)

    def _prepare_6d_ot_cloud(self, pred_dict, obs_dict):
        return self._prepare_6d_ot_cloud_pred(pred_dict), self._prepare_6d_ot_cloud_obs(obs_dict)

    def compute_w_d_per_point_from_noise(
        self,
        # First entry point for manual data-dimension rebalancing in 6D OT.
        # - scalar: global multiplier for all four data dimensions
        # - length-4: per-dimension multipliers [rhoxy, phsxy, rhoyx, phsyx]
        w_d_scale: Union[float, List[float], Tuple[float, ...]] = 1.0,
        normalize: str = "mean",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compute per-observation-point w_d from propagated noise std.

        Weight rule (per point, per dim): w = 1 / sigma_norm^2.
        - rhoxy/rhoyx: data_noise_std stores std of log10(rho); convert to normalized cloud units by /8.
        - phsxy/phsyx: data_noise_std stores std of (phi/90), already in normalized cloud units.

        Mode weights are applied consistently with the original 6D /sigma scheme:
            sigma_eff = sigma_norm / mode_weight  ->  w ∝ mode_weight^2 / sigma_norm^2

        Args:
            w_d_scale: overall scaling (scalar or 4-seq) applied AFTER normalization.
            normalize: 'mean' | 'none'
                - mean: normalize each dim so mean(w)=1, then multiply by w_d_scale
                - none: no normalization; directly use 1/sigma^2, then multiply by w_d_scale
        Returns:
            Tensor of shape (M, 4) on self.device, where M=n_freq*n_station.
        """
        if not hasattr(self, "data_noise_std"):
            raise RuntimeError("data_noise_std not found. Call load_obs_data/create_synthetic_data first.")

        sigma_min = float(self.ot_config.get("sigma_min", 0.03))
        te_w = float(getattr(self, "te_weight", 1.0))
        tm_w = float(getattr(self, "tm_weight", 1.0))

        def get_sigma_norm(key: str, mode_weight: float) -> torch.Tensor:
            # Use the same noise-floor-clipped std-dev as OT weights/cost.
            t = self.get_effective_cloud_noise_std(key, eps=eps)
            if t is None:
                raise RuntimeError(f"Missing data_noise_std[{key!r}].")
            s = t.to(self.device, dtype=torch.float64)
            # Mode weights: sigma_eff = sigma / mode_weight -> w ∝ mode_weight^2 / sigma^2
            s = s / max(mode_weight, eps)
            return torch.clamp(s, min=sigma_min)

        sig_rhoxy = get_sigma_norm("rhoxy", te_w)
        sig_phsxy = get_sigma_norm("phsxy", te_w)
        sig_rhoyx = get_sigma_norm("rhoyx", tm_w)
        sig_phsyx = get_sigma_norm("phsyx", tm_w)
        # Component balancing is implemented here:
        # - TE/TM balance via te_w/tm_w (different mode_weight for xy vs yx branches)
        # - rho/phs balance via independent sigma fields and optional 4-vector w_d_scale below

        # sig_* already clamped to sigma_min (0.03), so denominator >= 0.0009
        w_rhoxy = 1.0 / (sig_rhoxy ** 2)
        w_phsxy = 1.0 / (sig_phsxy ** 2)
        w_rhoyx = 1.0 / (sig_rhoyx ** 2)
        w_phsyx = 1.0 / (sig_phsyx ** 2)

        # Flatten in the same order as point clouds: (n_freq, n_station) -> (M,)
        w = torch.stack(
            [w_rhoxy.reshape(-1), w_phsxy.reshape(-1), w_rhoyx.reshape(-1), w_phsyx.reshape(-1)],
            dim=1,
        )
        # [关键] 先取 valid 点，再在其上归一化，保证返回的 w 满足 mean(w,dim=0)=1（四分量均值均为 1）
        valid_mask = self._get_6d_valid_mask()
        w = w[valid_mask]

        if normalize not in ("mean", "none"):
            raise ValueError(f"normalize must be 'mean' or 'none', got {normalize!r}")
        if normalize == "mean":
            mean = w.mean(dim=0).clamp(min=1e-12)
            w = w / mean

        if isinstance(w_d_scale, (list, tuple)):
            if len(w_d_scale) != 4:
                raise ValueError("w_d_scale must be scalar or length-4")
            scale_vec = torch.tensor([float(x) for x in w_d_scale], device=self.device, dtype=torch.float64)
        else:
            scale_vec = torch.full((4,), float(w_d_scale), device=self.device, dtype=torch.float64)
        w = w * scale_vec.view(1, 4)
        return w

    def update_ot_w_d_per_point_from_noise(
        self,
        w_d_scale: Union[float, List[float], Tuple[float, ...]] = 1.0,
        normalize: str = "mean",
    ) -> torch.Tensor:
        """Convenience: compute per-point w_d from noise and re-init sinkhorn."""
        w_d_point = self.compute_w_d_per_point_from_noise(w_d_scale=w_d_scale, normalize=normalize)
        self.ot_config["w_d"] = w_d_point
        self._init_sinkhorn(**self.ot_config)
        return w_d_point

    def print_ot_dimension_contributions(self, mode: str = "6dot"):
        """Print per-dimension contributions for OT diagnostic.

        Shows E[(Δx_k)²] (raw) and weighted contribution w_k*E[(Δx_k)²] when using
        WeightedCost, so the sum reflects the actual cost under same-(f,s) matching.
        """
        last_mode = getattr(self, "_last_inversion_mode", None)
        if last_mode == "mse":
            warnings.warn(
                "Last inversion used mode='mse' (no OT loss). "
                "OT dimension contributions shown here are for diagnostic only, "
                "they do not reflect the actual minimized misfit.",
                UserWarning,
                stacklevel=2,
            )
        if self.forward_operator is None or self.model_log_sigma is None:
            print("Call set_forward_operator and initialize_model first.")
            return
        with torch.no_grad():
            sigma_full = self._assemble_sigma_full(torch.exp(self.model_log_sigma))
            pred_dict = self.forward_operator(sigma_full)
        obs_dict = self.obs_data

        if mode == "6dot":
            cloud_pred, cloud_obs = self._prepare_6d_ot_cloud(pred_dict, obs_dict)
            dim_names = ["freq(norm)", "station(norm)", "rhoxy", "phsxy", "rhoyx", "phsyx"]
        else:
            key = list(obs_dict.keys())[0]
            cloud_pred = self._prepare_3d_ot_cloud(pred_dict[key], key)
            cloud_obs = self._prepare_3d_ot_cloud(obs_dict[key], key)
            dim_names = ["freq(norm)", "station(norm)", "value(norm)"]

        pred = cloud_pred.squeeze(0)
        obs = cloud_obs.squeeze(0)
        n_dim = pred.shape[1]

        # Get cost weights for weighted contribution (WeightedCost only)
        cw = getattr(self, "_cost_weights", None)
        w_vec = None
        if cw is not None and mode == "6dot":
            w_s, w_f = float(cw.get("w_s", 1.0)), float(cw.get("w_f", 1.0))
            w_d = cw.get("w_d")
            if isinstance(w_d, (list, tuple)) and len(w_d) == 4:
                w_vec = [w_f, w_s] + [float(x) for x in w_d]
            elif isinstance(w_d, (int, float)):
                w_vec = [w_f, w_s] + [float(w_d)] * 4
            elif hasattr(w_d, "mean"):  # per-point tensor: use mean per dim
                w_d_arr = w_d.mean(dim=0).cpu().numpy() if w_d.dim() == 2 else [float(w_d.mean())] * 4
                w_vec = [w_f, w_s] + [float(x) for x in w_d_arr[:4]]
            else:
                w_vec = [w_f, w_s, 1.0, 1.0, 1.0, 1.0]

        print("\n" + "=" * 86)
        print("OT dimension contributions (same-(f,s) matching)")
        print("-" * 86)
        if w_vec is not None:
            print(f"{'dim':<3} {'name':<16} {'E[(Δx_k)^2]':>16} {'w_k':>10} {'w_k*E':>16}")
        else:
            print(f"{'dim':<3} {'name':<16} {'E[(Δx_k)^2]':>16}")
        print("-" * 86)
        total_raw = 0.0
        total_weighted = 0.0
        for k in range(n_dim):
            delta_sq = ((pred[:, k] - obs[:, k]) ** 2).mean().item()
            total_raw += delta_sq
            w_k = float(w_vec[k]) if w_vec is not None and k < len(w_vec) else 1.0
            contrib = w_k * delta_sq
            total_weighted += contrib
            pred_min, pred_max = pred[:, k].min().item(), pred[:, k].max().item()
            obs_min, obs_max = obs[:, k].min().item(), obs[:, k].max().item()
            if w_vec is not None:
                print(f"{k:<3} {dim_names[k]:<16} {delta_sq:>16.6e} {w_k:>10.4f} {contrib:>16.6e}")
            else:
                print(f"{k:<3} {dim_names[k]:<16} {delta_sq:>16.6e}")
            print(f"    range pred=[{pred_min:.4f}, {pred_max:.4f}]  obs=[{obs_min:.4f}, {obs_max:.4f}]")
        print("-" * 86)
        print(f"  Sum E[(Δx_k)²] (raw)     = {total_raw:.6e}")
        if w_vec is not None:
            print(f"  Sum w_k*E[(Δx_k)²] (cost) = {total_weighted:.6e}")
        st_min = self.stations.min().item()
        st_max = self.stations.max().item()
        L_s = st_max - st_min if st_max > st_min else 1.0
        log_f = torch.log10(self.freqs)
        L_f = (log_f.max() - log_f.min()).item() if log_f.numel() > 1 else 1.0
        station_in_01 = 0 <= pred[:, 1].min().item() <= pred[:, 1].max().item() <= 1
        print(f"\n  [Check] station normalized to [0,1]: {station_in_01}")
        print(f"  Physical scales: L_s={L_s/1e3:.4f} km, L_f={L_f:.4f}")
        print("=" * 86 + "\n")
