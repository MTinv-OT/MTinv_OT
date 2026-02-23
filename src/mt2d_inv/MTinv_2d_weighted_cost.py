"""
MT2D inverter with weighted OT cost.

Cost function:
    C_ij = w_s*(Δs)² + w_f*(Δlog f)² + w_d*(Δdata)²

Point cloud (6dot): [norm_freq, norm_station, rhoxy, phsxy, rhoyx, phsyx].
Set w_d=0 for geometry-only cost; tune w_f for cross-frequency transport penalty.
"""
import torch
from typing import Optional

from .MTinv_2d import MT2DInverter


def make_weighted_cost_fn(w_s: float = 1.0, w_f: float = 1.0, w_d: float = 1.0):
    """
    Build weighted cost C_ij = w_s*(Δs)² + w_f*(Δlog f)² + w_d*(Δdata)².

    Point cloud: dim 0=freq, 1=station, 2..D-1=data (rhoxy, phsxy, rhoyx, phsyx).
    geomloss expects cost(x,y) -> (B,N,M) with x(B,N,D), y(B,M,D).
    """
    def cost_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), y: (B, M, D) -> (B,N,1) - (B,1,M) -> (B,N,M)
        d_freq = (x[:, :, 0:1] - y[:, :, 0:1].permute(0, 2, 1)).pow(2)   # (Δlog f)²
        d_stn = (x[:, :, 1:2] - y[:, :, 1:2].permute(0, 2, 1)).pow(2)   # (Δs)²
        # (Δdata)² = sum over dim 2..D-1
        n_data = x.shape[2] - 2
        if n_data > 0:
            d_data = sum(
                (x[:, :, 2 + k:3 + k] - y[:, :, 2 + k:3 + k].permute(0, 2, 1)).pow(2)
                for k in range(n_data)
            )
        else:
            d_data = torch.zeros(x.shape[0], x.shape[1], y.shape[1], device=x.device, dtype=x.dtype)
        return w_f * d_freq.squeeze(-1) + w_s * d_stn.squeeze(-1) + w_d * d_data
    return cost_fn


class MT2DInverterWeightedCost(MT2DInverter):
    """
    MT2D inverter with weighted cost C_ij = w_s*(Δs)² + w_f*(Δlog f)² + w_d*(Δdata)².

    ot_options adds:
        w_s: station weight (default 1.0)
        w_f: log-freq weight (default 1.0)
        w_d: data weight (default 1.0); set 0 for geometry-only cost
    """

    def __init__(self, **kwargs):
        ot_options = kwargs.get("ot_options") or {}
        ot_options.setdefault("w_s", 1.0)
        ot_options.setdefault("w_f", 1.0)
        ot_options.setdefault("w_d", 1.0)
        kwargs["ot_options"] = ot_options
        super().__init__(**kwargs)

    def _init_sinkhorn(self, p: int, blur: float, scaling: float, reach: Optional[float],
                      backend: str, w_s: float = 1.0, w_f: float = 1.0, w_d: float = 1.0, **kwargs):
        """Init Sinkhorn with custom weighted cost. Ignores p (cost is explicit)."""
        cost_fn = make_weighted_cost_fn(w_s=w_s, w_f=w_f, w_d=w_d)
        try:
            self.sinkhorn_loss = self.opt_config.create_sinkhorn_loss_with_cost(
                cost=cost_fn,
                blur=blur,
                scaling=scaling,
                reach=reach,
                debias=True,
                backend=backend
            )
            print(
                f"✓ Sinkhorn OT Loss (weighted): w_s={w_s}, w_f={w_f}, w_d={w_d}, "
                f"blur={blur}, scale={scaling}, reach={reach}"
            )
        except Exception as e:
            print(f"[Warning] Sinkhorn init failed: {e}")
            self.sinkhorn_loss = None

    def print_ot_dimension_contributions(self, mode: str = "6dot"):
        """Print per-dimension contributions E[(Δx_k)²] for OT diagnostic."""
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

        print("\n" + "=" * 60)
        print("OT dimension contributions E[(Δx_k)²]")
        print("=" * 60)
        for k in range(n_dim):
            delta_sq = ((pred[:, k] - obs[:, k]) ** 2).mean().item()
            pred_min, pred_max = pred[:, k].min().item(), pred[:, k].max().item()
            obs_min, obs_max = obs[:, k].min().item(), obs[:, k].max().item()
            print(f"  k={k} {dim_names[k]:16s}: E[(Δx_k)²] = {delta_sq:.6e}")
            print(f"         pred [{pred_min:.4f}, {pred_max:.4f}]  obs [{obs_min:.4f}, {obs_max:.4f}]")
        print("=" * 60)
        total = sum(((pred[:, k] - obs[:, k]) ** 2).mean().item() for k in range(n_dim))
        print(f"  Total (1/2 * sum_k E[(Δx_k)²]) ≈ {0.5 * total:.6e}")
        st_min = self.stations.min().item()
        st_max = self.stations.max().item()
        L_s = st_max - st_min if st_max > st_min else 1.0
        log_f = torch.log10(self.freqs)
        L_f = (log_f.max() - log_f.min()).item() if log_f.numel() > 1 else 1.0
        station_in_01 = 0 <= pred[:, 1].min().item() <= pred[:, 1].max().item() <= 1
        print(f"\n  [Check] station normalized to [0,1]: {station_in_01}")
        print(f"  Physical: L_s={L_s/1e3:.2f} km, L_f={L_f:.4f}")
        print("=" * 60 + "\n")
