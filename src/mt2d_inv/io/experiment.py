"""
数据存储模块 - 统一实验结果保存

用法:
    from mt2d_inv.io.experiment import ExperimentLogger

    logger = ExperimentLogger(
        model_tag="cross",
        output_root=Path(__file__).parent
    )

    # 反演完成后
    logger.save_from_inverter(inv)
    
    # 或指定 run_name
    logger.save_from_inverter(inv, run_name="test_001")
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import warnings

# 尝试导入 matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ExperimentLogger:
    """
    实验结果日志记录器
    
    目录结构:
        output_root/
        └── model_tag/
            └── 2026-06-18_10-21-33_0/
                ├── figures/
                │   ├── loss_curves.png
                │   ├── gradient_history.png
                │   ├── model_comparison.png
                │   ├── initial_model.png
                │   ├── data_fitting/
                │   │   ├── station_00.png
                │   │   └── ... (one per station by default)
                │   └── profiles_1d.png
                ├── summary.csv
                ├── history.csv
                ├── config.json
                ├── timing.json
                ├── meta.json
                ├── final_model.npz
                ├── static_shift.json
                └── apparent_resistivity.npz
    """
    
    def __init__(
        self,
        model_tag: str,
        output_root: Union[str, Path],
        auto_mkdir: bool = True,
    ):
        self.model_tag = model_tag
        self.output_root = Path(output_root)
        self.base_dir = self.output_root / model_tag
        
        if auto_mkdir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_run_dir(self, run_name: Optional[str] = None) -> Path:
        """生成运行目录"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if run_name is None:
            existing = list(self.base_dir.glob(f"{timestamp}_*"))
            idx = len(existing)
            run_dir = self.base_dir / f"{timestamp}_{idx}"
        else:
            run_dir = self.base_dir / f"{timestamp}_{run_name}"
        
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _extract_summary(self, inv, run_name: str, timestamp: str, run_dir: Path) -> Dict[str, Any]:
        """提取 summary.csv 数据"""
        h = pd.DataFrame(inv.loss_history)
        p = inv.time_stats.get("profile", {})
        
        row = {
            "run_name": run_name,
            "timestamp": timestamp,
            "run_dir": run_dir.name,
            "epochs": len(h),
            "final_misfit": float(h["misfit"].iloc[-1]) if len(h) else np.nan,
            "best_misfit": float(h["misfit"].min()) if len(h) else np.nan,
            "total_time_s": float(inv.time_stats.get("total_inversion_time", np.nan)),
            "avg_epoch_time_s": float(inv.time_stats.get("avg_epoch_time", np.nan)),
            "forward_ms": float(p.get("forward_ms", np.nan)),
            "data_prep_ms": float(p.get("data_prep_ms", np.nan)),
            "data_term_ms": float(p.get("data_term_ms", np.nan)),
            "backward_ms": float(p.get("backward_ms", np.nan)),
            "step_ms": float(p.get("step_ms", np.nan)),
            "reg_ms": float(p.get("regularization_ms", np.nan)),
        }
        if hasattr(inv, 'compute_recovery_rate'):
            try:
                recovery = inv.compute_recovery_rate()
                row["rmse"] = recovery.get("rmse", np.nan)
                row["mape"] = recovery.get("mape", np.nan)
                row["correlation"] = recovery.get("correlation", np.nan)
                row["ssim"] = recovery.get("ssim", np.nan)
                row["anomaly_rmse"] = recovery.get("anomaly_rmse", np.nan)   # ← 新增
                row["anomaly_mape"] = recovery.get("anomaly_mape", np.nan)   # ← 新增
            except Exception as e:
                print(f"计算恢复率时报错了: {e}")
        # 显存
        if str(inv.device).startswith("cuda"):
            try:
                row["peak_gpu_mem_GB"] = float(torch.cuda.max_memory_allocated(inv.device) / 1024**3)
            except:
                row["peak_gpu_mem_GB"] = np.nan
        
        # 恢复率
        if hasattr(inv, 'compute_recovery_rate'):
            try:
                recovery = inv.compute_recovery_rate()
                row["rmse"] = recovery.get("rmse", np.nan)
                row["mape"] = recovery.get("mape", np.nan)
                row["correlation"] = recovery.get("correlation", np.nan)
                row["ssim"] = recovery.get("ssim", np.nan)
            except Exception:
                pass
        
        return row

    @staticmethod
    def _to_numpy(value):
        """Convert torch.Tensor / list / scalar to JSON/npz-friendly Python types."""
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            return [ExperimentLogger._to_serializable(v) for v in value]
        return value

    @staticmethod
    def _to_serializable(value):
        """Convert a scalar/array to JSON-serializable types."""
        if value is None:
            return None
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
            if arr.ndim == 0:
                return float(arr)
            return arr.tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return float(value)
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [ExperimentLogger._to_serializable(v) for v in value]
        return value

    def _extract_static_shift_info(self, inv) -> Dict[str, Any]:
        """提取静位移配置及随机生成的系数。"""
        n_station = len(inv.stations) if hasattr(inv, "stations") else None
        shift_ids = getattr(inv, "shift_station_ids", None) or []

        info: Dict[str, Any] = {
            "static_shift_std": getattr(inv, "static_shift_std", None),
            "shift_ratio": getattr(inv, "shift_ratio", None),
            "shift_stations_mode": getattr(inv, "shift_stations", None),
            "shift_modes": list(getattr(inv, "shift_modes", ()) or ()),
            "n_stations_total": n_station,
            "n_shift_stations": getattr(inv, "n_shift_stations", len(shift_ids)),
            "shift_fraction_actual": getattr(inv, "shift_fraction_actual", None),
            "shift_station_indices": shift_ids,
            "requested_shift_station_indices": getattr(inv, "shift_station_indices", None),
            "shift_station_positions_m": None,
            "shift_factors": {},
            "shift_log10": {},
            # 如果使用了固定强度接口，这里会记录用户传入的规格
            "static_shift_log_input": None,
        }

        if hasattr(inv, "stations") and shift_ids:
            stations_np = self._to_numpy(inv.stations)
            info["shift_station_positions_m"] = [
                float(stations_np[i]) for i in shift_ids
            ]

        for mode in ("xy", "yx"):
            factors = getattr(inv, "static_shift_factors", {}) or {}
            logs = getattr(inv, "static_shift_log", {}) or {}
            if mode in factors:
                info["shift_factors"][mode] = self._to_serializable(factors[mode])
            if mode in logs:
                info["shift_log10"][mode] = self._to_serializable(logs[mode])

        # 记录固定静位移输入规格（如果有）
        fixed_in = getattr(inv, "static_shift_log_input", None)
        if fixed_in is not None:
            info["static_shift_log_input"] = self._to_serializable(fixed_in)

        return info

    def _save_apparent_resistivity(self, inv, run_dir: Path, verbose: bool = True) -> bool:
        """保存各频点视电阻率/相位，供后续重新绘图。"""
        arrays: Dict[str, Any] = {}

        if hasattr(inv, "freqs"):
            arrays["freqs"] = self._to_numpy(inv.freqs)
        if hasattr(inv, "stations"):
            arrays["stations"] = self._to_numpy(inv.stations)

        true_no_shift = getattr(inv, "true_data_no_shift", None) or {}
        for key in ("rhoxy", "rhoyx", "phsxy", "phsyx"):
            if key in true_no_shift:
                arrays[f"true_{key}"] = self._to_numpy(true_no_shift[key])

        obs_data = getattr(inv, "obs_data", None) or {}
        for key in ("rhoxy", "rhoyx", "phsxy", "phsyx"):
            if key in obs_data:
                arrays[f"obs_{key}"] = self._to_numpy(obs_data[key])

        try:
            with torch.no_grad():
                sigma_full = inv.get_sigma_full()
                pred_dict = inv.forward_operator(sigma_full)
            for key in ("rhoxy", "rhoyx", "phsxy", "phsyx"):
                if key in pred_dict:
                    arrays[f"pred_{key}"] = self._to_numpy(pred_dict[key])
        except Exception as e:
            if verbose:
                print(f"  ✗ apparent_resistivity.npz (pred) failed: {e}")

        if not arrays:
            if verbose:
                print("  ✗ apparent_resistivity.npz skipped: no data")
            return False

        np.savez_compressed(run_dir / "apparent_resistivity.npz", **arrays)
        return True
        
    def _extract_config(self, inv) -> Dict[str, Any]:
        """提取 config.json 数据"""
        config = {}
        
        # 1) 反演运行参数 (_last_run_config)
        last_run_config = getattr(inv, "_last_run_config", {})
        if last_run_config:
            config.update(last_run_config)
        
        # 2) 网格参数
        config["nza"] = getattr(inv, "nza", None)
        config["ny"] = len(inv.yn) - 1 if hasattr(inv, "yn") else None
        config["nz"] = len(inv.zn) - 1 if hasattr(inv, "zn") else None
        
        # 3) 观测系统
        config["n_freqs"] = len(inv.freqs) if hasattr(inv, "freqs") else None
        config["n_stations"] = len(inv.stations) if hasattr(inv, "stations") else None
        config["device"] = str(getattr(inv, "device", "cpu"))
        
        # 4) 反演权重
        config["te_weight"] = getattr(inv, "te_weight", None)
        config["tm_weight"] = getattr(inv, "tm_weight", None)
        config["data_loss_scale"] = getattr(inv, "data_loss_scale", None)
        
        # 5) 噪声参数
        config["noise_level"] = getattr(inv, "noise_level", None)
        config["noise_floor"] = getattr(inv, "noise_floor", None)

        # 6) 静位移参数
        config["static_shift"] = self._extract_static_shift_info(inv)

        return config
    
    def _extract_meta(self, inv, run_name: str, timestamp: str) -> Dict[str, Any]:
        """提取 meta.json 数据"""
        meta = {
            "model_tag": self.model_tag,
            "run_name": run_name,
            "timestamp": timestamp,
            "mode": getattr(inv, "_last_inversion_mode", "unknown"),
        }
        
        # OT 配置
        ot_config = getattr(inv, "ot_config", {})
        if ot_config:
            meta["ot_config"] = {
                "p": ot_config.get("p"),
                "blur": ot_config.get("blur"),
                "scaling": ot_config.get("scaling"),
                "reach": ot_config.get("reach"),
                "sigma_min": ot_config.get("sigma_min"),
                "sigma_6d": ot_config.get("sigma_6d"),
            }
        
        # 加权成本配置
        cost_weights = getattr(inv, "_cost_weights", None)
        if cost_weights:
            meta["cost_weights"] = {
                "w_s": cost_weights.get("w_s"),
                "w_f": cost_weights.get("w_f"),
            }
            w_d = cost_weights.get("w_d")
            if isinstance(w_d, (list, tuple)) and len(w_d) == 4:
                meta["cost_weights"]["w_d"] = {
                    "rhoxy": w_d[0],
                    "phsxy": w_d[1],
                    "rhoyx": w_d[2],
                    "phsyx": w_d[3],
                }
            elif torch.is_tensor(w_d):
                meta["cost_weights"]["w_d_shape"] = tuple(w_d.shape)
                meta["cost_weights"]["w_d_mean"] = float(w_d.mean().item()) if w_d.numel() > 0 else None
        
        return meta
    
    def _save_figures(self, inv, run_dir: Path, plot_kwargs: Optional[Dict] = None):
        """保存所有图片到 figures/ 子目录"""
        if not HAS_MPL:
            print("Warning: matplotlib not available, skipping plots")
            return
        
        plot_kwargs = plot_kwargs or {}
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 保存原始 plt.show（移到 try 块之前，确保 finally 能访问）
        original_show = plt.show
        
        try:
            h = pd.DataFrame(inv.loss_history)
            
            # 1) loss_curves.png
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            if len(h) > 0:
                ax1.semilogy(h["epoch"], h["misfit"], 'b-', linewidth=2)
                ax1.axhline(y=1.05, color='r', linestyle='--', label='Target (χ²=1.05)')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('RMS χ²')
                ax1.set_title('Data Misfit')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                ax2.semilogy(h["epoch"], h["data_loss"], 'c-', linewidth=2, label='Data Loss')
                ax2.semilogy(h["epoch"], h["model_loss"], 'm-', linewidth=2, label='Model Roughness')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss Components')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.suptitle('Inversion Convergence', fontsize=14)
            plt.tight_layout()
            fig.savefig(figures_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 2) gradient_history.png
            if len(h) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                epochs = [log['epoch'] for log in inv.loss_history[1:]]
                g_d = [log.get('grad_data_norm', np.nan) for log in inv.loss_history[1:]]
                g_m = [log.get('grad_model_norm', np.nan) for log in inv.loss_history[1:]]
                
                ax.plot(epochs, g_d, 'b-', label='||∇Φ_d||', linewidth=2)
                ax.plot(epochs, g_m, 'r-', label='λ·||∇Φ_m||', linewidth=2)
                ax.set_yscale('log')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Gradient norm')
                ax.set_title('Gradient History')
                ax.grid(True, alpha=0.5)
                ax.legend()
                plt.tight_layout()
                fig.savefig(figures_dir / "gradient_history.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # 3) 使用 inv 的绘图函数，禁用 plt.show()
            plt.show = lambda: None

            def _save_plot_fig(fig, filename: str):
                if fig is not None and fig.get_axes():
                    fig.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            if hasattr(inv, 'sig_true') and inv.sig_true is not None:
                try:
                    fig = inv.plot_model_comparison(**plot_kwargs)
                    _save_plot_fig(fig, "model_comparison.png")
                except Exception as e:
                    print(f"Warning: model_comparison plot failed: {e}")
            
            if hasattr(inv, 'initial_model_sigma') and inv.initial_model_sigma is not None:
                try:
                    fig = inv.plot_initial_model(show=False, **plot_kwargs)
                    _save_plot_fig(fig, "initial_model.png")
                except Exception as e:
                    print(f"Warning: initial_model plot failed: {e}")
            
            try:
                fit_kw: Dict[str, Any] = {"show": False}
                if plot_kwargs.get("plot_noise_cap") is not None:
                    fit_kw["plot_noise_cap"] = plot_kwargs["plot_noise_cap"]
                if plot_kwargs.get("data_fitting_station_indices") is not None:
                    fit_kw["station_indices"] = plot_kwargs["data_fitting_station_indices"]
                batch_size = int(plot_kwargs.get("data_fitting_batch_size", 1))
                fit_kw["stations_per_figure"] = batch_size

                station_indices = fit_kw.get("station_indices")
                if station_indices is None:
                    station_indices = list(range(len(inv.stations)))
                elif isinstance(station_indices, int):
                    station_indices = [station_indices]
                else:
                    station_indices = list(station_indices)

                result = inv.plot_data_fitting(**fit_kw)
                figs = result if isinstance(result, list) else ([result] if result is not None else [])
                df_dir = figures_dir / "data_fitting"
                df_dir.mkdir(exist_ok=True)

                n_saved = 0
                for fig_idx, fig in enumerate(figs):
                    batch = station_indices[fig_idx * batch_size : (fig_idx + 1) * batch_size]
                    if batch_size == 1 and len(batch) == 1:
                        out_name = f"station_{batch[0]:02d}.png"
                    else:
                        out_name = f"batch_{fig_idx:03d}.png"
                    if fig is not None and fig.get_axes():
                        fig.savefig(df_dir / out_name, dpi=300, bbox_inches="tight")
                        plt.close(fig)
                        n_saved += 1
                print(
                    f"  ✓ data_fitting: {n_saved} plots saved to {df_dir} "
                    f"({len(station_indices)} stations)"
                )
            except Exception as e:
                print(f"Warning: data_fitting plot failed: {e}")
            
            try:
                fig = inv.plot_1d_profiles(depth_limit_km=50)
                _save_plot_fig(fig, "profiles_1d.png")
            except Exception as e:
                print(f"Warning: profiles_1d plot failed: {e}")
            
            print(f"  ✓ Figures saved to {figures_dir}")
            
        except Exception as e:
            print(f"Warning: Error saving figures: {e}")
        finally:
            # 恢复原始 plt.show
            plt.show = original_show
    
    def save_from_inverter(
        self,
        inv,
        run_name: Optional[str] = None,
        save_figures: bool = True,
        plot_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ) -> Path:
        """从反演器保存所有结果"""
        
        # 1) 先检查是否有损失历史（放在最前面）
        if not hasattr(inv, 'loss_history') or len(inv.loss_history) == 0:
            raise ValueError("inv.loss_history is empty. Run inversion first.")
        
        # 2) 生成目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_name is None:
            existing = list(self.base_dir.glob(f"{timestamp}_*"))
            idx = len(existing)
            run_name = str(idx)
        
        run_dir = self.base_dir / f"{timestamp}_{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Saving results to: {run_dir}")
            print(f"{'='*60}")
        
        # 3) summary.csv
        summary_data = self._extract_summary(inv, run_name, timestamp, run_dir)
        pd.DataFrame([summary_data]).to_csv(
            run_dir / "summary.csv", index=False, encoding="utf-8-sig"
        )
        if verbose:
            print(f"  ✓ summary.csv")
        
        # 4) history.csv
        h = pd.DataFrame(inv.loss_history)
        h.to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  ✓ history.csv")
        
        # 5) config.json
        config_data = self._extract_config(inv)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"  ✓ config.json")
        
        # 6) timing.json
        timing_data = getattr(inv, "time_stats", {})
        with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(timing_data, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"  ✓ timing.json")
        
        # 7) meta.json
        meta_data = self._extract_meta(inv, run_name, timestamp)
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"  ✓ meta.json")
        
        # 8) final_model.npz
        try:
            final_sigma = inv.get_sigma_full().detach().cpu().numpy()
            np.savez_compressed(
                run_dir / "final_model.npz",
                sigma=final_sigma,
                zn=inv.zn.cpu().numpy() if torch.is_tensor(inv.zn) else inv.zn,
                yn=inv.yn.cpu().numpy() if torch.is_tensor(inv.yn) else inv.yn,
                nza=inv.nza,
            )
            if verbose:
                print(f"  ✓ final_model.npz")
        except Exception as e:
            if verbose:
                print(f"  ✗ final_model.npz failed: {e}")

        # 9) static_shift.json
        static_shift_data = self._extract_static_shift_info(inv)
        with open(run_dir / "static_shift.json", "w", encoding="utf-8") as f:
            json.dump(static_shift_data, f, indent=2, ensure_ascii=False, default=str)
        if verbose:
            print(f"  ✓ static_shift.json")

        # 10) apparent_resistivity.npz
        if self._save_apparent_resistivity(inv, run_dir, verbose=verbose):
            if verbose:
                print(f"  ✓ apparent_resistivity.npz")
        
        # 11) figures/
        if save_figures and HAS_MPL:
            self._save_figures(inv, run_dir, plot_kwargs)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Results saved successfully!")
            print(f"  Location: {run_dir}")
            if "final_misfit" in summary_data:
                print(f"  Final RMS χ²: {summary_data['final_misfit']:.4f}")
            if "rmse" in summary_data and not np.isnan(summary_data["rmse"]):
                print(f"  RMSE: {summary_data['rmse']:.4f}")
            if "mape" in summary_data and not np.isnan(summary_data["mape"]):
                print(f"  MAPE: {summary_data['mape']:.2f}%")
            print(f"{'='*60}\n")
        
        return run_dir


# ==================== 向后兼容的函数接口 ====================

def save_inversion_results(
    inverter,
    model_name: str,
    run_name: Optional[str] = None,
    save_dir_name: str = "实验结果",
    save_plots: bool = True,
    plot_kwargs: Optional[Dict] = None,
    save_model: bool = True,
    save_params: bool = True,
    verbose: bool = True,
) -> Path:
    """向后兼容的旧接口"""
    logger = ExperimentLogger(
        model_tag=model_name,
        output_root=Path.cwd() / save_dir_name,
    )
    return logger.save_from_inverter(
        inv=inverter,
        run_name=run_name,
        save_figures=save_plots,
        plot_kwargs=plot_kwargs,
        verbose=verbose,
    )


def save_multiple_runs(
    inverter_list: List,
    model_name: str,
    run_names: Optional[List[str]] = None,
    save_dir_name: str = "实验结果",
    save_plots: bool = True,
    plot_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> List[Path]:
    """向后兼容的旧接口"""
    logger = ExperimentLogger(
        model_tag=model_name,
        output_root=Path.cwd() / save_dir_name,
    )
    
    if run_names is None:
        run_names = [f"run_{i+1:02d}" for i in range(len(inverter_list))]
    
    if len(inverter_list) != len(run_names):
        raise ValueError("inverter_list and run_names must have the same length")
    
    saved_paths = []
    for inv, name in zip(inverter_list, run_names):
        path = logger.save_from_inverter(
            inv=inv,
            run_name=name,
            save_figures=save_plots,
            plot_kwargs=plot_kwargs,
            verbose=verbose,
        )
        saved_paths.append(path)
    
    return saved_paths


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """自动找最新的 checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    
    def get_epoch_num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    return str(max(checkpoints, key=get_epoch_num))


# ==================== Checkpoint 保存方法（作为独立函数） ====================

def save_checkpoint(inv, optimizer, epoch: int, checkpoint_dir: Path) -> None:
    """
    保存 checkpoint（独立函数，非类方法）
    
    Parameters
    ----------
    inv : MT2DInverter
        反演器对象
    optimizer : torch.optim.Optimizer
        优化器对象
    epoch : int
        当前 epoch 编号
    checkpoint_dir : Path
        保存目录
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state': inv.model_log_sigma.data,
        'optimizer_state': optimizer.state_dict(),
        'loss_history': inv.loss_history,
    }, checkpoint_dir / f"epoch_{epoch:04d}.pt")
