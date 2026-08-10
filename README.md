# MTinv_OT

A research toolbox for 1D/2D magnetotelluric (MT) forward modeling and inversion with **Optimal Transport (OT)** data misfits.

> Within a conventional MT inversion framework, geomloss Sinkhorn geometric OT distances are used as alternatives or complements to classical L2 data misfits, aiming for improved robustness to noise and non-Gaussian errors.

**Repository**: https://github.com/MTinv-OT/MTinv_OT

---

## Features

| Module | Description |
|------|------|
| **MT 1D OT inversion** (`mt1d_inv`) | Layered 1D models; apparent resistivity/phase embedded as 3D point clouds; Sinkhorn OT + Occam-style constraints |
| **MT 2D FD forward** (`mt2d_inv.forward`) | TE/TM total-field finite differences; PyTorch-differentiable; real-block solver |
| **MT 2D OT inversion** (`mt2d_inv.inversion`) | Multi-frequency, multi-station, four-component joint inversion; 3D/6D Sinkhorn OT; MSE comparison mode |
| **Data preparation** (`mt2d_inv.data_prep`) | EDI reading, strike estimation, profile projection, cleaning, and export |
| **Experiment I/O** (`mt2d_inv.io`) | `ExperimentLogger` for config / history / figures / metrics |
| **Plotting** (`mt2d_inv.plotting`) | Model comparison, data fitting, pseudosections, OT vs MSE convergence |

---

## Code structure

The refactored `mt2d_inv` package uses **Mixin composition** instead of a single monolithic `MTinv_2d.py`:

```
src/
├── mt1d_inv/                    # 1D inversion
│   ├── MTinv.py                 # MT1DInverter
│   ├── model.py, constraints.py, optimizer.py
│   └── visualize.py
│
└── mt2d_inv/
    ├── __init__.py              # Public API
    ├── models.py                # MT2DTrueModels (COMMEMI, Rubic, ...)
    ├── constraints.py           # 2D smoothness / roughness constraints
    ├── optimizer.py             # OptimizerConfig
    │
    ├── forward/
    │   └── solver.py            # MT2DFD_Torch (2D FD forward)
    │
    ├── inversion/               # 2D inversion core (Mixin composition)
    │   ├── base.py              # MT2DInverter
    │   ├── weighted_cost.py     # MT2DInverterWeightedCost
    │   ├── data.py              # Synthetic/observed data, static shift, error propagation
    │   ├── ot.py                # Sinkhorn OT data term
    │   ├── regularization.py    # Regularization and adaptive λ
    │   └── metrics.py           # Recovery metrics: RMSE / SSIM / correlation
    │
    ├── data_prep/               # Field-data preparation pipeline
    │   ├── prepare.py           # PrepareData (main entry)
    │   ├── edi.py               # EDI parsing and impedance transforms
    │   ├── strike.py            # Strike estimation and profile projection
    │   ├── cleaning.py          # Data cleaning
    │   ├── export.py            # Export tensors for inversion
    │   └── grid.py              # Grid extent helpers
    │
    ├── plotting/                # Visualization
    │   ├── inversion.py         # Model comparison, data fitting, sections
    │   ├── pseudosection.py     # Apparent-resistivity pseudosections
    │   ├── comparison.py        # OT vs MSE convergence
    │   └── prepare_data.py      # Plots for the preparation stage
    │
    └── io/
        └── experiment.py        # ExperimentLogger
```

### Main public API

```python
from mt2d_inv import MT2DInverter, MT2DInverterWeightedCost, MT2DFD_Torch, MT2DTrueModels
from mt2d_inv.data_prep import PrepareData
from mt2d_inv.io import ExperimentLogger
from mt2d_inv.plotting import (
    plot_model_comparison,
    plot_data_fitting,
    plot_ot_mse_convergence,
    plot_ot_mse_pseudosection_from_npz,
)
```

---

## Requirements

- Python ≥ 3.10
- PyTorch (CPU or GPU)
- numpy, matplotlib, scikit-image, pandas

```bash
pip install -e ".[ot,dev]"
# or step by step
pip install -e .
pip install geomloss jupyter   # OT inversion and notebook experiments
```

If you hit an OpenMP conflict on Windows:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Installation

```bash
git clone https://github.com/MTinv-OT/MTinv_OT.git
cd MTinv_OT
pip install -e ".[ot,dev]"
```

If a notebook is not launched from the project root, add `src` to `sys.path`:

```python
import sys
from pathlib import Path

def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src" / "mt2d_inv" / "__init__.py").is_file():
            return p
    raise RuntimeError("Cannot find project root")

root = find_project_root(Path.cwd())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
```

---

## Quick start

### 2D synthetic inversion (OT vs MSE)

```python
import torch
from mt2d_inv import MT2DInverterWeightedCost, MT2DTrueModels
from mt2d_inv.io import ExperimentLogger

device = "cuda" if torch.cuda.is_available() else "cpu"
yn, zn, nza, sig_true = MT2DTrueModels.create_commemi_2d4(nza=10, device=device)
freqs = torch.logspace(0, -5, 30, device=device)
stations = torch.linspace(-15000, 15000, 21, device=device)

inv = MT2DInverterWeightedCost(
    yn=torch.tensor(yn, dtype=torch.float64, device=device),
    zn=torch.tensor(zn, dtype=torch.float64, device=device),
    nza=nza, freqs=freqs, stations=stations,
    device=device, random_seed=123,
)
inv.set_forward_operator()
inv.sig_true = sig_true
inv.create_synthetic_data(noise_level=0.01, noise_type="gaussian")
inv.initialize_model(initial_sigma=0.01)

inv.run_inversion(n_epochs=300, mode="6dot")   # OT
logger = ExperimentLogger(model_tag="demo", output_root="test_results")
logger.save_from_inverter(inv, run_name="ot_demo")
```

### Field-data preparation (EDI → inversion tensors)

```python
from mt2d_inv.data_prep import PrepareData

prep = PrepareData(
    edi_dir="path/to/edi",
    n_freq_target=20,
    freq_min_hz=1e-4,
    freq_max_hz=1e4,
)
prep.run_all_simple(rotate=True, strike_true_deg=45.0)
data_dict = prep.export_data_dict_for_2d_inversion()
```

### Static shift

Synthetic data supports two static-shift modes:

**1. Random mode (original interface)**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.15,          # std σ of the log10 multiplier (not variance)
    shift_modes=("xy", "yx"),       # xy=TE, yx=TM
    shift_stations="random",        # "all" | "middle" | "random"
    shift_ratio=0.2,                # fraction of affected stations
)
```

**2. Fixed-amplitude mode (new interface)**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.0,
    shift_station_indices=[8, 9, 10, 11],   # 0-based station indices
    static_shift_log={
        "xy": 0.15,    # TE: direct log10 multiplier (×10^0.15 ≈ 1.41)
        "yx": 0.15,    # TM: same; set one or both
    },
)
```

| Argument | Meaning |
|------|------|
| `static_shift_std` | Random mode: Gaussian **std** σ of the log10 multiplier |
| `static_shift_log` | Fixed mode: **direct log10 multiplier** (not variance / std) |
| `shift_station_indices` | Explicit affected stations; overrides `shift_stations` |

Static-shift settings are written to `static_shift.json`; recovery metrics (including SSIM) are written to `summary.csv`.

---

## Experiment notebooks

Notebooks under `tests/` are organized by experiment type.

### Synthetic experiments — `tests/synthetic/`

COMMEMI 2D-1 / 2D-4 and Rubic models; OT (`6dot`) vs MSE; random static shift.

| Notebook | Model | Static shift |
|----------|------|--------|
| `commemi_2d1_21_shift*.ipynb` | COMMEMI 2D-1, 21 stations | random (`shift_ratio` = 0 / 0.1 / 0.15 / 0.2) |
| `commemi_2d4_21_shift*.ipynb` | COMMEMI 2D-4, 21 stations | same |
| `rubic_ot_21.ipynb` / `rubic_mse_21.ipynb` | Rubic, 21 stations | random |
| `rubic_ot_31.ipynb` / `rubic_mse_31.ipynb` | Rubic, 31 stations | random |

### Fixed static-shift comparison — `tests/synthetic/shift_compare/`

Uses `static_shift_log` to compare TE/TM static-shift effects (3 scenarios each for COMMEMI 2D-1 and 2D-4):

| Notebook | Scenario |
|----------|------|
| `*_static_te_only.ipynb` | TE (xy) only |
| `*_static_tm_only.ipynb` | TM (yx) only |
| `*_static_both_same.ipynb` | Same fixed static shift on TE and TM |

### Field-data experiments

| Directory | Description |
|------|------|
| `tests/AKBST-AMT-L08/` | AKBST AMT profile; MSE / TE5 OT inversion |
| `tests/Cascadia/` | Cascadia profile; 6dot TE/TM weight comparison |

### Output layout

Each `ExperimentLogger` run writes:

```
test_results/<model_tag>/<timestamp>_<run_name>/
├── config.json
├── summary.csv
├── history.csv
├── static_shift.json
├── apparent_resistivity.npz
├── final_model.npz
└── figures/
    ├── model_comparison.png
    ├── data_fitting/
    └── ...
```

---

## Plotting helpers

| Function | Purpose |
|------|------|
| `plot_model_comparison(inv)` | True vs inverted log10(ρ); prints SSIM |
| `plot_data_fitting(inv, station_indices=...)` | Station data-fit curves |
| `plot_ot_mse_convergence(hist_ot, hist_mse)` | OT vs MSE convergence |
| `plot_ot_mse_pseudosection_from_npz(npz_ot, npz_mse)` | Pseudosection comparison |
| `plot_rho_fitting_from_npz(npz_path)` | Reload data fits from saved npz |

SSIM from `inv.compute_recovery_rate()['ssim']` matches `plot_model_comparison` (log10(ρ) domain, air layer excluded) and is saved in `summary.csv`.

---

## MT 1D inversion

The 1D module is independent; entry point `mt1d_inv`:

```python
from mt1d_inv import MT1D, MT1DInverter

inv = MT1DInverter(device="cuda", use_sinkhorn=True)
inv.generate_synthetic_data(true_dz=..., true_sig=..., noise_level=0.05)
inv.run_inversion(num_epochs=800, use_adaptive_lambda=True)
inv.plot_data_fit()
inv.plot_model_comparison()
```

---

## Contributors

- Authors: Xinran Liu, Xuanzhang Chen, Bo Yang, Ziyu Tang
- Contact: xinran.liu@zju.edu.cn, bo.yang@zju.edu.cn

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt).

# MTinv_OT

基于最优传输（Optimal Transport, OT）的 MT 1D/2D 反演与正演研究工具箱。

> 在传统 MT 反演框架中，以 geomloss Sinkhorn 几何 OT 距离替代/补充经典 L2 数据拟合，提升对噪声与非高斯误差的鲁棒性。

**仓库**: [https://github.com/MTinv-OT/MTinv_OT](https://github.com/MTinv-OT/MTinv_OT)

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **MT 1D OT 反演** (`mt1d_inv`) | 一维分层模型；视电阻率/相位嵌入 3D 点云；Sinkhorn OT + Occam 约束 |
| **MT 2D FD 正演** (`mt2d_inv.forward`) | TE/TM 总场法有限差分；PyTorch 可微；real-block 求解 |
| **MT 2D OT 反演** (`mt2d_inv.inversion`) | 多频、多台站、四分量联合反演；3D/6D Sinkhorn OT；MSE 对比模式 |
| **数据准备** (`mt2d_inv.data_prep`) | EDI 读取、strike 估计、剖面投影、数据清洗与导出 |
| **实验 I/O** (`mt2d_inv.io`) | `ExperimentLogger` 统一保存 config / history / figures / metrics |
| **绘图** (`mt2d_inv.plotting`) | 模型对比、数据拟合、伪剖面、OT vs MSE 收敛对比 |

---

## 代码结构

重构后的 `mt2d_inv` 采用 **Mixin 组合**，替代原先单文件 `MTinv_2d.py`：

```
src/
├── mt1d_inv/                    # 1D 反演
│   ├── MTinv.py                 # MT1DInverter
│   ├── model.py, constraints.py, optimizer.py
│   └── visualize.py
│
└── mt2d_inv/
    ├── __init__.py              # 公开 API 入口
    ├── models.py                # MT2DTrueModels（COMMEMI、Rubic 等标准模型）
    ├── constraints.py           # 2D 平滑/粗糙度约束
    ├── optimizer.py             # OptimizerConfig
    │
    ├── forward/
    │   └── solver.py            # MT2DFD_Torch（2D 有限差分正演）
    │
    ├── inversion/               # 2D 反演核心（Mixin 组合）
    │   ├── base.py              # MT2DInverter
    │   ├── weighted_cost.py     # MT2DInverterWeightedCost
    │   ├── data.py              # 合成/观测数据、静位移、误差传播
    │   ├── ot.py                # Sinkhorn OT 数据项
    │   ├── regularization.py    # 正则化与自适应 λ
    │   └── metrics.py           # 恢复率 RMSE / SSIM / correlation
    │
    ├── data_prep/               # 实测数据准备流水线
    │   ├── prepare.py           # PrepareData（主入口）
    │   ├── edi.py               # EDI 解析与阻抗变换
    │   ├── strike.py            # Strike 估计与剖面投影
    │   ├── cleaning.py          # 数据清洗
    │   ├── export.py            # 反演张量导出
    │   └── grid.py              # 网格范围计算
    │
    ├── plotting/                # 反演结果可视化
    │   ├── inversion.py         # 模型对比、数据拟合、剖面等
    │   ├── pseudosection.py     # 视电阻率伪剖面
    │   ├── comparison.py        # OT vs MSE 收敛对比
    │   └── prepare_data.py      # 数据准备阶段绘图
    │
    └── io/
        └── experiment.py        # ExperimentLogger
```

### 主要公开 API

```python
from mt2d_inv import MT2DInverter, MT2DInverterWeightedCost, MT2DFD_Torch, MT2DTrueModels
from mt2d_inv.data_prep import PrepareData
from mt2d_inv.io import ExperimentLogger
from mt2d_inv.plotting import (
    plot_model_comparison,
    plot_data_fitting,
    plot_ot_mse_convergence,
    plot_ot_mse_pseudosection_from_npz,
)
```

---

## 环境与依赖

- Python ≥ 3.10
- PyTorch（CPU 或 GPU）
- numpy, matplotlib, scikit-image, pandas

```bash
pip install -e ".[ot,dev]"
# 或分步安装
pip install -e .
pip install geomloss jupyter   # OT 反演与 notebook 实验
```

Windows 上若遇 OpenMP 冲突，可设置：

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 安装与导入

```bash
git clone https://github.com/MTinv-OT/MTinv_OT.git
cd MTinv_OT
pip install -e ".[ot,dev]"
```

Notebook 中若不在项目根目录运行，需将 `src` 加入路径：

```python
import sys
from pathlib import Path

def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src" / "mt2d_inv" / "__init__.py").is_file():
            return p
    raise RuntimeError("Cannot find project root")

root = find_project_root(Path.cwd())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
```

---

## 快速上手

### 2D 合成数据反演（OT vs MSE）

```python
import torch
from mt2d_inv import MT2DInverterWeightedCost, MT2DTrueModels
from mt2d_inv.io import ExperimentLogger

device = "cuda" if torch.cuda.is_available() else "cpu"
yn, zn, nza, sig_true = MT2DTrueModels.create_commemi_2d4(nza=10, device=device)
freqs = torch.logspace(0, -5, 30, device=device)
stations = torch.linspace(-15000, 15000, 21, device=device)

inv = MT2DInverterWeightedCost(
    yn=torch.tensor(yn, dtype=torch.float64, device=device),
    zn=torch.tensor(zn, dtype=torch.float64, device=device),
    nza=nza, freqs=freqs, stations=stations,
    device=device, random_seed=123,
)
inv.set_forward_operator()
inv.sig_true = sig_true
inv.create_synthetic_data(noise_level=0.01, noise_type="gaussian")
inv.initialize_model(initial_sigma=0.01)

inv.run_inversion(n_epochs=300, mode="6dot")   # OT
logger = ExperimentLogger(model_tag="demo", output_root="test_results")
logger.save_from_inverter(inv, run_name="ot_demo")
```

### 实测数据准备（EDI → 反演张量）

```python
from mt2d_inv.data_prep import PrepareData

prep = PrepareData(
    edi_dir="path/to/edi",
    n_freq_target=20,
    freq_min_hz=1e-4,
    freq_max_hz=1e4,
)
prep.run_all_simple(rotate=True, strike_true_deg=45.0)
data_dict = prep.export_data_dict_for_2d_inversion()
```

### 静位移（Static Shift）

合成数据支持两种静位移施加方式：

**1. 随机模式（原有接口）**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.15,          # log10 乘子的标准差 σ（非方差）
    shift_modes=("xy", "yx"),       # xy=TE, yx=TM
    shift_stations="random",        # "all" | "middle" | "random"
    shift_ratio=0.2,                # 受影响台站比例
)
```

**2. 固定强度模式（新接口）**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.0,
    shift_station_indices=[8, 9, 10, 11],   # 0-based 台站序号
    static_shift_log={
        "xy": 0.15,    # TE：直接 log10 乘子（×10^0.15 ≈ 1.41）
        "yx": 0.15,    # TM：同上；可分别设置或只设其中一个
    },
)
```

| 参数 | 含义 |
|------|------|
| `static_shift_std` | 随机模式：log10 乘子的高斯 **标准差** σ |
| `static_shift_log` | 固定模式：**直接的 log10 乘子**（非方差、非标准差） |
| `shift_station_indices` | 显式指定受影响台站，优先级高于 `shift_stations` |

静位移配置与系数会写入 `static_shift.json`，恢复率指标（含 SSIM）写入 `summary.csv`。

---

## 实验 Notebook

`tests/` 目录按实验类型组织：

### 合成数据实验 — `tests/synthetic/`

COMMEMI 2D-1 / 2D-4 与 Rubic 模型；OT（6dot）与 MSE 对比；随机静位移。

| Notebook | 模型 | 静位移 |
|----------|------|--------|
| `commemi_2d1_21_shift*.ipynb` | COMMEMI 2D-1, 21 台站 | 随机（`shift_ratio` = 0 / 0.1 / 0.15 / 0.2） |
| `commemi_2d4_21_shift*.ipynb` | COMMEMI 2D-4, 21 台站 | 同上 |
| `rubic_ot_21.ipynb` / `rubic_mse_21.ipynb` | Rubic, 21 台站 | 随机 |
| `rubic_ot_31.ipynb` / `rubic_mse_31.ipynb` | Rubic, 31 台站 | 随机 |

### 固定静位移对比 — `tests/synthetic/shift_compare/`

使用新接口 `static_shift_log`，对比 TE/TM 静位移影响（COMMEMI 2D-1 与 2D-4 各 3 种场景）：

| Notebook | 场景 |
|----------|------|
| `*_static_te_only.ipynb` | 仅 TE (xy) 有静位移 |
| `*_static_tm_only.ipynb` | 仅 TM (yx) 有静位移 |
| `*_static_both_same.ipynb` | TE 与 TM 相同固定静位移 |

### 实测数据实验

| 目录 | 说明 |
|------|------|
| `tests/AKBST-AMT-L08/` | AKBST AMT 剖面数据；MSE / TE5 OT 反演 |
| `tests/Cascadia/` | Cascadia 剖面数据；6dot TE/TM 权重对比 |

### 实验结果目录结构

`ExperimentLogger` 每次运行生成：

```
test_results/<model_tag>/<timestamp>_<run_name>/
├── config.json              # 反演与静位移配置
├── summary.csv              # RMSE, SSIM, misfit, timing 等
├── history.csv              # 逐 epoch 损失历史
├── static_shift.json        # 静位移参数与实际系数
├── apparent_resistivity.npz # 各频点 ρ/φ（供伪剖面重绘）
├── final_model.npz
└── figures/
    ├── model_comparison.png
    ├── data_fitting/
    └── ...
```

---

## 绘图函数速查

| 函数 | 用途 |
|------|------|
| `plot_model_comparison(inv)` | 真/反演模型 log10(ρ) 对比；含 SSIM 打印 |
| `plot_data_fitting(inv, station_indices=...)` | 台站数据拟合曲线 |
| `plot_ot_mse_convergence(hist_ot, hist_mse)` | OT vs MSE 收敛对比 |
| `plot_ot_mse_pseudosection_from_npz(npz_ot, npz_mse)` | 伪剖面对比（共享/独立色标） |
| `plot_rho_fitting_from_npz(npz_path)` | 从保存的 npz 重绘数据拟合 |

SSIM 指标：`inv.compute_recovery_rate()['ssim']` 与 `plot_model_comparison` 打印值一致（log10(ρ) 域，排除空气层），并写入 `summary.csv`。

---

## MT 1D 反演

1D 模块保持独立，入口为 `mt1d_inv`：

```python
from mt1d_inv import MT1D, MT1DInverter

inv = MT1DInverter(device="cuda", use_sinkhorn=True)
inv.generate_synthetic_data(true_dz=..., true_sig=..., noise_level=0.05)
inv.run_inversion(num_epochs=800, use_adaptive_lambda=True)
inv.plot_data_fit()
inv.plot_model_comparison()
```

---

## 贡献者

- 作者：Xinran Liu, Xuanzhang Chen, Bo Yang, Ziyu Tang
- 联系：xinran.liu@zju.edu.cn, bo.yang@zju.edu.cn

---

## License

见仓库根目录 [LICENSE](LICENSE)。
