## MTinv_OT: MT 1D/2D Inversion and Forward Modelling with Optimal Transport (OT)

MTinv_OT is a Python research toolbox for magnetotelluric (MT) problems. The core idea:

> Within the traditional MT inversion framework, replace or supplement classical L2 data misfit with geometric Optimal Transport (OT) distance (geomloss Sinkhorn operator) for more robust fitting under noise and non-Gaussian errors.

Features:

- **MT 1D OT inversion**: Embed apparent resistivity and phase into 3D point cloud (logρ, normalized phase, logf) with geomloss `SamplesLoss(loss="sinkhorn")`, plus Occam smoothness, reference model constraints, and adaptive regularization.
- **MT 2D finite-difference forward**: TE/TM total-field 2D FD forward operator as a differentiable front-end for 2D OT inversion.
- **MT 2D experimental OT inversion**: Embed multi-frequency, multi-station, four-component (ρ/φ) observations in 3D/6D OT space for joint fitting with Sinkhorn distance and depth-weighted roughness.
- **1D/2D regularization and diagnostics**: Roughness/curvature matrices, χ² statistics, adaptive λ, gradient history, and visualizations.

Modules are implemented in PyTorch and geomloss, run on CPU or GPU, and are designed for integration with deep learning, joint inversion, and other physics constraints.

---

## Overview

### 1. MT 1D Inversion (mt1d_inv)

Key files:

- [src/mt1d_inv/MTinv.py](src/mt1d_inv/MTinv.py): Main inverter `MT1DInverter`
- [src/mt1d_inv/model.py](src/mt1d_inv/model.py): 1D geo-electric model `MT1D`
- [src/mt1d_inv/constraints.py](src/mt1d_inv/constraints.py): Roughness/curvature constraints
- [src/mt1d_inv/optimizer.py](src/mt1d_inv/optimizer.py): Optimizer and loss setup

Features: 1D MT forward and inversion, OT data misfit with geomloss Sinkhorn, Occam constraints, reference model, adaptive λ, error propagation and weighting, χ² diagnostics and plots.

### 2. MT 2D Forward (mt2d_inv)

Key files:

- [src/mt2d_inv/MT2D.py](src/mt2d_inv/MT2D.py): 2D FD forward `MT2DFD_Torch`
- [src/mt2d_inv/constraints.py](src/mt2d_inv/constraints.py): 2D smoothness constraints

Features: Total-field TE/TM 2D MT forward, complex FD, PyTorch auto-diff, real-block solve, 1D background for boundary conditions.

### 3. MT 2D Inversion (Experimental, OT-based)

Key file: [src/mt2d_inv/MTinv_2d.py](src/mt2d_inv/MTinv_2d.py): `MT2DInverter`

Features: TE/TM joint inversion, 3D/6D Sinkhorn OT, mode switch (`3dot`/`6dot`/`mse`), error propagation and χ², depth-weighted roughness, reference model, adaptive λ, diagnostics.

---

## Environment and Dependencies

- Python ≥ 3.8
- PyTorch (CPU or GPU)
- numpy, matplotlib
- Optional: **geomloss** for Sinkhorn OT

```bash
pip install torch numpy matplotlib
pip install geomloss  # for Sinkhorn OT
```

---

## Installation and Import

Clone the repo and add `src` to the Python path:

```python
import sys, os
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, "src"))

from mt1d_inv import MT1D, MT1DInverter
from mt2d_inv.MT2D import MT2DFD_Torch
```

Or use `pip install -e .` with [setup.py](setup.py).

---

## Quick Start: MT 1D Synthetic Example

```python
import os, sys
import torch
import numpy as np
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, "src"))
from mt1d_inv import MT1D, MT1DInverter

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    true_dz = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float32)
    true_sig = torch.tensor([0.01, 0.1, 0.01, 0.001], dtype=torch.float32)

    inv = MT1DInverter(device=device, use_sinkhorn=True, sinkhorn_dim=3, use_data_weighting=True)
    inv.generate_synthetic_data(true_dz=true_dz, true_sig=true_sig, freq_range=(-1, 4), n_freq=60, noise_level=0.05, seed=2025)
    inv.initialize_model(n_layers=true_sig.numel(), total_depth=float(true_dz.sum()), initial_sig=0.01, thickness_mode="equal")
    inv.setup_optimizer(lr=3e-3, reg_weight_sig=1e-4, optimizer_type="AdamW", p=2)
    inv.setup_constraints(constraint_type="roughness", use_occam_constraint=True, ref_weight=0.0, reference_sig=None)
    loss_history = inv.run_inversion(num_epochs=800, print_interval=50, use_adaptive_lambda=True, current_lambda=1e-4, warmup_epochs=50, update_interval=20)
    inv.plot_data_fit()
    inv.plot_model_comparison()

if __name__ == "__main__":
    main()
```

---

## Simple MT 2D Forward Example

```python
import numpy as np, torch
from mt2d_inv.MT2D import MT2DFD_Torch

zn = np.linspace(0, 2000, 41)
yn = np.linspace(-1000, 1000, 81)
nz, ny = len(zn) - 1, len(yn) - 1
sig = np.ones((nz, ny)) * 0.01
freq = np.logspace(-1, 3, 10)
ry = np.linspace(-500, 500, 21)
model = MT2DFD_Torch(nza=0, zn=zn, yn=yn, freq=freq, ry=ry, sig=sig, device="cpu")
result = model(mode="TETM")
rhoxy = result["rhoxy"].detach().cpu().numpy()
```

---

## Tests and Notebooks

Example Jupyter notebooks are in the `tests` directory. From the project root, run:

```bash
pip install -e .
pip install geomloss jupyter   # for OT and notebooks
```

Or in one command: `pip install -e ".[ot,dev]"`. Then open and run:

- [tests/test_mt1d/example_1d.ipynb](tests/test_mt1d/example_1d.ipynb) — 1D L2 + OT inversion
- [tests/test_mt2d/example_simple.ipynb](tests/test_mt2d/example_simple.ipynb) — 2D single-anomaly MSE vs OT
- [tests/test_mt2d/test_three_block.ipynb](tests/test_mt2d/test_three_block.ipynb) — 2D three-block MSE vs OT
- [tests/test_mt2d/test_CM2D-0.ipynb](tests/test_mt2d/test_CM2D-0.ipynb) — 2D CM2D-0 forward benchmark

---

## Contributors

- Authors: Xinran Liu, Xuanzhang Chen, Bo Yang, Ziyu Tang
- Contact: xinran.liu@zju.edu.cn, bo.yang@zju.edu.cn

---

## License

See [LICENSE](LICENSE) in the repository root.

---

---
# 中文版本 / Chinese Version
---

## MTinv_OT：基于最优传输（OT）的 MT 1D/2D 反演与正演工具箱

MTinv_OT 是一个专门面向大地电磁（Magnetotelluric, MT）问题的 Python 研究型工具箱，核心思想是：

> 在传统 MT 反演框架中，以几何最优传输（Optimal Transport, OT）距离（基于 geomloss 的 Sinkhorn 算子）替代/补充经典的 L2 数据拟合差，从而获得对噪声与非高斯误差更鲁棒的拟合方式。

围绕这一思想，工具箱实现了：

- **MT 1D OT 反演**：在一维分层模型中，将视电阻率与相位数据嵌入 3D 点云（logρ、归一化相位、logf）空间，借助 geomloss 的 `SamplesLoss(loss="sinkhorn")` 作为数据项，并联合 Occam 平滑约束、参考模型约束与自适应正则化权重；
- **MT 2D 有限差分正演**：TE/TM 总场法的二维有限差分正演算子，作为 2D OT 反演的“可微分前端”；
- **MT 2D 实验性 OT 反演**：将多频率、多台站、四分量（ρ/φ）的观测嵌入 3D/6D OT 空间，使用 Sinkhorn 距离实现多分量联合拟合，并叠加深度加权粗糙度与参考模型等约束；
- **一维/二维正则化与诊断工具**：粗糙度/曲率矩阵、χ² 统计、自适应 λ 更新、梯度历史与多种反演过程可视化。

所有模块基于 PyTorch 与 geomloss 实现，可在 CPU 或 GPU 上运行，便于与深度学习、联合反演或其他物理约束模型进行集成与对比实验。

---

## 功能概览

### 1. MT 1D 反演（mt1d_inv）

核心文件：[src/mt1d_inv/MTinv.py](src/mt1d_inv/MTinv.py)、[model.py](src/mt1d_inv/model.py)、[constraints.py](src/mt1d_inv/constraints.py)、[optimizer.py](src/mt1d_inv/optimizer.py)

主要特性：一维 MT 正反演；基于 geomloss 的 OT 数据拟合差；Occam 约束；参考模型约束；自适应正则化；误差传播与加权；完整诊断与可视化。

### 2. MT 2D 正演（mt2d_inv）

核心文件：[src/mt2d_inv/MT2D.py](src/mt2d_inv/MT2D.py)、[constraints.py](src/mt2d_inv/constraints.py)

主要特性：总场法 TE/TM 二维正演；PyTorch 自动求导；real-block 求解；1D 背景场边界条件。

### 3. MT 2D 反演（实验性）

核心文件：[src/mt2d_inv/MTinv_2d.py](src/mt2d_inv/MTinv_2d.py)

主要特性：TE/TM 全分量联合反演；3D/6D Sinkhorn OT；OT/MSE 模式切换；误差传播与 χ²；加权粗糙度与参考模型；自适应 λ；诊断可视化。

---

## 环境与依赖

Python ≥ 3.8，PyTorch，numpy，matplotlib；可选 geomloss（用于 Sinkhorn OT）。

---

## 安装与导入

克隆后，将 `src` 加入 Python 路径；或使用 `pip install -e .`。

---

## 快速上手、2D 正演与反演示例

参见上方英文版对应章节，代码逻辑相同。

---

## 测试与示例 Notebook

`tests` 目录下提供 Jupyter Notebook 示例。在项目根目录执行 `pip install -e .` 及 `pip install geomloss jupyter` 后，可直接打开并运行 `tests/test_mt1d/example_1d.ipynb`、`tests/test_mt2d/example_simple.ipynb`、`tests/test_mt2d/test_three_block.ipynb` 与 `tests/test_mt2d/test_CM2D-0.ipynb`。

---

## 贡献与致谢

- 作者：lxr, cxz
- 联系方式：xinran.liu@zju.edu.cn

欢迎在此基础上扩展更多功能或集成到更大的 MT 处理工作流中。

---

## License

见仓库根目录 [LICENSE](LICENSE)。
