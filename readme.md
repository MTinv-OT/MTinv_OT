## MTinv_OT：基于最优传输（OT）的 MT 1D/2D 反演与正演工具箱

MTinv_OT 是一个专门面向大地电磁（Magnetotelluric, MT）问题的 Python 研究型工具箱，核心思想是：

> 在传统 MT 反演框架中，以几何最优传输（Optimal Transport, OT）距离（基于 geomloss 的 Sinkhorn 算子）替代/补充经典的 L2 数据失配，从而获得对噪声与非高斯误差更鲁棒的拟合方式。

围绕这一思想，工具箱实现了：

- **MT 1D OT 反演**：在一维分层模型中，将视电阻率与相位数据嵌入 3D 点云（logρ、归一化相位、logf）空间，借助 geomloss 的 `SamplesLoss(loss="sinkhorn")` 作为数据项，并联合 Occam 平滑约束、参考模型约束与自适应正则化权重；
- **MT 2D 有限差分正演**：TE/TM 总场法的二维有限差分正演算子，作为 2D OT 反演的“可微分前端”；
- **MT 2D 实验性 OT 反演**：将多频率、多台站、四分量（ρ/φ）的观测嵌入 3D/6D OT 空间，使用 Sinkhorn 距离实现多分量联合拟合，并叠加深度加权粗糙度与参考模型等约束；
- **一维/二维正则化与诊断工具**：粗糙度/曲率矩阵、χ² 统计、自适应 λ 更新、梯度历史与多种反演过程可视化。

所有模块基于 PyTorch 与 geomloss 实现，可在 CPU 或 GPU 上运行，便于与深度学习、联合反演或其他物理约束模型进行集成与对比实验。

---

## 功能概览

### 1. MT 1D 反演（mt1d_inv）

核心文件：

- [src/mt1d_inv/MTinv.py](src/mt1d_inv/MTinv.py)：主反演类 `MT1DInverter`
- [src/mt1d_inv/model.py](src/mt1d_inv/model.py)：一维电性模型 `MT1D`
- [src/mt1d_inv/constraints.py](src/mt1d_inv/constraints.py)：粗糙度 / 曲率等约束
- [src/mt1d_inv/optimizer.py](src/mt1d_inv/optimizer.py)：优化器与损失配置

主要特性：

- 一维 MT 正反演：实现经典分层地球 MT 1D 正演与参数反演；
- 基于 geomloss 的 OT 数据失配：使用 `SamplesLoss(loss="sinkhorn")` 在 3D 点云空间（logρ、归一化相位、logf）上度量预测与观测的 OT 距离，可与传统 L2/L1 损失切换对比；
- Occam 约束：支持粗糙度 / 曲率等模型平滑约束，接近 de Groot-Hedlin & Constable (1990) 思想；
- 参考模型约束：在损失函数中加入参考模型罚项，使结果向先验模型靠拢；
- 自适应正则化权重：基于数据项与模型项梯度平衡自适应调整正则化系数 λ；
- 误差传播与加权：从阻抗噪声推导视电阻率和相位误差，自动构造权重与 χ² 统计量；
- 完整的诊断与可视化：损失曲线、χ² 演化、OT/L2 拟合效果对比以及模型对比图。

### 2. MT 2D 正演（mt2d_inv）

核心文件：

- [src/mt2d_inv/MT2D.py](src/mt2d_inv/MT2D.py)：二维有限差分正演类 `MT2DFD_Torch`
- [src/mt2d_inv/constraints.py](src/mt2d_inv/constraints.py)：二维模型平滑约束（横向 / 纵向）

主要特性：

- 总场法二维 TE/TM 模式 MT 正演，使用复数有限差分离散
- 使用 PyTorch 自动求导与线性代数库，可运行在 GPU
- 稳定的复数线性方程求解（real-block 转换）
- 兼容一维 1D 背景场插值，用于边界条件处理

### 3. MT 2D 反演（实验性，基于 OT）

核心文件：

- [src/mt2d_inv/MTinv_2d.py](src/mt2d_inv/MTinv_2d.py)：二维 MT 反演器 `MT2DInverter`

主要特性：

- TE/TM 全分量联合反演（`rhoxy / phsxy / rhoyx / phsyx`）；
- 使用 geomloss-Sinkhorn 的 3D/6D OT 点云损失：
	- 3D：单分量 OT，点为 `[freq, station, value]`；
	- 6D：多分量 OT，点为 `[freq, station, rhoxy, phsxy, rhoyx, phsyx]`；
- 支持 OT 模式（`3dot` / `6dot`）与加权 MSE 模式（`mse`）切换，便于对比 OT 与传统 L2 的表现；
- 从阻抗误差出发的二维误差传播与 χ² 统计（`compute_rms_chi2`），保证 OT/L2 都在“物理合理”的误差尺度下评估；
- 加权粗糙度 + 参考模型约束：
	- 深度加权的一阶平滑约束（`calculate_weighted_roughness`）；
	- 可选参考模型项（`calculate_combined_constraint`）；
- 自适应正则化参数 λ（`update_lambda_by_gradient_balance`），只允许 λ 单调减小，使约束逐步放松；
- 反演过程诊断：损失分解、λ 演化、梯度范数历史、数据拟合与 1D 剖面/2D 剖面可视化。

---

## 环境与依赖

建议环境：

- Python ≥ 3.8
- PyTorch（CPU 或 GPU 版本均可）
- numpy
- matplotlib

可选依赖（与 OT 相关）：

- geomloss：提供 `SamplesLoss(loss="sinkhorn")`，用于计算 OT / Sinkhorn 损失；

使用 pip 安装基础依赖示例：

```bash
pip install torch numpy matplotlib

# 若需要使用 Sinkhorn OT 损失
pip install geomloss
```

---

## 安装与导入方式

当前仓库未发布到 PyPI，可通过源码直接使用。

克隆仓库后，在项目根目录（包含 src 文件夹的位置）下，将 src 加入 Python 路径即可：

```python
import sys, os

project_root = os.path.dirname(__file__)  # 根据你的运行脚本位置调整
sys.path.append(os.path.join(project_root, "src"))

from mt1d_inv import MT1D, MT1DInverter
from mt2d_inv.MT2D import MT2DFD_Torch
```

也可以自行完善 [setup.py](setup.py)，通过 `pip install -e .` 的方式以包形式安装。

---

## 快速上手：MT 1D 合成数据与反演示例

下面给出一个最小可运行的一维示例流程，演示如何生成合成数据并进行反演。

```python
import os, sys
import torch
import numpy as np

# 确保可以找到 src 目录
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, "src"))

from mt1d_inv import MT1D, MT1DInverter


def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# 1. 构造真实模型（厚度 dz，电导率 sig）
	true_dz = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float32)
	true_sig = torch.tensor([0.01, 0.1, 0.01, 0.001], dtype=torch.float32)  # 层数 = len(dz)+1

	# 可选：使用 MT1D 封装模型
	true_model = MT1D(true_dz, true_sig)
	print("True model:", true_model)

	# 2. 初始化反演器（可选 Sinkhorn OT）
	inv = MT1DInverter(device=device,
					   use_sinkhorn=True,   # True 使用 OT；False 使用传统 L2/L1
					   sinkhorn_dim=3,
					   use_data_weighting=True)

	# 3. 生成带噪声的合成数据
	inv.generate_synthetic_data(true_dz=true_dz,
								true_sig=true_sig,
								freq_range=(-1, 4),  # 0.1–10000 Hz
								n_freq=60,
								noise_level=0.05,
								seed=2025)

	# 4. 初始化反演模型（层数需与 true_sig 一致）
	inv.initialize_model(n_layers=true_sig.numel(),
						 total_depth=float(true_dz.sum()),
						 initial_sig=0.01,
						 thickness_mode="equal")

	# 5. 设置优化器
	inv.setup_optimizer(lr=3e-3,
						reg_weight_sig=1e-4,
						optimizer_type="AdamW",
						p=2)

	# 6. 设置约束（Occam + 可选参考模型）
	inv.setup_constraints(constraint_type="roughness",   # 或 "curvature"
						  use_occam_constraint=True,
						  ref_weight=0.0,                 # 若 >0 则使用参考模型罚项
						  reference_sig=None)

	# 7. 运行反演
	loss_history = inv.run_inversion(num_epochs=800,
									 print_interval=50,
									 use_adaptive_lambda=True,
									 current_lambda=1e-4,
									 warmup_epochs=50,
									 update_interval=20)

	# 8. 可视化
	inv.plot_data_fit()            # 观测 vs 反演响应（ρ 和相位）
	inv.plot_model_comparison()    # 真实模型 vs 反演模型


if __name__ == "__main__":
	main()
```

常见可调参数：

- `use_sinkhorn`：是否使用 OT 损失；关闭时使用加权 L2/L1
- `noise_level`：合成数据噪声水平（相对阻抗幅度的比例）
- `constraint_type`：`"roughness"` 或 `"curvature"`
- `use_adaptive_lambda` 与 `current_lambda`：控制正则化系数自适应方案

更多高级绘图（χ² 演化、梯度平衡等）可参考 [src/mt1d_inv/MTinv.py](src/mt1d_inv/MTinv.py) 中的 `plot_chi2_history`、`plot_gradient_history` 等函数。

---

## 简单示例：MT 2D 正演

二维正演入口类为 `MT2DFD_Torch`，位于 [src/mt2d_inv/MT2D.py](src/mt2d_inv/MT2D.py)。下面给出一个极简骨架示例，展示基本调用方式：

```python
import os, sys
import numpy as np
import torch

project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, "src"))

from mt2d_inv.MT2D import MT2DFD_Torch


def run_mt2d_forward():
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# 1. 构建网格（深度 zn、水平距离 yn，单位 m）
	zn = np.linspace(0, 2000, 41)   # 40 层
	yn = np.linspace(-1000, 1000, 81)  # 80 个水平单元

	# 2. 构造二维电导率模型 sig[z, y]
	nz, ny = len(zn) - 1, len(yn) - 1
	sig = np.ones((nz, ny)) * 0.01  # 均匀半空间

	# 3. 频率与测点位置
	freq = np.logspace(-1, 3, 10)   # 0.1–1000 Hz
	ry = np.linspace(-500, 500, 21) # 测点水平位置

	# 4. 初始化正演器
	model = MT2DFD_Torch(nza=0,     # 地表层索引（可根据需要调整）
						 zn=zn,
						 yn=yn,
						 freq=freq,
						 ry=ry,
						 sig=sig,
						 device=device)

	# 5. 计算 TE/TM 模式响应
	result = model(mode="TETM")
	rhoxy = result["rhoxy"].detach().cpu().numpy()  # TE 视电阻率
	phsxy = result["phsxy"].detach().cpu().numpy()

	print("rhoxy shape:", rhoxy.shape)  # (nf, nry)


if __name__ == "__main__":
	run_mt2d_forward()
```

在真实应用中，你可以：

- 将 `sig` 替换为更复杂的二维结构（如异常体、电阻率梯度等）
- 更精细地控制 `nza`、网格加密方式和边界条件
- 在 2D 上构建类似 1D 的正则化与反演流程（参考 mt1d_inv 的设计）

---

## 示例：MT 2D 反演（基于 MT2DInverter）

下面给出一个简化的二维 OT 反演示例骨架，基于 [src/mt2d_inv/MTinv_2d.py](src/mt2d_inv/MTinv_2d.py) 中的 `MT2DInverter` 类。实际工程中可根据需要替换为实测数据、加入更复杂的正则化配置。

```python
import os, sys
import numpy as np
import torch

project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, "src"))

from mt2d_inv import MT2DFD_Torch, MT2DInverter


def run_mt2d_inversion():

	device = "cuda" if torch.cuda.is_available() else "cpu"

	# 1. 构建网格与观测配置
	zn = torch.linspace(0.0, 2000.0, 41)      # 深度节点 (m)
	yn = torch.linspace(-10000.0, 10000.0, 81) # 水平节点 (m)

	freqs = torch.logspace(-1, 3, 20)         # 频率 (Hz)
	stations = torch.linspace(-8000.0, 8000.0, 21)  # 台站位置 (m)

	# 2. 构建一个简单的“真实”模型 sig_true
	nz, ny = len(zn) - 1, len(yn) - 1
	sig_true = torch.ones((nz, ny)) * 0.01    # 均匀半空间

	# 可以在这里插入异常体，例如：
	# sig_true[10:20, 30:40] = 0.1

	# 3. 初始化反演器
	inverter = MT2DInverter(
		yn=yn,
		zn=zn,
		freqs=freqs,
		stations=stations,
		device=device,
		ot_options={"p": 2, "blur": 0.01, "backend": "tensorized"},
	)

	# 将真实模型传入（用于合成数据和绘图对比）
	inverter.sig_true = sig_true.to(device)

	# 4. 设置正演算子（内部会构造 MT2DFD_Torch）
	inverter.set_forward_operator(nza=0)

	# 5. 生成带噪声的二维合成数据
	inverter.create_synthetic_data(noise_level=0.05)

	# 6. 运行反演（这里只是一个示意配置）
	_ = inverter.run_inversion(
		n_epochs=100,
		mode="6dot",           # '3dot' / '6dot' / 'mse'
		lr=0.05,
		current_lambda=0.01,
		use_adaptive_lambda=True,
		warmup_epochs=50,
		update_interval=20,
		use_reference_model=False,
	)

	# 7. 绘制结果（模型对比 / 初始模型 / 损失与 λ 演化等）
	inverter.plot_model_comparison()
	inverter.plot_initial_model()
	inverter.plot_loss_history()
	inverter.plot_gradient_history()


if __name__ == "__main__":
	run_mt2d_inversion()
```

注意：

- 示例中使用 `create_synthetic_data` 生成合成观测；若使用实测数据，可直接填充 `inverter.obs_data` 结构（`rhoxy / phsxy / rhoyx / phsyx`）。
- `mode` 参数决定 OT/MSE 形式，推荐噪声水平较高时优先尝试 OT 模式（`3dot`/`6dot`）。
- 2D 反演模块目前仍偏实验性，参数（OT、正则化、λ 自适应等）需要结合具体模型和噪声水平做调优。

---

## 测试与示例 Notebook

在 tests 目录中提供了一些 Jupyter Notebook 示例，可根据这些 Notebook 进一步理解参数设置与反演行为。

---

## 贡献与致谢

- 作者：lxr, cxz
- 联系方式：xinran.liu@zju.edu.cn

欢迎在此基础上扩展更多维度的联合反演、各向异性、各向同性约束等功能，或将其集成到更大的 MT 处理工作流中。

---

## License

See [LICENSE](LICENSE) in the repository root.
