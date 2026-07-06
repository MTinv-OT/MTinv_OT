import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import math
from typing import Any, Tuple, Optional

def visualize_1d_model(model, title="地球物理电性模型"):
    """
    可视化一维模型（从MT1D类移出）
    """
    # 计算各层深度
    depths = [0]
    for i, thickness in enumerate[Any](model.dz):
        depths.append(depths[i] + thickness.item())
    
    # 扩展电导率用于绘图
    sig_plot = []
    for i in range(model.n_layers + 1):
        sig_plot.append(model.sig[i].item())
        if i < model.n_layers:
            sig_plot.append(model.sig[i].item())
    
    depth_plot = [0]
    for depth in depths[1:]:
        depth_plot.append(depth)
        depth_plot.append(depth)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(sig_plot, depth_plot, 'r-', linewidth=2)
    plt.yscale('linear')
    plt.gca().invert_yaxis()
    plt.xlabel('sig (S/m)')
    plt.ylabel('dz (m)')
    plt.title("1D True model")   # 再设置标题
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 添加层边界线
    for depth in depths[1:-1]:
        plt.axhline(y=depth, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return plt

class MT1D_3DVisualizer:
    """
    MT 三维可视化类
    用于绘制视电阻率、相位和频率的三维点云图
    """
    
    # 常量定义
    MU = 4e-7 * math.pi  # 磁导率
    PI = math.pi
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MT 1D 正演计算
        
        参数:
            freq: 频率张量
            dz: 厚度张量
            sig: 电导率张量
            
        返回:
            zxy: 阻抗张量
            rho: 视电阻率张量
            phs: 相位张量
        """
        nf = len(freq)
        zxy = torch.zeros(nf, dtype=torch.complex64, device=self.device)
        rho = torch.zeros(nf, dtype=torch.float32, device=self.device)
        phs = torch.zeros(nf, dtype=torch.float32, device=self.device)
        
        n_layers = sig.shape[0]
        
        for kf in range(nf):
            omega = 2.0 * self.PI * freq[kf]
            
            # 计算半空间阻抗
            sqrt_arg = torch.complex(torch.tensor(0.0, device=self.device), -omega * self.MU) / sig[-1]
            Z = torch.sqrt(sqrt_arg)
            
            # 从底层向上递归计算阻抗
            for m in range(n_layers-2, -1, -1):
                km_arg = torch.complex(torch.tensor(0.0, device=self.device), omega * self.MU * sig[m])
                km = torch.sqrt(km_arg)
                
                Z0 = -1j * omega * self.MU / km
                R = torch.exp(-2.0 * km * dz[m]) * (Z - Z0) / (Z + Z0)
                Z = Z0 * (1.0 + R) / (1. - R)
            
            zxy[kf] = Z
            rho[kf] = torch.abs(Z)**2 / (omega * self.MU)
            phs[kf] = torch.atan2(Z.imag, Z.real) * 180.0 / self.PI
        
        return zxy, rho, phs
    
    def create_3d_pointcloud_from_data(self, freq: torch.Tensor, rho: torch.Tensor, phs: torch.Tensor, 
                                     title: str = "MT 3D Point Cloud") -> plt.Figure:
        """
        从视电阻率、相位和频率数据创建三维点云
        
        参数:
            freq: 频率数据
            rho: 视电阻率数据
            phs: 相位数据
            title: 图表标题
            
        返回:
            fig:  matplotlib图形对象
        """
        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_np = rho.detach().cpu().numpy() if torch.is_tensor(rho) else np.array(rho)
        phs_np = phs.detach().cpu().numpy() if torch.is_tensor(phs) else np.array(phs)
        
        # 创建三维图形
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 创建散点图，颜色映射根据频率
        scatter = ax.scatter(np.log10(rho_np), phs_np, np.log10(freq_np), 
                           c=np.log10(freq_np), cmap='jet', 
                           s=50, alpha=0.8, marker='o')
        
        # 设置坐标轴标签
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=15, labelpad=15)
        ax.set_ylabel('Phase [degrees]', fontsize=15, labelpad=15)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=15, labelpad=0, rotation=180)
        
        # 设置标题
        ax.set_title(title, fontsize=25, pad=10)
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('log10(Frequency [Hz])', fontsize=15)
        
        # 设置视角
        ax.view_init(elev=20, azim=30)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_3d_pointcloud_from_model(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor,
                                      title: str = "MT 3D Point Cloud from Model") -> plt.Figure:
        """
        从电阻率模型通过正演计算创建三维点云
        
        参数:
            freq: 频率数据
            dz: 层厚度
            sig: 电导率
            title: 图表标题
            
        返回:
            fig: matplotlib图形对象
        """
        # 正演计算
        zxy, rho, phs = self.mt1d_forward(freq, dz, sig)
        
        # 创建三维点云
        fig = self.create_3d_pointcloud_from_data(freq, rho, phs, title)
        
        return fig
    
    def plot_comparison_3d_pointcloud(self, freq: torch.Tensor, 
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT 3D Point Cloud Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制观测数据和模型预测数据的对比三维点云
        
        参数:
            freq: 频率数据
            rho_obs: 观测视电阻率
            phs_obs: 观测相位
            dz_model: 模型层厚度
            sig_model: 模型电导率
            title: 图表标题
            save_path: 图片保存路径（可选）
        返回:
            fig: matplotlib图形对象
        """
        # 正演计算模型预测数据
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)
        
        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)
        
        # 创建三维图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制观测数据点（红色）
        scatter_obs = ax.scatter(np.log10(rho_obs_np), phs_obs_np, np.log10(freq_np), 
                               c='red', s=60, alpha=0.8, marker='o', label='Observed Data')
        
        # 绘制模型预测数据点（蓝色）
        scatter_pred = ax.scatter(np.log10(rho_pred_np), phs_pred_np, np.log10(freq_np), 
                                c='blue', s=60, alpha=0.6, marker='^', label='Model Prediction')
        
        # 设置坐标轴标签
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=12, labelpad=15)
        ax.set_ylabel('Phase [degrees]', fontsize=12, labelpad=15,rotation=180)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=12, labelpad=1, rotation=0)

        # 设置标题
        ax.set_title(title, fontsize=14, pad=15)
        
        # 添加图例
        ax.legend(fontsize=12, loc='upper left')
        
        # 设置视角
        ax.view_init(elev=20, azim=30)
        
        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_comparison_3d_with_lines(self, freq: torch.Tensor, 
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT 3D Point Cloud with Connections",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制观测数据和模型预测数据的对比三维点云，并连接对应的点
        左右各增加一个空白子图
        """
        # 正演计算模型预测数据
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)

        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)

        # 创建GridSpec布局
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])

        # 左侧空白子图
        ax_left = fig.add_subplot(gs[0], frameon=False)
        ax_left.axis('off')

        # 主三维图
        ax = fig.add_subplot(gs[1], projection='3d')
        scatter_obs = ax.scatter(np.log10(rho_obs_np), phs_obs_np, np.log10(freq_np), 
                                c='red', s=60, alpha=0.8, marker='o', label='Observed Data')
        scatter_pred = ax.scatter(np.log10(rho_pred_np), phs_pred_np, np.log10(freq_np), 
                                c='blue', s=60, alpha=0.6, marker='^', label='Inverted Data')
        for i in range(len(freq_np)):
            ax.plot([np.log10(rho_obs_np[i]), np.log10(rho_pred_np[i])],
                    [phs_obs_np[i], phs_pred_np[i]],
                    [np.log10(freq_np[i]), np.log10(freq_np[i])],
                    'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=24, labelpad=25)
        ax.set_ylabel('Phase [degrees]', fontsize=24, labelpad=20)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=24, labelpad=15, rotation=180)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(title, fontsize=40)
        ax.legend(fontsize=18, loc='upper right')
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

        # 右侧空白子图
        ax_right = fig.add_subplot(gs[2], frameon=False)
        ax_right.axis('off')

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_rho_freq_2d(self, freq: torch.Tensor, rho: torch.Tensor, 
                        title: str = "Apparent Resistivity vs Frequency",
                        label: str = "Data", color: str = 'blue',
                        figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制视电阻率-频率二维图
        
        参数:
            freq: 频率数据
            rho: 视电阻率数据
            title: 图表标题
            label: 数据标签
            color: 线条颜色
            figsize: 图形尺寸
            
        返回:
            fig: matplotlib图形对象
        """
        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_np = rho.detach().cpu().numpy() if torch.is_tensor(rho) else np.array(rho)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制双对数坐标图
        ax.loglog(freq_np, rho_np, 'o-', color=color, markersize=6, linewidth=2, label=label)
        
        # 设置坐标轴标签
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Apparent Resistivity [Ω·m]', fontsize=12)
        
        # 设置标题
        ax.set_title(title, fontsize=14)
        
        # 添加网格
        ax.grid(True, which='both', alpha=0.3)
        
        # 添加图例
        if label:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig

    # 文件：src/mt1d_inv/visualize.py

    def plot_sensitivity_matrix(self, J, z_grid, freqs, save_path=None):
        """
        绘制灵敏度矩阵热力图
        Args:
            J: 灵敏度矩阵 [n_freq, n_layer]
            z_grid: 深度网格 [n_layer + 1]
            freqs: 频率列表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 使用 pcolormesh 或 imshow
        # 注意：通常横轴是深度(层)，纵轴是频率(周期)
        
        # 为了画图方便，构建网格
        # y轴：频率索引 (0 到 n_freq)
        # x轴：层索引 (0 到 n_layer)
        
        im = ax.imshow(J, aspect='auto', cmap='RdBu_r', origin='upper',
                       extent=[0, len(z_grid)-1, 0, len(freqs)])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sensitivity (Partial Derivative)')
        
        # 设置轴标签
        ax.set_ylabel('Frequency Index (High -> Low)')
        ax.set_xlabel('Layer Index (Shallow -> Deep)')
        ax.set_title('Sensitivity Matrix (Jacobian)')
        
        # 可选：把刻度换成真实的频率和深度数值（稍微麻烦点，看你需要不需要）
        # ax.set_yticks(...)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def plot_phs_freq_2d(self, freq: torch.Tensor, phs: torch.Tensor,
                        title: str = "Phase vs Frequency",
                        label: str = "Data", color: str = 'red',
                        figsize: tuple = (10, 6)) -> plt.Figure:
        """
        绘制相位-频率二维图
        
        参数:
            freq: 频率数据
            phs: 相位数据
            title: 图表标题
            label: 数据标签
            color: 线条颜色
            figsize: 图形尺寸
            
        返回:
            fig: matplotlib图形对象
        """
        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        phs_np = phs.detach().cpu().numpy() if torch.is_tensor(phs) else np.array(phs)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制半对数坐标图（频率对数，相位线性）
        ax.semilogx(freq_np, phs_np, 's-', color=color, markersize=5, linewidth=2, label=label)
        
        # 设置坐标轴标签
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Phase [degrees]', fontsize=12)
        
        # 设置标题
        ax.set_title(title, fontsize=14)
        
        # 添加网格
        ax.grid(True, which='both', alpha=0.3)
        
        # 添加图例
        if label:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig

    def plot_comparison_2d(self, freq: torch.Tensor,
                          rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                          rho_pred: torch.Tensor, phs_pred: torch.Tensor,
                          title: str = "MT Data Comparison") -> plt.Figure:
        """
        绘制观测数据和预测数据的二维对比图
        
        参数:
            freq: 频率数据
            rho_obs: 观测视电阻率
            phs_obs: 观测相位
            rho_pred: 预测视电阻率
            phs_pred: 预测相位
            title: 图表标题
            
        返回:
            fig: matplotlib图形对象
        """
        # 转换为numpy数组
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)
        
        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制视电阻率对比
        ax1.loglog(freq_np, rho_obs_np, 'ro-', markersize=6, linewidth=2, label='Observed')
        ax1.loglog(freq_np, rho_pred_np, 'b^-', markersize=5, linewidth=2, label='Inverted')
        ax1.set_xlabel('Frequency [Hz]', fontsize=20)
        ax1.set_ylabel('Apparent Resistivity [Ω·m]', fontsize=20)
        ax1.set_title('Apparent Resistivity Comparison', fontsize=25)
        ax1.legend(fontsize=20)
        ax1.tick_params(axis='both', labelsize=16)  # 坐标数字字体
        
        # 绘制相位对比
        ax2.semilogx(freq_np, phs_obs_np, 'ro-', markersize=6, linewidth=2, label='Observed')
        ax2.semilogx(freq_np, phs_pred_np, 'b^-', markersize=5, linewidth=2, label='Inverted')
        ax2.set_xlabel('Frequency [Hz]', fontsize=20)
        ax2.set_ylabel('Phase [degrees]', fontsize=20)
        ax2.legend(fontsize=20)
        ax2.tick_params(axis='both', labelsize=16)  # 坐标数字字体
        
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        # 设置总标题
        fig.suptitle(title, fontsize=30, y=0.98)
        
        plt.tight_layout()
        return fig

    def plot_comparison_2d_from_model(self, freq: torch.Tensor,
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT Data Comparison with Model") -> plt.Figure:
        """
        从模型预测数据绘制二维对比图
        
        参数:
            freq: 频率数据
            rho_obs: 观测视电阻率
            phs_obs: 观测相位
            dz_model: 模型层厚度
            sig_model: 模型电导率
            title: 图表标题
            
        返回:
            fig: matplotlib图形对象
        """
        # 正演计算模型预测数据
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)
        
        # 绘制对比图
        fig = self.plot_comparison_2d(freq, rho_obs, phs_obs, rho_pred, phs_pred, title)
        
        return fig



