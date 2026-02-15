"""
地球物理电性模型定义模块
用于定义地层厚度和电导率参数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict
import math

class MT1D:
    """
    地球物理电性模型类
    
    属性:
        dz (torch.Tensor): 各层厚度（米）
        sig (torch.Tensor): 各层电导率（S/m）
        n_layers (int): 层数
    """
    
    def __init__(self, dz, sig):
        """
        初始化电性模型
        
        参数:
            dz (torch.Tensor or list): 各层厚度
            sig (torch.Tensor or list): 各层电导率
        """
        self.dz = torch.tensor(dz, dtype=torch.float32)
        self.sig = torch.tensor(sig, dtype=torch.float32)
        
        # 验证参数
        if len(self.dz) + 1 != len(self.sig):
            raise ValueError("电导率数量应该比厚度数量多1（包括顶层和底层半空间）")
    
    @property
    def n_layers(self):
        """返回模型层数（不包括半空间）"""
        return len(self.dz)
    
    def __repr__(self):
        return f"GeoElectricModel(dz={self.dz.tolist()}, sig={self.sig.tolist()})"
    
    def to_dict(self):
        """将模型参数转换为字典"""
        return {
            'dz': self.dz.tolist(),
            'sig': self.sig.tolist()
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建模型"""
        return cls(data['dz'], data['sig'])
    
    def visualize(self, title="地球物理电性模型"):
        """
        可视化模型
        
        参数:
            title (str): 图表标题
        """
        # 计算各层深度
        depths = [0]
        for i, thickness in enumerate(self.dz):
            depths.append(depths[i] + thickness.item())
        
        # 扩展电导率用于绘图
        sig_plot = []
        for i in range(self.n_layers + 1):
            sig_plot.append(self.sig[i].item())
            if i < self.n_layers:
                sig_plot.append(self.sig[i].item())
        
        depth_plot = [0]
        for depth in depths[1:]:
            depth_plot.append(depth)
            depth_plot.append(depth)
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(sig_plot, depth_plot, 'r-', linewidth=2)
        plt.yscale('linear')
        plt.gca().invert_yaxis()
        plt.xlabel('电导率 (S/m)')
        plt.ylabel('深度 (m)')
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # 添加层边界线
        for depth in depths[1:-1]:
            plt.axhline(y=depth, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return plt
