"""
约束项和正则化模块
包含1D和2D反演中的各种约束项计算，2D反演理论 (deGroot-Hedlin & Constable, 1990)
"""

import torch
from typing import Optional, Tuple


class ConstraintCalculator:
    """
    约束项计算器
    支持1D和2D模型的各种约束类型
    """
    
    def __init__(self, device: str = "cpu"):
        """
        初始化约束计算器
        
        参数:
            device: 计算设备
        """
        self.device = device
    
    def build_roughness_matrix(self, n_layers: int) -> torch.Tensor:
        """
        构建1D粗糙度矩阵 (一阶差分)
        
        参数:
            n_layers: 层数
            
        返回:
            R: 粗糙度矩阵 [n_layers-1, n_layers]
        """
        R = torch.zeros((n_layers-1, n_layers), device=self.device)
        for i in range(n_layers-1):
            R[i, i] = -1
            R[i, i+1] = 1
        return R
    
    def build_curvature_matrix(self, n_layers: int) -> torch.Tensor:
        """
        构建1D曲率矩阵 (二阶差分)
        
        参数:
            n_layers: 层数
            
        返回:
            C: 曲率矩阵 [n_layers-2, n_layers]
        """
        C = torch.zeros((n_layers-2, n_layers), device=self.device)
        for i in range(n_layers-2):
            C[i, i] = 1
            C[i, i+1] = -2
            C[i, i+2] = 1
        return C
    
    def calculate_1d_model_norm(self, model: torch.Tensor, 
                                constraint_type: str = "roughness",
                                dz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算1D模型约束项
        
        参数:
            model: 模型参数 [n_layers]
            constraint_type: 约束类型 ("roughness" 或 "curvature")
            dz: 层厚度 [n_layers] (可选，用于深度加权)
            
        返回:
            norm: 约束项标量值
        """
        n_layers = len(model)
        
        if constraint_type == "roughness":
            R = self.build_roughness_matrix(n_layers)
            model_diff = R @ model
        elif constraint_type == "curvature":
            C = self.build_curvature_matrix(n_layers)
            model_diff = C @ model
        else:
            # 默认：与均值差
            model_diff = model - torch.mean(model)
        
        return torch.sum(model_diff ** 2)
    
    def calculate_2d_smoothness(self, model: torch.Tensor,
                               lateral_weight: float = 1.0,
                               vertical_weight: float = 1.0) -> torch.Tensor:
        """
        计算2D平滑度约束 (核心正则化)
        
        参数:
            model: 2D模型参数 [n_positions, n_layers+1]
            lateral_weight: 横向平滑权重
            vertical_weight: 纵向平滑权重
            
        返回:
            smooth_norm: 平滑度标量值
        """
        n_pos, n_lay = model.shape
        
        # 横向粗糙度 (沿测点方向)
        if n_pos > 1:
            lateral_diff = model[1:, :] - model[:-1, :]
            lateral_norm = torch.sum(lateral_diff ** 2)
        else:
            lateral_norm = torch.tensor(0.0, device=self.device)
        
        # 纵向粗糙度 (沿深度方向)
        if n_lay > 1:
            vertical_diff = model[:, 1:] - model[:, :-1]
            vertical_norm = torch.sum(vertical_diff ** 2)
        else:
            vertical_norm = torch.tensor(0.0, device=self.device)
        
        # 加权组合
        total_norm = (
            lateral_weight * lateral_norm +
            vertical_weight * vertical_norm
        )
        
        return total_norm
    


