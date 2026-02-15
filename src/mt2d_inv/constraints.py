"""
约束项和正则化模块
包含1D和2D反演中的各种约束项计算方法
"""

import torch
from typing import Optional


class ConstraintCalculator:
    """
    约束项计算器
    支持1D和2D模型的各种约束类型
    """
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float, device: str = "cpu"):
        """
        初始化约束计算器，存储网格信息
        """
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.device = device
    
    
    def calculate_weighted_roughness(self, model_log_sigma, weights=None, norm_type="L2"):
        """
        带权重的粗糙度计算 (支持 L1 或 L2 范数)
        
        Args:
            model_log_sigma: 当前模型的对数电导率
            weights: 权重矩阵，默认为 None
            norm_type: 范数类型，"L1" 或 "L2" (默认 "L1")
            
        Returns:
            计算得到的加权粗糙度
        """
        # 1. 计算差分 (梯度)
        diff_x = (model_log_sigma[:, 1:] - model_log_sigma[:, :-1])
        diff_z = (model_log_sigma[1:, :] - model_log_sigma[:-1, :])
        
        # 2. 如果提供了权重
        if weights is not None:
            # 权重取相邻节点的平均
            w_x = (weights[:, 1:] + weights[:, :-1]) * 0.5
            w_z = (weights[1:, :] + weights[:-1, :]) * 0.5
            
            # 将权重应用到差分上
            diff_x = diff_x * w_x
            diff_z = diff_z * w_z
        
        # 3. 根据 norm_type 选择 L1 或 L2 范数
        if norm_type == "L1":
            # L1 范数：使用绝对值
            loss_x = torch.sum(torch.abs(diff_x))
            loss_z = torch.sum(torch.abs(diff_z))
        elif norm_type == "L2":
            # L2 范数：平方和开方
            loss_x = torch.sum(diff_x ** 2)
            loss_z = torch.sum(diff_z ** 2)
        else:
            raise ValueError("Unsupported norm_type. Please choose 'L1' or 'L2'.")
        
        # 正则化项 - 加入一个非常小的常数来防止矩阵奇异
        epsilon = 1e-12  # 小的正则化常数
        loss_x += epsilon * torch.sum(diff_x ** 2)  # 加强正则化
        loss_z += epsilon * torch.sum(diff_z ** 2)  # 加强正则化
    
        return loss_x + loss_z
    
    def calculate_reference_model_constraint(self, 
                                            model_log_sigma: torch.Tensor,
                                            reference_model_log_sigma: torch.Tensor,
                                            weights: Optional[torch.Tensor] = None,
                                            norm_type: str = "L2") -> torch.Tensor:
        """
        计算参考模型约束项
        约束模型不要偏离参考模型太远
        
        Args:
            model_log_sigma: 当前模型的对数电导率 [nz, nx]
            reference_model_log_sigma: 参考模型的对数电导率 [nz, nx]
            weights: 权重矩阵 [nz, nx]，用于空间加权（可选）
            norm_type: 范数类型，"L1" 或 "L2" (默认 "L2")
            
        Returns:
            计算得到的参考模型约束项
        """
        # 1. 计算模型与参考模型的偏差
        diff = model_log_sigma - reference_model_log_sigma
        
        # 2. 如果提供了权重，应用空间加权
        if weights is not None:
            diff = diff * weights
        
        # 3. 根据 norm_type 选择 L1 或 L2 范数
        if norm_type == "L1":
            # L1 范数：使用绝对值
            loss = torch.sum(torch.abs(diff))
        elif norm_type == "L2":
            # L2 范数：平方和
            loss = torch.sum(diff ** 2)
        else:
            raise ValueError("Unsupported norm_type. Please choose 'L1' or 'L2'.")
        
        return loss
    
    def calculate_combined_constraint(self,
                                     model_log_sigma: torch.Tensor,
                                     reference_model_log_sigma: Optional[torch.Tensor] = None,
                                     roughness_weights: Optional[torch.Tensor] = None,
                                     reference_weights: Optional[torch.Tensor] = None,
                                     roughness_norm: str = "L2",
                                     reference_norm: str = "L2",
                                     reference_weight: float = 0.0) -> torch.Tensor:
        """
        计算组合约束项：粗糙度约束 + 参考模型约束
        
        Args:
            model_log_sigma: 当前模型的对数电导率 [nz, nx]
            reference_model_log_sigma: 参考模型的对数电导率 [nz, nx] (可选)
            roughness_weights: 粗糙度计算的权重矩阵 (可选)
            reference_weights: 参考模型约束的权重矩阵 (可选)
            roughness_norm: 粗糙度范数类型，"L1" 或 "L2"
            reference_norm: 参考模型约束范数类型，"L1" 或 "L2"
            reference_weight: 参考模型约束的权重系数 (0.0 表示不使用参考模型约束)
            
        Returns:
            组合约束项值
        """
        # 1. 计算粗糙度约束
        roughness_loss = self.calculate_weighted_roughness(
            model_log_sigma, roughness_weights, roughness_norm
        )
        
        # 2. 如果提供了参考模型且权重 > 0，计算参考模型约束
        if reference_model_log_sigma is not None and reference_weight > 0.0:
            reference_loss = self.calculate_reference_model_constraint(
                model_log_sigma, reference_model_log_sigma, 
                reference_weights, reference_norm
            )
            return roughness_loss + reference_weight * reference_loss
        else:
            return roughness_loss
    
    