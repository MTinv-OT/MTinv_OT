"""
优化器配置和损失函数模块（改进版）

改进点:
1. ✅ 添加学习率调度器
2. ✅ 更好的参数初始化建议
3. ✅ 梯度监控功能
4. ✅ 可选的geomloss依赖
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict

# 可选的geomloss导入
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("⚠️  geomloss未安装,Sinkhorn损失不可用")


class OptimizerConfig:
    """
    优化器配置类（改进版）
    
    新增功能:
    - 学习率调度器
    - 梯度监控
    - 自适应优化器选择
    """
    
    def __init__(self, device: str = "cpu"):
        """初始化"""
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.optimizer_type = None
        self.learning_rate = None
        self.sinkhorn_loss = None
        self.data_loss_fn = None
        
        # 梯度监控
        self.grad_history = []
    
    def create_optimizer(self, 
                        params: List[torch.Tensor],
                        optimizer_type: str = "Adam",
                        lr: float = 0.01,
                        weight_decay: float = 0.0,
                        betas: Tuple[float, float] = (0.9, 0.999),
                        eps: float = 1e-8,
                        momentum: float = 0.9) -> optim.Optimizer:
        """
        创建优化器
        
        参数:
            params: 待优化参数
            optimizer_type: 'Adam', 'AdamW', 'SGD', 'RMSprop', 'LBFGS'
            lr: 学习率
            weight_decay: L2正则化系数
            betas: Adam系列的动量参数
            eps: 数值稳定性
            momentum: SGD/RMSprop的动量
        
        返回:
            optimizer: 优化器实例
        """
        if optimizer_type == "Adam":
            optimizer = optim.Adam(
                params, lr=lr, betas=betas, 
                eps=eps, weight_decay=weight_decay
            )
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params, lr=lr, betas=betas,
                eps=eps, weight_decay=weight_decay
            )
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(
                params, lr=lr, momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(
                params, lr=lr, momentum=momentum,
                eps=eps, weight_decay=weight_decay
            )
        elif optimizer_type == "LBFGS":
            optimizer = optim.LBFGS(
                params, lr=lr, max_iter=20,
                history_size=10, line_search_fn="strong_wolfe"
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        
        self.optimizer = optimizer
        self.optimizer_type = optimizer_type
        self.learning_rate = lr
        
        print(f"✓ 创建优化器: {optimizer_type}")
        print(f"  - 学习率: {lr}")
        print(f"  - 权重衰减: {weight_decay}")
        
        return optimizer
    
    def create_scheduler(self,
                        scheduler_type: str = "ReduceLROnPlateau",
                        factor: float = 0.5,
                        patience: int = 20,
                        min_lr: float = 1e-6,
                        **kwargs):
        """
        创建学习率调度器
        
        参数:
            scheduler_type: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'
            factor: 学习率衰减因子
            patience: 容忍轮数
            min_lr: 最小学习率
        
        返回:
            scheduler: 调度器实例
        """
        if self.optimizer is None:
            raise ValueError("请先创建优化器")
        
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )
        elif scheduler_type == "StepLR":
            step_size = kwargs.get('step_size', 50)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=factor
            )
        elif scheduler_type == "CosineAnnealing":
            T_max = kwargs.get('T_max', 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=min_lr
            )
        elif scheduler_type == "ExponentialLR":
            gamma = kwargs.get('gamma', 0.95)
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        else:
            raise ValueError(f"不支持的调度器: {scheduler_type}")
        
        self.scheduler = scheduler
        print(f"✓ 创建学习率调度器: {scheduler_type}")
        
        return scheduler
    
    def create_sinkhorn_loss(self,
                            p: int = 2,
                            blur: float = 0.08,
                            scaling: float = 0.9,
                            reach: Optional[float] = None,
                            debias: bool = True,
                            backend: str = "tensorized") -> Optional[SamplesLoss]:
        """
        创建Sinkhorn损失（如果geomloss可用）
        
        参数:
            p: 距离的p范数
            blur: 模糊参数(熵正则化)
            scaling: Sinkhorn迭代缩放
            reach: 非平衡OT松弛距离。None=平衡OT(质量守恒,默认); >0(如1.0)=非平衡OT(不要求质量守恒)
            debias: 是否去偏
            backend: 'tensorized', 'online', 'multiscale'
        
        返回:
            sinkhorn_loss: Sinkhorn损失实例或None
        """
        if not GEOMLOSS_AVAILABLE:
            print("⚠️  geomloss未安装,无法使用Sinkhorn损失")
            return None
        
        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            reach=reach,
            debias=debias,
            backend=backend
        )
        
        self.sinkhorn_loss = sinkhorn_loss
        
        print(f"✓ 创建Sinkhorn损失:")
        print(f"  - p范数: {p}")
        print(f"  - blur: {blur}")
        print(f"  - scaling: {scaling}")
        print(f"  - reach: {reach} ({'非平衡/不质量守恒' if reach else '平衡/质量守恒'})")
        
        return sinkhorn_loss
    
    def create_data_loss(self, p: int = 2) -> nn.Module:
        """
        创建标准数据损失
        
        参数:
            p: 1=L1, 2=MSE
        
        返回:
            loss_fn: 损失函数
        """
        if p == 1:
            loss_fn = nn.L1Loss()
            print("✓ 创建L1损失")
        else:
            loss_fn = nn.MSELoss()
            print("✓ 创建MSE损失")
        
        self.data_loss_fn = loss_fn
        return loss_fn
    
    def clip_gradients(self, 
                      parameters: List[torch.Tensor],
                      max_norm: float = 10.0,
                      record: bool = False) -> float:
        """
        梯度裁剪
        
        参数:
            parameters: 参数列表
            max_norm: 最大梯度范数
            record: 是否记录梯度历史
        
        返回:
            grad_norm: 裁剪前的梯度范数
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        
        if record:
            self.grad_history.append(grad_norm.item())
        
        return grad_norm.item()
    
    def clamp_parameters(self,
                        parameters: torch.Tensor,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None,
                        use_log_space: bool = False) -> None:
        """
        参数范围约束
        
        参数:
            parameters: 参数张量
            min_val: 最小值
            max_val: 最大值
            use_log_space: 是否在对数空间
        """
        with torch.no_grad():
            if min_val is not None:
                parameters.clamp_(min=min_val)
            if max_val is not None:
                parameters.clamp_(max=max_val)
    
    def step_scheduler(self, metric: Optional[float] = None):
        """
        学习率调度器步进
        
        参数:
            metric: 监控指标(用于ReduceLROnPlateau)
        """
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]['lr']
    
    def get_grad_stats(self) -> Dict[str, float]:
        """获取梯度统计信息"""
        if not self.grad_history:
            return {}
        
        import numpy as np
        grad_array = np.array(self.grad_history)
        
        return {
            'mean': float(np.mean(grad_array)),
            'std': float(np.std(grad_array)),
            'min': float(np.min(grad_array)),
            'max': float(np.max(grad_array)),
            'last': float(grad_array[-1])
        }
    
    @staticmethod
    def suggest_learning_rate(parameter_count: int, 
                             optimizer_type: str = "Adam") -> float:
        """
        根据参数数量建议学习率
        
        参数:
            parameter_count: 参数总数
            optimizer_type: 优化器类型
        
        返回:
            suggested_lr: 建议的学习率
        """
        if optimizer_type in ["Adam", "AdamW"]:
            if parameter_count < 1000:
                return 0.01
            elif parameter_count < 10000:
                return 0.005
            else:
                return 0.001
        elif optimizer_type == "SGD":
            if parameter_count < 1000:
                return 0.1
            elif parameter_count < 10000:
                return 0.05
            else:
                return 0.01
        else:
            return 0.01
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "="*50)
        print("优化器配置")
        print("="*50)
        print(f"优化器类型: {self.optimizer_type}")
        print(f"学习率: {self.learning_rate}")
        print(f"当前学习率: {self.get_current_lr()}")
        print(f"调度器: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"数据损失: {type(self.data_loss_fn).__name__ if self.data_loss_fn else 'None'}")
        print(f"Sinkhorn损失: {'可用' if self.sinkhorn_loss else '不可用'}")
        print("="*50)


# 测试代码
if __name__ == "__main__":
    print("优化器配置测试\n")
    
    # 创建配置
    config = OptimizerConfig(device='cpu')
    
    # 创建示例参数
    test_params = [torch.randn(100, 100, requires_grad=True)]
    
    # 创建优化器
    optimizer = config.create_optimizer(
        test_params,
        optimizer_type="Adam",
        lr=0.01
    )
    
    # 创建调度器
    scheduler = config.create_scheduler(
        scheduler_type="ReduceLROnPlateau",
        patience=10
    )
    
    # 创建损失函数
    mse_loss = config.create_data_loss(p=2)
    sinkhorn_loss = config.create_sinkhorn_loss(p=2, blur=0.05)
    
    # 打印配置
    config.print_config()
    
    # 学习率建议
    print(f"\n参数数量: {sum(p.numel() for p in test_params)}")
    suggested_lr = config.suggest_learning_rate(10000, "Adam")
    print(f"建议学习率: {suggested_lr}")
    
    print("\n✓ 测试完成")