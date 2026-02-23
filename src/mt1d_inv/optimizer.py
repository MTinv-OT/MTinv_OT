"""
Optimizer configuration and loss function module (improved).

Improvements:
1. Learning rate scheduler
2. Better parameter initialization suggestions
3. Gradient monitoring
4. Optional geomloss dependency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict

# Optional geomloss import
try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("⚠️  geomloss not installed, Sinkhorn loss unavailable")


class OptimizerConfig:
    """
    Optimizer configuration class (improved).
    
    Additional features:
    - Learning rate scheduler
    - Gradient monitoring
    - Adaptive optimizer selection
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize"""
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.optimizer_type = None
        self.learning_rate = None
        self.sinkhorn_loss = None
        self.data_loss_fn = None
        
        # Gradient monitoring
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
        Create optimizer.
        
        Args:
            params: Parameters to optimize
            optimizer_type: 'Adam', 'AdamW', 'SGD', 'RMSprop', 'LBFGS'
            lr: Learning rate
            weight_decay: L2 regularization coefficient
            betas: Adam momentum parameters
            eps: Numerical stability
            momentum: SGD/RMSprop momentum
        
        Returns:
            optimizer: Optimizer instance
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
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        self.optimizer = optimizer
        self.optimizer_type = optimizer_type
        self.learning_rate = lr
        
        print(f"✓ Created optimizer: {optimizer_type}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        
        return optimizer
    
    def create_scheduler(self,
                        scheduler_type: str = "ReduceLROnPlateau",
                        factor: float = 0.5,
                        patience: int = 20,
                        min_lr: float = 1e-6,
                        **kwargs):
        """
        Create learning rate scheduler.
        
        Args:
            scheduler_type: 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'
            factor: LR decay factor
            patience: Patience epochs
            min_lr: Minimum learning rate
        
        Returns:
            scheduler: Scheduler instance
        """
        if self.optimizer is None:
            raise ValueError("Create optimizer first")
        
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
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        self.scheduler = scheduler
        print(f"✓ Created LR scheduler: {scheduler_type}")
        
        return scheduler
    
    def create_sinkhorn_loss(self,
                            p: int = 2,
                            blur: float = 0.08,
                            scaling: float = 0.9,
                            reach: Optional[float] = None,
                            debias: bool = True,
                            backend: str = "tensorized") -> Optional[SamplesLoss]:
        """
        Create Sinkhorn loss (if geomloss available).
        
        Args:
            p: p-norm for distance
            blur: Blur parameter (entropy regularization)
            scaling: Sinkhorn iteration scaling
            reach: Unbalanced OT relaxation distance. None=balanced OT (mass conservation, default); >0 (e.g. 1.0)=unbalanced OT (no mass conservation)
            debias: Whether to debias
            backend: 'tensorized', 'online', 'multiscale'
        
        Returns:
            sinkhorn_loss: Sinkhorn loss instance or None
        """
        if not GEOMLOSS_AVAILABLE:
            print("⚠️  geomloss not installed, Sinkhorn loss unavailable")
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
        
        print(f"✓ Created Sinkhorn loss:")
        print(f"  - p-norm: {p}")
        print(f"  - blur: {blur}")
        print(f"  - scaling: {scaling}")
        print(f"  - reach: {reach} ({'unbalanced/no mass cons.' if reach else 'balanced/mass cons.'})")
        
        return sinkhorn_loss
    
    def create_data_loss(self, p: int = 2) -> nn.Module:
        """
        Create standard data loss.
        
        Args:
            p: 1=L1, 2=MSE
        
        Returns:
            loss_fn: Loss function
        """
        if p == 1:
            loss_fn = nn.L1Loss()
            print("✓ Created L1 loss")
        else:
            loss_fn = nn.MSELoss()
            print("✓ Created MSE loss")
        
        self.data_loss_fn = loss_fn
        return loss_fn
    
    def clip_gradients(self, 
                      parameters: List[torch.Tensor],
                      max_norm: float = 10.0,
                      record: bool = False) -> float:
        """
        Gradient clipping.
        
        Args:
            parameters: Parameter list
            max_norm: Max gradient norm
            record: Whether to record gradient history
        
        Returns:
            grad_norm: Gradient norm before clipping
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
        Parameter range constraint.
        
        Args:
            parameters: Parameter tensor
            min_val: Minimum value
            max_val: Maximum value
            use_log_space: Whether in log space
        """
        with torch.no_grad():
            if min_val is not None:
                parameters.clamp_(min=min_val)
            if max_val is not None:
                parameters.clamp_(max=max_val)
    
    def step_scheduler(self, metric: Optional[float] = None):
        """
        Learning rate scheduler step.
        
        Args:
            metric: Monitor metric (for ReduceLROnPlateau)
        """
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        if self.optimizer is None:
            return None
        return self.optimizer.param_groups[0]['lr']
    
    def get_grad_stats(self) -> Dict[str, float]:
        """Get gradient statistics"""
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
        Suggest learning rate based on parameter count.
        
        Args:
            parameter_count: Total parameter count
            optimizer_type: Optimizer type
        
        Returns:
            suggested_lr: Suggested learning rate
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
        """Print current configuration"""
        print("\n" + "="*50)
        print("Optimizer Configuration")
        print("="*50)
        print(f"Optimizer type: {self.optimizer_type}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Current LR: {self.get_current_lr()}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"Data loss: {type(self.data_loss_fn).__name__ if self.data_loss_fn else 'None'}")
        print(f"Sinkhorn loss: {'Available' if self.sinkhorn_loss else 'Unavailable'}")
        print("="*50)


# Test code
if __name__ == "__main__":
    print("Optimizer config test\n")
    
    # Create config
    config = OptimizerConfig(device='cpu')
    
    # Create sample params
    test_params = [torch.randn(100, 100, requires_grad=True)]
    
    # Create optimizer
    optimizer = config.create_optimizer(
        test_params,
        optimizer_type="Adam",
        lr=0.01
    )
    
    # Create scheduler
    scheduler = config.create_scheduler(
        scheduler_type="ReduceLROnPlateau",
        patience=10
    )
    
    # Create loss functions
    mse_loss = config.create_data_loss(p=2)
    sinkhorn_loss = config.create_sinkhorn_loss(p=2, blur=0.05)
    
    # Print config
    config.print_config()
    
    # Learning rate suggestion
    print(f"\n参数数量: {sum(p.numel() for p in test_params)}")
    suggested_lr = config.suggest_learning_rate(10000, "Adam")
    print(f"建议学习率: {suggested_lr}")
    
    print("\n✓ Test complete")