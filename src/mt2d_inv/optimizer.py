"""
Optimizer configuration and loss functions for MT2D inversion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Callable

from geomloss import SamplesLoss


class OptimizerConfig:
    """Optimizer configuration for MT2D inversion."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.optimizer = None
        self.optimizer_type = None
        self.learning_rate = None
        self.sinkhorn_loss = None

    def create_optimizer(self,
                        params: List[torch.Tensor],
                        optimizer_type: str = "AdamW",
                        lr: float = 0.01,
                        weight_decay: float = 0.0,
                        betas: Tuple[float, float] = (0.9, 0.999),
                        eps: float = 1e-8,
                        momentum: float = 0.9,) -> optim.Optimizer:
        """
        Create optimizer.

        Args:
            params: Parameters to optimize
            optimizer_type: 'Adam', 'AdamW', 'SGD', 'RMSprop', 'LBFGS'
            lr: Learning rate
            weight_decay: L2 regularization
            betas: Adam betas
            eps: Numerical stability
            momentum: SGD/RMSprop momentum

        Returns:
            Optimizer instance
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

        print(f"✓ Optimizer: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")
        return optimizer

    def create_sinkhorn_loss(self,
                             p: int = 2,
                             blur: float = 0.08,
                             scaling: float = 0.9,
                             reach: Optional[float] = None,
                             debias: bool = True,
                             backend: str = "tensorized") -> SamplesLoss:
        """
        Create Sinkhorn OT loss.

        Args:
            p: p-norm for cost
            blur: Entropy regularization
            scaling: Sinkhorn scaling
            reach: None=balanced OT; >0=unbalanced OT
            debias: Debiased Sinkhorn
            backend: 'tensorized', 'online', 'multiscale'

        Returns:
            SamplesLoss instance
        """
        diameter = max(2.0, blur * 2.0)
        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",
            p=p,
            blur=blur,
            scaling=scaling,
            reach=reach,
            diameter=diameter,
            debias=debias,
            backend=backend
        )
        self.sinkhorn_loss = sinkhorn_loss
        mode = "unbalanced" if reach else "balanced"
        print(f"✓ Sinkhorn: p={p}, blur={blur}, scaling={scaling}, reach={reach} ({mode})")
        return sinkhorn_loss

    def create_sinkhorn_loss_with_cost(self,
                                       cost: Callable,
                                       blur: float = 0.08,
                                       scaling: float = 0.9,
                                       reach: Optional[float] = None,
                                       debias: bool = True,
                                       backend: str = "tensorized") -> SamplesLoss:
        """
        Create Sinkhorn loss with custom cost.
        cost(x, y) must return (B, N, M) with x(B,N,D), y(B,M,D).
        """
        diameter = max(2.0, blur * 2.0)
        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",
            cost=cost,
            blur=blur,
            scaling=scaling,
            reach=reach,
            diameter=diameter,
            debias=debias,
            backend=backend
        )
        self.sinkhorn_loss = sinkhorn_loss
        print(f"✓ Sinkhorn (custom cost): blur={blur}, scaling={scaling}")
        return sinkhorn_loss
