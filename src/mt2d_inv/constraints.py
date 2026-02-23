"""
Constraint and regularization module.
Contains constraint calculations for 1D and 2D inversion.
"""

import torch
from typing import Optional


class ConstraintCalculator:
    """
    Constraint calculator.
    Supports various constraint types for 1D and 2D models.
    """
    
    def __init__(self, nx: int, nz: int, dx, dz, device: str = "cpu"):
        """
        Initialize constraint calculator; store grid info.
        dx, dz: grid spacing (m). Can be float (uniform) or 1D tensor/array (non-uniform).
        """
        self.nx = nx
        self.nz = nz
        self.dx = torch.as_tensor(dx, device=device) if not isinstance(dx, torch.Tensor) else dx.to(device)
        self.dz = torch.as_tensor(dz, device=device) if not isinstance(dz, torch.Tensor) else dz.to(device)
        self.device = device
    
    
    def calculate_weighted_roughness(self, model_log_sigma, weights=None, norm_type="L2"):
        """
        Weighted roughness as integral of gradient squared: ∫|∇m|² dA.
        Uses physical gradient (per meter) and integration area (dz for vertical
        interfaces, dx for horizontal interfaces). Distinguishes 0.1/10m vs 0.1/10km
        while balancing shallow/deep via area weighting.
        
        Args:
            model_log_sigma: Current model log conductivity [nz, nx]
            weights: Weight matrix (optional)
            norm_type: "L1" or "L2"
            
        Returns:
            Roughness value
        """
        # 1. Compute diff
        diff_x = (model_log_sigma[:, 1:] - model_log_sigma[:, :-1])
        diff_z = (model_log_sigma[1:, :] - model_log_sigma[:-1, :])

        # 2. Interface spacing: (cell_i + cell_i+1) / 2
        dx, dz = self.dx, self.dz
        if dx.ndim == 0:
            sp_x = dx
            sp_z = dz
        else:
            sp_x = (dx[:-1] + dx[1:]) * 0.5
            sp_z = (dz[:-1] + dz[1:]) * 0.5

        # 3. Physical gradient (per meter)
        grad_x = diff_x / sp_x.reshape(1, -1) if sp_x.ndim > 0 else diff_x / sp_x
        grad_z = diff_z / sp_z.reshape(-1, 1) if sp_z.ndim > 0 else diff_z / sp_z

        # 4. Integration area: vertical interface -> dz, horizontal -> dx
        area_for_x = dz.reshape(-1, 1) if dz.ndim > 0 else dz
        area_for_z = dx.reshape(1, -1) if dx.ndim > 0 else dx

        # 5. Optional spatial weights
        if weights is not None:
            w_x = (weights[:, 1:] + weights[:, :-1]) * 0.5
            w_z = (weights[1:, :] + weights[:-1, :]) * 0.5
            grad_x = grad_x * w_x
            grad_z = grad_z * w_z

        # 6. Integral of |grad|^p over area
        if norm_type == "L1":
            loss_x = torch.sum(torch.abs(grad_x) * area_for_x)
            loss_z = torch.sum(torch.abs(grad_z) * area_for_z)
        elif norm_type == "L2":
            loss_x = torch.sum(grad_x ** 2 * area_for_x)
            loss_z = torch.sum(grad_z ** 2 * area_for_z)
        else:
            raise ValueError("Unsupported norm_type. Please choose 'L1' or 'L2'.")

        epsilon = 1e-12
        return loss_x + loss_z + epsilon * (loss_x + loss_z)
    
    def calculate_reference_model_constraint(self, 
                                            model_log_sigma: torch.Tensor,
                                            reference_model_log_sigma: torch.Tensor,
                                            weights: Optional[torch.Tensor] = None,
                                            norm_type: str = "L2") -> torch.Tensor:
        """
        Reference model constraint (keep model close to reference).
        
        Args:
            model_log_sigma: Current log conductivity [nz, nx]
            reference_model_log_sigma: Reference log conductivity [nz, nx]
            weights: Spatial weights [nz, nx] (optional)
            norm_type: "L1" or "L2"
            
        Returns:
            Reference model constraint value
        """
        # 1. Deviation from reference
        diff = model_log_sigma - reference_model_log_sigma
        
        # 2. Apply spatial weights if provided
        if weights is not None:
            diff = diff * weights
        
        # 3. 根据 norm_type 选择 L1 或 L2 范数
        if norm_type == "L1":
            # L1 范数：使用绝对值
            loss = torch.sum(torch.abs(diff))
        elif norm_type == "L2":
            # L2: sum of squares
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
        Combined constraint: roughness + reference model.
        
        Args:
            model_log_sigma: Current log conductivity [nz, nx]
            reference_model_log_sigma: Reference log conductivity [nz, nx] (optional)
            roughness_weights: Roughness weights (optional)
            reference_weights: Reference constraint weights (optional)
            roughness_norm: "L1" or "L2"
            reference_norm: "L1" or "L2"
            reference_weight: Reference weight (0.0 = disabled)
            
        Returns:
            Combined constraint value
        """
        # 1. Roughness constraint
        roughness_loss = self.calculate_weighted_roughness(
            model_log_sigma, roughness_weights, roughness_norm
        )
        
        # 2. Reference model constraint if provided and weight > 0
        if reference_model_log_sigma is not None and reference_weight > 0.0:
            reference_loss = self.calculate_reference_model_constraint(
                model_log_sigma, reference_model_log_sigma, 
                reference_weights, reference_norm
            )
            return roughness_loss + reference_weight * reference_loss
        else:
            return roughness_loss
    
    