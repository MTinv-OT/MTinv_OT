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

    def compute_depth_weights_from_zn(
        self,
        zn: torch.Tensor,
        nza: int,
        beta: float = 0.3,
        clamp_max: float = 500.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute depth weights for roughness regularization.

        Heuristic form (earth layers only, excluding air):
            w(z) = (z / z0)^beta
        Then clamp and (optionally) normalize to max=1.

        Returns a matrix shaped like the earth model domain: (nz_earth, nx).
        """
        zn_tensor = (
            zn.to(self.device, dtype=torch.float64)
            if isinstance(zn, torch.Tensor)
            else torch.as_tensor(zn, device=self.device, dtype=torch.float64)
        )
        if zn_tensor.ndim != 1 or zn_tensor.numel() < 2:
            raise ValueError("zn must be a 1D tensor/array with length >= 2")

        nza_i = int(max(0, nza))
        z_centers_full = (zn_tensor[:-1] + zn_tensor[1:]) * 0.5
        z_centers_earth = z_centers_full[nza_i:]

        if int(z_centers_earth.numel()) != int(self.nz):
            raise ValueError(
                f"Depth weights size mismatch: expected nz_earth={self.nz}, got {int(z_centers_earth.numel())}. "
                "Check zn and nza consistency."
            )

        z_pos = torch.clamp(z_centers_earth, min=1.0)
        z0 = z_pos.min().clamp(min=1.0)
        w_z = (z_pos / z0).pow(float(beta)).unsqueeze(1).expand(-1, self.nx)
        w_z = torch.clamp(w_z, max=float(clamp_max))
        if normalize:
            w_z = w_z / w_z.max().clamp(min=1e-12)
        return w_z.to(device=self.device, dtype=torch.float64)
    
    
    def calculate_weighted_roughness(
        self,
        model_log_sigma,
        weights=None,
        norm_type: str = "L2",
        alpha_x: float = 1.0,
        alpha_z: float = 1.0,
    ):
        """
        Weighted roughness as integral of gradient squared: ∫|∇m|² dA.
                Uses physical gradient (per meter) and integration area (cell area) so that
                the discretization matches ∫|∇m|^p dA:
                    - x-gradient term uses area ~ dz * dx
                    - z-gradient term uses area ~ dx * dz
        
        Args:
            model_log_sigma: Current model log conductivity [nz, nx]
            weights: Weight matrix (optional)
            norm_type: "L1" or "L2"
            alpha_x: Weight for horizontal (x) roughness term
            alpha_z: Weight for vertical (z) roughness term
            
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

        # 4. Integration area (cell area associated with each gradient sample)
        # For x-gradient (between columns): area ≈ dz * sp_x
        # For z-gradient (between rows):    area ≈ dx * sp_z
        if dz.ndim == 0:
            dz_col = dz
        else:
            dz_col = dz.reshape(-1, 1)
        if dx.ndim == 0:
            dx_row = dx
        else:
            dx_row = dx.reshape(1, -1)

        if sp_x.ndim == 0:
            area_for_x = dz_col * sp_x
        else:
            area_for_x = dz_col * sp_x.reshape(1, -1)

        if sp_z.ndim == 0:
            area_for_z = dx_row * sp_z
        else:
            area_for_z = dx_row * sp_z.reshape(-1, 1)

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

        ax = float(alpha_x)
        az = float(alpha_z)
        total = ax * loss_x + az * loss_z
        epsilon = 1e-12
        return total + epsilon * total
    
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
                                     reference_weight: float = 0.0,
                                     alpha_x: float = 1.0,
                                     alpha_z: float = 1.0) -> torch.Tensor:
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
            alpha_x: Weight for horizontal (x) roughness term
            alpha_z: Weight for vertical (z) roughness term
            
        Returns:
            Combined constraint value
        """
        # 1. Roughness constraint
        roughness_loss = self.calculate_weighted_roughness(
            model_log_sigma,
            roughness_weights,
            roughness_norm,
            alpha_x=alpha_x,
            alpha_z=alpha_z,
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
    
    