"""Adaptive regularization (lambda) updates."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch



class InversionRegularizationMixin:
    def update_lambda_by_gradient_balance(
        self,
        loss_data: torch.Tensor,
        loss_model: torch.Tensor,
        current_lambda: float,
        alpha: float = 0.8,
        lambda_min: float = 1e-6,
        bl: float = 2.0,
        min_ratio_for_update: float = 0.01
    ):
        """
        Adaptively update lambda based on gradient magnitudes.
        Lambda is only allowed to decrease (gradually relax regularization).
        Target (soft): ||∇Phi_d|| ≲ bl * lambda * ||∇Phi_m||.

        Args:
            loss_data: Data loss tensor.
            loss_model: Model/regularization loss tensor.
            current_lambda: Current regularization weight.
            alpha: Exponent for exponential decrease when relaxing lambda (default: 0.8).
            lambda_min: Lower bound for lambda.
            bl: Balance factor scaling the gradient-norm target. Larger bl = more tolerant,
            min_ratio_for_update: Skip lambda update if ratio < this (avoid overly
                aggressive decrease when ratio is tiny). Default: 0.01.

        Returns:
            Tuple of (new_lambda, norm_d, norm_m) where norm_m is **raw** ||∇Φ_m|| (not ×λ).
            Callers that log/plot the model term should multiply by λ: λ·||∇Φ_m||.
        """

        # 1) Current gradient norms
        grad_d = torch.autograd.grad(
            loss_data,
            self.model_log_sigma,
            retain_graph=True,
            create_graph=False
        )[0]

        grad_m = torch.autograd.grad(
            loss_model,
            self.model_log_sigma,
            retain_graph=True,
            create_graph=False
        )[0]
        
        norm_d_raw = torch.sqrt(torch.mean(grad_d**2))
        norm_m_raw = torch.sqrt(torch.mean(grad_m**2)) + 1e-12
        
        # Record raw values
        norm_d_item = norm_d_raw.item()
        norm_m_item = norm_m_raw.item()
        
        # 2) Update histories
        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)
        # Keep histories bounded (avoid unbounded memory growth)
        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        # 3) Moving average (smooth gradient norms)
        # Fixed smoothing window (per request): only keep the last 3 samples.
        window_size = 3
        if len(self.grad_norm_d_history) >= window_size:
            norm_d_smooth = np.mean(self.grad_norm_d_history[-window_size:])
            norm_m_smooth = np.mean(self.grad_norm_m_history[-window_size:])
        else:
            # If history is short, fall back to current values
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        # 4) Ratio (using smoothed gradient norms)
        # Use smoothed norms directly to avoid single-step noise.
        # Note: do not use historical ratio statistics; gradients typically decay rapidly.
        ratio = norm_d_smooth / (bl * current_lambda * norm_m_smooth + 1e-12)
        ratio = float(ratio)
        # Store ratio history (monitoring only)
        self.ratio_history.append(ratio)

        # 5) Lambda can only decrease (exponential decrease)
        if ratio < 1.0:
            # If ratio is extremely small, decreasing lambda using the raw ratio can be too aggressive.
            # Instead of skipping updates (which can freeze lambda at a too-large value),
            # clamp ratio from below to allow a gradual monotonic decrease.
            ratio_eff = max(ratio, float(min_ratio_for_update))
            new_lambda = current_lambda * (ratio_eff ** alpha)
        else:
            new_lambda = current_lambda

        # 6) Safety constraints
        new_lambda = float(max(new_lambda, lambda_min))
        new_lambda = min(current_lambda, new_lambda)  # monotonic non-increase

        return new_lambda, norm_d_item, norm_m_item

