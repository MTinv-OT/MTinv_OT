"""Adaptive regularization-weight updates.

The expensive part of the gradient balance is the data-term gradient because it
triggers the sparse adjoint solve.  This module can return the already computed
data/model gradients so the inversion loop can reuse them for the optimizer
update instead of calling ``total_loss.backward()`` a second time.
"""
from __future__ import annotations

from typing import Optional

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
        min_ratio_for_update: float = 0.01,
        *,
        return_gradients: bool = False,
        retain_graph: bool = True,
        grad_data: Optional[torch.Tensor] = None,
        grad_model: Optional[torch.Tensor] = None,
    ):
        """Adaptively update ``lambda`` from data/model gradient magnitudes.

        Lambda is only allowed to decrease.  The soft balance target is

            ||grad Phi_d|| <= bl * lambda * ||grad Phi_m||.

        Parameters
        ----------
        loss_data, loss_model
            Data and model objective tensors.
        current_lambda
            Current regularization weight.
        alpha
            Exponent controlling the gradual decrease of lambda.
        lambda_min
            Lower bound for lambda.
        bl
            Balance factor in the target above.
        min_ratio_for_update
            Lower clamp used when the ratio is extremely small.
        return_gradients
            When ``True``, also return the already-computed ``grad Phi_d`` and
            ``grad Phi_m``.  The caller can then set
            ``model.grad = grad_data + lambda * grad_model`` and avoid a second
            data-term adjoint solve.
        retain_graph
            Passed to ``torch.autograd.grad``.  Set to ``False`` when the
            returned gradients will be reused for the optimizer step and no
            later backward pass is required.
        grad_data, grad_model
            Optional precomputed gradients.  Supplying these makes this method
            purely a norm/history/lambda update.

        Returns
        -------
        tuple
            ``(new_lambda, norm_d, norm_m)`` by default.  With
            ``return_gradients=True`` the tuple additionally contains detached
            ``(grad_data, grad_model)``.
        """
        model_param = self.model_log_sigma

        if grad_data is None:
            grad_data = torch.autograd.grad(
                loss_data,
                model_param,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]
        if grad_model is None:
            grad_model = torch.autograd.grad(
                loss_model,
                model_param,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]

        # RMS gradient norms make the balance independent of the number of
        # active model parameters.
        norm_d_raw = torch.sqrt(torch.mean(grad_data.detach() ** 2))
        norm_m_raw = torch.sqrt(torch.mean(grad_model.detach() ** 2)) + 1e-12

        norm_d_item = float(norm_d_raw.item())
        norm_m_item = float(norm_m_raw.item())

        self.grad_norm_d_history.append(norm_d_item)
        self.grad_norm_m_history.append(norm_m_item)

        max_history = 100
        if len(self.grad_norm_d_history) > max_history:
            self.grad_norm_d_history = self.grad_norm_d_history[-max_history:]
            self.grad_norm_m_history = self.grad_norm_m_history[-max_history:]
            if len(self.ratio_history) > max_history:
                self.ratio_history = self.ratio_history[-max_history:]

        window_size = 3
        if len(self.grad_norm_d_history) >= window_size:
            norm_d_smooth = float(np.mean(self.grad_norm_d_history[-window_size:]))
            norm_m_smooth = float(np.mean(self.grad_norm_m_history[-window_size:]))
        else:
            norm_d_smooth = norm_d_item
            norm_m_smooth = norm_m_item

        ratio = norm_d_smooth / (
            float(bl) * float(current_lambda) * norm_m_smooth + 1e-12
        )
        ratio = float(ratio)
        self.ratio_history.append(ratio)

        if ratio < 1.0:
            ratio_eff = max(ratio, float(min_ratio_for_update))
            new_lambda = float(current_lambda) * (ratio_eff ** float(alpha))
        else:
            new_lambda = float(current_lambda)

        new_lambda = max(new_lambda, float(lambda_min))
        new_lambda = min(float(current_lambda), float(new_lambda))

        if return_gradients:
            return (
                new_lambda,
                norm_d_item,
                norm_m_item,
                grad_data.detach(),
                grad_model.detach(),
            )
        return new_lambda, norm_d_item, norm_m_item
