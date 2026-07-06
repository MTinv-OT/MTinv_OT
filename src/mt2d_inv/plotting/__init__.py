"""MT 2D inversion plotting utilities."""
from __future__ import annotations

from ._style import apply_plot_style
from .comparison import plot_ot_mse_convergence, plot_ot_mse_convergence_from_dirs
from .data_fitting import (
    plot_rho_fitting_comparison,
    plot_rho_fitting_from_npz,
    plot_rho_fitting_from_single_npz,
)
from .inversion import (
    plot_1d_profiles,
    plot_data_fitting,
    plot_gradient_history,
    plot_initial_model,
    plot_loss_history,
    plot_model_comparison,
    plot_roughness_misfit_curve,
    plot_sensitivity,
)
from .pseudosection import (
    plot_apparent_resistivity_pseudosection,
    plot_inversion_pseudosection_comparison,
    plot_ot_mse_pseudosection_from_npz,
    plot_pseudosection_from_npz,
)

__all__ = [
    "apply_plot_style",
    "plot_model_comparison",
    "plot_initial_model",
    "plot_loss_history",
    "plot_roughness_misfit_curve",
    "plot_gradient_history",
    "plot_sensitivity",
    "plot_data_fitting",
    "plot_1d_profiles",
    "plot_ot_mse_convergence",
    "plot_ot_mse_convergence_from_dirs",
    "plot_rho_fitting_comparison",
    "plot_rho_fitting_from_npz",
    "plot_rho_fitting_from_single_npz",
    "plot_apparent_resistivity_pseudosection",
    "plot_inversion_pseudosection_comparison",
    "plot_pseudosection_from_npz",
    "plot_ot_mse_pseudosection_from_npz",
]
