"""Shared matplotlib style for mt2d_inv plotting."""
from __future__ import annotations

import matplotlib.pyplot as plt


def apply_plot_style() -> None:
    """Apply a paper-friendly Matplotlib style (font-related only)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
