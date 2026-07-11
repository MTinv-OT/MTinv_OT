"""Inversion result plotting (extracted from MT2DInverter)."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
from skimage.metrics import structural_similarity as ssim

from ._style import apply_plot_style

def _ylim_snap_to_mesh_bottom(zn, ylim: list, ylim_auto: bool = True) -> list:
    """Snap depth-axis bottom to the outer edge of the deepest visible mesh row.

    When ylim cuts between log-spaced cell centers (e.g. 50 km between centers at
    ~44 km and ~51 km), pcolormesh leaves empty margin above the axis. This snaps
    the bottom limit to the lower edge of the last row inside ylim.
    """
    if not ylim_auto or ylim is None:
        return list(ylim) if ylim is not None else [50, 0]
    ylim_out = list(ylim)
    y_cap = max(ylim_out)

    zn_np = zn.detach().cpu().numpy() if torch.is_tensor(zn) else np.asarray(zn)
    zn_km = zn_np.astype(float) * 0.001
    zc = 0.5 * (zn_km[:-1] + zn_km[1:])
    zc_ground = zc[zc >= 0]
    if zc_ground.size == 0:
        return ylim_out

    zc_visible = zc_ground[zc_ground <= y_cap]
    if zc_visible.size == 0:
        return ylim_out

    idx = int(np.where(zc_ground == zc_visible[-1])[0][0])
    if idx + 1 < zc_ground.size:
        mesh_bottom = float(zc_ground[idx] + 0.5 * (zc_ground[idx + 1] - zc_ground[idx]))
    else:
        mesh_bottom = float(zn_km.max())

    if y_cap > mesh_bottom:
        ylim_out[0] = mesh_bottom
    return ylim_out


def plot_model_comparison(
            inv,
            cmap: str = "jet_r",
            xlim: list = [-20, 20],     # X-axis bounds (ignored when clip_to_stations=True)
            ylim: list = [50, 0],      # Y-axis bounds
            ylim_auto: bool = True,
            clip_to_stations: bool = False,
            profile_extend_km: float = 5.0,
            profile_axis_width_km: float = None,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            synthetic_figwidth_scale: float = 1.0,
            axes_aspect: Optional[Union[str, float]] = None,
            figsize: Optional[Tuple[float, float]] = None,
        ):
        """
        Plot inverted model (log10 domain). If a true model (inv.sig_true) exists,
        also plot a side-by-side comparison (true vs inverted). Air layer is masked,
        and a view mask is applied according to the given bounds.

        clip_to_stations: If True, horizontal extent follows stations. For real data,
            use True to get profile-based display (stations + profile_extend_km each side).
        profile_extend_km: For real data, extend display this many km beyond stations
            on each side. Default 5 km.
        profile_axis_width_km: If set (e.g. 50) with real-data profile mode, force matplotlib
            x-axis to [0, width] km (left = st_min - profile_extend_km in physical coords).
            Default None uses (st_max - st_min) + 2 * profile_extend_km.

        ylim_auto: If True (default), shrink the depth-axis bottom to the mesh extent so
            no empty margin appears below the model (paper-ready layout).
        vmin/vmax: Optional color scale limits for the plotted quantity (log10 resistivity).
            - If provided, both true/inverted panels (if present) use the same limits.
            - If omitted (None), auto-scale using masked min/max ± 0.5 (current behavior).
        synthetic_figwidth_scale: Synthetic (true model) plots only. >1 widens x vs depth on screen
            and expands figure width; 1.0 is 1:1 km scaling. Ignored for real-data profile plots.
        axes_aspect: Override the x–z axes aspect (Matplotlib "aspect").
            - None: keep default behavior (synthetic: 1/synthetic_figwidth_scale; real data: 'auto')
            - 1.0: 1 km (x) equals 1 km (z)
            - 0.5: x is stretched 2× relative to z
        figsize: If provided, overrides the computed figure size in inches, e.g. (10, 6).
        """
        apply_plot_style()
        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = inv.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true_model = hasattr(inv, "sig_true") and isinstance(inv.sig_true, torch.Tensor)
        # Real data: profile extent = stations + extend_km each side; 0-based display, triangles above
        use_profile = not has_true_model
        if use_profile and clip_to_stations:
            extend_km = profile_extend_km
            offset_km = st_min - extend_km
            if profile_axis_width_km is not None:
                w = float(profile_axis_width_km)
                xlim_use = [0.0, max(w, 1e-6)]
                xlim_orig = [offset_km, offset_km + w]
            else:
                xlim_orig = [st_min - extend_km, st_max + extend_km]
                xlim_use = [0.0, (st_max - st_min) + 2 * extend_km]
            xlabel_str = "Distance along profile (km)"
        else:
            xlim_use = [st_min, st_max] if clip_to_stations else xlim
            xlim_orig = xlim_use
            xlabel_str = "Distance (km)"
        # -------- Model values --------
        sigma_inv = inv.get_sigma_full().detach().cpu().numpy()
        sigma_true = None
        if has_true_model:
            sigma_true = inv.sig_true.detach().cpu().numpy()
            try:
                if sigma_true.shape == sigma_inv.shape:
                    eps_ssim = 1e-12
                    # Match compute_recovery_rate(): SSIM on log10(ρ), earth only (mask air).
                    zc_ssim = 0.5 * (inv.zn[:-1] + inv.zn[1:])
                    earth_mask = zc_ssim >= 0
                    if torch.is_tensor(zc_ssim):
                        earth_mask = earth_mask.cpu().numpy()
                    rho_true = 1.0 / (sigma_true[earth_mask] + eps_ssim)
                    rho_inv = 1.0 / (sigma_inv[earth_mask] + eps_ssim)
                    log_rho_true = np.log10(rho_true)
                    log_rho_inv = np.log10(rho_inv)
                    data_range = log_rho_true.max() - log_rho_true.min()
                    if data_range > eps_ssim:
                        score = ssim(log_rho_inv, log_rho_true, data_range=data_range)
                        print(f"Model structural similarity (SSIM): {score:.4f}")
            except Exception:
                pass
        eps = 1e-12
        model_inv = np.log10(1.0 / (sigma_inv + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_true = "True log10 Resistivity"
        title_inv = "Inverted log10 Resistivity" if has_true_model else "Inverted log10 Resistivity"
        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (inv.zn[:-1] + inv.zn[1:])
        yc = 0.001 * 0.5 * (inv.yn[:-1] + inv.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        if use_profile and clip_to_stations:
            offset_km = st_min - profile_extend_km
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
        else:
            YY_plot = YY
            st_x_plot = st_km
        mask_air = ZZ < 0
        ylim = _ylim_snap_to_mesh_bottom(inv.zn, ylim, ylim_auto=ylim_auto)

        # -------- Build view mask (based on xlim_orig/ylim; YY in original coords) --------
        mask_x_min, mask_x_max = min(xlim_orig), max(xlim_orig)
        y_bottom, y_top = max(ylim), min(ylim)
        mask_view = (YY >= mask_x_min) & (YY <= mask_x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
        # -------- Figure size / aspect --------
        x_range = max(float(abs(max(xlim_use) - min(xlim_use))), 1e-6)
        y_range = max(float(abs(y_bottom - y_top)), 1e-6)

        # Station marker: keep surface border at z=0 (ylim top), and draw markers ABOVE the axes box.
        # Use axes-fraction y (>1) so it is always visible and does not change data limits.
        # Keep it close to the border to avoid colliding with the title.
        station_y_axes = 1.03
        panel_h = 5.0

        sx = max(float(synthetic_figwidth_scale), 1e-9) if has_true_model else 1.0
        axes_w_data = panel_h * (x_range / y_range) * sx
        fig_w = axes_w_data + 1.2
        fig_h = panel_h * (2.0 if has_true_model else 1.0)
        if figsize is not None:
            fig_w, fig_h = float(figsize[0]), float(figsize[1])
        aspect_default = (1.0 / sx) if has_true_model else "auto"
        aspect_use = aspect_default if axes_aspect is None else axes_aspect
        if has_true_model:
            model_true = np.log10(1.0 / (sigma_true + eps))
            model_true_masked = np.ma.masked_where(~mask_view, model_true)
        model_inv_masked = np.ma.masked_where(~mask_view, model_inv)
        # Min/max inside the mask for colorbar scaling (auto); allow user override.
        if has_true_model:
            auto_vmin = model_true_masked.min() - 0.5
            auto_vmax = model_true_masked.max() + 0.5
        else:
            auto_vmin = model_inv_masked.min() - 0.5
            auto_vmax = model_inv_masked.max() + 0.5

        vmin_use = float(auto_vmin) if vmin is None else float(vmin)
        vmax_use = float(auto_vmax) if vmax is None else float(vmax)
        int_ticks = np.arange(int(np.ceil(vmin_use)), int(np.floor(vmax_use)) + 1, dtype=int)

        # Font sizes (tuned for readability; per-plot, no global settings)
        title_fs = 30
        label_fs = 26
        tick_fs = 24
        cbar_label_fs = 25
        cbar_tick_fs = 24
        title_pad = 24
        # Mark station locations (km)
        st_x_km = inv.stations.cpu().numpy() / 1000.0
        if has_true_model:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
            im1 = ax1.pcolormesh(YY, ZZ, model_true_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax1.invert_yaxis()
            ax1.set_title(title_true, fontsize=title_fs, pad=title_pad)
            ax1.set_ylabel('Depth (km)', fontsize=label_fs)
            ax1.set_xlim(xlim_use)
            ax1.set_ylim(ylim)
            ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax1.set_aspect(aspect_use, adjustable='box')
            ax1.tick_params(axis='both', labelsize=tick_fs)
            cb1 = plt.colorbar(im1, ax=ax1)
            cb1.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb1.set_ticks(int_ticks)
                cb1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb1.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb1.ax.tick_params(labelsize=cbar_tick_fs)
            ax1.scatter(
                st_x_km,
                np.full(st_x_km.shape, station_y_axes, dtype=float),
                transform=ax1.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
            im2 = ax2.pcolormesh(YY, ZZ, model_inv_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax2.invert_yaxis()
            ax2.set_title(title_inv, fontsize=title_fs, pad=title_pad)
            ax2.set_ylabel('Depth (km)', fontsize=label_fs)
            ax2.set_xlabel('Distance (km)', fontsize=label_fs)
            ax2.set_xlim(xlim_use)
            ax2.set_ylim(ylim)
            ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax2.set_aspect(aspect_use, adjustable='box')
            ax2.tick_params(axis='both', labelsize=tick_fs)
            cb2 = plt.colorbar(im2, ax=ax2)
            cb2.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb2.set_ticks(int_ticks)
                cb2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb2.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb2.ax.tick_params(labelsize=cbar_tick_fs)
            ax2.scatter(
                st_x_km,
                np.full(st_x_km.shape, station_y_axes, dtype=float),
                transform=ax2.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
        else:
            # Inverted only (real data, no true model): default aspect='auto' (fill); set axes_aspect=1.0 for 1:1 km
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), sharex=False)
            im = ax.pcolormesh(YY_plot, ZZ, model_inv_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
            ax.invert_yaxis()
            ax.set_title(title_inv, fontsize=title_fs, pad=title_pad)
            ax.set_ylabel('Depth (km)', fontsize=label_fs)
            ax.set_xlabel(xlabel_str, fontsize=label_fs)
            ax.set_xlim(xlim_use)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax.set_aspect(aspect_use, adjustable='box')
            ax.tick_params(axis='both', labelsize=tick_fs)
            cb = plt.colorbar(im, ax=ax, pad=0.01)
            cb.set_label(label, fontsize=cbar_label_fs)
            if int_ticks.size > 0:
                cb.set_ticks(int_ticks)
                cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
                cb.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
            cb.ax.tick_params(labelsize=cbar_tick_fs)
            ax.scatter(
                st_x_plot,
                np.full(st_x_plot.shape, station_y_axes, dtype=float),
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                c='k',
                s=30,
                marker='v',
                label='Stations',
                zorder=10,
            )
        # More padding to avoid title overlapping with axes/markers.
        if has_true_model:
            fig.subplots_adjust(hspace=0.28)
        plt.tight_layout(pad=2.0)
        plt.show()
        return fig

def plot_initial_model(
            inv,
            cmap: str = "jet_r",
            xlim: list = None,
            ylim: list = None,
            clip_to_stations: bool = False,
            profile_extend_km: float = 5.0,
            profile_axis_width_km: float = None,
            ylim_auto: bool = True,
            synthetic_figwidth_scale: float = 1.0,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            show: bool = True,
        ):
        """Plot the initial model (log10 domain), masking air and applying a view mask.

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim as provided.
        profile_extend_km: For real data (no sig_true), extend display this many km beyond
            stations on each side. Default 5 km.
        profile_axis_width_km: If set with profile mode, force x-axis to [0, width] km.
        ylim_auto: If True, ylim is auto-set to grid depth extent to avoid empty space below.
        synthetic_figwidth_scale: Ignored in real-data profile mode. Otherwise same as plot_model_comparison.
        """
        apply_plot_style()
        if xlim is None:
            xlim = [-20, 20]
        if ylim is None:
            ylim = [50, 0]
        if inv.initial_model_sigma is None:
            print("Initial model is not saved; call initialize_model first")
            return None
        # -------- Resolve xlim (clip to stations if requested) --------
        st_km = inv.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true = hasattr(inv, "sig_true") and inv.sig_true is not None
        use_profile = not has_true and clip_to_stations
        if use_profile:
            offset_km = st_min - profile_extend_km
            if profile_axis_width_km is not None:
                w = float(profile_axis_width_km)
                xlim_use = [0.0, max(w, 1e-6)]
                xlim_orig = [offset_km, offset_km + w]
            else:
                xlim_orig = [st_min - profile_extend_km, st_max + profile_extend_km]
                xlim_use = [0.0, (st_max - st_min) + 2 * profile_extend_km]
            xlabel_str = "Distance along profile (km)"
            st_y_plot = 0.0  # triangle tip at z=0 km (surface)
        else:
            xlim_use = [st_min, st_max] if clip_to_stations else xlim
            xlim_orig = xlim_use
            xlabel_str = "Distance (km)"
            st_y_plot = 0.0
        # -------- Model values --------
        sigma_init = inv.initial_model_sigma.detach().cpu().numpy()
        eps = 1e-12

        model_init = np.log10(1.0 / (sigma_init + eps))
        label = r"log$_{10}$ Resistivity (Ω·m)"
        title_init = "Initial log10 Resistivity"
        # -------- Mask air layer (z < 0) --------
        zc = 0.001 * 0.5 * (inv.zn[:-1] + inv.zn[1:])
        yc = 0.001 * 0.5 * (inv.yn[:-1] + inv.yn[1:])
        YY, ZZ = np.meshgrid(yc.cpu().numpy(), zc.cpu().numpy())
        ylim = _ylim_snap_to_mesh_bottom(inv.zn, ylim, ylim_auto=ylim_auto)
        if use_profile:
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
        else:
            YY_plot = YY
            st_x_plot = st_km
        mask_air = ZZ < 0
        mask_ground = ZZ >= 0
        # -------- Build view mask (based on xlim_orig/ylim; YY in original coords) --------
        mask_x_min, mask_x_max = min(xlim_orig), max(xlim_orig)
        y_bottom, y_top = max(ylim), min(ylim)
        mask_view = (YY >= mask_x_min) & (YY <= mask_x_max) & (ZZ >= y_top) & (ZZ <= y_bottom) & ~mask_air
        # -------- Figure size / aspect --------
        x_range = max(float(abs(max(xlim_use) - min(xlim_use))), 1e-6)
        y_range = max(float(abs(y_bottom - y_top)), 1e-6)
        panel_h = 5.0
        sx = 1.0 if use_profile else max(float(synthetic_figwidth_scale), 1e-9)
        axes_w = panel_h * (x_range / y_range) * sx
        fig_w = axes_w + 1.2  # colorbar
        fig_h = panel_h
        aspect_xz = (1.0 / sx) if not use_profile else 1.0
        model_init_masked = np.ma.masked_where(~mask_view, model_init)
        auto_vmin = model_init_masked.min() - 0.5
        auto_vmax = model_init_masked.max() + 0.5
        vmin_use = float(auto_vmin) if vmin is None else float(vmin)
        vmax_use = float(auto_vmax) if vmax is None else float(vmax)
        int_ticks = np.arange(int(np.ceil(vmin_use)), int(np.floor(vmax_use)) + 1, dtype=int)

        # Match font sizes with plot_model_comparison() for paper-ready figures
        title_fs = 30
        label_fs = 26
        tick_fs = 24
        cbar_label_fs = 25
        cbar_tick_fs = 24
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax.pcolormesh(YY_plot, ZZ, model_init_masked, cmap=cmap, vmin=vmin_use, vmax=vmax_use, shading='auto')
        ax.invert_yaxis()
        ax.set_title(title_init, fontsize=title_fs, pad=24)
        ax.set_ylabel('Depth (km)', fontsize=label_fs)
        ax.set_xlabel(xlabel_str, fontsize=label_fs)
        ax.set_xlim(xlim_use)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.set_aspect(aspect_xz, adjustable='box')
        ax.tick_params(axis='both', labelsize=tick_fs)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(label, fontsize=cbar_label_fs)
        if int_ticks.size > 0:
            cb.set_ticks(int_ticks)
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            cb.ax.set_yticklabels([f"{int(t)}" for t in int_ticks])
        cb.ax.tick_params(labelsize=cbar_tick_fs)
        plt.tight_layout()
        if show:
            plt.show()
            return None
        return fig

def plot_loss_history(inv, target_misfit: float = 1.0, plot_roughness_vs_misfit: bool = False):
        """
        Plot loss terms and parameter evolution during inversion.

        plot_roughness_vs_misfit
            If True, also open a second figure: model roughness (x) vs data misfit (y),
            see ``plot_roughness_misfit_curve``.
        """
        apply_plot_style()
        # Extract series
        epochs = [log['epoch'] for log in inv.loss_history]
        misfit = [log['misfit'] for log in inv.loss_history]
        lambdas = [log['lambda'] for log in inv.loss_history]
        data_loss = [log['data_loss'] for log in inv.loss_history]
        model_loss = [log['model_loss'] for log in inv.loss_history]

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # --- Panel 1: Misfit (RMS) ---
        axes[0].plot(epochs, misfit, 'b-', linewidth=2, label='Current RMS')
        axes[0].axhline(y=target_misfit, color='r', linestyle='--', label='Target')
        axes[0].set_title("Data Misfit Convergence", fontsize=title_fs)
        axes[0].set_xlabel("Epoch", fontsize=label_fs)
        axes[0].set_ylabel("RMS Error", fontsize=label_fs)
        axes[0].set_yscale('log')  # RMS often spans orders of magnitude
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].tick_params(axis='both', labelsize=tick_fs)
        ot_distance = [log.get('ot_distance', float('nan')) for log in inv.loss_history]
        if np.any(np.isfinite(ot_distance)):
            ax0_twin = axes[0].twinx()
            ax0_twin.plot(epochs, ot_distance, 'g--', linewidth=2, label='OT Distance')
            ax0_twin.set_ylabel('OT Distance (scaled)', color='g', fontsize=label_fs)
            ax0_twin.tick_params(axis='y', labelsize=tick_fs, colors='g')
            ax0_twin.set_yscale('log')
            lines, labels = axes[0].get_legend_handles_labels()
            twin_lines, twin_labels = ax0_twin.get_legend_handles_labels()
            axes[0].legend(lines + twin_lines, labels + twin_labels, fontsize=legend_fs)
        else:
            axes[0].legend(fontsize=legend_fs)
        # --- Panel 2: Data Loss vs Model Loss (Roughness) ---
        ax2_twin = axes[1].twinx()
        p1, = axes[1].plot(epochs, data_loss, 'c-', label='Data Loss')
        p2, = ax2_twin.plot(epochs, model_loss, 'm-', label='Model Roughness')
        axes[1].set_title("Loss Components Trade-off", fontsize=title_fs)
        axes[1].set_xlabel("Epoch", fontsize=label_fs)
        axes[1].set_ylabel("Data Loss", color='c', fontsize=label_fs)
        axes[1].set_yscale('log')
        ax2_twin.set_ylabel("Roughness (Model Loss)", color='m', fontsize=label_fs)
        # Merge legends
        axes[1].tick_params(axis='both', labelsize=tick_fs)
        ax2_twin.tick_params(axis='both', labelsize=tick_fs)
        axes[1].legend(handles=[p1, p2], fontsize=legend_fs)
        axes[1].grid(True, alpha=0.3)
        # --- Panel 3: Lambda evolution ---
        axes[2].plot(epochs, lambdas, 'g-', linewidth=2)
        axes[2].set_title("Regularization Parameter (Lambda)", fontsize=title_fs)
        axes[2].set_xlabel("Epoch", fontsize=label_fs)
        axes[2].set_ylabel("Lambda Value", fontsize=label_fs)
        axes[2].set_yscale('log')  # Lambda can span orders of magnitude
        axes[2].grid(True, which="both", ls="-", alpha=0.5)
        axes[2].tick_params(axis='both', labelsize=tick_fs)
        plt.tight_layout()
        plt.show()

        if plot_roughness_vs_misfit:
            inv.plot_roughness_misfit_curve()


def plot_roughness_misfit_curve(
        inv,
        *,
        y_metric: str = "rms",
        log_x: bool = True,
        log_y: bool = True,
        mark_epochs: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        """
        L-curve style plot: model roughness (regularization term Φ_m) vs data misfit.

        Uses ``loss_history`` recorded by ``run_inversion`` (one point per logged epoch).

        Parameters
        ----------
        y_metric:
            ``"rms"`` — normalized RMS χ² from ``compute_rms_chi2`` (same as log ``Misfit(RMS χ²)``).
            ``"data_loss"`` — raw data objective ``loss_data`` used in the total loss.
        log_x, log_y:
            If True, use log scale on roughness / misfit axis (typical when both span orders of magnitude).
        mark_epochs:
            If True, scatter points colored by epoch index (line still connects in epoch order).
        ax:
            Optional matplotlib Axes; if None, a new figure is created.
        """
        if not getattr(inv, "loss_history", None) or len(inv.loss_history) == 0:
            raise RuntimeError("loss_history is empty; run inversion first.")

        apply_plot_style()
        rough = np.array([float(log["model_loss"]) for log in inv.loss_history], dtype=float)
        if y_metric.lower() in ("rms", "chi2", "misfit"):
            yvals = np.array([float(log["misfit"]) for log in inv.loss_history], dtype=float)
            y_label = r"Data misfit (RMS $\chi^2$)"
        elif y_metric.lower() in ("data_loss", "data", "loss_data"):
            yvals = np.array([float(log["data_loss"]) for log in inv.loss_history], dtype=float)
            y_label = "Data loss (optimization objective)"
        else:
            raise ValueError(f"y_metric must be 'rms' or 'data_loss', got {y_metric!r}")

        epochs = np.array([int(log["epoch"]) for log in inv.loss_history], dtype=int)
        created_fig = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5.5))

        valid = np.isfinite(rough) & np.isfinite(yvals) & (rough > 0) & (yvals > 0)
        if not np.any(valid):
            raise RuntimeError("No finite positive (roughness, misfit) pairs to plot.")

        r_p, y_p, e_p = rough[valid], yvals[valid], epochs[valid]
        ax.plot(r_p, y_p, "b-", linewidth=1.5, alpha=0.85, zorder=1, label="Trajectory (epoch order)")
        if mark_epochs:
            sc = ax.scatter(r_p, y_p, c=e_p, cmap="viridis", s=28, zorder=2, edgecolors="k", linewidths=0.3)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Epoch", fontsize=11)

        ax.set_xlabel(r"Model roughness $\Phi_m$ (regularization loss)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title("Roughness vs data misfit (L-curve trace)", fontsize=14)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.tick_params(axis="both", labelsize=10)
        if not mark_epochs:
            ax.legend(fontsize=10, loc="best")
        if created_fig:
            plt.tight_layout()
            plt.show()


def plot_gradient_history(inv):
        """Plot gradient-norm histories: data ||∇Φ_d|| and model λ·||∇Φ_m|| (scaled)."""
        apply_plot_style()
        logs = inv.loss_history
        if len(logs) > 1:
            logs = logs[1:]

        epochs = [log['epoch'] for log in logs]
        g_d = [log['grad_data_norm'] for log in logs]
        g_m = [log['grad_model_norm'] for log in logs]

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, g_d, 'b-', label='||∇Φ_d|| (Data)', linewidth=2)
        plt.plot(epochs, g_m, 'r-', label='λ·||∇Φ_m|| (Model, scaled)', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=label_fs)
        plt.ylabel('Gradient norm', fontsize=label_fs)
        plt.title('Data vs model gradient norms (model: λ·||∇Φ_m||)', fontsize=title_fs)
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)
        plt.grid(True, which='both', ls='-', alpha=0.5)
        plt.legend(fontsize=legend_fs)
        plt.tight_layout()
        plt.show()


def plot_sensitivity(inv, xlim=None, ylim=None, cmap: str = "viridis", clip_to_stations: bool = True,
                         profile_extend_km: float = 5.0):
        """Plot sensitivity matrix heatmap (per-cell sensitivity to observations).

        clip_to_stations: If True (default), horizontal extent is limited to between
            the leftmost and rightmost stations. If False, use xlim when provided.
        profile_extend_km: For real data (no sig_true), extend display this many km beyond
            stations on each side. Default 5 km.
        """
        apply_plot_style()
        sens_2d, YY, ZZ = inv.compute_sensitivity_matrix()
        eps = 1e-16
        sens_log = np.log10(sens_2d + eps)
        mask_air = ZZ < 0
        sens_masked = np.ma.masked_where(mask_air, sens_log)
        st_km = inv.stations.cpu().numpy() / 1000.0
        st_min, st_max = float(st_km.min()), float(st_km.max())
        has_true = hasattr(inv, "sig_true") and inv.sig_true is not None
        use_profile = not has_true and clip_to_stations
        if clip_to_stations and not use_profile:
            x_min, x_max = st_min, st_max
        elif not clip_to_stations:
            x_min, x_max = float(YY.min()), float(YY.max())
        else:
            x_min, x_max = st_min - profile_extend_km, st_max + profile_extend_km
        if xlim is not None:
            x_min, x_max = xlim[0], xlim[1]
        if use_profile:
            offset_km = st_min - profile_extend_km
            YY_plot = YY - offset_km
            st_x_plot = st_km - offset_km
            x_min, x_max = 0.0, (st_max - st_min) + 2 * profile_extend_km
            xlabel_str = "Distance along profile (km)"
            st_y_plot = 0.0  # triangle tip at z=0 km (surface)
        else:
            YY_plot = YY
            st_x_plot = st_km
            xlabel_str = "Distance (km)"
            st_y_plot = 0.0
        y_min, y_max = (float(ZZ.min()), float(ZZ.max())) if ylim is None else (ylim[0], ylim[1])
        x_range = max(abs(x_max - x_min), 1e-6)
        y_range = max(abs(y_max - y_min), 1e-6)
        panel_h = 5.0
        axes_w = panel_h * (x_range / y_range) if use_profile else 2.0 * panel_h * (x_range / y_range)
        fig_w = axes_w + 1.2
        fig, ax = plt.subplots(figsize=(fig_w, panel_h))

        title_fs = 14
        label_fs = 12
        tick_fs = 10
        legend_fs = 11
        cbar_label_fs = 12
        cbar_tick_fs = 10
        im = ax.pcolormesh(YY_plot, ZZ, sens_masked, cmap=cmap, shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel_str, fontsize=label_fs)
        ax.set_ylabel("Depth (km)", fontsize=label_fs)
        ax.set_title("Sensitivity Matrix (∂log pred / ∂log σ)", fontsize=title_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(r"log$_{10}$ Sensitivity", fontsize=cbar_label_fs)
        cb.ax.tick_params(labelsize=cbar_tick_fs)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.set_aspect(1.0 if use_profile else 0.5, adjustable='box')  # Real data: 1:1; synthetic: 2x
        ax.scatter(st_x_plot, np.full_like(st_x_plot, st_y_plot), c="k", s=10, marker="v", label="Stations")
        ax.legend(loc="upper right", fontsize=legend_fs)
        plt.tight_layout()
        plt.show()



def plot_data_fitting(
        inv,
        station_indices=None,
        *,
        stations_per_figure: int = 3,
        plot_noise_cap: Optional[float] = None,
        show: bool = True,
        plot_true_data: bool = False,
        yscale = 5,
    ) -> Union[Figure, List[Figure]]:
    
    with torch.no_grad():
        sigma_full = inv.get_sigma_full()
        pred_dict = inv.forward_operator(sigma_full)
    freqs = inv.freqs.cpu().numpy()
    n_stations = len(inv.stations)
    if station_indices is None:
        station_indices = list(range(n_stations))
    elif isinstance(station_indices, int):
        station_indices = [station_indices]
    else:
        station_indices = list(station_indices)

    stations_per_figure = max(1, int(stations_per_figure))
    figures: List[Figure] = []

    title_fs = 13
    label_fs = 12
    tick_fs = 10
    legend_fs = 8 

    noise_floor = float(getattr(inv, "noise_floor", 0.01) or 0.01)
    sigma_rho_floor = noise_floor / float(np.log(10.0))
    phase_error_deg_floor = max(noise_floor * 28.6, 0.5)
    bar_cap = plot_noise_cap if plot_noise_cap is not None else getattr(inv, "plot_noise_cap", None)
    if bar_cap is not None:
        bar_cap = float(bar_cap)

    true_dict = None
    if plot_true_data and hasattr(inv, "sig_true") and inv.sig_true is not None:
        with torch.no_grad():
            true_dict = inv.forward_operator(inv.sig_true)

    for batch_start in range(0, len(station_indices), stations_per_figure):
        batch = station_indices[batch_start : batch_start + stations_per_figure]
        n_plots = len(batch)
        n_cols = 3 if n_plots == 1 else min(n_plots, max(1, int(stations_per_figure)))
        
        fig, axes = plt.subplots(
            2, n_cols,
            figsize=(5 * n_cols, 7),
            sharex=True,
            gridspec_kw={'height_ratios': [1.0, 0.5], 'hspace': 0.05, 'width_ratios': [1.0] * n_cols} 
        )
        
        if np.ndim(axes) == 1:
            axes = axes.reshape(2, -1)
        
        rho_obs_all = []
        for i, st_idx in enumerate(batch):
            st_id = str(inv.station_ids[int(st_idx)]) if getattr(inv, "station_ids", None) is not None and int(st_idx) < len(inv.station_ids) else f"S{int(st_idx) + 1}"
            if hasattr(inv, "shift_mask") and bool(inv.shift_mask[int(st_idx)].item()):
                st_id += "*"
            
            ax_rho = axes[0, i]
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_rho = f"rho{mode}"
                if key_rho not in inv.obs_data: continue
                rho_obs = inv.obs_data[key_rho][:, st_idx].cpu().numpy()
                rho_pred = pred_dict[key_rho][:, st_idx].cpu().numpy()
                rho_true = true_dict[key_rho][:, st_idx].cpu().numpy() if (plot_true_data and true_dict and key_rho in true_dict) else None
                
                valid = np.isfinite(rho_obs) & (rho_obs > 0)
                if np.any(valid):
                    rho_obs_valid = rho_obs[valid]
                    rho_obs_all.append(rho_obs_valid) # 用于计算全局ymin/ymax
                    freqs_valid = freqs[valid]
                    sigma_log_eff = inv.get_effective_data_noise_std(key_rho)
                    sigma_log_eff = sigma_log_eff[:, st_idx].detach().cpu().numpy()[valid] if sigma_log_eff is not None else np.full_like(rho_obs_valid, sigma_rho_floor)
                    if bar_cap is not None: sigma_log_eff = np.minimum(sigma_log_eff, bar_cap)
                    
                    # 修正：强制 XY 为黑色(k)，YX 为灰色
                    if plot_true_data and rho_true is not None and np.all(np.isfinite(rho_true[valid])):
                        true_color = 'k' if mode == 'xy' else 'gray'
                        ax_rho.plot(freqs_valid, rho_true[valid], "-", color=true_color, lw=2, label=f"True {mode.upper()}")
                    
                    yerr = [rho_obs_valid - rho_obs_valid * 10.0**(-sigma_log_eff), rho_obs_valid * 10.0**(sigma_log_eff) - rho_obs_valid]
                    ax_rho.errorbar(freqs_valid, rho_obs_valid, yerr=yerr, fmt='o', ms=4, alpha=0.6, color=color, ecolor=color, elinewidth=1, capsize=2, label=f"Obs {mode.upper()}")
                
                ax_rho.plot(freqs, np.clip(np.nan_to_num(rho_pred, nan=1e-2), 1e-6, 1e10), f'{color}-', lw=1.5, label=f"Pred {mode.upper()}")

            ax_rho.set_xscale("log"); ax_rho.set_yscale("log"); ax_rho.set_box_aspect(1.0)
            ax_rho.set_title(st_id, fontsize=title_fs) 
            if i == 0: ax_rho.set_ylabel(r"Apparent Resistivity ($\Omega\cdot$m)", fontsize=label_fs)
            ax_rho.grid(True, which="both", alpha=0.3); ax_rho.legend(fontsize=legend_fs, loc="upper right")
            
            # --- Phase Plot ---
            ax_phs = axes[1, i]
            for mode, color in zip(["xy", "yx"], ["r", "b"]):
                key_phs = f"phs{mode}"
                if key_phs not in inv.obs_data: continue
                phs_obs = inv.obs_data[key_phs][:, st_idx].cpu().numpy()
                phs_true = true_dict[key_phs][:, st_idx].cpu().numpy() if (plot_true_data and true_dict and key_phs in true_dict) else None
                
                valid = np.isfinite(phs_obs)
                if np.any(valid):
                    # 修正：同样强制 Phase 的 True XY 为黑色
                    if plot_true_data and phs_true is not None and np.all(np.isfinite(phs_true[valid])):
                        ax_phs.plot(freqs[valid], phs_true[valid], "-", color='k' if mode == 'xy' else 'gray', lw=2)
                    ax_phs.errorbar(freqs[valid], phs_obs[valid], fmt='o', ms=4, alpha=0.6, color=color, ecolor=color, elinewidth=1, capsize=2)
                ax_phs.plot(freqs, np.clip(np.nan_to_num(pred_dict[key_phs][:, st_idx].cpu().numpy(), nan=45.0), 0, 90), f'{color}-', lw=1.5)
            
            ax_phs.set_xscale("log"); ax_phs.set_ylim(0, 90); ax_phs.set_box_aspect(0.5)
            ax_phs.set_xlabel("Frequency (Hz)", fontsize=label_fs)
            if i == 0: ax_phs.set_ylabel("Phase (deg)", fontsize=label_fs)
            ax_phs.grid(True, which="both", alpha=0.3)
            
        if rho_obs_all:
            rho_concat = np.concatenate(rho_obs_all)
            rho_valid = rho_concat[(np.isfinite(rho_concat) & (rho_concat > 0))]
            if rho_valid.size > 0:
                ymin, ymax = float(rho_valid.min()) / yscale, float(rho_valid.max()) * yscale
                for ax in axes[0, :n_plots]: ax.set_ylim(ymin, ymax)
        
        axes[0, 0].invert_xaxis(); axes[0, 0].set_xlim(freqs.max(), freqs.min())

        fig.text(0.5, 0.01, "* denotes stations affected by static shift.", ha="center", fontsize=11)
        figures.append(fig)
        if show: plt.show()

    return figures[0] if len(figures) == 1 else figures








def plot_1d_profiles(
        inv,
        station_indices: List[int] = None,
        depth_limit_km: float = None
    ):
        """
        Plot 1D vertical profiles at selected station locations.

        If a true model (inv.sig_true) is available, plot both true and inverted.
        Otherwise (real-data case), plot inverted only.
        """
        apply_plot_style()
        # 1) Model values
        sigma_inv = inv.get_sigma_full().detach().cpu().numpy()
        has_true_model = hasattr(inv, "sig_true") and isinstance(getattr(inv, "sig_true"), torch.Tensor)
        sigma_true = inv.sig_true.detach().cpu().numpy() if has_true_model else None

        # 2) Depth coordinates
        zn_km = 0.001 * inv.zn.cpu().numpy()
        zc_km = 0.5 * (zn_km[:-1] + zn_km[1:])
        cell_mask = zc_km >= 0
        edge_mask = zn_km >= 0
        zc_ground = zc_km[cell_mask]
        zn_ground = zn_km[edge_mask]

        # 3) Station selection
        if station_indices is None:
            n_stations = len(inv.stations)
            station_indices = [0, n_stations // 2, n_stations - 1]

        # 4) Shared x-axis limits
        all_vals = []
        y_centers = 0.5 * (inv.yn[:-1] + inv.yn[1:]).cpu().numpy()
        for st_idx in station_indices:
            col_idx = np.abs(y_centers - inv.stations[st_idx].item()).argmin()

            all_vals.append(1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12))
            if has_true_model and sigma_true is not None:
                all_vals.append(1.0 / (sigma_true[cell_mask, col_idx] + 1e-12))
        all_vals = np.hstack(all_vals)
        valid = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        if valid.size == 0:
            xmin, xmax = 1.0, 1.0
        else:
            xmin = valid.min() / 2
            xmax = valid.max() * 2

        # 5) Plot
        title_fs = 13
        label_fs = 12
        tick_fs = 10
        legend_fs = 10
        n_plots = len(station_indices)
        fig, axes = plt.subplots(
            1, n_plots, figsize=(4 * n_plots, 6), sharey=True
        )
        if n_plots == 1:
            axes = [axes]
        for i, st_idx in enumerate(station_indices):
            ax = axes[i]
            col_idx = np.abs(
                y_centers - inv.stations[st_idx].item()
            ).argmin()
            # -------- Inverted (cell-centered) --------
            val_inv = 1.0 / (sigma_inv[cell_mask, col_idx] + 1e-12)
            xlabel = r"Resistivity ($\Omega \cdot m$)"
            if has_true_model and sigma_true is not None:
                # -------- True (edge-based, perfect blocks) --------
                val_true_cell = 1.0 / (sigma_true[cell_mask, col_idx] + 1e-12)
                # Expand cell values into an edge-based step function
                val_true_step = np.repeat(val_true_cell, 2)
                z_true_step = np.repeat(zn_ground, 2)[1:-1]
                ax.plot(
                    val_true_step,
                    z_true_step,
                    'k--',
                    linewidth=1.5,
                    label='True'
                )
            ax.step(
                val_inv,
                zc_ground,
                where='mid',
                color='r',
                linewidth=2,
                label='Inverted')
            ax.set_xscale('log')
            ax.set_xlim(xmin, xmax)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel, fontsize=label_fs)
            st_y_km = inv.stations[st_idx].item() / 1000.0
            ax.set_title(f"Profile {st_y_km:.1f} km", fontsize=title_fs)
            ax.grid(True, which='both', alpha=0.3)
            if i == 0:
                ax.set_ylabel("Depth (km)", fontsize=label_fs)
            if depth_limit_km is not None:
                ax.set_ylim([depth_limit_km, 0])
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            ax.tick_params(axis='both', labelsize=tick_fs)
            ax.legend(fontsize=legend_fs)
        plt.tight_layout()
        plt.show()
        return fig

