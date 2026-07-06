"""PrepareData diagnostic plots (extracted)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_plot_style

class PrepareDataPlotMixin:
    def plot_pseudosection(
        self,
        components: Optional[List[str]] = None,
        cmap_rho: str = "jet_r",
        cmap_phs: str = "jet",
        profile_extend_km: float = 5.0,
        y_axis: str = "period",
    ):
        """Plot apparent resistivity and phase pseudosections (observation data).

        Call after run_all_simple() or export_data_dict_for_2d_inversion().
        Uses inv_freqs_torch, inv_stations_torch, inv_data_dict if available;
        otherwise calls export_data_dict_for_2d_inversion() first.

        X-axis: distance along profile (km), 0-based.
        Y-axis: period (s) or frequency (Hz), log scale, increasing period downward.

        Parameters
        ----------
        components : list of str, optional
            Which components to plot: "rhoxy", "phsxy", "rhoyx", "phsyx".
            Default: ["rhoxy", "phsxy", "rhoyx", "phsyx"] (2x2: TE + TM).
        cmap_rho : str
            Colormap for resistivity. Default "jet_r".
        cmap_phs : str
            Colormap for phase. Default "jet".
        profile_extend_km : float
            Horizontal extent beyond stations (km). Default 5.
        y_axis : str
            "period" (default) or "freq" for Y-axis.
        """
        if not hasattr(self, "inv_data_dict") or self.inv_data_dict is None:
            self.export_data_dict_for_2d_inversion()
        obs_data = self.inv_data_dict["obs_data"]
        freqs_t = self.inv_freqs_torch
        stations_t = self.inv_stations_torch

        def _to_np(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            return np.asarray(x)

        st_km = _to_np(stations_t) / 1000.0
        freqs_hz = _to_np(freqs_t)
        st_min, st_max = float(st_km.min()), float(st_km.max())
        offset_km = st_min - profile_extend_km
        st_x = st_km - offset_km

        if components is None:
            components = ["rhoxy", "phsxy", "rhoyx", "phsyx"]
        available = [k for k in components if k in obs_data]
        if not available:
            print(f"No requested components. Available: {list(obs_data.keys())}")
            return

        if y_axis == "period":
            y_vals = 1.0 / freqs_hz
            y_label = "Period (s)"
        else:
            y_vals = freqs_hz
            y_label = "Frequency (Hz)"

        n_f, n_s = _to_np(obs_data[available[0]]).shape
        if n_s > 1:
            x_mid = 0.5 * (st_x[:-1] + st_x[1:])
            x_edges = np.concatenate([
                [2 * st_x[0] - x_mid[0]], x_mid, [2 * st_x[-1] - x_mid[-1]],
            ])
        else:
            x_edges = np.array([st_x[0] - 0.5, st_x[0] + 0.5])
        if n_f > 1:
            log_y = np.log10(y_vals)
            y_mid = 0.5 * (log_y[:-1] + log_y[1:])
            y_edges = 10 ** np.concatenate([
                [2 * log_y[0] - y_mid[0]], y_mid, [2 * log_y[-1] - y_mid[-1]],
            ])
        else:
            y_edges = np.array([y_vals[0] * 0.5, y_vals[0] * 1.5])
        X, Y = np.meshgrid(x_edges, y_edges)

        n_plot = len(available)
        n_col = min(2, n_plot)
        n_row = (n_plot + n_col - 1) // n_col
        fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row), sharex=True, sharey=True)
        if n_plot == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, key in enumerate(available):
            ax = axes[idx]
            data = _to_np(obs_data[key])
            masked_data = np.ma.masked_invalid(data)
            if "rho" in key.lower():
                data_plot = np.ma.log10(np.ma.clip(masked_data, 1e-2, 1e6))
                im = ax.pcolormesh(X, Y, data_plot, cmap=cmap_rho, shading="flat")
                plt.colorbar(im, ax=ax, label=r"log$_{10}$ $\rho_a$ (Ω·m)")
            else:
                data_plot = np.ma.clip(masked_data, -90, 90)
                im = ax.pcolormesh(X, Y, data_plot, cmap=cmap_phs, shading="flat")
                plt.colorbar(im, ax=ax, label="Phase (°)")
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_xlabel("Distance along profile (km)")
            ax.set_ylabel(y_label)
            ax.set_title(key)
            ax.set_xlim(0, (st_max - st_min) + 2 * profile_extend_km)

        for j in range(n_plot, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()

    # -------------------------- export: mt_object -> txt --------------------------

    def plot_strike_period(self, mt: "PrepareData.CustomMT", unwrap_90: bool = False) -> None:
        """Plot strike vs period for a single station.

        unwrap_90: If True, unwrap by 90° for continuity. Can hide real variation.
        """
        s = self.station_strike(mt)
        period = 1 / mt.frequency
        if unwrap_90:
            s = self._unwrap_strike_90(period, s)
        valid_mask = np.isfinite(s)

        plt.figure(figsize=(10, 6))
        plt.semilogx(period[valid_mask], s[valid_mask], "b.", markersize=8, alpha=0.7)
        plt.xlabel("Period (s)", fontsize=12)
        plt.ylabel("Strike (deg)", fontsize=12)
        plt.title("Strike vs Period", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rose(strikes: np.ndarray) -> None:
        theta = np.deg2rad(np.asarray(strikes, dtype=float))
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.hist(theta, bins=36)
        ax.set_title("Strike Rose Diagram")
        plt.show()

    def plot_skew_period(self, mt: "PrepareData.CustomMT") -> None:
        skews = [self.phase_tensor_skew(mt.Z[i]) for i in range(len(mt.frequency))]
        period = 1 / mt.frequency
        plt.figure()
        plt.semilogx(period, skews, ".")
        plt.xlabel("Period (s)")
        plt.ylabel("Phase Tensor Skew (deg)")
        plt.grid()
        plt.show()

    def plot_all_strikes_subplots(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        n_cols: int = 4,
        unwrap_90: bool = False,
    ) -> None:
        """Plot strike vs period for each station.

        unwrap_90: If True, unwrap strike by 90° to reduce discontinuity from ambiguity.
            Use only when jumps are known to be from 90° ambiguity; it can hide real
            depth-dependent strike variation (e.g., 3D structure).
        """
        n_stations = len(mt_objects)
        n_rows = (n_stations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        fig.suptitle("Strike vs Period - All Stations", fontsize=16, y=1.02)

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, mt in enumerate(mt_objects):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_mask = np.isfinite(s)

            ax.semilogx(period[valid_mask], s[valid_mask], "b.", markersize=5, alpha=0.6)
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Strike (deg)")
            ax.set_title(f"Station {idx + 1}")
            ax.grid(True, alpha=0.3)
            if unwrap_90 and np.any(valid_mask):
                ymin, ymax = float(np.nanmin(s[valid_mask])), float(np.nanmax(s[valid_mask]))
                margin = max(10, (ymax - ymin) * 0.1)
                ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.set_ylim(-90, 90)

        for idx in range(n_stations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_comprehensive_strike_analysis(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        regional_strike: Optional[float] = None,
        all_strikes: Optional[np.ndarray] = None,
        unwrap_90: bool = False,
    ) -> None:
        """Comprehensive strike analysis (Strike vs Period, Rose, histogram, boxplot).

        unwrap_90: If True, unwrap strike by 90° for continuity. Can hide real variation.
        """
        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, (1, 2))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(mt_objects)))

        all_strikes_from_curves: list[float] = []
        for i, mt in enumerate(mt_objects):
            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_mask = np.isfinite(s)

            ax1.semilogx(
                period[valid_mask],
                s[valid_mask],
                ".",
                color=colors[i],
                markersize=4,
                alpha=0.5,
                label=f"St{i + 1}" if i < 10 else None,
            )

            if all_strikes is None:
                all_strikes_from_curves.extend(s[valid_mask].tolist())

        ax1.set_xlabel("Period (s)", fontsize=12)
        ax1.set_ylabel("Strike (deg)", fontsize=12)
        ax1.set_title("All Stations: Strike vs Period", fontsize=14)
        ax1.grid(True, alpha=0.3)
        if unwrap_90 and all_strikes_from_curves:
            arr = np.asarray(all_strikes_from_curves, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                ymin, ymax = float(np.nanmin(finite)), float(np.nanmax(finite))
                margin = max(10, (ymax - ymin) * 0.1)
                ax1.set_ylim(ymin - margin, ymax + margin)
            else:
                ax1.set_ylim(-90, 90)
        else:
            ax1.set_ylim(-90, 90)
        if len(mt_objects) <= 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if all_strikes is None:
            all_strikes_arr = np.asarray(all_strikes_from_curves, dtype=float)
        else:
            all_strikes_arr = np.asarray(all_strikes, dtype=float)

        ax2 = plt.subplot(2, 3, 3, polar=True)
        theta = np.deg2rad(all_strikes_arr[np.isfinite(all_strikes_arr)])
        ax2.hist(theta, bins=36, alpha=0.7)
        ax2.set_title("Rose Diagram (All Stations)", fontsize=12)

        ax3 = plt.subplot(2, 3, 4)
        finite_strikes = all_strikes_arr[np.isfinite(all_strikes_arr)]
        ax3.hist(finite_strikes, bins=30, edgecolor="black", alpha=0.7)
        ax3.set_xlabel("Strike (deg)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Strike Distribution", fontsize=12)
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 3, (5, 6))

        station_data = []
        station_labels = []
        for i, mt in enumerate(mt_objects):
            s = self.station_strike(mt)
            period = 1 / mt.frequency
            if unwrap_90:
                s = self._unwrap_strike_90(period, s)
            valid_s = s[np.isfinite(s)]
            if len(valid_s) > 0:
                station_data.append(valid_s)
                station_labels.append(f"S{i + 1}")

        bp = ax4.boxplot(station_data, tick_labels=station_labels, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor("lightblue")
            box.set_alpha(0.7)

        ax4.set_xlabel("Station")
        ax4.set_ylabel("Strike (deg)")
        ax4.set_title("Strike Statistics by Station", fontsize=12)
        ax4.grid(True, alpha=0.3)
        if not unwrap_90:
            ax4.set_ylim(-90, 90)

        if regional_strike is None:
            regional_strike = self._select_strike_true_deg()
        if regional_strike is None or not np.isfinite(float(regional_strike)):
            regional_strike, _ = self.estimate_regional_strike(mt_objects)

        ax4.axhline(
            y=float(regional_strike),
            color="r",
            linestyle="--",
            label=f"Regional: {float(regional_strike):.1f}°",
        )
        ax4.axhline(y=float(regional_strike) + 90, color="r", linestyle=":", alpha=0.5)
        ax4.axhline(y=float(regional_strike) - 90, color="r", linestyle=":", alpha=0.5)
        ax4.legend()

        plt.tight_layout()
        plt.show()

    # -------------------------- station projection / profile --------------------------

    def plot_station_profile(self, mt_objects: Sequence["PrepareData.CustomMT"], strike: Optional[float] = None):
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        profile_pos_m, profile_dist = self.assign_profile_pos_m(mt_objects, strike_use)

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(
            profile_dist,
            [0] * len(profile_dist),
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )
        plt.colorbar(scatter, label="Station ID")

        for i, (pd, lab) in enumerate(zip(profile_dist, station_id_labels)):
            plt.annotate(
                f"{lab}",
                (pd, 0),
                xytext=(0, 15 if i % 2 == 0 else -25),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        plt.xlabel(
            f"Distance perpendicular to strike (km) - Strike: {float(strike_use):.1f}°",
            fontsize=12,
        )
        plt.ylabel("Profile line", fontsize=12)
        plt.title("Station Projection onto Profile Perpendicular to Strike", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)

        margin = (profile_dist.max() - profile_dist.min()) * 0.1
        plt.xlim(profile_dist.min() - margin, profile_dist.max() + margin)

        plt.text(
            0.02,
            0.98,
            f"Profile direction: {float(strike_use) + 90:.1f}°\n(⊥ to strike)",
            transform=plt.gca().transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        sorted_idx = np.argsort(station_ids_num)
        print("\n=== Profile Information (Perpendicular to Strike) ===")
        print(f"Strike: {float(strike_use):.1f}°, Profile direction: {float(strike_use) + 90:.1f}°")
        print(f"Number of stations: {len(profile_dist)}")
        print("\nStation positions (sorted by ID):")
        for idx in sorted_idx:
            print(
                f"  Station {station_id_labels[idx]}: {profile_dist[idx]:8.2f} km  "
                f"(Lat: {lats[idx]:.4f}°, Lon: {lons[idx]:.4f}°)  "
                f"profile_pos_m: {profile_pos_m[idx]:9.1f} m"
            )

        return profile_dist, station_ids_num

    @staticmethod
    def add_scale_bar(ax, lon_lim, lat_lim, bar_length_km: float = 10.0) -> None:
        lon_bar = lon_lim[0] + 0.05 * (lon_lim[1] - lon_lim[0])
        lat_bar = lat_lim[0] + 0.05 * (lat_lim[1] - lat_lim[0])

        lat_center = np.mean(lat_lim)
        lon_per_km = 1 / (111.32 * np.cos(np.deg2rad(lat_center)))
        bar_length_deg = float(bar_length_km) * lon_per_km

        ax.plot([lon_bar, lon_bar + bar_length_deg], [lat_bar, lat_bar], "k-", linewidth=3)
        ax.plot([lon_bar, lon_bar], [lat_bar - 0.001, lat_bar + 0.001], "k-", linewidth=2)
        ax.plot(
            [lon_bar + bar_length_deg, lon_bar + bar_length_deg],
            [lat_bar - 0.001, lat_bar + 0.001],
            "k-",
            linewidth=2,
        )

        ax.text(
            lon_bar + bar_length_deg / 2,
            lat_bar - 0.002,
            f"{bar_length_km:g} km",
            ha="center",
            va="top",
            fontsize=8,
        )

    def plot_station_map_with_profile(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        strike: Optional[float] = None,
        basemap=None,
    ) -> None:
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        if strike_use is None or not np.isfinite(float(strike_use)):
            raise ValueError(
                "Strike angle is not available for plotting. Provide strike explicitly, "
                "or call set_user_strike(), or call compute_strike() first."
            )

        plt.figure(figsize=(12, 10))

        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        lat_margin = lat_range * 0.1
        lon_margin = lon_range * 0.1
        lat_lim = [lats.min() - lat_margin, lats.max() + lat_margin]
        lon_lim = [lons.min() - lon_margin, lons.max() + lon_margin]

        try:
            if basemap == "google":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                import cartopy.feature as cfeature  # type: ignore[import-untyped]

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], crs=ccrs.PlateCarree())

                ax.add_feature(cfeature.LAND, facecolor="lightgray")
                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                ax.add_feature(cfeature.RIVERS, linewidth=0.5)

                gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
                gl.top_labels = False
                gl.right_labels = False

            elif basemap == "terrain":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                import cartopy.feature as cfeature  # type: ignore[import-untyped]

                ax = plt.axes(projection=ccrs.PlateCarree())
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], crs=ccrs.PlateCarree())

                ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
                ax.add_feature(cfeature.LAND, facecolor="lightgreen", alpha=0.3)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
                ax.add_feature(cfeature.LAKES, alpha=0.5)
                ax.add_feature(cfeature.RIVERS, linewidth=0.5)

                gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.3)
                gl.top_labels = False
                gl.right_labels = False

            elif basemap == "osm":
                import cartopy.crs as ccrs  # type: ignore[import-untyped]
                from cartopy.io.img_tiles import OSM  # type: ignore[import-untyped]

                imagery = OSM()
                ax = plt.axes(projection=imagery.crs)
                ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
                ax.add_image(imagery, 10)

            else:
                ax = plt.gca()
                ax.set_xlabel("Longitude (°)", fontsize=12)
                ax.set_ylabel("Latitude (°)", fontsize=12)

        except ImportError:
            print("Note: Install cartopy for better basemaps. Using simple plot.")
            ax = plt.gca()
            ax.set_xlabel("Longitude (°)", fontsize=12)
            ax.set_ylabel("Latitude (°)", fontsize=12)

        scatter = ax.scatter(
            lons,
            lats,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )
        plt.colorbar(scatter, ax=ax, label="Station ID", shrink=0.8)

        for lat, lon, lab in zip(lats, lons, station_id_labels):
            ax.annotate(
                f"{lab}",
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        lon_center = np.mean(lons)
        lat_center = np.mean(lats)

        profile_length = 1.2 * max(
            lat_range * 111.132,
            lon_range * 111.32 * np.cos(np.deg2rad(lat_center)),
        )

        rad = np.deg2rad(float(strike_use) + 90)
        dx_deg = profile_length * np.sin(rad) / (111.32 * np.cos(np.deg2rad(lat_center)))
        dy_deg = profile_length * np.cos(rad) / 111.132

        ax.plot(
            [lon_center - dx_deg, lon_center + dx_deg],
            [lat_center - dy_deg, lat_center + dy_deg],
            "r-",
            linewidth=3,
            alpha=0.7,
            zorder=4,
            label="Profile (⊥ strike)",
        )

        rad_strike = np.deg2rad(float(strike_use))
        dx_strike_deg = (
            profile_length
            * 0.5
            * np.sin(rad_strike)
            / (111.32 * np.cos(np.deg2rad(lat_center)))
        )
        dy_strike_deg = profile_length * 0.5 * np.cos(rad_strike) / 111.132

        ax.plot(
            [lon_center - dx_strike_deg, lon_center + dx_strike_deg],
            [lat_center - dy_strike_deg, lat_center + dy_strike_deg],
            "b--",
            linewidth=2,
            alpha=0.5,
            zorder=3,
            label=f"Strike: {float(strike_use):.1f}°",
        )

        ax.annotate(
            "SW",
            (lon_center - dx_deg, lat_center - dy_deg),
            xytext=(-10, -10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red",
        )
        ax.annotate(
            "NE",
            (lon_center + dx_deg, lat_center + dy_deg),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

        if not hasattr(ax, "projection"):
            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)
            ax.set_xlabel("Longitude (°)", fontsize=12)
            ax.set_ylabel("Latitude (°)", fontsize=12)

        ax.set_title(
            f"Station Location Map with Profile\n"
            f"Strike: {float(strike_use):.1f}°, Profile: {float(strike_use) + 90:.1f}°",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=10)

        self.add_scale_bar(ax, lon_lim, lat_lim)

        plt.tight_layout()
        plt.show()

        print("\n=== Station Map Information ===")
        print(f"Number of stations: {len(mt_objects)}")
        print(f"Latitude range: {lats.min():.4f}° - {lats.max():.4f}°")
        print(f"Longitude range: {lons.min():.4f}° - {lons.max():.4f}°")
        print(f"Strike direction: {float(strike_use):.1f}°")
        print(f"Profile direction (⊥ strike): {float(strike_use) + 90:.1f}°")
        print(f"Profile length: {profile_length:.1f} km")

    def plot_2d_profile_coordinates(self, mt_objects: Sequence["PrepareData.CustomMT"], strike: Optional[float] = None):
        lats = np.array([mt.lat for mt in mt_objects])
        lons = np.array([mt.lon for mt in mt_objects])
        station_ids_num, station_id_labels = self._coerce_station_ids_for_plot(mt_objects)

        strike_use = self._select_strike_true_deg(strike)
        if strike_use is None or not np.isfinite(float(strike_use)):
            raise ValueError(
                "Strike angle is not available for plotting. Provide strike explicitly, "
                "or call set_user_strike(), or call compute_strike() first."
            )

        lat0, lon0 = lats[0], lons[0]
        d_lat = (lats - lat0) * 111.132
        d_lon = (lons - lon0) * 111.32 * np.cos(np.deg2rad(lat0))

        rad_perp = np.deg2rad(float(strike_use) + 90)
        profile_dist = d_lon * np.sin(rad_perp) + d_lat * np.cos(rad_perp)

        rad_para = np.deg2rad(float(strike_use))
        parallel_dist = d_lon * np.sin(rad_para) + d_lat * np.cos(rad_para)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter1 = ax1.scatter(
            lons,
            lats,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.colorbar(scatter1, ax=ax1, label="Station ID")

        for lat, lon, lab in zip(lats, lons, station_id_labels):
            ax1.annotate(
                f"{lab}",
                (lon, lat),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        ax1.set_xlabel("Longitude (°)", fontsize=12)
        ax1.set_ylabel("Latitude (°)", fontsize=12)
        ax1.set_title("Original Coordinates", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        scatter2 = ax2.scatter(
            profile_dist,
            parallel_dist,
            c=station_ids_num,
            cmap="viridis",
            s=200,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.5,
        )
        plt.colorbar(scatter2, ax=ax2, label="Station ID")

        for pd, pl, lab in zip(profile_dist, parallel_dist, station_id_labels):
            ax2.annotate(
                f"{lab}",
                (pd, pl),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Profile line")
        ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

        ax2.set_xlabel(
            f"Distance ⊥ to strike (km) - Profile direction: {float(strike_use) + 90:.1f}°",
            fontsize=12,
        )
        ax2.set_ylabel(
            f"Distance ∥ to strike (km) - Strike: {float(strike_use):.1f}°",
            fontsize=12,
        )
        ax2.set_title("Profile Coordinates", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis("equal")

        plt.suptitle("Station Distribution: Original vs Profile Coordinates", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        return profile_dist, parallel_dist

    # -------------------------- phase tensor ellipse glyph map --------------------------

    @staticmethod
    def skew_cmap_bgr():
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "skew_bgr",
            [(0.0, "#2b6cb0"), (0.5, "#a3d9a5"), (1.0, "#c92a2a")],
            N=256,
        )

    @staticmethod
    def _phase_tensor_ellipse_params(Phi: np.ndarray, normalize: bool = True):
        Phi = np.asarray(Phi, dtype=float)
        if Phi.shape != (2, 2) or not np.isfinite(Phi).all():
            return None
        try:
            U, s, _ = np.linalg.svd(Phi)
        except Exception:
            return None
        if s.size != 2 or not np.isfinite(s).all():
            return None
        if normalize and s[0] > 1e-12:
            s = s / s[0]
        a = float(s[0])
        b = float(s[1])
        angle = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
        return a, b, angle

    def plot_phase_tensor_ellipses_station_freq(
        self,
        mt_objects: Sequence["PrepareData.CustomMT"],
        normalize: bool = True,
        x_scale: float = 0.8,
        y_scale: float = 0.8,
        x_step: float = 1.5,
        skew_threshold: Optional[float] = None,
        skew_clip: Optional[float] = 5.0,
        cmap=None,
        alpha: float = 0.85,
        linewidth: float = 0.3,
        use_station_id: bool = True,
        sort_by_station_id: bool = True,
        figsize=(16, 12),
    ):
        from matplotlib.colors import Normalize
        from matplotlib.patches import Ellipse

        if mt_objects is None or len(mt_objects) == 0:
            raise ValueError("mt_objects is empty")

        try:
            x_step = float(x_step)
        except Exception:
            x_step = 1.0
        if (not np.isfinite(x_step)) or x_step <= 0:
            x_step = 1.0

        if skew_clip is not None:
            skew_clip = float(skew_clip)
            if (not np.isfinite(skew_clip)) or skew_clip <= 0:
                skew_clip = None

        if cmap is None:
            cmap = self.skew_cmap_bgr()

        station_ids = [getattr(mt, "station_id", None) for mt in mt_objects]
        has_ids = all(sid is not None for sid in station_ids)

        if use_station_id and has_ids and sort_by_station_id:
            # Robust sort: numeric IDs sort numerically; non-numeric keep stable order after numeric ones.
            def _try_int(x):
                try:
                    return int(str(x))
                except Exception:
                    return None

            numeric = [_try_int(sid) for sid in station_ids]
            order = sorted(
                range(len(station_ids)),
                key=lambda i: (numeric[i] is None, numeric[i] if numeric[i] is not None else i),
            )
            mts = [mt_objects[i] for i in order]
            station_ids_sorted = [station_ids[i] for i in order]
        else:
            mts = list(mt_objects)
            station_ids_sorted = station_ids

        if use_station_id and has_ids:
            # Use numeric label when possible; otherwise use original string ID.
            x_tick_labels = []
            for sid in station_ids_sorted:
                try:
                    x_tick_labels.append(f"{int(str(sid))}")
                except Exception:
                    x_tick_labels.append(str(sid))
            x_label = "Station ID"
        else:
            x_tick_labels = [f"{i + 1}" for i in range(len(mts))]
            x_label = "Station #"

        n_stations = len(mts)
        x_positions = 1.0 + np.arange(n_stations, dtype=float) * x_step

        records = []
        skew_vals: list[float] = []
        y_logs: list[float] = []
        a_vals: list[float] = []
        b_vals: list[float] = []

        for si, mt in enumerate(mts):
            freqs = np.asarray(mt.frequency)
            if freqs.ndim != 1:
                continue
            x = float(x_positions[si])
            for fi, f in enumerate(freqs):
                if not np.isfinite(f) or float(f) <= 0:
                    continue
                Phi = self.phase_tensor(mt.Z[fi])
                if Phi is None:
                    continue
                skew = self.phase_tensor_skew(mt.Z[fi])
                if skew_threshold is not None and np.isfinite(skew) and abs(skew) >= float(skew_threshold):
                    continue
                params = self._phase_tensor_ellipse_params(Phi, normalize=normalize)
                if params is None:
                    continue
                a, b, angle = params
                if not (np.isfinite(a) and np.isfinite(b) and a >= 0 and b >= 0):
                    continue
                y = float(np.log10(float(f)))
                records.append((x, y, float(skew) if np.isfinite(skew) else np.nan, a, b, angle))
                y_logs.append(y)
                a_vals.append(float(a))
                b_vals.append(float(b))
                if np.isfinite(skew):
                    skew_vals.append(float(skew))

        if len(records) == 0:
            raise RuntimeError("No valid phase tensor ellipses to plot.")

        if skew_clip is not None:
            norm = Normalize(vmin=-skew_clip, vmax=skew_clip, clip=True)
            skew_max = skew_clip
        else:
            if len(skew_vals) == 0:
                skew_max = 1.0
            else:
                skew_max = float(np.nanpercentile(np.abs(skew_vals), 95))
                if not np.isfinite(skew_max) or skew_max < 1e-6:
                    skew_max = float(np.nanmax(np.abs(skew_vals)))
                if not np.isfinite(skew_max) or skew_max < 1e-6:
                    skew_max = 1.0
            norm = Normalize(vmin=-skew_max, vmax=skew_max)

        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        fig, ax = plt.subplots(figsize=figsize)
        for x, y, skew, a, b, angle in records:
            if np.isfinite(skew):
                skew_for_color = float(skew)
                if skew_clip is not None:
                    skew_for_color = float(np.clip(skew_for_color, -skew_clip, skew_clip))
                color = cmap_obj(norm(skew_for_color))
            else:
                color = (0.6, 0.6, 0.6, 1.0)

            e = Ellipse(
                (x, y),
                width=2.0 * float(x_scale) * float(a),
                height=2.0 * float(y_scale) * float(b),
                angle=float(angle),
                facecolor=color,
                edgecolor="black",
                alpha=float(alpha),
                linewidth=float(linewidth),
            )
            ax.add_patch(e)

        max_a = float(np.nanmax(a_vals)) if len(a_vals) else 1.0
        max_b = float(np.nanmax(b_vals)) if len(b_vals) else 1.0
        x_pad = max(0.6 * x_step, 1.3 * float(x_scale) * max_a)
        y_pad = max(0.15, 1.3 * float(y_scale) * max_b)

        ax.set_xlim(float(x_positions[0] - x_pad), float(x_positions[-1] + x_pad))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels, rotation=90)
        ax.set_xlabel(x_label)

        y_min = float(np.nanmin(y_logs))
        y_max = float(np.nanmax(y_logs))
        y0 = np.floor(y_min)
        y1 = np.ceil(y_max)
        decade_ticks = np.arange(y0, y1 + 1)
        ax.set_yticks(decade_ticks)
        ax.set_yticklabels([f"{(10 ** t):g}" for t in decade_ticks])
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_ylabel("Frequency (Hz)")

        title = "Phase Tensor Ellipses (x=station, y=frequency, color=skew)"
        if skew_threshold is not None:
            title += f"  (|skew| < {float(skew_threshold):g}°)"
        if skew_clip is not None:
            title += f"  (skew clipped to ±{float(skew_max):g}°)"
        ax.set_title(title)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Phase tensor skew (deg)")

        plt.tight_layout()
        plt.show()

        return records

    # -------------------------- inspection / rotate-all --------------------------
