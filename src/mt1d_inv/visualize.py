import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import math
from typing import Any, Tuple, Optional

def visualize_1d_model(model, title="Geo-electric Model"):
    """
    Visualize 1D model (moved from MT1D class).
    """
    # Compute layer depths
    depths = [0]
    for i, thickness in enumerate(model.dz):
        depths.append(depths[i] + thickness.item())
    
    # Expand conductivity for plotting
    sig_plot = []
    for i in range(model.n_layers + 1):
        sig_plot.append(model.sig[i].item())
        if i < model.n_layers:
            sig_plot.append(model.sig[i].item())
    
    depth_plot = [0]
    for depth in depths[1:]:
        depth_plot.append(depth)
        depth_plot.append(depth)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(sig_plot, depth_plot, 'r-', linewidth=2)
    plt.yscale('linear')
    plt.gca().invert_yaxis()
    plt.xlabel('sig (S/m)')
    plt.ylabel('dz (m)')
    plt.title("1D True model")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Add layer boundary lines
    for depth in depths[1:-1]:
        plt.axhline(y=depth, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return plt

class MT1D_3DVisualizer:
    """
    MT 3D visualization class.
    For plotting apparent resistivity, phase and frequency in 3D point cloud.
    """
    
    # Constants
    MU = 4e-7 * math.pi  # 磁导率
    PI = math.pi
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MT 1D forward modelling.
        
        Args:
            freq: Frequency tensor
            dz: Thickness tensor
            sig: Conductivity tensor
            
        Returns:
            zxy: Impedance tensor
            rho: Apparent resistivity tensor
            phs: Phase tensor
        """
        nf = len(freq)
        zxy = torch.zeros(nf, dtype=torch.complex64, device=self.device)
        rho = torch.zeros(nf, dtype=torch.float32, device=self.device)
        phs = torch.zeros(nf, dtype=torch.float32, device=self.device)
        
        n_layers = sig.shape[0]
        
        for kf in range(nf):
            omega = 2.0 * self.PI * freq[kf]
            
            # Compute half-space impedance
            sqrt_arg = torch.complex(torch.tensor(0.0, device=self.device), -omega * self.MU) / sig[-1]
            Z = torch.sqrt(sqrt_arg)
            
            # Recursive impedance from bottom to top
            for m in range(n_layers-2, -1, -1):
                km_arg = torch.complex(torch.tensor(0.0, device=self.device), omega * self.MU * sig[m])
                km = torch.sqrt(km_arg)
                
                Z0 = -1j * omega * self.MU / km
                R = torch.exp(-2.0 * km * dz[m]) * (Z - Z0) / (Z + Z0)
                Z = Z0 * (1.0 + R) / (1. - R)
            
            zxy[kf] = Z
            rho[kf] = torch.abs(Z)**2 / (omega * self.MU)
            phs[kf] = torch.atan2(Z.imag, Z.real) * 180.0 / self.PI
        
        return zxy, rho, phs
    
    def create_3d_pointcloud_from_data(self, freq: torch.Tensor, rho: torch.Tensor, phs: torch.Tensor, 
                                     title: str = "MT 3D Point Cloud") -> plt.Figure:
        """
        Create 3D point cloud from apparent resistivity, phase and frequency data.
        
        Args:
            freq: Frequency data
            rho: Apparent resistivity data
            phs: Phase data
            title: Plot title
            
        Returns:
            fig: matplotlib figure object
        """
        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_np = rho.detach().cpu().numpy() if torch.is_tensor(rho) else np.array(rho)
        phs_np = phs.detach().cpu().numpy() if torch.is_tensor(phs) else np.array(phs)
        
        # Create 3D figure
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Create scatter plot, color mapped by frequency
        scatter = ax.scatter(np.log10(rho_np), phs_np, np.log10(freq_np), 
                           c=np.log10(freq_np), cmap='jet', 
                           s=50, alpha=0.8, marker='o')
        
        # Set axis labels
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=15, labelpad=15)
        ax.set_ylabel('Phase [degrees]', fontsize=15, labelpad=15)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=15, labelpad=0, rotation=180)
        
        # Set title
        ax.set_title(title, fontsize=25, pad=10)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('log10(Frequency [Hz])', fontsize=15)
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_3d_pointcloud_from_model(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor,
                                      title: str = "MT 3D Point Cloud from Model") -> plt.Figure:
        """
        Create 3D point cloud from resistivity model via forward modelling.
        
        Args:
            freq: Frequency data
            dz: Layer thickness
            sig: Conductivity
            title: Plot title
            
        Returns:
            fig: matplotlib figure object
        """
        # Forward modelling
        zxy, rho, phs = self.mt1d_forward(freq, dz, sig)
        
        # Create 3D point cloud
        fig = self.create_3d_pointcloud_from_data(freq, rho, phs, title)
        
        return fig
    
    def plot_comparison_3d_pointcloud(self, freq: torch.Tensor, 
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT 3D Point Cloud Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D point cloud comparison of observed vs model-predicted data.
        
        Args:
            freq: Frequency data
            rho_obs: Observed apparent resistivity
            phs_obs: Observed phase
            dz_model: Model layer thickness
            sig_model: Model conductivity
            title: Plot title
            save_path: Image save path (optional)
        Returns:
            fig: matplotlib figure object
        """
        # Forward modelling for model prediction
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)
        
        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot observed data (red)
        scatter_obs = ax.scatter(np.log10(rho_obs_np), phs_obs_np, np.log10(freq_np), 
                               c='red', s=60, alpha=0.8, marker='o', label='Observed Data')
        
        # Plot model prediction (blue)
        scatter_pred = ax.scatter(np.log10(rho_pred_np), phs_pred_np, np.log10(freq_np), 
                                c='blue', s=60, alpha=0.6, marker='^', label='Model Prediction')
        
        # Set axis labels
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=12, labelpad=15)
        ax.set_ylabel('Phase [degrees]', fontsize=12, labelpad=15,rotation=180)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=12, labelpad=1, rotation=0)

        # Set title
        ax.set_title(title, fontsize=14, pad=15)
        
        # Add legend
        ax.legend(fontsize=12, loc='upper left')
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_comparison_3d_with_lines(self, freq: torch.Tensor, 
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT 3D Point Cloud with Connections",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 3D point cloud comparison of observed vs model-predicted with connecting lines.
        Adds empty subplots on left and right.
        """
        # Forward modelling for model prediction
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)

        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)

        # Create GridSpec layout
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])

        # Left empty subplot
        ax_left = fig.add_subplot(gs[0], frameon=False)
        ax_left.axis('off')

        # Main 3D plot
        ax = fig.add_subplot(gs[1], projection='3d')
        scatter_obs = ax.scatter(np.log10(rho_obs_np), phs_obs_np, np.log10(freq_np), 
                                c='red', s=60, alpha=0.8, marker='o', label='Observed Data')
        scatter_pred = ax.scatter(np.log10(rho_pred_np), phs_pred_np, np.log10(freq_np), 
                                c='blue', s=60, alpha=0.6, marker='^', label='Inverted Data')
        for i in range(len(freq_np)):
            ax.plot([np.log10(rho_obs_np[i]), np.log10(rho_pred_np[i])],
                    [phs_obs_np[i], phs_pred_np[i]],
                    [np.log10(freq_np[i]), np.log10(freq_np[i])],
                    'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('log10(Apparent Resistivity [Ω·m])', fontsize=24, labelpad=25)
        ax.set_ylabel('Phase [degrees]', fontsize=24, labelpad=20)
        ax.set_zlabel('log10(Frequency [Hz])', fontsize=24, labelpad=15, rotation=180)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(title, fontsize=40)
        ax.legend(fontsize=18, loc='upper right')
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

        # Right empty subplot
        ax_right = fig.add_subplot(gs[2], frameon=False)
        ax_right.axis('off')

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    def plot_rho_freq_2d(self, freq: torch.Tensor, rho: torch.Tensor, 
                        title: str = "Apparent Resistivity vs Frequency",
                        label: str = "Data", color: str = 'blue',
                        figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot apparent resistivity vs frequency (2D).
        
        Args:
            freq: Frequency data
            rho: Apparent resistivity data
            title: Plot title
            label: Data label
            color: Line color
            figsize: Figure size
            
        Returns:
            fig: matplotlib figure object
        """
        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_np = rho.detach().cpu().numpy() if torch.is_tensor(rho) else np.array(rho)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot log-log
        ax.loglog(freq_np, rho_np, 'o-', color=color, markersize=6, linewidth=2, label=label)
        
        # Set axis labels
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Apparent Resistivity [Ω·m]', fontsize=12)
        
        # Set title
        ax.set_title(title, fontsize=14)
        
        # Add grid
        ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        if label:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig

    def plot_sensitivity_matrix(self, J, z_grid, freqs, save_path=None):
        """
        Plot sensitivity matrix heatmap.
        Args:
            J: Sensitivity matrix [n_freq, n_layer]
            z_grid: Depth grid [n_layer + 1]
            freqs: Frequency list
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use pcolormesh or imshow
        # Note: typically x-axis is depth (layer), y-axis is frequency (period)
        
        # Build grid for plotting
        # y-axis: frequency index (0 to n_freq)
        # x-axis: layer index (0 to n_layer)
        
        im = ax.imshow(J, aspect='auto', cmap='RdBu_r', origin='upper',
                       extent=[0, len(z_grid)-1, 0, len(freqs)])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sensitivity (Partial Derivative)')
        
        # Set axis labels
        ax.set_ylabel('Frequency Index (High -> Low)')
        ax.set_xlabel('Layer Index (Shallow -> Deep)')
        ax.set_title('Sensitivity Matrix (Jacobian)')
        
        # Optional: convert ticks to actual frequency and depth values
        # ax.set_yticks(...)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    def plot_phs_freq_2d(self, freq: torch.Tensor, phs: torch.Tensor,
                        title: str = "Phase vs Frequency",
                        label: str = "Data", color: str = 'red',
                        figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot phase vs frequency (2D).
        
        Args:
            freq: Frequency data
            phs: Phase data
            title: Plot title
            label: Data label
            color: Line color
            figsize: Figure size
            
        Returns:
            fig: matplotlib figure object
        """
        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        phs_np = phs.detach().cpu().numpy() if torch.is_tensor(phs) else np.array(phs)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot semilog (log frequency, linear phase)
        ax.semilogx(freq_np, phs_np, 's-', color=color, markersize=5, linewidth=2, label=label)
        
        # Set axis labels
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('Phase [degrees]', fontsize=12)
        
        # Set title
        ax.set_title(title, fontsize=14)
        
        # Add grid
        ax.grid(True, which='both', alpha=0.3)
        
        # Add legend
        if label:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig

    def plot_comparison_2d(self, freq: torch.Tensor,
                          rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                          rho_pred: torch.Tensor, phs_pred: torch.Tensor,
                          title: str = "MT Data Comparison") -> plt.Figure:
        """
        Plot 2D comparison of observed vs predicted data.
        
        Args:
            freq: Frequency data
            rho_obs: Observed apparent resistivity
            phs_obs: Observed phase
            rho_pred: Predicted apparent resistivity
            phs_pred: Predicted phase
            title: Plot title
            
        Returns:
            fig: matplotlib figure object
        """
        # Convert to numpy arrays
        freq_np = freq.detach().cpu().numpy() if torch.is_tensor(freq) else np.array(freq)
        rho_obs_np = rho_obs.detach().cpu().numpy() if torch.is_tensor(rho_obs) else np.array(rho_obs)
        phs_obs_np = phs_obs.detach().cpu().numpy() if torch.is_tensor(phs_obs) else np.array(phs_obs)
        rho_pred_np = rho_pred.detach().cpu().numpy() if torch.is_tensor(rho_pred) else np.array(rho_pred)
        phs_pred_np = phs_pred.detach().cpu().numpy() if torch.is_tensor(phs_pred) else np.array(phs_pred)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot apparent resistivity comparison
        ax1.loglog(freq_np, rho_obs_np, 'ro-', markersize=6, linewidth=2, label='Observed')
        ax1.loglog(freq_np, rho_pred_np, 'b^-', markersize=5, linewidth=2, label='Inverted')
        ax1.set_xlabel('Frequency [Hz]', fontsize=20)
        ax1.set_ylabel('Apparent Resistivity [Ω·m]', fontsize=20)
        ax1.set_title('Apparent Resistivity Comparison', fontsize=25)
        ax1.legend(fontsize=20)
        ax1.tick_params(axis='both', labelsize=16)
        
        # Plot phase comparison
        ax2.semilogx(freq_np, phs_obs_np, 'ro-', markersize=6, linewidth=2, label='Observed')
        ax2.semilogx(freq_np, phs_pred_np, 'b^-', markersize=5, linewidth=2, label='Inverted')
        ax2.set_xlabel('Frequency [Hz]', fontsize=20)
        ax2.set_ylabel('Phase [degrees]', fontsize=20)
        ax2.legend(fontsize=20)
        ax2.tick_params(axis='both', labelsize=16)
        
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)

        # Set overall title
        fig.suptitle(title, fontsize=30, y=0.98)
        
        plt.tight_layout()
        return fig

    def plot_comparison_2d_from_model(self, freq: torch.Tensor,
                                    rho_obs: torch.Tensor, phs_obs: torch.Tensor,
                                    dz_model: torch.Tensor, sig_model: torch.Tensor,
                                    title: str = "MT Data Comparison with Model") -> plt.Figure:
        """
        Plot 2D comparison from model prediction.
        
        Args:
            freq: Frequency data
            rho_obs: Observed apparent resistivity
            phs_obs: Observed phase
            dz_model: Model layer thickness
            sig_model: Model conductivity
            title: Plot title
            
        Returns:
            fig: matplotlib figure object
        """
        # Forward modelling for model prediction
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(freq, dz_model, sig_model)
        
        # Plot comparison
        fig = self.plot_comparison_2d(freq, rho_obs, phs_obs, rho_pred, phs_pred, title)
        
        return fig



