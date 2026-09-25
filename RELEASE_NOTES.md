# MTinv_OT v1.0.0

## Release notes

### Overview

MTinv_OT is a research toolbox for one-dimensional and two-dimensional magnetotelluric (MT) forward modeling and inversion with optimal-transport (OT) data misfits.

This release provides a modular PyTorch-based framework for comparing Sinkhorn OT and classical MSE data misfits in MT inversion problems.

### Main features

- 1D layered MT inversion with OT data misfits and Occam-style regularization
- 2D TE/TM finite-difference forward modeling
- Differentiable PyTorch-based forward solver
- Multi-frequency and multi-station 2D inversion
- 3D and 6D Sinkhorn OT data misfits
- MSE-based inversion for comparison
- Noise modeling and static-shift simulation
- EDI data preparation and profile projection
- Data cleaning, export, and experiment logging
- Model comparison, data-fitting, pseudosection, and convergence plots
- Recovery metrics including RMSE, SSIM, and correlation

### Installation

```bash
git clone https://github.com/MTinv-OT/MTinv_OT.git
cd MTinv_OT
pip install -e ".[ot,dev]"
```

Python 3.10 or newer is required. PyTorch can be installed with CPU or GPU support according to the user's environment.

### License

This project is released under the MIT License.

### Citation

If you use MTinv_OT in your research, please cite the corresponding Zenodo record and the specific version used in your work.

Repository: https://github.com/MTinv-OT/MTinv_OT
