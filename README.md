# MTinv_OT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22636468.svg)](https://doi.org/10.5281/zenodo.22636468)

A research toolbox for 1D/2D magnetotelluric (MT) forward modeling and inversion with **Optimal Transport (OT)** data misfits.

> Within a conventional MT inversion framework, geomloss Sinkhorn geometric OT distances are used as alternatives or complements to classical L2 data misfits, aiming for improved robustness to noise and non-Gaussian errors.

**Repository**: https://github.com/MTinv-OT/MTinv_OT

---

## Features

| Module | Description |
|------|------|
| **MT 1D OT inversion** (`mt1d_inv`) | Layered 1D models; apparent resistivity/phase embedded as 3D point clouds; Sinkhorn OT + Occam-style constraints |
| **MT 2D FD forward** (`mt2d_inv.forward`) | TE/TM total-field finite differences; PyTorch-differentiable; real-block solver |
| **MT 2D OT inversion** (`mt2d_inv.inversion`) | Multi-frequency, multi-station, four-component joint inversion; 3D/6D Sinkhorn OT; MSE comparison mode |
| **Data preparation** (`mt2d_inv.data_prep`) | EDI reading, strike estimation, profile projection, cleaning, and export |
| **Experiment I/O** (`mt2d_inv.io`) | `ExperimentLogger` for config / history / figures / metrics |
| **Plotting** (`mt2d_inv.plotting`) | Model comparison, data fitting, pseudosections, OT vs MSE convergence |

---

## Code structure

The refactored `mt2d_inv` package uses **Mixin composition** instead of a single monolithic `MTinv_2d.py`:

```
src/
├── mt1d_inv/                    # 1D inversion
│   ├── MTinv.py                 # MT1DInverter
│   ├── model.py, constraints.py, optimizer.py
│   └── visualize.py
│
└── mt2d_inv/
    ├── __init__.py              # Public API
    ├── models.py                # MT2DTrueModels (COMMEMI, checkerboard, ...)
    ├── constraints.py           # 2D smoothness / roughness constraints
    ├── optimizer.py             # OptimizerConfig
    │
    ├── forward/
    │   └── solver.py            # MT2DFD_Torch (2D FD forward)
    │
    ├── inversion/               # 2D inversion core (Mixin composition)
    │   ├── base.py              # MT2DInverter
    │   ├── weighted_cost.py     # MT2DInverterWeightedCost
    │   ├── data.py              # Synthetic/observed data, static shift, error propagation
    │   ├── ot.py                # Sinkhorn OT data term
    │   ├── regularization.py    # Regularization and adaptive λ
    │   └── metrics.py           # Recovery metrics: RMSE / SSIM / correlation
    │
    ├── data_prep/               # Field-data preparation pipeline
    │   ├── prepare.py           # PrepareData (main entry)
    │   ├── edi.py               # EDI parsing and impedance transforms
    │   ├── strike.py            # Strike estimation and profile projection
    │   ├── cleaning.py          # Data cleaning
    │   ├── export.py            # Export tensors for inversion
    │   └── grid.py              # Grid extent helpers
    │
    ├── plotting/                # Visualization
    │   ├── inversion.py         # Model comparison, data fitting, sections
    │   ├── pseudosection.py     # Apparent-resistivity pseudosections
    │   ├── comparison.py        # OT vs MSE convergence
    │   └── prepare_data.py      # Plots for the preparation stage
    │
    └── io/
        └── experiment.py        # ExperimentLogger
```

### Main public API

```python
from mt2d_inv import MT2DInverter, MT2DInverterWeightedCost, MT2DFD_Torch, MT2DTrueModels
from mt2d_inv.data_prep import PrepareData
from mt2d_inv.io import ExperimentLogger
from mt2d_inv.plotting import (
    plot_model_comparison,
    plot_data_fitting,
    plot_ot_mse_convergence,
    plot_ot_mse_pseudosection_from_npz,
)
```

---

## Requirements

- Python ≥ 3.10
- PyTorch (CPU or GPU)
- numpy, matplotlib, scikit-image, pandas

```bash
pip install -e ".[ot,dev]"
# or step by step
pip install -e .
pip install geomloss jupyter   # OT inversion and notebook experiments
```

If you hit an OpenMP conflict on Windows:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Installation

```bash
git clone https://github.com/MTinv-OT/MTinv_OT.git
cd MTinv_OT
pip install -e ".[ot,dev]"
```

If a notebook is not launched from the project root, add `src` to `sys.path`:

```python
import sys
from pathlib import Path

def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src" / "mt2d_inv" / "__init__.py").is_file():
            return p
    raise RuntimeError("Cannot find project root")

root = find_project_root(Path.cwd())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
```

---

## Quick start

### 2D synthetic inversion (OT vs MSE)

```python
import torch
from mt2d_inv import MT2DInverterWeightedCost, MT2DTrueModels
from mt2d_inv.io import ExperimentLogger

device = "cuda" if torch.cuda.is_available() else "cpu"
yn, zn, nza, sig_true = MT2DTrueModels.create_commemi_2d4(nza=10, device=device)
freqs = torch.logspace(0, -5, 30, device=device)
stations = torch.linspace(-15000, 15000, 21, device=device)

inv = MT2DInverterWeightedCost(
    yn=torch.tensor(yn, dtype=torch.float64, device=device),
    zn=torch.tensor(zn, dtype=torch.float64, device=device),
    nza=nza, freqs=freqs, stations=stations,
    device=device, random_seed=123,
)
inv.set_forward_operator()
inv.sig_true = sig_true
inv.create_synthetic_data(noise_level=0.01, noise_type="gaussian")
inv.initialize_model(initial_sigma=0.01)

inv.run_inversion(n_epochs=300, mode="6dot")   # OT
logger = ExperimentLogger(model_tag="demo", output_root="test_results")
logger.save_from_inverter(inv, run_name="ot_demo")
```

### Field-data preparation (EDI → inversion tensors)

```python
from mt2d_inv.data_prep import PrepareData

prep = PrepareData(
    edi_dir="path/to/edi",
    n_freq_target=20,
    freq_min_hz=1e-4,
    freq_max_hz=1e4,
)
prep.run_all_simple(rotate=True, strike_true_deg=45.0)
data_dict = prep.export_data_dict_for_2d_inversion()
```

### Static shift

Synthetic data supports two static-shift modes:

**1. Random mode (original interface)**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.15,          # std σ of the log10 multiplier (not variance)
    shift_modes=("xy", "yx"),       # xy=TE, yx=TM
    shift_stations="random",        # "all" | "middle" | "random"
    shift_ratio=0.2,                # fraction of affected stations
)
```

**2. Fixed-amplitude mode (new interface)**

```python
inv.create_synthetic_data(
    noise_level=0.01,
    static_shift_std=0.0,
    shift_station_indices=[8, 9, 10, 11],   # 0-based station indices
    static_shift_log={
        "xy": 0.15,    # TE: direct log10 multiplier (×10^0.15 ≈ 1.41)
        "yx": 0.15,    # TM: same; set one or both
    },
)
```

| Argument | Meaning |
|------|------|
| `static_shift_std` | Random mode: Gaussian **std** σ of the log10 multiplier |
| `static_shift_log` | Fixed mode: **direct log10 multiplier** (not variance / std) |
| `shift_station_indices` | Explicit affected stations; overrides `shift_stations` |

Static-shift settings are written to `static_shift.json`; recovery metrics (including SSIM) are written to `summary.csv`.

---

## Experiment notebooks

Notebooks under `tests/` locate the project root automatically. Synthetic runs invert **6D OT** (`mode="6dot"`) and/or **MSE** (`mode="mse"`). Field notebooks expect EDI files next to the notebook (or in the folder named in `EDI_DIR`).

### Convergence — `tests/synthetic/convergence/`

Same synthetic observations inverted with OT then MSE; overlay RMS χ² curves. COMMEMI 2D-1 / 2D-4, 21 stations, 1% Gaussian noise.

| Notebook | Model |
|----------|------|
| `commemi_2d1_21-ot-mse-noise0.01.ipynb` | COMMEMI 2D-1 |
| `commemi_2d4_21-ot-mse-noise0.01.ipynb` | COMMEMI 2D-4 |

### Noise comparison — `tests/synthetic/noise_compare/`

Separate OT and MSE notebooks at Gaussian impedance noise 1%, 2%, and 4% (`noise_level` = 0.01 / 0.02 / 0.04). 21 stations.

| Notebook pattern | Model | Mode |
|----------|------|------|
| `commemi_2d1_21-{ot,mse}-0.01.ipynb` (also `0.02`, `0.04`) | COMMEMI 2D-1 | OT or MSE |
| `commemi_2d4_21-{ot,mse}-0.01.ipynb` (also `0.02`, `0.04`) | COMMEMI 2D-4 | OT or MSE |

### Random static shift — `tests/synthetic/random_static/`

Random log10 static shift (`static_shift_std=0.15`) on a fraction of stations (`shift_ratio`). OT and MSE share the same `obs_data`. 21 stations.

| Notebook | Model | `shift_ratio` |
|----------|------|--------|
| `commemi_2d1_21_shift-0.1.ipynb` | COMMEMI 2D-1 | 0.1 |
| `commemi_2d1_21_shift-0.15.ipynb` | COMMEMI 2D-1 | 0.15 |
| `commemi_2d1_21_shift-0.2.ipynb` | COMMEMI 2D-1 | 0.2 |
| `commemi_2d4_21_shift-0.1.ipynb` | COMMEMI 2D-4 | 0.1 |
| `commemi_2d4_21_shift-0.15.ipynb` | COMMEMI 2D-4 | 0.15 |
| `commemi_2d4_21_shift-0.2.ipynb` | COMMEMI 2D-4 | 0.2 |

### Field data — `tests/AKBST-AMT-L08/` and `tests/Cascadia/`

| Notebook | Profile |
|----------|------|
| `tests/AKBST-AMT-L08/ot.ipynb`, `mse.ipynb` | AKBST AMT line L08 |
| `tests/Cascadia/ot.ipynb`, `mse.ipynb` | Cascadia |

### Output layout

Each `ExperimentLogger` run writes:

```
test_results/<model_tag>/<timestamp>_<run_name>/
├── config.json
├── summary.csv
├── history.csv
├── static_shift.json
├── apparent_resistivity.npz
├── final_model.npz
└── figures/
    ├── model_comparison.png
    ├── data_fitting/
    └── ...
```

---

## Plotting helpers

| Function | Purpose |
|------|------|
| `plot_model_comparison(inv)` | True vs inverted log10(ρ); prints SSIM |
| `plot_data_fitting(inv, station_indices=...)` | Station data-fit curves |
| `plot_ot_mse_convergence(hist_ot, hist_mse)` | OT vs MSE convergence |
| `plot_ot_mse_pseudosection_from_npz(npz_ot, npz_mse)` | Pseudosection comparison |
| `plot_rho_fitting_from_npz(npz_path)` | Reload data fits from saved npz |

SSIM from `inv.compute_recovery_rate()['ssim']` matches `plot_model_comparison` (log10(ρ) domain, air layer excluded) and is saved in `summary.csv`.

---

## MT 1D inversion

The 1D module is independent; entry point `mt1d_inv`:

```python
from mt1d_inv import MT1D, MT1DInverter

inv = MT1DInverter(device="cuda", use_sinkhorn=True)
inv.generate_synthetic_data(true_dz=..., true_sig=..., noise_level=0.05)
inv.run_inversion(num_epochs=800, use_adaptive_lambda=True)
inv.plot_data_fit()
inv.plot_model_comparison()
```

---

## Contributors

- Authors: Xinran Liu, Xuanzhang Chen, Bo Yang, Ziyu Tang
- Contact: xinran.liu@zju.edu.cn, bo.yang@zju.edu.cn

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt).
