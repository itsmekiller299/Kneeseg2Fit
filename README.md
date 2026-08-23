# knee-seg2fit

**Knee Osteoarthritis Analysis & Implant Sizing Pipeline**

A complete end-to-end pipeline for knee OA analysis from MRI, covering segmentation, quantitative measurement, statistical analysis, and implant sizing recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Data Loader](#stage-1---data-loader)
  - [Stage 2 — 2D U-Net Segmentation](#stage-2---2d-unet-segmentation)
  - [Stage 3 — Measurement](#stage-3---measurement)
  - [Stage 4 — Analysis](#stage-4---analysis)
  - [Stage 5 — Implant Matching](#stage-5---implant-matching)
- [Usage](#usage)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Generating Mock Data](#generating-mock-data)
  - [Running Individual Stages](#running-individual-stages)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

`knee-seg2fit` is a Python pipeline for analyzing knee osteoarthritis (OA) from MRI data. It progresses through five stages:

1. **Data Loading** — Load OAI-style MRI datasets (NIfTI or NPY slices) with metadata (KL grade, age, sex)
2. **Segmentation** — 2D U-Net trained on synthetic data to segment femur, tibia, and meniscus
3. **Measurement** — Classical computer vision metrics: meniscus thickness, bone dimensions, tibial area
4. **Analysis** — Statistical tests (t-test, ANOVA), classifier training (logistic regression, random forest), volcano-style plots
5. **Implant Matching** — Nearest-neighbor lookup against an implant size catalog to recommend implant sizes based on measured dimensions

The pipeline uses **Monai** for U-Net training, **SciPy/Scikit-learn** for statistics, and **Streamlit**-ready structure for potential UI deployment.

---

## Project Structure

```
knee-seg2fit/
├── data/                    # Input/derived data
│   ├── create_mock_oai.py   # Generate mock OAI directory structure
│   ├── generate_synth_seg.py # Generate synthetic segmentation dataset
│   ├── generate_synthetic_implants.py  # Generate implant size catalog
│   ├── mock_oai/            # Mock OAI dataset (5 subjects)
│   ├── synth_seg/           # Synthetic segmentation data (train/val meta CSVs)
│   ├── synthetic_implant_sizes.csv  # 100 implant records (small/medium/large)
│   ├── measurements_demo.csv      # Demo measurements from analysis stage
│   └── measurements_30.csv        # Full measurements (30 subjects)
├── src/
│   ├── __init__.py          # Package init
│   ├── config.py            # Pipeline configuration (UNET, measurement, matching params)
│   ├── segmentation/        # Stage 1-2
│   │   ├── __init__.py
│   │   ├── data_loader.py   # OAI dataset loader
│   │   └── train_unet.py    # 2D U-Net training script
│   ├── measurement/         # Stage 3
│   │   ├── __init__.py
│   │   ├── measurements.py  # Meniscus thickness, bone dimensions from masks
│   │   └── tests/           # Measurement tests
│   ├── matching/            # Stage 5
│   │   ├── __init__.py
│   │   └── matcher.py       # Implant nearest-neighbor matcher
│   └── analysis/            # Stage 4
│       ├── __init__.py
│       └── analyze.py       # Statistical analysis & classifier training
├── checks/                  # Validation / CI checks
├── notebooks/               # Jupyter notebooks
├── checkpoints/             # Trained model checkpoints
└── README.md                # This file
```

---

## Installation

```bash
# Clone and enter directory
git clone <repo-url>
cd knee-seg2fit

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
# Or with dev dependencies
pip install -e ".[dev]"
```

The package requires Python >= 3.10. Key dependencies are listed in `pyproject.toml`:
- `torch >= 2.0`
- `monai >= 1.3`
- `numpy >= 1.24`
- `scipy >= 1.11`
- `pandas >= 2.1`
- `scikit-learn >= 1.3`
- `matplotlib >= 3.8`
- `streamlit >= 1.28`
- `nibabel >= 5.0`

---

## Pipeline Stages

### Stage 1 — Data Loader

Loads OAI-style MRI datasets. Supports both mock directory structure (created via `data/create_mock_oai.py`) and real OAI data.

- **Class**: `src.segmentation.data_loader.OAI`
- **Function**: `get_loaders(root_dir, batch_size, num_workers)` returns train/val loaders + metadata CSV
- Loads `.npy` slices or falls back to `.nii.gz` NIfTI volumes
- Extracts middle slice from 3D volumes
- Returns `{image: Tensor, metadata: dict}` per item

### Stage 2 — 2D U-Net Segmentation

Trains a 2D U-Net to segment femur, tibia, and medial meniscus from MRI slices.

- **Script**: `src/segmentation/train_unet.py`
- **Architecture**: MONAI U-Net, `spatial_dims=2`, 4 classes (bg, femur, tibia, meniscus)
- **Training**: 5 epochs, AdamW lr=1e-3, CrossEntropyLoss
- **Data**: Synthetic data from `data/synth_seg/` (train_meta.csv, val_meta.csv)
- **Checkpoints**: Saved to `checkpoints/unet_best.pth` based on best validation Dice
- **Output**: Mean Dice per class printed per epoch

### Stage 3 — Measurement

From segmentation masks, computes quantitative measurements using classical CV techniques.

- **Function**: `src.measurement.measurements.measure_from_masks(mask_femur, mask_tibia, mask_meniscus, n_thickness_samples)`
- **Returns dict** with:
  - `femoral_width_px`, `femoral_ap_px` — bone widths in pixels
  - `tibial_width_px`, `tibial_ap_px` — tibial AP dimensions
  - `tibial_area_px2` — approximate tibial area (pixel count)
  - `meniscus_thickness_px` — thickness at sampled points along centerline
  - `meniscus_thickness_mean_px`, `meniscus_thickness_std_px` — statistics

### Stage 4 — Analysis

Statistical analysis and classifier training on measurements.

- **Script**: `src/analysis/analyze.py`
- **`main()`**: Loads measurements + metadata, runs:
  - **Basic stats**: ANOVA comparing meniscus thickness vs KL grade; t-tests vs OA status (KL>=2) and sex
  - **Classifier training**: Logistic regression + random forest to predict OA from thickness + age + sex
  - **Reports**: Accuracy/AUC for both models, random forest feature importances
  - **Volcano plot data**: Distribution of measurements grouped by OA status
- **`train_classifier()`**: Trains on user-specified feature columns, returns test splits and trained models

### Stage 5 — Implant Matching

Nearest-neighbor lookup against implant size catalog to recommend implant sizes.

- **Class**: `src.matching.matcher.ImplantMatcher`
- **`find_matches(patient_dims, top_k=3)`**: Returns top_k closest implant sizes ranked by Euclidean distance
- **4D distance**: `(femoral_width, femoral_AP, tibial_width, tibial_AP)` with optional z-score normalization
- **Schema**: Implant CSV must have columns `femoral_width, femoral_AP, tibial_width, tibial_AP, size_label`
- **Demo**: `demo_matching()` prints top 3 matches with dimensions

---

## Usage

### Running the Full Pipeline

```bash
# 1. Generate mock OAI data (if you don't have real data)
python data/create_mock_oai.py

# 2. Generate implant size catalog
python data/generate_synthetic_implants.py

# 3. Train U-Net segmentation (5 epochs)
python -m src.segmentation.train_unet

# 4. Run analysis on generated measurements
python -m src.analysis.analyze

# 5. Demo implant matching
python -m src.matching.matcher
```

### Generating Mock Data

```bash
# Create mock OAI directory with 5 synthetic subjects
python data/create_mock_oai.py

# Generate synthetic implant catalog (100 records, small/medium/large)
python data/generate_synthetic_implants.py

# Generate synthetic segmentation dataset
python data/generate_synth_seg.py
```

### Running Individual Stages

**Load OAI dataset and create dataloaders:**

```python
from src.segmentation.data_loader import get_loaders

train_loader, val_loader, meta = get_loaders("data/mock_oai", batch_size=2)
print(f"Dataset: {len(meta)} subjects")
```

**Measure from segmentation masks:**

```python
from src.measurement.measurements import measure_from_masks
import numpy as np

# masks should be numpy arrays (H, W) binary: 0=background, >0=label
measurements = measure_from_masks(mask_femur, mask_tibia, mask_meniscus=None, n_thickness_samples=100)
print(measurements)
```

**Match patient dimensions to implant sizes:**

```python
from src.matching.matcher import ImplantMatcher

matcher = ImplantMatcher("data/synthetic_implant_sizes.csv", normalize=True)
patient = {"femoral_width": 75.0, "femoral_AP": 43.0, "tibial_width": 78.0, "tibial_AP": 36.0}
matches = matcher.find_matches(patient, top_k=3)
# Prints: Top 3 implant matches with size labels and distances
```

**Run analysis on measurements:**

```python
from src.analysis.analyze import main
main("data/measurements_30.csv", "data/mock_oai/metadata.csv")
```

---

## Configuration

Edit `src/config.py` to adjust pipeline parameters:

```python
# U-Net hyperparameters
UNET_CONFIG = {
    "in_channels": 1,
    "out_channels": 4,       # femur, tibia, meniscus, background
    "channels": (16, 32, 64, 128),
    "strides": (2, 2, 2),
    "num_res_blocks": 2,
}

# Measurement config
MEASUREMENT_CONFIG = {
    "meniscus_sample_points": 100,
    "distance_transform_threshold": 0.5,
}

# Implant matching config
MATCHING_CONFIG = {
    "distance_metric": "euclidean",
    "top_k": 3,
}
```

---

## Dependencies

See `pyproject.toml` for the full dependency list. Key packages:

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `monai` | U-Net training, transforms, loss functions |
| `numpy` | Numerical operations |
| `scipy` | Statistical tests (t-test, ANOVA) |
| `pandas` | Data manipulation |
| `scikit-learn` | Logistic regression, random forest, train/test split |
| `matplotlib` | Plotting |
| `nibabel` | NIfTI file I/O |
| `streamlit` | UI deployment (structure ready) |

Dev dependencies: `pytest`, `black`, `ruff`.

---

## License

MIT License. See `LICENSE` (or add one if needed) for full terms.