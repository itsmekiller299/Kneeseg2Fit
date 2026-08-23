#!/usr/bin/env python
"""
Generate a synthetic implant size CSV spanning realistic ranges.
Schema: femoral_width, femoral_AP, tibial_width, tibial_AP
This can be swapped in with real manufacturer data later.
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

n_small = 30
n_medium = 50
n_large = 30

# Realistic ranges based on typical knee implant sizes (mm)
# Femoral width: ~50-90 mm, Femoral AP: ~30-55 mm
# Tibial width: ~50-95 mm, Tibial AP: ~25-45 mm

small = pd.DataFrame(
    {
        "femoral_width": np.random.normal(65, 5, n_small).round(1),
        "femoral_AP": np.random.normal(38, 3, n_small).round(1),
        "tibial_width": np.random.normal(68, 5, n_small).round(1),
        "tibial_AP": np.random.normal(32, 3, n_small).round(1),
    }
)

medium = pd.DataFrame(
    {
        "femoral_width": np.random.normal(75, 5, n_medium).round(1),
        "femoral_AP": np.random.normal(43, 3, n_medium).round(1),
        "tibial_width": np.random.normal(78, 5, n_medium).round(1),
        "tibial_AP": np.random.normal(36, 3, n_medium).round(1),
    }
)

large = pd.DataFrame(
    {
        "femoral_width": np.random.normal(85, 5, n_large).round(1),
        "femoral_AP": np.random.normal(48, 3, n_large).round(1),
        "tibial_width": np.random.normal(88, 5, n_large).round(1),
        "tibial_AP": np.random.normal(41, 3, n_large).round(1),
    }
)

implants = pd.concat([small, medium, large], ignore_index=True)
implants["size_label"] = (
    ["small"] * n_small + ["medium"] * n_medium + ["large"] * n_large
)

output_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic_implant_sizes.csv"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
implants.to_csv(output_path, index=False)

print(f"Generated {len(implants)} synthetic implant records -> {output_path}")
print(implants.describe())
