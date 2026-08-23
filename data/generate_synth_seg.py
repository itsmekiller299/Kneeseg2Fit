#!/usr/bin/env python
"""
Generate synthetic 2D segmentation data for U-Net training.
Creates images with: femur (outer circle), tibia (inner circle), meniscus (ring).
Labels: 0=background, 1=femur, 2=tibia, 3=meniscus
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd

base = os.path.join(os.path.dirname(__file__), "..", "data", "synth_seg")
img_dir = os.path.join(base, "images")
lbl_dir = os.path.join(base, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

n_train = 80
n_val = 20


def make_slice():
    """Create one synthetic segmentation slice."""
    H, W = 128, 128
    img = np.zeros((H, W), dtype=np.float32)
    lbl = np.zeros((H, W), dtype=np.int64)

    # Random centers (avoid edges)
    cx, cy = np.random.randint(30, 98), np.random.randint(30, 98)

    # Random radii
    fem_radius = np.random.randint(35, 55)
    tib_radius = np.random.randint(20, 35)  # must be < fem_radius - 4

    # Create meshgrid
    y, x = np.ogrid[:H, :W]
    dist2 = (x - cx) ** 2 + (y - cy) ** 2

    # Femur: outer region
    fem_mask = dist2 <= fem_radius**2
    # Tibia: inner region (inside smaller circle)
    tib_mask = dist2 <= tib_radius**2
    # Meniscus: ring between tibia and femur
    men_mask = (dist2 <= (fem_radius - 2) ** 2) & (dist2 > (tib_radius + 2) ** 2)

    # Assign label values
    lbl[~fem_mask] = 0  # background
    lbl[fem_mask & ~tib_mask] = 1  # femur tissue
    lbl[tib_mask] = 2  # tibia tissue
    lbl[men_mask] = 3  # meniscus

    # Add some intensity variation
    img[lbl == 1] = np.random.uniform(0.7, 1.0, np.sum(lbl == 1))
    img[lbl == 2] = np.random.uniform(0.4, 0.6, np.sum(lbl == 2))
    img[lbl == 3] = np.random.uniform(0.8, 0.9, np.sum(lbl == 3))
    img[lbl == 0] = np.random.uniform(0.0, 0.3, np.sum(lbl == 0))

    return img, lbl


for i in range(n_train + n_val):
    img, lbl = make_slice()
    # Save as npy for quick loading
    np.save(os.path.join(img_dir, f"train_{i:03d}.npy"), img)
    np.save(os.path.join(lbl_dir, f"train_{i:03d}.npy"), lbl)

    # Also save as NIfTI for MONAI compatibility (cast to float32 to avoid int64 warning)
    if i < 3:  # first few as NIfTI too
        nib.save(
            nib.Nifti1Image(img.astype(np.float32)[np.newaxis, ...], np.eye(4)),
            os.path.join(img_dir, f"train_{i:03d}.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(lbl.astype(np.float32)[np.newaxis, ...], np.eye(4)),
            os.path.join(lbl_dir, f"train_{i:03d}.nii.gz"),
        )

# Create train/val split metadata
train_meta = pd.DataFrame(
    {
        "image_path": [
            os.path.join(img_dir, f"train_{i:03d}.npy") for i in range(n_train)
        ],
        "label_path": [
            os.path.join(lbl_dir, f"train_{i:03d}.npy") for i in range(n_train)
        ],
    }
)
val_meta = pd.DataFrame(
    {
        "image_path": [
            os.path.join(img_dir, f"train_{i:03d}.npy")
            for i in range(n_train, n_train + n_val)
        ],
        "label_path": [
            os.path.join(lbl_dir, f"train_{i:03d}.npy")
            for i in range(n_train, n_train + n_val)
        ],
    }
)

train_meta.to_csv(os.path.join(base, "train_meta.csv"), index=False)
val_meta.to_csv(os.path.join(base, "val_meta.csv"), index=False)

print(f"Generated {n_train} training + {n_val} val synthetic segmentation slices")
print(f"Saved to {base}")
train_meta.head()
