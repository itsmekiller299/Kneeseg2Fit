#!/usr/bin/env python
"""
Create a small mock OAI-style dataset directory structure for development/testing.
Expected structure (what real OAI data would look like):
  data/mock_oai/
    subjects/
      sub-01/
        ses-01/
          anat/
            sub-01_ses-01_T1w.nii.gz
            sub-01_ses-01_T2w.nii.gz
          mri/
            slice_001.png / .dcm
      sub-02/
        ...
    metadata.csv with columns: subject_id, kl_grade, age, sex
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd

base = os.path.join(os.path.dirname(__file__), "mock_oai")
subjects_dir = os.path.join(base, "subjects")
os.makedirs(subjects_dir, exist_ok=True)

n_subjects = 5

for subj_idx in range(n_subjects):
    subj_id = f"sub-{subj_idx+1:03d}"
    subj_path = os.path.join(subjects_dir, subj_id)
    ses_path = os.path.join(subj_path, "ses-01")
    anat_path = os.path.join(ses_path, "anat")
    mri_path = os.path.join(ses_path, "mri")
    os.makedirs(anat_path, exist_ok=True)
    os.makedirs(mri_path, exist_ok=True)

    # Create a synthetic T1-like volume (64x64x30) with simple anatomy
    # Simple shapes: femur (outer), tibia (inner), meniscus (between)
    vol = np.zeros((64, 64, 30), dtype=np.float32)

    # Femur: outer ellipse at each slice
    for z in range(30):
        y, x = np.ogrid[:64, :64]
        # Outer femur boundary
        fem_radius = 28 - 0.3 * z
        fem_mask = (x - 32) ** 2 + (y - 32) ** 2 <= fem_radius**2
        # Tibia: smaller inner ellipse
        tib_radius = 20 - 0.2 * z
        tib_mask = (x - 32) ** 2 + (y - 32) ** 2 <= tib_radius**2
        # Meniscus: thin ring between femur and tibia
        men_mask = ((x - 32) ** 2 + (y - 32) ** 2 <= (fem_radius - 2) ** 2) & (
            (x - 32) ** 2 + (y - 32) ** 2 > (tib_radius + 2) ** 2
        )
        vol[fem_mask & ~tib_mask, z] = 1.0  # femur tissue
        vol[tib_mask, z] = 2.0  # tibia tissue
        vol[men_mask, z] = 3.0  # meniscus

    # Save NIfTI
    nib.save(
        nib.Nifti1Image(vol, np.eye(4)),
        os.path.join(anat_path, f"{subj_id}_T1w.nii.gz"),
    )

    # Also save a mid-sagittal slice as PNG for quick 2D testing
    mid_slice = vol[:, :, 15]
    np.save(os.path.join(mri_path, "slice_015.npy"), mid_slice)

    # Create metadata
    # KL grade: 0-4, age 40-80, sex: 0=F, 1=M
    kl = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
    age = np.random.randint(45, 80)
    sex = np.random.choice([0, 1])  # 0=Female, 1=Male

    # Store metadata
    if subj_idx == 0:
        meta_rows = []
    meta_rows.append(
        {"subject_id": subj_id, "kl_grade": int(kl), "age": int(age), "sex": sex}
    )

meta_df = pd.DataFrame(meta_rows)
meta_path = os.path.join(base, "metadata.csv")
meta_df.to_csv(meta_path, index=False)
print(f"Mock OAI dataset created at {base}")
print(meta_df)
