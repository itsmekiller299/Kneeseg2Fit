#!/usr/bin/env python
"""
Stage 1 — Data loader for OAI-style dataset.

Loads MRI slices + KL grade + age/sex metadata.
Works with real OAI data or the mock directory created in data/mock_oai/.
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset


class OAI(Dataset):
    """
    Minimal OAI-style dataset loader.
    Returns: {image: Tensor, metadata: dict} per item.
    """

    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: path to .../mock_oai/ (or real OAI root)
        split: not used here but kept for compatibility
        """
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        self.transform = transform or Compose([])

        # Build list of subject folders that have MRI slices
        self.subjects = []
        subj_base = os.path.join(root_dir, "subjects")
        if os.path.isdir(subj_base):
            for subj_name in sorted(os.listdir(subj_base)):
                subj_path = os.path.join(subj_base, subj_name)
                mri_path = os.path.join(subj_path, "ses-01", "mri")
                if os.path.isdir(mri_path):
                    # Check for at least one .npy slice
                    if any(f.endswith(".npy") for f in os.listdir(mri_path)):
                        self.subjects.append(subj_name)
        # Also include any .nii.gz subjects with mri subfolder
        for subj_name in sorted(os.listdir(subj_base)):
            subj_path = os.path.join(subj_base, subj_name)
            mri_path = os.path.join(subj_path, "ses-01", "mri")
            if os.path.isdir(mri_path):
                continue  # already handled by .npy check
            # fallback: check for NIfTI files
            anat_path = os.path.join(subj_path, "ses-01", "anat")
            if os.path.isdir(anat_path) and any(
                f.endswith(".nii.gz") for f in os.listdir(anat_path)
            ):
                # Extract slices from NIfTI later
                self.subjects.append(subj_name)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj_id = self.subjects[idx]

        # --- Load MRI slice ---
        subj_path = os.path.join(self.root_dir, "subjects", subj_id)
        mri_path = os.path.join(subj_path, "ses-01", "mri")

        # Try .npy first (our mock format)
        npy_files = [f for f in os.listdir(mri_path) if f.endswith(".npy")]
        if npy_files:
            slice_data = np.load(os.path.join(mri_path, npy_files[0]))
            # If 3D volume, pick middle slice; if 2D, use as-is
            if slice_data.ndim == 3:
                slice_data = slice_data[:, :, slice_data.shape[2] // 2]
        else:
            # Fallback: load NIfTI and extract middle slice
            nii_path = os.path.join(
                self.root_dir,
                "subjects",
                subj_id,
                "ses-01",
                "anat",
                f"{subj_id}_T1w.nii.gz",
            )
            vol = nib.load(nii_path).get_fdata()
            slice_data = vol[:, :, vol.shape[2] // 2]

        # Add channel dim if grayscale
        if slice_data.ndim == 2:
            slice_data = slice_data[np.newaxis, ...]  # (1, H, W)

        # --- Load metadata ---
        meta_row = self.metadata[self.metadata["subject_id"] == subj_id].iloc[0]
        kl_grade = float(meta_row["kl_grade"])
        age = float(meta_row["age"])
        sex = float(meta_row["sex"])  # 0=F, 1=M

        metadata = {"subject_id": subj_id, "kl_grade": kl_grade, "age": age, "sex": sex}

        # --- Apply transforms ---
        if self.transform:
            slice_data = self.transform(slice_data)

        return {"image": slice_data, "metadata": metadata}


def get_loaders(root_dir="data/mock_oai", batch_size=2, num_workers=0):
    """Create train/val loaders for the OAI dataset."""
    dataset = OAI(root_dir=root_dir)
    # Simple 80/20 split by subject
    n = len(dataset)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    # Use torch.utils.data.Subset
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, dataset.metadata


if __name__ == "__main__":
    train_loader, val_loader, meta = get_loaders()
    print(f"Dataset size: {len(meta)} subjects")
    for batch in train_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Sample metadata: {batch['metadata']}")
        break
