#!/usr/bin/env python
"""
Stage 2 — 2D U-Net segmentation using MONAI.

Trains a U-Net to segment femur, tibia, and medial meniscus from MRI slices.
Uses synthetic data generated in data/synth_seg/.
Includes checkpointing, validation, and Dice-score evaluation.
"""

import os

import numpy as np
import pandas as pd
import torch
from monai.networks.nets import UNet
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SynthSegDataset(Dataset):
    """Simple synthetic segmentation dataset loading .npy pairs."""

    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx])  # shape (H, W) or (1, H, W)
        lbl = np.load(self.label_paths[idx])  # shape (H, W) with values 0-3

        # Ensure channel dim for image
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # (1, H, W)
        # Label stays (H, W) with class indices 0-3

        img_tensor = torch.from_numpy(img).float()
        lbl_tensor = torch.from_numpy(lbl).long()

        if self.transform:
            img_tensor, lbl_tensor = self.transform(img_tensor, lbl_tensor)

        return {"image": img_tensor, "label": lbl_tensor}


def get_synth_dataloaders(data_dir="data/synth_seg", batch_size=2, num_workers=0):
    """Create train/val dataloaders from synthetic segmentation data."""
    train_meta = pd.read_csv(os.path.join(data_dir, "train_meta.csv"))
    val_meta = pd.read_csv(os.path.join(data_dir, "val_meta.csv"))

    train_dataset = SynthSegDataset(
        image_paths=train_meta["image_path"].tolist(),
        label_paths=train_meta["label_path"].tolist(),
    )
    val_dataset = SynthSegDataset(
        image_paths=val_meta["image_path"].tolist(),
        label_paths=val_meta["label_path"].tolist(),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train one epoch."""
    model.train()
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)  # (B, H, W) with values 0-3

        optimizer.zero_grad()
        preds = model(images)  # raw logits (B, C, H, W)
        loss = loss_fn(preds, labels)  # CrossEntropyLoss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate_epoch(model, loader, loss_fn, device, num_classes=4):
    """Validate one epoch: CrossEntropyLoss + mean Dice score."""
    model.eval()
    running_loss = 0.0
    all_dice = {c: [] for c in range(1, num_classes)}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)  # (B, H, W) with values 0-3

        preds = model(images)  # raw logits (B, C, H, W)

        loss = loss_fn(preds, labels)
        running_loss += loss.item() * images.size(0)

        pred_argmax = preds.argmax(dim=1)  # (B, H, W) with values 0-3

        for c in range(1, num_classes):
            pred_c = (pred_argmax == c).float()  # (B, H, W)
            label_c = (labels == c).float()  # (B, H, W)

            for b in range(pred_c.shape[0]):
                intersection = (pred_c[b] * label_c[b]).sum().item()
                union = pred_c[b].sum().item() + label_c[b].sum().item()
                if union > 0:
                    all_dice[c].append(2.0 * intersection / union)
                else:
                    all_dice[c].append(1.0)

    mean_dice = {
        c: np.mean(all_dice[c]) if all_dice[c] else 0.0 for c in range(1, num_classes)
    }
    overall_mean = (
        np.mean([mean_dice[c] for c in mean_dice]) if any(mean_dice.values()) else 0.0
    )
    mean_loss = running_loss / len(loader.dataset) if loader.dataset else 0.0
    return mean_loss, overall_mean, mean_dice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lr = 1e-3
    epochs = 5
    batch_size = 4
    num_classes = 4  # bg, femur, tibia, meniscus

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    train_loader, val_loader = get_synth_dataloaders(batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_dice = 0.0
    ckpt_path = "checkpoints/unet_best.pth"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_dice, per_class_dice = validate_epoch(
            model, val_loader, loss_fn, device, num_classes
        )

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean Dice: {val_dice:.4f}"
        )
        print(
            f"  Per-class Dice (femur, tibia, meniscus): {[(k, f'{v:.4f}') for k, v in per_class_dice.items()]}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint (dice={val_dice:.4f})")

    print(f"\nBest val Dice: {best_dice:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
