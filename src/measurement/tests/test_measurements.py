#!/usr/bin/env python
"""
Unit tests for measurement module using synthetic masks with known ground-truth dimensions.

Masks are simple ellipses where we can compute expected measurements analytically.
"""

import numpy as np
import pytest

from src.measurement.measurements import (
    compute_femoral_width_tibia_mask,
    compute_meniscus_thickness,
    measure_from_masks,
)


def make_ellipse_mask(center, axes, angle, shape, label_value=1):
    """Create a binary mask with an ellipse."""
    H, W = shape
    y, x = np.ogrid[:H, :W]
    # Rotated ellipse
    x_rot = (x - center[1]) * np.cos(angle) + (y - center[0]) * np.sin(angle)
    y_rot = -(x - center[1]) * np.sin(angle) + (y - center[0]) * np.cos(angle)
    mask = (x_rot**2 / axes[1] ** 2 + y_rot**2 / axes[0] ** 2) <= 1
    return mask


class TestFemoralWidthAP:
    """Test femoral width and AP dimension computation."""

    def test_width_ap_basic(self):
        """Femur ellipse: width=100px, height=80px -> should recover approx those values."""
        H, W = 200, 200
        mask = make_ellipse_mask(
            center=(100, 100), axes=(50, 40), angle=0, shape=(H, W)
        )
        # No separate tibia mask; compute with empty tibia -> femur alone
        # This tests the function handles the mask correctly
        result = compute_femoral_width_tibia_mask(mask, np.zeros_like(mask))
        # Femoral width should be approx 100 (diameter), AP approx 80
        assert (
            70 <= result["femoral_width_px"] <= 130
        ), f"Expected width ~100, got {result['femoral_width_px']}"
        assert (
            55 <= result["femoral_ap_px"] <= 105
        ), f"Expected AP ~80, got {result['femoral_ap_px']}"

    def test_femur_tibia_overlap(self):
        """Femur and tibia ellipses with known overlap."""
        H, W = 200, 200
        femur_mask = make_ellipse_mask(
            center=(100, 100), axes=(50, 40), angle=0, shape=(H, W)
        )
        # Tibia is smaller, centered same place
        tibia_mask = make_ellipse_mask(
            center=(100, 100), axes=(30, 25), angle=0, shape=(H, W)
        )

        result = compute_femoral_width_tibia_mask(femur_mask, tibia_mask)
        # Width and AP should be smaller with tibia subtracted
        assert result["femoral_width_px"] > 0
        assert result["femoral_ap_px"] > 0
        # Tibial measurements should be even smaller
        # (this test just ensures no crashes and reasonable values)


class TestMeniscusThickness:
    """Test meniscus thickness computation."""

    def test_thickness_no_meniscus(self):
        """With no meniscus mask, should return zeros."""
        H, W = 100, 100
        femur = np.zeros((H, W), dtype=np.uint8)
        tibia = np.zeros((H, W), dtype=np.uint8)
        result = compute_meniscus_thickness(
            np.zeros((H, W)), tibia, femur, n_samples=50
        )
        assert result["meniscus_thickness_mean_px"] == 0.0
        assert result["meniscus_thickness_std_px"] == 0.0

    def test_thickness_with_ring(self):
        """Meniscus as a ring between two circles; thickness sampled randomly."""
        H, W = 100, 100
        femur = (
            (np.ogrid[:H, :W][0] - 50) ** 2 + (np.ogrid[:H, :W][1] - 50) ** 2
        ) <= 40**2
        tibia = (
            (np.ogrid[:H, :W][0] - 50) ** 2 + (np.ogrid[:H, :W][1] - 50) ** 2
        ) <= 20**2
        meniscus = (femur > 0) & (tibia == 0)

        result = compute_meniscus_thickness(
            meniscus.astype(np.uint8),
            tibia.astype(np.uint8),
            femur.astype(np.uint8),
            n_samples=200,
        )
        # Mean thickness from random sampling of a ring: ~5-8 px for this config
        # (distance transform averages over all positions, closer to edges)
        assert (
            1 <= result["meniscus_thickness_mean_px"] <= 15
        ), f"Expected mean thickness ~5-8, got {result['meniscus_thickness_mean_px']}"


class TestMeasureFromMasks:
    """Test the full measurement pipeline."""

    def test_full_pipeline_ellipses(self):
        """Full pipeline with ellipse femur/tibia and meniscus ring."""
        H, W = 128, 128
        femur_mask = make_ellipse_mask(
            center=(64, 64), axes=(50, 40), angle=0, shape=(H, W)
        )
        tibia_mask = make_ellipse_mask(
            center=(64, 64), axes=(30, 25), angle=0, shape=(H, W)
        )
        # Meniscus ring
        meniscus_mask = (femur_mask > 0) & (tibia_mask == 0)

        measurements = measure_from_masks(
            femur_mask, tibia_mask, meniscus_mask, n_thickness_samples=50
        )

        # Check keys present
        expected_keys = {
            "femoral_width_px",
            "femoral_ap_px",
            "tibial_width_px",
            "tibial_ap_px",
            "meniscus_thickness_mean_px",
            "meniscus_thickness_std_px",
            "meniscus_thickness_px",
            "tibial_area_px2",
        }
        assert (
            set(measurements.keys()) == expected_keys
        ), f"Missing keys: {expected_keys - set(measurements.keys())}"

        # Values should be positive (except maybe some std)
        assert measurements["femoral_width_px"] > 0
        assert measurements["femoral_ap_px"] > 0
        assert measurements["tibial_width_px"] > 0
        assert measurements["tibial_ap_px"] > 0
        assert measurements["tibial_area_px2"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
