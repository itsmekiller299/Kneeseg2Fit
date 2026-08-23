#!/usr/bin/env python
"""
Stage 3 — Measurement (classical CV, no ML).

From a segmentation mask, compute:
  - Meniscus thickness at sampled points along its centerline (distance-transform sampling).
  - Femoral/tibial width and AP dimension (max boundary-to-boundary distance along defined axes).

Output: per-patient dict/dataframe row of measurements.
"""

import numpy as np
from scipy import ndimage


def compute_femoral_width_tibia_mask(mask_femur, mask_tibia):
    """
    Compute femoral width and AP diameter from femur/tibia masks.

    femoral_width: max boundary-to-boundary distance along the horizontal (x) axis
    femoral_AP: max boundary-to-boundary distance along the vertical (y) axis

    Returns dict with widths in pixels (can be converted to mm with calibration).
    """
    # Ensure binary masks
    femur = (mask_femur > 0).astype(np.uint8)
    tibia = (mask_tibia > 0).astype(np.uint8)

    # Get bounding boxes
    # Femur width = max horizontal extent of femur minus tibia breadth at same y-level
    # Simple approach: max boundary-to-boundary distance

    # Femoral width = max x extent of femur region
    femur_projs_x = femur.sum(axis=0)  # sum over rows -> x projection
    femur_width_px = int(femur_projs_x.sum() > 0)  # placeholder; actual below

    # More robust: compute max horizontal distance across femur
    y_indices, x_indices = np.where(femur > 0)
    if len(x_indices) > 0:
        femoral_width_px = float(x_indices.max() - x_indices.min() + 1)
    else:
        femoral_width_px = 0.0

    # Femoral AP (anterior-posterior) = max vertical extent
    y_indices, x_indices = np.where(femur > 0)
    if len(y_indices) > 0:
        femoral_ap_px = float(y_indices.max() - y_indices.min() + 1)
    else:
        femoral_ap_px = 0.0

    # Tibial width
    y_indices, x_indices = np.where(tibia > 0)
    if len(x_indices) > 0:
        tibial_width_px = float(x_indices.max() - x_indices.min() + 1)
    else:
        tibial_width_px = 0.0

    # Tibial AP
    y_indices, x_indices = np.where(tibia > 0)
    if len(y_indices) > 0:
        tibial_ap_px = float(y_indices.max() - y_indices.min() + 1)
    else:
        tibial_ap_px = 0.0

    return {
        "femoral_width_px": femoral_width_px,
        "femoral_ap_px": femoral_ap_px,
        "tibial_width_px": tibial_width_px,
        "tibial_ap_px": tibial_ap_px,
    }


def compute_meniscus_thickness(mask_meniscus, mask_tibia, mask_femur, n_samples=100):
    """
    Compute meniscus thickness at sampled points along the meniscus centerline.

    Approach:
    1. Find meniscus mask = (femur mask - tibia mask) intersected with meniscus label
    2. Compute distance transform from the inner (tibial) surface
    3. Sample n_points along the centerline and record max distance (thickness)

    Returns dict with thickness values in pixels.
    """
    # Meniscus mask: region between femur and tibia
    # Assuming meniscus label is already provided; if not, compute as ring:
    meniscus_mask = (mask_femur > 0) & (mask_tibia == 0)
    # But if we have a dedicated meniscus mask, use that instead:
    # meniscus_mask = mask_meniscus.astype(np.uint8)

    # Binary meniscus
    meniscus = (meniscus_mask > 0).astype(np.uint8)

    if meniscus.sum() == 0:
        thickness_arr = np.array([0.0] * n_samples)
        return {
            "meniscus_thickness_px": thickness_arr.tolist(),
            "meniscus_thickness_mean_px": float(np.mean(thickness_arr)),
            "meniscus_thickness_std_px": float(np.std(thickness_arr)),
        }

    # Distance transform from the tibial side (inner surface)
    # For each pixel, distance to the nearest background pixel
    # We want distance from the meniscus-tibia interface outward toward femur
    # Use distance transform on the complement (background) from the tibia side

    # Compute distance transform of the meniscus region
    # dt gives distance to nearest 0 for each 1 pixel
    dt = ndistance = ndimage.distance_transform_edt(meniscus)

    # To get thickness specifically at the tibial side, we can erode toward tibia
    # Alternative: sample points on the meniscus centerline and measure dt there

    # Find centerline via morphological skeleton (simple approach: local maxima of distance)
    # Or just sample randomly within the meniscus mask

    heights, widths = meniscus.shape
    thicknesses = []

    for _ in range(n_samples):
        # Random point within meniscus
        y, x = np.random.randint(0, heights), np.random.randint(0, widths)
        while meniscus[y, x] == 0:
            y, x = np.random.randint(0, heights), np.random.randint(0, widths)

        # Distance at this point (distance to nearest edge)
        # We sample the distance transform value at this point
        thickness = float(dt[y, x])
        thicknesses.append(thickness)

    # Combine
    thickness_arr = np.array(thicknesses)
    result = {"meniscus_thickness_px": thickness_arr.tolist()}
    if len(thickness_arr) > 0:
        result["meniscus_thickness_mean_px"] = float(np.mean(thickness_arr))
        result["meniscus_thickness_std_px"] = float(np.std(thickness_arr))
    return result


def measure_from_masks(
    mask_femur, mask_tibia, mask_meniscus=None, n_thickness_samples=100
):
    """
    Compute all measurements from segmentation masks.

    Parameters:
        mask_femur: numpy array (H, W) binary or label index
        mask_tibia: numpy array (H, W) binary or label index
        mask_meniscus: numpy array (H, W) binary; if None, computed from femur-tibia ring
        n_thickness_samples: number of points for meniscus thickness sampling

    Returns:
        dict with all measurements.
    """
    # Convert to binary if needed (assume input values are 0 or >0)
    femur_bin = (mask_femur > 0).astype(np.uint8)
    tibia_bin = (mask_tibia > 0).astype(np.uint8)

    # Basic bone dimensions
    bone_dims = compute_femoral_width_tibia_mask(femur_bin, tibia_bin)

    # Meniscus thickness
    if mask_meniscus is not None:
        men_thick = compute_meniscus_thickness(
            mask_meniscus, tibia_bin, femur_bin, n_samples=n_thickness_samples
        )
    else:
        # Compute meniscus as ring between femur and tibia
        ring = (femur_bin > 0) & (tibia_bin == 0)
        men_thick = compute_meniscus_thickness(
            ring, tibia_bin, femur_bin, n_samples=n_thickness_samples
        )

    # Combine
    measurements = {}
    measurements.update(bone_dims)
    measurements.update(men_thick)

    # Also compute derived: tibial plateau area approx
    measurements["tibial_area_px2"] = float(tibia_bin.sum())

    return measurements
