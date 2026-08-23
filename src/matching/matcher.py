#!/usr/bin/env python
"""
Stage 5 — Implant matching.

Nearest-neighbor lookup: given a patient's measured dimensions,
return the closest implant size(s) from the CSV, ranked by distance.

Distance metric: Euclidean distance in the 4-dimensional space
(femoral_width, femoral_AP, tibial_width, tibial_AP), after optional
normalization/standardization.

Schema of implant CSV (generated in data/synthetic_implant_sizes.csv):
  femoral_width, femoral_AP, tibial_width, tibial_AP, size_label
"""

import numpy as np
import pandas as pd


class ImplantMatcher:
    """Nearest-neighbor matcher against implant size catalog."""

    def __init__(self, implant_csv_path, metric="euclidean", normalize=False):
        """
        implant_csv_path: path to synthetic_implant_sizes.csv (or real manufacturer CSV)
        metric: distance metric, currently only 'euclidean' supported
        normalize: if True, standardize dimensions (z-score) before matching
        """
        self.implant_df = pd.read_csv(implant_csv_path)
        self.metric = metric
        self.normalize = normalize

        if normalize:
            self._fit_scaler()

    def _fit_scaler(self):
        """Compute mean and std for normalization (store for transform)."""
        self.dim_mean = self.implant_df[
            ["femoral_width", "femoral_AP", "tibial_width", "tibial_AP"]
        ].mean()
        self.dim_std = self.implant_df[
            ["femoral_width", "femoral_AP", "tibial_width", "tibial_AP"]
        ].std()

    def _normalize_dims(self, dims):
        """Apply z-score normalization to 4D dimension vector."""
        if self.normalize:
            return (dims - self.dim_mean) / self.dim_std
        return dims

    def find_matches(self, patient_dims, top_k=3):
        """
        Given patient's {femoral_width, femoral_AP, tibial_width, tibial_AP},
        return top_k closest implant sizes ranked by distance.

        patient_dims: dict or dict-like with the 4 dimension keys
        top_k: number of closest matches to return

        Returns list of dicts: [{"size_label": ..., "distance": ..., "dimensions": ...}, ...]
        """
        dims = np.array(
            [
                patient_dims["femoral_width"],
                patient_dims["femoral_AP"],
                patient_dims["tibial_width"],
                patient_dims["tibial_AP"],
            ],
            dtype=float,
        ).reshape(1, -1)

        dims = self._normalize_dims(dims)

        # Compute Euclidean distances to all implant entries
        implant_dims = self.implant_df[
            ["femoral_width", "femoral_AP", "tibial_width", "tibial_AP"]
        ].values

        if self.metric == "euclidean":
            distances = np.linalg.norm(implant_dims - dims, axis=1).flatten()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Combine with size labels and sort
        results = []
        for i, dist in enumerate(distances):
            results.append(
                {
                    "size_label": self.implant_df.iloc[i]["size_label"],
                    "distance": float(dist),
                    "femoral_width": float(implant_dims[i, 0]),
                    "femoral_AP": float(implant_dims[i, 1]),
                    "tibial_width": float(implant_dims[i, 2]),
                    "tibial_AP": float(implant_dims[i, 3]),
                }
            )

        results.sort(key=lambda x: x["distance"])
        return results[:top_k]


def load_implant_catalog(csv_path="data/synthetic_implant_sizes.csv"):
    """Load the implant size catalog and return matcher instance."""
    return ImplantMatcher(csv_path)


def demo_matching(
    patient_dims=None, top_k=3, csv_path="data/synthetic_implant_sizes.csv"
):
    """Run a demo of implant matching with printed results."""
    if patient_dims is None:
        # Example: medium-sized patient
        patient_dims = {
            "femoral_width": 75.0,
            "femoral_AP": 43.0,
            "tibial_width": 78.0,
            "tibial_AP": 36.0,
        }

    matcher = ImplantMatcher(csv_path)
    matches = matcher.find_matches(patient_dims, top_k=top_k)

    print(f"Patient dimensions: {patient_dims}")
    print(f"\nTop {top_k} implant matches:")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['size_label']}: distance = {m['distance']:.2f} mm")
        print(
            f"     Dimensions: FW={m['femoral_width']:.1f}, FAP={m['femoral_AP']:.1f}, "
            f"TW={m['tibial_width']:.1f}, TAP={m['tibial_AP']:.1f}"
        )

    return matches


if __name__ == "__main__":
    demo_matching()
