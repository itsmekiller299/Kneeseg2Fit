# Pipeline configuration
DEVICE = "cuda" if __name__ == "__main__" else "cpu"  # simplified; set per-component

# U-Net hyperparameters
UNET_CONFIG = {
    "in_channels": 1,
    "out_channels": 4,  # femur, tibia, meniscus, background
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
    "distance_metric": "euclidean",  # or "cosine"
    "top_k": 3,
}
