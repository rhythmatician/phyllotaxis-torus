#!/usr/bin/env python3
"""
Test suite to verify GPU renderer produces same output as CPU renderer.

Run with: pytest test_renderers.py -v
Or just: python test_renderers.py
"""

# Add parent directory to path to import torus_net_gpu

import numpy as np
from scipy import ndimage


def count_colored_spheres(img: np.ndarray, min_area: int = 50) -> dict:
    """
    Count distinct colored spheres in an image by detecting colored blobs.

    Background is expected to be black (0, 0, 0).
    Colored spheres have RGB values that are not all similar.

    Args:
        img: Image as numpy array [H, W, 3] in range [0, 1]
        min_area: Minimum pixel area to count as a sphere

    Returns:
        dict with sphere count and labeled regions
    """

    # Drop alpha channel if present
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Create a mask for non-black pixels (potential sphere pixels)
    # A pixel is "colored" if it's not black and has distinct RGB channels
    is_colored = np.any(img > 0.1, axis=2)  # Not pure black

    # Additional check: RGB channels should not all be the same (exclude gray/white)
    rgb_max = np.max(img, axis=2)
    rgb_min = np.min(img, axis=2)
    has_color_variation = (rgb_max - rgb_min) > 0.05  # Channels differ significantly

    # Combine conditions: colored and has variation
    sphere_mask = is_colored & has_color_variation

    # Label connected components
    labeled_array, num_features = ndimage.label(sphere_mask)

    # Count regions larger than min_area
    sphere_count = 0
    region_sizes = []
    for label_id in range(1, num_features + 1):
        region_size = np.sum(labeled_array == label_id)
        if region_size >= min_area:
            sphere_count += 1
            region_sizes.append(region_size)

    return {
        "count": sphere_count,
        "region_sizes": sorted(region_sizes, reverse=True),
        "total_colored_pixels": np.sum(sphere_mask),
        "labeled_array": labeled_array,
    }
