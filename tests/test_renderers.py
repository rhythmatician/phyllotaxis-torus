#!/usr/bin/env python3
"""
Test suite to verify GPU renderer produces same output as CPU renderer.

Run with: pytest test_renderers.py -v
Or just: python test_renderers.py
"""

import sys
from pathlib import Path
import pytest
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from count_colored_spheres import count_colored_spheres
import os

# Add parent directory to path to import torus_net_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))


from torus_net_gpu import (  # noqa: E402
    render_cpu,
    render_uv_gpu,
    Scene,
)


# Test output directory
TEST_OUTPUT_DIR = Path("test_output")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def compare_images(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.995):
    """
    Compare two images using SSIM (Structural Similarity Index).

    Args:
        img1, img2: Images as numpy arrays [H, W, 3] in range [0, 1]
        threshold: Minimum SSIM score to consider images similar (default 0.995)

    Returns:
        dict with comparison metrics
    """
    # Compute SSIM
    ssim_score: float = ssim(img1, img2, channel_axis=2, data_range=1.0)  # type: ignore

    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)

    # Compute max absolute difference
    max_diff = np.max(np.abs(img1 - img2))

    # Count colored spheres in both images
    spheres1 = count_colored_spheres(img1)
    spheres2 = count_colored_spheres(img2)
    diff_img = np.abs(img1 - img2)
    spheres_diff = count_colored_spheres(diff_img)

    # Check if images are similar
    is_similar = ssim_score >= threshold
    spheres_match = (spheres1["count"] == spheres2["count"]) and (
        spheres_diff["count"] == 0
    )

    return {
        "ssim": ssim_score,
        "mse": mse,
        "max_diff": max_diff,
        "is_similar": is_similar,
        "threshold": threshold,
        "spheres_count_1": spheres1["count"],
        "spheres_count_2": spheres2["count"],
        "spheres_count_diff": spheres_diff["count"],
        "spheres_match": spheres_match,
        "region_sizes_1": spheres1["region_sizes"],
        "region_sizes_2": spheres2["region_sizes"],
    }


def load_rendered_image(path: Path) -> np.ndarray:
    """Load a rendered PNG image as numpy array [H, W, 3] in range [0, 1]."""
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[:, :, :3]  # Drop alpha if present


def render_test_scene(scene: Scene, steps: list[int], renderer: str, test_name: str):
    output_path = TEST_OUTPUT_DIR / f"{test_name}_{renderer}.png"

    # Load path in png directory (where render functions save)
    png_path = Path("png") / output_path.name

    # CPU cache: reuse existing rendered image unless forced
    if renderer == "cpu":
        force = os.getenv("FORCE_CPU_RENDER", "").strip() not in (
            "",
            "0",
            "false",
            "False",
        )
        if png_path.exists() and not force:
            return png_path

        render_cpu(scene, steps, str(output_path))
        return png_path

    if renderer == "gpu":
        render_uv_gpu(scene, steps, str(output_path))
    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    # Verify that GPU renderer didn't fall back to CPU (unchanged)
    if renderer == "gpu":
        json_path = Path("json") / f"{output_path.stem}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)
                actual_renderer = metadata.get("renderer", "unknown")
                if "UV-GPU" not in actual_renderer:
                    raise AssertionError(
                        "UV-GPU renderer fell back to CPU! Check shader compilation errors."
                    )

    return png_path


# ============================================================================
# Phase 1: Basic Geometry Tests
# ============================================================================


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "case_id": "test1_single_node",
                "scene_kwargs": dict(
                    N=1,
                    W=400,
                    H=400,
                    dot_radius_px=10,
                    dot_radius_dynamic=False,
                ),
                "comparison_png": "test1_comparison.png",
            },
            id="single_node",
        ),
        pytest.param(
            {
                "case_id": "test2_few_nodes",
                "scene_kwargs": dict(
                    N=13,
                    W=800,
                    H=800,
                    dot_radius_px=8,
                    dot_radius_dynamic=False,
                ),
                "comparison_png": "test2_comparison.png",
            },
            id="few_nodes",
        ),
        pytest.param(
            {
                "case_id": "test3_medium_nodes",
                "scene_kwargs": dict(
                    N=89,
                    W=1024,
                    H=1024,
                    dot_radius_px=5,
                    dot_radius_dynamic=False,
                ),
                "comparison_png": "test3_comparison.png",
            },
            id="medium_nodes",
        ),
    ],
)
def test_nodes_cpu_vs_gpu(case):
    # Common settings across your three tests (kept consistent with existing code)
    scene = Scene(
        **case["scene_kwargs"],
        f=1.2,
        t_step=0.4,
        aa_factor=1,
        line_radius_px=1,
        line_alpha=0.4,
    )
    steps = []  # No edges yet (matches all three tests)

    # Render with GPU and CPU renderers
    gpu_path = render_test_scene(scene, steps, "gpu", case["case_id"])
    gpu_img = load_rendered_image(gpu_path)

    # Preserve the "GPU single-color output" failure check for the N=13 case

    gpu_img = gpu_img
    if np.allclose(gpu_img, gpu_img[0, 0, :], atol=1e-3):
        raise AssertionError(
            "GPU renderer output is a single color image! Check shader compilation errors."
        )
    cpu_path = render_test_scene(scene, steps, "cpu", case["case_id"])
    cpu_img = load_rendered_image(cpu_path)

    metrics = compare_images(cpu_img, gpu_img)

    # Save comparison image (same pattern you already use)
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cpu_img)
    axes[0].set_title("CPU")
    axes[0].axis("off")
    axes[1].imshow(gpu_img)
    axes[1].set_title(f"GPU (SDF)\nSSIM: {metrics['ssim']:.3f}")
    axes[1].axis("off")
    diff = np.abs(cpu_img - gpu_img)
    axes[2].imshow(diff)
    axes[2].set_title(f"Difference\nMax: {metrics['max_diff']:.3f}")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(TEST_OUTPUT_DIR / case["comparison_png"], dpi=100)
    plt.close()

    assert (
        metrics["spheres_count_1"] == metrics["spheres_count_2"]
    ), "CPU and GPU sphere counts differ"
    if not (metrics["spheres_count_diff"] == 0):
        print("WARNING: Colored spheres differ between CPU and GPU renders")
    assert metrics["is_similar"], f"Images differ too much: SSIM={metrics['ssim']:.4f}"


# ============================================================================
# Phase 2: Line Tests (TODO - implement after nodes work)
# ============================================================================


@pytest.mark.skip(reason="Lines not yet working in GPU renderer")
def test_single_edge():
    """Test 4: Render 2 nodes with 1 connecting edge."""
    pass


@pytest.mark.skip(reason="Lines not yet working in GPU renderer")
def test_few_edges():
    """Test 5: Render simple network (N=13, step=5)."""
    pass
