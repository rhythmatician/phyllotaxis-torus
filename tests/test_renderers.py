#!/usr/bin/env python3
"""
Test suite to verify GPU renderer produces same output as CPU renderer.

Run with: pytest test_renderers.py -v
Or just: python test_renderers.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import torus_net_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from torus_net_gpu import (
    render_cpu,
    render_gpu,
    render_sdf_gpu,
    Scene,
)


# Test output directory
TEST_OUTPUT_DIR = Path("test_output")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)


def compare_images(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.95):
    """
    Compare two images using SSIM (Structural Similarity Index).
    
    Args:
        img1, img2: Images as numpy arrays [H, W, 3] in range [0, 1]
        threshold: Minimum SSIM score to consider images similar (default 0.95)
    
    Returns:
        dict with comparison metrics
    """
    # Compute SSIM
    ssim_score = ssim(img1, img2, channel_axis=2, data_range=1.0)
    
    # Compute MSE
    mse = np.mean((img1 - img2) ** 2)
    
    # Compute max absolute difference
    max_diff = np.max(np.abs(img1 - img2))
    
    # Check if images are similar
    is_similar = ssim_score >= threshold
    
    return {
        "ssim": ssim_score,
        "mse": mse,
        "max_diff": max_diff,
        "is_similar": is_similar,
        "threshold": threshold,
    }


def load_rendered_image(path: Path) -> np.ndarray:
    """Load a rendered PNG image as numpy array [H, W, 3] in range [0, 1]."""
    img = plt.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img[:, :, :3]  # Drop alpha if present


def render_test_scene(scene: Scene, steps: list[int], renderer: str, test_name: str):
    """
    Render a test scene with specified renderer.
    
    Args:
        scene: Scene configuration
        steps: Edge step sizes
        renderer: "cpu", "gpu", or "sdf-gpu"
        test_name: Name for output file
    
    Returns:
        Path to rendered image
    """
    output_path = TEST_OUTPUT_DIR / f"{test_name}_{renderer}.png"
    
    if renderer == "cpu":
        render_cpu(scene, steps, str(output_path))
    elif renderer == "gpu":
        render_gpu(scene, steps, str(output_path))
    elif renderer == "sdf-gpu":
        render_sdf_gpu(scene, steps, str(output_path))
    else:
        raise ValueError(f"Unknown renderer: {renderer}")
    
    # Load the image from png directory (where render functions save)
    png_path = Path("png") / output_path.name
    return png_path


# ============================================================================
# Phase 1: Basic Geometry Tests
# ============================================================================

def test_single_node():
    """Test 1: Render a single node (N=1) with CPU and GPU."""
    scene = Scene(
        N=1,
        W=400,
        H=400,
        f=1.2,
        t_step=0.4,
        aa_factor=1,
        dot_radius_px=10,  # Large dot so it's clearly visible
        dot_radius_dynamic=False,  # Uniform size
        line_radius_px=1,
        line_alpha=0.4,
    )
    
    steps = []  # No edges
    
    # Render with both
    cpu_path = render_test_scene(scene, steps, "cpu", "test1_single_node")
    gpu_path = render_test_scene(scene, steps, "sdf-gpu", "test1_single_node")
    
    # Load images
    cpu_img = load_rendered_image(cpu_path)
    gpu_img = load_rendered_image(gpu_path)

    # Print filenames for debugging
    print(f"CPU Image Path: {cpu_path}")
    print(f"GPU Image Path: {gpu_path}")
    
    # Compare
    metrics = compare_images(cpu_img, gpu_img, threshold=0.986)
    
    print(f"\n[Test 1: Single Node]")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Max Diff: {metrics['max_diff']:.6f}")
    print(f"  Similar: {metrics['is_similar']}")
    
    # Save comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    plt.savefig(TEST_OUTPUT_DIR / "test1_comparison.png", dpi=100)
    plt.close()
    
    assert metrics['is_similar'], f"Images differ too much: SSIM={metrics['ssim']:.4f}"


def test_few_nodes():
    """Test 2: Render a small phyllotaxis pattern (N=13) with CPU and GPU."""
    scene = Scene(
        N=13,  # Small Fibonacci number
        W=800,
        H=800,
        f=1.2,
        t_step=0.4,
        aa_factor=1,
        dot_radius_px=8,
        dot_radius_dynamic=False,
        line_radius_px=1,
        line_alpha=0.4,
    )
    
    steps = []  # No edges yet
    
    # Render with both
    cpu_path = render_test_scene(scene, steps, "cpu", "test2_few_nodes")
    gpu_path = render_test_scene(scene, steps, "sdf-gpu", "test2_few_nodes")
    
    # Load images
    cpu_img = load_rendered_image(cpu_path)
    gpu_img = load_rendered_image(gpu_path)
    
    # Compare
    metrics = compare_images(cpu_img, gpu_img, threshold=0.986)
    
    print(f"\n[Test 2: Few Nodes (N=13)]")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Max Diff: {metrics['max_diff']:.6f}")
    print(f"  Similar: {metrics['is_similar']}")
    
    # Save comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    plt.savefig(TEST_OUTPUT_DIR / "test2_comparison.png", dpi=100)
    plt.close()
    
    assert metrics['is_similar'], f"Images differ too much: SSIM={metrics['ssim']:.4f}"


def test_medium_nodes():
    """Test 3: Render a medium phyllotaxis pattern (N=89) with CPU and GPU."""
    scene = Scene(
        N=89,  # Medium Fibonacci number
        W=1024,
        H=1024,
        f=1.2,
        t_step=0.4,
        aa_factor=1,
        dot_radius_px=5,
        dot_radius_dynamic=False,
        line_radius_px=1,
        line_alpha=0.4,
    )
    
    steps = []  # No edges yet
    
    # Render with both
    cpu_path = render_test_scene(scene, steps, "cpu", "test3_medium_nodes")
    gpu_path = render_test_scene(scene, steps, "sdf-gpu", "test3_medium_nodes")
    
    # Load images
    cpu_img = load_rendered_image(cpu_path)
    gpu_img = load_rendered_image(gpu_path)
    
    # Compare
    metrics = compare_images(cpu_img, gpu_img, threshold=0.986)
    
    print(f"\n[Test 3: Medium Nodes (N=89)]")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Max Diff: {metrics['max_diff']:.6f}")
    print(f"  Similar: {metrics['is_similar']}")
    
    # Save comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
    plt.savefig(TEST_OUTPUT_DIR / "test3_comparison.png", dpi=100)
    plt.close()
    
    assert metrics['is_similar'], f"Images differ too much: SSIM={metrics['ssim']:.4f}"


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


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GPU vs CPU Renderer Comparison Tests")
    print("=" * 80)
    
    # Run tests individually (not using pytest)
    try:
        test_single_node()
        print("✓ Test 1 passed")
    except AssertionError as e:
        print(f"✗ Test 1 failed: {e}")
    except Exception as e:
        print(f"✗ Test 1 error: {e}")
    
    try:
        test_few_nodes()
        print("✓ Test 2 passed")
    except AssertionError as e:
        print(f"✗ Test 2 failed: {e}")
    except Exception as e:
        print(f"✗ Test 2 error: {e}")
    
    try:
        test_medium_nodes()
        print("✓ Test 3 passed")
    except AssertionError as e:
        print(f"✗ Test 3 failed: {e}")
    except Exception as e:
        print(f"✗ Test 3 error: {e}")
    
    print("\n" + "=" * 80)
    print("Test results saved to:", TEST_OUTPUT_DIR)
    print("=" * 80)
