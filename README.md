# Torus Phyllotaxis Net (GPU)

A GPU-accelerated renderer for beautiful phyllotaxis-based point distributions on a torus, with correct depth-tested occlusion and customizable edge networks.

## Features

- **GPU Acceleration**: CUDA-enabled PyTorch for fast rendering (or CPU fallback)
- **Correct Occlusion**: Uses ray–sphere intersection per pixel to properly occlude dots and lines behind the torus surface
- **Flexible Edge Networks**: Specify any combination of step sizes to create custom phyllotaxis networks (e.g., Fibonacci spirals)
- **High-Quality Output**: 1920×1080 default with configurable resolution, point styles, and line rendering
- **Colormap Support**: Seam-free "pingpong" magma colormap via matplotlib

## Installation

1. **Clone or download** this repository
2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   Or manually:
   ```powershell
   pip install numpy matplotlib torch
   ```

   For GPU acceleration, install a CUDA-enabled PyTorch build from https://pytorch.org/

## Usage

### Basic Examples

Generate a simple 13–21 Fibonacci spiral:
```powershell
python torus_net_gpu.py --steps 13 21 --out fibonacci.png
```

Generate with nearest neighbors (step size 1):
```powershell
python torus_net_gpu.py --steps 1 --out neighbors.png
```

Generate a complex multi-scale network:
```powershell
python torus_net_gpu.py --steps 1 13 21 34 55 --out multi_scale.png
```

Outputs will be organized by file type:
- `png/fibonacci.png` — the rendered image
- `json/fibonacci.json` — scene metadata and parameters

## File Organization

Outputs are automatically organized by type:
- **PNG images** → `png/` folder
- **JSON metadata** → `json/` folder

Each render produces a matching pair:
- `png/your_image.png` — the rendered visualization
- `json/your_image.json` — complete scene configuration (geometry, camera, rendering params, step sizes)

This makes it easy to reproduce renders or tweak existing ones!

## Command-Line Arguments

#### Required
- `--steps`: Space-separated list of step sizes (e.g., `13 21`)
- `--out`: Output PNG filename

#### Optional (Scene & Geometry)
- `--N`: Number of points on torus (default: 1500)
- `--W`: Image width in pixels (default: 1920)
- `--H`: Image height in pixels (default: 1080)

#### Optional (Camera)
- `--f`: Camera focal length (default: 1.2)
- `--t_step`: Camera step toward center (default: 0.40)

#### Optional (Rendering Style)
- `--dot_px`: Dot radius in pixels (default: 4)
- `--line_px`: Line radius in pixels (default: 1)
- `--line_alpha`: Line alpha blending [0–1] (default: 0.40)
- `--line_samples`: Samples per edge for line interpolation (default: 44)

### Example with Custom Parameters

```powershell
python torus_net_gpu.py --steps 1 13 21 --out output.png --N 2000 --W 2560 --H 1440 --dot_px 6 --line_alpha 0.5
```

## How It Works

### Phyllotaxis on a Torus

Points are distributed across the torus surface using a golden-angle-based phyllotaxis algorithm:
- **u-coordinate** (major circle): Uses the golden angle for even azimuthal spread
- **v-coordinate** (minor circle): Computed via bisection to achieve roughly equal area distribution

### Edges & Networks

The `--steps` parameter defines which points get connected:
- `--steps 1` connects nearest neighbors (i → i+1)
- `--steps 13 21` connects points 13 and 21 steps apart
- `--steps 1 13 21` combines all three networks

### Depth-Tested Occlusion

Unlike traditional splatting, this renderer uses **per-pixel ray–sphere intersection**:
1. Precompute ray directions for every pixel
2. For each dot/line sample, solve the ray–sphere equation to find intersection distance
3. Compare against the torus depth buffer to determine occlusion
4. Only render if the sphere is in front of (or touching) the torus surface

This eliminates the "dots visible on the back side of the donut hole" problem.

### Rendering Pipeline

1. **Depth pass**: Sphere-trace the torus and store ray parameter (t_hit) at first surface hit
2. **Color setup**: Assign colors to points/edges using a seam-free pingpong magma colormap
3. **Line rendering**: Sample curves along (u,v) interpolation on the torus surface; alpha-blend into image with **Phong shading**
4. **Dot rendering**: Render point spheres with per-pixel occlusion testing and **Phong shading**

### Phong Shading

Each sphere (dot or line sample) is shaded using **Phong lighting**, which combines:
- **Ambient**: Global illumination base (default: 0.3)
- **Diffuse**: Directional lighting based on surface normal (default strength: 0.6)
- **Specular**: Bright highlights from the light source (default strength: 0.5, shininess: 32.0)

The lighting direction is set to `(-0.5, 0.3, 1.0)` by default, creating natural-looking highlights on the spheres. Per-pixel normals are computed from the ray–sphere intersection geometry, making even small spheres look convincingly 3D.

## Colormap Note

The current implementation uses matplotlib's exact magma colormap via `plt.get_cmap("magma")`. Colors are computed in `build_lut()` and converted to a PyTorch tensor for efficient GPU lookup.

The colormap is applied with a "pingpong" effect that maps [0, 2π] → [0, 1, 0], creating a seam-free loop from dark (bottom) through bright (middle) and back to dark (top).

## Performance

- **GPU (CUDA)**: ~1–2 seconds per frame (1920×1080, N=1500 points)
- **CPU**: ~30–60 seconds per frame

Rendering time is dominated by the sphere-trace depth pass and per-pixel occlusion tests.

## File Structure

```
torus/
├── torus_net_gpu.py               # Main renderer script
├── test_shading.py                # Demo test suite
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation (this file)
├── docs/                          # Detailed documentation
│   ├── SHADING_INTEGRATION.md    # Technical Phong shading details
│   ├── SHADING_GUIDE.md          # Visual guide & parameter tuning
│   ├── PROJECT_STATUS.md         # Complete feature list & status
│   ├── INTEGRATION_SUMMARY.md    # Integration summary
│   ├── COMPLETION_CHECKLIST.md   # Status checklist
│   ├── QUICK_REFERENCE.py        # Command-line examples
│   └── BUG_FIX.md                # Recent bug fixes
├── png/                           # Output images (auto-created)
└── json/                          # Metadata exports (auto-created)
```

## Tips & Tricks

### Finding Interesting Step Sizes

Fibonacci numbers often produce visually interesting patterns:
- `--steps 3 5 8` (Fibonacci triplet)
- `--steps 5 8 13` (shifted Fibonacci)
- `--steps 1 2 3 5 8 13 21` (entire sequence, can be dense)

### Adjusting Aesthetics

- **Larger dots**: Increase `--dot_px` (e.g., 6–8)
- **Visible lines**: Decrease `--line_alpha` and/or increase `--line_px`
- **Smoother curves**: Increase `--line_samples` (more accurate but slower)
- **Sparse network**: Use fewer/larger step sizes

### High-Resolution Output

For poster-quality images:
```powershell
python torus_net_gpu.py --steps 13 21 --W 4096 --H 2304 --dot_px 8 --out poster.png
```

## License

This code is provided as-is. Feel free to modify and share!

## References

- Golden angle & phyllotaxis: https://en.wikipedia.org/wiki/Phyllotaxis
- Torus SDF: Classic parametric surface distance formulation
- Ray–sphere intersection: Standard computational geometry technique
