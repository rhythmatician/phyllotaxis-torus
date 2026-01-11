# Torus Phyllotaxis Net (GPU)

A GPU-accelerated renderer for beautiful phyllotaxis-based point distributions on a torus, with correct depth-tested occlusion and customizable edge networks.

## Features

- **Dual Rendering Modes**:
  - **CPU Mode** (default): PyTorch-based splatting renderer, works on any system
  - **GPU Mode** (`--gpu` flag): OpenGL compute shader ray marching, requires OpenGL 4.3+ GPU
- **OpenGL GPU Acceleration**: Works with Intel HD Graphics 630 and other OpenGL 4.3+ GPUs
- **Correct Occlusion**: Uses ray–sphere intersection per pixel to properly occlude dots and lines behind the torus surface
- **Flexible Edge Networks**: Specify any combination of step sizes to create custom phyllotaxis networks (e.g., Fibonacci spirals)
- **High-Quality Output**: 1920×1080 default with configurable resolution, point styles, and line rendering
- **Colormap Support**: Seam-free "pingpong" colormap via matplotlib (gnuplot, magma, viridis, etc.)

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
   pip install numpy matplotlib torch moderngl
   ```

## Usage

### Basic Examples

Generate a simple 13–21 Fibonacci spiral (CPU mode):
```powershell
python torus_net_gpu.py --steps 13 21 --out fibonacci.png
```

Generate with SDF-based GPU rendering (NEW - true ink lines on torus interior):
```powershell
python torus_net_gpu.py --sdf --steps 13 21 --out sdf_render.png
```

Generate with GPU acceleration (OpenGL compute shaders, sphere-based):
```powershell
python torus_net_gpu.py --gpu --steps 13 21 --out fibonacci_gpu.png
```

Generate with smooth blending (GPU mode only):
```powershell
python torus_net_gpu.py --gpu --smooth --out smooth_blend.png
```

Generate with nearest neighbors (step size 1):
```powershell
python torus_net_gpu.py --steps 1 --out neighbors.png
```

Generate dots only (no edges) for faster GPU rendering:
```powershell
python torus_net_gpu.py --gpu --out dots_only.png
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
- `--out`: Output PNG filename

#### Optional (Rendering Mode)
- `--gpu`: Use OpenGL GPU acceleration (ray marching with compute shaders). Requires OpenGL 4.3+ compatible GPU.
- `--sdf`: Use SDF-based GPU rendering with proper primitives (capsule lines on torus interior, not sampled spheres) - **RECOMMENDED**
- `--smooth`: Enable smooth blending between all spheres (GPU mode only, computationally expensive)
- `--smooth_k`: Smoothing factor for smooth minimum (default: 0.3, higher = more blending)
- `--steps`: Space-separated list of step sizes (e.g., `13 21`). If omitted, only dots are rendered (faster for sphere-based GPU mode).

#### Optional (Scene & Geometry)
- `--N`: Number of points on torus (default: 1597)
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

## SDF-Based GPU Rendering (NEW)

### Overview

The `--sdf` flag enables a new SDF-based GPU renderer that uses proper signed distance field primitives instead of sampled spheres. This is the recommended mode for high-quality ink-style visualization.

### How SDF Mode Works

- **SDF Primitives**: Uses proper geometric primitives instead of sampled points
  - Node spheres at phyllotaxis positions
  - Edge capsules (rounded line segments) for connections
  - Torus shell as background surface
- **Boolean Operations**: Sophisticated SDF composition
  - Intersection: Edge capsules ∩ torus inside shell = ink lines on interior surface
  - Smooth union: Nodes ⊔ edges at junctions (no sharp corners)
  - Difference: Creates the inside-only shell band
- **True Surface Lines**: Edges appear as ink on the torus interior, not floating spheres
- **Smooth Junctions**: Connections between nodes and edges are smoothly blended (no creases)
- **Efficient**: Only N nodes + E edges (not thousands of sampled spheres)

### Key Advantages

1. **Massive performance improvement**: N=1597 with steps uses only ~1600 primitives vs ~200k spheres
2. **Clean line rendering**: True capsule tubes on surface, not blobby sphere chains
3. **Proper shading**: SDF gradients give smooth normals everywhere
4. **Scalable**: Performance scales with N+E, not N×samples

### Usage

```powershell
# SDF-based rendering (recommended for quality)
python torus_net_gpu.py --sdf --steps 13 21 --out sdf_render.png

# Works great with complex networks
python torus_net_gpu.py --sdf --steps 1 13 21 34 55 --N 1597 --out complex_sdf.png
```

**Technical Details**:
- Shell thickness: 5% of torus minor radius
- Node radius: Scaled from `--dot_px`
- Edge radius: Half of `--line_px` (capsules appear thinner)
- Smooth k: Controlled by `--smooth_k` (default 0.3)

## GPU Acceleration Mode (Sphere-Based)

### Overview

The `--gpu` flag enables OpenGL compute shader-based ray marching, which uses a fundamentally different rendering approach than the default CPU splatting method. This mode samples edges into many small spheres.

### How GPU Mode Works

- **Ray Marching**: Each pixel shoots a ray into the scene and marches along it using sphere tracing
- **Sphere SDF**: Distance to the nearest sphere is computed at each step
- **Smooth Blending**: Optional smooth minimum blending between ALL spheres (dots AND lines) using Inigo Quilez's formula
  - Enable with `--smooth` flag (GPU mode only)
  - Disabled by default for performance
  - When enabled, creates organic blob-like connections between nearby spheres
  - Adjust smoothness with `--smooth_k` parameter (default 0.3, higher = more blending)
- **Phong Shading**: Per-pixel lighting computation on sphere surfaces

### Smooth Blending Feature

The `--smooth` flag enables smooth minimum blending in GPU mode, which merges spheres (both dots and line samples) into a unified organic surface:

```powershell
# Enable smooth blending (slower but creates unified surfaces)
python torus_net_gpu.py --gpu --smooth --out organic.png

# Adjust blend smoothness (higher k = more blending)
python torus_net_gpu.py --gpu --smooth --smooth_k 0.5 --out very_smooth.png
```

**Note**: Smooth blending is computationally expensive. It works best with:
- Smaller datasets (N < 500)
- Dots only (no `--steps` argument)
- Dedicated GPU hardware

### Performance Characteristics

**SDF Mode** (`--sdf`) **is best for:**
- High-quality ink-style visualization
- Complex networks with many edges
- Any dataset size (N up to several thousand)
- Clean, professional output

**GPU Mode** (`--gpu`) **is best for:**
- Systems with OpenGL 4.3+ compatible GPUs (Intel HD Graphics 630, NVIDIA, AMD, etc.)
- Rendering with fewer spheres (dots only mode, no `--steps` argument)
- Exploring smooth blending effects when enabled

**CPU Mode is better for:**
- Rendering with many line samples (complex edge networks with `--steps`)
- Systems without dedicated GPU hardware
- Guaranteed consistent performance

### GPU Performance Tips

1. **Dots only** (fastest): Omit `--steps` argument
   ```powershell
   python torus_net_gpu.py --gpu --out fast.png
   ```

2. **Simple edges**: Use 1-2 step sizes
   ```powershell
   python torus_net_gpu.py --gpu --steps 13 --out medium.png
   ```

3. **Complex edges**: Many step sizes may be slow on integrated GPUs
   ```powershell
   # May be slow with --gpu on Intel HD Graphics
   python torus_net_gpu.py --steps 1 13 21 34 55 --out complex.png
   ```

### GPU Requirements

- **Minimum**: OpenGL 4.3 with compute shader support
- **Tested on**: Intel HD Graphics 630, Mesa llvmpipe (software)
- **Recommended**: Dedicated GPU (NVIDIA GTX/RTX, AMD Radeon, etc.)

### Troubleshooting

If GPU rendering is slow or times out:
1. Try dots-only mode (no `--steps`)
2. Reduce `--N` (number of points)
3. Use CPU mode instead (omit `--gpu` flag)

## Colormap Note

The current implementation uses matplotlib's exact colormap via `plt.get_cmap()`. Colors are computed in `build_lut()` and converted to a PyTorch tensor for efficient GPU lookup.

The colormap is applied with a "pingpong" effect that maps [0, 2π] → [0, 1, 0], creating a seam-free loop from dark (bottom) through bright (middle) and back to dark (top).

## Performance

### CPU Mode (Default)
- **Typical**: ~10–60 seconds per frame at 1920×1080
- **Scales with**: Number of spheres (dots + line samples), resolution, anti-aliasing factor
- **Best for**: Complex edge networks, consistent performance across systems

### GPU Mode (`--gpu` flag)
- **Dots only** (N=1597, no edges): ~1–5 seconds on dedicated GPU, ~10–30 seconds on integrated GPU
- **With edges**: Performance depends heavily on GPU hardware
  - Dedicated GPU: ~5–30 seconds
  - Integrated GPU (Intel HD 630): May be slow with >1000 spheres
  - Software rendering (llvmpipe): Very slow, not recommended for production

**Performance bottleneck**: GPU ray marching checks all spheres at each ray step. With line samples, sphere count can reach 4000+, making it compute-intensive.

Rendering time in GPU mode is dominated by the ray marching loop and sphere distance calculations.

## File Structure

```
phyllotaxis-torus/
├── torus_net_gpu.py               # Main renderer script (CPU & GPU modes)
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation (this file)
├── src/                           # Source modules
│   ├── types.py                   # Scene dataclass and configuration
│   ├── shade.py                   # Phong shading implementation
│   ├── gpu_renderer.py            # Sphere-based GPU renderer (compute shaders)
│   └── sdf_gpu_renderer.py        # SDF-based GPU renderer (NEW - proper primitives)
├── shaders/                       # OpenGL compute shaders
│   ├── raymarch.comp              # Sphere-based ray marching shader
│   ├── raymarch_sdf.comp          # SDF-based ray marching shader (NEW)
│   └── common.glsl                # Shared GLSL functions
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
