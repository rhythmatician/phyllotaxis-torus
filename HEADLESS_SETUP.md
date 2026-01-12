# Headless OpenGL Setup Guide

This guide explains how to enable OpenGL rendering in headless environments (servers, CI/CD pipelines, Docker containers) using Mesa's software renderer (llvmpipe).

## Overview

The problem: OpenGL applications typically require an X11 display server, which isn't available in:
- CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- Headless servers
- Docker containers
- SSH sessions without X forwarding

The solution: Use Mesa's EGL (Embedded-System Graphics Library) backend with llvmpipe software renderer, which provides CPU-based OpenGL without requiring a physical GPU or display.

## Quick Setup

### Ubuntu/Debian

```bash
# Install EGL libraries
sudo apt-get update
sudo apt-get install -y libegl1 libgbm1

# Verify the mesa driver is installed (usually already present)
sudo apt-get install -y libgl1-mesa-dri
```

### RHEL/CentOS/Fedora

```bash
# Install EGL libraries
sudo yum install -y mesa-libEGL mesa-libgbm

# Verify the mesa driver is installed
sudo yum install -y mesa-dri-drivers
```

### Alpine Linux (Docker)

```dockerfile
RUN apk add --no-cache \
    mesa-gl \
    mesa-egl \
    mesa-gbm \
    mesa-dri-gallium
```

## Verification

Test that headless OpenGL works:

```bash
python3 -c "
import moderngl
try:
    ctx = moderngl.create_standalone_context(backend='egl')
    print(f'✓ Success!')
    print(f'  Renderer: {ctx.info.get(\"GL_RENDERER\", \"Unknown\")}')
    print(f'  Vendor: {ctx.info.get(\"GL_VENDOR\", \"Unknown\")}')
    print(f'  OpenGL Version: {ctx.info.get(\"GL_VERSION\", \"Unknown\")}')
    ctx.release()
except Exception as e:
    print(f'✗ Failed: {e}')
"
```

Expected output:
```
✓ Success!
  Renderer: llvmpipe (LLVM 20.1.2, 256 bits)
  Vendor: Mesa
  OpenGL Version: 4.5 (Core Profile) Mesa 25.0.7
```

## How It Works

### Backend Selection

ModernGL's `create_standalone_context()` tries backends in this order:

1. **X11/GLX** - Requires X11 display (fails in headless environments)
2. **EGL** - Works headless with Mesa (requires libegl1)
3. **OSMesa** - Fallback software renderer (slow, limited features)

Our code automatically tries EGL first:

```python
try:
    ctx = moderngl.create_standalone_context(backend='egl')
except Exception:
    # Fallback to default backend (may require X11)
    ctx = moderngl.create_standalone_context()
```

### Mesa llvmpipe

llvmpipe is Mesa's high-performance software OpenGL renderer that uses LLVM for JIT compilation:

- Implements OpenGL 4.5+ Core Profile
- Runs on CPU (no GPU required)
- Uses SIMD instructions (SSE, AVX) for performance
- Multi-threaded rendering
- Slower than hardware GPU but much faster than old software renderers

## Performance Considerations

### Rendering Times (1920×1080, N=1597)

| Mode | Hardware GPU | llvmpipe (CPU) |
|------|-------------|----------------|
| CPU (default) | N/A | 10–60 sec |
| GPU (`--gpu`) | 1–30 sec | 60–300 sec |
| SDF (`--sdf`) | 5–20 sec | 80–400 sec |

**Recommendation**: For headless environments, use CPU mode (default) for best performance.

### Optimization Tips

1. **Use CPU mode** (don't use `--gpu` or `--sdf` flags in headless environments)
2. **Reduce resolution**: Use `--W 1280 --H 720` for faster renders
3. **Fewer points**: Use `--N 800` instead of default 1597
4. **Disable anti-aliasing**: Use `--aa 1` (default is 1, so no change needed)
5. **Parallel jobs**: llvmpipe uses multiple CPU cores automatically

## CI/CD Integration

### GitHub Actions

```yaml
name: Render Phyllotaxis

on: [push]

jobs:
  render:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libegl1 libgbm1 libgl1-mesa-dri
      
      - name: Install Python dependencies
        run: pip install -r requirements.txt
      
      - name: Render image
        run: |
          python torus_net_gpu.py --steps 13 21 --out render.png
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: renders
          path: png/*.png
```

### GitLab CI

```yaml
image: python:3.12

before_script:
  - apt-get update
  - apt-get install -y libegl1 libgbm1 libgl1-mesa-dri
  - pip install -r requirements.txt

render:
  stage: build
  script:
    - python torus_net_gpu.py --steps 13 21 --out render.png
  artifacts:
    paths:
      - png/
    expire_in: 1 week
```

### Docker

```dockerfile
FROM python:3.12-slim

# Install OpenGL dependencies for headless rendering
RUN apt-get update && apt-get install -y \
    libegl1 \
    libgbm1 \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run renderer
CMD ["python", "torus_net_gpu.py", "--steps", "13", "21", "--out", "render.png"]
```

Build and run:

```bash
docker build -t phyllotaxis .
docker run -v $(pwd)/output:/app/png phyllotaxis
```

## Troubleshooting

### Error: "libEGL.so not loaded"

**Cause**: EGL library is not installed.

**Solution**:
```bash
sudo apt-get install -y libegl1 libgbm1
```

### Error: "XOpenDisplay: cannot open display"

**Cause**: ModernGL is trying to use X11 backend instead of EGL.

**Solution**: Ensure EGL libraries are installed. The code now automatically tries EGL first.

### Error: "Failed to create OpenGL context"

**Possible causes**:

1. **Missing EGL**: Install `libegl1` and `libgbm1`
2. **Missing Mesa driver**: Install `libgl1-mesa-dri`
3. **Corrupted Mesa**: Reinstall Mesa packages

**Diagnostic steps**:

```bash
# Check if EGL library exists
ls -la /usr/lib/x86_64-linux-gnu/libEGL.so*

# Check if Mesa driver exists
ls -la /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so

# Check Mesa version
glxinfo | grep "OpenGL version" || echo "Install mesa-utils for glxinfo"

# Try loading EGL manually
python3 -c "from ctypes import CDLL; CDLL('libEGL.so.1')"
```

### Performance Issues

If rendering is very slow in headless mode:

1. **Don't use GPU mode**: Omit `--gpu` and `--sdf` flags
2. **Check CPU usage**: llvmpipe should use multiple cores
3. **Reduce workload**: Lower `--N`, `--W`, `--H` parameters
4. **Monitor resources**: Use `htop` or `top` to check CPU/memory usage

### Segmentation Fault

If you encounter segfaults:

1. **Update Mesa**: Ensure you have Mesa 20.0+
2. **Check llvmpipe**: Run `LIBGL_ALWAYS_SOFTWARE=1 glxinfo` (requires mesa-utils)
3. **Update LLVM**: Ensure LLVM version matches Mesa build

## Environment Variables

Useful environment variables for debugging and configuration:

```bash
# Force software rendering (useful for testing)
export LIBGL_ALWAYS_SOFTWARE=1

# Enable Mesa debug output
export MESA_DEBUG=1
export LIBGL_DEBUG=verbose

# Set number of threads for llvmpipe (default: auto)
export LP_NUM_THREADS=4

# Disable llvmpipe's JIT compiler (for debugging, very slow)
export LP_FORCE_SSE2=1
```

## Alternative: OSMesa (Not Recommended)

OSMesa (Off-Screen Mesa) is an older alternative to EGL:

```bash
# Install OSMesa
sudo apt-get install -y libosmesa6

# Use in Python
import moderngl
ctx = moderngl.create_standalone_context(backend='osmesa')
```

**Why EGL is better:**
- Faster (uses modern Mesa drivers)
- Better OpenGL support (4.5+ vs 2.1)
- Actively maintained
- Supports compute shaders

OSMesa is only recommended if EGL is unavailable on your system.

## Summary

For headless OpenGL rendering:

1. **Install**: `libegl1` and `libgbm1` packages
2. **Verify**: Mesa llvmpipe renderer loads correctly
3. **Use**: CPU mode (default) for best performance
4. **Test**: Run verification script to ensure setup works

The renderer will automatically use Mesa's llvmpipe software renderer when no GPU is available, enabling OpenGL rendering in any headless environment.
