# Copilot Instructions for Torus Phyllotaxis Project

## Environment Setup

### PowerShell Activation
This project uses **PowerShell** (not cmd or bash). Always activate the virtual environment first:

```powershell
& C:\Users\rhyth\git\torus\.venv\Scripts\Activate.ps1
```

After activation, you can run Python scripts directly:
```powershell
python torus_net_gpu.py --steps 13 21 --out test.png
python -m pip install package_name
```

**DO NOT** assume bash/Linux commands. Use PowerShell equivalents:
- `Get-ChildItem` instead of `ls`
- `Remove-Item` instead of `rm`
- `Copy-Item` instead of `cp`
- `Move-Item` instead of `mv`

## Project Structure

### Root Level (Keep Minimal)
Only essential files in root:
- `torus_net_gpu.py` - Main renderer script
- `README.md` - Entry point for users
- `requirements.txt` - Python dependencies
- `.venv/` - Virtual environment (don't commit)
- `.git/`, `.gitignore` - Version control

### Documentation Organization
**ALL documentation files go in `docs/` folder**, NOT root:
- `docs/SHADING_INTEGRATION.md` - Technical Phong shading details
- `docs/SHADING_GUIDE.md` - Visual guide & parameter tuning
- `docs/PROJECT_STATUS.md` - Complete feature list
- `docs/INTEGRATION_SUMMARY.md` - Integration summary
- `docs/COMPLETION_CHECKLIST.md` - Status checklist
- `docs/QUICK_REFERENCE.py` - Command-line examples
- `docs/BUG_FIX.md` - Bug fix documentation

**Never add `.md` files to root** — they clutter the workspace. Put them in `docs/`.

### Testing Organization
**IF tests are needed in future:**
- Create a `tests/` folder at root level
- Use `pytest` framework (not custom scripts)
- Only add if testing is actually necessary
- Currently: NO tests needed (renderer is stable)

**DO NOT create ad-hoc test scripts** like `test_shading.py` in root. Either use proper pytest in `tests/` or don't test at all.

### Output Folders (Auto-Created)
These are created automatically when rendering:
- `png/` - Rendered images (gitignored)
- `json/` - Scene metadata (gitignored)

## Code Modifications

### Main Renderer (`torus_net_gpu.py`)
Key functions:
- `phong_shade()` - Phong lighting calculation
- `splat_spheres()` - Sphere rendering with shading (fixed tensor indexing)
- `render()` - Main rendering pipeline
- `Scene` dataclass - Scene parameters including lighting

**Recent fix**: Tensor indexing in `splat_spheres()` uses explicit variable naming:
- `_inb` suffix for boundary-filtered tensors
- `_ok` suffix for ray-sphere filtered tensors
- Final names for occlusion-filtered tensors

### README.md
- Main entry point for users
- Include features, installation, usage, and file structure
- Link to detailed docs in `docs/` folder
- Keep reasonably concise (~200 lines)

## Common Tasks

### Running a Render
```powershell
& C:\Users\rhyth\git\torus\.venv\Scripts\Activate.ps1
python torus_net_gpu.py --steps 13 21 --out spiral.png
# Output: png/spiral.png, json/spiral.json
```

### Adding Documentation
1. Create file in `docs/` folder (not root)
2. Use `.md` extension for markdown
3. Link from `README.md` if it's important
4. Reference in this file if it affects future development

### Testing Code Changes
1. Verify syntax: `python -m py_compile torus_net_gpu.py`
2. Test render: `python torus_net_gpu.py --steps 1 --W 960 --H 540 --out test.png`
3. Check output: `Get-ChildItem png/, json/`

### Installing Dependencies
```powershell
& C:\Users\rhyth\git\torus\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU support:
> TBD: CUDA unavailable with Intel(R) HD Graphics 630

## File Naming Conventions

### Documentation Files
- Use SCREAMING_SNAKE_CASE for important docs: `SHADING_GUIDE.md`
- Lowercase for minor docs or code examples: `quick_reference.py`

### Output Files
- User can specify any name via `--out` flag
- Images go to `png/` with given name
- Metadata goes to `json/` with same base name

## Important Notes

### Terminal Context
- Always assume working directory is `C:\Users\rhyth\git\torus`
- Always activate venv first with the PowerShell command above
- Use PowerShell syntax for all commands

### Do NOT
- Create test files in root (`test_*.py`)
- Create `.md` files in root (use `docs/`)
- Use bash/Linux commands (use PowerShell)
- Forget to activate venv (leads to path issues)

### DO
- Activate venv at start of each terminal session
- Put all docs in `docs/` folder
- Use pytest if tests are ever needed (in `tests/` folder)
- Run `python torus_net_gpu.py` directly (after venv activation)
- Keep root level minimal and clean

## Feature Overview

### Current Capabilities
✅ GPU-accelerated torus phyllotaxis rendering
✅ Flexible edge networks (any step sizes)
✅ Phong-shaded 3D spheres (ambient + diffuse + specular)
✅ Multiple colormaps (magma, inferno, viridis, plasma, gnuplot)
✅ Automatic output organization (png/ and json/ folders)
✅ JSON metadata export
✅ CPU fallback support
✅ Correct occlusion (per-pixel ray-sphere intersection)

### Known Limitations
- Single directional light source (can adjust direction)
- Fixed camera position (can edit `u0_cam`, `v_cam`, `eps_wall` in Scene)
- No per-edge material properties
- No CLI args for lighting (edit Scene in code)

## Quick Reference Commands

```powershell
# Activate venv (DO THIS FIRST)
& C:\Users\rhyth\git\torus\.venv\Scripts\Activate.ps1

# Basic render
python torus_net_gpu.py --steps 13 21 --out test.png

# Nearest neighbors
python torus_net_gpu.py --steps 1 --out neighbors.png

# Multi-scale
python torus_net_gpu.py --steps 1 13 21 --out multi.png

# Different colormap
python torus_net_gpu.py --steps 13 21 --cmap inferno --out inferno.png

# High resolution
python torus_net_gpu.py --steps 13 21 --W 3840 --H 2160 --out 4k.png

# Check syntax
python -m py_compile torus_net_gpu.py

# Show docs folder
Get-ChildItem docs/

# Show output
Get-ChildItem png/, json/
```

## When to Reference This File

Read this if:
- You're about to create a new file (ask: docs/ or tests/? or root?)
- You're about to run terminal commands (use PowerShell syntax)
- You're starting a new terminal session (activate venv first)
- You're unsure about project structure or conventions

---

**Last Updated**: Current session
**Status**: Production-ready, stable, well-organized
