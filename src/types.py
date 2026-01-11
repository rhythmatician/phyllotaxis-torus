import math

from dataclasses import dataclass
# -------------------------
# Scene
# -------------------------
@dataclass
class Scene:
    R: float = 3.0
    r_outer: float = 2.6
    r_inner: float = 2.3

    N: int = 1597  # Fibonacci number for seamless phyllotaxis pattern

    W: int = 1920
    H: int = 1080
    f: float = 1.2
    t_step: float = 0.40
    target: tuple = (0.0, 0.0, 1.0)
    
    # anti-aliasing
    aa_factor: int = 1  # 1=no AA, 2=2x2 SSAA (4x pixels), 3=3x3 SSAA (9x pixels)

    # inside-tube camera recipe
    u0_cam: float = -math.pi / 4
    v_cam: float = -0.28
    eps_wall: float = 0.05

    # raymarch
    hit_eps: float = 1.2e-3
    t_max: float = 160.0
    max_steps: int = 150

    # occlusion tolerance in t-space
    eps_t: float = 0.01

    # style
    cmap_name: str = "gnuplot"

    dot_radius_px: int = 4
    dot_radius_dynamic: bool = True     # Use dynamic sizing based on local density
    dot_size_min: float = 0.5           # Minimum size multiplier for dynamic dots
    dot_size_max: float = 2.5           # Maximum size multiplier for dynamic dots

    line_radius_px: int = 1
    line_radius_dynamic: bool = True    # Use dynamic line thickness (inverse of dots)
    line_size_min: float = 0.5          # Minimum size multiplier for dynamic lines
    line_size_max: float = 2.5          # Maximum size multiplier for dynamic lines
    line_alpha: float = 0.40
    line_samples_per_edge: int = 44

    # lighting for sphere shading
    light_dir: tuple = (-0.5, 0.3, 1.0)  # directional light direction (will be normalized)
    ambient: float = 0.3
    diffuse_strength: float = 0.6
    specular_strength: float = 0.5 
    shininess: float = 32.0
    
    # smooth blending (SDF operations)
    enable_smooth_blend: bool = False    # Enable smooth min blending between dots
    smooth_k: float = 0.3                # Smoothing factor (higher = more blending)
    blend_radius_multiplier: float = 1.5 # How far to consider neighbors for blending
