#!/usr/bin/env python3
"""
Torus phyllotaxis net (GPU) with CORRECT OCCLUSION.

Fixes the "dots visible on back side of donut hole" problem by:
- Depth buffer stores t_hit (ray parameter), not z_cam
- Ray directions D[y,x] are precomputed
- Dots/line-samples are rendered as small 3D spheres using ray–sphere intersection per pixel

Deps:
  pip install numpy matplotlib
  pip install torch  (install CUDA build for GPU)

Examples:
  python torus_net_gpu.py --steps 13 21 --out out_13_21.png
  python torus_net_gpu.py --steps 34 55 --out out_34_55.png
  python torus_net_gpu.py --steps 1 13 21 --out spiral.png
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# -------------------------
# Math helpers
# -------------------------
def wrap_pi_torch(a: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return (a + math.pi) % two_pi - math.pi


def torus_sdf(P: torch.Tensor, R: float, r: float) -> torch.Tensor:
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    qx = torch.sqrt(x * x + y * y) - R
    return torch.sqrt(qx * qx + z * z) - r


def torus_point_from_uv(u: torch.Tensor, v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    cu, su = torch.cos(u), torch.sin(u)
    cv, sv = torch.cos(v), torch.sin(v)
    x = (R + r * cv) * cu
    y = (R + r * cv) * su
    z = r * sv
    return torch.stack([x, y, z], dim=-1)


def camera_basis_np(camera_xyz, target_xyz, up_xyz=(0.0, 0.0, 1.0)):
    C = np.array(camera_xyz, dtype=float)
    T = np.array(target_xyz, dtype=float)
    up = np.array(up_xyz, dtype=float)

    forward = T - C
    nrm = np.linalg.norm(forward)
    if nrm < 1e-12:
        raise ValueError("Camera and target coincide.")
    forward /= nrm

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
    else:
        right /= rn

    true_up = np.cross(right, forward)
    return right, true_up, forward


def pingpong01(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.abs(2.0 * t - 1.0)


def build_lut(device, cmap_name="gnuplot", n=256) -> torch.Tensor:
    cmap = plt.get_cmap(cmap_name)
    lut = np.asarray([cmap(i / (n - 1))[:3] for i in range(n)], dtype=np.float32)
    return torch.tensor(lut, device=device)


# -------------------------
# Scene
# -------------------------
@dataclass
class Scene:
    R: float = 3.0
    r_outer: float = 2.6
    r_inner: float = 2.3

    N: int = 1500

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


def build_camera(scene: Scene):
    u0 = scene.u0_cam
    e_r = np.array([math.cos(u0), math.sin(u0), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])

    centerline = scene.R * e_r
    rho = scene.r_inner - scene.eps_wall
    cam0 = centerline + rho * (math.cos(scene.v_cam) * e_r + math.sin(scene.v_cam) * e_z)

    target_old = np.array([0.0, 0.0, scene.r_outer], dtype=float)
    dir_to_old = target_old - cam0
    dir_to_old /= np.linalg.norm(dir_to_old)
    cam = cam0 + scene.t_step * dir_to_old

    right, up, forward = camera_basis_np(cam, scene.target)
    return cam, right, up, forward


def compute_dynamic_radii(v: torch.Tensor, R: float, r: float, base_radius: int, size_min: float, size_max: float) -> torch.Tensor:
    """
    Compute dynamic radii for phyllotaxis dots based on local spacing.

    In phyllotaxis on a torus, local density varies with v (minor circle angle):
    - Near v=0 (outer equator): points are more spread out → larger dots
    - Near v=±π (inner equator): points are compressed → smaller dots

    The local "circumference" at angle v is: 2π(R + r·cos(v))
    Larger circumference → more space → larger dots

    Args:
        v: Minor circle angles [N]
        R: Major radius (torus)
        r: Minor radius (tube)
        base_radius: Base pixel radius
        size_min: Minimum size multiplier
        size_max: Maximum size multiplier

    Returns:
        radii: Dynamic pixel radii [N]
    """
    # Local circumference factor: cos(v) ranges from -1 (inner) to +1 (outer)
    # R + r·cos(v) is the distance from torus center to the point's circular path
    circ_factor = R + r * torch.cos(v)

    # Normalize to [0, 1] where 0=inner equator, 1=outer equator
    circ_min = R - r  # Inner equator
    circ_max = R + r  # Outer equator
    normalized = (circ_factor - circ_min) / (circ_max - circ_min)

    # Map to size range [size_min, size_max]
    size_mult = size_min + (size_max - size_min) * normalized

    # Apply to base radius
    radii = base_radius * size_mult

    return radii


# -------------------------
# Phyllotaxis uv
# -------------------------
def torus_phyllotaxis_uv(N: int, R: float, r: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    phi = (1 + 5**0.5) / 2
    alpha = 2 * math.pi / (phi * phi)
    two_pi = 2.0 * math.pi

    idx = torch.arange(N, device=device, dtype=torch.float32)
    u = (idx * alpha) % two_pi

    s = (idx + 0.5) / N
    target = two_pi * R * s

    lo = torch.zeros((N,), device=device)
    hi = torch.full((N,), two_pi, device=device)

    for _ in range(70):
        mid = 0.5 * (lo + hi)
        gmid = R * mid + r * torch.sin(mid)
        lo = torch.where(gmid < target, mid, lo)
        hi = torch.where(gmid < target, hi, mid)

    v = 0.5 * (lo + hi)
    return u, v


# -------------------------
# Rays + depth (t_hit) on GPU
# -------------------------
def build_rays(scene: Scene, device, right, up, forward) -> torch.Tensor:
    """
    Returns D[y,x,3] normalized ray directions.
    """
    W, H = scene.W, scene.H
    aspect = W / H

    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2.0

    # [H,W]
    y_img, x_img = torch.meshgrid(ys, xs, indexing="ij")
    x_img = x_img * aspect

    dx = x_img / scene.f
    dy = y_img / scene.f
    dz = torch.ones_like(dx)

    D = (
        dx[..., None] * right[None, None, :]
        + dy[..., None] * up[None, None, :]
        + dz[..., None] * forward[None, None, :]
    )
    D = D / torch.linalg.norm(D, dim=-1, keepdim=True)
    return D  # [H,W,3]


def render_depth_t(scene: Scene, device, C, D) -> torch.Tensor:
    """
    Sphere-traces with abs(SDF) and records t_hit (ray parameter) at first hit.
    depth_t[y,x] = t_hit, inf if miss.
    """
    H, W = scene.H, scene.W
    Df = D.reshape(-1, 3)
    Pcount = Df.shape[0]

    depth_t = torch.full((Pcount,), float("inf"), device=device)
    t = torch.zeros((Pcount,), device=device)
    alive = torch.ones((Pcount,), device=device, dtype=torch.bool)

    for _ in range(scene.max_steps):
        if not alive.any():
            break

        P = C[None, :] + Df * t[:, None]
        dist = torch.abs(torus_sdf(P, scene.R, scene.r_inner))

        hit = alive & (dist < scene.hit_eps)
        if hit.any():
            depth_t[hit] = t[hit]
            alive[hit] = False

        t_next = t + dist
        alive = alive & (t_next < scene.t_max) & torch.isfinite(t_next)
        t = torch.where(alive, t_next, t)

    return depth_t.reshape(H, W)


# -------------------------
# Projection helpers
# -------------------------
def project_points(P: torch.Tensor, C, right, up, forward, f: float, W: int, H: int):
    """
    Returns valid mask, z_cam, px, py.
    """
    aspect = W / H
    V = P - C[None, :]
    x_cam = torch.matmul(V, right)
    y_cam = torch.matmul(V, up)
    z_cam = torch.matmul(V, forward)

    valid = z_cam > 1e-6

    x_img = f * x_cam / z_cam
    y_img = f * y_cam / z_cam

    px = torch.round(((x_img / aspect) + 1.0) * 0.5 * (W - 1)).to(torch.int64)
    py = torch.round(((1.0 - y_img) * 0.5) * (H - 1)).to(torch.int64)

    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    valid = valid & inb
    return valid, z_cam, px, py


def disk_offsets(radius_px: int):
    r2 = radius_px * radius_px
    return [(dx, dy)
            for dy in range(-radius_px, radius_px + 1)
            for dx in range(-radius_px, radius_px + 1)
            if dx * dx + dy * dy <= r2]


# -------------------------
# Correct occluded splatting: ray–sphere per pixel
# -------------------------
def phong_shade(
    base_color: torch.Tensor,     # [K,3]
    normal: torch.Tensor,          # [K,3]
    view_dir: torch.Tensor,        # [K,3]
    light_dir: torch.Tensor,       # [3]
    ambient: float,
    diffuse_strength: float,
    specular_strength: float,
    shininess: float,
) -> torch.Tensor:
    """
    Compute Phong shading: ambient + diffuse + specular.
    All inputs normalized.
    Returns shaded color [K,3].
    """
    # Normalize inputs
    light_dir = light_dir / (torch.linalg.norm(light_dir) + 1e-6)

    # Ambient
    ambient_col = ambient * base_color

    # Diffuse
    diff = torch.clamp(torch.sum(normal * light_dir[None, :], dim=-1, keepdim=True), 0.0, 1.0)
    diffuse_col = diffuse_strength * diff * base_color

    # Specular (Blinn-Phong variant: half-vector)
    half_vec = (view_dir + light_dir[None, :]) / (torch.linalg.norm(view_dir + light_dir[None, :], dim=-1, keepdim=True) + 1e-6)
    spec = torch.clamp(torch.sum(normal * half_vec, dim=-1, keepdim=True), 0.0, 1.0)
    spec_pow = torch.pow(spec, shininess)
    specular_col = specular_strength * spec_pow * torch.ones_like(base_color)

    return ambient_col + diffuse_col + specular_col


def splat_spheres(
    img: torch.Tensor,
    depth_t: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    centers: torch.Tensor,    # [M,3]
    colors: torch.Tensor,     # [M,3]
    radius_px: int | torch.Tensor,  # int for uniform, Tensor[M] for per-point
    scene: Scene,
    right, up, forward,
    alpha: float | None = None,   # None => overwrite
    use_shading: bool = True,      # Apply Phong shading?
    sphere_depth: torch.Tensor | None = None,  # Shared depth buffer for sphere-to-sphere occlusion
):
    """
    Render each center as a small sphere, per covered pixel:
      - gather ray direction D[y,x]
      - solve ray-sphere intersection
      - compare t_hit_sphere to depth_t[y,x] (torus)
      - if use_shading: compute per-pixel normal and apply Phong shading
      - radius_px can be a scalar (uniform) or Tensor (per-point dynamic sizing)
      - sphere_depth: optional shared depth buffer for proper occlusion between multiple splat calls
    """
    H, W = scene.H, scene.W
    
    # Sphere depth buffer to ensure proper occlusion between spheres
    if sphere_depth is None:
        sphere_depth = torch.full((H, W), float("inf"), device=img.device, dtype=torch.float32)
    
    valid, z_cam, px, py = project_points(centers, C, right, up, forward, scene.f, W, H)

    centers = centers[valid]
    colors = colors[valid]
    z_cam = z_cam[valid]
    px = px[valid]
    py = py[valid]

    # Handle dynamic radii
    if isinstance(radius_px, torch.Tensor):
        radius_px = radius_px[valid]  # Filter per-point radii

    if centers.numel() == 0:
        return

    # Convert requested pixel radius to an approximate world radius *per point* (keeps apparent size stable):
    # r_px ≈ r_world * f/(z_cam*aspect)*(W-1)/2  => r_world ≈ r_px * z_cam * aspect * 2 / (f*(W-1))
    aspect = W / H

    # Handle both uniform and per-point radii
    if isinstance(radius_px, torch.Tensor):
        r_world = (radius_px * z_cam * aspect * 2.0) / (scene.f * (W - 1))
        max_radius_px = int(torch.ceil(torch.max(radius_px)).item())
    else:
        r_world = (radius_px * z_cam * aspect * 2.0) / (scene.f * (W - 1))
        max_radius_px = radius_px

    r2 = r_world * r_world

    offsets = disk_offsets(max_radius_px)

    # Prepare lighting direction for shading
    light_dir_np = np.array(scene.light_dir, dtype=np.float32)
    light_dir = torch.tensor(light_dir_np, device=D.device, dtype=torch.float32)

    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not inb.any():
            continue

        x_inb = x[inb]
        y_inb = y[inb]
        P0_inb = centers[inb]
        col_inb = colors[inb]
        rr2_inb = r2[inb]

        # ray data for these pixels
        Di_inb = D[y_inb, x_inb, :]  # [K,3]

        # ray-sphere: |(C + tD) - P0|^2 = r^2
        L = P0_inb - C[None, :]               # [K,3]
        b = torch.sum(Di_inb * L, dim=-1)     # [K]
        c = torch.sum(L * L, dim=-1) - rr2_inb
        disc = b * b - c

        ok = disc > 0.0
        if not ok.any():
            continue

        x_ok = x_inb[ok]
        y_ok = y_inb[ok]
        col_ok = col_inb[ok]
        b_ok = b[ok]
        disc_ok = disc[ok]
        Di_ok = Di_inb[ok]
        P0_ok = P0_inb[ok]

        t_sphere = b_ok - torch.sqrt(disc_ok)
        ok2 = t_sphere > 0.0
        if not ok2.any():
            continue

        x = x_ok[ok2]
        y = y_ok[ok2]
        col = col_ok[ok2]
        t_sphere = t_sphere[ok2]
        Di = Di_ok[ok2]
        P0 = P0_ok[ok2]

        # occlusion vs torus
        dt = depth_t[y, x]
        vis = t_sphere <= (dt + scene.eps_t)

        if not vis.any():
            continue

        x = x[vis]
        y = y[vis]
        col = col[vis]
        t_sphere = t_sphere[vis]
        Di = Di[vis]
        P0 = P0[vis]

        # Check sphere-to-sphere occlusion: only render if closer than existing spheres
        current_sphere_depth = sphere_depth[y, x]
        closer = t_sphere < current_sphere_depth
        
        if not closer.any():
            continue
            
        x = x[closer]
        y = y[closer]
        col = col[closer]
        t_sphere = t_sphere[closer]
        Di = Di[closer]
        P0 = P0[closer]

        # Apply Phong shading if requested
        if use_shading:
            # Compute 3D hit point on sphere surface
            hit_point = C[None, :] + Di * t_sphere[:, None]  # [K,3]
            
            # Normal is radial direction from center
            normal = (hit_point - P0) / (torch.linalg.norm(hit_point - P0, dim=-1, keepdim=True) + 1e-6)  # [K,3]
            
            # View direction (from hit point toward camera)
            view_dir = (C[None, :] - hit_point) / (torch.linalg.norm(C[None, :] - hit_point, dim=-1, keepdim=True) + 1e-6)  # [K,3]
            
            # Apply Phong shading
            col = phong_shade(
                base_color=col,
                normal=normal,
                view_dir=view_dir,
                light_dir=light_dir,
                ambient=scene.ambient,
                diffuse_strength=scene.diffuse_strength,
                specular_strength=scene.specular_strength,
                shininess=scene.shininess,
            )

        # Update sphere depth buffer and render
        sphere_depth[y, x] = torch.minimum(sphere_depth[y, x], t_sphere)
        
        if alpha is None:
            img[y, x, :] = col
        else:
            img[y, x, :] = (1.0 - alpha) * img[y, x, :] + alpha * col
    
    return sphere_depth


# -------------------------
# Render
# -------------------------
def render(scene: Scene, steps: list[int], out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} (cuda_available={torch.cuda.is_available()})")

    # Organize outputs by file type
    out_path = Path(out_path)
    png_dir = Path("png")
    json_dir = Path("json")
    png_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    png_path = png_dir / out_path.name
    json_path = json_dir / out_path.with_suffix(".json").name

    cam_np, right_np, up_np, forward_np = build_camera(scene)

    C = torch.tensor(cam_np, device=device, dtype=torch.float32)
    right = torch.tensor(right_np, device=device, dtype=torch.float32)
    up = torch.tensor(up_np, device=device, dtype=torch.float32)
    forward = torch.tensor(forward_np, device=device, dtype=torch.float32)

    # Apply supersampling for anti-aliasing
    W_render = scene.W * scene.aa_factor
    H_render = scene.H * scene.aa_factor
    
    # Temporarily modify scene dimensions for rendering
    original_W, original_H = scene.W, scene.H
    scene.W, scene.H = W_render, H_render

    # Rays + depth (t_hit)
    D = build_rays(scene, device, right, up, forward)              # [H,W,3]
    depth_t = render_depth_t(scene, device, C, D).to(torch.float32)  # [H,W]

    # Base: white background, black where torus is visible
    img = torch.ones((scene.H, scene.W, 3), device=device, dtype=torch.float32)
    img[torch.isfinite(depth_t), :] = 0.0

    # Points
    u, v = torus_phyllotaxis_uv(scene.N, scene.R, scene.r_inner, device=device)
    P = torus_point_from_uv(u, v, scene.R, scene.r_inner)

    # Colors (pingpong gnuplot, seam-free)
    lut = build_lut(device, scene.cmap_name, 256)
    t_raw = (v % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp = pingpong01(t_raw)
    cidx = torch.clamp((t_pp * 255.0).to(torch.int64), 0, 255)
    dot_cols = lut[cidx]  # [N,3]

    # Compute dynamic dot radii based on local density (v-coordinate)
    if scene.dot_radius_dynamic:
        dot_radii = compute_dynamic_radii(
            v, scene.R, scene.r_inner, 
            scene.dot_radius_px, 
            scene.dot_size_min, 
            scene.dot_size_max
        )
    else:
        dot_radii = scene.dot_radius_px  # Uniform radius

    # Edges from step sizes
    edges = []
    N = scene.N
    for step in steps:
        for i in range(N):
            if i + step < N:
                edges.append((i, i + step))

    # Build line samples on surface (E * S points)
    samples = scene.line_samples_per_edge + 1
    A = torch.linspace(0.0, 1.0, samples, device=device, dtype=torch.float32)  # [S]

    i0 = torch.tensor([e[0] for e in edges], device=device, dtype=torch.int64)
    i1 = torch.tensor([e[1] for e in edges], device=device, dtype=torch.int64)

    u0 = u[i0]; v0 = v[i0]
    u1 = u[i1]; v1 = v[i1]

    du = wrap_pi_torch(u1 - u0)
    dv = wrap_pi_torch(v1 - v0)

    uu = u0[:, None] + du[:, None] * A[None, :]
    vv = v0[:, None] + dv[:, None] * A[None, :]

    line_pts = torus_point_from_uv(uu.reshape(-1), vv.reshape(-1), scene.R, scene.r_inner)

    # Color per edge from midpoint-v (then broadcast to samples)
    v_mid = v0 + 0.5 * dv
    t_raw_e = (v_mid % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp_e = pingpong01(t_raw_e)
    eidx = torch.clamp((t_pp_e * 255.0).to(torch.int64), 0, 255)
    edge_cols = lut[eidx]  # [E,3]
    line_cols = edge_cols[:, None, :].expand(-1, samples, -1).reshape(-1, 3)

    # Compute dynamic line radii (INVERSE of dot sizing - thick where dots are small)
    if scene.line_radius_dynamic:
        # Compute per-edge radii at midpoint
        edge_radii = compute_dynamic_radii(
            v_mid, scene.R, scene.r_inner,
            scene.line_radius_px,
            scene.line_size_min,
            scene.line_size_max
        )
        # INVERT: max where dots are min, min where dots are max
        # Map [min, max] → [max, min]
        inverted_radii = scene.line_size_max + scene.line_size_min - edge_radii
        # Broadcast to all samples along each edge [E,S] → [E*S]
        line_radii = inverted_radii[:, None].expand(-1, samples).reshape(-1)
    else:
        line_radii = scene.line_radius_px  # Uniform radius

    # Render lines first (alpha blend, light shading, with dynamic radii)
    sphere_depth = splat_spheres(
        img=img,
        depth_t=depth_t,
        D=D,
        C=C,
        centers=line_pts,
        colors=line_cols,
        radius_px=line_radii,
        scene=scene,
        right=right,
        up=up,
        forward=forward,
        alpha=scene.line_alpha,
        use_shading=True,
    )

    # Render dots on top (overwrite, full shading, with dynamic radii)
    # Pass the same sphere_depth buffer so dots properly occlude with lines
    splat_spheres(
        img=img,
        depth_t=depth_t,
        D=D,
        C=C,
        centers=P,
        colors=dot_cols,
        radius_px=dot_radii,  # Use dynamic radii
        scene=scene,
        right=right,
        up=up,
        forward=forward,
        alpha=None,
        use_shading=True,
        sphere_depth=sphere_depth,  # Share depth buffer!
    )

    # Restore original dimensions
    scene.W, scene.H = original_W, original_H

    # Downsample if anti-aliasing was used
    if scene.aa_factor > 1:
        # Reshape to [H_out, aa_factor, W_out, aa_factor, 3]
        img = img.reshape(original_H, scene.aa_factor, original_W, scene.aa_factor, 3)
        # Average over the aa_factor dimensions
        img = img.mean(dim=(1, 3))  # Result: [H_out, W_out, 3]

    img_cpu = img.clamp(0.0, 1.0).detach().cpu().numpy()
    plt.imsave(png_path, img_cpu)
    print(f"[saved] {png_path}")

    # Save scene metadata as JSON
    metadata = {
        "steps": steps,
        "scene": asdict(scene),
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {json_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="*", default=[], help="List of step sizes (e.g., --steps 1 13 21). If omitted, only dots are rendered.")
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--f", type=float, default=1.2)
    ap.add_argument("--t_step", type=float, default=0.40)
    
    ap.add_argument("--aa", type=int, default=1, help="Anti-aliasing factor: 1=off, 2=2x2 SSAA (4x pixels), 3=3x3 SSAA (9x pixels)")

    ap.add_argument("--cmap", type=str, default="gnuplot", help="Matplotlib colormap name (e.g., magma, inferno, viridis, plasma, gnuplot)")

    ap.add_argument("--dot_px", type=int, default=4)
    ap.add_argument("--dot_dynamic", action="store_true", default=True, help="Use dynamic dot sizing based on local density (default: True)")
    ap.add_argument("--no_dot_dynamic", action="store_false", dest="dot_dynamic", help="Disable dynamic dot sizing")
    ap.add_argument("--dot_size_min", type=float, default=0.5, help="Minimum size multiplier for dynamic dots (default: 0.5)")
    ap.add_argument("--dot_size_max", type=float, default=2.5, help="Maximum size multiplier for dynamic dots (default: 2.5)")

    ap.add_argument("--line_px", type=int, default=1)
    ap.add_argument("--line_dynamic", action="store_true", default=True, help="Use dynamic line thickness (inverse of dots) (default: True)")
    ap.add_argument("--no_line_dynamic", action="store_false", dest="line_dynamic", help="Disable dynamic line thickness")
    ap.add_argument("--line_size_min", type=float, default=0.5, help="Minimum size multiplier for dynamic lines (default: 0.5)")
    ap.add_argument("--line_size_max", type=float, default=2.5, help="Maximum size multiplier for dynamic lines (default: 2.5)")
    ap.add_argument("--line_alpha", type=float, default=0.40)
    ap.add_argument("--line_samples", type=int, default=44)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scene = Scene(
        N=args.N,
        W=args.W,
        H=args.H,
        f=args.f,
        t_step=args.t_step,
        aa_factor=args.aa,
        cmap_name=args.cmap,
        dot_radius_px=args.dot_px,
        dot_radius_dynamic=args.dot_dynamic,
        dot_size_min=args.dot_size_min,
        dot_size_max=args.dot_size_max,
        line_radius_px=args.line_px,
        line_radius_dynamic=args.line_dynamic,
        line_size_min=args.line_size_min,
        line_size_max=args.line_size_max,
        line_alpha=args.line_alpha,
        line_samples_per_edge=args.line_samples,
    )

    render(
        scene=scene,
        steps=args.steps,
        out_path=args.out,
    )
