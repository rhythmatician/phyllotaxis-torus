#!/usr/bin/env python3
"""
Torus phyllotaxis net (GPU) with CORRECT OCCLUSION.

Fixes the "dots visible on back side of donut hole" problem by:
- Depth buffer stores t_hit (ray parameter), not z_cam
- Ray directions D[y,x] are precomputed
- Dots/line-samples are rendered as small 3D spheres using ray–sphere intersection per pixel

Deps:
  pip install numpy matplotlib
  pip install torch  (Must be `Intel(R) HD Graphics 630` compatible)

Examples:
  python torus_net_gpu.py --steps 13 21 --out out_13_21.png
  python torus_net_gpu.py --steps 34 55 --out out_34_55.png
  python torus_net_gpu.py --steps 1 13 21 --out spiral.png
  python torus_net_gpu.py --sdf --out out_sdf.png
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.types import Scene
from src.shade import phong_shade


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


def hollow_torus_sdf(
    P: torch.Tensor, R: float, r_outer: float, thickness: float = 0.115
) -> torch.Tensor:
    """
    Hollow torus (shell with thickness).
    Returns negative inside the hollow interior, positive outside the shell.
    """
    torus_outer = torus_sdf(P, R, r_outer)
    torus_inner = torus_sdf(P, R, r_outer - thickness)
    # Shell is the region between outer and inner surfaces
    return torch.maximum(torus_outer, -torus_inner)


def torus_point_from_uv(
    u: torch.Tensor, v: torch.Tensor, R: float, r: float
) -> torch.Tensor:
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


def build_camera(scene: Scene):
    u0 = scene.u0_cam
    e_r = np.array([math.cos(u0), math.sin(u0), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])

    centerline = scene.R * e_r
    rho = scene.r_inner - scene.eps_wall
    cam0 = centerline + rho * (
        math.cos(scene.v_cam) * e_r + math.sin(scene.v_cam) * e_z
    )

    target_old = np.array([0.0, 0.0, scene.r_outer], dtype=float)
    dir_to_old = target_old - cam0
    dir_to_old /= np.linalg.norm(dir_to_old)
    cam = cam0 + scene.t_step * dir_to_old

    right, up, forward = camera_basis_np(cam, scene.target)
    return cam, right, up, forward


def compute_dynamic_radii(
    v: torch.Tensor,
    R: float,
    r: float,
    base_radius: int,
    size_min: float,
    size_max: float,
) -> torch.Tensor:
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
def torus_phyllotaxis_uv(
    N: int, R: float, r: float, device
) -> tuple[torch.Tensor, torch.Tensor]:
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
    Uses (W-1) and (H-1) convention to match project_points pixel grid.
    """
    W, H = scene.W, scene.H
    aspect = W / H

    # Map pixel centers to [-1, 1] using (W-1) convention to match project_points
    xs = (
        torch.arange(W, device=device, dtype=torch.float32) / (W - 1) * 2.0 - 1.0
    ) * aspect
    ys = 1.0 - (torch.arange(H, device=device, dtype=torch.float32) / (H - 1) * 2.0)

    # [H,W]
    y_img, x_img = torch.meshgrid(ys, xs, indexing="ij")

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
    return [
        (dx, dy)
        for dy in range(-radius_px, radius_px + 1)
        for dx in range(-radius_px, radius_px + 1)
        if dx * dx + dy * dy <= r2
    ]


def splat_spheres(
    img: torch.Tensor,
    depth_t: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    centers: torch.Tensor,  # [M,3]
    colors: torch.Tensor,  # [M,3]
    radius_px: int | torch.Tensor,  # int for uniform, Tensor[M] for per-point
    scene: Scene,
    right,
    up,
    forward,
    alpha: float | None = None,  # None => overwrite
    use_shading: bool = True,  # Apply Phong shading?
    sphere_depth: (
        torch.Tensor | None
    ) = None,  # Shared depth buffer for sphere-to-sphere occlusion
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
        sphere_depth = torch.full(
            (H, W), float("inf"), device=img.device, dtype=torch.float32
        )

    valid, z_cam, px, py = project_points(centers, C, right, up, forward, scene.f, W, H)

    if centers.numel() > 0:
        print(
            f"[CPU splat] Processing {len(centers)} centers, {valid.sum().item()} valid after projection"
        )

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
        L = P0_inb - C[None, :]  # [K,3]
        b = torch.sum(Di_inb * L, dim=-1)  # [K]
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
            normal = (hit_point - P0) / (
                torch.linalg.norm(hit_point - P0, dim=-1, keepdim=True) + 1e-6
            )  # [K,3]

            # View direction (from hit point toward camera)
            view_dir = (C[None, :] - hit_point) / (
                torch.linalg.norm(C[None, :] - hit_point, dim=-1, keepdim=True) + 1e-6
            )  # [K,3]

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
# CPU Renderer
# -------------------------
def render_cpu(scene: Scene, steps: list[int], out_path: str):
    device = torch.device("cpu")
    print(f"[device] {device}")

    # Organize outputs by file type
    out_path_p = Path(out_path)
    png_dir = Path("png")
    json_dir = Path("json")
    png_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    png_path = png_dir / out_path_p.name
    json_path = json_dir / out_path_p.with_suffix(".json").name

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
    D = build_rays(scene, device, right, up, forward)  # [H,W,3]
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
            v,
            scene.R,
            scene.r_inner,
            scene.dot_radius_px,
            scene.dot_size_min,
            scene.dot_size_max,
        )
    else:
        dot_radii = torch.full(
            (scene.N,), scene.dot_radius_px, device=device, dtype=torch.float32
        )  # Uniform radius

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

    u0 = u[i0]
    v0 = v[i0]
    u1 = u[i1]
    v1 = v[i1]

    du = wrap_pi_torch(u1 - u0)
    dv = wrap_pi_torch(v1 - v0)

    uu = u0[:, None] + du[:, None] * A[None, :]
    vv = v0[:, None] + dv[:, None] * A[None, :]

    line_pts = torus_point_from_uv(
        uu.reshape(-1), vv.reshape(-1), scene.R, scene.r_inner
    )

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
            v_mid,
            scene.R,
            scene.r_inner,
            scene.line_radius_px,
            scene.line_size_min,
            scene.line_size_max,
        )
        # INVERT: max where dots are min, min where dots are max
        # Map [min, max] → [max, min]
        inverted_radii = scene.line_size_max + scene.line_size_min - edge_radii
        # Broadcast to all samples along each edge [E,S] → [E*S]
        line_radii = inverted_radii[:, None].expand(-1, samples).reshape(-1)
    else:
        line_radii = torch.full(
            (samples * len(edges),),
            scene.line_radius_px,
            device=device,
            dtype=torch.float32,
        )  # Uniform radius

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


def render_sdf_gpu(scene: Scene, steps: list[int], out_path: str):
    """
    SDF-based GPU rendering using OpenGL compute shaders.
    Uses proper SDF primitives (capsules, torus shell) instead of sampled spheres.
    """
    try:
        from src.sdf_gpu_renderer import SDFGPURenderer
    except ImportError as e:
        print(f"[error] Failed to import SDF GPU renderer: {e}")
        print("[fallback] Using CPU renderer instead")
        render_cpu(scene, steps, out_path)
        return

    print("[SDF-GPU] Initializing OpenGL SDF renderer...")

    gpu_renderer = None  # Initialize to avoid NameError in exception handlers
    try:
        # Create SDF GPU renderer
        gpu_renderer = SDFGPURenderer(scene.W, scene.H)

        # Print GPU info
        info = gpu_renderer.gpu_info
        print(f"[SDF-GPU] Vendor: {info['vendor']}")
        print(f"[SDF-GPU] Renderer: {info['renderer']}")
        print(f"[SDF-GPU] OpenGL Version: {info['version']}")
        print(f"[SDF-GPU] GLSL Version: {info['glsl_version']}")

    except Exception as e:
        print(f"[error] Failed to initialize SDF GPU renderer: {e}")
        print("[fallback] Using CPU renderer instead")
        render_cpu(scene, steps, out_path)
        return

    # Organize outputs by file type
    out_path_p = Path(out_path)
    png_dir = Path("png")
    json_dir = Path("json")
    png_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    png_path = png_dir / out_path_p.name
    json_path = json_dir / out_path_p.with_suffix(".json").name

    # Build camera
    cam_np, right_np, up_np, forward_np = build_camera(scene)

    # Compute CPU depth buffer for GPU to use (ensures perfect occlusion parity)
    device = torch.device("cpu")
    C = torch.tensor(cam_np, device=device, dtype=torch.float32)
    right_t = torch.tensor(right_np, device=device, dtype=torch.float32)
    up_t = torch.tensor(up_np, device=device, dtype=torch.float32)
    fwd_t = torch.tensor(forward_np, device=device, dtype=torch.float32)
    D_cpu = build_rays(scene, device, right_t, up_t, fwd_t)
    depth_t_cpu = render_depth_t(scene, device, C, D_cpu)  # [H, W]
    depth_t_np = depth_t_cpu.numpy().astype(np.float32)

    # Generate phyllotaxis points on CPU
    u, v = torus_phyllotaxis_uv(scene.N, scene.R, scene.r_inner, device)
    P = torus_point_from_uv(u, v, scene.R, scene.r_inner)

    # Colors (pingpong colormap, seam-free)
    lut = build_lut(device, scene.cmap_name, 256)
    t_raw = (v % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp = pingpong01(t_raw)
    cidx = torch.clamp((t_pp * 255.0).to(torch.int64), 0, 255)
    node_cols = lut[cidx]  # [N,3]

    # Build edge list (endpoint indices)
    edges = []
    N = scene.N
    for step in steps:
        for i in range(N):
            if i + step < N:
                edges.append((i, i + step))

    # Convert to numpy and add alpha channel
    node_positions_np = P.numpy()
    node_colors_rgba = torch.cat(
        [node_cols, torch.ones((len(node_cols), 1), device=device)], dim=-1
    ).numpy()

    # Build edge arrays
    if len(edges) > 0:
        edge_indices_np = np.array(edges, dtype=np.int32)

        # Compute edge colors from midpoint v-coordinate (same as CPU renderer)
        edge_i0 = torch.tensor([e[0] for e in edges], device=device, dtype=torch.int64)
        edge_i1 = torch.tensor([e[1] for e in edges], device=device, dtype=torch.int64)
        v0 = v[edge_i0]
        v1 = v[edge_i1]
        dv = wrap_pi_torch(v1 - v0)
        v_mid = v0 + 0.5 * dv

        # Color from midpoint v (pingpong colormap)
        t_raw_e = (v_mid % (2.0 * math.pi)) / (2.0 * math.pi)
        t_pp_e = pingpong01(t_raw_e)
        eidx = torch.clamp((t_pp_e * 255.0).to(torch.int64), 0, 255)
        edge_cols = lut[eidx]  # [E,3]
        edge_colors_rgba = torch.cat(
            [edge_cols, torch.ones((len(edge_cols), 1), device=device)], dim=-1
        ).numpy()
    else:
        edge_indices_np = np.empty((0, 2), dtype=np.int32)
        edge_colors_rgba = np.empty((0, 4), dtype=np.float32)

    # Calculate SDF parameters (DRY with CPU renderer)
    # Use the exact CPU projection logic to determine visibility and z_cam,
    # then convert the requested pixel radius to world-space per node.
    aspect = scene.W / scene.H

    # Reuse CPU projection: get valid mask and z_cam
    device = torch.device("cpu")
    C_t = torch.tensor(cam_np, device=device, dtype=torch.float32)
    right_t = torch.tensor(right_np, device=device, dtype=torch.float32)
    up_t = torch.tensor(up_np, device=device, dtype=torch.float32)
    fwd_t = torch.tensor(forward_np, device=device, dtype=torch.float32)
    P_t = torch.tensor(node_positions_np, device=device, dtype=torch.float32)
    valid_t, z_cam_t, px_t, py_t = project_points(
        P_t, C_t, right_t, up_t, fwd_t, scene.f, scene.W, scene.H
    )

    valid_nodes = valid_t.cpu().numpy().astype(bool)
    z_cam = z_cam_t.cpu().numpy()

    # Filter arrays to only valid nodes (like CPU splat)
    node_positions_np = node_positions_np[valid_nodes]
    node_colors_rgba = node_colors_rgba[valid_nodes]

    # Handle both uniform and per-point radii (scene.dot_radius_dynamic=False in tests)
    # r_world ≈ r_px * z_cam * aspect * 2 / (f * (W-1))
    node_radii = (scene.dot_radius_px * z_cam[valid_nodes] * aspect * 2.0) / (
        scene.f * (scene.W - 1)
    )

    # Calculate world_scale for edges using a typical distance
    # For edges, we use the mean distance of the edge endpoints
    cam_origin = np.array([0.0, 0.0, 0.0])  # Torus is centered at origin
    cam_to_center = float(np.linalg.norm(cam_np - cam_origin))
    typical_distance = (
        max(1.0, scene.r_inner - cam_to_center)
        if cam_to_center < scene.r_inner
        else cam_to_center - scene.R
    )
    typical_distance = abs(typical_distance)
    world_scale = (typical_distance * aspect * 2.0) / (scene.f * (scene.W - 1))

    # Capsules tend to render optically thicker than equivalent line/sphere chains,
    # so we deliberately scale them down by a fixed ratio in world space.
    edge_to_line_radius_ratio = 0.5
    edge_radius = scene.line_radius_px * world_scale * edge_to_line_radius_ratio

    # Shell thickness: should be thin enough to see detail.
    # Allow overriding the default 5% of minor radius via scene.shell_thickness_factor.
    shell_thickness_factor = getattr(scene, "shell_thickness_factor", 0.05)
    shell_thickness = scene.r_inner * shell_thickness_factor
    # Smooth k: use scene's smooth_k for junctions (fallback to 0.1 if absent)
    smooth_k = getattr(scene, "smooth_k", 0.1)

    # Use scene's hit_eps to match CPU torus silhouette in tests
    adjusted_hit_eps = float(scene.hit_eps)

    # Upload scene to GPU
    print(
        f"[SDF-GPU] Uploading {len(node_positions_np)} nodes and {len(edge_indices_np)} edges..."
    )
    print(f"[SDF-GPU] Camera: pos={cam_np}, forward={forward_np}")
    print(f"[SDF-GPU] First 3 nodes: {node_positions_np[:3]}")
    print(f"[SDF-GPU] First 3 radii: {node_radii[:3]}")
    print(f"[SDF-GPU] First 3 colors: {node_colors_rgba[:3, :3]}")  # RGB only
    print(
        f"[SDF-GPU] Total nodes before filter: {len(P)}, after filter: {len(node_positions_np)}"
    )

    # Build per-pixel ray directions to ensure parity with CPU
    D_cpu = build_rays(
        scene, device=torch.device("cpu"), right=right_t, up=up_t, forward=fwd_t
    )
    ray_dirs_np3 = D_cpu.numpy().reshape(-1, 3).astype(np.float32)
    # Pad to vec4 for std430 16-byte alignment
    zeros = np.zeros((ray_dirs_np3.shape[0], 1), dtype=np.float32)
    ray_dirs_np = np.concatenate([ray_dirs_np3, zeros], axis=1)

    print(
        f"[SDF-GPU] Ray dirs shape: {ray_dirs_np.shape}, depth shape: {depth_t_np.shape}"
    )
    print(
        f"[SDF-GPU] Depth stats: min={depth_t_np.min():.3f}, max={depth_t_np.max():.3f}, finite={np.isfinite(depth_t_np).sum()}/{depth_t_np.size}"
    )
    # Sample depth values at key locations
    center_y, center_x = scene.H // 2, scene.W // 2
    print(
        f"[SDF-GPU] Sample depths: center={depth_t_np[center_y, center_x]:.3f}, [100,100]={depth_t_np[100, 100]:.3f}, [200,200]={depth_t_np[200, 200]:.3f}"
    )

    # Debug: compute distances from camera to each uploaded sphere
    print(f"[SDF-GPU] Analyzing {len(node_positions_np)} uploaded nodes:")
    for i in range(len(node_positions_np)):
        pos = node_positions_np[i]
        dist_from_cam = np.linalg.norm(pos - cam_np)
        # Compute actual ray parameter t (distance along forward ray)
        L = pos - cam_np
        t_along_forward = np.dot(
            L, forward_np
        )  # Ray parameter along camera forward direction
        # Project to pixel coordinates to see where it appears
        pos_t = torch.tensor(pos, dtype=torch.float32)
        C_t = torch.tensor(cam_np, dtype=torch.float32)
        valid_proj, z_proj, px_proj, py_proj = project_points(
            pos_t.unsqueeze(0), C_t, right_t, up_t, fwd_t, scene.f, scene.W, scene.H
        )
        px, py = int(px_proj[0].item()), int(py_proj[0].item())
        depth_at_proj = (
            depth_t_np[py, px]
            if 0 <= py < scene.H and 0 <= px < scene.W
            else float("inf")
        )
        visible = t_along_forward <= depth_at_proj + scene.eps_t
        print(
            f"  Node {i}: t={t_along_forward:.3f}, projects to [{py:3d},{px:3d}], depth_there={depth_at_proj:.3f}, visible={visible}"
        )

    try:
        gpu_renderer.upload_scene(
            node_positions=node_positions_np,
            node_colors=node_colors_rgba,
            node_radii=node_radii,  # Per-node radii based on distance from camera
            edge_indices=edge_indices_np,
            edge_colors=edge_colors_rgba,
            torus_R=scene.R,
            torus_r=scene.r_inner,
            shell_thickness=shell_thickness,
            edge_radius=float(edge_radius),
            smooth_k=smooth_k,
            hit_eps=adjusted_hit_eps,
            eps_t=float(scene.eps_t),
            t_max=scene.t_max,
            max_steps=scene.max_steps,
            camera_pos=cam_np,
            camera_right=right_np,
            camera_up=up_np,
            camera_forward=forward_np,
            focal_length=scene.f,
            light_dir=np.array(scene.light_dir, dtype=np.float32),
            ambient=scene.ambient,
            diffuse_strength=scene.diffuse_strength,
            specular_strength=scene.specular_strength,
            shininess=scene.shininess,
            ray_dirs=ray_dirs_np,
            depth_buffer=depth_t_np,  # Pass CPU depth for perfect occlusion parity
        )

        # Render
        img_rgba = gpu_renderer.render()

        # Convert RGBA to RGB for output
        img_rgb = img_rgba[:, :, :3]

        # Cleanup GPU resources
        try:
            gpu_renderer.cleanup()
        except Exception as cleanup_err:
            print(
                f"[warning] GPU cleanup failed after successful render: {cleanup_err}"
            )

    except Exception as e:
        print(f"[error] SDF GPU rendering failed: {e}")
        print("[fallback] Using CPU renderer instead")
        if gpu_renderer:
            try:
                gpu_renderer.cleanup()
            except Exception as cleanup_err:
                # Ignore cleanup errors in fallback path, but log for diagnostics
                print(
                    f"[warning] GPU cleanup failed during fallback after render error: {cleanup_err}"
                )
        render_cpu(scene, steps, out_path)
        return

    # Save output
    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    plt.imsave(png_path, img_rgb)
    print(f"[saved] {png_path}")

    # Save scene metadata as JSON
    metadata = {
        "steps": steps,
        "scene": asdict(scene),
        "renderer": "SDF-GPU (OpenGL compute shader with SDF primitives)",
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {json_path}")


def render_uv_gpu(scene: Scene, steps: list[int], out_path: str):
    """
    UV-texture-based two-pass GPU rendering using OpenGL compute shaders.

    Pass A: Build 2D ink SDF texture in UV space
    Pass B: Ray march torus and sample ink texture

    This approach scales O(pixels) instead of O(pixels × primitives), enabling
    efficient rendering of thousands of nodes/edges.
    """
    try:
        from src.uv_gpu_renderer import UVGPURenderer
    except ImportError as e:
        print(f"[error] Failed to import UV GPU renderer: {e}")
        print("[fallback] Using CPU renderer instead")
        render_cpu(scene, steps, out_path)
        return

    print("[UV-GPU] Initializing OpenGL UV-texture-based renderer...")

    gpu_renderer = None
    try:
        # Create UV GPU renderer (with 2048x2048 UV texture by default)
        gpu_renderer = UVGPURenderer(scene.W, scene.H, uv_texture_size=2048)

        # Print GPU info
        info = gpu_renderer.gpu_info
        print(f"[UV-GPU] Vendor: {info['vendor']}")
        print(f"[UV-GPU] Renderer: {info['renderer']}")
        print(f"[UV-GPU] OpenGL Version: {info['version']}")
        print(f"[UV-GPU] GLSL Version: {info['glsl_version']}")
        print(
            f"[UV-GPU] UV Texture Resolution: {gpu_renderer.uv_texture_size}x{gpu_renderer.uv_texture_size}"
        )

    except Exception as e:
        print(f"[error] Failed to initialize UV GPU renderer: {e}")
        print("[fallback] Using CPU renderer instead")
        render_cpu(scene, steps, out_path)
        return

    # Organize outputs by file type
    out_path_p = Path(out_path)
    png_dir = Path("png")
    json_dir = Path("json")
    png_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)

    png_path = png_dir / out_path_p.name
    json_path = json_dir / out_path_p.with_suffix(".json").name

    # Build camera
    cam_np, right_np, up_np, forward_np = build_camera(scene)

    # Generate phyllotaxis points on CPU
    device = torch.device("cpu")
    u, v = torus_phyllotaxis_uv(scene.N, scene.R, scene.r_inner, device)
    P = torus_point_from_uv(u, v, scene.R, scene.r_inner)

    # Convert u, v to numpy in [-PI, PI] range (they already are)
    node_uv_np = torch.stack([u, v], dim=-1).numpy()  # [N, 2]

    # Build edge list
    edges = []
    N = scene.N
    for step in steps:
        for i in range(N):
            if i + step < N:
                edges.append((i, i + step))

    # Build edge UV array (u0, v0, u1, v1)
    if len(edges) > 0:
        edge_uv_list = []
        for i0, i1 in edges:
            # For each edge, store shortest-wrap endpoints in UV space
            u0, v0 = node_uv_np[i0]
            u1, v1 = node_uv_np[i1]
            edge_uv_list.append([u0, v0, u1, v1])
        edge_uv_np = np.array(edge_uv_list, dtype=np.float32)
    else:
        edge_uv_np = np.empty((0, 4), dtype=np.float32)

    # Convert pixel radii to UV-space radii
    # The torus surface metric is: ds² = (R + r·cos(v))²·du² + r²·dv²
    # At v=0 (outer equator), the circumference in u is 2π(R+r)
    # At v=π/2 (top), the circumference in u is 2πR
    # Average metric scale: ~2πR or ~2π(R+r/2)

    # For dot radius in UV space, we want dots to appear as a certain pixel size.
    # The CPU renderer uses world-space radii, and we need to convert those to UV radii.
    #
    # Heuristic: A pixel radius of r_px should map to a UV radius that represents
    # a similar angular extent on the torus surface.
    #
    # Using the outer radius R+r as reference:
    # Angular extent ≈ (r_world / (R + r)) radians
    #
    # First, convert pixel radius to world radius (same as before)
    aspect = scene.W / scene.H
    V = P.numpy() - cam_np[np.newaxis, :]
    z_cam = np.dot(V, forward_np)
    min_z_cam = max(0.1, scene.R * 0.5)
    z_cam_clamped = np.maximum(z_cam, min_z_cam)

    # World-space radii
    node_radii_world = (scene.dot_radius_px * z_cam_clamped * aspect * 2.0) / (
        scene.f * (scene.W - 1)
    )

    # Convert to UV-space radii (angular radians)
    # At outer edge: radius_uv ≈ radius_world / (R + r)
    # At inner edge: radius_uv ≈ radius_world / (R - r)
    # Use average: radius_uv ≈ radius_world / R
    node_radii_uv = node_radii_world / scene.R  # [N]

    # Similarly for edges
    cam_to_center = float(np.linalg.norm(cam_np))
    typical_distance = max(1.0, cam_to_center)
    world_scale = (typical_distance * aspect * 2.0) / (scene.f * (scene.W - 1))
    edge_radius_world = scene.line_radius_px * world_scale * 0.5
    edge_radii_uv = np.full(len(edges), edge_radius_world / scene.R, dtype=np.float32)

    # UV texture parameters
    smooth_k = getattr(scene, "smooth_k", 0.1)

    # Ink threshold: SDF < threshold means "inside ink"
    # With smooth blending, the ink alpha will transition smoothly around threshold=0
    ink_threshold = 0.0
    ink_smoothness = smooth_k  # Smoothness of ink edge (same as smooth_k)

    # Colors
    ink_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # White ink
    torus_color = np.array([0.1, 0.1, 0.1, 1.0], dtype=np.float32)  # Dark gray torus
    background_color = np.array(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )  # Black background

    # Upload scene to GPU
    print(f"[UV-GPU] Uploading {len(node_uv_np)} nodes and {len(edges)} edges...")
    print("[UV-GPU] Building ink SDF texture (Pass A)...")

    try:
        gpu_renderer.upload_scene(
            node_uv=node_uv_np,
            node_radii_uv=node_radii_uv,
            edge_uv=edge_uv_np,
            edge_radii_uv=edge_radii_uv,
            torus_R=scene.R,
            torus_r=scene.r_inner,
            smooth_k=smooth_k,
            ink_threshold=ink_threshold,
            ink_smoothness=ink_smoothness,
            hit_eps=scene.hit_eps,
            t_max=scene.t_max,
            max_steps=min(
                scene.max_steps, 200
            ),  # Fewer steps needed for single primitive
            camera_pos=cam_np,
            camera_right=right_np,
            camera_up=up_np,
            camera_forward=forward_np,
            light_dir=np.array(scene.light_dir, dtype=np.float32),
            ambient=scene.ambient,
            diffuse_strength=scene.diffuse_strength,
            specular_strength=scene.specular_strength,
            shininess=scene.shininess,
            ink_color=ink_color,
            torus_color=torus_color,
            background_color=background_color,
        )

        # Render (Pass A + Pass B)
        print("[UV-GPU] Ray marching torus and sampling ink texture (Pass B)...")
        img_rgba = gpu_renderer.render()

        # Convert RGBA to RGB for output
        img_rgb = img_rgba[:, :, :3]

        # Cleanup GPU resources
        try:
            gpu_renderer.cleanup()
        except Exception as cleanup_err:
            print(
                f"[warning] GPU cleanup failed after successful render: {cleanup_err}"
            )

    except Exception as e:
        print(f"[error] UV GPU rendering failed: {e}")
        import traceback

        traceback.print_exc()
        print("[fallback] Using CPU renderer instead")
        if gpu_renderer:
            try:
                gpu_renderer.cleanup()
            except Exception as cleanup_err:
                print(f"[warning] GPU cleanup failed during fallback: {cleanup_err}")
        render_cpu(scene, steps, out_path)
        return

    # Save output
    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    plt.imsave(png_path, img_rgb)
    print(f"[saved] {png_path}")

    # Save scene metadata as JSON
    metadata = {
        "steps": steps,
        "scene": asdict(scene),
        "renderer": "UV-GPU (OpenGL two-pass UV-texture-based ray marching)",
        "uv_texture_size": gpu_renderer.uv_texture_size,
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] {json_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=[],
        help="List of step sizes (e.g., --steps 1 13 21). If omitted, only dots are rendered.",
    )
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument(
        "--N", type=int, default=1597
    )  # Fibonacci number for seamless phyllotaxis pattern
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--f", type=float, default=1.2)
    ap.add_argument("--t_step", type=float, default=0.40)

    ap.add_argument(
        "--aa",
        type=int,
        default=1,
        help="Anti-aliasing factor: 1=off, 2=2x2 SSAA (4x pixels), 3=3x3 SSAA (9x pixels)",
    )

    ap.add_argument(
        "--cmap",
        type=str,
        default="gnuplot",
        help="Matplotlib colormap name (e.g., magma, inferno, viridis, plasma, gnuplot)",
    )

    ap.add_argument("--dot_px", type=int, default=4)
    ap.add_argument(
        "--dot_dynamic",
        action="store_true",
        default=True,
        help="Use dynamic dot sizing based on local density (default: True)",
    )
    ap.add_argument(
        "--no_dot_dynamic",
        action="store_false",
        dest="dot_dynamic",
        help="Disable dynamic dot sizing",
    )
    ap.add_argument(
        "--dot_size_min",
        type=float,
        default=0.5,
        help="Minimum size multiplier for dynamic dots (default: 0.5)",
    )
    ap.add_argument(
        "--dot_size_max",
        type=float,
        default=2.5,
        help="Maximum size multiplier for dynamic dots (default: 2.5)",
    )

    ap.add_argument("--line_px", type=int, default=1)
    ap.add_argument(
        "--line_dynamic",
        action="store_true",
        default=True,
        help="Use dynamic line thickness (inverse of dots) (default: True)",
    )
    ap.add_argument(
        "--no_line_dynamic",
        action="store_false",
        dest="line_dynamic",
        help="Disable dynamic line thickness",
    )
    ap.add_argument(
        "--line_size_min",
        type=float,
        default=0.5,
        help="Minimum size multiplier for dynamic lines (default: 0.5)",
    )
    ap.add_argument(
        "--line_size_max",
        type=float,
        default=2.5,
        help="Maximum size multiplier for dynamic lines (default: 2.5)",
    )
    ap.add_argument("--line_alpha", type=float, default=0.40)
    ap.add_argument("--line_samples", type=int, default=44)

    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Use OpenGL GPU acceleration (ray marching with compute shaders)",
    )
    ap.add_argument(
        "--sdf",
        action="store_true",
        help="Use SDF-based GPU rendering (true capsule lines on torus interior, not sampled spheres)",
    )
    ap.add_argument(
        "--uv",
        action="store_true",
        help="Use UV-texture-based GPU rendering (two-pass: build ink SDF texture, then ray march torus). Scales to thousands of nodes/edges efficiently.",
    )
    ap.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth blending between spheres (GPU mode only, may be slow with many spheres)",
    )
    ap.add_argument(
        "--smooth_k",
        type=float,
        default=0.3,
        help="Smoothing factor for smooth minimum (default: 0.3, higher = more blending)",
    )

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
        enable_smooth_blend=args.smooth,
        smooth_k=args.smooth_k,
    )

    # Choose renderer based on flags  - 2 renders max! one for CPU, and one for GPU (with smoothing)
    if args.uv:
        print("[mode] UV-GPU rendering (OpenGL two-pass UV-texture-based)")
        render_uv_gpu(
            scene=scene,
            steps=args.steps,
            out_path=args.out,
        )
    elif args.sdf:
        print("[mode] SDF-GPU rendering (OpenGL with SDF primitives)")
        render_sdf_gpu(
            scene=scene,
            steps=args.steps,
            out_path=args.out,
        )
    elif args.gpu:
        print("[mode] SDF-GPU rendering (OpenGL with SDF primitives)")
        render_sdf_gpu(
            scene=scene,
            steps=args.steps,
            out_path=args.out,
        )
    else:
        print("[mode] CPU rendering (PyTorch)")
        render_cpu(
            scene=scene,
            steps=args.steps,
            out_path=args.out,
        )
