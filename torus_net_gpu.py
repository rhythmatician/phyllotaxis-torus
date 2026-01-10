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
  python torus_net_gpu_fixed_occlusion.py --s1 13 --s2 21 --out out_13_21.png
  python torus_net_gpu_fixed_occlusion.py --s1 34 --s2 55 --out out_34_55.png
  python torus_net_gpu_fixed_occlusion.py --s1 13 --s2 21 --include_i_plus_1 --out spiral.png
"""

import argparse
import math
from dataclasses import dataclass

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


def build_lut(device, cmap_name="magma", n=256) -> torch.Tensor:
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
    cmap_name: str = "magma"
    reverse_cmap: bool = False

    dot_radius_px: int = 4
    line_radius_px: int = 1
    line_alpha: float = 0.40
    line_samples_per_edge: int = 44


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
def splat_spheres(
    img: torch.Tensor,
    depth_t: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    centers: torch.Tensor,    # [M,3]
    colors: torch.Tensor,     # [M,3]
    radius_px: int,
    scene: Scene,
    right, up, forward,
    alpha: float | None = None,   # None => overwrite
):
    """
    Render each center as a small sphere, per covered pixel:
      - gather ray direction D[y,x]
      - solve ray-sphere intersection
      - compare t_hit_sphere to depth_t[y,x] (torus)
    """
    H, W = scene.H, scene.W
    valid, z_cam, px, py = project_points(centers, C, right, up, forward, scene.f, W, H)

    centers = centers[valid]
    colors = colors[valid]
    z_cam = z_cam[valid]
    px = px[valid]
    py = py[valid]

    if centers.numel() == 0:
        return

    # Convert requested pixel radius to an approximate world radius *per point* (keeps apparent size stable):
    # r_px ≈ r_world * f/(z_cam*aspect)*(W-1)/2  => r_world ≈ r_px * z_cam * aspect * 2 / (f*(W-1))
    aspect = W / H
    r_world = (radius_px * z_cam * aspect * 2.0) / (scene.f * (W - 1))
    r2 = r_world * r_world

    offsets = disk_offsets(radius_px)

    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not inb.any():
            continue

        x = x[inb]
        y = y[inb]
        P0 = centers[inb]
        col = colors[inb]
        rr2 = r2[inb]

        # ray data for these pixels
        Di = D[y, x, :]  # [K,3]

        # ray-sphere: |(C + tD) - P0|^2 = r^2
        L = P0 - C[None, :]               # [K,3]
        b = torch.sum(Di * L, dim=-1)     # [K]
        c = torch.sum(L * L, dim=-1) - rr2
        disc = b * b - c

        ok = disc > 0.0
        if not ok.any():
            continue

        x = x[ok]
        y = y[ok]
        col = col[ok]
        b = b[ok]
        disc = disc[ok]

        t_sphere = b - torch.sqrt(disc)
        ok2 = t_sphere > 0.0
        if not ok2.any():
            continue

        x = x[ok2]
        y = y[ok2]
        col = col[ok2]
        t_sphere = t_sphere[ok2]

        # occlusion vs torus
        dt = depth_t[y, x]
        vis = t_sphere <= (dt + scene.eps_t)

        if not vis.any():
            continue

        x = x[vis]
        y = y[vis]
        col = col[vis]

        if alpha is None:
            img[y, x, :] = col
        else:
            img[y, x, :] = (1.0 - alpha) * img[y, x, :] + alpha * col


# -------------------------
# Render
# -------------------------
def render(scene: Scene, s1: int, s2: int, include_i_plus_1: bool, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} (cuda_available={torch.cuda.is_available()})")

    cam_np, right_np, up_np, forward_np = build_camera(scene)

    C = torch.tensor(cam_np, device=device, dtype=torch.float32)
    right = torch.tensor(right_np, device=device, dtype=torch.float32)
    up = torch.tensor(up_np, device=device, dtype=torch.float32)
    forward = torch.tensor(forward_np, device=device, dtype=torch.float32)

    # Rays + depth (t_hit)
    D = build_rays(scene, device, right, up, forward)              # [H,W,3]
    depth_t = render_depth_t(scene, device, C, D).to(torch.float32)  # [H,W]

    # Base: white background, black where torus is visible
    img = torch.ones((scene.H, scene.W, 3), device=device, dtype=torch.float32)
    img[torch.isfinite(depth_t), :] = 0.0

    # Points
    u, v = torus_phyllotaxis_uv(scene.N, scene.R, scene.r_inner, device=device)
    P = torus_point_from_uv(u, v, scene.R, scene.r_inner)

    # Colors (pingpong magma, seam-free)
    lut = build_lut(device, scene.cmap_name, 256)
    t_raw = (v % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp = pingpong01(t_raw)
    if scene.reverse_cmap:
        t_pp = 1.0 - t_pp
    cidx = torch.clamp((t_pp * 255.0).to(torch.int64), 0, 255)
    dot_cols = lut[cidx]  # [N,3]

    # Edges (optionally exclude i->i+1)
    edges = []
    N = scene.N
    if include_i_plus_1:
        for i in range(N - 1):
            edges.append((i, i + 1))
    for i in range(N):
        if i + s1 < N:
            edges.append((i, i + s1))
        if i + s2 < N:
            edges.append((i, i + s2))

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
    if scene.reverse_cmap:
        t_pp_e = 1.0 - t_pp_e
    eidx = torch.clamp((t_pp_e * 255.0).to(torch.int64), 0, 255)
    edge_cols = lut[eidx]  # [E,3]
    line_cols = edge_cols[:, None, :].expand(-1, samples, -1).reshape(-1, 3)

    # Render lines first (alpha blend)
    splat_spheres(
        img=img,
        depth_t=depth_t,
        D=D,
        C=C,
        centers=line_pts,
        colors=line_cols,
        radius_px=scene.line_radius_px,
        scene=scene,
        right=right,
        up=up,
        forward=forward,
        alpha=scene.line_alpha,
    )

    # Render dots on top (overwrite)
    splat_spheres(
        img=img,
        depth_t=depth_t,
        D=D,
        C=C,
        centers=P,
        colors=dot_cols,
        radius_px=scene.dot_radius_px,
        scene=scene,
        right=right,
        up=up,
        forward=forward,
        alpha=None,
    )

    img_cpu = img.clamp(0.0, 1.0).detach().cpu().numpy()
    plt.imsave(out_path, img_cpu)
    print(f"[saved] {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=int, required=True)
    ap.add_argument("--s2", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--include_i_plus_1", action="store_true")

    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--f", type=float, default=1.2)
    ap.add_argument("--t_step", type=float, default=0.40)

    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--dot_px", type=int, default=4)
    ap.add_argument("--line_px", type=int, default=1)
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
        reverse_cmap=args.reverse,
        dot_radius_px=args.dot_px,
        line_radius_px=args.line_px,
        line_alpha=args.line_alpha,
        line_samples_per_edge=args.line_samples,
    )

    render(
        scene=scene,
        s1=args.s1,
        s2=args.s2,
        include_i_plus_1=args.include_i_plus_1,
        out_path=args.out,
    )
