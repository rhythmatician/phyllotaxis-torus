"""
Torus phyllotaxis net (GPU) — depth-tested solid torus + colored dots + on-surface curve “net”.

Key changes vs the earlier “tight spiral” look:
- By default, we DO NOT add edges (i -> i+1). Only (i -> i+s1) and (i -> i+s2).
- Net “lines” are drawn as curves constrained to the torus surface by interpolating in (u,v)
  with shortest wrap-around. This removes the interior “chord” look.

Note: these curves are on-surface parameter interpolations, not exact torus geodesics
(which would require solving the geodesic ODE). Visually, they behave like “surface wires”.

Deps:
  pip install numpy matplotlib
  pip install torch  (install a CUDA build if you want GPU acceleration)

Examples:
  # (13, 21) no i->i+1
  python torus_net_gpu.py --s1 13 --s2 21 --out torus_13_21.png

  # (34, 55) no i->i+1
  python torus_net_gpu.py --s1 34 --s2 55 --out torus_34_55.png

  # If you *do* want i->i+1 back:
  python torus_net_gpu.py --s1 13 --s2 21 --include_i_plus_1

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
    """Wrap angles to [-pi, pi)."""
    two_pi = 2.0 * math.pi
    return (a + math.pi) % two_pi - math.pi


def torus_sdf(P: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """
    Signed distance to a torus centered at origin around Z axis.
    P: (..., 3)
    """
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    qx = torch.sqrt(x * x + y * y) - R
    return torch.sqrt(qx * qx + z * z) - r


def torus_point_from_uv(u: torch.Tensor, v: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """
    Map (u,v) -> R^3 on torus surface.
    u: angle around donut (major circle)
    v: angle around tube (minor circle)
    """
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


@dataclass
class Scene:
    # Geometry
    R: float = 3.0
    r_outer: float = 2.6
    r_inner: float = 2.3

    # Points
    N: int = 1500

    # Camera / projection
    W: int = 1920
    H: int = 1080
    f: float = 1.2
    t_step: float = 0.40
    target: tuple = (0.0, 0.0, 1.0)

    # “Inside tube” camera recipe (matches what we used)
    u0_cam: float = -math.pi / 4
    v_cam: float = -0.28
    eps_wall: float = 0.05

    # Rendering / depth test
    eps_depth: float = 0.02
    hit_eps: float = 1.2e-3
    t_max: float = 160.0
    max_steps: int = 150

    # Stylization
    cmap_name: str = "magma"
    reverse_cmap: bool = False  # we already “pingpong”; this flips that if desired
    dot_radius_px: int = 4
    line_radius_px: int = 1
    line_alpha: float = 0.40
    line_samples_per_edge: int = 44


def pingpong01(t: torch.Tensor) -> torch.Tensor:
    # maps [0,1] -> [0,1] with a forward+reverse pingpong (no seam)
    return 1.0 - torch.abs(2.0 * t - 1.0)


def build_magma_lut(device, cmap_name="magma", n=256) -> torch.Tensor:
    cmap = plt.get_cmap(cmap_name)
    lut = np.asarray([cmap(i / (n - 1))[:3] for i in range(n)], dtype=np.float32)
    return torch.tensor(lut, device=device)


# -------------------------
# Phyllotaxis points on torus (u,v)
# -------------------------
def torus_phyllotaxis_uv(N: int, R: float, r: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return u,v for N points. u uses golden angle. v is chosen so that points spread in “area-ish”
    along the major direction by inverting g(v) = R*v + r*sin(v) via bisection.
    """
    if N <= 0:
        return torch.empty((0,), device=device), torch.empty((0,), device=device)

    phi = (1 + 5**0.5) / 2
    alpha = 2 * math.pi / (phi * phi)
    two_pi = 2.0 * math.pi

    # u is straightforward
    idx = torch.arange(N, device=device, dtype=torch.float32)
    u = (idx * alpha) % two_pi

    # v needs bisection (do it on GPU in parallel)
    s = (idx + 0.5) / N
    target = two_pi * R * s  # desired g(v)

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
# GPU raymarch for opaque torus depth
# -------------------------
def render_depth_gpu(scene: Scene, device, C, right, up, forward) -> torch.Tensor:
    """
    Returns depth buffer in camera-forward units:
      depth[y, x] = z_cam (dot(P - C, forward)) at first hit of |SDF| < hit_eps
    """
    W, H = scene.W, scene.H
    aspect = W / H

    # Pixel grid in normalized image coords
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2.0
    xs = xs * aspect

    # Build per-pixel ray directions
    # x_img = xs, y_img = ys; camera space uses [x_img/f, y_img/f, 1]
    x_img, y_img = torch.meshgrid(xs, ys, indexing="xy")  # [W,H] each
    dx = x_img / scene.f
    dy = y_img / scene.f
    dz = torch.ones_like(dx)

    # Convert to world: D = dx*right + dy*up + dz*forward
    # Shapes: right/up/forward are (3,)
    D = (
        dx[..., None] * right[None, None, :]
        + dy[..., None] * up[None, None, :]
        + dz[..., None] * forward[None, None, :]
    )  # [W,H,3]
    D = D / torch.linalg.norm(D, dim=-1, keepdim=True)

    # Flatten for marching
    Df = D.reshape(-1, 3)  # [P,3]
    Pcount = Df.shape[0]

    depth = torch.full((Pcount,), float("inf"), device=device)
    t = torch.zeros((Pcount,), device=device)
    alive = torch.ones((Pcount,), device=device, dtype=torch.bool)

    # March
    for _ in range(scene.max_steps):
        if not alive.any():
            break

        P = C[None, :] + Df * t[:, None]
        dist = torch.abs(torus_sdf(P, scene.R, scene.r_inner))

        hit = alive & (dist < scene.hit_eps)
        if hit.any():
            Ph = P[hit]
            z_cam = torch.matmul(Ph - C[None, :], forward)  # [nhit]
            depth[hit] = z_cam
            alive[hit] = False

        t_next = t + dist
        alive = alive & (t_next < scene.t_max) & torch.isfinite(t_next)
        t = torch.where(alive, t_next, t)

    # Reshape to [H,W] in usual image indexing (y,x)
    # We flattened in [W,H] order; convert carefully:
    # Our meshgrid was (xs, ys) indexing="xy" -> shape [W,H]
    depth_wh = depth.reshape(scene.W, scene.H).transpose(0, 1).contiguous()  # -> [H,W]
    return depth_wh


# -------------------------
# Projection + splatting (GPU)
# -------------------------
def project_points_gpu(P: torch.Tensor, C, right, up, forward, f: float, W: int, H: int):
    """
    P: [N,3]
    Returns:
      valid mask, z_cam, px, py (int64 pixel coords)
    """
    aspect = W / H
    V = P - C[None, :]
    x_cam = torch.matmul(V, right)
    y_cam = torch.matmul(V, up)
    z_cam = torch.matmul(V, forward)

    valid = z_cam > 1e-6

    x_img = f * x_cam / z_cam
    y_img = f * y_cam / z_cam

    # map to pixels
    px = torch.round(((x_img / aspect) + 1.0) * 0.5 * (W - 1)).to(torch.int64)
    py = torch.round(((1.0 - y_img) * 0.5) * (H - 1)).to(torch.int64)

    in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    valid = valid & in_bounds

    return valid, z_cam, px, py


def disk_offsets(radius_px: int) -> list[tuple[int, int]]:
    out = []
    r2 = radius_px * radius_px
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy <= r2:
                out.append((dx, dy))
    return out


def splat_dots(
    img: torch.Tensor,
    depth: torch.Tensor,
    P: torch.Tensor,
    v: torch.Tensor,
    scene: Scene,
    C, right, up, forward,
    lut: torch.Tensor,
):
    """
    Draw dots as depth-tested splats.
    img: [H,W,3] float32
    depth: [H,W] float32 (inf for background)
    """
    H, W = scene.H, scene.W
    valid, z_cam, px, py = project_points_gpu(P, C, right, up, forward, scene.f, W, H)

    px = px[valid]
    py = py[valid]
    z_cam = z_cam[valid]
    v_use = v[valid]

    # pingpong magma based on v
    t_raw = (v_use % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp = pingpong01(t_raw)
    if scene.reverse_cmap:
        t_pp = 1.0 - t_pp

    idx = torch.clamp((t_pp * 255.0).to(torch.int64), 0, 255)
    cols = lut[idx]  # [M,3]

    offsets = disk_offsets(scene.dot_radius_px)

    # z-buffer for dots to avoid overwriting closer dots with farther ones
    zbuf = torch.full((H, W), float("inf"), device=img.device)

    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not inb.any():
            continue

        x = x[inb]
        y = y[inb]
        z = z_cam[inb]
        c = cols[inb]

        # depth test: must be on/above visible surface
        dref = depth[y, x]
        ok = z <= (dref + scene.eps_depth)

        if not ok.any():
            continue

        x = x[ok]
        y = y[ok]
        z = z[ok]
        c = c[ok]

        # dot z-order: only write if this dot is closer than prior dot at that pixel
        prior = zbuf[y, x]
        closer = z < prior
        if closer.any():
            xw = x[closer]
            yw = y[closer]
            zw = z[closer]
            cw = c[closer]

            zbuf[yw, xw] = zw
            img[yw, xw, :] = cw


def splat_lines_on_surface(
    img: torch.Tensor,
    depth: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    edges: list[tuple[int, int]],
    scene: Scene,
    C, right, up, forward,
    lut: torch.Tensor,
):
    """
    Draw surface curves by sampling along (u,v) interpolation (shortest wrap).
    Depth-tested + alpha blended.
    """
    device = img.device
    H, W = scene.H, scene.W

    offsets = disk_offsets(scene.line_radius_px)

    # Prebuild all samples for all edges in one batch (fast enough at N~1500)
    samples = scene.line_samples_per_edge + 1
    A = torch.linspace(0.0, 1.0, samples, device=device, dtype=torch.float32)  # [S]

    # edge endpoint uv
    i0 = torch.tensor([e[0] for e in edges], device=device, dtype=torch.int64)
    i1 = torch.tensor([e[1] for e in edges], device=device, dtype=torch.int64)

    u0 = u[i0]
    v0 = v[i0]
    u1 = u[i1]
    v1 = v[i1]

    du = wrap_pi_torch(u1 - u0)
    dv = wrap_pi_torch(v1 - v0)

    # [E,S]
    uu = u0[:, None] + du[:, None] * A[None, :]
    vv = v0[:, None] + dv[:, None] * A[None, :]

    # color per edge from midpoint v
    v_mid = v0 + 0.5 * dv
    t_raw = (v_mid % (2.0 * math.pi)) / (2.0 * math.pi)
    t_pp = pingpong01(t_raw)
    if scene.reverse_cmap:
        t_pp = 1.0 - t_pp
    cidx = torch.clamp((t_pp * 255.0).to(torch.int64), 0, 255)
    edge_col = lut[cidx]  # [E,3]
    # expand to samples
    cols = edge_col[:, None, :].expand(-1, samples, -1).reshape(-1, 3)  # [E*S,3]

    # 3D points on surface
    P = torus_point_from_uv(uu.reshape(-1), vv.reshape(-1), scene.R, scene.r_inner)

    valid, z_cam, px, py = project_points_gpu(P, C, right, up, forward, scene.f, W, H)
    px = px[valid]
    py = py[valid]
    z_cam = z_cam[valid]
    cols = cols[valid]

    # alpha blend into image
    alpha = scene.line_alpha

    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not inb.any():
            continue

        x = x[inb]
        y = y[inb]
        z = z_cam[inb]
        c = cols[inb]

        # depth test
        dref = depth[y, x]
        ok = z <= (dref + scene.eps_depth)
        if not ok.any():
            continue

        x = x[ok]
        y = y[ok]
        c = c[ok]

        img[y, x, :] = (1.0 - alpha) * img[y, x, :] + alpha * c


# -------------------------
# Main render
# -------------------------
def build_camera(scene: Scene):
    # This matches the “inside tube looking up toward the funnel” recipe we’ve been using.
    u0 = scene.u0_cam
    e_r = np.array([math.cos(u0), math.sin(u0), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])

    centerline = scene.R * e_r
    rho = scene.r_inner - scene.eps_wall
    cam0 = centerline + rho * (math.cos(scene.v_cam) * e_r + math.sin(scene.v_cam) * e_z)

    # step camera slightly along direction toward “top of donut hole” at (0,0,r_outer)
    target_old = np.array([0.0, 0.0, scene.r_outer], dtype=float)
    dir_to_old = target_old - cam0
    dir_to_old /= np.linalg.norm(dir_to_old)
    cam = cam0 + scene.t_step * dir_to_old

    right, up, forward = camera_basis_np(cam, scene.target)
    return cam, right, up, forward


def render(scene: Scene, s1: int, s2: int, include_i_plus_1: bool, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} (cuda_available={torch.cuda.is_available()})")

    cam_np, right_np, up_np, forward_np = build_camera(scene)

    C = torch.tensor(cam_np, device=device, dtype=torch.float32)
    right = torch.tensor(right_np, device=device, dtype=torch.float32)
    up = torch.tensor(up_np, device=device, dtype=torch.float32)
    forward = torch.tensor(forward_np, device=device, dtype=torch.float32)

    # Depth buffer for solid torus (inner surface)
    depth = render_depth_gpu(scene, device, C, right, up, forward).to(torch.float32)

    # Base image: white background, black where torus is visible
    img = torch.ones((scene.H, scene.W, 3), device=device, dtype=torch.float32)
    torus_mask = torch.isfinite(depth)
    img[torus_mask, :] = 0.0

    # LUT for magma
    lut = build_magma_lut(device, cmap_name=scene.cmap_name, n=256)

    # Points on torus
    u, v = torus_phyllotaxis_uv(scene.N, scene.R, scene.r_inner, device=device)
    P = torus_point_from_uv(u, v, scene.R, scene.r_inner)

    # Build edges (optionally omit i->i+1)
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

    # Lines first
    splat_lines_on_surface(
        img, depth,
        u=u, v=v,
        edges=edges,
        scene=scene,
        C=C, right=right, up=up, forward=forward,
        lut=lut,
    )

    # Dots on top
    splat_dots(
        img, depth,
        P=P, v=v,
        scene=scene,
        C=C, right=right, up=up, forward=forward,
        lut=lut,
    )

    # Save (move to CPU)
    img_cpu = img.clamp(0.0, 1.0).detach().cpu().numpy()
    plt.imsave(out_path, img_cpu)
    print(f"[saved] {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=int, required=True)
    ap.add_argument("--s2", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--include_i_plus_1", action="store_true", help="Add i->i+1 edges (tight spiral look). Default OFF.")
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--W", type=int, default=1920)
    ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--f", type=float, default=1.2)
    ap.add_argument("--t_step", type=float, default=0.40)

    ap.add_argument("--reverse", action="store_true", help="Reverse the ping-pong colormap direction.")
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
    )

    render(
        scene=scene,
        s1=args.s1,
        s2=args.s2,
        include_i_plus_1=args.include_i_plus_1,
        out_path=args.out,
    )
