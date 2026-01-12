import torch


# -------------------------
# Correct occluded splatting: ray–sphere per pixel
# -------------------------
def phong_shade(
    base_color: torch.Tensor,  # [K,3]
    normal: torch.Tensor,  # [K,3]
    view_dir: torch.Tensor,  # [K,3]
    light_dir: torch.Tensor,  # [3]
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
    diff = torch.clamp(
        torch.sum(normal * light_dir[None, :], dim=-1, keepdim=True), 0.0, 1.0
    )
    diffuse_col = diffuse_strength * diff * base_color

    # Specular (Blinn-Phong variant: half-vector)
    half_vec = (view_dir + light_dir[None, :]) / (
        torch.linalg.norm(view_dir + light_dir[None, :], dim=-1, keepdim=True) + 1e-6
    )
    spec = torch.clamp(torch.sum(normal * half_vec, dim=-1, keepdim=True), 0.0, 1.0)
    spec_pow = torch.pow(spec, shininess)
    specular_col = specular_strength * spec_pow * torch.ones_like(base_color)

    return ambient_col + diffuse_col + specular_col
