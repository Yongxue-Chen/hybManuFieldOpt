"""
Tool geometry definitions and sampling for AM and SM tools.

AM tool: inverted cone (apex down) + infinite cylinder above, always vertical (+z).
SM tool: hemisphere (ball tip) + cylindrical shank + cylindrical holder,
         oriented by the neural-network field3 output.
"""
import torch
import numpy as np
from typing import Tuple


def _normalize_vectors(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize batched vectors along the last dimension."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _stratified_01(n: int, device, dtype) -> torch.Tensor:
    """Deterministic stratified samples in (0, 1)."""
    if n <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=dtype)
    return (idx + 0.5) / n


def _golden_theta(n: int, device, dtype) -> torch.Tensor:
    """Deterministic quasi-uniform azimuths in [0, 2pi)."""
    if n <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=dtype)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    return idx * golden_angle


# ---------------------------------------------------------------------------
# Rotation utility (ported from loss_multi_field.py)
# ---------------------------------------------------------------------------

def _skew_batch(v):
    """Build skew-symmetric matrices for batched 3-D vectors.

    Args:
        v: Tensor of shape ``(M, 3)``.

    Returns:
        Tensor of shape ``(M, 3, 3)`` where each matrix is ``[v]_x``.
    """
    x, y, z = v.unbind(-1)
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([O, -z, y], dim=-1),
        torch.stack([z, O, -x], dim=-1),
        torch.stack([-y, x, O], dim=-1),
    ], dim=-2)


def _align_z_to_v_batch(v, eps=1e-12):
    """Rotation matrices that map canonical +Z to each target direction.

    Args:
        v: Target direction tensor of shape ``(M, 3)``.
        eps: Numerical epsilon for normalization/division stability.

    Returns:
        Rotation tensor of shape ``(M, 3, 3)``.
    """
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    m = v.shape[0]
    dev, dt = v.device, v.dtype
    k = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt).expand_as(v)
    c = (k * v).sum(-1)
    vx = torch.cross(k, v, dim=-1)
    s = vx.norm(dim=-1)
    K = _skew_batch(vx)
    I = torch.eye(3, device=dev, dtype=dt).expand(m, 3, 3)
    f = ((1.0 - c) / (s * s + eps)).view(-1, 1, 1)
    R_gen = I + K + K @ K * f
    R_180 = torch.tensor(
        [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], device=dev, dtype=dt,
    ).expand(m, 3, 3)
    par = (s < 1e-12).view(-1, 1, 1)
    pos = (c > 0).view(-1, 1, 1)
    return torch.where(par, torch.where(pos, I, R_180), R_gen)


def _align_z_to_u_upper_half_robust(u, eps=1e-12):
    """
    针对上半球优化的坐标系对齐函数
    将 +z 轴旋转至 u 方向 (u_z >= 0)
    """
    # 1. 强制归一化，确保旋转矩阵不含缩放
    norm = torch.norm(u, dim=-1, keepdim=True)
    u = u / (norm + eps)
    
    m = u.shape[0]
    device, dtype = u.device, u.dtype
    
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    zero = torch.zeros(m, device=device, dtype=dtype)

    # 2. 构造反对称矩阵 K = [k x u]_x
    # 此时旋转轴 v = (0,0,1) x (ux, uy, uz) = (-uy, ux, 0)
    K = torch.stack([
        torch.stack([zero,  zero,  ux],   dim=-1), # Row 0
        torch.stack([zero,  zero,  uy],   dim=-1), # Row 1
        torch.stack([-ux,   -uy,   zero], dim=-1)  # Row 2
    ], dim=-2) 

    I = torch.eye(3, device=device, dtype=dtype).expand(m, 3, 3)
    
    # 3. Rodrigues 简化公式：R = I + K + K^2 * (1/(1+uz))
    # 因为 uz >= 0，所以 (1 + uz) 绝不会除以 0
    f = (1.0 / (1.0 + uz)).view(-1, 1, 1)
    # 使用 bmm 进行矩阵批处理乘法
    R = I + K + torch.bmm(K, K) * f
    
    return R


# ---------------------------------------------------------------------------
# AM tool sampling  (model-space coordinates)
# ---------------------------------------------------------------------------

def sample_am_tool(
    path_points: torch.Tensor,
    cone_height: float,
    cone_half_angle: float,
    AM_collision_margin: float,
    aabb_max_z: float,
    n_cone: int = 100,
    n_cylinder: int = 100,
    n_cone_surface: int = 0,
    n_cylinder_surface: int = 0,
    tool_axis: torch.Tensor = None,
    cone_tip_bias: float = 2.0,
) -> torch.Tensor:
    """
    Generate sample points representing the volume of the AM tool.

    The AM tool is modeled as an inverted cone (material deposition nozzle)
    topped by a cylinder (the machine head/spindle).

    If 'tool_axis' is provided, the tool shape is rotated to align with this
    axis. Otherwise, it defaults to standing vertically (+Z direction).

    Args:
        path_points: (M, 3) tensor of path waypoints (tool tip positions).
        cone_height: Height of the conical section.
        cone_half_angle: Angle of the cone opening (radians).
        aabb_max_z: The maximum Z height of the printing volume (for cylinder cap).
        n_cone: Number of random samples to generate inside the cone volume.
        n_cylinder: Number of random samples inside the cylinder volume above the cone.
        n_cone_surface: Number of quasi-uniform samples on cone lateral surface.
                If <= 0, uses max(8, n_cone // 4).
        n_cylinder_surface: Number of quasi-uniform samples on cylinder side surface.
                    If <= 0, uses max(8, n_cylinder // 4).
        tool_axis: Optional (M, 3) vector specifying the tool orientation.
                   If None, defaults to [0, 0, 1] (vertical).
        cone_tip_bias: Exponent for cone height sampling; >1 gives denser sampling
                       near the tip (t=rand()^cone_tip_bias). Default 2.0.

    Returns:
        (M, n_cone + n_cylinder, 3) tensor of sample points relative to each path point.
    """
    M = path_points.shape[0]
    dev, dt = path_points.device, path_points.dtype
    Rmax = cone_height * np.tan(cone_half_angle)
    n_cone_surface = n_cone_surface if n_cone_surface > 0 else max(8, n_cone // 4)
    n_cylinder_surface = n_cylinder_surface if n_cylinder_surface > 0 else max(8, n_cylinder // 4)
    n_total = n_cone + n_cylinder + n_cone_surface + n_cylinder_surface
    samples = torch.empty(M, n_total, 3, device=dev, dtype=dt)
    base = path_points.unsqueeze(1)  # (M,1,3)
    cursor = 0

    # --- Cone (canonical along +z) ---
    # Tip-heavy sampling: t = rand^cone_tip_bias puts more samples near the tip (small t).
    if n_cone > 0:
        h_min = AM_collision_margin
        t_min = h_min / cone_height
        t_raw = torch.rand(M, n_cone, device=dev, dtype=dt)
        t = t_min + (1.0 - t_min) * (t_raw ** cone_tip_bias)
        r_max = t * Rmax
        theta = 2.0 * torch.pi * torch.rand(M, n_cone, device=dev, dtype=dt)
        r = torch.sqrt(torch.rand(M, n_cone, device=dev, dtype=dt)) * r_max
        dx = r * torch.cos(theta)
        dy = r * torch.sin(theta)
        dz = t * cone_height
        samples[:, cursor:cursor + n_cone, 0] = dx
        samples[:, cursor:cursor + n_cone, 1] = dy
        samples[:, cursor:cursor + n_cone, 2] = dz
        cursor += n_cone

    # --- Cone lateral surface (quasi-uniform) ---
    if n_cone_surface > 0:
        h_min = AM_collision_margin
        t_min = h_min / cone_height
        u = _stratified_01(n_cone_surface, dev, dt)
        # Lateral-area-aware mapping: area ~ t dt => t = sqrt(...)
        t_surface = torch.sqrt(t_min * t_min + (1.0 - t_min * t_min) * u)
        theta_surface = _golden_theta(n_cone_surface, dev, dt)
        r_surface = t_surface * Rmax
        dx_s = r_surface * torch.cos(theta_surface)
        dy_s = r_surface * torch.sin(theta_surface)
        dz_s = t_surface * cone_height
        samples[:, cursor:cursor + n_cone_surface, 0] = dx_s.unsqueeze(0).expand(M, -1)
        samples[:, cursor:cursor + n_cone_surface, 1] = dy_s.unsqueeze(0).expand(M, -1)
        samples[:, cursor:cursor + n_cone_surface, 2] = dz_s.unsqueeze(0).expand(M, -1)
        cursor += n_cone_surface

    # --- Cylinder above cone (canonical along +z) ---
    if n_cylinder > 0:
        theta_c = 2.0 * torch.pi * torch.rand(M, n_cylinder, device=dev, dtype=dt)
        r_c = torch.sqrt(torch.rand(M, n_cylinder, device=dev, dtype=dt)) * Rmax
        dx_c = r_c * torch.cos(theta_c)
        dy_c = r_c * torch.sin(theta_c)
        h_cyl_default = max(float(aabb_max_z) - cone_height, 0.0)
        uz = torch.rand(M, n_cylinder, device=dev, dtype=dt)
        samples[:, cursor:cursor + n_cylinder, 0] = dx_c
        samples[:, cursor:cursor + n_cylinder, 1] = dy_c
        samples[:, cursor:cursor + n_cylinder, 2] = cone_height + uz * h_cyl_default
        cursor += n_cylinder

    # --- Cylinder side surface (quasi-uniform) ---
    if n_cylinder_surface > 0:
        h_cyl_default = max(float(aabb_max_z) - cone_height, 0.0)
        theta_cs = _golden_theta(n_cylinder_surface, dev, dt)
        u_z = _stratified_01(n_cylinder_surface, dev, dt)
        dx_cs = Rmax * torch.cos(theta_cs)
        dy_cs = Rmax * torch.sin(theta_cs)
        dz_cs = cone_height + u_z * h_cyl_default
        samples[:, cursor:cursor + n_cylinder_surface, 0] = dx_cs.unsqueeze(0).expand(M, -1)
        samples[:, cursor:cursor + n_cylinder_surface, 1] = dy_cs.unsqueeze(0).expand(M, -1)
        samples[:, cursor:cursor + n_cylinder_surface, 2] = dz_cs.unsqueeze(0).expand(M, -1)
        cursor += n_cylinder_surface

    # --- Rotation (if custom axis provided) ---
    if tool_axis is not None:
        tool_axis = _normalize_vectors(tool_axis)
        # Create rotation matrices that align +Z to the custom 'tool_axis'
        # R = _align_z_to_v_batch(tool_axis)  # (M, 3, 3)
        R = _align_z_to_u_upper_half_robust(tool_axis)
        # Apply rotation to all sample points
        samples = torch.einsum('ikl,ijl->ijk', R, samples)

    samples = samples + base
    return samples


# ---------------------------------------------------------------------------
# SM tool sampling  (model-space coordinates)
# ---------------------------------------------------------------------------

def _sample_sm_canonical(
    r_tip: float,
    r_shank: float,
    R: float,
    h: float,
    H: float,
    collision_margin: float,
    m: int,
    n_sphere: int,
    n_shank: int,
    n_holder: int,
    n_sphere_surface: int,
    n_shank_surface: int,
    n_holder_surface: int,
    n_holder_bottom_surface: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    Sample canonical SM tool points (before rotation / translation).

    Geometry (axis = +z from ball centre):
    1. sphere, radius r_tip   -> shrunk by collision_margin
    2. Shank cylinder, radius r_shank, z in [0, h]  -> NOT shrunk
      3. Holder cylinder, radius R, z in [h, h+H]  -> NOT shrunk

    Returns (m, N1+N2+N3+Ns1+Ns2+Ns3+Ns4, 3).
    """
    r_sphere = max(r_tip - collision_margin, 1e-6)

    # 1) sphere
    if n_sphere > 0:
        dirs = torch.randn(m, n_sphere, 3, device=device, dtype=dtype)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)
        radii = r_sphere * torch.rand(
            m, n_sphere, 1, device=device, dtype=dtype
        ).pow(1.0 / 3.0)
        p1 = dirs * radii
    else:
        p1 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 1s) sphere surface (quasi-uniform via Fibonacci sphere)
    if n_sphere_surface > 0:
        idx = _stratified_01(n_sphere_surface, device, dtype)
        z = 1.0 - 2.0 * idx
        rr = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        theta = _golden_theta(n_sphere_surface, device, dtype)
        xs = rr * torch.cos(theta)
        ys = rr * torch.sin(theta)
        zs = z
        p1s_single = torch.stack([xs, ys, zs], dim=-1) * r_sphere
        p1s = p1s_single.unsqueeze(0).expand(m, -1, -1)
    else:
        p1s = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 2) Shank (small cylinder)
    if n_shank > 0:
        th2 = 2 * torch.pi * torch.rand(m, n_shank, 1, device=device, dtype=dtype)
        rho2 = r_shank * torch.sqrt(torch.rand(m, n_shank, 1, device=device, dtype=dtype))
        x2 = rho2 * torch.cos(th2)
        y2 = rho2 * torch.sin(th2)
        z2 = h * torch.rand(m, n_shank, 1, device=device, dtype=dtype)
        p2 = torch.cat([x2, y2, z2], dim=-1)
    else:
        p2 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 2s) shank side surface (quasi-uniform)
    if n_shank_surface > 0:
        th2s = _golden_theta(n_shank_surface, device, dtype)
        z2s = h * _stratified_01(n_shank_surface, device, dtype)
        x2s = r_shank * torch.cos(th2s)
        y2s = r_shank * torch.sin(th2s)
        p2s_single = torch.stack([x2s, y2s, z2s], dim=-1)
        p2s = p2s_single.unsqueeze(0).expand(m, -1, -1)
    else:
        p2s = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 3) Holder (big cylinder, NOT shrunk)
    if n_holder > 0:
        th3 = 2 * torch.pi * torch.rand(m, n_holder, 1, device=device, dtype=dtype)
        rho3 = R * torch.sqrt(torch.rand(m, n_holder, 1, device=device, dtype=dtype))
        x3 = rho3 * torch.cos(th3)
        y3 = rho3 * torch.sin(th3)
        z3 = h + H * (torch.rand(m, n_holder, 1, device=device, dtype=dtype) ** 3.0)
        nBottom = max(n_holder // 4, 1)
        z3[:,:nBottom,:] = h
        p3 = torch.cat([x3, y3, z3], dim=-1)
    else:
        p3 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 3s) holder side surface (quasi-uniform)
    if n_holder_surface > 0:
        th3s = _golden_theta(n_holder_surface, device, dtype)
        z3s = h + H * _stratified_01(n_holder_surface, device, dtype)
        x3s = R * torch.cos(th3s)
        y3s = R * torch.sin(th3s)
        p3s_single = torch.stack([x3s, y3s, z3s], dim=-1)
        p3s = p3s_single.unsqueeze(0).expand(m, -1, -1)
    else:
        p3s = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 3b) holder bottom surface (z = h, deterministic quasi-uniform disk)
    if n_holder_bottom_surface > 0:
        th3b = _golden_theta(n_holder_bottom_surface, device, dtype)
        u3b = _stratified_01(n_holder_bottom_surface, device, dtype)
        rho3b = R * torch.sqrt(u3b)
        x3b = rho3b * torch.cos(th3b)
        y3b = rho3b * torch.sin(th3b)
        z3b = torch.full_like(x3b, h)
        p3b_single = torch.stack([x3b, y3b, z3b], dim=-1)
        p3b = p3b_single.unsqueeze(0).expand(m, -1, -1)
    else:
        p3b = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # Group sphere-related samples first so a single index splits sphere vs. shank+holder.
    return torch.cat([p1, p1s, p2, p2s, p3, p3s, p3b], dim=1)


def sample_sm_tool(
    path_points: torch.Tensor,
    model,
    sm_tip_diameter: float,
    sm_shank_diameter: float,
    sm_tool_length: float,
    sm_holder_diameter: float,
    sm_holder_length: float,
    collision_margin: float = 0.0,
    n_sphere: int = 50,
    n_shank: int = 80,
    n_holder: int = 10,
    n_sphere_surface: int = 0,
    n_shank_surface: int = 0,
    n_holder_surface: int = 0,
    n_holder_bottom_surface: int = 0,
    override_normal_vec: torch.Tensor = None,
    override_tool_vec: torch.Tensor = None,
) -> torch.Tensor:
    """
    Sample points inside the SM tool for a batch of path points.

    Tool orientation is queried from model's field3 unless overrides are
    given.  The hemisphere volume/surface are shrunk inward by
    *collision_margin* (model-space units); shank and holder are **not**
    shrunk.

    Args:
        path_points: (M, 3) in **model space**.
        model:       Neural-network model (must support field_type='field3').
        sm_tip_diameter, sm_shank_diameter, sm_tool_length,
        sm_holder_diameter, sm_holder_length:
            Tool geometry parameters **in model-space units** (already /SCALE).
        collision_margin: Inward shrinkage applied to hemisphere volume/surface
                  only (**model-space units**).
        n_hemisphere, n_shank, n_holder: Volume sample counts per part.
        n_sphere_surface, n_shank_surface, n_holder_surface, n_holder_bottom_surface:
            Quasi-uniform surface sample counts. If <= 0, auto set to
            max(8, volume_count // 4) for each part.
        override_normal_vec: (M, 3) surface normal override.  ``None`` →
            read from field3.
        override_tool_vec: (M, 3) tool axis override.  ``None`` → read from
            field3.

    Returns:
        (M, N_total, 3) tensor of sample points in model space.
    """
    M = path_points.shape[0]
    dev, dt = path_points.device, path_points.dtype

    r_tip = sm_tip_diameter / 2.0
    r_shank = sm_shank_diameter / 2.0
    R = sm_holder_diameter / 2.0
    h = sm_tool_length
    H = sm_holder_length

    if override_normal_vec is not None and override_tool_vec is not None:
        normal_vec = override_normal_vec
        tool_vec = override_tool_vec
    else:
        with torch.no_grad():
            two_vec = model.forward(path_points, field_type='field3')
        normal_vec = two_vec[..., :3] if override_normal_vec is None else override_normal_vec
        tool_vec = two_vec[..., 3:] if override_tool_vec is None else override_tool_vec

    normal_vec = _normalize_vectors(normal_vec)
    tool_vec = _normalize_vectors(tool_vec)

    n_sphere_surface = n_sphere_surface if n_sphere_surface > 0 else max(8, n_sphere // 4)
    n_shank_surface = n_shank_surface if n_shank_surface > 0 else max(8, n_shank // 4)
    n_holder_surface = n_holder_surface if n_holder_surface > 0 else max(8, n_holder // 4)
    n_holder_bottom_surface = (
        n_holder_bottom_surface if n_holder_bottom_surface > 0 else max(8, n_holder // 4)
    )

    ball_center = path_points + normal_vec * r_tip

    # print(f"r_tip: {r_tip}, r_shank: {r_shank}, R: {R}, h: {h}, H: {H}, collision_margin: {collision_margin}")

    canonical = _sample_sm_canonical(
        r_tip, r_shank, R, h, H, collision_margin,
        M,
        n_sphere, n_shank, n_holder,
        n_sphere_surface, n_shank_surface, n_holder_surface,
        n_holder_bottom_surface,
        dev, dt,
    )

    # Rot = _align_z_to_v_batch(tool_vec)  # (M,3,3)
    Rot = _align_z_to_u_upper_half_robust(tool_vec)
    rotated = torch.einsum('ikl,ijl->ijk', Rot, canonical)  # (M, N, 3)
    return rotated + ball_center.unsqueeze(1)


def query_sm_orientation(model, path_points: torch.Tensor):
    """Query field3 for SM tool normal and tool vectors.

    Args:
        model: Trained neural model supporting ``field_type='field3'``.
        path_points: Tensor ``(M, 3)`` of model-space path points.

    Returns:
        (normal_vec, tool_vec) each of shape (M, 3).
    """
    with torch.no_grad():
        two_vec = model.forward(path_points, field_type='field3')
    normal_vec = _normalize_vectors(two_vec[..., :3])
    tool_vec = _normalize_vectors(two_vec[..., 3:])
    return normal_vec, tool_vec
