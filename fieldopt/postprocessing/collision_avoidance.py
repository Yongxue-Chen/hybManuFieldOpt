"""
Collision avoidance by adjusting tool axis orientation.

For each path point that causes a collision with the default tool axis, this
module searches candidate orientations on a cone around the default axis.  The
first collision-free orientation is accepted; otherwise the point is marked as
*unresolved*.

Two helper entry points are provided:

* ``avoid_am_collisions`` – searches within a cone around +z.
* ``avoid_sm_collisions`` – two-phase search for SM ball-end mills:

    - **Phase 1** varies ``tool_vec`` only (ball centre fixed) to resolve
      shank / holder collisions.
    - **Phase 2** varies ``normal_vec`` to shift the ball centre (keeping
      ``tool_vec`` at its default).  If the ball is then clear but the
      shank / holder still collides, an inner search over ``tool_vec``
      candidates finds a collision-free axis.

  The initial detection classifies collisions by tool region so that each
  point only runs the phase(s) that can actually resolve its collision type.

The unified ``check_and_avoid_collisions`` function dispatches to the correct
variant based on ``layer_type``.
"""
import math
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm

from .tool_shape import (
    sample_am_tool,
    sample_sm_tool,
    query_sm_orientation,
    _align_z_to_v_batch,
    _align_z_to_u_upper_half_robust
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolOrientation:
    """Per-point tool axis information."""
    am_axis: np.ndarray            # (3,) AM tool axis (+z or adjusted)
    sm_normal_vec: np.ndarray      # (3,) SM surface normal (zeros for AM)
    sm_tool_vec: np.ndarray        # (3,) SM tool axis (zeros for AM)
    adjusted: bool                 # True if axis was changed from default


@dataclass
class CollisionAvoidanceResult:
    """Result of collision avoidance for all path points."""
    points: np.ndarray                      # (N, 3) path points (real space)
    orientations: List[ToolOrientation]     # per-point orientation
    collision_resolved: np.ndarray          # (N,) bool – resolved by adjustment
    collision_unresolved: np.ndarray        # (N,) bool – cannot resolve


# ---------------------------------------------------------------------------
# Cone candidate generation
# ---------------------------------------------------------------------------

def _generate_cone_candidates(
    default_axis: torch.Tensor,
    half_angle_deg: float,
    n_candidates: int,
    device,
    dtype,
    enforce_upper_hemisphere: bool = True,
) -> torch.Tensor:
    """Generate *n_candidates* unit vectors uniformly spaced INSIDE a cone (spherical cap).

    This function creates a set of alternative tool orientations using a 
    Fibonacci spiral to ensure uniform distribution over the spherical cap area.
    The points are generated from the center (smallest tilt) to the edge (max tilt).

    Args:
        default_axis: (3,) unit vector – the central axis of the cone.
        half_angle_deg: The maximum opening angle of the cone (degrees).
        n_candidates: How many directions to sample inside the cone area.
        enforce_upper_hemisphere: If True, filter out candidates that point downwards (z < 0).

    Returns:
        (K, 3) tensor of valid unit vectors, where K <= n_candidates.
    """
    if n_candidates == 0:
        return torch.empty((0, 3), dtype=dtype, device=device)
        
    half_rad = math.radians(half_angle_deg)
    cos_h = math.cos(half_rad)
    
    # 1. 生成从 0 到 n_candidates-1 的索引
    indices = torch.arange(n_candidates, device=device, dtype=dtype)
    
    # 2. 计算每个点在局部 Z 轴上的投影高度
    # fraction 范围在 (0, 1) 之间。加上 0.5 是为了避免点完全重合在极点(中心)或边界上
    fraction = (indices + 0.5) / n_candidates
    # Z 值从 1.0 (不倾斜) 递减到 cos_h (最大倾斜角)
    z = 1.0 - fraction * (1.0 - cos_h)
    
    # 3. 计算每个点在局部 XY 平面上的投影半径
    r = torch.sqrt(1.0 - z * z)
    
    # 4. 黄金角 (Golden Angle)，约为 137.5 度 (2.39996 弧度)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    # 每个点的方位角
    theta = indices * golden_angle
    
    # 5. 构建局部坐标系下的候选向量
    canonical = torch.stack([
        r * torch.cos(theta),
        r * torch.sin(theta),
        z
    ], dim=-1)  # (n_candidates, 3)

    # 6. 将局部坐标系的 Z 轴旋转对齐到实际的 'default_axis'
    # R = _align_z_to_v_batch(default_axis.unsqueeze(0))  # (1, 3, 3)
    R = _align_z_to_u_upper_half_robust(default_axis.unsqueeze(0))
    rotated = (R[0] @ canonical.T).T  # (n_candidates, 3)
    
    # 确保数值稳定性 (归一化)
    candidates = rotated / (rotated.norm(dim=-1, keepdim=True) + 1e-12)

    if enforce_upper_hemisphere:
        # 仅保留 Z >= 0 的向量 (指向上方或水平)
        mask = candidates[:, 2] >= -1e-6
        candidates = candidates[mask]

    return candidates

def _generate_cone_candidates_batched(
    vecs: torch.Tensor,
    half_angle_deg: float,
    n_candidates: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    基于 Fibonacci 螺旋的批量化圆锥内部采样。
    参数:
        vecs: (M, 3) 中心轴向量张量。
    返回:
        (M, n_candidates, 3) 每个中心轴对应的均匀分布采样向量。
    """
    M = vecs.shape[0]
    if M == 0 or n_candidates == 0:
        return torch.empty((M, n_candidates, 3), dtype=dtype, device=device)

    # 1. 计算角度边界
    half_rad = math.radians(half_angle_deg)
    cos_h = math.cos(half_rad)
    
    # 2. 在局部坐标系生成通用的 Fibonacci 采样模板 (n_candidates, 3)
    indices = torch.arange(n_candidates, device=device, dtype=dtype)
    
    # 黄金角
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = indices * golden_angle
    
    # Z 从 1.0 (中心) 到 cos_h (边缘) 均匀分布
    fraction = (indices + 0.5) / n_candidates
    z = 1.0 - fraction * (1.0 - cos_h)
    
    # 根据 Z 计算半径 r
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    
    # 构建局部坐标系的候选向量 (n_candidates, 3)
    canonical = torch.stack([
        r * torch.cos(theta),
        r * torch.sin(theta),
        z
    ], dim=-1)

    # 3. 计算旋转矩阵 (M, 3, 3)
    R = _align_z_to_v_batch(vecs)

    # 4. 批量旋转
    # 将模板扩展为 (M, 3, n_candidates) 以便 bmm
    canonical_T = canonical.T.unsqueeze(0).expand(M, 3, n_candidates)
    
    # (M, 3, 3) @ (M, 3, n_candidates) -> (M, 3, n_candidates)
    rotated = torch.bmm(R, canonical_T)
    
    # 转置为 (M, n_candidates, 3)
    candidates = rotated.transpose(1, 2)


    # --- E1. 强制包含一个正 Z 轴向量 (M, 1, 3) ---
    z_axis = torch.zeros((M, 1, 3), device=device, dtype=dtype)
    z_axis[:, :, 2] = 1.0  # 绝对的 (0, 0, 1)

    # --- E2. 在 Z 轴附近生成受限角度的随机单位向量 (M, n_extra, 3) ---
    n_extra = 4  # 额外再加 4 个随机的，总共增加 5 个
    
    # 随机弧度和受限的高度
    rand_theta = torch.rand((M, n_extra), device=device, dtype=dtype) * 2 * math.pi
    # 随机 fraction 从 0 到 1，决定了在 half_angle_deg 范围内的分布
    rand_fraction = torch.rand((M, n_extra), device=device, dtype=dtype)
    
    z_rand = 1.0 - rand_fraction * (1.0 - cos_h)
    r_rand = torch.sqrt(torch.clamp(1.0 - z_rand * z_rand, min=0.0))
    
    extra_rand_vecs = torch.stack([
        r_rand * torch.cos(rand_theta),
        r_rand * torch.sin(rand_theta),
        z_rand
    ], dim=-1)

    # --- E3. 拼接并执行最后的统一归一化 ---
    # 将原有的 candidates、正 Z 轴、以及随机 Z 轴向量合并
    candidates = torch.cat([candidates, z_axis, extra_rand_vecs], dim=1)

    # 5. 归一化确保单位长度
    candidates = candidates / (candidates.norm(dim=-1, keepdim=True) + 1e-12)

    return candidates

# ---------------------------------------------------------------------------
# Material query (thin wrapper – avoids circular import)
# ---------------------------------------------------------------------------

def _query_exists_flat(model, check_func, flat_pts, time, max_time,
                       aabb_min, aabb_max, query_batch_size,
                       *, use_amp: bool = False):
    """Boolean existence mask for a flat tensor of 3-D points.

    Args:
        model: Trained neural model.
        check_func: In-model occupancy helper callable.
        flat_pts: Tensor ``(N, 3)`` in model coordinates.
        time: Query process time.
        max_time: Global max process time.
        aabb_min: AABB minimum corner tensor ``(3,)``.
        aabb_max: AABB maximum corner tensor ``(3,)``.
        query_batch_size: Inference batch size for flattened points.

    Returns:
        Boolean tensor ``(N,)`` where ``True`` means material exists.
    """
    from .collision_checker import _query_material_exists
    return _query_material_exists(
        model, check_func, flat_pts, time, max_time,
        aabb_min, aabb_max, query_batch_size,
        use_amp=use_amp,
    )


# ---------------------------------------------------------------------------
# AM collision avoidance
# ---------------------------------------------------------------------------

def avoid_am_collisions(
    ctx,
    pts_model: torch.Tensor,
    collision_mask: torch.Tensor,
    layer_time: float,
    *,
    cone_half_angle_deg: float = 15.0,
    n_candidates: int = 16,
    n_cone: int = 100,
    n_cylinder: int = 100,
    n_cone_surface: int = 0,
    n_cyl_surface: int = 0,
    query_batch_size: int = 32768,
    m_chunk: int = 32768,
    collision_use_amp: bool = False,
) -> tuple:
    """Try alternative AM tool axes for colliding points.

    Vectorised algorithm (replaces per-point nested loop):

    For each candidate axis on a cone around +Z (in ascending tilt order):
      - Batch-test ALL still-unresolved colliding points simultaneously.
      - Mark safe points as resolved with this candidate and drop them from
        the active set.
      - Repeat until all points are resolved or all candidates exhausted.

    This transforms O(N * K) sequential GPU queries into O(K) batched queries,
    drastically improving GPU utilisation.

    Args:
        ctx: Postprocess context.
        pts_model: Tensor ``(N, 3)`` path points in model space.
        collision_mask: Bool mask ``(N,)`` from default-orientation collision check.
        layer_time: Layer time for material query.
        cone_half_angle_deg: Max tilt angle around default axis.
        n_candidates: Number of candidate directions per point.
        n_cone: AM cone sample count per candidate.
        n_cylinder: AM cylinder sample count per candidate.
        query_batch_size: Inference batch size in material existence checks.
        m_chunk: Maximum number of colliding points to process in one GPU call.
            Caps peak VRAM to ``m_chunk × n_per_pt × 3 × 4`` bytes per thread.
            Default 32768; reduce if multi-thread OOM is observed.

    Returns:
        (resolved_mask, unresolved_mask, orientations_list)

        - resolved_mask: ``(N,)`` bool numpy array.
        - unresolved_mask: ``(N,)`` bool numpy array.
        - orientations_list: ``List[ToolOrientation]`` aligned with input points.
    """
    dev, dt = pts_model.device, pts_model.dtype
    manu = ctx.manu_config
    aabb_min = torch.as_tensor(ctx.spaceBox[0], dtype=dt, device=dev)
    aabb_max = torch.as_tensor(ctx.spaceBox[1], dtype=dt, device=dev)
    cone_h = manu['AMConeHeight']
    cone_a = manu['AMConeHalfAngle']
    am_collision_margin = manu['AMCollisionMargin']
    z_top = float(aabb_max[2])

    N = pts_model.shape[0]
    default_axis = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dt)

    resolved = np.zeros(N, dtype=bool)
    unresolved = np.zeros(N, dtype=bool)

    collision_np = (collision_mask.cpu().numpy()
                    if torch.is_tensor(collision_mask)
                    else np.asarray(collision_mask, dtype=bool))

    # Pre-fill all N orientations with the safe default (+Z axis).
    orientations: List[ToolOrientation] = [
        ToolOrientation(
            am_axis=np.array([0., 0., 1.]),
            sm_normal_vec=np.zeros(3),
            sm_tool_vec=np.zeros(3),
            adjusted=False,
        )
        for _ in range(N)
    ]

    # Candidate axes are the same for every AM point (default axis is always +Z).
    candidates = _generate_cone_candidates(
        default_axis, cone_half_angle_deg, n_candidates, dev, dt,
        enforce_upper_hemisphere=True,
    )
    n_actual_candidates = candidates.shape[0]

    # still_unresolved: colliding points that no candidate has cleared yet.
    still_unresolved = collision_np.copy()

    for ci in range(n_actual_candidates):
        idx = np.where(still_unresolved)[0]   # global indices of active points
        if len(idx) == 0:
            break

        M = len(idx)
        axis_i = candidates[ci]
        axis_np = axis_i.cpu().numpy()

        # Chunked GPU query: cap peak VRAM to m_chunk * n_per_pt * 3 * 4 bytes.
        safe_gpu = torch.zeros(M, dtype=torch.bool, device=dev)
        for cs in range(0, M, m_chunk):
            ce = min(cs + m_chunk, M)
            bp_c = pts_model[idx[cs:ce]]                              # (Mc, 3)
            axis_c = axis_i.unsqueeze(0).expand(ce - cs, -1)         # (Mc, 3)

            tool_pts = sample_am_tool(
                bp_c, cone_h, cone_a, am_collision_margin,
                z_top,
                n_cone=n_cone, n_cylinder=n_cylinder,
                n_cone_surface=n_cone_surface, n_cylinder_surface=n_cyl_surface,
                tool_axis=axis_c,
            )  # (Mc, n_per_pt, 3)
            n_per_pt = tool_pts.shape[1]
            flat = tool_pts.reshape(-1, 3)

            exists = _query_exists_flat(
                ctx.model, ctx.check_func, flat,
                layer_time, ctx.max_time,
                aabb_min, aabb_max, query_batch_size,
                use_amp=collision_use_amp,
            )
            safe_gpu[cs:ce] = ~exists.reshape(ce - cs, n_per_pt).any(dim=1)
            del tool_pts, flat, exists

        safe = safe_gpu.cpu().numpy()
        safe_global = idx[safe]
        resolved[safe_global] = True
        for g in safe_global:
            orientations[g] = ToolOrientation(
                am_axis=axis_np.copy(),
                sm_normal_vec=np.zeros(3),
                sm_tool_vec=np.zeros(3),
                adjusted=True,
            )
        still_unresolved[safe_global] = False

    # Anything still unresolved after all candidates is truly unresolvable.
    unresolved[still_unresolved & collision_np] = True

    return resolved, unresolved, orientations


# ---------------------------------------------------------------------------
# SM collision avoidance
# ---------------------------------------------------------------------------

# def avoid_sm_collisions(
#     ctx,
#     pts_model: torch.Tensor,
#     collision_mask,
#     layer_time: float,
#     *,
#     hemi_collision_mask=None,
#     shank_holder_collision_mask=None,
#     cone_half_angle_deg: float = 15.0,
#     n_candidates: int = 16,
#     sm_collision_margin_model: float = 0.0,
#     n_hemisphere: int = 50,
#     n_shank: int = 80,
#     n_holder: int = 70,
#     query_batch_size: int = 32768,
#     m_chunk: int = 32768,
#     collision_use_amp: bool = False,
# ) -> tuple:
#     """Try alternative SM tool axes for colliding points.

#     Two-phase avoidance strategy (same logic as before), now vectorised:
#     each phase iterates over candidate *indices* and processes all eligible
#     points in a single batched GPU query per index instead of one-point-at-a-time.

#     Phase 1 – vary ``tool_vec`` (skip hemisphere-only points):
#       For each candidate index ci, batch-test all still-unresolved Phase-1
#       points with their respective ci-th tool_vec candidate.

#     Phase 2 – vary ``normal_vec`` (skip shank/holder-only points):
#       For each nci, batch-test all still-unresolved Phase-2 points with their
#       ci-th normal_vec candidate + original tool_vec. Points where only the
#       shank/holder remains colliding enter an inner tool_vec search (also
#       batched per tci).

#     Selection logic (same as before):

#     +-----------------------+---------+---------+
#     | Collision region      | Phase 1 | Phase 2 |
#     +=======================+=========+=========+
#     | Hemisphere only       | skipped | run     |
#     +-----------------------+---------+---------+
#     | Shank / holder only   | run     | skipped |
#     +-----------------------+---------+---------+
#     | Both (mixed)          | run     | if fail |
#     +-----------------------+---------+---------+

#     Args:
#         ctx: Postprocess context.
#         pts_model: Tensor ``(N, 3)`` path points in model space.
#         collision_mask: Bool mask ``(N,)`` from default-orientation collision check.
#         layer_time: Layer time for material query.
#         hemi_collision_mask: ``(N,)`` bool array – True when the hemisphere
#             region of the tool has a collision.  ``None`` → treat every
#             colliding point as involving the hemisphere (runs Phase 2 for all).
#         shank_holder_collision_mask: ``(N,)`` bool array – True when the
#             shank/holder region of the tool has a collision.  ``None`` →
#             treat every colliding point as involving shank/holder (runs Phase 1
#             for all).
#         cone_half_angle_deg: Max tilt angle around per-point default axis.
#         n_candidates: Number of candidate directions per colliding point.
#         sm_collision_margin_model: SM shrink margin in model-space units.
#         n_hemisphere: Hemisphere sample count per candidate.
#         n_shank: Shank sample count per candidate.
#         n_holder: Holder sample count per candidate.
#         query_batch_size: Inference batch size in material existence checks.

#     Returns:
#         (resolved_mask, unresolved_mask, orientations_list)

#         - resolved_mask: ``(N,)`` bool numpy array.
#         - unresolved_mask: ``(N,)`` bool numpy array.
#         - orientations_list: ``List[ToolOrientation]`` aligned with input points.
#     """
#     dev, dt = pts_model.device, pts_model.dtype
#     manu = ctx.manu_config
#     aabb_min = torch.as_tensor(ctx.spaceBox[0], dtype=dt, device=dev)
#     aabb_max = torch.as_tensor(ctx.spaceBox[1], dtype=dt, device=dev)

#     N = pts_model.shape[0]
#     normal_vec, tool_vec = query_sm_orientation(ctx.model, pts_model)

#     resolved = np.zeros(N, dtype=bool)
#     unresolved = np.zeros(N, dtype=bool)

#     collision_np = (collision_mask.cpu().numpy()
#                     if torch.is_tensor(collision_mask)
#                     else np.asarray(collision_mask, dtype=bool))

#     _tool_kw = dict(
#         sm_tool_diameter=manu['SMToolParas']['SMToolDiameter'],
#         sm_tool_length=manu['SMToolParas']['SMToolLength'],
#         sm_holder_diameter=manu['SMToolParas']['SMHolderDiameter'],
#         sm_holder_length=manu['SMToolParas']['SMHolderLength'],
#         collision_margin=sm_collision_margin_model,
#         n_hemisphere=n_hemisphere,
#         n_shank=n_shank,
#         n_holder=n_holder,
#     )

#     # Cache per-point numpy vectors for orientation construction.
#     normal_np = normal_vec.detach().cpu().numpy()   # (N, 3)
#     tool_np   = tool_vec.detach().cpu().numpy()     # (N, 3)

#     # Pre-fill all N orientations with their respective default axes.
#     orientations: List[ToolOrientation] = [
#         ToolOrientation(
#             am_axis=np.zeros(3),
#             sm_normal_vec=normal_np[i],
#             sm_tool_vec=tool_np[i],
#             adjusted=False,
#         )
#         for i in range(N)
#     ]

#     # Classify collision type per point.
#     hemi_coll = (np.asarray(hemi_collision_mask, dtype=bool)
#                  if hemi_collision_mask is not None else collision_np.copy())
#     shank_holder_coll = (np.asarray(shank_holder_collision_mask, dtype=bool)
#                          if shank_holder_collision_mask is not None else collision_np.copy())

#     only_hemi_mask  = hemi_coll & ~shank_holder_coll   # hemisphere-only → Phase 2 only
#     only_shank_mask = ~hemi_coll & collision_np         # shank/holder-only → Phase 1 only

#     # ------------------------------------------------------------------
#     # Phase 1: vary tool_vec (batch over candidate index ci).
#     # Skipped for hemisphere-only points.
#     # ------------------------------------------------------------------
#     phase1_idx = np.where(collision_np & ~only_hemi_mask)[0]

#     if len(phase1_idx) > 0:
#         # Pre-generate tv candidates per eligible point (each around its own tool_vec).
#         tv_cands_list = [
#             _generate_cone_candidates(
#                 tool_vec[g], cone_half_angle_deg, n_candidates, dev, dt,
#                 enforce_upper_hemisphere=True,
#             )
#             for g in phase1_idx
#         ]
#         max_K1 = max(c.shape[0] for c in tv_cands_list)
#         phase1_still = np.ones(len(phase1_idx), dtype=bool)  # local active mask

#         for ci in range(max_K1):
#             has_ci = np.array([ci < c.shape[0] for c in tv_cands_list])
#             local_sel = np.where(phase1_still & has_ci)[0]   # positions in phase1_idx
#             if len(local_sel) == 0:
#                 break

#             global_sel = phase1_idx[local_sel]               # positions in pts_model
#             M = len(local_sel)

#             # Chunked GPU query to cap peak VRAM per candidate iteration.
#             safe_gpu = torch.zeros(M, dtype=torch.bool, device=dev)
#             for cs in range(0, M, m_chunk):
#                 ce = min(cs + m_chunk, M)
#                 chunk_ls  = local_sel[cs:ce]
#                 chunk_gs  = global_sel[cs:ce]
#                 bp_c      = pts_model[chunk_gs]               # (Mc, 3)
#                 nv_c      = normal_vec[chunk_gs]              # (Mc, 3)
#                 tv_c      = torch.stack([tv_cands_list[j][ci]
#                                          for j in chunk_ls])  # (Mc, 3)

#                 tool_pts = sample_sm_tool(bp_c, ctx.model,
#                                           override_normal_vec=nv_c,
#                                           override_tool_vec=tv_c,
#                                           **_tool_kw)
#                 flat   = tool_pts.reshape(-1, 3)
#                 exists = _query_exists_flat(
#                     ctx.model, ctx.check_func, flat,
#                     layer_time, ctx.max_time,
#                     aabb_min, aabb_max, query_batch_size,
#                     use_amp=collision_use_amp,
#                 )
#                 safe_gpu[cs:ce] = ~exists.reshape(ce - cs, -1).any(dim=1)
#                 del tool_pts, flat, exists
#             safe = safe_gpu.cpu().numpy()

#             for lj in np.where(safe)[0]:
#                 li = local_sel[lj]
#                 g  = phase1_idx[li]
#                 resolved[g] = True
#                 orientations[g] = ToolOrientation(
#                     am_axis=np.zeros(3),
#                     sm_normal_vec=normal_np[g],
#                     sm_tool_vec=tv_cands_list[li][ci].cpu().numpy(),
#                     adjusted=True,
#                 )
#                 phase1_still[li] = False

#     # ------------------------------------------------------------------
#     # Phase 2: vary normal_vec (batch over candidate index nci).
#     # Skipped for shank/holder-only points and for points already resolved.
#     # ------------------------------------------------------------------
#     phase2_eligible = collision_np & ~only_shank_mask & ~resolved
#     phase2_idx = np.where(phase2_eligible)[0]

#     if len(phase2_idx) > 0:
#         # Pre-generate nv candidates and inner tv candidates per eligible point.
#         nv_cands_list = [
#             _generate_cone_candidates(
#                 normal_vec[g], cone_half_angle_deg, n_candidates, dev, dt,
#                 enforce_upper_hemisphere=True,
#             )
#             for g in phase2_idx
#         ]
#         # Inner tv candidates depend on the point's tool_vec, not on nci.
#         inner_tv_cands_list = [
#             _generate_cone_candidates(
#                 tool_vec[g], cone_half_angle_deg, n_candidates, dev, dt,
#                 enforce_upper_hemisphere=True,
#             )
#             for g in phase2_idx
#         ]
#         max_K_nci = max(c.shape[0] for c in nv_cands_list)
#         phase2_still = np.ones(len(phase2_idx), dtype=bool)  # local active mask

#         for nci in range(max_K_nci):
#             has_nci = np.array([nci < c.shape[0] for c in nv_cands_list])
#             local_sel = np.where(phase2_still & has_nci)[0]  # positions in phase2_idx
#             if len(local_sel) == 0:
#                 break

#             global_sel = phase2_idx[local_sel]
#             M          = len(local_sel)

#             # Step (a): probe with candidate normal_vec + original tool_vec.
#             # Chunked to bound peak VRAM = m_chunk * n_per_pt * 3 * 4 bytes.
#             fully_safe_gpu = torch.zeros(M, dtype=torch.bool, device=dev)
#             hemi_still_gpu = torch.zeros(M, dtype=torch.bool, device=dev)
#             for cs in range(0, M, m_chunk):
#                 ce           = min(cs + m_chunk, M)
#                 chunk_ls     = local_sel[cs:ce]
#                 chunk_gs     = global_sel[cs:ce]
#                 bp_c         = pts_model[chunk_gs]
#                 cand_nv_c    = torch.stack([nv_cands_list[j][nci] for j in chunk_ls])
#                 tv_orig_c    = tool_vec[chunk_gs]

#                 tool_pts = sample_sm_tool(bp_c, ctx.model,
#                                           override_normal_vec=cand_nv_c,
#                                           override_tool_vec=tv_orig_c,
#                                           **_tool_kw)
#                 n_per_pt  = tool_pts.shape[1]
#                 flat      = tool_pts.reshape(-1, 3)
#                 exists    = _query_exists_flat(
#                     ctx.model, ctx.check_func, flat,
#                     layer_time, ctx.max_time,
#                     aabb_min, aabb_max, query_batch_size,
#                     use_amp=collision_use_amp,
#                 )
#                 exists_2d_c = exists.reshape(ce - cs, n_per_pt)
#                 fully_safe_gpu[cs:ce] = ~exists_2d_c.any(dim=1)
#                 hemi_still_gpu[cs:ce] = exists_2d_c[:, :n_hemisphere].any(dim=1)
#                 del tool_pts, flat, exists, exists_2d_c

#             fully_safe = fully_safe_gpu.cpu().numpy()
#             hemi_still = hemi_still_gpu.cpu().numpy()
#             shank_only = ~fully_safe & ~hemi_still   # ball clear, shank/holder still hits

#             # Mark fully-safe points as resolved.
#             for lj in np.where(fully_safe)[0]:
#                 li = local_sel[lj]
#                 g  = phase2_idx[li]
#                 resolved[g] = True
#                 orientations[g] = ToolOrientation(
#                     am_axis=np.zeros(3),
#                     sm_normal_vec=nv_cands_list[li][nci].cpu().numpy(),
#                     sm_tool_vec=tool_np[g],
#                     adjusted=True,
#                 )
#                 phase2_still[li] = False

#             # Inner search: ball is clear but shank/holder still collides.
#             # Search over tool_vec candidates keeping this nci's normal_vec.
#             inner_local_sel  = local_sel[shank_only]    # positions in phase2_idx
#             inner_global_sel = global_sel[shank_only]   # positions in pts_model

#             if len(inner_local_sel) > 0:
#                 max_K_tci  = max(inner_tv_cands_list[j].shape[0]
#                                  for j in inner_local_sel)
#                 inner_still = np.ones(len(inner_local_sel), dtype=bool)

#                 for tci in range(max_K_tci):
#                     has_tci = np.array([tci < inner_tv_cands_list[j].shape[0]
#                                         for j in inner_local_sel])
#                     inner_active = np.where(inner_still & has_tci)[0]  # positions in inner_local_sel
#                     if len(inner_active) == 0:
#                         break

#                     act_li = inner_local_sel[inner_active]    # positions in phase2_idx
#                     act_g  = inner_global_sel[inner_active]   # positions in pts_model
#                     M2     = len(inner_active)

#                     # Chunked inner query.
#                     safe2_gpu = torch.zeros(M2, dtype=torch.bool, device=dev)
#                     for cs2 in range(0, M2, m_chunk):
#                         ce2      = min(cs2 + m_chunk, M2)
#                         bp2      = pts_model[act_g[cs2:ce2]]
#                         cand_nv2 = torch.stack([nv_cands_list[li][nci]
#                                                 for li in act_li[cs2:ce2]])
#                         tv_cand2 = torch.stack([inner_tv_cands_list[li][tci]
#                                                 for li in act_li[cs2:ce2]])

#                         tool_pts2 = sample_sm_tool(bp2, ctx.model,
#                                                    override_normal_vec=cand_nv2,
#                                                    override_tool_vec=tv_cand2,
#                                                    **_tool_kw)
#                         flat2   = tool_pts2.reshape(-1, 3)
#                         exists2 = _query_exists_flat(
#                             ctx.model, ctx.check_func, flat2,
#                             layer_time, ctx.max_time,
#                             aabb_min, aabb_max, query_batch_size,
#                             use_amp=collision_use_amp,
#                         )
#                         safe2_gpu[cs2:ce2] = ~exists2.reshape(ce2 - cs2, -1).any(dim=1)
#                         del tool_pts2, flat2, exists2
#                     safe2 = safe2_gpu.cpu().numpy()

#                     for ia_j in np.where(safe2)[0]:
#                         # inner_active[ia_j] → position in inner_local_sel / inner_still
#                         ia_pos = inner_active[ia_j]
#                         ia_li  = act_li[ia_j]    # position in phase2_idx
#                         ia_g   = act_g[ia_j]     # position in pts_model
#                         resolved[ia_g] = True
#                         orientations[ia_g] = ToolOrientation(
#                             am_axis=np.zeros(3),
#                             sm_normal_vec=nv_cands_list[ia_li][nci].cpu().numpy(),
#                             sm_tool_vec=inner_tv_cands_list[ia_li][tci].cpu().numpy(),
#                             adjusted=True,
#                         )
#                         phase2_still[ia_li] = False
#                         inner_still[ia_pos] = False

#     # Any remaining colliding points that neither phase could resolve.
#     unresolved[collision_np & ~resolved] = True

#     return resolved, unresolved, orientations


def avoid_sm_collisions(
    ctx,
    pts_model: torch.Tensor,
    collision_mask,
    layer_time: float,
    *,
    sphere_collision_mask=None,
    cone_half_angle_deg: float = 15.0,
    max_tilt_angle_deg: float = 89.0,
    n_candidates: int = 16,
    n_sphere: int = 50,
    n_shank: int = 80,
    n_holder: int = 70,
    n_sphere_surface: int = 0,
    n_shank_surface: int = 0,
    n_holder_surface: int = 0,
    n_holder_bottom_surface: int = 0,
    query_batch_size: int = 32768,
    m_chunk: int = 32768,
    collision_use_amp: bool = False,
) -> tuple:
    dev, dt = pts_model.device, pts_model.dtype
    manu = ctx.manu_config
    aabb_min = torch.as_tensor(ctx.spaceBox[0], dtype=dt, device=dev)
    aabb_max = torch.as_tensor(ctx.spaceBox[1], dtype=dt, device=dev)

    N = pts_model.shape[0]
    normal_vec, tool_vec = query_sm_orientation(ctx.model, pts_model)

    # 标准化输入掩码到 GPU
    coll_gpu = (collision_mask.clone().detach().to(dtype=torch.bool, device=dev) 
                if torch.is_tensor(collision_mask) 
                else torch.tensor(collision_mask, dtype=torch.bool, device=dev))
    
    sphere_coll_gpu = (torch.tensor(sphere_collision_mask, dtype=torch.bool, device=dev) 
                       if sphere_collision_mask is not None else coll_gpu.clone())
    
    # shank_coll_gpu = (torch.tensor(shank_holder_collision_mask, dtype=torch.bool, device=dev) 
    #                   if shank_holder_collision_mask is not None else coll_gpu.clone())

    resolved = torch.zeros(N, dtype=torch.bool, device=dev)
    unresolved = torch.zeros(N, dtype=torch.bool, device=dev)
    
    # 全局维护最终的安全姿态和修改标记 (In-place 更新)
    out_normal_vec = normal_vec.clone()
    out_tool_vec = tool_vec.clone()
    adjusted = torch.zeros(N, dtype=torch.bool, device=dev)

    only_shank_mask = ~sphere_coll_gpu & coll_gpu

    # Fallback: if callers pass 0, mirror the same auto-formula used in sample_sm_tool.
    if n_sphere_surface <= 0:
        n_sphere_surface = max(8, n_sphere // 4)
    if n_shank_surface <= 0:
        n_shank_surface = max(8, n_shank // 4)
    if n_holder_surface <= 0:
        n_holder_surface = max(8, n_holder // 4)
    if n_holder_bottom_surface <= 0:
        # Bottom disk is a frequent leakage spot; enforce high-density sampling.
        n_holder_bottom_surface = max(128, 8 * n_holder_surface)

    _tool_kw = dict(
        sm_tip_diameter=manu['SMToolParas']['SMTipDiameter'],
        sm_shank_diameter=manu['SMToolParas']['SMShankDiameter'],
        sm_tool_length=manu['SMToolParas']['SMToolLength'],
        sm_holder_diameter=manu['SMToolParas']['SMHolderDiameter'],
        sm_holder_length=manu['SMToolParas']['SMHolderLength'],
        collision_margin=manu['SMToolParas']['SMCollisionMargin'],
        n_sphere=n_sphere, n_shank=n_shank, n_holder=n_holder,
        n_sphere_surface=n_sphere_surface, n_shank_surface=n_shank_surface, n_holder_surface=n_holder_surface,
        n_holder_bottom_surface=n_holder_bottom_surface,
    )
    # Split index: layout is [sphere_vol, sphere_surf, shank_vol, shank_surf, holder_vol, holder_surf, holder_bottom_surf]
    _n_sphere_total = n_sphere + n_sphere_surface   # index of first shank sample

    cos_theta = math.cos(math.radians(max_tilt_angle_deg))
    sin_theta = math.sin(math.radians(max_tilt_angle_deg))

    # ==================================================================
    # 内部工具：纯 GPU 批处理碰撞查询 (收敛 m_chunk 循环，减少代码嵌套)
    # ==================================================================
    def _check_collisions_batched(bp: torch.Tensor, nv: torch.Tensor, tv: torch.Tensor, check_target: str) -> torch.Tensor:
        """
        check_target: 'shank_only' (跳过所有球体采样) | 'sphere_only' (仅测球体采样)
        Sample layout (from _sample_sm_canonical):
                    [sphere_vol | sphere_surf | shank_vol | shank_surf | holder_vol | holder_surf | holder_bottom_surf]
           ← _n_sphere_total →|←           rest                                       →
        返回 shape (M,) 的 boolean tensor, True 表示安全
        """
        M = bp.shape[0]
        safe_gpu = torch.zeros(M, dtype=torch.bool, device=dev)
        for cs in range(0, M, m_chunk):
            ce = min(cs + m_chunk, M)
            tool_pts = sample_sm_tool(bp[cs:ce], ctx.model, override_normal_vec=nv[cs:ce], override_tool_vec=tv[cs:ce], **_tool_kw)
            
            if check_target == 'shank_only':
                # Skip all sphere-related samples (volume + surface)
                flat = tool_pts[:, _n_sphere_total:, :].reshape(-1, 3)
                pts_per_tool = tool_pts.shape[1] - _n_sphere_total
            else:  # sphere_only – include both volume and surface sphere samples
                flat = tool_pts[:, :_n_sphere_total, :].reshape(-1, 3)
                pts_per_tool = _n_sphere_total
                
            exists = _query_exists_flat(
                ctx.model, ctx.check_func, flat, layer_time, ctx.max_time,
                aabb_min, aabb_max, query_batch_size, use_amp=collision_use_amp,
            )
            safe_gpu[cs:ce] = ~exists.reshape(ce - cs, pts_per_tool).any(dim=1)
        return safe_gpu

    # ==================================================================
    # 内部工具：向量拉回计算 (保持纯 Tensor 计算)
    # ==================================================================
    def _compute_ref_tv(c_nv: torch.Tensor, t_orig: torch.Tensor) -> torch.Tensor:
        """
        纯向量化 (Pure Vectorized) 的参考刀轴计算，无 CPU-GPU 同步，无条件分支。
        """
        # 1. 计算原始向量与法向量的夹角余弦
        dot_v = (c_nv * t_orig).sum(dim=-1, keepdim=True)
        
        # ---------------------------------------------------------
        # Step A: 全局计算 Slerp 拉回向量 (向 theta 边界靠拢)
        # ---------------------------------------------------------
        u = t_orig - dot_v * c_nv
        u_norm = torch.norm(u, dim=-1, keepdim=True)
        # 安全归一化：如果 t_orig 与 c_nv 共线，u_norm 为 0，避免除零错误
        u_safe = torch.where(u_norm > 1e-6, u / u_norm, torch.zeros_like(u))
        
        # 拉回到夹角刚好为 theta 的位置
        pulled = c_nv * cos_theta + u_safe * sin_theta
        
        # ---------------------------------------------------------
        # Step B: 全局计算保底策略 (Fallback) 向量
        # 当拉回后的向量 Z <= 0 时启用
        # ---------------------------------------------------------
        # 候选 B1: 投影 cand_nv 到 XY 平面
        proj_nv = c_nv.clone()
        proj_nv[:, 2] = 0.0
        p_norm = torch.norm(proj_nv, dim=-1, keepdim=True)
        
        # [物理连贯性优化] 如果 cand_nv 垂直向下导致投影极小，
        # 则使用原刀轴 t_orig 的 XY 投影方向，避免武断的硬编码导致方向突变
        proj_t = t_orig.clone()
        proj_t[:, 2] = 0.0
        pt_norm = torch.norm(proj_t, dim=-1, keepdim=True)
        # 极小概率下 t_orig 也是垂直的，才使用硬编码兜底
        fallback_dir = torch.where(pt_norm > 1e-6, proj_t / pt_norm, torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt))
        
        proj_nv_safe = torch.where(p_norm > 1e-6, proj_nv / p_norm, fallback_dir)
        
        # 决策 B: 如果 c_nv 朝上，保底用 c_nv；否则用 XY 投影
        c_nv_z_gt_0 = c_nv[:, 2:3] > 0
        fallback_tv = torch.where(c_nv_z_gt_0, c_nv, proj_nv_safe)
        
        # ---------------------------------------------------------
        # Step C: 层级决断 (Select)
        # ---------------------------------------------------------
        # 决策 1: 拉回的向量是否合法 (Z > 0) ? 合法用拉回的，不合法用保底的
        pulled_z_gt_0 = pulled[:, 2:3] > 0
        adjusted_tv = torch.where(pulled_z_gt_0, pulled, fallback_tv)
        
        # 决策 2: 原始向量是否本来就合法 ? 合法直接保留原向量，否则用调整后的
        is_valid_orig = (t_orig[:, 2:3] > 0) & (dot_v >= cos_theta - 1e-6)
        
        return torch.where(is_valid_orig, t_orig, adjusted_tv)

    # ==================================================================
    # Phase 1: 仅处理刀柄/刀夹碰撞
    # ==================================================================
    phase1_idx = torch.where(only_shank_mask)[0]

    if len(phase1_idx) > 0:
        # 极致优化 2: 大张量批处理生成，消除小 Tensor 列表，形如 (M, n_candidates, 3)
        tv_cands_batch = _generate_cone_candidates_batched(
            tool_vec[phase1_idx], cone_half_angle_deg, n_candidates, dev, dt
        )
        phase1_still = torch.ones(len(phase1_idx), dtype=torch.bool, device=dev)
        n_p1_cands = tv_cands_batch.shape[1]  # n_candidates + 5 extra (+Z and random)

        for ci in range(n_p1_cands):
            local_sel = torch.where(phase1_still)[0]
            if len(local_sel) == 0: break

            g_idx = phase1_idx[local_sel]
            bp_c = pts_model[g_idx]
            nv_c = normal_vec[g_idx]
            tv_c = tv_cands_batch[local_sel, ci, :] # 直接切片，零显存碎片

            dot_val = (tv_c * nv_c).sum(dim=1)
            valid_mask = (tv_c[:, 2] > 0) & (dot_val >= cos_theta - 1e-6)

            if valid_mask.any():
                v_local = local_sel[valid_mask]
                v_g_idx = g_idx[valid_mask]
                v_safe = _check_collisions_batched(bp_c[valid_mask], nv_c[valid_mask], tv_c[valid_mask], 'shank_only')
                
                # 获取真正安全的索引
                safe_v_idx = torch.where(v_safe)[0]
                if len(safe_v_idx) > 0:
                    final_g_idx = v_g_idx[safe_v_idx]
                    final_l_idx = v_local[safe_v_idx]
                    
                    # 批量写回结果
                    resolved[final_g_idx] = True
                    adjusted[final_g_idx] = True
                    out_tool_vec[final_g_idx] = tv_c[valid_mask][safe_v_idx]
                    phase1_still[final_l_idx] = False

    # ==================================================================
    # Phase 2: 球体碰撞 + Phase 1 未解决的点
    # ==================================================================
    phase2_idx = torch.where(coll_gpu & ~resolved)[0]

    if len(phase2_idx) > 0:
        nv_cands_batch = _generate_cone_candidates_batched(
            normal_vec[phase2_idx], cone_half_angle_deg, n_candidates, dev, dt
        )
        phase2_still = torch.ones(len(phase2_idx), dtype=torch.bool, device=dev)
        n_p2_cands = nv_cands_batch.shape[1]  # n_candidates + 5 extra (+Z and random)

        for nci in range(n_p2_cands):
            local_sel = torch.where(phase2_still)[0]
            if len(local_sel) == 0: break

            g_idx = phase2_idx[local_sel]
            bp_c = pts_model[g_idx]
            cand_nv_c = nv_cands_batch[local_sel, nci, :]
            
            # [外层] 法线 Z 分量过滤
            valid_nv_mask = cand_nv_c[:, 2] >= -sin_theta - 1e-6
            
            if valid_nv_mask.any():
                v_local = local_sel[valid_nv_mask]
                v_g_idx = g_idx[valid_nv_mask]
                v_nv = cand_nv_c[valid_nv_mask]
                v_tv_orig = tool_vec[v_g_idx]
                
                # 检测球体
                sphere_safe = _check_collisions_batched(bp_c[valid_nv_mask], v_nv, v_tv_orig, 'sphere_only')
                
                pass_idx = torch.where(sphere_safe)[0]
                if len(pass_idx) > 0:
                    i_local = v_local[pass_idx]
                    i_g_idx = v_g_idx[pass_idx]
                    i_nv = v_nv[pass_idx]
                    i_tv_orig = v_tv_orig[pass_idx]
                    
                    # [内层 Step 1] 批量计算 ref_tv
                    ref_tv = _compute_ref_tv(i_nv, i_tv_orig)
                    
                    # [内层 Step 2] 批量验证除球体外的部分
                    step2_safe = _check_collisions_batched(pts_model[i_g_idx], i_nv, ref_tv, 'shank_only')
                    
                    # 批量写回 Step 2 成功的点
                    s2_pass = torch.where(step2_safe)[0]
                    if len(s2_pass) > 0:
                        s2_g_idx = i_g_idx[s2_pass]
                        resolved[s2_g_idx] = True
                        adjusted[s2_g_idx] = True
                        out_normal_vec[s2_g_idx] = i_nv[s2_pass]
                        out_tool_vec[s2_g_idx] = ref_tv[s2_pass]
                        phase2_still[i_local[s2_pass]] = False
                    
                    # [内层 Step 3] 局部重采样 (针对 Step 2 失败的点)
                    s3_fail = torch.where(~step2_safe)[0]
                    if len(s3_fail) > 0:
                        s3_local = i_local[s3_fail]
                        s3_g_idx = i_g_idx[s3_fail]
                        s3_nv = i_nv[s3_fail]
                        s3_ref_tv = ref_tv[s3_fail]
                        
                        s3_cands_batch = _generate_cone_candidates_batched(
                            s3_ref_tv, cone_half_angle_deg, n_candidates, dev, dt
                        )
                        s3_still = torch.ones(len(s3_fail), dtype=torch.bool, device=dev)
                        n_s3_cands = s3_cands_batch.shape[1]  # n_candidates + 5 extra

                        for tci in range(n_s3_cands):
                            act_s3 = torch.where(s3_still)[0]
                            if len(act_s3) == 0: break
                            
                            c_g_idx = s3_g_idx[act_s3]
                            c_nv = s3_nv[act_s3]
                            c_tv = s3_cands_batch[act_s3, tci, :]
                            
                            dot_val = (c_tv * c_nv).sum(dim=1)
                            v_mask = (c_tv[:, 2] > 0) & (dot_val >= cos_theta - 1e-6)
                            
                            if v_mask.any():
                                act_v_mask = torch.where(v_mask)[0]
                                f_g_idx = c_g_idx[act_v_mask]
                                f_nv = c_nv[act_v_mask]
                                f_tv = c_tv[act_v_mask]
                                
                                s3_safe = _check_collisions_batched(pts_model[f_g_idx], f_nv, f_tv, 'shank_only')
                                
                                final_s3_pass = torch.where(s3_safe)[0]
                                if len(final_s3_pass) > 0:
                                    win_g_idx = f_g_idx[final_s3_pass]
                                    orig_s3_idx = act_s3[act_v_mask[final_s3_pass]]
                                    
                                    resolved[win_g_idx] = True
                                    adjusted[win_g_idx] = True
                                    out_normal_vec[win_g_idx] = f_nv[final_s3_pass]
                                    out_tool_vec[win_g_idx] = f_tv[final_s3_pass]
                                    phase2_still[s3_local[orig_s3_idx]] = False
                                    s3_still[orig_s3_idx] = False

    unresolved[coll_gpu & ~resolved] = True

    # ==================================================================
    # 极致优化 1: 延迟对象构建。仅在退出前统一拉取到 CPU 并构建对象。
    # ==================================================================
    resolved_np = resolved.cpu().numpy()
    unresolved_np = unresolved.cpu().numpy()
    
    # 将修改结果批量迁移至 CPU
    adj_np = adjusted.cpu().numpy()
    out_nv_np = out_normal_vec.cpu().numpy()
    out_tv_np = out_tool_vec.cpu().numpy()
    orig_nv_np = normal_vec.cpu().numpy()

    # 高速单次循环生成对象列表
    orientations = [
        ToolOrientation(
            am_axis=np.zeros(3),
            sm_normal_vec=out_nv_np[i] if adj_np[i] else orig_nv_np[i],
            sm_tool_vec=out_tv_np[i],
            adjusted=bool(adj_np[i]),
        )
        for i in range(N)
    ]

    return resolved_np, unresolved_np, orientations

# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def check_and_avoid_collisions(
    ctx,
    path_points_real: np.ndarray,
    layer_time: float,
    layer_type: str,
    *,
    batch_size: int = 4096,
    n_tool_samples: int = 900,
    query_batch_size: int = 32768,
    avoidance_cone_half_angle: float = 15.0,
    avoidance_n_candidates: int = 16,
    avoidance_m_chunk: int = 32768,
    show_progress: bool = True,
    verbose: bool = True,
    collision_use_amp: Optional[bool] = None,
) -> CollisionAvoidanceResult:
    """Check collisions and attempt axis adjustment for colliding points.

    The flow mirrors :func:`collision_checker.check_collisions` but adds a
    two-phase avoidance search for SM points that initially collide.

    For SM layers the initial detection also classifies *where* in the tool
    the collision occurs (hemisphere vs shank/holder).  This lets
    :func:`avoid_sm_collisions` skip phases that cannot resolve the detected
    collision type, saving unnecessary NN queries:

    * Hemisphere-only collision  → Phase 2 only (tilt whole tool to shift ball
      centre; Phase 1 cannot move the ball).
    * Shank/holder-only collision → Phase 1 only (vary ``tool_vec``; Phase 2
      is not needed).
    * Mixed collision             → Phase 1 first, Phase 2 as fallback.

    Args:
        ctx: Postprocessing context (model, config, etc.).
        path_points_real: (N, 3) path points in real-world coordinates.
        layer_time: Time value for material query.
        layer_type: 'AM' or 'SM'.
        batch_size: GPU batch size for tool sampling.
        n_tool_samples: Number of random samples inside the tool volume.
            Must be >= 3.
        query_batch_size: GPU batch size for NN inference.
        avoidance_cone_half_angle: How far (degrees) to tilt the tool when
            searching for a safe orientation.
        avoidance_n_candidates: How many tilt directions to test per phase.
        avoidance_m_chunk: Maximum colliding-point batch size in the avoidance
            GPU queries.  Bounds peak VRAM to
            ``avoidance_m_chunk × n_per_pt × 3 × 4`` bytes per thread.
            Default 32768; reduce if multi-thread OOM is observed.

    Returns:
        :class:`CollisionAvoidanceResult` with per-point orientations and
        resolved / unresolved masks.
    """
    scale = ctx.scale
    device = ctx.device if isinstance(ctx.device, torch.device) else torch.device(ctx.device)
    manu = ctx.manu_config
    is_cuda = (device.type == 'cuda')
    use_amp = is_cuda if collision_use_amp is None else (
        bool(collision_use_amp) and is_cuda
    )

    if n_tool_samples < 3:
        raise ValueError(
            f"n_tool_samples must be >= 3, got {n_tool_samples}. "
            "This ensures non-negative SM part sample counts."
        )

    aabb_min = torch.as_tensor(ctx.spaceBox[0], dtype=torch.float32, device=device)
    aabb_max = torch.as_tensor(ctx.spaceBox[1], dtype=torch.float32, device=device)

    N = len(path_points_real)
    pts_model = torch.tensor(
        path_points_real / scale, dtype=torch.float32, device=device,
    )

    if layer_type == 'AM':
        # Allocate more samples to cone (tip) for better collision detection at AM nozzle
        n_cone = (4 * n_tool_samples) // 10
        n_cyl  = n_tool_samples // 10
        # Surface sample counts: ~1/4 of volume count for each part
        n_cone_surface = n_cone
        n_cyl_surface  = n_cyl
    else:
        n_sphere = max(n_tool_samples // 8, 1)
        n_shank  = max(n_tool_samples // 4, 1)
        n_holder = max(n_tool_samples // 8, 1)
        # Surface sample counts: ~1/4 of volume count for each part
        n_sphere_surface = n_sphere
        n_shank_surface  = n_shank
        n_holder_surface = n_holder
        # Use high-density holder-bottom sampling to reduce false negatives.
        n_holder_bottom_surface = max(256, 8 * n_holder_surface)
        # Precompute split index for sphere vs. shank detection
        # Layout: [sphere_vol, sphere_surf, shank_vol, shank_surf, holder_vol, holder_surf]
        n_sphere_total = n_sphere + n_sphere_surface

    # --- Phase 1: Default-axis collision detection (batched) ---
    # This phase checks all points using their standard orientation.
    # It is fast because it processes points in large batches on the GPU.
    collision_mask_gpu = torch.zeros(N, dtype=torch.bool, device=device)
    if layer_type != 'AM':
        sphere_collision_mask_gpu = torch.zeros(N, dtype=torch.bool, device=device)
        shank_holder_collision_mask_gpu = torch.zeros(N, dtype=torch.bool, device=device)

    if verbose:
        amp_tag = "on" if use_amp else "off"
        print(f"Collision avoidance: {N} points, batch={batch_size}, amp={amp_tag}")
    with torch.inference_mode():
        for b_start in tqdm(range(0, N, batch_size),
                            desc=f'{layer_type} default collision',
                            disable=not show_progress):
            b_end = min(b_start + batch_size, N)
            bp = pts_model[b_start:b_end]
            M = bp.shape[0]

            if layer_type == 'AM':
                # Default AM tool: vertical (+Z)
                tool_pts = sample_am_tool(
                    bp, manu['AMConeHeight'], manu['AMConeHalfAngle'],
                    manu['AMCollisionMargin'],
                    float(aabb_max[2]),
                    n_cone=n_cone, n_cylinder=n_cyl,
                    n_cone_surface=n_cone_surface, n_cylinder_surface=n_cyl_surface,
                )
            else:
                # Default SM tool: oriented by field3
                tool_pts = sample_sm_tool(
                    bp, ctx.model,
                    sm_tip_diameter=manu['SMToolParas']['SMTipDiameter'],
                    sm_shank_diameter=manu['SMToolParas']['SMShankDiameter'],
                    sm_tool_length=manu['SMToolParas']['SMToolLength'],
                    sm_holder_diameter=manu['SMToolParas']['SMHolderDiameter'],
                    sm_holder_length=manu['SMToolParas']['SMHolderLength'],
                    collision_margin=manu['SMToolParas']['SMCollisionMargin'],
                    n_sphere=n_sphere, n_shank=n_shank, n_holder=n_holder,
                    n_sphere_surface=n_sphere_surface, n_shank_surface=n_shank_surface,
                    n_holder_surface=n_holder_surface,
                    n_holder_bottom_surface=n_holder_bottom_surface,
                )

            flat = tool_pts.reshape(-1, 3)
            exists = _query_exists_flat(
                ctx.model, ctx.check_func, flat,
                layer_time, ctx.max_time,
                aabb_min, aabb_max, query_batch_size,
                use_amp=use_amp,
            )
            n_per_pt = tool_pts.shape[1]
            exists_2d = exists.reshape(M, n_per_pt)
            # If any sample point inside the tool hits material, it's a collision
            collision_mask_gpu[b_start:b_end] = exists_2d.any(dim=1)

            if layer_type != 'AM':
                # Layout: [sphere_vol, sphere_surf, shank_vol, shank_surf, holder_vol, holder_surf, holder_bottom_surf]
                sphere_collision_mask_gpu[b_start:b_end] = exists_2d[:, :n_sphere_total].any(dim=1)
                shank_holder_collision_mask_gpu[b_start:b_end] = exists_2d[:, n_sphere_total:].any(dim=1)

            del tool_pts, flat, exists, exists_2d

    collision_mask = collision_mask_gpu.cpu().numpy()
    if layer_type != 'AM':
        sphere_collision_mask = sphere_collision_mask_gpu.cpu().numpy()
        # shank_holder_collision_mask = shank_holder_collision_mask_gpu.cpu().numpy()

    n_collisions = int(collision_mask.sum())
    if verbose:
        print(f"  Default collisions: {n_collisions}/{N}")

    # Short-circuit if no collisions found
    if n_collisions == 0:
        orientations = []
        if layer_type == 'AM':
            for _ in range(N):
                orientations.append(ToolOrientation(
                    am_axis=np.array([0., 0., 1.]),
                    sm_normal_vec=np.zeros(3),
                    sm_tool_vec=np.zeros(3),
                    adjusted=False,
                ))
        else:
            normal_vec, tool_vec = query_sm_orientation(ctx.model, pts_model)
            normal_np = normal_vec.detach().cpu().numpy()
            tool_np = tool_vec.detach().cpu().numpy()
            for i in range(N):
                orientations.append(ToolOrientation(
                    am_axis=np.zeros(3),
                    sm_normal_vec=normal_np[i],
                    sm_tool_vec=tool_np[i],
                    adjusted=False,
                ))
        return CollisionAvoidanceResult(
            points=path_points_real,
            orientations=orientations,
            collision_resolved=np.zeros(N, dtype=bool),
            collision_unresolved=np.zeros(N, dtype=bool),
        )

    # --- Phase 2: Avoidance search for colliding points ---
    # This phase is slower as it iterates per-point or per-candidate,
    # but only runs on the subset of points that actually collided.
    with torch.inference_mode():
        if layer_type == 'AM':
            resolved, unresolved, orientations = avoid_am_collisions(
                ctx, pts_model, collision_mask, layer_time,
                cone_half_angle_deg=avoidance_cone_half_angle,
                n_candidates=avoidance_n_candidates,
                n_cone=n_cone, n_cylinder=n_cyl,
                n_cone_surface=n_cone_surface, n_cyl_surface=n_cyl_surface,
                query_batch_size=query_batch_size,
                m_chunk=avoidance_m_chunk,
                collision_use_amp=use_amp,
            )
        else:
            resolved, unresolved, orientations = avoid_sm_collisions(
                ctx, pts_model, collision_mask, layer_time,
                sphere_collision_mask=sphere_collision_mask,
                cone_half_angle_deg=avoidance_cone_half_angle,
                max_tilt_angle_deg=89.0,
                n_candidates=avoidance_n_candidates,
                n_sphere=n_sphere, n_shank=n_shank, n_holder=n_holder,
                n_sphere_surface=n_sphere_surface, n_shank_surface=n_shank_surface,
                n_holder_surface=n_holder_surface,
                n_holder_bottom_surface=n_holder_bottom_surface,
                query_batch_size=query_batch_size,
                m_chunk=avoidance_m_chunk,
                collision_use_amp=use_amp,
            )

            # Conservative post-check with denser samples, then retry unresolved residuals.
            verify_n_sphere = max(64, n_sphere * 2)
            verify_n_shank = max(128, n_shank * 2)
            verify_n_holder = max(128, n_holder * 2)
            verify_n_sphere_surface = max(64, n_sphere_surface * 2)
            verify_n_shank_surface = max(128, n_shank_surface * 2)
            verify_n_holder_surface = max(128, n_holder_surface * 2)
            verify_n_holder_bottom_surface = max(2048, n_holder_bottom_surface * 2)
            verify_n_sphere_total = verify_n_sphere + verify_n_sphere_surface

            ori_nv = np.stack([o.sm_normal_vec for o in orientations], axis=0)
            ori_tv = np.stack([o.sm_tool_vec for o in orientations], axis=0)
            ori_nv_t = torch.as_tensor(ori_nv, dtype=torch.float32, device=device)
            ori_tv_t = torch.as_tensor(ori_tv, dtype=torch.float32, device=device)

            residual_collision_gpu = torch.zeros(N, dtype=torch.bool, device=device)
            residual_sphere_gpu = torch.zeros(N, dtype=torch.bool, device=device)

            with torch.inference_mode():
                for b_start in tqdm(range(0, N, batch_size),
                                    desc='SM strict recheck',
                                    disable=not show_progress):
                    b_end = min(b_start + batch_size, N)
                    bp = pts_model[b_start:b_end]
                    nv_b = ori_nv_t[b_start:b_end]
                    tv_b = ori_tv_t[b_start:b_end]
                    M = bp.shape[0]

                    tool_pts = sample_sm_tool(
                        bp, ctx.model,
                        sm_tip_diameter=manu['SMToolParas']['SMTipDiameter'],
                        sm_shank_diameter=manu['SMToolParas']['SMShankDiameter'],
                        sm_tool_length=manu['SMToolParas']['SMToolLength'],
                        sm_holder_diameter=manu['SMToolParas']['SMHolderDiameter'],
                        sm_holder_length=manu['SMToolParas']['SMHolderLength'],
                        collision_margin=manu['SMToolParas']['SMCollisionMargin'],
                        n_sphere=verify_n_sphere, n_shank=verify_n_shank, n_holder=verify_n_holder,
                        n_sphere_surface=verify_n_sphere_surface,
                        n_shank_surface=verify_n_shank_surface,
                        n_holder_surface=verify_n_holder_surface,
                        n_holder_bottom_surface=verify_n_holder_bottom_surface,
                        override_normal_vec=nv_b,
                        override_tool_vec=tv_b,
                    )

                    flat = tool_pts.reshape(-1, 3)
                    exists = _query_exists_flat(
                        ctx.model, ctx.check_func, flat,
                        layer_time, ctx.max_time,
                        aabb_min, aabb_max, query_batch_size,
                        use_amp=use_amp,
                    )
                    exists_2d = exists.reshape(M, tool_pts.shape[1])
                    residual_collision_gpu[b_start:b_end] = exists_2d.any(dim=1)
                    residual_sphere_gpu[b_start:b_end] = exists_2d[:, :verify_n_sphere_total].any(dim=1)

                    del tool_pts, flat, exists, exists_2d

            residual_collision = residual_collision_gpu.cpu().numpy()
            residual_sphere = residual_sphere_gpu.cpu().numpy()

            n_residual = int(residual_collision.sum())
            if n_residual > 0 and verbose:
                print(f"  Strict recheck found residual SM collisions: {n_residual}/{N}, retrying...")

            if n_residual > 0:
                retry_resolved, retry_unresolved, retry_orientations = avoid_sm_collisions(
                    ctx, pts_model, residual_collision, layer_time,
                    sphere_collision_mask=residual_sphere,
                    cone_half_angle_deg=min(avoidance_cone_half_angle + 5.0, 30.0),
                    max_tilt_angle_deg=89.0,
                    n_candidates=max(avoidance_n_candidates * 2, 24),
                    n_sphere=verify_n_sphere,
                    n_shank=verify_n_shank,
                    n_holder=verify_n_holder,
                    n_sphere_surface=verify_n_sphere_surface,
                    n_shank_surface=verify_n_shank_surface,
                    n_holder_surface=verify_n_holder_surface,
                    n_holder_bottom_surface=verify_n_holder_bottom_surface,
                    query_batch_size=query_batch_size,
                    m_chunk=avoidance_m_chunk,
                    collision_use_amp=use_amp,
                )

                retry_resolved_idx = np.where(retry_resolved)[0]
                for i in retry_resolved_idx:
                    orientations[i] = retry_orientations[i]

                resolved = np.logical_or(resolved, retry_resolved)
                unresolved = np.logical_or(unresolved, retry_unresolved)

                # Keep any point still failing strict recheck as unresolved.
                still_bad = np.logical_and(residual_collision, np.logical_not(retry_resolved))
                unresolved = np.logical_or(unresolved, still_bad)
                resolved[still_bad] = False

    n_resolved = int(resolved.sum())
    n_unresolved = int(unresolved.sum())
    if verbose:
        print(f"  Avoidance: {n_resolved} resolved, {n_unresolved} unresolved "
              f"(of {n_collisions} collisions)")

    return CollisionAvoidanceResult(
        points=path_points_real,
        orientations=orientations,
        collision_resolved=resolved,
        collision_unresolved=unresolved,
    )