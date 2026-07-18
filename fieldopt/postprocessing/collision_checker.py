"""
Collision detection between manufacturing tools and existing material.

For every path point the tool shape is sampled, and the neural network is
queried to decide whether material exists at each sample.  If *any* sample
collides, the path point is marked unsafe.

All heavy GPU work is batched to stay within VRAM limits.  Three knobs
control memory usage:

  * ``batch_size``       – path points processed per outer batch
  * ``n_tool_samples``   – total samples inside the tool per point
  * ``query_batch_size`` – points fed to the NN in one forward pass
"""
import torch
import numpy as np
from typing import Optional, Tuple, Union
from tqdm import tqdm

from .tool_shape import sample_am_tool, sample_sm_tool, query_sm_orientation


# ---------------------------------------------------------------------------
# Material existence query
# ---------------------------------------------------------------------------

def _query_material_exists(
    model,
    check_func,
    points: torch.Tensor,
    time: float,
    max_time: float,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    query_batch_size: int,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Determine whether material exists at *points* at the given *time*.

    Returns a boolean tensor of shape (N,).

    Material exists at point p when BOTH:
      1. probs1(p) >= 0.5 **and** time1(p) <= time   (already deposited)
      2. NOT (probs2(p) >= 0.5 **and** time2(p) <= time)  (not yet removed)

    Args:
        model: Trained neural model.
        check_func: In-model occupancy helper callable.
        points: Flattened tool sample tensor ``(N, 3)`` in model space.
        time: Query process time.
        max_time: Global max process time.
        aabb_min: Model-space AABB minimum corner tensor ``(3,)``.
        aabb_max: Model-space AABB maximum corner tensor ``(3,)``.
        query_batch_size: Inference mini-batch size for ``points``.
    """
    N = points.shape[0]
    exists = torch.zeros(N, dtype=torch.bool, device=points.device)

    for start in range(0, N, query_batch_size):
        end = min(start + query_batch_size, N)
        pts = points[start:end]

        inside_aabb = ((pts >= aabb_min) & (pts <= aabb_max)).all(dim=-1)
        if not torch.any(inside_aabb):
            continue

        pts_valid = pts[inside_aabb]
        with torch.autocast(
            device_type=points.device.type,
            enabled=use_amp,
            dtype=amp_dtype,
        ):
            isInModel, _ = check_func(pts_valid)
            f1, f2, lM1_raw, lM2_raw = model(pts_valid, field_type='timesAndMasks')
        lM1 = torch.where(isInModel == 1, 5.0, lM1_raw)
        lM2 = torch.where(isInModel == 1, -5.0, lM2_raw)

        t1 = f1.squeeze(-1)
        t2 = (f1 + f2 * (max_time - f1)).squeeze(-1)
        p1 = torch.sigmoid(lM1).squeeze(-1)
        p2 = torch.sigmoid(lM2).squeeze(-1)

        deposited = (p1 >= 0.5) & (t1 <= time)
        removed = (p2 >= 0.5) & (t2 <= time)
        mat = deposited & (~removed)

        local_exists = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
        local_exists[inside_aabb] = mat

        is_below = (
            (pts[:, 0] >= aabb_min[0]) & (pts[:, 0] <= aabb_max[0]) &
            (pts[:, 1] >= aabb_min[1]) & (pts[:, 1] <= aabb_max[1]) &
            (pts[:, 2] <= aabb_min[2])
        )
        local_exists[is_below] = True

        exists[start:end] = local_exists

    return exists


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# def check_collisions(
#     ctx,
#     path_points_real: np.ndarray,
#     layer_time: float,
#     layer_type: str,
#     *,
#     sm_collision_margin: float = 0.0,
#     batch_size: int = 4096,
#     n_tool_samples: int = 200,
#     query_batch_size: int = 32768,
#     show_progress: bool = True,
#     verbose: bool = True,
#     collision_use_amp: Optional[bool] = None,
#     return_orientations: bool = False,
# ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     Check whether the tool collides with existing material at each path point.

#     Args:
#         ctx:  PostprocessContext.
#         path_points_real: (N, 3) path points in **real-world** coordinates.
#         layer_time:  Time value associated with the current layer.
#         layer_type:  ``'AM'`` or ``'SM'``.
#         sm_collision_margin: Safety margin (real-world units) by which SM
#             hemisphere + shank are shrunk inward.  Holder is NOT shrunk.
#             Ignored for AM.
#         batch_size:  Path points per GPU batch (lower = less VRAM).
#         n_tool_samples: Total samples per path point inside the tool.
#             Must be >= 3.
#         query_batch_size: Points per NN forward pass (lower = less VRAM).
#         return_orientations: If True, also return SM orientations queried
#             from the model (SM layers only).

#     Returns:
#         If ``return_orientations`` is False (default):
#             Boolean ndarray of shape ``(N,)``.  ``True`` = collision detected.
#         If ``return_orientations`` is True:
#             Tuple ``(collision_mask, normal_np, tool_np)`` where
#             ``normal_np`` and ``tool_np`` are ``(N, 3)`` ndarrays.
#             For AM layers, both orientation arrays are ``None``.
#     """
#     scale = ctx.scale
#     model = ctx.model
#     check_func = ctx.check_func
#     device = ctx.device if isinstance(ctx.device, torch.device) else torch.device(ctx.device)
#     max_time = ctx.max_time
#     manu = ctx.manu_config
#     is_cuda = (device.type == 'cuda')
#     use_amp = is_cuda if collision_use_amp is None else (
#         bool(collision_use_amp) and is_cuda
#     )

#     if n_tool_samples < 3:
#         raise ValueError(
#             f"n_tool_samples must be >= 3, got {n_tool_samples}. "
#             "This ensures non-negative SM part sample counts."
#         )

#     aabb_min = torch.as_tensor(ctx.spaceBox[0], dtype=torch.float32, device=device)
#     aabb_max = torch.as_tensor(ctx.spaceBox[1], dtype=torch.float32, device=device)

#     N = len(path_points_real)
#     collision_mask_gpu = torch.zeros(N, dtype=torch.bool, device=device)

#     # Convert to model space
#     pts_model = torch.tensor(
#         path_points_real / scale, dtype=torch.float32, device=device,
#     )

#     # Distribute sample counts across tool parts
#     if layer_type == 'AM':
#         n_cone = (4 * n_tool_samples) // 5
#         n_cyl = n_tool_samples - n_cone
#     else:
#         n_hemi = max(n_tool_samples // 3, 1)
#         n_shank = max(n_tool_samples // 2, 1)
#         n_holder = n_tool_samples - n_hemi - n_shank

#     # Bulk-query SM orientations once instead of per-batch.
#     sm_normal_vec = sm_tool_vec = None
#     if layer_type != 'AM' and N > 0:
#         sm_normal_vec, sm_tool_vec = query_sm_orientation(model, pts_model)

#     if verbose:
#         amp_tag = "on" if use_amp else "off"
#         print(f"Collision check: {N} points, batch={batch_size}, "
#               f"samples/pt={n_tool_samples}, query_batch={query_batch_size}, amp={amp_tag}")

#     with torch.inference_mode():
#         for b_start in tqdm(
#             range(0, N, batch_size),
#             desc=f'{layer_type} collision',
#             disable=not show_progress,
#         ):
#             b_end = min(b_start + batch_size, N)
#             bp = pts_model[b_start:b_end]
#             M = bp.shape[0]

#             # --- Generate tool samples (model space) ---
#             if layer_type == 'AM':
#                 cone_h = manu['AMConeHeight']
#                 cone_a = manu['AMConeHalfAngle']
#                 tool_pts = sample_am_tool(
#                     bp, cone_h, cone_a, float(aabb_max[2]),
#                     n_cone=n_cone, n_cylinder=n_cyl,
#                 )  # (M, n_total, 3)
#             else:
#                 margin_model = sm_collision_margin / scale
#                 tool_pts = sample_sm_tool(
#                     bp, model,
#                     sm_tool_diameter=manu['SMToolParas']['SMToolDiameter'],
#                     sm_tool_length=manu['SMToolParas']['SMToolLength'],
#                     sm_holder_diameter=manu['SMToolParas']['SMHolderDiameter'],
#                     sm_holder_length=manu['SMToolParas']['SMHolderLength'],
#                     collision_margin=margin_model,
#                     n_hemisphere=n_hemi,
#                     n_shank=n_shank,
#                     n_holder=n_holder,
#                     override_normal_vec=sm_normal_vec[b_start:b_end],
#                     override_tool_vec=sm_tool_vec[b_start:b_end],
#                 )  # (M, n_total, 3)

#             n_per_pt = tool_pts.shape[1]

#             flat = tool_pts.reshape(-1, 3)
#             exists = _query_material_exists(
#                 model, check_func, flat, layer_time, max_time,
#                 aabb_min, aabb_max, query_batch_size,
#                 use_amp=use_amp,
#             )

#             exists_2d = exists.reshape(M, n_per_pt)
#             collision_mask_gpu[b_start:b_end] = exists_2d.any(dim=1)

#             del tool_pts, flat, exists, exists_2d
            
#     collision_mask = collision_mask_gpu.cpu().numpy()

#     if return_orientations:
#         normal_np = sm_normal_vec.detach().cpu().numpy() if sm_normal_vec is not None else None
#         tool_np = sm_tool_vec.detach().cpu().numpy() if sm_tool_vec is not None else None
#         return collision_mask, normal_np, tool_np
#     return collision_mask


# Re-export the avoidance-aware entry point so callers can use either module.
from .collision_avoidance import check_and_avoid_collisions  # noqa: E402, F401
