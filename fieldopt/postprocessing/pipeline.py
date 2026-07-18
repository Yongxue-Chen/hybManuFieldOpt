"""
End-to-end hybrid manufacturing post-processing pipeline.

Usage::

    from fieldopt.postprocessing import run_pipeline

    result = run_pipeline(
        config_name='fertility',
        model_path='output/multi_field_model_joint_trained.pth',
    )

    # result['layers']  – interleaved AM/SM Layer objects (real coords)
    # result['paths']   – per-layer FilteredPath results
"""
import os
import time
import threading
import contextlib
import multiprocessing as mp
import numpy as np
import torch
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .model_loader import load_model_and_config, PostprocessContext
from .layer_generator import (
    generate_all_layers,
    interleave_layers,
    scale_layer_mesh,
    Layer,
)
from .path_generator import generate_path
from .path_postprocessor import process_path_with_avoidance

import pyvista as pv

def run_pipeline(
    config_name: str,
    model_path: str,
    *,
    # Model / STL
    stl_dir: str = 'stlFiles',
    device: Optional[str] = None,
    voxel_resolution: int = 512,
    geometry_backend: str = 'voxel_artifact',
    geometry_artifact_path: Optional[str] = None,
    # Layer generation
    grid_resolution: int = 256,
    am_coarse_count: int = 100,
    sm_coarse_count: int = 100,
    am_min_fragment_size: int = 8,
    sm_min_fragment_size: int = 3,
    min_layer_points: int = 0,
    # Path generation
    path_method: str = 'zigzag',
    am_path_width: float = 0.8,
    sm_path_width: float = 0.8,

    direction_angle: float = 0.0,
    sample_spacing: float = 1.0,
    # Collision detection
    collision_batch_size: int = 65536,
    n_tool_samples: int = 900,
    query_batch_size: int = 262144,
    # Collision avoidance (NEW)
    enable_avoidance: bool = True,
    avoidance_cone_half_angle: float = 30.0,
    avoidance_n_candidates: int = 20,
    avoidance_m_chunk: int = 32768,
    # Parallelism
    n_workers: int = 16,
    # Logging / progress
    progress_style: str = 'bar',
    verbose: bool = False,
    # Collision acceleration
    collision_use_amp: Optional[bool] = None,
    # Misc
    skip_collision_check: bool = False,
    skip_paths: bool = False,
    sm_only: bool = False,
) -> Dict[str, Any]:
    """
    Run the full hybrid manufacturing post-processing pipeline.

    This function orchestrates the entire workflow from trained model to manufacturing paths.

    Args:
        config_name (str): 
            Suffix for the configuration file to load.
            Example: 'bracket' loads 'configs.config_multi_field_bracket'.
        model_path (str):
            Absolute or relative path to the trained model weights (.pth file).
        
        -- Model / STL / Environment --
        stl_dir (str):
            Directory containing the ground-truth STL file (named <config_name>.stl).
            Used for normalization and coordinate scaling.
        device (str, optional):
            Computation device, e.g., 'cuda:0' or 'cpu'. If None, uses the device defined in config.
        voxel_resolution (int):
            Resolution for voxelizing the ground truth STL (e.g., 512 means a 512^3 grid).
            Used to create the implicit function for checking "is point inside model?".
        geometry_backend (str):
            Implicit geometry backend: 'voxel_artifact', 'voxel', or 'siren'.
        geometry_artifact_path (str, optional):
            Optional saved voxel artifact or neural-SDF checkpoint path.
        -- Layer Generation Parameters --
        grid_resolution (int):
            Resolution of the 3D grid used to extract layer meshes via Marching Cubes.
            Higher values (e.g. 256) give smoother layers but consume more memory.
        am_coarse_count (int):
            Number of initial candidate time-steps to evaluate for Additive Manufacturing (AM) layers.
            The algorithm selects valid layers from these candidates based on layer height constraints.
        sm_coarse_count (int):
            Number of initial candidate time-steps for Subtractive Manufacturing (SM) layers.
        am_min_fragment_size (int):
            Minimum number of vertices an AM mesh fragment must have to be kept.
            Default 10.
        sm_min_fragment_size (int):
            Minimum number of vertices an SM mesh fragment must have to be kept.
            Can be set smaller than AM when SM layers are naturally finer. Default 10.
        am_max_layer_height (float, optional):
            Maximum allowed local distance (in real-world units, mm) between two
            consecutive AM layers.  Wherever the gap exceeds this value a local
            intermediate layer is inserted adaptively.  When None (default) the
            value from ``manu_config['max_layer_AM']`` in the config file is used.
        sm_max_layer_height (float, optional):
            Same as ``am_max_layer_height`` but for Subtractive Manufacturing layers.
            Overrides ``manu_config['max_layer_SM']`` when provided.
        min_layer_points (int):
            After layer generation, discard any layer whose total vertex count is
            below this threshold.  Useful for removing near-empty layers that survive
            fragment filtering but cover only a negligible area.  Default 0 keeps all.

        -- Path Generation Parameters --
        path_method (str):
            Pattern strategy for covering the layer. Currently supports: 'zigzag'.
        am_path_width (float, optional):
            AM layer zigzag spacing in **real-world units (mm)**.
            If None, path generator default is used.
        sm_path_width (float, optional):
            SM layer zigzag spacing in **real-world units (mm)**.
            If None, path generator default is used.
        direction_angle (float):
            Rotation angle of the zigzag pattern in degrees (0 = horizontal scan lines).
        sample_spacing (float):
            Fixed arc-length spacing (in **real-world units**, e.g. mm) used to
            resample zigzag paths before collision detection.  Smaller values
            give denser path points and finer collision resolution but increase
            compute cost.  Set to ``0`` or negative to skip resampling.
        project_to_surface (bool):
            If True, 2D zigzag points are projected onto the 3D curved layer surface.
            Crucial for non-planar layers to ensure the tool follows the curvature.

        -- Collision Detection Parameters --
        collision_batch_size (int):
            Number of path points to process in parallel on the GPU during tool sampling.
            Reduce this if you run out of VRAM.
        n_tool_samples (int):
            Number of random points sampled *inside* the tool volume for each path point.
            More samples = more accurate collision detection but slower.
            Must be >= 3.
            Sampling is stochastic; set global random seeds before calling
            this function if you need repeatable collision masks.
        query_batch_size (int):
            Number of points sent to the Neural Network in a single forward pass.
            Reduce this if you run out of VRAM.

        -- Collision Avoidance Parameters (NEW) --
        enable_avoidance (bool):
            If True, when a collision is detected, the system tries to find a collision-free
            tool axis by tilting the tool.
            If False, colliding points are simply discarded (path is interrupted).
        avoidance_cone_half_angle (float):
            Maximum tilt angle (in degrees) from the default axis to search for a safe orientation.
            Imagine a cone around the tool axis; we search along the surface of this cone.
        avoidance_n_candidates (int):
            Number of candidate directions to test around the cone (e.g., 16 means test every 22.5 degrees).
        avoidance_m_chunk (int):
            Maximum number of colliding points processed in one GPU call during the
            avoidance search.  Bounds peak VRAM to
            ``avoidance_m_chunk × n_per_pt × 3 × 4`` bytes per thread regardless of
            how many points collide.  Default 32768; reduce if OOM is observed with
            many concurrent workers.

        -- Safe Move Parameters (NEW) --
        retract_height (float):
            Safe retraction height in **real-world units (mm)**.
            When moving between unconnected path segments (or disjoint islands), the tool
            will lift up by this amount, travel horizontally, and then descend.

        n_workers (int):
            Number of worker threads used in Phase B (collision detection).
            Phase A (path generation) always uses up to ``os.cpu_count()``
            process workers via ``ProcessPoolExecutor`` with the ``spawn``
            start method – this bypasses the GIL so alphashape/Shapely run
            on all available CPU cores simultaneously.
            Each Phase-B thread gets its own CUDA stream so GPU kernels from
            different layers can be overlapped by the hardware scheduler.
            Increase if you have spare VRAM (~50 MB per extra worker).
            Default is 16.

        -- Debugging --
        skip_collision_check (bool):
            If True, skips all collision detection logic. Returns raw paths immediately.
            Useful for quick visualization of the geometry.

    Returns:
        dict containing:
          - 'layers': List of Layer objects (mesh, type, time).
          - 'paths':  List of path data dicts (raw path, filtered path, statistics).
          - 'scale':  Scaling factor used (model space <-> real space).
          - 'ctx':    The PostprocessContext object.
    """

    t_pipeline_start = time.perf_counter()

    progress_style = (progress_style or 'bar').lower()
    if progress_style not in ('bar', 'dots', 'none'):
        raise ValueError(
            "progress_style must be one of {'bar', 'dots', 'none'}."
        )

    # ---- 1. Load model & config ----
    print("=" * 60)
    print(f"  Hybrid Post-Processing Pipeline  [{config_name}]")
    print("=" * 60)
    t0 = time.perf_counter()
    ctx = load_model_and_config(
        config_name, model_path,
        stl_dir=stl_dir, device=device,
        voxel_resolution=voxel_resolution,
        geometry_backend=geometry_backend,
        geometry_artifact_path=geometry_artifact_path,
    )
    t_load = time.perf_counter() - t0
    scale = ctx.scale

    # ---- 2. Generate layers (model space) ----
    t0 = time.perf_counter()
    am_layers, sm_layers = generate_all_layers(
        ctx,
        resolution=grid_resolution,
        am_coarse_count=am_coarse_count,
        sm_coarse_count=sm_coarse_count,
        am_min_fragment_size=am_min_fragment_size,
        sm_min_fragment_size=sm_min_fragment_size,
        min_layer_points=min_layer_points,
        build_am=not sm_only,
        build_sm=True,
    )
    t_layer_gen = time.perf_counter() - t0

    interleaved = interleave_layers(am_layers, sm_layers)
    print(f"\nInterleaved sequence: {len(interleaved)} layers "
          f"({len(am_layers)} AM + {len(sm_layers)} SM)")
    if sm_only:
        selected_layers = list(sm_layers)
        print(f"SM-only mode: keeping {len(selected_layers)} SM layers and skipping AM outputs.")
    else:
        selected_layers = interleaved

    # ---- 3. Scale meshes to real-world coordinates ----
    t0 = time.perf_counter()
    real_layers = [scale_layer_mesh(l, scale) for l in selected_layers]
    t_scale_mesh = time.perf_counter() - t0

    # ---- 4. Optional early exit: layer-only mode ----
    if skip_paths:
        t_pipeline_total = time.perf_counter() - t_pipeline_start
        print("\n" + "=" * 60)
        print("  Pipeline complete (layer-only mode, paths skipped).")
        print("  Timing summary (seconds):")
        print(f"    load_model_and_config : {t_load:.3f}")
        print(f"    generate_all_layers   : {t_layer_gen:.3f}")
        print(f"    scale_layer_mesh      : {t_scale_mesh:.3f}")
        print(f"    pipeline_total_wall   : {t_pipeline_total:.3f}")
        print("=" * 60)
        return {
            'layers': real_layers,
            'paths': [],
            'scale': scale,
            'config_name': config_name,
            'ctx': ctx,
        }

    # ---- 5. Per-layer path generation + optional collision avoidance ----
    if skip_collision_check:
        print("\nGenerating raw paths for each layer (collision checks skipped) ...")
    else:
        print("\nGenerating collision-aware paths for each layer ...")
    path_verbose = verbose
    paths = []
    total_path_points = 0
    total_removed_points = 0
    layers_with_removed = 0

    # Helper to select per-layer path width
    def _path_width_for_layer(layer: Layer) -> float:
        if layer.layer_type == 'AM':
            return am_path_width
        return sm_path_width

    # NOTE:
    #   * For now we always use the new generate_path + process_path_with_avoidance
    #     pipeline used in debug_path.py, so paths are already collision-free.
    #   * The old skip_collision_check / enable_avoidance flags are honoured
    #     in a simplified way:
    #       - skip_collision_check=True: raw geometric paths only, no filtering.
    #       - skip_collision_check=False: always run avoidance (if it fails,
    #         those points are dropped and reported via removed_points).

    for layer in tqdm(real_layers, desc="Per-layer paths", disable=(progress_style == 'none')):
        if layer.mesh is None or layer.mesh.n_points < 3:
            paths.append({
                'segments': [],
                'removed_points': np.empty((0, 3), dtype=float),
            })
            continue

        # Generate raw geometric zigzag paths in real-world coordinates
        width = _path_width_for_layer(layer)
        angle = direction_angle

        raw_segments = generate_path(
            layer_mesh=layer.mesh,
            layer_type=layer.layer_type,
            path_width=width,
            direction_angle=angle,
            sample_spacing=sample_spacing,
        )

        if skip_collision_check or not raw_segments:
            if raw_segments and layer.layer_type in ('AM', 'SM'):
                from .tool_shape import query_sm_orientation
                pts_list = [seg['points'] for seg in raw_segments]
                n_pts_total = sum(len(pts) for pts in pts_list)
                if n_pts_total > 0:
                    import torch
                    device = ctx.device if isinstance(ctx.device, torch.device) else torch.device(ctx.device)
                    pts_flat = np.vstack(pts_list)
                    pts_model = torch.tensor(pts_flat / scale, dtype=torch.float32, device=device)
                    
                    if layer.layer_type == 'SM':
                        with torch.inference_mode():
                            normal_vec, tool_vec = query_sm_orientation(ctx.model, pts_model)
                            normal_np = normal_vec.cpu().numpy()
                            tool_np = tool_vec.cpu().numpy()
                    else:
                        am_axis = np.tile(np.array([0., 0., 1.]), (n_pts_total, 1))

                    cursor = 0
                    for seg in raw_segments:
                        n_seg = len(seg['points'])
                        if layer.layer_type == 'SM':
                            seg['orientations'] = {
                                'sm_normal_vec': normal_np[cursor:cursor+n_seg],
                                'sm_tool_vec': tool_np[cursor:cursor+n_seg]
                            }
                        else:
                            seg['orientations'] = {
                                'am_axis': am_axis[cursor:cursor+n_seg]
                            }
                        cursor += n_seg

            safe_segments = raw_segments
            removed_points = np.empty((0, 3), dtype=float)
        else:
            # When enable_avoidance=False, set n_candidates to 0 (detection + drop only)
            effective_n_candidates = avoidance_n_candidates if enable_avoidance else 0
            safe_segments, removed_points = process_path_with_avoidance(
                ctx,
                raw_segments,
                layer.layer_type,
                layer.time_value,
                collision_batch_size=collision_batch_size,
                n_tool_samples=n_tool_samples,
                query_batch_size=query_batch_size,
                avoidance_cone_half_angle=avoidance_cone_half_angle,
                avoidance_n_candidates=effective_n_candidates,
                avoidance_m_chunk=avoidance_m_chunk,
                collision_use_amp=collision_use_amp,
                verbose=path_verbose,
            )

        n_kept = sum(seg['points'].shape[0] for seg in safe_segments)
        n_removed = len(removed_points)
        total_path_points += n_kept + n_removed
        total_removed_points += n_removed
        if n_removed > 0:
            layers_with_removed += 1

        paths.append({
            'segments': safe_segments,
            'removed_points': removed_points,
        })

    t_pipeline_total = time.perf_counter() - t_pipeline_start

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Timing summary (seconds):")
    print(f"    load_model_and_config : {t_load:.3f}")
    print(f"    generate_all_layers   : {t_layer_gen:.3f}")
    print(f"    scale_layer_mesh      : {t_scale_mesh:.3f}")
    print(f"    pipeline_total_wall   : {t_pipeline_total:.3f}")
    if not skip_collision_check:
        print("  Path summary:")
        print(f"    total path points (before removal) : {total_path_points}")
        print(f"    removed (unresolved collisions)     : {total_removed_points}")
        print(f"    layers with at least one removal   : {layers_with_removed}")
    else:
        print("  Raw path summary:")
        print(f"    total path points : {total_path_points}")
        print("    collision checks  : skipped")
    print("=" * 60)

    # ---- 6. Recenter output: shift origin to AABB bottom-face center ----
    # Current origin is the AABB minimum corner (xmin, ymin, zmin) of the STL.
    # We shift all coordinates so that the new origin is the center of the
    # bottom face, i.e. subtract (Lx/2, Ly/2, 0) from every point.
    real_extent = ctx.spaceBox[1] * scale  # [Lx, Ly, Lz] in mm
    xy_center_offset = np.array([real_extent[0] / 2.0, real_extent[1] / 2.0, 0.0])

    print(f"\n  Recentering: shifting origin by -{xy_center_offset} mm "
          f"(AABB bottom-face center).")

    # Shift layer meshes in-place
    for layer in real_layers:
        if layer.mesh is not None and layer.mesh.n_points > 0:
            layer.mesh.points -= xy_center_offset

    # Shift all path segment points
    for path_dict in paths:
        for seg in path_dict.get('segments', []):
            pts = seg.get('points')
            if pts is not None and len(pts) > 0:
                seg['points'] = np.asarray(pts) - xy_center_offset
            # Also shift orientation vectors' base positions if stored as points
            # (orientation vectors themselves are directions, not positions, so
            # they are not shifted)
        rp = path_dict.get('removed_points')
        if rp is not None and len(rp) > 0:
            path_dict['removed_points'] = np.asarray(rp) - xy_center_offset

    return {
        'layers': real_layers,
        'paths': paths,
        'scale': scale,
        'config_name': config_name,
        'ctx': ctx,
        'xy_center_offset': xy_center_offset,
    }
