"""
Layer generation for hybrid additive/subtractive manufacturing.

Generates AM (time1) and SM (time2) layers from the neural network model,
then interleaves them by time value to produce the manufacturing sequence.
"""
import numpy as np
import torch
import pyvista as pv
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


@dataclass
class Layer:
    """A single manufacturing layer."""
    mesh: pv.PolyData
    time_value: float
    layer_type: str          # 'AM' or 'SM'
    index: int               # sequential index within its type
    global_index: int = -1   # set after interleaving


# ---------------------------------------------------------------------------
# Time-field grid generation (runs in model space)
# ---------------------------------------------------------------------------

def _compute_grid_shape(spaceBox, resolution):
    """
    Compute per-axis point counts so that every voxel is isotropic (cubic).

    ``resolution`` sets the number of sample points along the longest AABB
    edge; the other two axes are scaled proportionally so that the voxel
    edge length is identical in all three directions.

    Args:
        spaceBox: Model-space AABB as ``(2, 3)`` array.
        resolution: Number of grid samples along the longest axis.

    Returns:
        Tuple ``(nx, ny, nz)`` grid dimensions.
    """
    min_b = np.asarray(spaceBox[0], dtype=float)
    max_b = np.asarray(spaceBox[1], dtype=float)
    extents = max_b - min_b
    max_extent = float(np.max(extents))
    if max_extent < 1e-12:
        return resolution, resolution, resolution
    voxel_size = max_extent / (resolution - 1)
    nx = max(2, int(round(extents[0] / voxel_size)) + 1)
    ny = max(2, int(round(extents[1] / voxel_size)) + 1)
    nz = max(2, int(round(extents[2] / voxel_size)) + 1)
    return nx, ny, nz


def _generate_time_fields_grid(
    model, spaceBox, device, max_time, check_func, resolution, batch_size=8192,
):
    """Evaluate both AM/SM time fields on one shared 3-D grid pass.

    Args:
        model: Trained neural model supporting ``field_type='timesAndMasks'``.
        spaceBox: Model-space axis-aligned bounding box as ``(2, 3)`` array.
        device: Torch device used for inference.
        max_time: Scalar max process time used in time2 reconstruction.
        check_func: Callable that returns in-model mask for query points.
        resolution: Grid samples along the longest AABB axis.
        batch_size: Number of grid points per inference mini-batch.

    Returns:
        Tuple ``(time1_field, mask1_field, time2_field, mask2_field)`` where
        each field has shape ``(nx, ny, nz)``. Invalid voxels are ``np.nan`` in
        time fields and ``False`` in mask fields.
    """
    min_b, max_b = spaceBox[0], spaceBox[1]
    # Use uniform resolution for all axes to match resultsChecking/LayerGene
    nx = ny = nz = resolution
    x = np.linspace(float(min_b[0]), float(max_b[0]), nx)
    y = np.linspace(float(min_b[1]), float(max_b[1]), ny)
    z = np.linspace(float(min_b[2]), float(max_b[2]), nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    time1_values, mask1_values = [], []
    time2_values, mask2_values = [], []

    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            bp = torch.tensor(
                points[i:i + batch_size], dtype=torch.float32, device=device,
            )
            isInModel, _ = check_func(bp)
            f1, f2, fM1_raw, fM2_raw = model(bp, field_type='timesAndMasks')
            fM1 = torch.where(isInModel == 1, 5.0, fM1_raw)
            fM2 = torch.where(isInModel == 1, -5.0, fM2_raw)

            t1 = f1.squeeze()
            p1 = torch.sigmoid(fM1).squeeze()
            mask1 = p1 >= 0.5

            t2 = (f1 + f2 * (max_time - f1)).squeeze()
            p2 = torch.sigmoid(fM2).squeeze()
            mask2 = (p1 >= 0.5) & (p2 >= 0.5)

            time1_values.append(t1.cpu().numpy())
            mask1_values.append(mask1.cpu().numpy())
            time2_values.append(t2.cpu().numpy())
            mask2_values.append(mask2.cpu().numpy())

    time1_field = np.concatenate(time1_values).reshape(nx, ny, nz)
    mask1_field = np.concatenate(mask1_values).reshape(nx, ny, nz)
    time1_field = np.where(mask1_field, time1_field, np.nan)

    time2_field = np.concatenate(time2_values).reshape(nx, ny, nz)
    mask2_field = np.concatenate(mask2_values).reshape(nx, ny, nz)
    time2_field = np.where(mask2_field, time2_field, np.nan)
    return time1_field, mask1_field, time2_field, mask2_field


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _setup_grid(spaceBox, time_field, time_filter):
    """Build a PyVista structured grid for contour extraction.

    Args:
        spaceBox: Model-space AABB as ``(2, 3)`` array.
        time_field: Dense scalar field ``(nx, ny, nz)``.
        time_filter: Boolean mask ``(nx, ny, nz)`` selecting valid voxels.

    Returns:
        Tuple ``(grid, min_val, max_val)``. If no valid voxels exist, returns
        ``(None, None, None)``.
    """
    filtered = np.where(time_filter, time_field, np.nan)
    valid = filtered[~np.isnan(filtered)]
    if len(valid) == 0:
        return None, None, None
    min_val, max_val = np.min(valid), np.max(valid)
    min_b, max_b = spaceBox[0], spaceBox[1]
    nx, ny, nz = time_field.shape  # uniform resolution from _generate_time_fields_grid
    x = np.linspace(float(min_b[0]), float(max_b[0]), nx)
    y = np.linspace(float(min_b[1]), float(max_b[1]), ny)
    z = np.linspace(float(min_b[2]), float(max_b[2]), nz)
    grid = pv.StructuredGrid()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid.points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    grid.dimensions = (nx, ny, nz)
    grid.point_data['time_field'] = filtered.flatten()
    return grid, min_val, max_val


def _get_base_points(spaceBox, layer_type, resolution):
    """Create base anchor points for initial layer distance constraints.

    Args:
        spaceBox: Model-space AABB as ``(2, 3)`` array.
        layer_type: Layer family string, ``'AM'`` or ``'SM'``.
        resolution: Grid resolution (used for both X and Y).

    Returns:
        For AM: ``(resolution*resolution, 3)`` points on the AABB minimum-Z plane.
        For SM: ``None`` (no base plane anchor is used).
    """
    min_b, max_b = spaceBox[0], spaceBox[1]
    if layer_type == 'AM':
        x = np.linspace(float(min_b[0]), float(max_b[0]), resolution)
        y = np.linspace(float(min_b[1]), float(max_b[1]), resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, float(min_b[2]))
        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    return None


def _filter_fragments(mesh, min_size):
    """Drop disconnected mesh fragments smaller than ``min_size`` points.

    Args:
        mesh: Input ``pv.PolyData`` contour mesh.
        min_size: Minimum vertex count per connected component.

    Returns:
        Filtered mesh, or ``None`` when no component survives.
    """
    if mesh is None or mesh.n_points < min_size:
        return None
    try:
        conn = mesh.connectivity(largest=False)
        region_ids = conn['RegionId']
        unique_ids, counts = np.unique(region_ids, return_counts=True)
        valid = unique_ids[counts >= min_size]
        if len(valid) == 0:
            return None
        if len(valid) == len(unique_ids):
            return mesh
        return mesh.extract_points(np.isin(region_ids, valid))
    except Exception:
        return mesh


def _sanitize_mesh(mesh):
    """Remove vertices with NaN or Inf coordinates from a PyVista mesh.

    ``pv.StructuredGrid.contour`` and ``pv.PolyData.extract_points`` can
    produce non-finite coordinates when the underlying scalar field contains
    NaN sentinels (used here to mark invalid / out-of-mask voxels).  This
    helper strips those degenerate vertices so downstream code (cKDTree,
    Shapely, etc.) never sees non-finite data.

    Args:
        mesh: ``pv.PolyData`` mesh to sanitize.

    Returns:
        Cleaned mesh, or ``None`` if no finite points remain.
    """
    if mesh is None or mesh.n_points == 0:
        return mesh
    finite_mask = np.all(np.isfinite(mesh.points), axis=1)
    if np.all(finite_mask):
        return mesh
    if not np.any(finite_mask):
        return None
    try:
        # Use adjacent_cells=True (the default) so that all faces whose vertices
        # lie within the finite mask are preserved.  Using adjacent_cells=False
        # would drop every face that has even one non-selected vertex, which in
        # practice removes *all* faces from a marching-cubes mesh and leaves an
        # empty point cloud that cannot be rendered as a surface.
        cleaned = mesh.extract_points(finite_mask, adjacent_cells=True)
        if cleaned is None or cleaned.n_points == 0:
            return None
        # Second pass: guard against residual NaN points that may appear via
        # face connectivity on some PyVista versions.
        fm2 = np.all(np.isfinite(cleaned.points), axis=1)
        if np.all(fm2):
            return cleaned
        if not np.any(fm2):
            return None
        return cleaned.extract_points(fm2, adjacent_cells=True)
    except Exception:
        return None


def _strip_degenerate_cells(mesh):
    """Convert mesh to a clean PolyData surface, removing all non-face cells.

    ``extract_points()`` can produce ``UnstructuredGrid`` with mixed cell types
    (triangles + lines).  The most robust fix is ``extract_surface()`` which
    keeps only the outer surface faces as ``PolyData``.
    Additionally, we reconstruct the mesh using ONLY its faces (dropping any VTK
    line cells that might have survived) and filter out zero-area skinny triangles.

    Args:
        mesh: Any PyVista mesh.

    Returns:
        Cleaned ``PolyData`` with only surface cells, or ``None``.
    """
    if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
        return None
    try:
        # 1. extract_surface gets outer boundary
        surf = mesh.extract_surface()
        if surf is None or surf.n_points == 0 or surf.n_cells == 0:
            return None

        # 2. Force drop ALL line and vertex cells, keeping ONLY faces (polygons)
        # By passing only points and faces to PolyData, any lines/verts are destroyed.
        if surf.n_lines > 0 or surf.n_verts > 0 or surf.n_strips > 0:
            surf = pv.PolyData(surf.points, faces=surf.faces).clean()

        if surf.n_points == 0 or surf.n_cells == 0:
            return None

        # 3. Filter out zero-area / degenerate skinny triangles (which look like stray lines)
        areas = surf.compute_cell_sizes(length=False, area=True, volume=False)
        valid = areas['Area'] > 1e-8
        if not np.any(valid):
            return None
        if not np.all(valid):
            surf = surf.extract_cells(valid).extract_surface().clean()

        if surf.n_points == 0 or surf.n_cells == 0:
            return None

        return surf
    except Exception:
        return mesh if mesh.n_cells > 0 else None


def _adaptive_coarse_values(time_field, min_val, max_val, count, alpha=0.5):
    """Generate hybrid sampling values based on the time field distribution.

    Splits the total sample budget into a uniform portion and an empirical-CDF
    adaptive portion, then merges the two sets. This preserves some true
    uniform coverage for sparse regions while still densifying heavily occupied
    time ranges.

    * ``alpha=0.0``: purely uniform sampling (guarantees even time steps).
    * ``alpha=1.0``: purely adaptive sampling (can miss sparse top layers).
    * ``alpha=0.5``: hybrid sampling (recommended), catches top layers while
      keeping denser sampling in detailed regions.

    Args:
        time_field: 3D ``(nx, ny, nz)`` NumPy array with NaN for masked voxels.
        min_val: Minimum valid time value.
        max_val: Maximum valid time value.
        count: Total number of sample values to generate.
        alpha: Budget split between uniform (0) and adaptive (1).

    Returns:
        1D NumPy array of *count* sorted time values.
    """
    uniform_values = np.linspace(min_val, max_val, count)
    if count <= 2:
        return uniform_values
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return uniform_values

    valid = time_field[~np.isnan(time_field)]
    if len(valid) < count or max_val - min_val < 1e-12:
        return uniform_values

    n_uniform = max(2, int(round((1.0 - alpha) * count)))
    n_uniform = min(n_uniform, count)
    n_adaptive = max(0, count - n_uniform)

    uniform_values = np.linspace(min_val, max_val, n_uniform)
    adaptive_values = np.empty((0,), dtype=np.float64)

    if n_adaptive > 0:
        # Build CDF via histogram
        n_bins = max(count * 4, 500)
        hist, bin_edges = np.histogram(valid, bins=n_bins, range=(min_val, max_val))
        cdf = np.cumsum(hist).astype(np.float64)
        if cdf[-1] > 0.0:
            cdf /= cdf[-1]  # normalize to [0, 1]

            # Uniform quantiles in CDF space -> non-uniform values in time space
            quantiles = np.linspace(0.0, 1.0, n_adaptive)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Interpolate adaptive values
            adaptive_values = np.interp(quantiles, cdf, bin_centers)

    values = np.concatenate([uniform_values, adaptive_values])
    values = np.unique(np.concatenate([values, [min_val, max_val]]))

    if len(values) < count:
        dense_count = max(count * 4, count + 2)
        while len(values) < count:
            refill_pool = np.linspace(min_val, max_val, dense_count)[1:-1]
            refill_pool = np.setdiff1d(refill_pool, values, assume_unique=False)
            if len(refill_pool) == 0:
                break
            missing = count - len(values)
            take = min(missing, len(refill_pool))
            refill_idx = np.linspace(0, len(refill_pool) - 1, take, dtype=int)
            values = np.sort(np.concatenate([values, refill_pool[refill_idx]]))
            dense_count *= 2

    # if len(values) < count:
    #     values = np.linspace(min_val, max_val, count)
    # elif len(values) > count:
    #     inner = values[1:-1]
    #     keep_inner = count - 2
    #     if keep_inner <= 0:
    #         values = np.array([min_val, max_val], dtype=np.float64)
    #     elif len(inner) > keep_inner:
    #         keep_idx = np.linspace(0, len(inner) - 1, keep_inner, dtype=int)
    #         values = np.concatenate(([min_val], inner[keep_idx], [max_val]))

    values[0] = min_val
    values[-1] = max_val
    if len(values) >= 2:
        extra_bottom_layers = np.linspace(values[0], values[1], 52, dtype=np.float64)[1:-1]
        values = np.concatenate([values[:1], extra_bottom_layers, values[1:]])
    if len(values) >= 2:
        extra_top_layers = np.linspace(values[-2], values[-1], 52, dtype=np.float64)[1:-1]
        values = np.concatenate([values[:-1], extra_top_layers, values[-1:]])

    return values


def _extract_valid_components(mesh, valid_mask, min_fragment_size, keep_threshold=0.5):
    """
    Component-aware point extraction with a hybrid keep/discard strategy.

    For each connected component of *mesh*:
      - If the fraction of points satisfying *valid_mask* >= keep_threshold,
        extract only the valid points of that component (point-level trimming,
        may produce jagged edges at the boundary).
      - Otherwise discard the entire component.

    This prevents large, mostly-valid regions from being discarded entirely
    just because a minority of their boundary points are too close to a
    neighbouring layer, while still cleanly rejecting regions that are
    predominantly invalid.

    Args:
        mesh: Input contour mesh with ``mesh.n_points`` vertices.
        valid_mask: Boolean array ``(mesh.n_points,)``; ``True`` means keepable.
        min_fragment_size: Minimum component size to consider.
        keep_threshold: Component-level valid ratio threshold in ``[0, 1]``.

    Returns:
        Filtered mesh containing retained points, or ``None`` if empty.
    """
    if mesh is None or mesh.n_points == 0:
        return None
    if not np.any(valid_mask):
        return None
    if np.all(valid_mask):
        return _sanitize_mesh(mesh)
    try:
        conn = mesh.connectivity(largest=False)
        region_ids = conn['RegionId']
        keep_pts = np.zeros(mesh.n_points, dtype=bool)
        for rid in np.unique(region_ids):
            in_region = region_ids == rid
            if np.sum(in_region) < max(min_fragment_size, 1):
                continue
            valid_frac = np.sum(valid_mask[in_region]) / np.sum(in_region)
            if valid_frac >= keep_threshold:
                keep_pts |= (in_region & valid_mask)
        if not np.any(keep_pts):
            return None
        if np.all(keep_pts):
            return _sanitize_mesh(mesh)
        return _sanitize_mesh(mesh.extract_points(keep_pts))
    except Exception:
        result = mesh.extract_points(valid_mask)
        if result is None or result.n_points == 0:
            return None
        return _sanitize_mesh(result)


def _filter_small_layers(layers: list, min_points: int) -> list:
    """Remove layers whose mesh has fewer than *min_points* vertices.

    Args:
        layers: List of ``Layer`` objects.
        min_points: Minimum ``mesh.n_points`` required to keep a layer.

    Returns:
        Filtered ``Layer`` list with contiguous per-type ``index`` reassigned.
    """
    if min_points <= 0:
        return layers
    kept = [l for l in layers if l.mesh.n_points >= min_points]
    for i, l in enumerate(kept):
        l.index = i + 1
    return kept


def _compute_min_distance_to_layers(query_pts, layer_trees):
    """Compute nearest-neighbor distance to any previous layer tree.

    Args:
        query_pts: Query points array of shape ``(N, 3)``.
        layer_trees: List of ``cKDTree`` objects built from earlier layers.

    Returns:
        ``(N,)`` distance array. Returns ``np.inf`` when no trees are provided.
    """
    if len(layer_trees) == 0:
        return np.full(len(query_pts), np.inf)
    dists = np.full(len(query_pts), np.inf)
    for tree in layer_trees:
        try:
            d, _ = tree.query(query_pts, k=1, workers=-1)
        except TypeError:
            # Fallback for older SciPy versions that do not support workers.
            d, _ = tree.query(query_pts, k=1)
        dists = np.minimum(dists, d)
    return dists


def _sample_indices(indices, max_samples, rng):
    """Subsample global point indices without replacement."""
    if len(indices) <= max_samples:
        return indices
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def _approximate_dual_distance_mask(
    points, finite_mask, trees_a, trees_b, min_h, max_samples, rng,
):
    """Approximate a two-sided distance-valid mask from sampled contour points.

    Distance checks are evaluated on a sampled subset, then propagated to the
    full finite point set using nearest sampled neighbors. This trades some
    accuracy for a large reduction in KD-tree query cost during refinement.
    """
    valid_mask = np.zeros(points.shape[0], dtype=bool)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        return valid_mask

    sample_indices = _sample_indices(finite_indices, max_samples, rng)
    sample_points = points[sample_indices]
    d_a_sample = _compute_min_distance_to_layers(sample_points, trees_a)
    d_b_sample = _compute_min_distance_to_layers(sample_points, trees_b)
    sample_valid = (d_a_sample >= min_h) & (d_b_sample >= min_h)

    if len(sample_indices) == len(finite_indices):
        valid_mask[sample_indices] = sample_valid
        return valid_mask

    sample_tree = cKDTree(sample_points)
    try:
        _, nn = sample_tree.query(points[finite_indices], k=1, workers=-1)
    except TypeError:
        _, nn = sample_tree.query(points[finite_indices], k=1)
    valid_mask[finite_indices] = sample_valid[nn]
    return valid_mask


# ---------------------------------------------------------------------------
# Skeleton-then-fill layer generation (ported from LayerGene)
# ---------------------------------------------------------------------------

def _generate_skeleton_layers(
    spaceBox, time_field, mask_field, min_h, max_h, layer_type,
    coarse_layer_count=300, min_fragment_size=10, coverage_ratio=0.3,
):
    """
    Two-phase layer generation: skeleton -> refinement.

    Args:
        spaceBox: Model-space AABB as ``(2, 3)`` array.
        time_field: Scalar field ``(nx, ny, nz)`` used for contouring.
        mask_field: Boolean validity mask ``(nx, ny, nz)``.
        min_h: Minimum allowed local distance to previous layers.
        max_h: Maximum allowed local distance before gap refinement.
        layer_type: ``'AM'`` or ``'SM'``.
        coarse_layer_count: Number of coarse candidate iso-values.
        min_fragment_size: Minimum component size to keep.
        coverage_ratio: Fraction of original contour points that must survive
            distance filtering for a skeleton layer to be accepted.  Set to 0.0
            to disable the coverage check entirely.  Default 0.5.

    Returns:
        List of tuples ``(mesh, time_value, index)`` in model space.
    """
    if cKDTree is None:
        raise ImportError("scipy.spatial.cKDTree is required")
    if coarse_layer_count < 2:
        raise ValueError(f"coarse_layer_count must be >= 2, got {coarse_layer_count}")
    if min_h <= 0:
        raise ValueError(f"min_h must be > 0, got {min_h}")
    if max_h <= 0:
        raise ValueError(f"max_h must be > 0, got {max_h}")
    if min_fragment_size < 0:
        raise ValueError(f"min_fragment_size must be >= 0, got {min_fragment_size}")
    if max_h < 2 * min_h:
        max_h = max(max_h, 2 * min_h)

    grid, min_val, max_val = _setup_grid(spaceBox, time_field, mask_field)
    if grid is None:
        return []

    max_lookback = 10
    skeleton_nodes = []
    all_prev_trees = []

    resolution = time_field.shape[0]  # uniform resolution
    base_pts = _get_base_points(spaceBox, layer_type, resolution)
    if base_pts is not None:
        finite = base_pts[np.all(np.isfinite(base_pts), axis=1)]
        base_tree = cKDTree(finite) if len(finite) else None
        skeleton_nodes.append({
            'mesh': None, 'val': min_val, 'pts': base_pts,
            'is_base': True, 'skeleton_id': 0, 'tree': base_tree,
        })
        if base_tree is not None:
            all_prev_trees.append(base_tree)

    # Use hybrid sampling (alpha=0.5 by default) to get both dense geometry coverage
    # and guarantee we don't miss the top/bottom sparse regions.
    coarse_values = _adaptive_coarse_values(time_field, min_val, max_val, coarse_layer_count, alpha=0.4)
    sk_id = 1
    max_phase_errors = 20
    skeleton_error_count = 0

    print(f"Phase 1: Generating {layer_type} skeleton layers ...")
    for val in tqdm(coarse_values, desc=f'{layer_type} skeleton'):
        try:
            contour = grid.contour(isosurfaces=[val], scalars='time_field')
            if contour is None or contour.n_points == 0:
                continue
            pts = contour.points
            fm = np.all(np.isfinite(pts), axis=1)
            if not np.any(fm):
                continue
            fp = pts[fm]

            if len(all_prev_trees) == 0 and layer_type == 'SM':
                contour = _filter_fragments(contour, min_fragment_size)
                contour = _sanitize_mesh(contour)
                if contour and contour.n_points > 0:
                    fp_c = contour.points[np.all(np.isfinite(contour.points), axis=1)]
                    fp_tree = cKDTree(fp_c) if len(fp_c) > 0 else None
                    skeleton_nodes.append({
                        'mesh': contour, 'val': val, 'pts': contour.points,
                        'is_base': False, 'skeleton_id': sk_id, 'tree': fp_tree,
                    })
                    sk_id += 1
                    if fp_tree is not None:
                        all_prev_trees.append(fp_tree)
                continue

            if len(all_prev_trees) == 0:
                continue

            dists = np.full(contour.n_points, -1.0)
            dists[fm] = _compute_min_distance_to_layers(fp, all_prev_trees)
            valid_mask = dists >= min_h
            if not np.any(valid_mask):
                continue

            # Same logic as original create_skeleton_based_level_sets:
            # 1. Simple point extraction (not component-aware)
            if np.all(valid_mask):
                part = contour
            else:
                part = contour.extract_points(valid_mask)

            # 2. Filter small fragments (skip for last value)
            frag_min = min_fragment_size if val != coarse_values[-1] else 1
            part = _filter_fragments(part, frag_min)
            if part is None or part.n_points == 0:
                continue

            # 3. Sanitize NaN/Inf points
            part = _sanitize_mesh(part)
            if part is None or part.n_points == 0:
                continue

            # 4. Coverage ratio check: 
            if val != coarse_values[-1] and coverage_ratio > 0.0:
                fm_part = np.all(np.isfinite(part.points), axis=1)
                num_finite_part = np.sum(fm_part)
                num_finite_all = np.sum(fm)
                ratio_finite = num_finite_part / num_finite_all if num_finite_all > 0 else 0
                if ratio_finite < coverage_ratio:
                    continue

            fp2 = part.points[np.all(np.isfinite(part.points), axis=1)]
            fp2_tree = cKDTree(fp2) if len(fp2) > 0 else None
            skeleton_nodes.append({
                'mesh': part, 'val': val, 'pts': part.points,
                'is_base': False, 'skeleton_id': sk_id, 'tree': fp2_tree,
            })
            sk_id += 1
            if fp2_tree is not None:
                all_prev_trees.append(fp2_tree)
            if len(all_prev_trees) > max_lookback:
                all_prev_trees.pop(0)
        except Exception as e:
            skeleton_error_count += 1
            print(
                f"  skeleton error at val={val:.6f} "
                f"({skeleton_error_count}/{max_phase_errors}): {e}"
            )
            if skeleton_error_count >= max_phase_errors:
                raise RuntimeError(
                    f"Too many skeleton errors ({skeleton_error_count})"
                ) from e
            continue

    if layer_type == 'AM':
        top_nodes = [
            node for node in skeleton_nodes
            if not node.get('is_base', False)
            and node.get('mesh') is not None
            and node['mesh'].n_points > 0
        ]
        if top_nodes:
            top_node = max(top_nodes, key=lambda node: float(node['mesh'].bounds[5]))
            top_z = float(top_node['mesh'].bounds[5])
            bbox_top_z = float(spaceBox[1][2])
            top_gap = bbox_top_z - top_z
            z_eps = max(1e-5, 0.1 * min_h)

            def _build_top_candidate(val, current_top_z):
                nonlocal skeleton_error_count
                try:
                    contour = grid.contour(isosurfaces=[val], scalars='time_field')
                    if contour is None or contour.n_points == 0:
                        return None
                    pts = contour.points
                    fm = np.all(np.isfinite(pts), axis=1)
                    if not np.any(fm):
                        return None
                    fp = pts[fm]

                    if len(all_prev_trees) == 0:
                        return None

                    dists = np.full(contour.n_points, -1.0)
                    dists[fm] = _compute_min_distance_to_layers(fp, all_prev_trees)
                    valid_mask = dists >= min_h
                    if not np.any(valid_mask):
                        return None

                    if np.all(valid_mask):
                        part = contour
                    else:
                        part = contour.extract_points(valid_mask)

                    part = _filter_fragments(part, 1)
                    if part is None or part.n_points == 0:
                        return None

                    part = _sanitize_mesh(part)
                    if part is None or part.n_points == 0:
                        return None

                    part_zmax = float(part.bounds[5])
                    if part_zmax <= current_top_z + z_eps:
                        return None

                    return {
                        'mesh': part,
                        'val': val,
                        'pts': part.points,
                        'zmax': part_zmax,
                        'gap': bbox_top_z - part_zmax,
                    }
                except Exception as e:
                    skeleton_error_count += 1
                    print(
                        f"  top skeleton error at val={val:.6f} "
                        f"({skeleton_error_count}/{max_phase_errors}): {e}"
                    )
                    if skeleton_error_count >= max_phase_errors:
                        raise RuntimeError(
                            f"Too many skeleton errors ({skeleton_error_count})"
                        ) from e
                    return None

            if top_gap > max_h and top_node['val'] < max_val:
                print(
                    f"  AM top-gap detected before refine: "
                    f"gap={top_gap:.6f}, incrementally adding highest valid top layers ..."
                )
                extra_added = 0
                while (bbox_top_z - top_z) > max_h and top_node['val'] < max_val:
                    top_candidates = np.linspace(top_node['val'], max_val, 101)[1:]
                    chosen_candidate = None

                    for val in top_candidates[::-1]:
                        cand = _build_top_candidate(val, top_z)
                        if cand is None:
                            continue
                        chosen_candidate = cand
                        break

                    if chosen_candidate is None:
                        break

                    fp2 = chosen_candidate['pts'][np.all(np.isfinite(chosen_candidate['pts']), axis=1)]
                    fp2_tree = cKDTree(fp2) if len(fp2) > 0 else None
                    new_node = {
                        'mesh': chosen_candidate['mesh'],
                        'val': chosen_candidate['val'],
                        'pts': chosen_candidate['pts'],
                        'is_base': False,
                        'skeleton_id': sk_id,
                        'tree': fp2_tree,
                    }
                    skeleton_nodes.append(new_node)
                    sk_id += 1
                    if fp2_tree is not None:
                        all_prev_trees.append(fp2_tree)
                    if len(all_prev_trees) > max_lookback:
                        all_prev_trees.pop(0)

                    top_node = new_node
                    top_z = chosen_candidate['zmax']
                    extra_added += 1
                    print(
                        f"  Added AM top skeleton layer at val={chosen_candidate['val']:.6f}. "
                        f"Current top gap={max(0.0, chosen_candidate['gap']):.6f}"
                    )

                if extra_added > 0:
                    skeleton_nodes.sort(key=lambda node: float(node['val']))
                print(
                    f"  AM top insertion finished. Current top gap={max(0.0, bbox_top_z - top_z):.6f}"
                )

    # Phase 2 refinement is currently enabled for AM only.
    enable_refine = True

    if layer_type != 'SM' and enable_refine:
        print(f"  {len(skeleton_nodes)} skeleton nodes. Phase 2: refining ...")

        # --- Phase 2: fill gaps ---
        for node in skeleton_nodes:
            node['parent_a'] = node['skeleton_id']
            node['parent_b'] = node['skeleton_id']
            node['depth'] = 0

        orig_count = len(skeleton_nodes)
        processed_gaps = set()
        max_depth = 5
        gap_sample_size = 4096
        vm_sample_size = 4096
        rng = np.random.default_rng(0)
        refine_error_count = 0
        pbar = tqdm(total=max(orig_count - 1, 1), desc=f'{layer_type} refine')
        i = 0

        while i < len(skeleton_nodes) - 1:
            na, nb = skeleton_nodes[i], skeleton_nodes[i + 1]
            pa, pb = na['pts'], nb['pts']
            if pa is None or pa.size == 0 or pb is None or pb.size == 0:
                i += 1
                continue

            fma = np.all(np.isfinite(pa), axis=1)
            fmb = np.all(np.isfinite(pb), axis=1)
            if not np.any(fma) or not np.any(fmb):
                i += 1
                continue

            trees_a = [
                skeleton_nodes[j]['tree']
                for j in range(max(0, i + 1 - max_lookback), i + 1)
                if skeleton_nodes[j].get('tree') is not None
            ]
            trees_b = [
                skeleton_nodes[j]['tree']
                for j in range(i + 1, min(len(skeleton_nodes), i + 1 + max_lookback))
                if skeleton_nodes[j].get('tree') is not None
            ]
            # Refinement requires valid constraints from both sides.
            # If either side has no usable tree, skip this pair to avoid
            # unconstrained insertions.
            if len(trees_a) == 0 or len(trees_b) == 0:
                i += 1
                continue

            pb_finite_idx = np.flatnonzero(fmb)
            pb_sample_idx = _sample_indices(pb_finite_idx, gap_sample_size, rng)
            dists_ba_sample = _compute_min_distance_to_layers(pb[pb_sample_idx], trees_a)
            gap_ratio = (
                np.count_nonzero(dists_ba_sample > max_h) / len(pb_sample_idx)
                if len(pb_sample_idx) > 0 else 0.0
            )
            est_gap_points = gap_ratio * len(pb_finite_idx)
            is_gap = est_gap_points > min_fragment_size
            sa = na.get('parent_a', na.get('skeleton_id'))
            sb = nb.get('parent_b', nb.get('skeleton_id'))
            gap_key = f"{sa}_{sb}"

            if gap_key not in processed_gaps and na.get('skeleton_id') is not None and nb.get('skeleton_id') is not None:
                processed_gaps.add(gap_key)
                pbar.update(1)

            if is_gap:
                if max(na.get('depth', 0), nb.get('depth', 0)) >= max_depth:
                    is_gap = False
                if abs(nb['val'] - na['val']) < 1e-5:
                    is_gap = False

            if is_gap:
                mid_val = (na['val'] + nb['val']) / 2
                try:
                    mc = grid.contour(isosurfaces=[mid_val], scalars='time_field')
                    if mc.n_points == 0:
                        i += 1; continue
                    fm_mc = np.all(np.isfinite(mc.points), axis=1)
                    if not np.any(fm_mc):
                        i += 1; continue

                    vm = _approximate_dual_distance_mask(
                        mc.points, fm_mc, trees_a, trees_b, min_h,
                        vm_sample_size, rng,
                    )
                    if np.any(vm):
                        # Same as original: simple extract_points + filter_fragments
                        if np.all(vm):
                            part = mc
                        else:
                            part = mc.extract_points(vm)
                        part = _filter_fragments(part, min_fragment_size)
                        part = _sanitize_mesh(part)
                        if part is not None and part.n_points > 0:
                            fp_part = part.points[np.all(np.isfinite(part.points), axis=1)]
                            part_tree = cKDTree(fp_part) if len(fp_part) > 0 else None
                            mid_node = {
                                'mesh': part, 'val': mid_val, 'pts': part.points,
                                'is_base': False, 'skeleton_id': None,
                                'parent_a': sa, 'parent_b': sb, 'tree': part_tree,
                                'depth': max(na.get('depth', 0), nb.get('depth', 0)) + 1,
                            }
                            skeleton_nodes.insert(i + 1, mid_node)
                            continue
                except Exception as e:
                    refine_error_count += 1
                    print(
                        f"  refine error at mid_val={mid_val:.6f} "
                        f"({refine_error_count}/{max_phase_errors}): {e}"
                    )
                    if refine_error_count >= max_phase_errors:
                        raise RuntimeError(
                            f"Too many refine errors ({refine_error_count})"
                        ) from e
            i += 1
        pbar.close()
    else:
        print(
            f"  {len(skeleton_nodes)} skeleton nodes. "
            f"Skipping Phase 2 (refining) for {layer_type} layers."
        )

    layers = []
    for node in skeleton_nodes:
        if node.get('is_base', False):
            continue
        mesh = node['mesh']
        if mesh is None or mesh.n_points == 0:
            continue
        # Post-process: strip degenerate line cells that render as lines
        mesh = _strip_degenerate_cells(mesh)
        if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
            continue  # skip empty layers entirely

        # Enhancement Pass: Fix topological issues and smooth geometry
        try:
            # 1. Deep clean: merge extremely close points and remove degenerate faces
            mesh = mesh.clean(tolerance=1e-5)
            # 2. Re-filter fragments: clean/strip might have disconnected small triangles
            mesh = _filter_fragments(mesh, min_fragment_size)
            if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
                continue
            # 3. Cast back to PolyData (filter_fragments returns UnstructuredGrid)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
                continue
            # 4. Taubin smooth: removes marching cubes terracing/aliasing with minimal shrinkage
            mesh = mesh.smooth_taubin(n_iter=20, pass_band=0.05)
            # 5. Compute normals: required for correct rendering and downstream collision checks
            mesh = mesh.compute_normals(cell_normals=True, point_normals=True)
        except Exception as e:
            print(f"Warning: layer mesh enhancement failed for val {node['val']}: {e}")

        if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
            continue

        mesh['level_value'] = np.full(mesh.n_points, node['val'])
        layers.append((mesh, node['val'], len(layers) + 1))
    return layers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_layers(
    ctx,
    resolution: int = 128,
    am_coarse_count: int = 300,
    sm_coarse_count: int = 300,
    am_min_fragment_size: int = 10,
    sm_min_fragment_size: int = 10,
    am_max_layer_height: Optional[float] = None,
    sm_max_layer_height: Optional[float] = None,
    min_layer_points: int = 0,
    build_am: bool = True,
    build_sm: bool = True,
) -> Tuple[List[Layer], List[Layer]]:
    """
    Generate AM and SM layers from the neural network model.
    All mesh coordinates remain in model space (scaled).

    Args:
        ctx: PostprocessContext.
        resolution: Grid resolution for marching-cubes isosurface extraction.
        am_coarse_count: Number of candidate time-steps for AM skeleton generation.
        sm_coarse_count: Number of candidate time-steps for SM skeleton generation.
        am_min_fragment_size: Minimum vertex count to keep a disconnected AM mesh
            fragment (island). Filters out tiny noise artifacts. Default 10.
        sm_min_fragment_size: Minimum vertex count to keep a disconnected SM mesh
            fragment. Can be set smaller than AM when SM layers are naturally finer.
            Default 10.
        am_max_layer_height: Maximum allowed gap (in model space) between consecutive
            AM layers.  If a local distance exceeds this value an intermediate layer
            is inserted adaptively.  Overrides ``manu_config['max_layer_AM']`` when
            provided.
        sm_max_layer_height: Same as above for SM layers; overrides
            ``manu_config['max_layer_SM']`` when provided.
        min_layer_points: After all layers are generated, discard any layer whose
            mesh has fewer than this many vertices.  Set to 0 (default) to keep all
            layers.
        build_am: Whether to build AM layers.
        build_sm: Whether to build SM layers.

    Returns:
        (am_layers, sm_layers) each a list of Layer objects.
    """
    manu = ctx.manu_config
    if not build_am and not build_sm:
        return [], []

    # Read layer-height bounds from config, fall back to sensible defaults when the
    # key is absent (e.g. when MANU_CONFIG_CHECK is used and lacks these entries).
    am_min_h = manu.get('min_layer_AM', manu.get('max_layer_AM', 1.0) * 0.5)
    am_max_h = manu.get('max_layer_AM', am_min_h * 2.0)
    sm_min_h = manu.get('min_layer_SM', manu.get('max_layer_SM', 0.1) * 0.5)
    sm_max_h = manu.get('max_layer_SM', sm_min_h * 2.0)

    print("=== Generating time fields (AM + SM) ===")
    t1_field, t1_mask, t2_field, t2_mask = _generate_time_fields_grid(
        ctx.model, ctx.spaceBox, ctx.device, ctx.max_time, ctx.check_func,
        resolution,
    )

    am_layers: List[Layer] = []
    sm_layers: List[Layer] = []

    if build_am:
        print(f"=== Building AM layers (min_h={am_min_h:.4f}, max_h={am_max_h:.4f}, min_fragment={am_min_fragment_size}) ===")
        am_raw = _generate_skeleton_layers(
            ctx.spaceBox, t1_field, t1_mask,
            min_h=am_min_h, max_h=am_max_h,
            layer_type='AM',
            coarse_layer_count=am_coarse_count,
            min_fragment_size=am_min_fragment_size,
        )
        am_layers = [
            Layer(mesh=m, time_value=v, layer_type='AM', index=idx)
            for m, v, idx in am_raw
        ]

    if build_sm:
        print(f"=== Building SM layers (min_h={sm_min_h:.4f}, max_h={sm_max_h:.4f}, min_fragment={sm_min_fragment_size}) ===")
        sm_raw = _generate_skeleton_layers(
            ctx.spaceBox, t2_field, t2_mask,
            min_h=sm_min_h, max_h=sm_max_h,
            layer_type='SM',
            coarse_layer_count=sm_coarse_count,
            min_fragment_size=sm_min_fragment_size,
        )
        sm_layers = [
            Layer(mesh=m, time_value=v, layer_type='SM', index=idx)
            for m, v, idx in sm_raw
        ]

    print(f"Generated {len(am_layers)} AM layers, {len(sm_layers)} SM layers.")

    # Final pass: discard layers that are too small (entire layer, not just fragments).
    if min_layer_points > 0:
        am_before, sm_before = len(am_layers), len(sm_layers)
        am_layers = _filter_small_layers(am_layers, min_layer_points)
        sm_layers = _filter_small_layers(sm_layers, min_layer_points)
        print(
            f"After small-layer filter (min_points={min_layer_points}): "
            f"{len(am_layers)} AM ({am_before - len(am_layers)} removed), "
            f"{len(sm_layers)} SM ({sm_before - len(sm_layers)} removed)."
        )

    return am_layers, sm_layers


def interleave_layers(
    am_layers: List[Layer],
    sm_layers: List[Layer],
) -> List[Layer]:
    """
    Merge AM and SM layers sorted by time_value and assign global indices.

    Args:
        am_layers: AM layer list.
        sm_layers: SM layer list.

    Returns:
        Combined list sorted by ``time_value`` with ``global_index`` set.
    """
    # 强制保证合并后的前两层必须是 AM layer：
    # 找到第二个 AM layer 的时间，将此时间之前的 SM layer 删掉
    if len(am_layers) >= 2:
        cutoff_time = am_layers[1].time_value
        sm_layers = [sm for sm in sm_layers if sm.time_value >= cutoff_time]

    combined = list(am_layers) + list(sm_layers)
    combined.sort(key=lambda l: l.time_value)
    for gi, layer in enumerate(combined):
        layer.global_index = gi
    return combined


def scale_layer_mesh(layer: Layer, scale: float) -> Layer:
    """
    Return a copy of the layer whose mesh coordinates are multiplied by *scale*.

    Args:
        layer: Input layer in model-space coordinates.
        scale: Scalar conversion factor from model space to real space.

    Returns:
        New ``Layer`` object with scaled mesh points.
    """
    mesh_copy = layer.mesh.copy()
    mesh_copy.points = mesh_copy.points * scale
    return Layer(
        mesh=mesh_copy,
        time_value=layer.time_value,
        layer_type=layer.layer_type,
        index=layer.index,
        global_index=layer.global_index,
    )
