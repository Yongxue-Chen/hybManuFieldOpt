import pyvista as pv
import numpy as np
import trimesh
import os
import torch
from stl import mesh


_CUBE_VERTS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=float,
)

_CUBE_FACES = [
    [0, 3, 1],
    [1, 3, 2],  # bottom
    [4, 5, 7],
    [5, 6, 7],  # top
    [0, 1, 4],
    [1, 5, 4],  # front
    [1, 2, 5],
    [2, 6, 5],  # right
    [2, 3, 6],
    [3, 7, 6],  # back
    [3, 0, 7],
    [0, 4, 7],  # left
]

_VOXEL_NEIGHBORS = {
    "bottom": (0, 0, -1),
    "top": (0, 0, 1),
    "front": (0, -1, 0),
    "back": (0, 1, 0),
    "left": (-1, 0, 0),
    "right": (1, 0, 0),
}

_FACE_INDICES = {
    "bottom": [0, 1],
    "top": [2, 3],
    "front": [4, 5],
    "right": [6, 7],
    "back": [8, 9],
    "left": [10, 11],
}


def _pyvista_surface_to_trimesh(surface_mesh):
    """
    Convert a triangulated PyVista surface mesh to a Trimesh object.
    """
    surface_mesh = surface_mesh.extract_surface().triangulate()
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(
        vertices=np.asarray(surface_mesh.points),
        faces=faces,
        process=False,
    )


def _split_surface_bodies(surface_mesh):
    """
    Split a surface mesh into connected bodies.
    Returns a list of non-empty PyVista surface meshes.
    """
    split = surface_mesh.split_bodies()
    bodies = []

    n_blocks = split.n_blocks if hasattr(split, "n_blocks") else len(split)
    for i in range(n_blocks):
        block = split[i]
        if block is None or block.n_points == 0:
            continue
        body = block.extract_surface().triangulate()
        if body.n_points == 0:
            continue
        bodies.append(body)

    if not bodies:
        return [surface_mesh]

    return bodies


def _load_normalized_trimesh(stl_file):
    """
    Load an STL with trimesh and apply the same [0, 1] normalization used by
    create_voxelization.
    """
    scale, p_min_translation, _ = get_normalization_parameters(stl_file)
    mesh = trimesh.load_mesh(stl_file, force='mesh')
    mesh.apply_translation(-p_min_translation)
    mesh.apply_scale(scale)
    return mesh


def get_normalization_parameters(stl_file):
    """
    Calculate normalization parameters to scale a mesh to the [0, 1] range.
    Uses PyVista to read the mesh and calculate its bounds.

    Args:
        stl_file (str): Path to the input STL file.

    Returns:
        tuple:
            - scale (float): The scale factor for normalization.
            - translation (np.ndarray): The translation vector (p_min) for normalization.
    """
    mesh = pv.read(stl_file).extract_surface()
    bounds = mesh.bounds
    p_min = np.array([bounds[0], bounds[2], bounds[4]])
    dims = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    
    max_len = dims.max()
    if max_len < 1e-9:  # Prevent division by zero
        scale = 1.0
    else:
        scale = 1.0 / max_len
    
    newBounds = np.array([p_min, p_min + dims * scale])
        
    return scale, p_min, newBounds

def create_voxelization(stl_file, resolution=100):
    """
    Voxelize STL file using PyVista, with results in normalized [0,1] space.
    
    Args:
        stl_file (str): Path to input STL file.
        resolution (int): Voxelization resolution.
    
    Returns:
        tuple:
            - cropped_voxel (np.ndarray): Cropped 3D voxel matrix (0/1 values).
            - voxel_centers (np.ndarray): Normalized coordinates of voxel centers.
            - norm_bounds (np.ndarray): The bounds of the normalized mesh.
    """
    # Get normalization params to ensure consistency
    scale, p_min_translation, _ = get_normalization_parameters(stl_file)

    # Load and extract geometry with PyVista
    mesh = pv.read(stl_file).extract_geometry()

    # Normalize mesh to [0,1] range
    normalized_mesh = mesh.copy()
    normalized_mesh.translate(-p_min_translation, inplace=True)
    normalized_mesh.scale([scale, scale, scale], inplace=True)
    normalized_surface = normalized_mesh.extract_surface().triangulate()
    
    # Create voxel grid
    norm_bounds = normalized_surface.bounds
    spacing_val = 1.0 / resolution # since the mesh is normalized to [0,1] range, the spacing is 1/resolution
    spacing = (spacing_val, spacing_val, spacing_val)
    
    grid_origin = np.array([norm_bounds[0], norm_bounds[2], norm_bounds[4]])
    print(f"Grid origin: {grid_origin}")

    norm_dims_size = np.array([
        norm_bounds[1]-norm_bounds[0], 
        norm_bounds[3]-norm_bounds[2], 
        norm_bounds[5]-norm_bounds[4]
    ])
    dims = np.ceil(norm_dims_size / spacing_val).astype(int) + 1

    # Compute occupancy per connected body and merge by union.
    voxel_matrix = np.zeros((dims[2], dims[1], dims[0]), dtype=bool)
    bodies = _split_surface_bodies(normalized_surface)
    print(f"Detected {len(bodies)} connected bodies for voxelization.")

    for body in bodies:
        grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=grid_origin)
        grid.compute_implicit_distance(body, inplace=True)
        implicit_distance_flat = grid.point_data["implicit_distance"]
        voxel_matrix |= (implicit_distance_flat <= 0).reshape(dims[2], dims[1], dims[0])
    
    true_voxel_coords = np.argwhere(voxel_matrix)
    if true_voxel_coords.size == 0:
        print("Warning: No voxels found inside mesh.")
        empty_shape = (0, 0, 0)
        return np.zeros(empty_shape, dtype=np.int8), np.zeros(empty_shape + (3,)), np.array([0,0,0,0,0,0])

    min_c = true_voxel_coords.min(axis=0)
    max_c = true_voxel_coords.max(axis=0)
    cropped_voxel = voxel_matrix[
        min_c[0]:max_c[0]+1,
        min_c[1]:max_c[1]+1,
        min_c[2]:max_c[2]+1,
    ].astype(np.int8)

    # Calculate voxel center coordinates
    # Create index grids for the cropped voxel matrix dimensions (z, y, x)
    z_indices, y_indices, x_indices = np.indices(cropped_voxel.shape)

    # Add the offset from the original grid to get absolute indices
    z_indices += min_c[0]
    y_indices += min_c[1]
    x_indices += min_c[2]

    # Stack the indices in (x, y, z) order for coordinate calculation
    stacked_indices = np.stack([x_indices, y_indices, z_indices], axis=-1)

    # Calculate centers by scaling indices and adding the origin.
    # We add 0.5 to the indices to get the center of each voxel.
    voxel_centers = grid_origin + (stacked_indices + 0.5) * spacing

    return cropped_voxel, voxel_centers, norm_bounds


def create_voxel_surface_mesh(cropped_voxel, voxel_centers, spacing_val):
    """
    Build a blocky surface mesh from the occupied voxel cells.

    The returned mesh is in the same normalized coordinate system as
    voxel_centers.
    """
    triangles = create_voxel_surface_triangles(
        cropped_voxel,
        voxel_centers,
        spacing_val,
    )
    if triangles.size == 0:
        return None

    return _triangles_to_trimesh(triangles)


def _triangles_to_trimesh(triangles):
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def create_voxel_surface_triangles(cropped_voxel, voxel_centers, spacing_val):
    """
    Generate exposed voxel faces as STL triangles using the voxel2STL approach:
    visit each occupied voxel and skip faces shared with occupied neighbors.
    """
    if cropped_voxel.size == 0 or voxel_centers.size == 0:
        return np.empty((0, 3, 3), dtype=float)

    z_count, y_count, x_count = cropped_voxel.shape
    origin = voxel_centers[0, 0, 0] - 0.5 * spacing_val
    cube_verts = _CUBE_VERTS * spacing_val

    triangles = []
    occupied_coords = np.argwhere(cropped_voxel == 1)
    for z, y, x in occupied_coords:
        base = origin + np.array([x, y, z]) * spacing_val
        cube = cube_verts + base

        for direction, (dx, dy, dz) in _VOXEL_NEIGHBORS.items():
            nx_, ny_, nz_ = x + dx, y + dy, z + dz
            if (
                0 <= nx_ < x_count
                and 0 <= ny_ < y_count
                and 0 <= nz_ < z_count
                and cropped_voxel[nz_, ny_, nx_] == 1
            ):
                continue

            for face_index in _FACE_INDICES[direction]:
                triangles.append([cube[i] for i in _CUBE_FACES[face_index]])

    if not triangles:
        return np.empty((0, 3, 3), dtype=float)

    return np.array(triangles, dtype=float)


def write_voxel_surface_stl(cropped_voxel, voxel_centers, spacing_val, filename):
    """
    Write voxelized surface triangles to STL using numpy-stl, matching
    forFigures/voxel2STL.py.
    """
    triangles = create_voxel_surface_triangles(
        cropped_voxel,
        voxel_centers,
        spacing_val,
    )
    if triangles.size == 0:
        return None

    stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, triangle in enumerate(triangles):
        stl_mesh.vectors[i] = triangle

    stl_mesh.save(filename)
    return _triangles_to_trimesh(triangles)


def compute_voxel_surface_error(stl_file, voxel_surface, sample_count=50000):
    """
    Compute voxelized-surface -> original-surface distance in normalized space.

    The mean distance is area-sampled on the voxelized surface. The max distance
    is evaluated over those samples plus the voxel surface vertices.
    """
    if voxel_surface is None:
        return None

    original_mesh = _load_normalized_trimesh(stl_file)
    if isinstance(voxel_surface, trimesh.Trimesh):
        voxel_mesh = voxel_surface
    else:
        voxel_mesh = _pyvista_surface_to_trimesh(voxel_surface)

    if len(voxel_mesh.vertices) == 0 or len(voxel_mesh.faces) == 0:
        return None

    sample_count = max(0, int(sample_count))
    if sample_count > 0:
        sampled_points, _ = trimesh.sample.sample_surface(voxel_mesh, sample_count)
        mean_query_points = sampled_points
        max_query_points = np.vstack([sampled_points, voxel_mesh.vertices])
    else:
        mean_query_points = voxel_mesh.vertices
        max_query_points = voxel_mesh.vertices

    _, mean_distances, _ = trimesh.proximity.closest_point(
        original_mesh, mean_query_points
    )
    _, max_distances, _ = trimesh.proximity.closest_point(
        original_mesh, max_query_points
    )

    return {
        "mean_error": float(np.mean(mean_distances)),
        "max_error": float(np.max(max_distances)),
        "sample_count": int(len(mean_query_points)),
        "vertex_count": int(len(voxel_mesh.vertices)),
        "face_count": int(len(voxel_mesh.faces)),
    }


def writeVoxelMatrix(voxel_matrix, filename):
    boardSize = voxel_matrix.shape
    vecBoard = voxel_matrix.flatten()
    with open(filename, "w") as f:
        f.write(
            str(boardSize[2])
            + ","
            + str(boardSize[1])
            + ","
            + str(boardSize[0])
            + "\n"
        )
        f.write("0,0,0\n")
        for i in range(len(vecBoard) - 1):
            f.write(str(int(vecBoard[i])) + ",")
        f.write(str(int(vecBoard[-1])) + "\n")


def write_voxel_centers_txt(voxel_centers, filename, aabb_min, aabb_max):
    flat_centers = voxel_centers.reshape(-1, 3)
    sort_idx = np.lexsort((flat_centers[:,0], flat_centers[:,1], flat_centers[:,2]))
    sorted_centers = flat_centers[sort_idx]
    with open(filename, "w") as f:
        # First line writes the AABB bounds
        f.write(f"{aabb_min[0]:.6f},{aabb_min[1]:.6f},{aabb_min[2]:.6f},{aabb_max[0]:.6f},{aabb_max[1]:.6f},{aabb_max[2]:.6f}\n")
        for c in sorted_centers:
            f.write(f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f}\n")

if __name__ == "__main__":
    MODEL_NAME = 'bracket'
    resolution = 140  # Voxelization resolution

    stl_path = f"stlFiles/{MODEL_NAME}.stl"

    print(f"Processing {stl_path} with resolution {resolution}...")
    
    # Generate voxel data and get the normalized bounds
    cropped_voxel, voxel_centers, norm_bounds = create_voxelization(stl_path, resolution)

    # --- Create output directory and define paths ---
    output_dir = f"initialFields/{MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(stl_path))[0]
    
    voxel_matrix_path = os.path.join(output_dir, f"{base_filename}_res{resolution}_voxels.txt")
    voxel_centers_path = os.path.join(output_dir, f"{base_filename}_res{resolution}_centers.txt")
    voxel_surface_path = os.path.join(output_dir, f"{base_filename}_res{resolution}_voxel_surface.stl")

    # --- Write Voxel Matrix ---
    if cropped_voxel.size > 0:
        print(f"Writing voxel matrix to {voxel_matrix_path}...")
        writeVoxelMatrix(cropped_voxel, voxel_matrix_path)
    else:
        print("Voxel matrix is empty, not writing to file.")

    # --- Write Voxel Centers and Normalized AABB ---
    if voxel_centers.size > 0:
        aabb_min = np.array([norm_bounds[0], norm_bounds[2], norm_bounds[4]])
        aabb_max = np.array([norm_bounds[1], norm_bounds[3], norm_bounds[5]])
        
        print(f"Writing voxel centers to {voxel_centers_path}...")
        write_voxel_centers_txt(voxel_centers, voxel_centers_path, aabb_min, aabb_max)
    else:
        print("Voxel centers are empty, not writing to file.")

    # --- Write Voxelized Surface and Normalized Surface Error ---
    if cropped_voxel.size > 0 and voxel_centers.size > 0:
        spacing_val = 1.0 / resolution
        print(f"Writing voxelized surface mesh to {voxel_surface_path}...")
        voxel_surface = write_voxel_surface_stl(
            cropped_voxel,
            voxel_centers,
            spacing_val,
            voxel_surface_path,
        )

        error_stats = compute_voxel_surface_error(stl_path, voxel_surface)
        if error_stats is not None:
            print("\nVoxelized surface -> original surface error (normalized [0,1] scale):")
            print(f"  Surface samples for mean : {error_stats['sample_count']:,}")
            print(f"  Voxel surface vertices   : {error_stats['vertex_count']:,}")
            print(f"  Voxel surface faces      : {error_stats['face_count']:,}")
            print(f"  Mean error               : {error_stats['mean_error']:.8f}")
            print(f"  Max error                : {error_stats['max_error']:.8f}")
            print(f"  Mean error * resolution  : {error_stats['mean_error'] * resolution:.8f}")
            print(f"  Max error * resolution   : {error_stats['max_error'] * resolution:.8f}")
        else:
            print("Could not compute voxelized surface error.")
    else:
        print("Voxel surface is empty, not writing surface mesh or error stats.")
        
    print("Processing complete.") 
