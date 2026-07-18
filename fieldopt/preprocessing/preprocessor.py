import numpy as np
import trimesh

def normalize_mesh_to_unit(mesh):
    """
    将 Mesh 归一化：最大边长缩放至 1.0，并对齐到原点
    """
    # 获取原始 AABB
    bounds = mesh.bounds
    extents = mesh.extents
    max_side = np.max(extents)
    
    # 平移：将最小点移动到 (0, 0, 0)
    mesh.apply_translation(-bounds[0])
    
    # 缩放：最大边长变为 1
    scale_factor = 1.0 / max_side
    mesh.apply_scale(scale_factor)
    
    print(f"归一化完成：当前 AABB 范围 [0, 0, 0] 到 {mesh.bounds[1]}")
    return mesh

def extract_overhang_points(mesh, angle_threshold=45, sample_density=2000, z_min=0.1,
                            xy_boundary_dist_max=None):
    """
    改进版：在符合高度要求的悬垂面上进行均匀采样
    z_min: 采样点的最低高度阈值
    xy_boundary_dist_max: if not None, restrict samples whose XY-distance
                          to the mesh AABB boundary is smaller than this value
    """
    # 1. 法向过滤 (角度阈值)
    normals = mesh.face_normals
    down_vec = np.array([0, 0, -1])
    cos_theta = np.dot(normals, down_vec)
    angle_mask = cos_theta > np.cos(np.radians(90 - angle_threshold))
    
    # 2. 高度过滤 (Z轴阈值)
    # 使用面片中心的高度进行判断
    face_centers = mesh.triangles_center
    height_mask = face_centers[:, 2] > z_min
    
    # 3. 合并掩码：必须同时满足“是悬垂面”且“高于 z_min”
    combined_mask = angle_mask & height_mask
    
    if not np.any(combined_mask):
        print(f"未发现高于 Z={z_min} 的悬垂面。")
        return np.array([])

    # 提取符合条件的子网格
    overhang_mesh = mesh.submesh([combined_mask], append=True)
    
    # 根据有效面积计算采样点数量
    num_samples = int(overhang_mesh.area * sample_density)
    num_samples = max(num_samples, 10)
    
    print(f"识别到有效悬垂面积: {overhang_mesh.area:.4f} (Z > {z_min}), 采样种子点数量: {num_samples}")
    
    # 在过滤后的表面均匀采样
    samples, _ = trimesh.sample.sample_surface(overhang_mesh, num_samples)

    # 4. Optional XY-boundary distance restriction w.r.t. mesh AABB
    if xy_boundary_dist_max is not None and samples.size > 0:
        bounds = mesh.bounds
        x_min, y_min = bounds[0][0], bounds[0][1]
        x_max, y_max = bounds[1][0], bounds[1][1]

        # distance to the four XY boundaries for each sample
        dx_min = samples[:, 0] - x_min
        dx_max = x_max - samples[:, 0]
        dy_min = samples[:, 1] - y_min
        dy_max = y_max - samples[:, 1]

        dist_to_boundary = np.minimum.reduce([dx_min, dx_max, dy_min, dy_max])
        mask = dist_to_boundary < xy_boundary_dist_max
        filtered = samples[mask]

        print(f"按 XY-AABB 边界距离<{xy_boundary_dist_max:.4f} 过滤后种子点数量: {filtered.shape[0]}")

        if filtered.size == 0:
            return filtered
        samples = filtered

    return samples