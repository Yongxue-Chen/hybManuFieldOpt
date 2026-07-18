import trimesh
import numpy as np
import torch

def create_cylinder_mesh(
    paths,
    r_supp,
    sections=8,
    z_min=None,
    z_eps=1e-6,
    radius_scale=1.0,
    top_embed_length=0.0,
):
    """
    将路径数据转换为圆柱体网格
    paths: List[List[Tuple(np.array/Tensor, np.array/None)]]

    当支撑路径碰到 AABB 的底面时，为了避免球帽穿出底面，
    会跳过这些底部节点的球体，只保留圆柱，从而得到平底支撑。

    参数
    -------
    paths : list
        支撑路径列表。
    r_supp : float
        支撑半径。
    sections : int
        球和圆柱的离散段数。
    z_min : float or None
        AABB 底面的 z 值；若为 None，则对每条路径使用该路径自身的最小 z。
    z_eps : float
        判断“贴底”的容差。
    radius_scale : float
        对支撑半径的统一放大倍率。
    top_embed_length : float
        沿每条支撑首段的反方向，将首段向主体内部额外延伸的长度。
        这样在不修改输入主体 STL 的前提下，让导出的支撑与主体产生稳定重叠，
        便于后续 boolean union 得到单一整体。
    """
    support_radius = float(r_supp) * float(radius_scale)
    all_meshes = []

    for path_data in paths:
        if len(path_data) < 2:
            continue
            
        # 提取并转换坐标点：确保所有点都是 CPU 上的 numpy 数组 (3,)
        points = []
        for node in path_data:
            pos = node[0]  # 提取元组中的第一个元素 (坐标)
            if torch.is_tensor(pos):
                pos = pos.detach().cpu().numpy()
            else:
                pos = np.array(pos)
            points.append(pos)

        # 转为数组方便后续按坐标处理
        points = np.asarray(points, dtype=float)
        segment_points = points.copy()

        # 将支撑首段向主体方向轻微埋入，避免只是“接触”而没有真实体积重叠。
        if top_embed_length > 0.0 and len(segment_points) >= 2:
            first_dir = segment_points[1] - segment_points[0]
            first_len = np.linalg.norm(first_dir)
            if first_len > 1e-9:
                segment_points[0] = segment_points[0] - first_dir / first_len * float(top_embed_length)

        # 确定用于“贴底”判断的 z 基准
        if z_min is not None:
            base_z = float(z_min)
        else:
            base_z = float(points[:, 2].min())

        # 1. 为路径中的每一个节点生成一个球体 (确保连接处无缝)
        for p in points:
            # 如果该点足够接近底面，则跳过球体，避免半球穿出 AABB
            if (p[2] - base_z) <= max(z_eps, 0.25 * support_radius):
                continue

            sphere = trimesh.creation.uv_sphere(radius=support_radius, count=[sections, sections])
            sphere.apply_translation(p)  # 现在 p 是正确的 (3,) 数组
            all_meshes.append(sphere)

        # 2. 为路径中的每一段线段生成一个圆柱体
        for i in range(len(segment_points) - 1):
            p1 = segment_points[i]
            p2 = segment_points[i + 1]
            
            height = np.linalg.norm(p2 - p1)
            if height < 1e-6:
                continue
            center = (p1 + p2) / 2.0
            
            cylinder = trimesh.creation.cylinder(radius=support_radius, height=height, sections=sections)
            
            # 计算方向并对齐
            direction = (p2 - p1) / height
            z_axis = [0, 0, 1]
            
            # 这里的 align_vectors 返回变换矩阵
            matrix = trimesh.geometry.align_vectors(z_axis, direction)
            cylinder.apply_transform(matrix)
            cylinder.apply_translation(center)
            
            all_meshes.append(cylinder)

    if not all_meshes:
        return trimesh.Trimesh()
        
    return trimesh.util.concatenate(all_meshes)
