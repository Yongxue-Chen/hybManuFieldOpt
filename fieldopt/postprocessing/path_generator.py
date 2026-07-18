import numpy as np
import pyvista as pv
from typing import List, Dict, Optional

def generate_path(
    layer_mesh: pv.PolyData,
    layer_type: str,
    path_width: float,
    direction_angle: float = 0.0,
    sample_spacing: float = 1.0
) -> List[Dict]:
    """
    生成的 Zigzag 路径将完美避开空洞飞线。
    逻辑：引入 Line ID，同一行内的断裂（空洞造成）不进行缝合。
    """
    if layer_mesh is None or layer_mesh.n_points < 3:
        return []

    # 1. 计算切片平面方向
    theta = np.radians(direction_angle)
    # 扫描线法向量（在XY平面内旋转）
    plane_normal = np.array([-np.sin(theta), np.cos(theta), 0.0])
    
    projections = np.dot(layer_mesh.points, plane_normal)
    global_min, global_max = np.min(projections), np.max(projections)
    
    # 计算扫描线的分布位置
    start_d = global_min + path_width * 0.2
    end_d = global_max - path_width * 0.2
    
    if start_d >= end_d:
        slice_vals = [(global_min + global_max) / 2.0]
    else:
        slice_vals = np.arange(start_d, end_d + 1e-5, path_width)
        
    raw_segments_info = []
    center_z = layer_mesh.center[2]
    
    # 设定微小空洞愈合容差
    hole_tolerance = path_width * 4.0 
    
    # 2. 执行切片并记录 Line ID
    for line_id, d in enumerate(slice_vals):
        plane_origin = plane_normal * d + np.array([0, 0, center_z])
        try:
            contour = layer_mesh.slice(normal=plane_normal, origin=plane_origin)
            if contour is None or contour.n_points < 2:
                continue
            
            # 提取线段
            stripped = contour.strip(join=True)
            slice_segs = []
            lines_array = stripped.lines
            offset = 0
            while offset < len(lines_array):
                n_pts = lines_array[offset]
                if n_pts >= 2:
                    pt_indices = lines_array[offset + 1 : offset + 1 + n_pts]
                    slice_segs.append(stripped.points[pt_indices])
                offset += 1 + n_pts
            
            # 片内愈合（只焊死极小的裂缝）
            healed_segs = _heal_tiny_holes(slice_segs, tolerance=hole_tolerance)
            
            for seg in healed_segs:
                if len(seg) >= 2:
                    # 直接存原始点，最终统一重采样
                    raw_segments_info.append({
                        'line_id': line_id,
                        'points': seg
                    })
        except Exception:
            continue

    if not raw_segments_info:
        return []

    # 扫描线方向（沿扫描线走向，垂直于 plane_normal，在 XY 平面内）
    scan_dir = np.array([np.cos(theta), np.sin(theta), 0.0])

    # 3. 按连通区域分组：相邻扫描线上端点足够近的段归为同一区域
    components = _find_connected_components(raw_segments_info, path_width)

    # 组间按各自最小坐标排序，保持整体有序
    components.sort(key=lambda comp: min(
        np.dot(s['points'][0], scan_dir) for s in comp
    ))

    # 4. 每个连通区域独立处理：组内排序 + 顺序检查首尾缝合
    all_stitched = []
    for comp in components:
        sorted_comp = _sort_component_by_scanline(comp, scan_dir)
        comp_paths  = _stitch_component(sorted_comp, path_width)
        all_stitched.extend(comp_paths)

    # 5. 全局二次缝合：尝试将各连通区域间靠近的路径端点进一步合并
    all_stitched = _stitch_global_final(all_stitched, path_width * 4.0)

    # 6. 对完整路径重新等间距采样，消除缝合连接处的点密度不均匀
    all_stitched = [_resample_polyline(p, spacing=sample_spacing) for p in all_stitched]

    # 7. 过滤掉物理长度小于 min_path_len 的连续路径
    min_path_len = sample_spacing * 3.0
    optimized_paths = _filter_short_paths(all_stitched, min_length=min_path_len)

    # 8. 格式化输出
    final_output = []
    for pts in optimized_paths:
        final_output.append({
            'points': pts,
            'type': layer_type,
            'orientations': None
        })

    return final_output


def _heal_tiny_holes(segments: List[np.ndarray], tolerance: float) -> List[np.ndarray]:
    """合并同一条扫描线上极近的断点（同时检查候选段的首端和尾端）。"""
    if not segments: return []
    pool = list(segments)
    healed = []
    curr = pool.pop(0)
    while pool:
        best_idx = -1
        best_dist = float('inf')
        best_flip = False
        for i, cand in enumerate(pool):
            d_start = np.linalg.norm(curr[-1] - cand[0])
            d_end   = np.linalg.norm(curr[-1] - cand[-1])
            if d_start < best_dist:
                best_dist, best_idx, best_flip = d_start, i, False
            if d_end < best_dist:
                best_dist, best_idx, best_flip = d_end,   i, True

        if best_dist <= tolerance:
            seg = pool.pop(best_idx)
            if best_flip:
                seg = seg[::-1]
            curr = np.vstack((curr, seg))
        else:
            healed.append(curr)
            curr = pool.pop(0)
    healed.append(curr)
    return healed


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """等间距重采样。"""
    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_len = cum_dist[-1]
    if total_len < 1e-6: return points[[0]]
    n_samples = max(2, int(np.ceil(total_len / spacing)))
    new_dists = np.linspace(0, total_len, n_samples)
    return np.stack([np.interp(new_dists, cum_dist, points[:, i]) for i in range(3)], axis=1)


def _find_connected_components(segments_info: List[Dict], path_width: float) -> List[List[Dict]]:
    """
    将所有段按连通区域分组（Union-Find）。
    判定两段连通的条件：
      - 属于相邻扫描线（line_id 差 <= 1）
      - 四对端点（首首/首尾/尾首/尾尾）中最小距离 <= 3 × path_width
    """
    from collections import defaultdict

    n = len(segments_info)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    threshold = path_width * 8.0

    for i in range(n):
        for j in range(i + 1, n):
            if abs(segments_info[i]['line_id'] - segments_info[j]['line_id']) > 1:
                continue
            pi0, pi1 = segments_info[i]['points'][0], segments_info[i]['points'][-1]
            pj0, pj1 = segments_info[j]['points'][0], segments_info[j]['points'][-1]
            min_d = min(
                np.linalg.norm(pi0 - pj0),
                np.linalg.norm(pi0 - pj1),
                np.linalg.norm(pi1 - pj0),
                np.linalg.norm(pi1 - pj1),
            )
            if min_d <= threshold:
                union(i, j)

    groups: Dict[int, list] = defaultdict(list)
    for i, seg in enumerate(segments_info):
        groups[find(i)].append(seg)

    return list(groups.values())


def _sort_component_by_scanline(component: List[Dict], scan_dir: np.ndarray) -> List[Dict]:
    """
    对单个连通区域内的段排序：
    - 按 line_id（扫描线编号）升序排列各行
    - 行内段按沿 scan_dir 方向的投影坐标排序
    - 奇数行整体翻转（包括段内点顺序），形成 Zigzag 回程
    """
    from collections import defaultdict

    lines: Dict[int, list] = defaultdict(list)
    for seg in component:
        lines[seg['line_id']].append(seg)

    sorted_result = []
    line_ids = sorted(lines.keys())

    for i, lid in enumerate(line_ids):
        segs = lines[lid]
        # 按沿扫描线方向的投影排序（即垂直于 plane_normal 的分量）
        segs_sorted = sorted(segs, key=lambda s: float(np.dot(s['points'][0], scan_dir)))

        if i % 2 == 1:
            segs_sorted = segs_sorted[::-1]
            for s in segs_sorted:
                s['points'] = s['points'][::-1]

        sorted_result.extend(segs_sorted)

    return sorted_result


def _stitch_component(sorted_segs: List[Dict], path_width: float) -> List[np.ndarray]:
    """
    在单个连通区域内顺序缝合：
    - 对每对相邻段，同时检查下一段的首端和尾端，选距离更近的方向（允许翻转）
    - 在阈值内则连接，超出则断开新起一段
    """
    if not sorted_segs:
        return []

    # 连通区域内缝合阈值：足够宽松以跨越 U 型弯
    stitch_threshold = path_width * 8.0

    results = []
    current_path = sorted_segs[0]['points'].copy()

    for i in range(1, len(sorted_segs)):
        next_pts = sorted_segs[i]['points'].copy()
        last_pt  = current_path[-1]

        d_start = np.linalg.norm(last_pt - next_pts[0])
        d_end   = np.linalg.norm(last_pt - next_pts[-1])

        # 选更近的端作为起点（允许翻转）
        if d_end < d_start:
            next_pts = next_pts[::-1]
            d_start  = d_end

        if d_start <= stitch_threshold:
            current_path = np.vstack((current_path, next_pts))
        else:
            results.append(current_path)
            current_path = next_pts

    results.append(current_path)
    return results

def _stitch_global_final(paths: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    """
    全局贪心二次缝合：对组内缝合后的所有路径段再做一轮全局最近邻搜索，
    将端点距离在 threshold 内的段进一步连接，减少抬刀次数。
    同时检查每段的首端和尾端，允许翻转以找到最近的连接方式。
    """
    if not paths:
        return []

    # 用可变列表存储，每项为 np.ndarray
    pool = [p.copy() for p in paths]

    # 起始段：选首点 x+y 最小的
    start_idx = min(range(len(pool)), key=lambda i: pool[i][0][0] + pool[i][0][1])
    current_path = pool.pop(start_idx)
    results = []

    while pool:
        last_pt   = current_path[-1]
        best_idx  = -1
        best_dist = float('inf')
        best_flip = False

        for i, cand in enumerate(pool):
            d_start = np.linalg.norm(last_pt - cand[0])
            d_end   = np.linalg.norm(last_pt - cand[-1])
            if d_start < best_dist:
                best_dist, best_idx, best_flip = d_start, i, False
            if d_end < best_dist:
                best_dist, best_idx, best_flip = d_end,   i, True

        next_path = pool.pop(best_idx)
        if best_flip:
            next_path = next_path[::-1]

        if best_dist <= threshold:
            current_path = np.vstack((current_path, next_path))
        else:
            results.append(current_path)
            current_path = next_path

    results.append(current_path)
    return results


def _filter_short_paths(paths: List[np.ndarray], min_length: float) -> List[np.ndarray]:
    """
    过滤掉物理长度小于 min_length 的连续路径。
    """
    filtered = []
    for pts in paths:
        if len(pts) < 2:
            continue
        
        # 计算整条路径的总长度
        # np.linalg.norm(pts[1:] - pts[:-1], axis=1) 计算段与段之间的距离
        total_len = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
        
        if total_len >= min_length:
            filtered.append(pts)
            
    return filtered
