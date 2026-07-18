import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from .collision_avoidance import check_and_avoid_collisions


def _normalize_np_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a (N,3) array row-wise; keeps zero rows unchanged."""
    arr = np.asarray(v, dtype=float)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > eps, norms, 1.0)
    out = arr / safe
    out[norms[:, 0] <= eps] = 0.0
    return out


def _pack_orientations(layer_type: str, orientations_list: list) -> Dict[str, np.ndarray]:
    """Pack per-point orientations into the format expected by visualizer/tool drawing.

    AM: {'am_axis': (n, 3)}
    SM: {'sm_normal_vec': (n, 3), 'sm_tool_vec': (n, 3)}
    """
    if not orientations_list:
        return {}
    if layer_type == 'AM':
        return {'am_axis': _normalize_np_rows(np.array(orientations_list))}
    # SM: list of dicts with 'normal_vec' / 'tool_vec'
    return {
        'sm_normal_vec': _normalize_np_rows(np.array([o['normal_vec'] for o in orientations_list])),
        'sm_tool_vec': _normalize_np_rows(np.array([o['tool_vec'] for o in orientations_list])),
    }


def process_path_with_avoidance(
    ctx: Any,
    segments: List[Dict],
    layer_type: str,
    layer_time: float,
    collision_batch_size: int = 4096,
    n_tool_samples: int = 900,
    query_batch_size: int = 32768,
    avoidance_cone_half_angle: float = 15.0,
    avoidance_n_candidates: int = 16,
    avoidance_m_chunk: int = 32768,
    collision_use_amp: Optional[bool] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], np.ndarray]:
    """
    对路径段进行批量避障处理，并额外输出被剔除的碰撞点。
    
    Returns:
        Tuple[List[Dict], np.ndarray]: (处理后的路径段列表, 被剔除的无法规避碰撞的点坐标集)
    """
    if not segments:
        return [], np.array([])

    # 1. 提取所有点进行批量检测
    all_pts_list = [seg['points'] for seg in segments]
    pts_flat = np.vstack(all_pts_list)
    
    # 2. 调用批量避障接口（带上 pipeline 中的所有控制参数）
    result = check_and_avoid_collisions(
        ctx=ctx,
        path_points_real=pts_flat,
        layer_time=layer_time,
        layer_type=layer_type,
        batch_size=collision_batch_size,
        n_tool_samples=n_tool_samples,
        query_batch_size=query_batch_size,
        avoidance_cone_half_angle=avoidance_cone_half_angle,
        avoidance_n_candidates=avoidance_n_candidates,
        avoidance_m_chunk=avoidance_m_chunk,
        collision_use_amp=collision_use_amp,
        verbose=verbose,
        show_progress=verbose,
    )

    # --- 新增：提取被剔除的点 ---
    removed_points = pts_flat[result.collision_unresolved] 
    # -----------------------

    final_segments = []
    cursor = 0
    
    for original_seg in segments:
        n_pts = len(original_seg['points'])
        is_unresolved = result.collision_unresolved[cursor : cursor + n_pts]
        orientations = result.orientations[cursor : cursor + n_pts]
        points = original_seg['points']
        
        active_pts = []
        active_orientations = []
        
        for i in range(n_pts):
            if not is_unresolved[i]:
                active_pts.append(points[i])
                ori = orientations[i]
                if layer_type == 'AM':
                    active_orientations.append(ori.am_axis)
                else:
                    active_orientations.append({
                        'normal_vec': ori.sm_normal_vec,
                        'tool_vec': ori.sm_tool_vec
                    })
            else:
                if len(active_pts) >= 2:
                    seg_orient = _pack_orientations(layer_type, active_orientations)
                    final_segments.append({
                        'points': np.array(active_pts),
                        'type': layer_type,
                        'orientations': seg_orient
                    })
                active_pts = []
                active_orientations = []
        
        if len(active_pts) >= 2:
            seg_orient = _pack_orientations(layer_type, active_orientations)
            final_segments.append({
                'points': np.array(active_pts),
                'type': layer_type,
                'orientations': seg_orient
            })
            
        cursor += n_pts

    # 修改返回值为元组
    return final_segments, removed_points