import torch
import numpy as np
import pyvista as pv
import argparse
from typing import List, Dict

# Support both `python -m fieldopt.postprocessing.debug_path` and direct script execution.
try:
    from .path_generator import generate_path
    from .path_postprocessor import process_path_with_avoidance
    from .model_loader import load_model_and_config
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fieldopt.postprocessing.path_generator import generate_path
    from fieldopt.postprocessing.path_postprocessor import process_path_with_avoidance
    from fieldopt.postprocessing.model_loader import load_model_and_config

def debug_path_visualization(
    layer, 
    segments: List[Dict],
    removed_points: np.ndarray = None, 
    title: str = "Path Debug"
):
    """可视化单层路径，并标记剔除的碰撞点（已修复图例错误）"""
    p = pv.Plotter(title=title)
    has_labels = False # 用于追踪是否真的有标签被添加
    
    # 1. 渲染基础 Layer Mesh
    if layer.mesh and layer.mesh.n_points > 0:
        p.add_mesh(layer.mesh, color='lightblue', opacity=0.4, style='surface', label='Layer Mesh')
        has_labels = True
        edges = layer.mesh.extract_feature_edges(boundary_edges=True)
        p.add_mesh(edges, color='blue', line_width=2)

    # 2. 渲染被剔除的碰撞点 (红色)
    if removed_points is not None and len(removed_points) > 0:
        p.add_points(
            removed_points, 
            color='red', 
            point_size=8, 
            render_points_as_spheres=True,
            label='Removed (Unresolved)' # 明确设置标签
        )
        has_labels = True

    # 3. 合并路径并渲染
    if not segments:
        print("警告: 没有剩余的可加工路径段。")
    else:
        all_points = []
        lines_array = []
        current_offset = 0
        start_pts = []
        
        for seg in segments:
            pts = seg['points']
            n = len(pts)
            if n < 2: continue
            
            all_points.append(pts)
            start_pts.append(pts[0])
            
            segment_indices = np.arange(current_offset, current_offset + n)
            lines_array.append(n)
            lines_array.extend(segment_indices)
            current_offset += n

        if all_points:
            all_points_np = np.vstack(all_points)
            path_poly = pv.PolyData(all_points_np)
            path_poly.lines = np.array(lines_array)
            path_poly.point_data['order'] = np.arange(all_points_np.shape[0])
            
            p.add_mesh(
                path_poly, 
                cmap='plasma', 
                line_width=4, 
                render_lines_as_tubes=True,
                scalars='order',
                show_scalar_bar=False,
                label='Safe Path', # 为路径添加标签
                name='paths'
            )
            has_labels = True
            
            # 起点标记
            if start_pts:
                p.add_points(np.array(start_pts), color='green', point_size=12, render_points_as_spheres=True, label='Entry Points')
                has_labels = True
            
    # 只有在确实存在标签时才添加图例，避免 ValueError
    if has_labels:
        p.add_legend()
        
    p.add_axes()
    p.show_grid()
    p.show()

def main():
    parser = argparse.ArgumentParser(description="单层路径生成调试工具")
    parser.add_argument("--cache-file", type=str, default="output/MBBSmooth_pipeline_result.pt",
                        help="包含 Layer 数据的 pipeline 缓存文件路径")
    parser.add_argument("--am-idx", type=int, default=0, help="要测试的 AM 层局部索引")
    parser.add_argument("--sm-idx", type=int, default=0, help="要测试的 SM 层局部索引")
    parser.add_argument("--am-width", type=float, default=0.5, help="AM 路径宽度 (mm)")
    parser.add_argument("--sm-width", type=float, default=0.5, help="SM 路径宽度 (mm)")
    parser.add_argument("--angle", type=float, default=0.0, help="Zigzag 扫描角度 (度)")
    parser.add_argument("--spacing", type=float, default=1.0, help="路径采样点间距 (mm)")
    
    args = parser.parse_args()

    ctx = load_model_and_config(
        "MBBSmooth", 
        "output/MBBSmooth_final_trained.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 读取缓存数据
    try:
        print(f"正在加载缓存文件: {args.cache_file} ...")
        data = torch.load(args.cache_file, weights_only=False)
        layers = data['layers']
        scale = data.get('scale', 1.0)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.cache_file}。请先运行 pipeline 生成缓存。")
        return

    # 拆分 AM 和 SM
    am_layers = [l for l in layers if l.layer_type == 'AM']
    sm_layers = [l for l in layers if l.layer_type == 'SM']
    
    print(f"加载成功: 共有 {len(layers)} 层 (AM: {len(am_layers)}, SM: {len(sm_layers)}). Scale: {scale}")

    # ===== 测试 AM 层 =====
    if 0 <= args.am_idx < len(am_layers):
        target_layer = am_layers[args.am_idx]
        print(f"\n[测试 AM 层] 索引: {args.am_idx} (全局索引: {target_layer.global_index})")
        print(f"  高度/时间: {target_layer.time_value:.4f}, 顶点数: {target_layer.mesh.n_points}")
        
        # 保护机制：如果保存的 mesh 还在模型空间(极小)，将其缩放至真实尺寸
        if target_layer.mesh.bounds[1] - target_layer.mesh.bounds[0] < 5.0 and scale > 10.0:
             print("  注意: Mesh似乎处于模型空间，正在临时放大以匹配毫米单位...")
             target_layer.mesh.points *= scale
        
        raw_segs = generate_path(
            layer_mesh=target_layer.mesh,
            layer_type='AM',
            path_width=args.am_width,
            direction_angle=args.angle,
            sample_spacing=args.spacing
        )
        
        # 执行碰撞避免和路径重组
        safe_segs, removed_points = process_path_with_avoidance(
            ctx, raw_segs, 'AM', target_layer.time_value
        )

        n_safe_pts = sum(len(s['points']) for s in safe_segs)
        print(f"  -> 生成了 {len(safe_segs)} 段路径，共 {n_safe_pts} 个无碰撞点。")
        if safe_segs and safe_segs[0].get('orientations'):
            ori = safe_segs[0]['orientations']
            if 'am_axis' in ori:
                ax = ori['am_axis'][0]
                print(f"  -> 无碰撞姿态已输出 (首点 am_axis 示例): [{ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f}]")
        debug_path_visualization(target_layer, safe_segs, removed_points, title=f"AM Layer #{args.am_idx} Path Debug")
    else:
        print(f"\n[测试 AM 层] 索引 {args.am_idx} 超出范围 (最大 {len(am_layers)-1})。")

    # # ===== 测试 SM 层 =====
    # if 0 <= args.sm_idx < len(sm_layers):
    #     target_layer = sm_layers[args.sm_idx]
    #     print(f"\n[测试 SM 层] 索引: {args.sm_idx} (全局索引: {target_layer.global_index})")
        
    #     if target_layer.mesh.bounds[1] - target_layer.mesh.bounds[0] < 5.0 and scale > 10.0:
    #          target_layer.mesh.points *= scale

    #     raw_segs = generate_path(
    #         layer_mesh=target_layer.mesh,
    #         layer_type='SM',
    #         path_width=args.sm_width,
    #         direction_angle=args.angle + 45.0,  # 给 SM 测试一个不同的角度
    #         sample_spacing=args.spacing
    #     )

    #     safe_segs, removed_points = process_path_with_avoidance(
    #         ctx, raw_segs, 'SM', target_layer.time_value
    #     )

    #     n_safe_pts = sum(len(s['points']) for s in safe_segs)
    #     print(f"  -> 生成了 {len(safe_segs)} 段路径，共 {n_safe_pts} 个无碰撞点。")
    #     if safe_segs and safe_segs[0].get('orientations'):
    #         ori = safe_segs[0]['orientations']
    #         if 'sm_normal_vec' in ori:
    #             nv, tv = ori['sm_normal_vec'][0], ori['sm_tool_vec'][0]
    #             print(f"  -> 无碰撞姿态已输出 (首点 normal/tool 示例): normal=[{nv[0]:.4f},{nv[1]:.4f},{nv[2]:.4f}], tool=[{tv[0]:.4f},{tv[1]:.4f},{tv[2]:.4f}]")
    #     debug_path_visualization(target_layer, safe_segs, removed_points, title=f"SM Layer #{args.sm_idx} Path Debug")
    # else:
    #     print(f"\n[测试 SM 层] 索引 {args.sm_idx} 超出范围 (最大 {len(sm_layers)-1})。")

if __name__ == "__main__":
    main()