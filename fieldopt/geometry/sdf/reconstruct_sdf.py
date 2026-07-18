import torch
import numpy as np
import pyvista as pv
try:
    from fieldopt.geometry.sdf.sdf_field import sdfModel
except ImportError:
    from fieldopt.geometry.sdf.sdf_field import sdfModel
import os

def reconstruct_and_visualize(
    stl_path="../stlFiles/bracket.stl",
    model_path="../stlFiles/bracketSDF.pt",
    resolution=128,
    show_original=True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading original mesh from {stl_path}...")
    if not os.path.exists(stl_path):
        print(f"Error: 找不到模型文件 {stl_path}")
        return
        
    mesh = pv.read(stl_path)
    meshVertices = np.array(mesh.points)
    
    # 严格按照你训练时的代码逻辑，恢复缩放和平移参数
    x_min, y_min, z_min = np.min(meshVertices, axis=0)
    x_max, y_max, z_max = np.max(meshVertices, axis=0)
    
    minVals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
    maxVals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
    midVals = 0.5 * (minVals + maxVals)
    
    max_range = np.max([x_max-x_min, y_max-y_min, z_max-z_min])
    rangeVals = 0.5 * torch.tensor([max_range, max_range, max_range], dtype=torch.float32, device=device)
    
    print(f"Generating {resolution}x{resolution}x{resolution} uniform grid for sampling...")
    pad = 0.1 * max_range  # 提供周围 10% 的填充
    x_vals = np.linspace(x_min - pad, x_max + pad, resolution)
    y_vals = np.linspace(y_min - pad, y_max + pad, resolution)
    z_vals = np.linspace(z_min - pad, z_max + pad, resolution)
    
    # 构建 pyvista 原生的结构化三维网格，这样抽取算法时不会乱序
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    grid = pv.StructuredGrid(X, Y, Z)
    
    # 将网格所有的点转为 GPU Tensor 并进行归一化
    points_world_tensor = torch.tensor(grid.points, dtype=torch.float32, device=device)
    points_norm = (points_world_tensor - midVals) / rangeVals
    
    print(f"Loading trained SDF model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: 找不到权重文件 {model_path}")
        return
        
    # 读取你跑通并保存好了的模型
    model = sdfModel(model_load_path=model_path, device=device)
    model.model.eval() # 设置为推理模式
    
    print("Predicting SDF values (this might take a few seconds)...")
    # 分批推理，防止网格点太多把显存一次性撑爆
    with torch.no_grad():
        outVals = model.predictOuts(points_norm, batch_size=200000)
        predicted_sdf_norm = outVals['scalars'].squeeze()
        
    # 距离还原：乘以原始几何放缩倍率，恢复真实世界毫米级别的绝对距离
    predicted_sdf_world = predicted_sdf_norm * rangeVals[0]
    predicted_sdf_world_np = predicted_sdf_world.cpu().numpy()
    
    print("Extracting isosurface from distance field...")
    grid.point_data["SDF"] = predicted_sdf_world_np
    
    try:
        # 换回经典 Marching Cubes 等值面重建
        reconstruct_mesh = grid.contour([0.0], scalars="SDF")
    except Exception as e:
        print(f"Warning: 从场中抽取 0 等值面失败了。原因: {e}")
        return

    print("Rendering visualization...")
    
    # 可视化准备
    plotter = pv.Plotter(title="SDF Field Marching Cubes")
    plotter.background_color = "white"
    
    # ================ 模型添加 ================
    original_actor = None
    if show_original:
        # 添加原始 STL (蓝色，透明带线框)
        original_actor = plotter.add_mesh(
            mesh, color='lightblue', opacity=0.35, show_edges=True, edge_color="gray", label='Original STL'
        )
        
    # 重建网格的实体显示 (橙色连续闭合表面)
    plotter.add_mesh(
        reconstruct_mesh, color='orange', smooth_shading=True, label='Reconstruct SDF'
    )
    
    # ================ 交互按钮 ================
    if show_original:
        def toggle_vis(state):
            if original_actor:
                original_actor.SetVisibility(state)
        
        # 添加一个 UI 勾选框
        plotter.add_checkbox_button_widget(
            toggle_vis,
            value=True,
            position=(10.0, 10.0),
            size=30,
            border_size=3,
            color_on='dodgerblue',
            color_off='lightgrey',
            background_color='white'
        )
        # 用文本在按钮旁边标注说明文字
        plotter.add_text("Show Original STL", position=(50, 15), font_size=12, color="black")

    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

if __name__ == '__main__':
    # 你可以在这里调整参数配置
    import os
    # 设定工作目录切换，保证相对路径安全
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    
    reconstruct_and_visualize(
        stl_path="stlFiles/bracket.pt" if not os.path.exists("stlFiles/bracket.stl") else "stlFiles/bracket.stl",
        model_path="stlFiles/bracketSDF.pt",
        resolution=128,          # 回滚至最适合曲面光滑度的 128 (约 200万采样点)
        show_original=True      
    )
