import numpy as np
from skimage import measure
from stl import mesh

def generate_tpms_stl(tpms_type='gyroid', iterations=(2, 2, 2), res=64, thickness=0.2, level=0, filename='tpms_model.stl'):
    """
    生成具有厚度的水密TPMS模型 STL 文件
    :param tpms_type: 'gyroid', 'schwarz_p', 'diamond'
    :param iterations: 在X, Y, Z方向上的周期数
    :param res: 分辨率（单方向体素数量）
    :param thickness: 壁厚
    :param level: 偏移量（控制孔隙率）
    :param filename: 保存的文件名
    """
    
    # 1. 构建坐标网格
    # 为了确保边界闭合，我们在边缘多加一圈像素进行“封口”处理
    x = np.linspace(0, 2 * np.pi * iterations[0], res)
    y = np.linspace(0, 2 * np.pi * iterations[1], res)
    z = np.linspace(0, 2 * np.pi * iterations[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 2. 定义 TPMS 函数
    if tpms_type.lower() == 'gyroid':
        F = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    elif tpms_type.lower() == 'schwarz_p':
        F = np.cos(X) + np.cos(Y) + np.cos(Z)
    elif tpms_type.lower() == 'diamond':
        F = (np.sin(X) * np.sin(Y) * np.sin(Z) + 
             np.sin(X) * np.cos(Y) * np.cos(Z) + 
             np.cos(X) * np.sin(Y) * np.cos(Z) + 
             np.cos(X) * np.cos(Y) * np.sin(Z))
    elif tpms_type.lower() == 'neovius':
        F = 3*(np.cos(X) + np.cos(Y) + np.cos(Z)) + 4*np.cos(X)*np.cos(Y)*np.cos(Z)
    else:
        raise ValueError("Unsupported TPMS type")

    # 3. 创建实体厚度逻辑
    # 原始表面是 F = level。有厚度的表面是 |F - level| = thickness/2
    # 我们定义一个新的场 G，使得 G=0 处就是厚度边界
    G = np.abs(F - level) - (thickness / 2)

    # 4. 强制边界水密 (Boundary Padding)
    # 将网格的最外层像素值设为一个较大的正数，强制 Marching Cubes 在边界处“收口”
    G[0, :, :] = G[-1, :, :] = 1
    G[:, 0, :] = G[:, -1, :] = 1
    G[:, :, 0] = G[:, :, -1] = 1

    # 5. 使用 Marching Cubes 提取等值面
    # 等值设为 0
    verts, faces, normals, values = measure.marching_cubes(G, level=0)

    # 6. 导出为 STL
    tpms_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            tpms_mesh.v0[i] = verts[f[0]]
            tpms_mesh.v1[i] = verts[f[1]]
            tpms_mesh.v2[i] = verts[f[2]]
    
    is_watertight = tpms_mesh.is_closed()
    print(f"Is mesh watertight: {is_watertight}")

    import os
    filepath = os.path.expanduser(filename)
    tpms_mesh.save(filepath)
    print(f"Model saved to: {filepath}")

# --- 示例调用 ---
if __name__ == "__main__":
    generate_tpms_stl(
        tpms_type='gyroid', 
        iterations=(8, 8, 8), 
        res=100,              # 分辨率越高越光滑，但计算越慢
        thickness=1.0,        # 实体壁厚
        level=0,              # 基础等值面
        filename='~/OneDrive/generated_models/my_gyroid_solid.stl'
    )