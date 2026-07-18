from __future__ import annotations

import argparse
import os

os.environ["PYOPENGL_PLATFORM"] = "egl" # 预防由于服务器 headless(无显示器) 导致的 pyrender 崩溃

import mesh_to_sdf
import numpy as np
import torch
import torch.nn as nn
import trimesh
from tqdm import tqdm
from skimage.measure import marching_cubes

try:
    from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
    from fieldopt.models.hash_encoder import MultiResHashEncoder
    from fieldopt.models.mlp import MLP
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
    from fieldopt.models.hash_encoder import MultiResHashEncoder
    from fieldopt.models.mlp import MLP

class NGPSDFNet(nn.Module):
    """
    基于 Instant-NGP (多分辨率哈希编码) + 轻量 MLP 的 SDF 预测网络
    """
    def __init__(
        self,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        device: str,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = MultiResHashEncoder(
            L=n_levels,
            T=2**log2_hashmap_size,
            F=n_features_per_level,
            N_min=base_resolution,
            N_max=finest_resolution,
            device=device,
            bounding_box=[aabb_min.astype(np.float32), aabb_max.astype(np.float32)],
        )
        self.mlp = MLP(
            input_dim=self.encoder.output_dim,
            output_dim=1,
            n_neurons=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation="relu",
            output_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x).float()
        return self.mlp(encoded)

def load_and_normalize_mesh(stl_file: str) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    1. 归一化并移动到原点：将 bounding box 的最小点移动到原点 (0,0,0)，并将最大边长缩放为 1。
    """
    scale, p_min, _ = get_normalization_parameters(stl_file)
    mesh = trimesh.load_mesh(stl_file)
    mesh.apply_translation(-p_min)
    mesh.apply_scale(scale)

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()

    aabb_min = mesh.bounds[0].astype(np.float32)
    aabb_max = mesh.bounds[1].astype(np.float32)
    return mesh, aabb_min, aabb_max

def get_mesh_to_sdf_scale(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """
    由于 mesh_to_sdf 内部会自动把模型位移并缩放进单位球 (r=1) 内进行计算，
    我们需要反向推导出它的缩放倍率，以便将 mesh_to_sdf 返回的坐标和 SDF 圆滑地还原回我们的 [0, 1] 空间。
    """
    center = mesh.bounding_box.centroid
    vertices = mesh.vertices - center
    radius = float(np.max(np.linalg.norm(vertices, axis=1)))
    return center, max(radius, 1e-8)

def generate_sdf_dataset(
    mesh: trimesh.Trimesh,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    n_surf: int,
    n_vol: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    2. 使用 mesh_to_sdf 库进行采样并获取目标真实 SDF。
    """
    center, radius = get_mesh_to_sdf_scale(mesh)
    
    # --- 表面点采样 ---
    print(f"  使用 mesh_to_sdf 采样 {n_surf:,} 个距离点...")
    surf_pts_unit, surf_sdf_unit_raw = mesh_to_sdf.sample_sdf_near_surface(
        mesh,
        number_of_points=n_surf,
        surface_point_method="sample",
        sign_method="normal"
    )
    # 将单位球体空间的点还原到项目的 [0,1] 空间
    surf_pts = surf_pts_unit * radius + center
    
    print(f"  [修正精度] 放弃 mesh_to_sdf 估算的符号，使用精准射线法判定表层点内外边界...")
    surf_dists = np.abs(surf_sdf_unit_raw * radius)
    surf_inside = mesh.contains(surf_pts)
    surf_sdf = np.where(surf_inside, -surf_dists, surf_dists).astype(np.float32)
    
    # --- 体素点均匀采样 (平衡内部/外部点) ---
    print(f"  在包含盒内生成 {n_vol:,} 个平衡体素点 (强制提升内部采样比例)...")
    pad = 0.1 * (aabb_max - aabb_min)
    v_min, v_max = aabb_min - pad, aabb_max + pad
    
    # 策略：大量过采样候选点，使用精准射线法筛选内部点，达到内部 50% 的平衡比例
    candidates = np.random.uniform(v_min, v_max, size=(n_vol * 6, 3)).astype(np.float32)
    c_sdf_unit_raw = mesh_to_sdf.mesh_to_sdf(
        mesh,
        candidates,
        surface_point_method="sample",
        sign_method="normal"
    )
    
    print(f"  [修正精度] 对 {len(candidates):,} 个候选体素点进行并行射线法，计算精准内部空间...")
    c_dists = np.abs(c_sdf_unit_raw * radius)
    c_inside_mask = mesh.contains(candidates)
    c_sdf = np.where(c_inside_mask, -c_dists, c_dists).astype(np.float32)
    
    in_mask = c_sdf < 0
    out_mask = ~in_mask
    
    # 尝试拿一半的内部点
    inside_pts = candidates[in_mask][:n_vol // 2]
    inside_sdf = c_sdf[in_mask][:n_vol // 2]
    
    # 剩下的名额全部用外部点填满
    out_needed = n_vol - len(inside_pts)
    outside_pts = candidates[out_mask][:out_needed]
    outside_sdf = c_sdf[out_mask][:out_needed]
    
    vol_pts = np.vstack([inside_pts, outside_pts]).astype(np.float32)
    vol_sdf = np.concatenate([inside_sdf, outside_sdf]).astype(np.float32)

    pts = np.vstack([surf_pts, vol_pts]).astype(np.float32)
    sdf = np.concatenate([surf_sdf, vol_sdf]).astype(np.float32)

    # 打乱点集
    idx = np.random.permutation(len(pts))
    return pts[idx], sdf[idx]

def train(
    stl_file: str,
    epochs: int = 400,
    batch_size: int = 65536,
    lr: float = 1e-3,
    n_surf: int = 200_000,
    n_vol: int = 300_000,
    lambda_eikonal: float = 0.01,
    device: str = "cuda"
):
    """
    完整的运行流水线：归一化网格 -> mesh_to_sdf采点 -> 喂入Instant-NGP -> 得到SDF场模型
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU。")
        device = "cpu"
        
    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    save_path = os.path.join(stl_dir, f"{stl_name}_mesh_to_sdf_ngp.pt")
    
    print("\n" + "=" * 70)
    print(f"任务: STL 模型 -> mesh_to_sdf 采样 -> Instant-NGP -> SDF 网络场")
    print(f"文件: {stl_file}")
    print("=" * 70)

    # 1. 归一化并移动到原点 (最大边长改为1，最小点到原点)
    print("\n[1/4] 正在归一化加载的 STL 网格...")
    mesh, aabb_min, aabb_max = load_and_normalize_mesh(stl_file)
    aabb_min_t = torch.tensor(aabb_min, dtype=torch.float32, device=device)
    aabb_max_t = torch.tensor(aabb_max, dtype=torch.float32, device=device)
    print(f"  空间极小点 AABB min: {aabb_min}")
    print(f"  空间极大点 AABB max: {aabb_max}")

    # 2. 采样获取 Ground Truth
    print("\n[2/4] 生成训练点云流与 SDF ...")
    pts_np, sdf_np = generate_sdf_dataset(mesh, aabb_min, aabb_max, n_surf, n_vol)
    pts_t = torch.tensor(pts_np, dtype=torch.float32, device=device)
    sdf_t = torch.tensor(sdf_np, dtype=torch.float32, device=device).unsqueeze(-1)
    
    # 3. 构造预定的 NGP 网络场
    print("\n[3/4] 正在搭建 Instant-NGP 神经 SDF 场...")
    net = NGPSDFNet(aabb_min, aabb_max, device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device == "cuda")) if hasattr(torch.amp, "GradScaler") else None
    if scaler is None and device == "cuda":
        # 兼容旧版 PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 4. 训练 SDF 网络场
    print(f"\n[4/4] 启动 SDF 场网络训练 (目标 {epochs} 轮) ...")
    n_train = len(pts_t)
    
    net.train()
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            batch_pts = pts_t[idx].detach().requires_grad_(True)
            batch_sdf = sdf_t[idx]
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", enabled=(device=="cuda")):
                pred_sdf = net(batch_pts).float()
                
                # 损失函数: 对靠近表面的点添加更高的权重 (Weight = 4.0) 增强细节重建
                weights = torch.where(batch_sdf.abs() < 0.02, 4.0, 1.0)
                sdf_loss = ((pred_sdf - batch_sdf) ** 2 * weights).mean()
                
            # 计算 Eikonal Loss
            grad_scale = 2.0 / (aabb_max_t - aabb_min_t).clamp(min=1e-8)
            grads = torch.autograd.grad(pred_sdf.sum(), batch_pts, create_graph=True)[0]
            grads_project = grads * grad_scale
            eikonal_loss = ((grads_project.norm(dim=-1) - 1.0) ** 2).mean()

            loss = sdf_loss + lambda_eikonal * eikonal_loss
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        
        # 每10轮打印一次指标
        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / max(n_batches, 1)
            tqdm.write(f"  [Epoch {epoch:03d}/{epochs}]  Total Loss: {avg_loss:.6f}  LR: {scheduler.get_last_lr()[0]:.2e}")
            
    # 保存训练成果
    torch.save({
        "state_dict": net.state_dict(),
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
        "model_space": "ngp_sdf_field",
    }, save_path)
    print(f"\nSDF场权重已成功保存至:\n{save_path}")

def visualize_ngp(stl_file, recon_only=False):
    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    pt_path = os.path.join(stl_dir, f"{stl_name}_mesh_to_sdf_ngp.pt")
    
    if not os.path.exists(pt_path):
        print(f"找不到预训练文件 {pt_path}，请先训练。")
        return

    # 1. 载入原始模型及其原包围盒信息
    orig_mesh, aabb_min, aabb_max = load_and_normalize_mesh(stl_file)
    
    # 2. 载入我们训练好的 Instant-NGP SDF 权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    net = NGPSDFNet(ckpt["aabb_min"], ckpt["aabb_max"], device).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # 3. 在包含盒空间内致密撒点 (构建 256x256x256 的密集网格)
    print("\n正在致密空间中提取 SDF 重建网格 (Marching Cubes)...")
    res = 256
    pad = 0.1 * (aabb_max - aabb_min)
    v_min, v_max = aabb_min - pad, aabb_max + pad
    
    xs = np.linspace(v_min[0], v_max[0], res)
    ys = np.linspace(v_min[1], v_max[1], res)
    zs = np.linspace(v_min[2], v_max[2], res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    
    # 4. 分块进行网络推理，防止显存爆满
    sdf_preds = []
    chunk_size = 100000
    with torch.no_grad():
        for i in range(0, len(grid_pts), chunk_size):
            pts_t = torch.tensor(grid_pts[i:i+chunk_size], device=device)
            sdf_preds.append(net(pts_t).cpu().numpy())
            
    sdf_vol = np.concatenate(sdf_preds).reshape(res, res, res)
    
    # 5. Marching Cubes 提取等值面
    try:
        spacing = (v_max - v_min) / (res - 1)
        verts, faces, _, _ = marching_cubes(sdf_vol, level=0.0, spacing=spacing)
        verts += v_min # 偏移回真正的物理空间
        recon_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        print(f"  重建完成！提取了 {len(verts)} 个顶点和 {len(faces)} 个面。")
    except ValueError:
        print("未检测到有效表面 (SDF 未发生正负号翻转)，可能是网络没学好。")
        return
        
    # 修改颜色以便区分：原始模型=蓝色半透明，重建模型=红色半透明
    orig_mesh.visual.face_colors = [50, 150, 250, 150]
    recon_mesh.visual.face_colors = [250, 50, 50, 150]
    
    # 6. 推送至 3D 窗口显示 （这步必须在有显示器的机器上运行）
    print("正在打开 3D 展示窗口...")
    meshes_to_show = [recon_mesh] if recon_only else [orig_mesh, recon_mesh]
    scene = trimesh.Scene(meshes_to_show)
    scene.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STL -> mesh_to_sdf -> Instant-NGP -> SDF Neural Field")
    parser.add_argument("--stl", required=True, help="输入 STL 模型路径")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_surf", type=int, default=200000, help="表面采样数量")
    parser.add_argument("--n_vol", type=int, default=300000, help="空间体素采样数量")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true", help="完成训练后，自动打开窗口显示结果（仅限有屏幕设备）")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练过程，直接调用最后保存的权重进行可视化")
    parser.add_argument("--recon_only", action="store_true", help="可视化时，仅显示重建的红色 SDF 模型，不显示对比用的原始蓝色模型")
    args = parser.parse_args()
    
    if not args.skip_train:
        train(
            stl_file=args.stl,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_surf=args.n_surf,
            n_vol=args.n_vol,
            device=args.device
        )
        
    if args.visualize or args.skip_train:
        visualize_ngp(args.stl, recon_only=args.recon_only)
