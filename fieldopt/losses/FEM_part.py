import torch
import torch.nn as nn
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# 初始化 Taichi
ti.init(arch=ti.gpu if torch.cuda.is_available() else ti.cpu)

# ==========================================
# 1. 全局配置
# ==========================================
RES = 32
BATCH_SIZE = 8
CG_ITER = 10       # 减少迭代次数，降低梯度累积
CG_TOL = 1e-6
E0 = 500.0
DX = 1.0 / RES
GRAD_CLIP_THRESHOLD = 1e10  # Taichi梯度裁剪阈值

# ==========================================
# 2. Taichi 物理场定义
# ==========================================
density_field = ti.field(dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)

# 【修复1】所有中间变量都需要梯度，否则 Tape 无法记录历史
u = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)
r = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)
p = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)
Ap = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)
b = ti.Vector.field(3, dtype=float, shape=(BATCH_SIZE, RES, RES, RES), needs_grad=True)

# 【修复2】使用 0D Field 存储标量，确保除法在 Kernel 内进行，Tape 可追踪
alpha = ti.field(dtype=float, shape=(BATCH_SIZE,), needs_grad=True)
beta  = ti.field(dtype=float, shape=(BATCH_SIZE,), needs_grad=True)
r_dot_r = ti.field(dtype=float, shape=(BATCH_SIZE,), needs_grad=True)
r_dot_r_old = ti.field(dtype=float, shape=(BATCH_SIZE,), needs_grad=True)
p_dot_Ap = ti.field(dtype=float, shape=(BATCH_SIZE,), needs_grad=True)

loss_field = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def init_physics():
    u.fill(0.0)
    u.grad.fill(0.0)
    r.fill(0.0)
    r.grad.fill(0.0)
    p.fill(0.0)
    p.grad.fill(0.0)
    Ap.fill(0.0)
    Ap.grad.fill(0.0)
    b.fill(0.0)
    b.grad.fill(0.0)
    
    # 标量场也需要清零
    alpha.fill(0.0)
    alpha.grad.fill(0.0)
    beta.fill(0.0)
    beta.grad.fill(0.0)
    
    loss_field[None] = 0.0
    loss_field.grad[None] = 0.0

@ti.kernel
def compute_rhs():
    for idx, i, j, k in b:
        if j == 0:
            b[idx, i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            rho = density_field[idx, i, j, k]
            # 简单的体积力：密度越大，受力越大
            b[idx, i, j, k] = ti.Vector([0.0, -9.8 * rho, 0.0])

@ti.kernel
def matrix_vector_product(x: ti.template(), result: ti.template()):
    """
    计算 A * x。
    【修复3】使用平均刚度确保矩阵对称性 (Symmetry)
    """
    for idx, i, j, k in result:
        if j == 0:
            # 边界条件：对角线为1，非对角线为0 -> result = x
            result[idx, i, j, k] = x[idx, i, j, k]
        else:
            # 获取当前节点的材料属性
            rho_self = density_field[idx, i, j, k]
            E_self = rho_self ** 3 * E0 + 1e-3
            
            res_vec = ti.Vector([0.0, 0.0, 0.0])
            total_diag_k = 0.0
            
            # 遍历邻居
            for axis in ti.static(range(3)):
                for d in ti.static([-1, 1]):
                    offset = ti.Vector([0, 0, 0])
                    offset[axis] = d
                    ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
                    
                    if 0 <= ni < RES and 0 <= nj < RES and 0 <= nk < RES:
                        # 获取邻居的材料属性
                        rho_neigh = density_field[idx, ni, nj, nk]
                        E_neigh = rho_neigh ** 3 * E0 + 1e-3
                        
                        # 【关键】刚度取平均！确保 K_ij == K_ji
                        # 弹簧刚度 k = (E1 + E2) / 2 / dx^2
                        k_link = (E_self + E_neigh) * 0.5 / (DX * DX)
                        
                        # 如果邻居不是边界固定点，则计算贡献
                        # 这里简化处理：假设所有内部点都参与相互作用
                        res_vec -= k_link * x[idx, ni, nj, nk]
                        total_diag_k += k_link
            
            # 对角线项：所有连接弹簧的刚度之和
            res_vec += total_diag_k * x[idx, i, j, k]
            
            result[idx, i, j, k] = res_vec

# --- 下面是用于 CG 的辅助 Kernel，全部在 Taichi 内完成以保证梯度 ---

@ti.kernel
def dot_product(v1: ti.template(), v2: ti.template(), out: ti.template()):
    # 归约求和前先清零
    for b_idx in range(BATCH_SIZE):
        out[b_idx] = 0.0
        
    for idx, i, j, k in v1:
        val = v1[idx, i, j, k].dot(v2[idx, i, j, k])
        ti.atomic_add(out[idx], val)

@ti.kernel
def update_alpha():
    # alpha = r_dot_r_old / p_dot_Ap
    for b_idx in range(BATCH_SIZE):
        denom = p_dot_Ap[b_idx]
        if abs(denom) > 1e-10:
            alpha[b_idx] = r_dot_r_old[b_idx] / denom
        else:
            alpha[b_idx] = 0.0

@ti.kernel
def update_u_r():
    # u = u + alpha * p
    # r = r - alpha * Ap
    for idx, i, j, k in u:
        a = alpha[idx]
        u[idx, i, j, k] += a * p[idx, i, j, k]
        r[idx, i, j, k] -= a * Ap[idx, i, j, k]

@ti.kernel
def update_beta():
    # beta = r_dot_r / r_dot_r_old
    for b_idx in range(BATCH_SIZE):
        denom = r_dot_r_old[b_idx]
        if abs(denom) > 1e-20:
            beta[b_idx] = r_dot_r[b_idx] / denom
        else:
            beta[b_idx] = 0.0

@ti.kernel
def update_p():
    # p = r + beta * p
    for idx, i, j, k in p:
        b_val = beta[idx]
        p[idx, i, j, k] = r[idx, i, j, k] + b_val * p[idx, i, j, k]

@ti.kernel
def copy_r_old():
    for i in range(BATCH_SIZE):
        r_dot_r_old[i] = r_dot_r[i]

@ti.kernel
def init_cg_vectors():
    # u = 0, r = b, p = b
    for I in ti.grouped(b):
        u[I] = ti.Vector([0.0, 0.0, 0.0])  # 显式初始化
        r[I] = b[I]
        p[I] = b[I]

def run_cg_iter():
    """Python 函数编排 Kernel 调用，Tape 会记录这个序列"""
    
    # 1. 初始化
    compute_rhs()
    init_cg_vectors()
    dot_product(r, r, r_dot_r_old)
    
    # 2. 迭代
    for _ in range(CG_ITER):
        # Ap = A * p
        matrix_vector_product(p, Ap)
        
        # alpha = ...
        dot_product(p, Ap, p_dot_Ap)
        update_alpha() # Kernel内做除法
        
        # u = ..., r = ...
        update_u_r()
        
        # beta = ...
        dot_product(r, r, r_dot_r)
        update_beta() # Kernel内做除法
        
        # p = ...
        update_p()
        
        # update scalar history
        copy_r_old()

@ti.kernel
def compute_compliance_loss():
    for idx, i, j, k in u:
        force_y = -9.8 * density_field[idx, i, j, k]
        # 注意：这里我们使用 atomic_add 累加到标量
        loss_field[None] += force_y * u[idx, i, j, k].y

# ==========================================
# 3. 辅助函数与网络
# ==========================================
def point_to_batch_grid(points, weights, resolution):
    """
    将点云权重映射到批量网格（使用scatter保持平均化效果）
    
    Args:
        points: [N_pts, 3] 采样点位置
        weights: [N_pts, Batch_Size] 每个点在不同时间的密度
        resolution: 网格分辨率
    Returns:
        grid: [Batch_Size, Res, Res, Res] 密度网格
    """
    N_pts, B_size = weights.shape
    device = points.device
    
    # 1. 坐标转体素索引
    coords = (points * resolution).long().clamp(0, resolution - 1)
    # 单个 Grid 内的线性索引
    voxel_idx = (coords[:, 0] * resolution * resolution + 
                 coords[:, 1] * resolution + 
                 coords[:, 2]) # [N_pts]
    
    # 2. 扩展为 Batch 索引
    # [N_pts, 1] -> [N_pts, B_size]
    voxel_idx_expanded = voxel_idx.unsqueeze(1).repeat(1, B_size)
    
    # [1, B_size] -> offsets [0, Res^3, 2*Res^3 ...]
    batch_offsets = (torch.arange(B_size, device=device) * (resolution**3)).unsqueeze(0)
    
    # 最终扁平索引
    flat_indices = (voxel_idx_expanded + batch_offsets).view(-1) # [N_pts * B_size]
    flat_weights = weights.view(-1)
    
    # 3. Scatter Add (分子)
    total_size = B_size * (resolution**3)
    grid_density = torch.zeros(total_size, device=device)
    grid_count = torch.zeros(total_size, device=device)
    
    grid_density.scatter_add_(0, flat_indices, flat_weights)
    
    # 4. 计数归一化 (分母)
    grid_count.scatter_add_(0, flat_indices, torch.ones_like(flat_weights))
    
    # 归一化：即使1:1映射也保持除法（数值稳定性）
    grid_final = grid_density / (grid_count + 1e-5)
    
    return grid_final.view(B_size, resolution, resolution, resolution)

class TimeFieldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 4. 主训练循环
# ==========================================
device = torch.device('cuda')

def run_demo():
    model = TimeFieldNet().to(device)
    # 采样点增加到32768，降低学习率避免梯度爆炸
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # 从0.001降到0.0003
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)  # 每100轮降低20%

    print(f"开始训练... CG Solver | Res: {RES} | CG_ITER: {CG_ITER} | LR: 0.0003")
    loss_history = []

    # 【关键改进】混合采样策略：FEM网格中心 + 随机点
    # 1. FEM网格中心采样 (32³ = 32768 个固定点)
    indices = torch.arange(RES, device=device, dtype=torch.float32)
    grid_centers = (indices + 0.5) / RES  # [0.5/32, 1.5/32, ..., 31.5/32]
    gx, gy, gz = torch.meshgrid(grid_centers, grid_centers, grid_centers, indexing='ij')
    GRID_POINTS_FIXED = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # [32768, 3]

    # 2. 额外的随机采样点 (每个epoch不同，作为正则化)
    NUM_RANDOM_POINTS = 8192  # 额外增加 25% 的随机点

    print(f"混合采样策略:")
    print(f"  - 固定点 (FEM网格中心): {GRID_POINTS_FIXED.shape[0]} 个")
    print(f"  - 随机点 (每epoch变化): {NUM_RANDOM_POINTS} 个")
    print(f"  - 总计: {GRID_POINTS_FIXED.shape[0] + NUM_RANDOM_POINTS} 个点")
    print(f"✓ FEM网格完美对齐 + 随机正则化")

    # 固定时间点
    torch.manual_seed(42)
    FIXED_TARGET_TIMES = torch.rand(BATCH_SIZE).to(device)

    for epoch in range(401):  # 增加到400轮
        optimizer.zero_grad()
        
        # 【混合采样】固定点 + 随机点
        random_points = torch.rand(NUM_RANDOM_POINTS, 3, device=device)
        points = torch.cat([GRID_POINTS_FIXED, random_points], dim=0)  # 拼接
        target_times = FIXED_TARGET_TIMES 
        
        out = model(points)
        t1, t2 = out[:, 0:1], out[:, 1:2]
        
        k = 20.0
        mask_t1 = torch.sigmoid(k * (target_times.unsqueeze(0) - t1))
        mask_t2 = torch.sigmoid(k * (t2 - target_times.unsqueeze(0)))
        weights = mask_t1 * mask_t2  # [N_total, Batch]
        
        grid_4d = point_to_batch_grid(points, weights, RES)
        grid_4d.retain_grad()  # 在使用前声明保留梯度，避免警告
        
        density_field.from_torch(grid_4d)
        init_physics()
        
        # --- 核心修改：Tape 录制完整的 CG 过程 ---
        with ti.ad.Tape(loss=loss_field):
            run_cg_iter() # 调用编排好的 Kernel 序列
            compute_compliance_loss()
        
        grad_grid = density_field.grad.to_torch(device)
        
        # 【关键修复1】梯度裁剪：防止Taichi梯度爆炸
        grad_max = grad_grid.abs().max().item()
        if grad_max > GRAD_CLIP_THRESHOLD:
            grad_grid = grad_grid.clamp(-GRAD_CLIP_THRESHOLD, GRAD_CLIP_THRESHOLD)
        
        # 【关键修复2】梯度缩放：固定缩放因子，不依赖当前梯度大小
        # 使用固定缩放避免梯度爆炸时归一化失效
        # grad_grid_normalized = grad_grid / (grad_max + 1e-8)  # 缩放到[-1, 1]
        # grad_grid_normalized = grad_grid_normalized * 1000.0  # 固定量级
        gradScale = 1e2
        grad_grid_normalized = grad_grid/ gradScale
        
        # 【修复】物理损失：保留梯度方向，让梯度引导优化
        # Taichi梯度的符号很重要：grad>0表示该位置增加密度会让结构变差
        loss_phys = (grid_4d * grad_grid_normalized).sum() * 0.1  # 去掉abs，提高权重
        
        # 【体积约束】超强力版本
        vol_fraction = weights.mean()
        TARGET_VOL = 0.30
        
        # 1. 基础体积约束（平方惩罚）
        vol_target_loss = ((vol_fraction - TARGET_VOL) ** 2) * 2000.0  # 提高到2000，防止偏离
        
        # 2. 强制最小密度（网格采样更稳定，可以用温和一些的惩罚）
        min_vol_threshold = 0.25  # 至少25%
        # 使用指数惩罚，但系数降低
        if vol_fraction < min_vol_threshold:
            violation = min_vol_threshold - vol_fraction
            min_vol_penalty = torch.exp(violation * 8.0) * 500.0  # 稍微温和一些
        else:
            min_vol_penalty = torch.tensor(0.0, device=vol_fraction.device)
        
        # 3. 防止过满
        max_vol_threshold = 0.35  # 降低到35%
        max_vol_penalty = torch.relu(vol_fraction - max_vol_threshold) * 5000.0
        
        # 4. 防止低于目标（新增！）
        # 当 vol_frac < 0.30 时，额外惩罚
        under_target_penalty = torch.relu(TARGET_VOL - vol_fraction) * 1000.0
        
        vol_loss = vol_target_loss + min_vol_penalty + max_vol_penalty + under_target_penalty
        
        # 【SIMP惩罚】几乎不用
        simp_penalty = (weights * (1.0 - weights)).mean() * 1.0
        
        # 【动态权重】去掉abs后，物理梯度更有意义，可以更早且更多地使用
        if epoch < 50:
            phys_weight = 0.01  # 前50轮，主要关注体积
        elif epoch < 150:
            # 50-150轮，增加到0.05
            phys_weight = 0.01 + (epoch - 50) / 100 * 0.04
        else:
            # 150轮后固定在0.05
            phys_weight = 0.05  # 固定权重
        
        # 【硬约束】体积低于阈值时，降低物理权重
        if vol_fraction < 0.25:
            phys_weight = phys_weight * 0.1  # 降低到10%
        
        # 组合损失
        total_loss = loss_phys * phys_weight + vol_loss + simp_penalty
        total_loss.backward()
        
        # 梯度裁剪防止 CG 早期不稳定导致的梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # 学习率衰减
        
        if epoch % 20 == 0:
            val = loss_field[None] / BATCH_SIZE # 平均真实柔度
            loss_history.append(val)
            
            # 添加详细调试信息
            u_norm = u.to_torch().abs().sum().item()
            grad_norm_display = density_field.grad.to_torch().abs().mean().item()
            density_mean = grid_4d.mean().item()
            weights_max = weights.max().item()
            
            print(f"Epoch {epoch} | Compliance: {val:.2f} | Phys×{phys_weight:.4f} | Vol: {vol_loss.item():.1f}")
            print(f"  -> Vol_frac: {vol_fraction.item():.4f} (target=0.30) | UnderPen: {under_target_penalty.item():.1f} | Grad_max: {grad_max:.2e}")

    # ==========================================
    # 5. 可视化结果
    # ==========================================
    print("\n" + "="*60)
    print("正在生成可视化...")
    print(f"训练使用的时间点: {FIXED_TARGET_TIMES.cpu().numpy()}")

    with torch.no_grad():
        # 检查模型输出范围（只用固定点）
        sample_out = model(GRID_POINTS_FIXED[:1000])
        sample_t1, sample_t2 = sample_out[:, 0], sample_out[:, 1]
        print(f"模型输出范围: t1=[{sample_t1.min():.4f}, {sample_t1.max():.4f}], t2=[{sample_t2.min():.4f}, {sample_t2.max():.4f}]")
        
        # 使用模型学到的中间时间点（t1和t2的中点）
        test_time = ((sample_t1.mean() + sample_t2.mean()) / 2).item()
        print(f"使用时间点: t = {test_time:.4f} (模型学习的中心时间)")
        
        k = 20.0  # Sigmoid温度系数
        viz_res = 40
        lin = torch.linspace(0, 1, viz_res)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing='ij')
        g_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)
        
        out = model(g_pts)
        t1, t2 = out[:, 0], out[:, 1]
        
        m1 = torch.sigmoid(k * (test_time - t1))
        m2 = torch.sigmoid(k * (t2 - test_time))
        val = (m1 * m2).reshape(viz_res, viz_res, viz_res).cpu().numpy()
        
        print(f"密度值范围: [{val.min():.4f}, {val.max():.4f}]")
        print(f"密度均值: {val.mean():.4f}")
        print(f"密度 > 0.1 的比例: {(val > 0.1).mean():.2%}")
        print(f"密度 > 0.3 的比例: {(val > 0.3).mean():.2%}")
        print(f"密度 > 0.5 的比例: {(val > 0.5).mean():.2%}")
        print("="*60 + "\n")

    # 简化可视化：只显示核心信息
    fig = plt.figure(figsize=(15, 5))

    # 绘制 Loss 曲线
    ax1 = fig.add_subplot(131)
    ax1.plot(loss_history)
    ax1.set_title("Compliance Loss (Lower is Better)")
    ax1.set_xlabel("Epoch / 20")
    ax1.set_ylabel("Compliance")
    ax1.grid(True, alpha=0.3)

    # 绘制密度分布直方图
    ax2 = fig.add_subplot(132)
    threshold = max(val.mean() + val.std() * 0.5, 0.01)
    ax2.hist(val.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    ax2.set_title("Density Distribution")
    ax2.set_xlabel("Density Value")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 绘制 3D 结构（唯一的3D视图）
    ax3 = fig.add_subplot(133, projection='3d')
    structure = val > threshold
    print(f"使用阈值: {threshold:.4f}, 结构体积分数: {structure.mean():.2%}")
    if structure.sum() > 0:
        ax3.voxels(structure, edgecolor='k', alpha=0.8, facecolors='cyan')
        ax3.set_title(f"Structure at t={test_time:.3f}\nVolume: {structure.mean():.1%}")
    else:
        ax3.text(0.5, 0.5, 0.5, 'No structure', ha='center', va='center', fontsize=10)
        ax3.set_title(f"Structure at t={test_time:.3f} (Empty)")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

    print(f"训练完成！最终柔度: {loss_history[-1]:.2f}")


if __name__ == "__main__":
    run_demo()
