import torch
import torch.nn.functional as F

class StructureLossCalculatorTorchSMForce:
    """
    纯 PyTorch 实现的 SM 切削力结构安全损失计算器。

    核心思想：
        - 使用有限差分近似的线弹性模型 + 共轭梯度法 (CG) 求解在重力和 SM 切削力下的位移场 u；
        - 对超过允许总位移范数的体素施加二次惩罚，构造“位移越界”型损失；
        - 在 batch 维度上对样本做 top‑k 截断平均，只关注最差的若干时间步 / 样本。
    """
    
    def __init__(self, 
                 physical_size, 
                 max_resolution,
                 batch_size, 
                 cg_iter, 
                 cg_tol, 
                 E0, 
                 grad_clip_threshold,
                 grad_scale,
                 device='cuda',
                 use_total_mass = 100.0,
                 sm_cutting_force_magnitude = 0.0,
                 sm_max_displacement_norm = 0.6):
        """
        初始化结构损失计算器。
        
        Args:
            physical_size (tuple[float, float, float]):
                物理尺寸 (L_X, L_Y, L_Z)，与采样点在同一物理坐标系下，单位自洽即可。
            max_resolution (int):
                在最长边方向上的网格划分数，另外两维按相同比例缩放得到体素数。
            batch_size (int):
                批次大小（通常对应时间步数量或多场景数量），用于预分配张量大小。
            cg_iter (int):
                共轭梯度法的最大迭代次数。
            cg_tol (float):
                共轭梯度法的相对残差容差（当前实现中早停逻辑已注释，可按需开启）。
            E0 (float):
                基础杨氏模量，对应 rho=1 的材料刚度；实际刚度使用 SIMP: E = rho^3 * E0。
            grad_clip_threshold (float):
                对 dL/drho 做数值裁剪的阈值，>0 时启用，避免梯度爆炸。
            grad_scale (float):
                对最终 dL/drho 的统一缩放因子，用于与其它 loss 平衡。
            device (str or torch.device):
                物理求解和损失计算所在设备，通常为 'cuda'。
            use_total_mass (float or None):
                如果为 None，则按每个样本自身的 rho 体素和作为“总质量”做归一化；
                如果为标量，则使用该固定常数作为归一化质量，方便跨样本对齐尺度。
        """
        # 保存参数
        self.L_X, self.L_Y, self.L_Z = physical_size
        max_length = max(physical_size)
        self.DX = max_length / max_resolution
        
        # 允许的最大总位移范数，超过该值的部分会产生二次惩罚
        self.sm_max_displacement_norm = sm_max_displacement_norm
        self.sm_cutting_force_magnitude = sm_cutting_force_magnitude
        # 损失归一化用的总质量配置，具体含义见上方 docstring
        self.use_total_mass = use_total_mass

        # self.smooth_kernel = torch.ones(1, 1, 3, 3, 3, device=device) / 27.0
        
        RES_X = round(self.L_X / self.DX)
        RES_Y = round(self.L_Y / self.DX)
        RES_Z = round(self.L_Z / self.DX)
        self.TUPLE_RES = (RES_X, RES_Y, RES_Z)
        print(f"RES_X: {RES_X}, RES_Y: {RES_Y}, RES_Z: {RES_Z}")
        
        
        self.BATCH_SIZE = batch_size
        self.CG_ITER = cg_iter
        self.CG_TOL = cg_tol
        self.E0 = E0
        self.GRAD_CLIP_THRESHOLD = grad_clip_threshold
        self.GRAD_SCALE = grad_scale
        self.rho_threshold = 0.1
        self.device = torch.device(device)
        
        print(f"[StructureLossCalculatorTorchSMForce] Initialized with:")
        print(f"  - Resolution: {self.TUPLE_RES}")
        print(f"  - Grid spacing (DX): {self.DX:.6f}")
        print(f"  - Batch size: {self.BATCH_SIZE}")
        print(f"  - CG iterations: {self.CG_ITER}")
        print(f"  - SM cutting force magnitude: {self.sm_cutting_force_magnitude}")
        print(f"  - SM max displacement norm: {self.sm_max_displacement_norm}")
        print(f"  - Device: {self.device}")

    def _precompute_stiffness_components(self, rho):
        """
        [优化] 预计算刚度矩阵系数，专为 torch.compile 设计
        将不变的系数计算提取到循环外
        """
        # 1. 计算基础系数
        E = (rho ** 3) * self.E0 + 1e-9 * self.E0
        coeff = 0.5 / (self.DX * self.DX)
        
        # 2. Pad 操作
        E_pad = F.pad(E, (1, 1, 1, 1, 1, 1), mode='constant', value=0.0)
        E_center = E_pad[:, 1:-1, 1:-1, 1:-1]
        
        # 3. 预计算 6 个方向的连接系数
        # 即使是 torch.compile，这种明确的列表展开通常比循环更容易被优化
        neighbor_slices = [
            (slice(0, -2), slice(1, -1), slice(1, -1)),  # x-
            (slice(2, None), slice(1, -1), slice(1, -1)),  # x+
            (slice(1, -1), slice(0, -2), slice(1, -1)),  # y-
            (slice(1, -1), slice(2, None), slice(1, -1)),  # y+
            (slice(1, -1), slice(1, -1), slice(0, -2)),  # z-
            (slice(1, -1), slice(1, -1), slice(2, None)),  # z+
        ]
        
        links = []
        diag_sum = torch.zeros_like(E_center)
        
        # 遍历6个邻居
        for idx, (sx, sy, sz) in enumerate(neighbor_slices):
            E_neigh = E_pad[:, sx, sy, sz]
            
            # 弹簧刚度 k = (E_center + E_neigh) / 2 / DX^2
            k_link = (E_center + E_neigh) * coeff
            
            # 构建简化的 mask (只针对当前方向)
            if idx == 0: k_link[:, 0, :, :] = 0.0      # x-
            elif idx == 1: k_link[:, -1, :, :] = 0.0   # x+
            elif idx == 2: k_link[:, :, 0, :] = 0.0    # y-
            elif idx == 3: k_link[:, :, -1, :] = 0.0   # y+
            elif idx == 4: k_link[:, :, :, 0] = 0.0    # z-
            elif idx == 5: k_link[:, :, :, -1] = 0.0   # z+
            
            links.append(k_link.unsqueeze(-1)) # [B, X, Y, Z, 1]
            diag_sum += k_link
            
        diag_term = diag_sum.unsqueeze(-1) # [B, X, Y, Z, 1]
        
        return diag_term, links

    def _solve_cg(self, rho, b_force):
        """
        共轭梯度法求解线性方程组 K*u = b (优化版)
        
        Args:
            rho: [B, X, Y, Z] 密度场
            b_force: [B, X, Y, Z, 3] 外力场
        
        Returns:
            u: [B, X, Y, Z, 3] 位移场
        """
        # === 优化部分：预计算刚度矩阵 ===
        diag_K, links = self._precompute_stiffness_components(rho)
        
        # 初始化
        u = torch.zeros_like(b_force)
        r = b_force.clone()
        
        # 边界条件：z=0处残差为0
        r[:, :, :, 0, :] = 0.0
        
        p = r.clone()
        r_dot_r = (r * r).sum(dim=(1, 2, 3, 4))  # [B]
        
        neighbor_slices = [
            (slice(0, -2), slice(1, -1), slice(1, -1)),  # x-
            (slice(2, None), slice(1, -1), slice(1, -1)),  # x+
            (slice(1, -1), slice(0, -2), slice(1, -1)),  # y-
            (slice(1, -1), slice(2, None), slice(1, -1)),  # y+
            (slice(1, -1), slice(1, -1), slice(0, -2)),  # z-
            (slice(1, -1), slice(1, -1), slice(2, None)),  # z+
        ]

        # CG迭代
        # r_dot_r_initial = r_dot_r.clone()

        for _ in range(self.CG_ITER):
            # === 矩阵乘法展开 (Matrix-Vector Product) ===
            # Pad p (注意这里是 p 不是 u，因为我们在算 Ap = K * p)
            p_pad = F.pad(p.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1), mode='constant', value=0.0)
            p_pad = p_pad.permute(0, 2, 3, 4, 1)
            
            # 对角项
            Ap = diag_K * p 
            
            # 非对角项 (直接使用预计算好的 links)
            for idx, (sx, sy, sz) in enumerate(neighbor_slices):
                p_neigh = p_pad[:, sx, sy, sz, :]
                Ap -= links[idx] * p_neigh
            
            # 强制边界条件
            Ap[:, :, :, 0, :] = 0.0
            
            # alpha = (r . r) / (p . Ap)
            p_dot_Ap = (p * Ap).sum(dim=(1, 2, 3, 4)) + 1e-10  # [B]
            alpha = r_dot_r / p_dot_Ap  # [B]
            
            # 更新解和残差
            alpha_expanded = alpha.view(-1, 1, 1, 1, 1)
            u = u + alpha_expanded * p
            r_new = r - alpha_expanded * Ap
            
            # 强制边界条件（防止浮点误差漂移）
            r_new[:, :, :, 0, :] = 0.0
            
            r_dot_r_new = (r_new * r_new).sum(dim=(1, 2, 3, 4))  # [B]

            # # 早停检查：基于相对残差收敛
            # # 计算相对残差：||r|| / ||r_initial||
            # relative_residual = torch.sqrt(r_dot_r_new / (r_dot_r_initial + 1e-20))  # [B]
            # # 如果所有批次的相对残差都小于容差，则提前停止
            # if torch.all(relative_residual < self.CG_TOL):
            #     break
            
            # beta = (r_new . r_new) / (r . r)
            beta = r_dot_r_new / (r_dot_r + 1e-20)  # [B]
            
            # 更新搜索方向
            p = r_new + beta.view(-1, 1, 1, 1, 1) * p
            
            # 更新残差
            r = r_new
            r_dot_r = r_dot_r_new
        
        return u

    def _compute_loss(self, u, rho):
        """
        计算基于位移约束的结构损失（top‑k 截断平均）。
        
        Args:
            u: [B, X, Y, Z, 3]
                位移场。
            rho: [B, X, Y, Z]
                密度场，通常在 [0, 1]，rho 较小表示空洞或软材料。
        
        Returns:
            loss (torch.Tensor):
                标量，batch 内 top‑k 样本（最差一半左右）的平均惩罚损失。
            top_k_indices (torch.Tensor):
                shape [K]，被选中的样本索引，用于反向传播中构建 batch mask。
        """
        displacement_norm = torch.linalg.norm(u, dim=-1)

        # # 先增加 mask_material 约束，只对有效材料位置进行统计和输出
        # mask_material = (rho > self.rho_threshold)
        # drop_distance_masked = drop_distance[mask_material > 0]
        # if drop_distance_masked.numel() > 0:
        #     print("drop_distance (masked): ", drop_distance_masked.max().item(), drop_distance_masked.min().item())
        #     # 输出 drop_distance 各个百分位对应的值（仅对mask后的体素）
        #     percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        #     drop_distance_flat = drop_distance_masked.flatten()
        #     percentile_values = torch.quantile(drop_distance_flat, torch.tensor([p/100.0 for p in percentiles], device=drop_distance.device))
        #     for p, v in zip(percentiles, percentile_values):
        #         print(f"drop_distance {p}th percentile: {v.item():.6f}")
        # else:
        #     print("No valid drop_distance values after applying mask_material.")


        # 构造“有效失败区域”掩码：
        #   1) 有足够材料 (rho > rho_threshold)
        #   2) 总位移超过允许阈值 (displacement_norm > sm_max_displacement_norm)
        with torch.no_grad():
            mask_material = (rho > self.rho_threshold).float()
            mask_fail = (displacement_norm > self.sm_max_displacement_norm).float()
            mask_target = mask_material * mask_fail

            # print("mask_material中1的个数:", mask_material.sum().item())
            # print("mask_fail中1的个数:", mask_fail.sum().item())
            # print("mask_target中1的个数:", mask_target.sum().item())
            # print("mask中最大u_z:", drop_distance[mask_target == 1].max().item())
            # print("dx: ", self.DX)
        
        excess_displacement = torch.relu(displacement_norm - self.sm_max_displacement_norm)
        penalty_field = rho * (excess_displacement ** 2)

        penalty_masked = penalty_field * mask_target  # 只对违规体素计入损失

        # 每个样本的总惩罚（数值上与“加权下沉能量”类似）
        total_penalty = penalty_masked.sum(dim=(1, 2, 3))
        
        if self.use_total_mass is None:
            total_mass = rho.sum(dim=(1, 2, 3)) + 1e-6
        else:
            total_mass = self.use_total_mass

        # 每个样本的归一化损失：总惩罚 / 总质量
        loss_per_sample = total_penalty / total_mass

        # 在 batch 维度上做 top‑k 截断，只保留最差的一部分样本
        B = loss_per_sample.shape[0]
        k = max(1, int(B * 0.8))
        top_k_loss, top_k_indices = torch.topk(loss_per_sample, k, largest = True)
        loss = top_k_loss.mean()
    
        return loss, top_k_indices

    def _compute_adjoint_gradient(self, u, lambda_vec, rho, excess_drop, mask, total_mass, batch_mask_view):
        """
        使用伴随方法计算梯度 dL/drho。

        三个主要来源：
            - Direct term:   损失显式依赖 rho 的部分（rho * excess^2）；
            - Force term:    重力载荷 f = -9.8 * rho * e_z 对 rho 的显式依赖；
            - Stiffness term:刚度矩阵 K(rho) 对 rho 的依赖，通过 (u_i - u_j)·(λ_i - λ_j) 体现。
        """

        # === Term 1: Direct term (dL/drho) ===
        # Loss = Sum(Mask * rho * excess^2) / Mass
        # d(Numerator)/drho = Mask * excess^2
        # 为了数值稳定，忽略分母 total_mass 对 rho 的导数
        term_direct = mask * (excess_drop ** 2) / total_mass

        # === Term 2: Adjoint force term ===
        # 自重载荷 f_z = -9.8 * rho，因此 df/drho 在 z 分量上为常数 -9.8
        # 伴随项贡献为 lambda^T * df/drho ~= lambda_z * (-9.8)
        # [Trick]: 乘以 0.1 系数来降低“减重”驱动力，避免过度削弱结构连接
        GRAVITY_FACTOR = 0.1
        term_adjoint_force = lambda_vec[..., 2] * (-9.8) * GRAVITY_FACTOR

        # === Term 3: Adjoint stiffness term ===
        # lambda^T * dK/drho * u
        # 这是产生"连接"动力的核心项，必须保留
        dE_drho = 3.0 * rho * rho * self.E0
        
        
        # 计算邻居差分项：sum over neighbors of (u_i - u_j) · (lambda_i - lambda_j)
        # 使用pad+slice向量化实现
        u_pad = F.pad(u.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1), mode='constant', value=0.0)
        u_pad = u_pad.permute(0, 2, 3, 4, 1)  # [B, X+2, Y+2, Z+2, 3]
        
        lambda_pad = F.pad(lambda_vec.permute(0, 4, 1, 2, 3), (1, 1, 1, 1, 1, 1), mode='constant', value=0.0)
        lambda_pad = lambda_pad.permute(0, 2, 3, 4, 1)  # [B, X+2, Y+2, Z+2, 3]
        
        u_center = u  # [B, X, Y, Z, 3]
        lambda_center = lambda_vec  # [B, X, Y, Z, 3]
        
        sum_mixed_diff = torch.zeros_like(rho)
        
        # 创建有效邻居mask（与_matrix_vector_product保持一致）
        device = u.device
        X, Y, Z = rho.shape[1], rho.shape[2], rho.shape[3]
        valid_masks = []
        
        # x- direction: valid if i > 0
        mask_xm = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_xm[:, 0, :, :] = 0.0
        valid_masks.append(mask_xm)
        
        # x+ direction: valid if i < X-1
        mask_xp = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_xp[:, -1, :, :] = 0.0
        valid_masks.append(mask_xp)
        
        # y- direction: valid if j > 0
        mask_ym = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_ym[:, :, 0, :] = 0.0
        valid_masks.append(mask_ym)
        
        # y+ direction: valid if j < Y-1
        mask_yp = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_yp[:, :, -1, :] = 0.0
        valid_masks.append(mask_yp)
        
        # z- direction: valid if k > 0
        mask_zm = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_zm[:, :, :, 0] = 0.0
        valid_masks.append(mask_zm)
        
        # z+ direction: valid if k < Z-1
        mask_zp = torch.ones((1, X, Y, Z), device=device, dtype=rho.dtype)
        mask_zp[:, :, :, -1] = 0.0
        valid_masks.append(mask_zp)
        
        # 6个邻居
        neighbor_slices = [
            (slice(0, -2), slice(1, -1), slice(1, -1)),  # x-
            (slice(2, None), slice(1, -1), slice(1, -1)),  # x+
            (slice(1, -1), slice(0, -2), slice(1, -1)),  # y-
            (slice(1, -1), slice(2, None), slice(1, -1)),  # y+
            (slice(1, -1), slice(1, -1), slice(0, -2)),  # z-
            (slice(1, -1), slice(1, -1), slice(2, None)),  # z+
        ]
        
        for idx, (sx, sy, sz) in enumerate(neighbor_slices):
            u_neigh = u_pad[:, sx, sy, sz, :]  # [B, X, Y, Z, 3]
            lambda_neigh = lambda_pad[:, sx, sy, sz, :]  # [B, X, Y, Z, 3]
            
            u_diff = u_center - u_neigh  # [B, X, Y, Z, 3]
            lambda_diff = lambda_center - lambda_neigh  # [B, X, Y, Z, 3]
            
            # 点积
            mixed_product = (u_diff * lambda_diff).sum(dim=-1)  # [B, X, Y, Z]
            
            # 应用有效邻居mask：只统计域内邻居的贡献
            sum_mixed_diff += mixed_product * valid_masks[idx]
        
        # === Term 3: Adjoint stiffness term ===
        # Under K * lambda = - dL/du, the stiffness contribution is: + lambda^T * dK/drho * u
        # For this stencil discretization, it corresponds to:
        #   + (dE/drho / (2*DX^2)) * Σ_neighbors (u_i-u_j) · (lambda_i-lambda_j)
        term_adjoint_stiffness = (dE_drho / (2.0 * self.DX * self.DX)) * sum_mixed_diff
        
        # 总梯度
        grad = term_direct - term_adjoint_force + term_adjoint_stiffness
        grad = grad * batch_mask_view

        # grad smooth （模糊梯度；均值滤波）
        # with torch.no_grad():
        #     padding = 1
        #     grad_smooth = grad.unsqueeze(1)
        #     grad_smooth = F.pad(grad_smooth, pad = (padding, padding, padding, padding, padding, padding), mode='replicate')
        #     grad_smooth = F.conv3d(grad_smooth, self.smooth_kernel).squeeze(1)
        #     grad = 0.5 * grad + 0.5 * grad_smooth
        
        # 边界条件：z=0处梯度为0
        grad[:, :, :, 0] = 0.0
        
        return grad

    def point_to_batch_grid(self, points, weights, tuple_resolution):
        """
        将点云权重映射到批量网格
        
        Args:
            points: [N_pts, 3]
                采样点物理坐标（与 physical_size / DX 同一坐标系）。
            weights: [N_pts, Batch_Size]
                每个点在不同 batch / 时间步上的密度值。
            tuple_resolution: (RES_X, RES_Y, RES_Z)
                目标体素网格的分辨率。
        
        Returns:
            grid: [Batch_Size, RES_X, RES_Y, RES_Z]
                体素化后的密度网格，对落入同一体素的多个点取简单平均。
        """
        N_pts, B_size = weights.shape
        device = points.device
        
        RES_X, RES_Y, RES_Z = tuple_resolution
        
        coords_x = (points[:, 0] / self.DX).long().clamp(0, RES_X - 1)
        coords_y = (points[:, 1] / self.DX).long().clamp(0, RES_Y - 1)
        coords_z = (points[:, 2] / self.DX).long().clamp(0, RES_Z - 1)
        
        voxel_idx = coords_x * (RES_Y * RES_Z) + coords_y * RES_Z + coords_z
        voxel_idx_expanded = voxel_idx.unsqueeze(1).repeat(1, B_size)
        
        batch_offsets = (torch.arange(B_size, device=device) * (RES_X * RES_Y * RES_Z)).unsqueeze(0)
        flat_indices = (voxel_idx_expanded + batch_offsets).view(-1)
        flat_weights = weights.view(-1)
        
        total_size = B_size * (RES_X * RES_Y * RES_Z)
        grid_density = torch.zeros(total_size, device=device, dtype=flat_weights.dtype)
        grid_count = torch.zeros(total_size, device=device, dtype=flat_weights.dtype)
        
        grid_density.scatter_add_(0, flat_indices, flat_weights)
        grid_count.scatter_add_(0, flat_indices, torch.ones_like(flat_weights))
        
        grid_final = grid_density / (grid_count + 1e-5)
        
        return grid_final.view(B_size, RES_X, RES_Y, RES_Z)

    def _points_to_voxel_indices(self, points, tuple_resolution):
        RES_X, RES_Y, RES_Z = tuple_resolution
        coords_x = (points[:, 0] / self.DX).long().clamp(0, RES_X - 1)
        coords_y = (points[:, 1] / self.DX).long().clamp(0, RES_Y - 1)
        coords_z = (points[:, 2] / self.DX).long().clamp(0, RES_Z - 1)
        return coords_x, coords_y, coords_z

    def structureLoss(self, samplePoints, density, sm_force_points, sm_normal_vectors):
        """
        结构拓扑优化 / 结构安全损失函数入口。
        
        Args:
            samplePoints: [N_pts, 3]
                采样点坐标（通常来自隐式场 / MLP 输出的采样）。
            density: [N_pts, Batch_size]
                对应采样点在不同时间步 / 场景下的密度（设计变量），范围一般在 [0, 1]。
            sm_force_points: [Batch_size, 3]
                与每个时间样本绑定的 SM base point，坐标需要与 samplePoints 一样为结构求解归一化坐标。
            sm_normal_vectors: [Batch_size, 3]
                从材料指向外部的 SM normal。切削力会取反，指向材料内部。
        
        Returns:
            loss_real (torch.Tensor):
                标量，结构损失（已经封装了物理求解 + top‑k + 伴随梯度，直接可反向）。
            simp_penalty (torch.Tensor):
                标量，SIMP 风格的二值化 / 平滑惩罚项，鼓励密度接近 0/1。
        """
        # 结构求解对数值精度比较敏感，强制使用 float32，避免 AMP/autocast 下 CG 和伴随求解不稳定。
        samplePoints = samplePoints.to(device=self.device, dtype=torch.float32)
        density = density.to(device=self.device, dtype=torch.float32)
        sm_force_points = sm_force_points.to(device=self.device, dtype=torch.float32)
        sm_normal_vectors = sm_normal_vectors.to(device=self.device, dtype=torch.float32)

        # Step 1: 将点云映射到网格
        grid_4d = self.point_to_batch_grid(samplePoints, density, self.TUPLE_RES)
        grid_4d = grid_4d.to(self.device, dtype=torch.float32)
        
        # 使用自定义Function进行前向和反向传播
        loss_real = StructureFunction.apply(
            grid_4d, 
            sm_force_points,
            sm_normal_vectors,
            self
        )
        
        # SIMP惩罚
        simp_penalty = (density * (1.0 - density)).mean()
        
        return loss_real, simp_penalty


# ========================================================================
# 自定义Autograd Function：实现伴随方法
# ========================================================================
class StructureFunction(torch.autograd.Function):
    """
    自定义的 PyTorch Function，实现 FEM + 伴随法的端到端求导。

    - Forward:
        给定密度场 rho，构造重力载荷，使用 CG 求解位移场 u，
        再根据位移越界构造 top‑k 截断损失。
    - Backward:
        先构造 dL/du 的等效“虚拟力”，再解伴随方程 K * lambda = -dL/du，
        最终组合 direct / force / stiffness 三部分得到 dL/drho。
    """
    
    @staticmethod
    def forward(ctx, density, sm_force_points, sm_normal_vectors, calculator):
        """
        前向传播：求解 FEM 并计算基于位移约束的 top‑k 截断损失。
        
        Args:
            density: [B, X, Y, Z]
                密度场，通常由上游网络输出（经过体素化后的结果）。
            calculator: StructureLossCalculatorTorchSMForce
                携带物理尺寸、刚度参数、阈值等配置的计算器实例。
        
        Returns:
            loss (torch.Tensor):
                标量，batch 内 top‑k 样本的平均结构损失。
        """
        ctx.calculator = calculator
        
        # 构建外力场（重力 + 指向材料内部的 SM 切削力）
        b_force = torch.zeros((*density.shape, 3), device=density.device, dtype=density.dtype)
        b_force[..., 2] = -9.8 * density
        if calculator.sm_cutting_force_magnitude != 0.0 and sm_force_points.numel() > 0:
            force_x, force_y, force_z = calculator._points_to_voxel_indices(
                sm_force_points, calculator.TUPLE_RES
            )
            batch_idx = torch.arange(density.shape[0], device=density.device)
            cutting_force = -calculator.sm_cutting_force_magnitude * sm_normal_vectors
            b_force[batch_idx, force_x, force_y, force_z, :] += cutting_force
        b_force[:, :, :, 0, :] = 0.0  # 边界条件
        
        # 求解位移场 K*u = f（前向物理模拟）
        with torch.no_grad():  # 不保存中间计算图，节省显存
            u = calculator._solve_cg(density, b_force)
        
        # 计算损失
        loss, top_k_indices = calculator._compute_loss(u, density)
        
        # 保存必要的张量供反向传播使用
        ctx.save_for_backward(density, u, top_k_indices)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_loss):
        """
        反向传播：使用伴随方法计算对密度场的精确梯度。
        
        Args:
            grad_loss:
                标量或 shape 可 broadcast 到标量的张量，来自上游 loss 的链式梯度（通常为 1）。
        
        Returns:
            grad_density: [B, X, Y, Z]
                对 density 的梯度；对 calculator 参数不求导，因此返回 None。
        """
        density, u, top_k_indices = ctx.saved_tensors
        calculator = ctx.calculator

        B = density.shape[0]
        K = top_k_indices.shape[0]

        # === 0. 构建mask ===
        batch_mask = torch.zeros(B, device=density.device, dtype=density.dtype)
        batch_mask[top_k_indices] = 1.0 / K #对应mean
        batch_mask_view = batch_mask.view(-1, 1, 1, 1)

        # === 1. 重算中间变量 (为了得到 Mask 和 excess_displacement) ===
        displacement_norm = torch.linalg.norm(u, dim=-1)
        
        with torch.no_grad():
            mask_material = (density > calculator.rho_threshold).float()
            mask_fail = (displacement_norm > calculator.sm_max_displacement_norm).float()
            mask_target = mask_material * mask_fail
            
        excess_displacement = torch.relu(displacement_norm - calculator.sm_max_displacement_norm)

        if calculator.use_total_mass is None:
            total_mass = density.sum(dim=(1,2,3)).view(-1, 1, 1, 1) + 1e-6
        else:
            total_mass = calculator.use_total_mass

        # === 2. 计算 dL/du (伴随方程的 RHS) ===
        # d||u||/du = u / (||u|| + eps)
        dL_du = mask_target.unsqueeze(-1) * density.unsqueeze(-1) * 2 * excess_displacement.unsqueeze(-1)
        dL_du = dL_du * u / (displacement_norm.unsqueeze(-1) + 1e-12)
        total_mass_vec = total_mass.unsqueeze(-1) if torch.is_tensor(total_mass) else total_mass
        dL_du = dL_du / total_mass_vec
        dL_du = dL_du * batch_mask_view.unsqueeze(-1)
        
        # === 3. 求解伴随方程 K * lambda = -dL/du ===
        with torch.no_grad():
            lambda_vec = calculator._solve_cg(density, -dL_du)

        # === 4. 计算对密度的梯度 ===
        with torch.no_grad():
            grad_density = calculator._compute_adjoint_gradient(
                u, lambda_vec, density, excess_displacement, mask_target, total_mass, batch_mask_view
            )

        if torch.isnan(grad_density).any() or torch.isinf(grad_density).any():
            print(f"[StructureLoss] Error: Gradient contains NaN or Inf!")
            print(f"  - Has NaN: {torch.isnan(grad_density).any().item()}")
            print(f"  - Has Inf: {torch.isinf(grad_density).any().item()}")
            raise ValueError("Gradient computation resulted in NaN/Inf")
        
        # 梯度裁剪
        if calculator.GRAD_CLIP_THRESHOLD > 0:
            grad_density = grad_density.clamp(-calculator.GRAD_CLIP_THRESHOLD, calculator.GRAD_CLIP_THRESHOLD)
        # 梯度缩放
        grad_density = grad_density * calculator.GRAD_SCALE
        
        # 乘以外部梯度（链式法则）
        grad_density = grad_density * grad_loss
        
        return grad_density, None, None, None


StructureLossCalculatorTorch = StructureLossCalculatorTorchSMForce
