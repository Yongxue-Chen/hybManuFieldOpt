import torch
import numpy as np

class CollisionDetector:
    def __init__(self, mesh, tool_params, device='cuda', floor_z=0.0):
        """
        tool_params = {
            'r_tip': 刀尖半径,
            'r_shank': 刀杆半径,
            'l_tool': 刀具从球心到 Holder 底端的长度 (刀杆长度),
            'r_holder': Holder 的半径,
            'l_shank': Holder 的长度
        }
        """
        self.device = device
        self.floor_z = float(floor_z)
        
        self.v0 = torch.tensor(mesh.vertices[mesh.faces[:, 0]], dtype=torch.float32).to(device)
        self.v1 = torch.tensor(mesh.vertices[mesh.faces[:, 1]], dtype=torch.float32).to(device)
        self.v2 = torch.tensor(mesh.vertices[mesh.faces[:, 2]], dtype=torch.float32).to(device)
        self.edge1 = self.v1 - self.v0
        self.edge2 = self.v2 - self.v0

        self.face_a = torch.sum(self.edge1 * self.edge1, dim=-1)
        self.face_b = torch.sum(self.edge1 * self.edge2, dim=-1)
        self.face_c = torch.sum(self.edge2 * self.edge2, dim=-1)
        self.face_det = self.face_a * self.face_c - self.face_b * self.face_b
        
        # 刀具参数
        self.r_tip = tool_params['r_tip']      # 球头半径
        self.r_shank = tool_params.get('r_shank', self.r_tip)  # 刀杆半径
        self.l_tool = tool_params['l_tool']    # 刀杆长度
        self.r_holder = tool_params.get('r_holder', tool_params.get('r_shank', self.r_tip))
        self.l_holder = tool_params['l_shank'] # holder 长度
        
        # 预生成高斯半球姿态向量 (N, 3)
        self.v_poses = self._generate_fibonacci_hemisphere(10).to(device)

    def _generate_fibonacci_hemisphere(self, n):
        """
        生成分布均匀的上半球采样姿态 (Z轴向上)
        """
        # 黄金角度 (弧度)
        phi = np.pi * (3. - np.sqrt(5.)) 
        
        indices = torch.arange(n)
        
        # 1. 垂直方向（极角方向）的线性分布
        # z 从 1.0 (正上方) 降到 0.0 (水平面)，代表 cos(theta)
        z = 1.0 - (indices / float(n - 1))
        
        # 2. 计算当前层在 XY 平面上的投影半径 (sin(theta))
        radius = torch.sqrt(torch.clamp(1 - z * z, min=0))
        
        # 3. 角度在水平面上螺旋展开
        theta = phi * indices
        
        x = torch.cos(theta) * radius
        y = torch.sin(theta) * radius
        
        # 返回 (n, 3) 张量，Z 轴现在是极轴（向上）
        return torch.stack([x, y, z], dim=-1)

    def _is_in_aabb_xy(self, points, aabb):
        """
        points: (N, 3) 形状的张量
        """
        # 显式提取列，确保返回 (N,) 形状的布尔张量
        x_in = (points[:, 0] >= aabb[0]) & (points[:, 0] <= aabb[1])
        y_in = (points[:, 1] >= aabb[2]) & (points[:, 1] <= aabb[3])
        return x_in & y_in

    def check_hit_as_stop(self, p_curr, p_cand, r_supp, segment_buffer_tensor):
        """
        Hit-as-Stop 检测：判断当前生长步 [p_curr, p_cand] 是否触达终点
        返回: (is_hit, hit_point)
        """
        # 转换输入为 Tensor 并移至 GPU
        p1 = torch.as_tensor(p_curr, dtype=torch.float32, device=self.device)
        p2 = torch.as_tensor(p_cand, dtype=torch.float32, device=self.device)
        
        # 1. 检查底面碰撞 (Z=floor_z)
        # 如果 p2 穿过或到达支撑最低平面，计算交点
        floor_z = torch.tensor(self.floor_z, dtype=torch.float32, device=self.device)
        if p2[2] <= floor_z:
            t = (p1[2] - floor_z) / (p1[2] - p2[2] + 1e-9)
            hit_p = p1 + t * (p2 - p1)
            hit_p[2] = floor_z
            return True, hit_p

        # 2. 检测与 Mesh 的干涉 (线段与三角形求交)
        # 使用 Möller-Trumbore 算法并行检测线段是否穿过任何面片
        hit_mesh, p_mesh = self._ray_mesh_intersect(p1, p2)
        if hit_mesh:
            return True, p_mesh

        # 3. 检测与已存支撑的干涉 (线段与线段距离)
        # 若距离 < 2 * r_supp，则认为支撑连接成功
        if segment_buffer_tensor.shape[0] > 0:
            hit_supp, p_supp = self._check_support_collision(p1, p2, r_supp, segment_buffer_tensor)
            if hit_supp:
                return True, p_supp

        return False, None

    def _ray_mesh_intersect(self, p1, p2):
        """
        极致优化版：预计算变量 + 减少显存拷贝
        """
        d = p2 - p1
        ray_len = torch.norm(d)
        if ray_len < 1e-9: return False, None
        dir_vec = d / ray_len

        # 直接使用缓存的 edge1, edge2, v0
        # 扩展 dir_vec 到 (N_faces, 3) 时尽量不产生数据副本
        h = torch.cross(dir_vec.expand(self.edge1.shape), self.edge2, dim=-1)
        a = torch.sum(self.edge1 * h, dim=-1)

        # 1. 第一层过滤：平行面
        eps = 1e-8
        mask = a.abs() > eps
        
        # 2. 第二层：重心坐标 u (利用广播减少冗余计算)
        s = p1 - self.v0
        # 提前用 torch.where 处理除零，避免产生 INF 干扰后续
        f = torch.where(mask, 1.0 / a, torch.tensor(0.0, device=self.device))
        u = f * torch.sum(s * h, dim=-1)
        mask &= (u >= 0.0) & (u <= 1.0)

        # 3. 如果所有三角形都已被过滤，提前返回 (CPU 端的 Short-circuit)
        if not mask.any(): return False, None

        # 4. 第三层：重心坐标 v
        q = torch.cross(s, self.edge1, dim=-1)
        v = f * torch.sum(dir_vec.expand(self.edge2.shape) * q, dim=-1)
        mask &= (v >= 0.0) & (u + v <= 1.0)
        
        # 5. 第四层：距离 t
        t = f * torch.sum(self.edge2 * q, dim=-1)
        mask &= (t > 1e-6) & (t <= ray_len)

        if mask.any():
            t_valid = torch.where(mask, t, torch.tensor(float('inf'), device=self.device))
            t_min, min_idx = t_valid.min(dim=0)
            if t_min == float('inf'): return False, None
            return True, p1 + t_min * dir_vec
        
        return False, None
    
    def _check_support_collision(self, p1, p2, r_supp, segment_buffer_tensor):
        """
        计算线段 p1-p2 到 segment_buffer 中所有线段的最短距离
        """
        segments = segment_buffer_tensor
        
        # 计算线段到线段的最短距离 (向量化实现)
        # 返回每个 buffer 段到当前段的最小距离
        dists, t_params = self._batch_line_line_dist(p1, p2, segments)
        
        threshold = 2.0 * r_supp
        mask = dists < threshold
        
        if torch.any(mask):
            # 找到第一个碰撞位置，并计算在该位置的坐标
            # 这里简化处理，取第一个碰撞段的对应点
            t_val = t_params[mask][0] # 当前线段上的比例参数
            hit_point = p1 + t_val * (p2 - p1)
            return True, hit_point
            
        return False, None
    
    def _batch_line_line_dist(self, p1, p2, segments):
        """
        向量化计算：单条线段 (p1-p2) 与一批线段 (segments) 的最短距离
        segments shape: (N, 2, 3) -> N条线段，每条2个端点，每个点3个坐标
        """
        # 1. 提取向量
        u = p2 - p1  # 方向向量 1 (3,)
        q1 = segments[:, 0, :] # 起点集 (N, 3)
        q2 = segments[:, 1, :] # 终点集 (N, 3)
        v = q2 - q1  # 方向向量集 2 (N, 3)
        w0 = p1 - q1 # 连接向量 (N, 3)

        # 2. 计算点积 (Dot Products)
        a = torch.dot(u, u)
        b = torch.mv(v, u)
        c = torch.sum(v * v, dim=-1)
        d = torch.mv(w0, u)
        e = torch.sum(v * w0, dim=-1)

        # 3. 计算参数 s, t
        denom = a * c - b * b
        # 处理平行情况 (denom 接近 0)
        eps = 1e-8
        
        # 默认 s = 0 时的计算
        s_num = b * e - c * d
        t_num = a * e - b * d

        # 4. 参数约束到 [0, 1] 范围内
        # 这是一个简化版的参数化求解，但在 GPU 上非常高效
        s = torch.clamp(s_num / (denom + eps), 0.0, 1.0)
        t = torch.clamp((b * s + e) / (c + eps), 0.0, 1.0)
        
        # 重新修正 s 以保证全局最优
        s = torch.clamp((b * t - d) / (a + eps), 0.0, 1.0)

        # 5. 计算最短距离向量
        # p1 + s*u - (q1 + t*v)
        closest_p1 = p1 + s.unsqueeze(-1) * u
        closest_p2 = q1 + t.unsqueeze(-1) * v
        diff = closest_p1 - closest_p2
        
        dists = torch.norm(diff, dim=-1)
        return dists, s # 返回距离和当前线段上的比例参数 s
    
    def is_sm_accessible(self, point_np, segment_buffer_tensor, aabb, r_supp):
        """
        点 point 处的避障姿态验证 (考虑刀尖偏移)
        point: 支撑点坐标 (3,)
        segment_buffer_tensor: 支撑线段集 (N, 2, 3)
        aabb: [xmin, xmax, ymin, ymax, zmin, zmax]
        """

        point = torch.as_tensor(point_np, dtype=torch.float32, device=self.device)

        # --- 1. 计算 256 个姿态下的关键点坐标 (Batching) ---
        # self.v_poses: (256, 3)
        # 所有的计算都基于广播机制
        
        # 球头中心：从刀尖点沿姿态向量偏移 R_tip
        p_centers = point + self.v_poses * self.r_tip
        
        # 刀杆末端：从球心偏移 L_tool
        p_shank_ends = p_centers + self.v_poses * self.l_tool
        
        # 刀柄末端：从刀杆末端偏移 L_holder
        p_holder_ends = p_shank_ends + self.v_poses * self.l_holder

        # --- 2. 批量碰撞判定 (256个姿态并行) ---
        is_pose_safe = torch.ones(self.v_poses.shape[0], dtype=torch.bool, device=self.device)

        # A. 球头 (Tip Sphere) 碰撞
        # 判定：球心在 p_centers，半径 r_tip
        sphere_collision = self._batch_dist_sphere_env(p_centers, self.r_tip, r_supp, segment_buffer_tensor, aabb)
        is_pose_safe &= ~sphere_collision

        # B. 刀杆 (Shank Cylinder) 碰撞
        # 判定：线段 [p_centers, p_shank_ends]，半径 r_shank
        shank_collision = self._batch_dist_segment_env(p_centers, p_shank_ends, self.r_shank, r_supp, segment_buffer_tensor, aabb)
        is_pose_safe &= ~shank_collision

        # C. 刀柄 (Holder Cylinder) 碰撞
        # 判定：线段 [p_shank_ends, p_holder_ends]，半径 r_holder
        holder_collision = self._batch_dist_segment_env(p_shank_ends, p_holder_ends, self.r_holder, r_supp, segment_buffer_tensor, aabb, noSupport = True)
        is_pose_safe &= ~holder_collision

        # --- 3. 挑选最优姿态 ---
        if is_pose_safe.any():
            # 找到所有安全姿态中索引最小的（因为 v_poses[0] 是 [0,0,1]，最垂直）
            idx = torch.where(is_pose_safe)[0][0]
            return True, self.v_poses[idx]

        return False, None
    
    def _batch_dist_sphere_env(self, centers, radius, r_supp, segments, aabb):
        """
        centers: (256, 3)
        """
        # 1. 只有 XY 在 AABB 内部的球体才需要检测
        is_inside = self._is_in_aabb_xy(centers, aabb)

        # 2. 计算点到 Mesh 的距离 (256,)
        dist_mesh = self._batch_dist_point_mesh(centers, aabb[4])
        
        # 3. 计算点到支撑线段的距离 (256,)
        if segments.shape[0] > 0:
            dist_supp = self._batch_dist_point_segments(centers, segments)
        else:
            dist_supp = torch.full((centers.shape[0],), float('inf'), device=self.device)
            
        # 碰撞判定：在内部 & (离模型太近 或 离支撑太近)
        hit_mesh = is_inside & (dist_mesh < radius)
        hit_supp = is_inside & (dist_supp < (radius + r_supp))

        return hit_mesh | hit_supp

    def _batch_dist_point_segments(self, points, segments):
        """
        计算一批点到一批线段的最短距离
        points: (M, 3) - 256 个候选球心
        segments: (N, 2, 3) - 已生成的 N 条支撑线段
        返回: (M,) - 每个点到整个支撑集的最小距离
        """
        # 提取线段端点
        q1 = segments[:, 0, :] # (N, 3)
        q2 = segments[:, 1, :] # (N, 3)
        v = q2 - q1            # 线段向量 (N, 3)
        
        # 利用广播扩展维度: points(M, 1, 3), q1(1, N, 3)
        # 计算从 q1 指向 points 的向量 w: (M, N, 3)
        w = points.unsqueeze(1) - q1.unsqueeze(0)
        
        # 计算投影参数 t = (w · v) / (v · v)
        # v_sq: (1, N), dot_wv: (M, N)
        v_sq = torch.sum(v * v, dim=-1).unsqueeze(0) + 1e-9
        dot_wv = torch.sum(w * v.unsqueeze(0), dim=-1)
        
        # 限制 t 在 [0, 1] 范围内，得到线段上最近点的参数
        t = torch.clamp(dot_wv / v_sq, 0.0, 1.0)
        
        # 计算点到线段上最近点的距离向量
        # closest_on_segment: (M, N, 3)
        closest = q1.unsqueeze(0) + t.unsqueeze(-1) * v.unsqueeze(0)
        dist_vec = points.unsqueeze(1) - closest
        
        # 计算欧式距离平方并取最小值: (M, N) -> (M,)
        dist_sq = torch.sum(dist_vec**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        
        return torch.sqrt(min_dist_sq)

    def _batch_dist_point_mesh(self, points, zmin, chunk_size=10000):
        """
        优化版：分批处理面片以节省显存 (Chunking)
        """
        M = points.shape[0]
        F = self.v0.shape[0]
        p = points.unsqueeze(1) # (M, 1, 3)
        
        min_dist_sq_global = torch.full((M,), float('inf'), device=self.device)

        # 将面片分成若干批次处理
        for i in range(0, F, chunk_size):
            end = min(i + chunk_size, F)
            v0_c = self.v0[i:end].unsqueeze(0)    # (1, chunk, 3)
            e0_c = self.edge1[i:end].unsqueeze(0)
            e1_c = self.edge2[i:end].unsqueeze(0)

            diff = v0_c - p
            a = torch.sum(e0_c * e0_c, dim=-1)
            b = torch.sum(e0_c * e1_c, dim=-1)
            c = torch.sum(e1_c * e1_c, dim=-1)
            d = torch.sum(e0_c * diff, dim=-1)
            e = torch.sum(e1_c * diff, dim=-1)

            det = a * c - b * b
            s = b * e - c * d
            t = b * d - a * e

            dist_v1v0 = self._dist_p_seg_batch(p, v0_c, v0_c + e0_c)
            dist_v2v0 = self._dist_p_seg_batch(p, v0_c, v0_c + e1_c)
            dist_v1v2 = self._dist_p_seg_batch(p, v0_c + e0_c, v0_c + e1_c)
            
            inv_det = 1.0 / (det + 1e-12)
            s *= inv_det
            t *= inv_det
            mask_inside = (s >= 0) & (t >= 0) & (s + t <= 1)
            
            normal = torch.cross(e0_c, e1_c, dim=-1)
            unit_normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-12)
            dist_plane = torch.abs(torch.sum(diff * unit_normal, dim=-1))
            
            dist_edges = torch.min(torch.min(dist_v1v0, dist_v2v0), dist_v1v2)
            res_sq = torch.where(mask_inside, dist_plane**2, dist_edges**2)
            
            # 更新当前批次的最小值
            min_dist_sq_chunk, _ = torch.min(res_sq, dim=1)
            min_dist_sq_global = torch.min(min_dist_sq_global, min_dist_sq_chunk)

        distToZmin = points[:, 2] - zmin
        min_dist_sq_global = torch.min(min_dist_sq_global, distToZmin**2)
        
        return torch.sqrt(torch.clamp(min_dist_sq_global, min=1e-12))

    def _dist_p_seg_batch(self, p, a, b):
        """辅助函数：批量计算点 p (M,1,3) 到线段 ab (1,F,3) 的距离平方"""
        ab = b - a
        ap = p - a
        # 计算投影比例 t
        t = torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-12)
        t = torch.clamp(t, 0.0, 1.0)
        # 最近点
        closest = a + t.unsqueeze(-1) * ab
        return torch.norm(p - closest, dim=-1)

    # def _batch_dist_segment_env(self, p1s, p2s, radius, r_supp, segments, aabb, num_samples=20):
    #     """
    #     批量计算线段与环境的碰撞状态
    #     p1s, p2s: (256, 3) 线段端点
    #     radius: 刀具组件半径 (r_tip 或 r_holder)
    #     segments: (N, 2, 3) 已存支撑线段集
    #     aabb: [xmin, xmax, ymin, ymax, zmin]
    #     num_samples: 每条线段上的采样点数，越多越精确 (建议 5-10)
    #     """
    #     # 1. 沿线段进行等距采样 (生成圆柱体的包络点)
    #     # t: (num_samples,) -> [0, 1]
    #     t = torch.linspace(0, 1, num_samples, device=self.device)
        
    #     # 利用广播生成采样点矩阵
    #     # p1s.unsqueeze(1): (256, 1, 3)
    #     # v.unsqueeze(1): (256, 1, 3)
    #     # t.view(1, -1, 1): (1, num_samples, 1)
    #     v = p2s - p1s
    #     samples = p1s.unsqueeze(1) + v.unsqueeze(1) * t.view(1, -1, 1) # (256, num_samples, 3)
        
    #     # 展开为扁平的点集进行批量计算: (256 * num_samples, 3)
    #     flat_samples = samples.view(-1, 3)

    #     # 2. 安全空间屏蔽逻辑 (XY 边界过滤)
    #     # 只有 XY 在 AABB 内的点才需要检测碰撞
    #     # z < aabb[4] 且 XY 在外部的点会被此逻辑自动过滤，视为安全
    #     mask_in_aabb = (flat_samples[:, 0] >= aabb[0]) & (flat_samples[:, 0] <= aabb[1]) & \
    #                 (flat_samples[:, 1] >= aabb[2]) & (flat_samples[:, 1] <= aabb[3]) & \
    #                     (flat_samples[:, 2] <= aabb[5])

    #     # 初始化碰撞结果 (256 * num_samples,)
    #     is_point_collided = torch.zeros(flat_samples.shape[0], dtype=torch.bool, device=self.device)

    #     # 3. 提取需要检测的子点集 (优化计算量)
    #     active_indices = torch.where(mask_in_aabb)[0]
    #     if active_indices.numel() > 0:
    #         active_samples = flat_samples[active_indices]

    #         # A. 检测与 Mesh 的距离
    #         dist_mesh = self._batch_dist_point_mesh(active_samples, aabb[4])
    #         hit_mesh = dist_mesh < radius

    #         # B. 检测与已存支撑的距离
    #         if segments.shape[0] > 0:
    #             dist_supp = self._batch_dist_point_segments(active_samples, segments)
    #             hit_supp = dist_supp < (radius + r_supp)
    #         else:
    #             hit_supp = torch.zeros_like(hit_mesh)

    #         # 记录子集碰撞状态
    #         is_point_collided[active_indices] = hit_mesh | hit_supp

    #     # 4. 归约：只要线段上有一个采样点碰撞，该姿态即为碰撞
    #     # reshape 回 (256, num_samples) 然后在 dim=1 取 any
    #     is_pose_collided = is_point_collided.view(p1s.shape[0], -1).any(dim=1)

    #     return is_pose_collided

    def _batch_dist_segment_env(self, p1s, p2s, radius, r_supp, segments, aabb, num_samples=20, noSupport=False):
        """
        改为圆盘驱动的段检测
        """
        # 1. 沿线段采样
        t = torch.linspace(0, 1, num_samples, device=self.device)
        v = p2s - p1s # 姿态向量
        norm_v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-9)
        
        # samples: (256, num_samples, 3)
        samples = p1s.unsqueeze(1) + v.unsqueeze(1) * t.view(1, -1, 1)
        
        # 展平采样点和对应的轴向
        flat_samples = samples.view(-1, 3)
        flat_axes = norm_v.unsqueeze(1).repeat(1, num_samples, 1).view(-1, 3)

        # 2. 批量调用圆盘检测
        is_point_collided = self._batch_dist_disk_mesh(flat_samples, flat_axes, radius, aabb[4])
        
        # 3. 支撑检测 (保持线段距离逻辑，因为支撑也是圆柱体)
        if segments.shape[0] > 0 and not noSupport:
            dist_supp = self._batch_dist_point_segments(flat_samples, segments)
            is_point_collided |= (dist_supp < (radius + r_supp))

        return is_point_collided.view(p1s.shape[0], -1).any(dim=1)
    

    def _batch_dist_disk_mesh(self, centers, axes, radius, z_min, chunk_size_f=5000, chunk_size_m=100):
        """
        全向量化圆盘-Mesh 碰撞检测
        """
        M = centers.shape[0]
        F = self.v0.shape[0]
        hit_global = torch.zeros(M, dtype=torch.bool, device=self.device)

        # 1. 考虑倾斜的地面检测
        # axes[:, 2] 是 cos(theta)，地面下降量是 R * sin(theta)
        z_drop = radius * torch.sqrt(torch.clamp(1 - axes[:, 2]**2, min=0))
        hit_global |= ((centers[:, 2] - z_drop) < z_min)

        # 2. 双重分批计算
        for m_s in range(0, M, chunk_size_m):
            m_e = min(m_s + chunk_size_m, M)
            batch_P = centers[m_s:m_e].unsqueeze(1) # (batch_M, 1, 3)
            batch_V = axes[m_s:m_e].unsqueeze(1)   # (batch_M, 1, 3)
            m_hits = torch.zeros(m_e - m_s, dtype=torch.bool, device=self.device)

            for f_s in range(0, F, chunk_size_f):
                f_e = min(f_s + chunk_size_f, F)
                
                # 提取预存的顶点和常量
                v0 = self.v0[f_s:f_e].unsqueeze(0)
                v1 = self.v1[f_s:f_e].unsqueeze(0)
                v2 = self.v2[f_s:f_e].unsqueeze(0)
                
                # 计算跨越平面掩码
                dA = torch.sum((v0 - batch_P) * batch_V, dim=-1)
                dB = torch.sum((v1 - batch_P) * batch_V, dim=-1)
                dC = torch.sum((v2 - batch_P) * batch_V, dim=-1)
                intersect_mask = ~((dA > 0) & (dB > 0) & (dC > 0) | (dA < 0) & (dB < 0) & (dC < 0))
                
                if not intersect_mask.any(): continue

                # 调用优化的距离计算
                dist_sq = self._point_to_tri_dist_sq_fast(batch_P, f_s, f_e)
                
                collision = intersect_mask & (dist_sq <= radius**2)
                m_hits |= collision.any(dim=1)
                
                if m_hits.all(): break

            hit_global[m_s:m_e] = m_hits

        return hit_global

    def _point_to_tri_dist_sq_fast(self, P, f_s, f_e):
        """
        利用预计算常量加速的点-三角形距离计算
        """
        B = self.v0[f_s:f_e].unsqueeze(0)
        E0 = self.edge1[f_s:f_e].unsqueeze(0)
        E1 = self.edge2[f_s:f_e].unsqueeze(0)
        D = B - P
        
        a = self.face_a[f_s:f_e].unsqueeze(0)
        b = self.face_b[f_s:f_e].unsqueeze(0)
        c = self.face_c[f_s:f_e].unsqueeze(0)
        det = self.face_det[f_s:f_e].unsqueeze(0)
        
        d = torch.sum(E0 * D, dim=-1)
        e = torch.sum(E1 * D, dim=-1)
        
        s = b * e - c * d
        t = b * d - a * e
        
        # 参数约束逻辑
        s = torch.clamp(s / (det + 1e-8), 0, 1)
        t = torch.clamp(t / (det + 1e-8), 0, 1)
        
        # 修正 s+t > 1 的情况
        over = torch.clamp((s + t) - 1, min=0)
        s = torch.clamp(s - over * 0.5, min=0)
        t = torch.clamp(t - over * 0.5, min=0)
        
        closest = B + s.unsqueeze(-1) * E0 + t.unsqueeze(-1) * E1
        return torch.sum((closest - P)**2, dim=-1)
