import numpy as np
import torch

class SupportGenerator:
    def __init__(self, detector, seeds, r_supp, step_len, logical_aabb, aabb):
        """
        logical_aabb: [xmin, xmax, ymin, ymax, zmin] (归一化空间下的逻辑边界)
        """
        self.detector = detector
        self.seeds = seeds
        self.r_supp = r_supp
        self.step_len = step_len
        self.logical_aabb = logical_aabb
        self.aabb = aabb
        self.finished_paths = []
        self.segment_buffer_tensor = torch.empty((0, 2, 3), device='cuda', dtype=torch.float32)
        self.max_fail_limit = 10
        # Minimum number of steps from seed in normal growth mode before allowing stop-on-hit
        self.min_steps_from_seed = 3

    def sort_seeds_by_center(self):
        """
        向心排序
        """
        center = np.array([(self.logical_aabb[0] + self.logical_aabb[1]) / 2, (self.logical_aabb[2] + self.logical_aabb[3]) / 2])
        dists = np.linalg.norm(self.seeds[:, :2] - center, axis=1)
        self.seeds = self.seeds[np.argsort(dists)]

    def _sample_cone_direction(self, angle_deg=45):
        """
        在正下方 [0, 0, -1] 附近的圆锥内随机采样一个步进方向
        """
        phi = np.random.uniform(0, 2 * np.pi)
        # 在指定的半张角范围内采样 cos_theta
        cos_theta = np.random.uniform(np.cos(np.radians(angle_deg)), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        
        # 生成向下生长的单位向量
        return np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            -cos_theta
        ])

    def _clip_to_boundary(self, p_curr, p_cand):
        """
        使用参数化裁剪算法将 p_cand 截断在逻辑 AABB 内。
        逻辑 AABB 格式: [xmin, xmax, ymin, ymax, zmin]
        """
        # 1. 初始化位移向量和比例 t
        v = p_cand - p_curr
        t_min = 1.0  # 默认不截断
        eps = 1e-9

        # 定义边界索引映射，方便循环或清晰展示
        # 0:xmin, 1:xmax, 2:ymin, 3:ymax, 4:zmin
        bounds = self.logical_aabb

        # 2. 检查各轴边界 (X, Y, Z)
        for i in range(3):
            if abs(v[i]) < eps:
                continue
                
            # 确定当前维度的最小值和最大值边界
            if i == 2: # Z 轴只有下界 zmin
                b_min, b_max = bounds[4], float('inf') 
            else: # X, Y 轴有双边界
                b_min, b_max = bounds[i*2], bounds[i*2 + 1]

            # 如果越过下界
            if p_cand[i] < b_min:
                t_i = (b_min - p_curr[i]) / v[i]
                t_min = min(t_min, max(0.0, t_i))
            # 如果越过上界 (仅限 X 和 Y)
            elif p_cand[i] > b_max:
                t_i = (b_max - p_curr[i]) / v[i]
                t_min = min(t_min, max(0.0, t_i))

        # 3. 计算初步交点
        p_int = p_curr + t_min * v

        # 4. 强制物理约束与浮点修正
        # 将 XY 严格限制在 AABB 内
        p_int[0] = np.clip(p_int[0], bounds[0], bounds[1])
        p_int[1] = np.clip(p_int[1], bounds[2], bounds[3])
        
        # Z 轴处理：既不能低于 zmin，通常也不希望在碰撞截断时向上漂移
        p_int[2] = max(p_int[2], bounds[4]) 
        p_int[2] = min(p_int[2], p_curr[2]) # 保持不向上漂移的原始逻辑

        return p_int

    def update_buffer(self, path):
        """
        修正版：直接在 GPU 上处理路径点，修复 TypeError 并提升性能
        """
        # 1. 提取路径中的所有点坐标
        points_raw = [p[0] for p in path]
        if len(points_raw) < 2:
            return

        # 2. 将所有点统一转换为 GPU 上的浮点 Tensor
        # 如果 p 是 numpy 则转换，如果是 tensor 则移动到对应设备
        points_t = []
        for p in points_raw:
            if not torch.is_tensor(p):
                p = torch.tensor(p, dtype=torch.float32, device=self.detector.device)
            else:
                p = p.to(dtype=torch.float32, device=self.detector.device)
            points_t.append(p)
        
        # 3. 构造线段张量 (N-1, 2, 3)
        pts_tensor = torch.stack(points_t) # 形状 (N, 3)
        p_starts = pts_tensor[:-1]         # 起点集 (N-1, 3)
        p_ends = pts_tensor[1:]           # 终点集 (N-1, 3)
        
        # 在 dim=1 堆叠，形成 (N-1, 2, 3)
        new_segments = torch.stack([p_starts, p_ends], dim=1)
        
        # 4. 合并到全局缓冲区
        self.segment_buffer_tensor = torch.cat([self.segment_buffer_tensor, new_segments], dim=0)


    def run(self):
        self.sort_seeds_by_center()
        for seed in self.seeds:
            path = self.grow_rrt(seed)
            if path:
                self.finished_paths.append(path)
                self.update_buffer(path)

    # def grow_one_path(self, seed):
    #     """
    #     单根支撑的 RRT 生长：包含边界截断和垂直降落逻辑
    #     """
    #     # 存储元组: (坐标点, 安全姿态)
    #     path = [(seed, None)] 
    #     curr_p = seed
    #     in_vertical_mode = False  # 标记是否已撞击边界进入垂直下降模式

    #     for _ in range(500): # 最大步进次数
    #         # --- 1. 确定步进方向 ---
    #         if not in_vertical_mode:
    #             direction = self._sample_cone_direction(angle_deg=45)
    #         else:
    #             direction = np.array([0, 0, -1]) # 边界点强制竖直向下

    #         # --- 2. 计算候选点 p_cand ---
    #         p_cand = curr_p + direction * self.step_len

    #         # --- 3. 逻辑 AABB 边界处理 ---
    #         if not in_vertical_mode:
    #             is_out_x = (p_cand[0] < self.logical_aabb[0] or p_cand[0] > self.logical_aabb[1])
    #             is_out_y = (p_cand[1] < self.logical_aabb[2] or p_cand[1] > self.logical_aabb[3])
                
    #             if is_out_x or is_out_y:
    #                 # 撞击侧墙：截断点位置并切换为垂直模式
    #                 p_cand = self._clip_to_boundary(curr_p, p_cand)
    #                 in_vertical_mode = True

    #         # --- 4. 判定“碰撞即停止” (Hit-as-Stop) ---
    #         # 这里的碰撞包括：Mesh、已有支撑线段、Z=0地面
    #         is_hit, hit_p = self.detector.check_hit_as_stop(curr_p, p_cand, self.r_supp, self.segment_buffer)
            
    #         if is_hit:
    #             # 碰到目标，生长成功。停止点不需要 SM 检测
    #             path.append((hit_p, None))
    #             return path 
            
    #         # --- 5. 中间点 SM 检测 ---
    #         # 检测在 p_cand 处，工具是否能以某个姿态避障
    #         accessible, v_safe = self.detector.is_sm_accessible(p_cand, self.segment_buffer, self.logical_aabb)
            
    #         if accessible:
    #             path.append((p_cand, v_safe))
    #             curr_p = p_cand
    #         else:
    #             # 如果此路不通（SM进不去）
    #             if in_vertical_mode:
    #                 # 垂直模式下撞墙则该路径宣告失败
    #                 return None
    #             else:
    #                 # 正常生长模式下尝试换个方向随机采样
    #                 continue 
                    
    #     return None
    
    def grow_rrt(self, seed, angle_deg=45):
        """
        带有侧墙截断逻辑和垂直下降模式的 RRT
        """

        # 节点结构：pos, v_safe, parent, is_dead
        tree = [{
            'pos': np.array(seed, dtype=float),
            'v_safe': None,
            'parent': -1,
            'fail_count': 0,
            'is_dead': False
        }]

        max_iter = 1000
        
        for _ in range(max_iter):
            # 1. 采样
            active_indices = [i for i, n in enumerate(tree) if not n['is_dead']]
            if not active_indices: break
            near_idx = np.random.choice(active_indices)
            q_near = tree[near_idx]['pos']
            direction = self._sample_cone_direction(angle_deg)
            q_cand = q_near + direction * self.step_len

            # --- 核心逻辑 A: 逻辑 AABB 边界处理 ---
            is_out_x = (q_cand[0] < self.logical_aabb[0] or q_cand[0] > self.logical_aabb[1])
            is_out_y = (q_cand[1] < self.logical_aabb[2] or q_cand[1] > self.logical_aabb[3])

            if is_out_x or is_out_y:
                # 撞击侧墙：截断并立即切换为垂直探测模式
                p_clip = self._clip_to_boundary(q_near, q_cand)
                
                # 从截断点直接向 logical_aabb 的最低平面探测（垂直下降）
                # 目标点只需要明显低于该平面，让 detector 在 floor_z 处截停
                p_ground_target = np.array([p_clip[0], p_clip[1], self.logical_aabb[4] - 1.0])
                is_hit, hit_p = self.detector.check_hit_as_stop(p_clip, p_ground_target, self.r_supp, self.segment_buffer_tensor)
                
                if is_hit:
                    # 将截断点和最终落点作为路径的末尾
                    # 先把 p_clip 加入树（可选，为了提取路径方便）
                    clip_idx = len(tree)
                    tree.append({'pos': p_clip, 'v_safe': None, 'parent': near_idx, 'is_dead': False})
                    # 最后的命中点
                    tree.append({'pos': hit_p, 'v_safe': None, 'parent': clip_idx, 'is_dead': False})
                    return self._extract_path(tree)

            # --- 核心逻辑 B: 判定“碰撞即停止” (正常生长模式) ---
            is_hit, hit_p = self.detector.check_hit_as_stop(q_near, q_cand, self.r_supp, self.segment_buffer_tensor)
            if is_hit:
                # Enforce a minimum number of steps from seed in normal growth mode
                steps_from_seed = self._compute_depth(tree, near_idx)
                if steps_from_seed < self.min_steps_from_seed:
                    # Too short to terminate here; penalize this node and continue growing
                    continue

                tree.append({
                    'pos': hit_p,
                    'v_safe': None,
                    'parent': near_idx,
                    'is_dead': False
                })
                return self._extract_path(tree) 

            # 4. 中间点 SM 检测
            accessible, v_safe = self.detector.is_sm_accessible(q_cand, self.segment_buffer_tensor, self.aabb, self.r_supp)
            if accessible:
                tree.append({
                    'pos': q_cand,
                    'v_safe': v_safe,
                    'parent': near_idx,
                    'fail_count': 0,
                    'is_dead': False
                })
            else:
                self._mark_fail(tree, near_idx)

        return None
    
    def _mark_fail(self, tree, idx):
        tree[idx]['fail_count'] += 1
        if tree[idx]['fail_count'] >= self.max_fail_limit:
            tree[idx]['is_dead'] = True

    def _compute_depth(self, tree, idx):
        """
        Compute the number of nodes from the seed (root) to the given node index, inclusive.
        This corresponds to the number of segments from seed to the next step when expanding
        from this node in normal growth mode.
        """
        depth = 0
        while idx != -1:
            depth += 1
            idx = tree[idx]['parent']
        return depth

    def _extract_path(self, tree):
        path = []
        curr_idx = len(tree) - 1
        while curr_idx != -1:
            node = tree[curr_idx]
            path.append((node['pos'], node['v_safe']))
            curr_idx = node['parent']
        return path[::-1]
