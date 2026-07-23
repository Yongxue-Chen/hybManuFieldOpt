import os
import torch
import trimesh
import numpy as np
from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
from torch.utils.data import IterableDataset


def _repair_trimesh_component(mesh):
    mesh = mesh.copy()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process()
    mesh.merge_vertices()
    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()
    return mesh


def _prepare_mesh_for_voxelization(mesh):
    """
    Treat a multi-component closed mesh as valid for voxelization.
    Only attempt repairs on components that are genuinely non-watertight.
    """
    if mesh.is_watertight:
        return mesh, True, 1, 0

    components = list(mesh.split(only_watertight=False))
    if not components:
        return mesh, False, 0, 0

    repaired_components = []
    non_watertight_count = 0

    for component in components:
        repaired = _repair_trimesh_component(component)
        repaired_components.append(repaired)
        non_watertight_count += int(not repaired.is_watertight)

    prepared_mesh = trimesh.util.concatenate(repaired_components)
    effective_watertight = (non_watertight_count == 0)
    return prepared_mesh, effective_watertight, len(repaired_components), non_watertight_count

class StreamedGPUTrainingDataGenerator:
    def __init__(self, aabb_min, aabb_max, batch_size, device, MANU_CONFIG, MAX_RES_STRUCTURE, sample_each_grid):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError("StreamedGPUTrainingDataGenerator requires a CUDA device.")

        self.aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32, device=self.device)
        self.aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32, device=self.device)
        self.batch_size = int(batch_size)

        # Manu configuration
        self.nSupport = int(MANU_CONFIG['nSupport'])
        self.hSupport = torch.tensor(MANU_CONFIG['hSupport'], dtype=torch.float32, device=self.device)
        self.supportR = self.hSupport / torch.tan(torch.tensor(MANU_CONFIG['supportAngle'], dtype=torch.float32, device=self.device))

        self.nAMCone = int(MANU_CONFIG['nAMCone'])
        self.nAMCylinder = int(MANU_CONFIG['nAMCylinder'])
        self.AMConeHeight = torch.tensor(MANU_CONFIG['AMConeHeight'], dtype=torch.float32, device=self.device)
        self.AMConeHalfAngle = torch.tensor(MANU_CONFIG['AMConeHalfAngle'], dtype=torch.float32, device=self.device)
        self.AMConeTanAngle = torch.tan(self.AMConeHalfAngle)
        self.AMCollisionMargin = torch.tensor(MANU_CONFIG['AMCollisionMargin'], dtype=torch.float32, device=self.device)

        self.SMTipDiameter = torch.tensor(MANU_CONFIG['SMToolParas']['SMTipDiameter'], dtype=torch.float32, device=self.device)
        self.SMShankDiameter = torch.tensor(MANU_CONFIG['SMToolParas']['SMShankDiameter'], dtype=torch.float32, device=self.device)
        self.SMHolderDiameter = torch.tensor(MANU_CONFIG['SMToolParas']['SMHolderDiameter'], dtype=torch.float32, device=self.device)
        self.SMToolLength = torch.tensor(MANU_CONFIG['SMToolParas']['SMToolLength'], dtype=torch.float32, device=self.device)
        self.SMHolderLength = torch.tensor(MANU_CONFIG['SMToolParas']['SMHolderLength'], dtype=torch.float32, device=self.device)
        self.SMCollisionMargin = torch.tensor(MANU_CONFIG['SMToolParas']['SMCollisionMargin'], dtype=torch.float32, device=self.device)
        self.nSMTip = int(MANU_CONFIG['SMToolParas']['nSMTip'])
        self.nSMTool = int(MANU_CONFIG['SMToolParas']['nSMTool'])
        self.nSMHolder = int(MANU_CONFIG['SMToolParas']['nSMHolder'])

        max_length = max(self.aabb_max - self.aabb_min)
        self.DX = max_length / MAX_RES_STRUCTURE
        self.RES_X = round(((self.aabb_max[0] - self.aabb_min[0]) / self.DX).item())
        self.RES_Y = round(((self.aabb_max[1] - self.aabb_min[1]) / self.DX).item())
        self.RES_Z = round(((self.aabb_max[2] - self.aabb_min[2]) / self.DX).item())
        self.sample_each_grid = sample_each_grid

        # Create CUDA streams on the target device
        self.stream1 = torch.cuda.Stream(device=self.device)
        self.stream2 = torch.cuda.Stream(device=self.device)

        self.preload()
    
    @torch.no_grad()
    def sample_grid_points(self):
        indices_x = torch.arange(self.RES_X, device=self.device, dtype=torch.float32)
        indices_y = torch.arange(self.RES_Y, device=self.device, dtype=torch.float32)
        indices_z = torch.arange(self.RES_Z, device=self.device, dtype=torch.float32)
        grid_base_x = indices_x * self.DX
        grid_base_y = indices_y * self.DX
        grid_base_z = indices_z * self.DX
        gx, gy, gz = torch.meshgrid(grid_base_x, grid_base_y, grid_base_z, indexing='ij')
        grid_points_base = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        grid_points_base = grid_points_base.repeat(self.sample_each_grid, 1)
        jitter = torch.rand(grid_points_base.shape, device=self.device, dtype=torch.float32) * self.DX
        grid_points = grid_points_base + jitter + self.aabb_min
        return grid_points

    @torch.no_grad()
    def sample_uniform_points(self, max_resolution):
        # Recompute grid spacing and resolution based on desired max_resolution
        max_length = max(self.aabb_max - self.aabb_min)
        self.DX = max_length / max_resolution
        self.RES_X = round(((self.aabb_max[0] - self.aabb_min[0]) / self.DX).item())
        self.RES_Y = round(((self.aabb_max[1] - self.aabb_min[1]) / self.DX).item())
        self.RES_Z = round(((self.aabb_max[2] - self.aabb_min[2]) / self.DX).item())

        # Build a uniform 3D grid inside the AABB (grid cell centers, no jitter)
        indices_x = torch.arange(self.RES_X, device=self.device, dtype=torch.float32)
        indices_y = torch.arange(self.RES_Y, device=self.device, dtype=torch.float32)
        indices_z = torch.arange(self.RES_Z, device=self.device, dtype=torch.float32)

        # Use cell centers: (i + 0.5) * DX, then shift by aabb_min
        grid_base_x = (indices_x + 0.5) * self.DX
        grid_base_y = (indices_y + 0.5) * self.DX
        grid_base_z = (indices_z + 0.5) * self.DX

        gx, gy, gz = torch.meshgrid(grid_base_x, grid_base_y, grid_base_z, indexing='ij')
        grid_points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        grid_points = grid_points + self.aabb_min  # shift into AABB

        return grid_points

    @torch.no_grad()
    def _generate_batch(self, points=None):
        # 1) Sample base points
        if points is None:
            rnd = torch.rand(self.batch_size, 3, device=self.device)
            points = (self.aabb_max - self.aabb_min) * rnd + self.aabb_min

        # 2) Generate support points as a flat tensor of shape (B*nS, 3)
        B = points.shape[0]
        nS = self.nSupport

        # Default empty supports if nS == 0
        if nS <= 0:
            NSupport = 0
            supports = torch.empty((0,3), device=self.device, dtype=points.dtype)
        else:
            # Sample offsets uniformly in disk of radius supportR in XY
            # r ~ sqrt(U) * R, theta ~ U[0, 2pi)
            NSupport = B * nS
            U = torch.rand(NSupport, device=self.device, dtype=points.dtype)
            r = torch.sqrt(U) * self.supportR
            theta = 2.0 * torch.pi * torch.rand(NSupport, device=self.device, dtype=points.dtype)
            dx = r * torch.cos(theta)
            dy = r * torch.sin(theta)

            # Compose supports by repeating base points and applying offsets; z lowered by h
            supports = points.repeat_interleave(nS, dim=0)  # (B*nS, 3)
            supports[:, 0] = supports[:, 0] + dx
            supports[:, 1] = supports[:, 1] + dy
            supports[:, 2] = supports[:, 2] - self.hSupport
            # note that supports may be outside the aabb, so we will add a mask later

        # 3) Generate AMTool points
        nCone = self.nAMCone
        nCyl  = self.nAMCylinder
        hCone = self.AMConeHeight
        Rmax  = hCone * self.AMConeTanAngle

        extras = [supports]

        am_count = nCone + nCyl
        am_grouped = torch.empty((B, am_count, 3), device=self.device, dtype=points.dtype)

        base = points.unsqueeze(1) # (B,1,3) for broadcasting

        # Cone: apex at base point, axis +z, height hCone, half-angle
        if nCone > 0:
            # Uniform in cone volume: t in [0,1], t = U^(1/3)
            # t_cone = torch.rand(B, nCone, device=self.device, dtype=points.dtype) ** (1.0 / 3.0)
            # r_max_cone = t_cone * Rmax
            # theta_cone = 2.0 * torch.pi * torch.rand(B, nCone, device=self.device, dtype=points.dtype)
            # r_cone = torch.sqrt(torch.rand(B, nCone, device=self.device, dtype=points.dtype)) * r_max_cone

            # dx_cone = r_cone * torch.cos(theta_cone)
            # dy_cone = r_cone * torch.sin(theta_cone)
            # dz_cone = t_cone * hCone

            h_min = self.AMCollisionMargin
            t_min = h_min / hCone
            power = 2.0

            # 在 [t_min, 1.0] 范围内采样高度 t
            t_raw = torch.rand(B, nCone, device=self.device, dtype=points.dtype)
            t_cone = t_min + (1.0 - t_min) * (t_raw ** power) 
            r_max_at_t = t_cone * Rmax 
            theta_cone = 2.0 * torch.pi * torch.rand(B, nCone, device=self.device, dtype=points.dtype)
            r_cone = torch.sqrt(torch.rand(B, nCone, device=self.device, dtype=points.dtype)) * r_max_at_t
            dx_cone = r_cone * torch.cos(theta_cone)
            dy_cone = r_cone * torch.sin(theta_cone)
            dz_cone = t_cone * hCone

            am_grouped[:,:nCone,0] = base[...,0] + dx_cone
            am_grouped[:,:nCone,1] = base[...,1] + dy_cone
            am_grouped[:,:nCone,2] = base[...,2] + dz_cone
            # note that am_cone may be outside the aabb, so we will add a mask later

        # Cylinder above cone: radius equals cone base Rmax, z in [p.z+hCone, aabb_max.z]
        if nCyl > 0:
            theta_cyl = 2.0 * torch.pi * torch.rand(B, nCyl, device=self.device, dtype=points.dtype)
            r_cyl = torch.sqrt(torch.rand(B, nCyl, device=self.device, dtype=points.dtype)) * Rmax
            dx_cyl = r_cyl * torch.cos(theta_cyl)
            dy_cyl = r_cyl * torch.sin(theta_cyl)

            z_min = base[...,2] + hCone
            z_top = self.aabb_max[2]
            height_cyl = torch.clamp(z_top - z_min, min=0.0)
            uz = torch.rand(B, nCyl, device=self.device, dtype=points.dtype)
            z_cyl = z_min + uz * height_cyl

            # if z_min > z_top, z_cyl will be z_min, which is outside the aabb

            am_grouped[:,nCone:,0] = base[...,0] + dx_cyl
            am_grouped[:,nCone:,1] = base[...,1] + dy_cyl
            am_grouped[:,nCone:,2] = z_cyl

            # note that am_cyl may be outside the aabb, so we will add a mask later

        extras.append(am_grouped.reshape(-1,3))

        # 4) Concatenate all points
        allPoints = torch.cat([points] + extras, dim=0)

        # add mask for points inside the aabb
        mask_inside = ((allPoints>=self.aabb_min) & (allPoints<=self.aabb_max)).all(dim=-1)
        mask_inside[:B] = True # base points are always inside the aabb

        points_inside = allPoints[mask_inside]

        toolPoints_raw = sample_SM_Tool(
            self.SMTipDiameter/2.0, self.SMShankDiameter/2.0, self.SMHolderDiameter/2.0,
            self.SMToolLength, self.SMHolderLength,
            self.SMCollisionMargin,
            B,
            [self.nSMTip, self.nSMTool, self.nSMHolder],
            points_inside.device,
            points_inside.dtype
        )

        return points_inside, mask_inside, [B, NSupport, B*am_count], toolPoints_raw

    def preload(self):
        # 将"生成下一批数据"这个任务放入 stream1 队列
        with torch.cuda.stream(self.stream1):
            self.next_batchP, self.next_batchMask, self.next_batchN, self.next_batchToolPoints_raw = self._generate_batch()

    def __iter__(self):
        return self

    def __next__(self):
        # 3. 主循环请求数据时，首先确保后台流上的数据已生成完毕
        #    在大多数情况下，由于训练耗时>>生成耗时，这里的等待时间几乎为0
        self.stream1.synchronize() 
        
        # 4. 立即返回已经预加载好的数据，主循环无需等待
        current_batchP, current_batchMask, current_batchN, current_batchToolPoints_raw = self.next_batchP, self.next_batchMask, self.next_batchN, self.next_batchToolPoints_raw
        
        # 5. 关键并行步骤：
        #    在主循环开始使用 current_batch 进行训练的同时，
        #    我们立刻将"生成下一批数据"的任务放入另一个后台流 (stream2)。
        #    这个任务会和主流的训练任务并行执行。
        with torch.cuda.stream(self.stream2):
            self.next_batchP, self.next_batchMask, self.next_batchN, self.next_batchToolPoints_raw = self._generate_batch()

        # Swap streams
        self.stream1, self.stream2 = self.stream2, self.stream1
        
        return current_batchP, current_batchMask, current_batchN, current_batchToolPoints_raw

# @torch.no_grad()
def sample_SM_Tool(r_tip, r_shank, R, h, H, collision_margin, m, n_per_shape, device, dtype):
    """
    Batched sampling on GPU with volume + surface points.
    - r_tip, r_shank, R, h, H: scalar
    - m: number of base points
    - n_per_shape: three integers indicating the number of points for each part
    - collision_margin: inward shrink applied only to tip hemisphere volume/surface
    Returns:
      pos: (m,N1+N1s+N2+N2s+N3+N3s+N3b,3)
    """
    N1, N2, N3 = n_per_shape[0], n_per_shape[1], n_per_shape[2]
    if isinstance(r_tip, torch.Tensor) or isinstance(collision_margin, torch.Tensor):
        r_t = torch.as_tensor(r_tip, device=device, dtype=dtype)
        cm_t = torch.as_tensor(collision_margin, device=device, dtype=dtype)
        r_sphere = torch.clamp(r_t - cm_t, min=1e-6)
    else:
        r_sphere = max(r_tip - collision_margin, 1e-6)

    # Surface counts follow tool_shape.py default policy.
    N1s = N1 if N1 > 0 else 0
    N2s = N2 if N2 > 0 else 0
    N3b = N3 if N3 > 0 else 0

    def _stratified_01(n):
        if n <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        idx = torch.arange(n, device=device, dtype=dtype)
        return (idx + 0.5) / n

    def _golden_theta(n):
        if n <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        idx = torch.arange(n, device=device, dtype=dtype)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        return idx * golden_angle

    # Canonical sampling for each part, sized exactly to counts
    # 1) Hemisphere (z<=0), centered at origin, radius r
    if N1 > 0:
        dirs = torch.randn(m, N1, 3, device=device, dtype=dtype)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-12)
        flip = (dirs[..., 2:3] > 0).to(dtype)
        dirs = dirs * (1.0 - 2.0 * flip)  # flip if z > 0
        radii = r_sphere * torch.rand(m, N1, 1, device=device, dtype=dtype).pow(1.0 / 3.0)
        pos1 = dirs * radii  # (m, N1, 3)
    else:
        pos1 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 1s) Sphere surface (quasi-uniform), collision margin applies here
    if N1s > 0:
        # Uniform random sampling on lower hemisphere surface (z <= 0)
        z1s = -torch.rand(m, N1s, device=device, dtype=dtype)
        r1s = torch.sqrt(torch.clamp(1.0 - z1s * z1s, min=0.0))
        th1s = 2.0 * torch.pi * torch.rand(m, N1s, device=device, dtype=dtype)
        x1s = r1s * torch.cos(th1s)
        y1s = r1s * torch.sin(th1s)
        pos1s = torch.stack([x1s, y1s, z1s], dim=-1) * r_sphere
    else:
        pos1s = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 2) Small cylinder: radius r_shank, z in [0, h] (no collision margin shrink)
    if N2 > 0:
        theta2 = 2 * torch.pi * torch.rand(m, N2, 1, device=device, dtype=dtype)
        rho2 = r_shank * torch.sqrt(torch.rand(m, N2, 1, device=device, dtype=dtype))
        x2 = rho2 * torch.cos(theta2)
        y2 = rho2 * torch.sin(theta2)
        z2 = h * torch.rand(m, N2, 1, device=device, dtype=dtype)
        pos2 = torch.cat([x2, y2, z2], dim=-1)  # (m, N2, 3)
    else:
        pos2 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 2s) Small cylinder side surface (quasi-uniform), no collision margin
    if N2s > 0:
        # Uniform random sampling on cylinder side surface
        th2s = 2.0 * torch.pi * torch.rand(m, N2s, device=device, dtype=dtype)
        z2s = h * torch.rand(m, N2s, device=device, dtype=dtype)
        x2s = r_shank * torch.cos(th2s)
        y2s = r_shank * torch.sin(th2s)
        pos2s = torch.stack([x2s, y2s, z2s], dim=-1)
    else:
        pos2s = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # 3) Big cylinder: radius R, z in [h, h+H]
    if N3 > 0:
        theta3 = 2 * torch.pi * torch.rand(m, N3, 1, device=device, dtype=dtype)
        rho3 = R * torch.sqrt(torch.rand(m, N3, 1, device=device, dtype=dtype))
        x3 = rho3 * torch.cos(theta3)
        y3 = rho3 * torch.sin(theta3)
        z3 = h + H * (torch.rand(m, N3, 1, device=device, dtype=dtype) ** 3.0)
        nBottom = max(N3 // 2, 1)
        z3[:, :nBottom, :] = h
        pos3 = torch.cat([x3, y3, z3], dim=-1)  # (m, N3, 3)
    else:
        pos3 = torch.empty(m, 0, 3, device=device, dtype=dtype)

    # Connect all parts to (m, N_total, 3)
    pos = torch.cat([pos1, pos1s, pos2, pos2s, pos3], dim=1)
    return pos


class CPUTrainingDataGenerator(IterableDataset):
    """
    高效的CPU数据生成器 - 在后台生成训练坐标点
    DataLoader会自动与GPU训练形成流水线
    """
    def __init__(self, aabb_min, aabb_max, batch_size):
        super().__init__()
        self.aabb_min = torch.tensor(aabb_min, dtype=torch.float32)
        self.aabb_max = torch.tensor(aabb_max, dtype=torch.float32)
        self.batch_size = batch_size
        self.extent = self.aabb_max - self.aabb_min

    def __iter__(self):
        """每个epoch生成指定数量的批次"""
        while True:
            random_vals = torch.rand(self.batch_size, 3, dtype=torch.float32)
            points = self.aabb_min + random_vals * self.extent
            yield points

    def __len__(self):
        return self.batch_size

@torch.jit.script
def nearest_neighbor_query_gpu(voxel_grid, coords_nearest):
    """
    GPU优化的最近邻查询函数 (输出严格的0或1)
    [已修正边界问题]
    
    Args:
        voxel_grid (torch.Tensor): 存放在GPU上的3D体素网格 (int8类型)
        coords_nearest (torch.Tensor): 查询点的坐标 (N, 3) (已经四舍五入到最近的整数索引)
    
    Returns:
        torch.Tensor: 查询结果，严格为0或1 (float32类型)
    """
    # # 将坐标四舍五入到最近的整数索引
    # coords_nearest = torch.round(coords).long()
    
    # 获取网格维度
    grid_shape = torch.tensor(voxel_grid.shape, device=voxel_grid.device)
    
    # --- 新增的边界检查 ---
    # 1. 创建一个布尔张量，标记所有出界的点
    #    检查三个轴中是否有任何一个轴出界
    out_of_bounds = (coords_nearest[:, 0] < 0) | (coords_nearest[:, 0] >= grid_shape[0]) | \
                    (coords_nearest[:, 1] < 0) | (coords_nearest[:, 1] >= grid_shape[1]) | \
                    (coords_nearest[:, 2] < 0) | (coords_nearest[:, 2] >= grid_shape[2])
    
    # --- 原始逻辑，但现在是安全的 ---
    # 2. 对坐标进行clamp，防止索引错误。对于出界的点，这里的索引是临时的，后续会被覆盖
    idx_x = torch.clamp(coords_nearest[:, 0], 0, grid_shape[0] - 1)
    idx_y = torch.clamp(coords_nearest[:, 1], 0, grid_shape[1] - 1)
    idx_z = torch.clamp(coords_nearest[:, 2], 0, grid_shape[2] - 1)
    
    # 3. 从网格中提取值
    results = voxel_grid[idx_x, idx_y, idx_z].to(torch.float32)
    
    # 4. 对于所有被标记为出界的点，将它们的结果强制设为0（外部）
    results[out_of_bounds] = 0.0
    
    return results

def create_pure_pytorch_implicit_function(stl_file, device='cuda', voxel_resolution=256):
    """
    [优化版] 基于体素的GPU加速隐式函数。
    使用高效的体素化方法，并优化内存占用。
    
    Returns:
        function: 一个优化的查询函数，用于点是否在网格内的测试 (输出0或1)
    """
    print('\n' + '='*50)
    print(f"创建体素隐式函数 (分辨率: {voxel_resolution}, 设备: {device})")
    
    # 1. 加载和归一化网格
    scale, p_min_translation, _ = get_normalization_parameters(stl_file)
    mesh = trimesh.load_mesh(stl_file)
    mesh.apply_translation(-p_min_translation)
    mesh.apply_scale(scale)

    # --- 在这里加入诊断代码 ---
    # print("\n--- Trimesh 诊断报告 ---")
    # print(f"模型是否水密 (is_watertight)?       : {mesh.is_watertight}")
    # print(f"模型法线是否一致 (is_orientable)? : {mesh.is_winding_consistent}")
    # print(f"模型体积 (volume):                  : {mesh.volume}")
    # print("------------------------\n")

    mesh, effective_watertight, component_count, non_watertight_count = _prepare_mesh_for_voxelization(mesh)
    if effective_watertight:
        if component_count > 1:
            print(f"检测到 {component_count} 个连通分量，且全部为闭合体，按有效水密网格处理。")
    else:
        print(
            "警告: 网格仍包含非水密分量 "
            f"({non_watertight_count}/{component_count})，体素化结果可能不准确。"
        )
    
    # 2. [优化] 使用trimesh内置的高效体素化引擎
    print(f"正在使用高效引擎进行体素化 (分辨率: {voxel_resolution})...")
    # 使用一个略大于1的pitch来确保边界正确
    pitch = np.max(mesh.bounds[1] - mesh.bounds[0]) / (voxel_resolution - 1)
    voxel_grid_trimesh = mesh.voxelized(pitch=pitch)

    voxel_matrix = voxel_grid_trimesh.matrix.astype(bool)
    # 从边界开始flood fill，标记所有外部体素
    from scipy.ndimage import binary_fill_holes
    filled_matrix = binary_fill_holes(voxel_matrix)
    voxel_grid = filled_matrix.astype(np.int8)
    
    # 使用padding来处理边界查询
    # voxel_grid = voxel_grid_trimesh.matrix.astype(np.int8)
    
    # 将体素网格和相关参数传输到GPU
    # [优化] 使用int8类型，节省75%显存
    voxel_tensor = torch.tensor(voxel_grid, dtype=torch.int8, device=device)
    
    # 计算用于坐标变换的参数
    # voxel_grid_trimesh.transform 包含了从网格坐标到世界坐标的变换
    # 我们需要它的逆变换
    world_to_grid_transform = torch.tensor(np.linalg.inv(voxel_grid_trimesh.transform), dtype=torch.float32, device=device)
    
    print(f"体素网格创建成功。内部体素数量: {torch.sum(voxel_tensor).item():,}")
    print('='*50 + '\n')
    
    def voxel_query_H(points):
        """
        使用最近邻查询来判断点是否在模型内部
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=device)
        else:
            points = points.to(device)

        # 将世界坐标点转换为体素网格的索引坐标
        # 添加一列1以进行齐次坐标变换
        points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=device)], dim=1)
        
        # 应用变换矩阵: (N, 4) @ (4, 4) -> (N, 4)
        # 我们只取前3个坐标 (x, y, z)
        grid_coords = (points_homo @ world_to_grid_transform.T)[:, :3]
        grid_coord_nearest = torch.round(grid_coords).long()
        
        pointsStates = nearest_neighbor_query_gpu(voxel_tensor, grid_coord_nearest)

        return pointsStates.view(-1, 1), grid_coord_nearest

    # Expose tensors for optional serialization.
    voxel_query_H.voxel_tensor = voxel_tensor
    voxel_query_H.world_to_grid_transform = world_to_grid_transform
    voxel_query_H.device = str(device)
    voxel_query_H.voxel_resolution = int(voxel_resolution)
    voxel_query_H.source_stl = os.path.abspath(stl_file)
    
    return voxel_query_H


def save_pure_pytorch_implicit_function(H_gpu: callable, output_path: str) -> str:
    """
    Save a pure-PyTorch voxel implicit function created by
    ``create_pure_pytorch_implicit_function``.

    The saved artifact contains tensors required to rebuild an equivalent
    callable later via ``load_pure_pytorch_implicit_function``.
    """
    required_attrs = ('voxel_tensor', 'world_to_grid_transform')
    missing = [name for name in required_attrs if not hasattr(H_gpu, name)]
    if missing:
        raise ValueError(
            "H_gpu does not contain serialization metadata. "
            "Please create it via create_pure_pytorch_implicit_function first. "
            f"Missing attributes: {missing}"
        )

    payload = {
        'format': 'pure_pytorch_voxel_implicit_v1',
        'voxel_tensor': getattr(H_gpu, 'voxel_tensor').detach().to('cpu', dtype=torch.int8),
        'world_to_grid_transform': getattr(H_gpu, 'world_to_grid_transform').detach().to('cpu', dtype=torch.float32),
        'voxel_resolution': int(getattr(H_gpu, 'voxel_resolution', -1)),
        'source_stl': str(getattr(H_gpu, 'source_stl', '')),
    }

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_pure_pytorch_implicit_function(artifact_path: str, device: str = 'cuda') -> callable:
    """
    Load a saved pure-PyTorch voxel implicit artifact and rebuild a callable
    with the same interface as ``create_pure_pytorch_implicit_function``.
    """
    artifact_path = os.path.abspath(artifact_path)
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(f"Saved H_gpu artifact not found: {artifact_path}")

    payload = torch.load(artifact_path, map_location='cpu', weights_only=False)
    fmt = payload.get('format', '')
    if fmt != 'pure_pytorch_voxel_implicit_v1':
        raise ValueError(
            f"Unsupported H_gpu artifact format: {fmt!r}. "
            "Expected 'pure_pytorch_voxel_implicit_v1'."
        )

    voxel_tensor = payload['voxel_tensor'].to(device=device, dtype=torch.int8)
    world_to_grid_transform = payload['world_to_grid_transform'].to(device=device, dtype=torch.float32)

    def voxel_query_H(points):
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=device)
        else:
            points = points.to(device)

        points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=device)], dim=1)
        grid_coords = (points_homo @ world_to_grid_transform.T)[:, :3]
        grid_coord_nearest = torch.round(grid_coords).long()
        pointsStates = nearest_neighbor_query_gpu(voxel_tensor, grid_coord_nearest)
        return pointsStates.view(-1, 1), grid_coord_nearest

    voxel_query_H.voxel_tensor = voxel_tensor
    voxel_query_H.world_to_grid_transform = world_to_grid_transform
    voxel_query_H.device = str(device)
    voxel_query_H.voxel_resolution = int(payload.get('voxel_resolution', -1))
    voxel_query_H.source_stl = str(payload.get('source_stl', ''))

    return voxel_query_H


def get_sdf_aabb(stl_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the AABB stored in a pre-trained SDF checkpoint without loading the
    full network weights.

    Returns ``(aabb_min, aabb_max)`` as float32 numpy arrays in normalised mesh
    space — the same coordinate system used by ``load_sdf_implicit_function``.

    Args:
        stl_file: Path to the STL file (used only to derive the checkpoint path).

    Raises:
        FileNotFoundError: If the SDF checkpoint does not exist yet.
    """
    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    ckpt_path = os.path.abspath(checkpoint_path) if checkpoint_path else os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"SDF checkpoint not found: {ckpt_path}\n"
            f"Please train it first:\n"
            f"  conda run -n myenv python fieldopt/geometry/sdf/train_sdf.py --stl {stl_file}"
        )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    aabb_min = np.asarray(ckpt['aabb_min'], dtype=np.float32)
    aabb_max = np.asarray(ckpt['aabb_max'], dtype=np.float32)
    return aabb_min, aabb_max


def load_sdf_implicit_function(stl_file: str, device: str = 'cuda', checkpoint_path: str | None = None) -> callable:
    """
    Load a pre-trained neural SDF (produced by fieldopt/geometry/sdf/train_sdf.py)
    and return a query callable with the same interface as
    ``create_pure_pytorch_implicit_function``.

    Both current SDFNet checkpoints and legacy ``siren-pytorch`` checkpoints
    containing ``model_state_dict`` are supported.

    Run the training script first if the checkpoint is missing:

        conda run -n fieldopt_hm_rtx5090 python -m fieldopt.geometry.sdf.train_sdf --stl <path>

    Args:
        stl_file: Path to the STL file (used to derive the checkpoint path when
                  ``checkpoint_path`` is not provided).
        device:   PyTorch device string, e.g. ``'cuda'`` or ``'cpu'``.
        checkpoint_path: Optional explicit neural-SDF checkpoint path.

    Returns:
        sdf_query(points) -> (binary_states, sdf_values)
            binary_states : ``torch.Tensor`` of shape ``(N, 1)``, dtype float32.
                            1.0 = inside the model, 0.0 = outside.
            sdf_values    : ``torch.Tensor`` of shape ``(N, 1)``, dtype float32.
                            Raw SDF output (negative inside, positive outside).
    """
    from .sdf.train_sdf import SDFNet, normalise_pts  # lazy import

    # ── Derive checkpoint path ────────────────────────────────────────────────
    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    ckpt_path = os.path.abspath(checkpoint_path) if checkpoint_path else os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"SDF checkpoint not found: {ckpt_path}\n"
            f"Please train it first:\n"
            "  conda run -n fieldopt_hm_rtx5090 python "
            f"-m fieldopt.geometry.sdf.train_sdf --stl {stl_file}"
        )

    print('\n' + '=' * 50)
    print(f"Loading neural SDF from: {ckpt_path}  (device: {device})")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        net = SDFNet(
            fourier_L=ckpt['fourier_L'],
            hidden_dim=ckpt['hidden_dim'],
            n_hidden=ckpt['n_hidden'],
        ).to(device)
        net.load_state_dict(ckpt['state_dict'])

        aabb_min_t = torch.tensor(ckpt['aabb_min'], dtype=torch.float32, device=device)
        aabb_max_t = torch.tensor(ckpt['aabb_max'], dtype=torch.float32, device=device)

        def normalise_query_points(points: torch.Tensor) -> torch.Tensor:
            return normalise_pts(points, aabb_min_t, aabb_max_t)

        format_description = (
            f"SDFNet (L={ckpt['fourier_L']}, "
            f"hidden={ckpt['hidden_dim']}×{ckpt['n_hidden']})"
        )
    elif 'model_state_dict' in ckpt:
        from siren_pytorch import SirenNet

        state_dict = ckpt['model_state_dict']
        layer_indices = {
            int(key.split('.')[1])
            for key in state_dict
            if key.startswith('layers.') and key.endswith('.weight')
        }
        first_weight = state_dict['layers.0.weight']
        last_weight = state_dict['last_layer.weight']
        net = SirenNet(
            int(first_weight.shape[1]),
            int(first_weight.shape[0]),
            int(last_weight.shape[0]),
            len(layer_indices),
            w0=30,
            w0_initial=30,
        ).to(device)
        net.load_state_dict(state_dict)

        mesh = trimesh.load_mesh(abs_stl, force='mesh')
        bounds = torch.tensor(mesh.bounds, dtype=torch.float32, device=device)
        centre = 0.5 * (bounds[0] + bounds[1])
        half_longest_edge = 0.5 * (bounds[1] - bounds[0]).max()
        if half_longest_edge <= 0:
            raise ValueError(f"STL has a degenerate AABB: {stl_file}")

        def normalise_query_points(points: torch.Tensor) -> torch.Tensor:
            return (points - centre) / half_longest_edge

        format_description = (
            f"legacy SirenNet (hidden={first_weight.shape[0]}×{len(layer_indices)})"
        )
    else:
        raise ValueError(
            f"Unsupported SDF checkpoint format: {ckpt_path}. "
            "Expected 'state_dict' or 'model_state_dict'."
        )

    net.eval()
    print(f"{format_description} loaded")
    print('=' * 50 + '\n')

    def sdf_query(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query the neural SDF for a batch of points.

        Args:
            points: ``(N, 3)`` tensor in normalised world coordinates
                    (same space as produced by ``get_normalization_parameters``).

        Returns:
            (binary_states, sdf_values)
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=device)
        else:
            points = points.to(device)

        with torch.no_grad():
            x_norm = normalise_query_points(points)
            sdf_values = net(x_norm)                        # (N, 1)

        # Negative SDF → inside the model (1.0), positive → outside (0.0)
        binary_states = (sdf_values < 0.0).float()          # (N, 1)
        return binary_states, sdf_values

    return sdf_query


def sample_points_gpu(H_gpu, aabb_min, aabb_max, n_samples, device='cuda', verbose=False):
    """
    GPU加速采样 - 已为内存管理优化
    
    Args:
        H_gpu: GPU隐式函数
        aabb_min: 边界框最小值
        aabb_max: 边界框最大值  
        n_samples: 采样点数量
        device: 设备类型，默认'cuda'
        batch_size: 批次大小，默认50000
        verbose: 是否显示打印信息和进度条，默认True
    """
    from tqdm import tqdm
    
    if verbose:
        print(f"GPU采样 {n_samples:,} 个点...")
    
    aabb_min_t = torch.tensor(aabb_min, dtype=torch.float32, device=device)
    aabb_max_t = torch.tensor(aabb_max, dtype=torch.float32, device=device)

    random_vals = torch.rand(n_samples, 3, device=device)
    final_points_gpu = aabb_min_t + random_vals * (aabb_max_t - aabb_min_t)

    with torch.no_grad():
        final_values_gpu = H_gpu(final_points_gpu).view(-1, 1)

    torch.cuda.empty_cache()
    
    if verbose:
        print(f"GPU采样完成! 内部点数量: {torch.sum(final_values_gpu).item():,}")
    
    return final_points_gpu, final_values_gpu

def visualize_results(stl_file_path, points, values, scale, p_min_translation):
    """
    使用 trimesh 可视化内部点、原始模型、坐标轴和边界框。
    """
    print("\n正在准备可视化...")
    
    # 1. 将数据移回CPU并转换为Numpy
    points_cpu = points.cpu().numpy()
    values_cpu = values.cpu().numpy().flatten()
    
    # 2. 筛选出所有内部点
    inside_points = points_cpu[values_cpu == 1]
    outside_points = points_cpu[values_cpu == 0]
    
    if inside_points.shape[0] == 0:
        print("警告: 未找到任何内部点进行可视化。")
        return
        
    print(f"将 {inside_points.shape[0]:,} 个内部点进行可视化。")

    # 3. 加载原始STL模型并应用与采样点相同的归一化变换
    mesh = trimesh.load_mesh(stl_file_path)
    mesh.apply_translation(-p_min_translation)
    mesh.apply_scale(scale)
    
    # 3a. 在控制台打印归一化后的坐标范围
    normalized_bounds = mesh.bounds
    print("\n--- 可视化坐标范围 (归一化后) ---")
    print(f"模型边界框 (Min): {np.round(normalized_bounds[0], 4)}")
    print(f"模型边界框 (Max): {np.round(normalized_bounds[1], 4)}")
    print("------------------------------------")

    # 3b. 创建一个代表坐标轴的3D对象
    axis_vis = trimesh.creation.axis()

    # 3c. 创建一个代表模型边界框的线框对象 (CORRECTED LINE)
    box_outline = trimesh.path.creation.box_outline(
        extents=mesh.extents,
        transform=mesh.bounding_box.transform
    )

    # 4. 将模型设置为半透明
    mesh.visual.face_colors = [150, 150, 150, 100]
    
    # 5. 创建内部点的点云对象
    point_cloud = trimesh.points.PointCloud(inside_points)
    point_cloud.colors = [255, 0, 0, 255]

    outside_point_cloud = trimesh.points.PointCloud(outside_points)
    outside_point_cloud.colors = [0, 0, 255, 255]
    
    # 6. 创建一个场景
    scene = trimesh.Scene([
        mesh,
        point_cloud,
        outside_point_cloud,
        axis_vis,
        box_outline
    ])
    
    # 7. 显示场景
    print("弹出可视化窗口...")
    scene.show()

# --- 使用示例 ---
if __name__ == '__main__':
    # 创建一个虚拟的STL文件用于测试
    stl_file_path = 'stlFiles/fertility.stl'
    
    # 获取归一化后的模型边界
    scale, p_min, newBounds = get_normalization_parameters(stl_file_path)
    aabb_min_norm = np.zeros(3)
    aabb_max_norm = newBounds[1] - newBounds[0]
    
    # 1. Load neural SDF (run train_sdf.py first if checkpoint is missing)
    H = load_sdf_implicit_function(stl_file_path)
    
    # # 2. 采样
    # for i in range(100):
    #     points, values = sample_points_gpu(H, aabb_min_norm, aabb_max_norm, n_samples=int(100),verbose=True)
    #     visualize_results(stl_file_path, points, values, scale, p_min)
    #     print(f"第{i}次采样完成")

    points, values = sample_points_gpu(H, aabb_min_norm, aabb_max_norm, n_samples=int(100),verbose=True)
    visualize_results(stl_file_path, points, values, scale, p_min)
    
    # # 3. 验证结果
    # print("\n结果抽样检查:")
    # print("采样点坐标 (前5个):")
    # print(points[:5].cpu().numpy())
    # print("判断结果 (0=外部, 1=内部) (前5个):")
    # print(values[:5].cpu().numpy())
    
    # # 检查值的分布，应该只有0和1
    # unique_values = torch.unique(values)
    # print(f"\n输出中唯一的值: {unique_values.cpu().numpy()}")
    # assert len(unique_values) <= 2 # 应该只有0和1（或其中之一）

    # visualize_results(stl_file_path, points, values, scale, p_min)
