import torch
import torch.nn.functional as F
import math
from .structure_loss_torch import StructureLossCalculatorTorch
from .structure_loss_torch_sm_force import StructureLossCalculatorTorchSMForce

def _skew_batch(v):
	# v: (m,3) -> (m,3,3)
	x, y, z = v.unbind(-1)
	O = torch.zeros_like(x)
	return torch.stack([
		torch.stack([ O, -z,  y], dim=-1),
		torch.stack([ z,  O, -x], dim=-1),
		torch.stack([-y,  x,  O], dim=-1),
	], dim=-2)

def _align_z_to_v_batch(v, eps=1e-12):
	# v: (m,3) -> R: (m,3,3), mapping +z to v/||v||
	v = v / (v.norm(dim=-1, keepdim=True) + eps)
	m = v.shape[0]
	device, dtype = v.device, v.dtype
	k = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(v)
	c = (k * v).sum(dim=-1)                   # (m,)
	vx = torch.cross(k, v, dim=-1)            # (m,3) = axis * sin(theta)
	s = vx.norm(dim=-1)                        # (m,)
	K = _skew_batch(vx)                        # (m,3,3)
	I = torch.eye(3, device=device, dtype=dtype).expand(m, 3, 3)
	f = ((1.0 - c) / (s * s + eps)).view(-1,1,1)
	R_general = I + K + K @ K * f

	R_180 = torch.tensor([[1.0, 0.0, 0.0],
	                      [0.0,-1.0, 0.0],
	                      [0.0, 0.0,-1.0]], device=device, dtype=dtype).expand(m,3,3)

	mask_parallel = (s < 1e-12).view(-1,1,1)
	mask_pos = (c > 0).view(-1,1,1)
	R = torch.where(mask_parallel, torch.where(mask_pos, I, R_180), R_general)
	return R

def _align_z_to_u_upper_half_robust(u, eps=1e-12):
    """
    针对上半球优化的坐标系对齐函数
    将 +z 轴旋转至 u 方向 (u_z >= 0)
    """
    # 1. 强制归一化，确保旋转矩阵不含缩放
    norm = torch.norm(u, dim=-1, keepdim=True)
    u = u / (norm + eps)
    
    m = u.shape[0]
    device, dtype = u.device, u.dtype
    
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    zero = torch.zeros(m, device=device, dtype=dtype)

    # 2. 构造反对称矩阵 K = [k x u]_x
    # 此时旋转轴 v = (0,0,1) x (ux, uy, uz) = (-uy, ux, 0)
    K = torch.stack([
        torch.stack([zero,  zero,  ux],   dim=-1), # Row 0
        torch.stack([zero,  zero,  uy],   dim=-1), # Row 1
        torch.stack([-ux,   -uy,   zero], dim=-1)  # Row 2
    ], dim=-2) 

    I = torch.eye(3, device=device, dtype=dtype).expand(m, 3, 3)
    
    # 3. Rodrigues 简化公式：R = I + K + K^2 * (1/(1+uz))
    # 因为 uz >= 0，所以 (1 + uz) 绝不会除以 0
    f = (1.0 / (1.0 + uz)).view(-1, 1, 1)
    # 使用 bmm 进行矩阵批处理乘法
    R = I + K + torch.bmm(K, K) * f
    
    return R

class ComprehensiveLoss:
    def __init__(self, weights, m, h, SMToolParas, aabb_min, aabb_max, max_time, paras_structure, device='cuda', check_func=None, struc_loss_calculator=None):
        self.weights = weights
        self.check_func = check_func
        self.m = m
        self.h = h
        self.aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32, device=device)
        self.aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32, device=device)
        self.max_time = max_time

        self.SMTipDiameter = SMToolParas['SMTipDiameter']
        self.SMShankDiameter = SMToolParas['SMShankDiameter']
        self.SMToolLength = SMToolParas['SMToolLength']
        self.SMHolderDiameter = SMToolParas['SMHolderDiameter']
        self.SMHolderLength = SMToolParas['SMHolderLength']
        self.SMCollisionMargin = SMToolParas['SMCollisionMargin']
        self.nSMTip = SMToolParas['nSMTip']
        self.nSMTool = SMToolParas['nSMTool']
        self.nSMHolder = SMToolParas['nSMHolder']
        self.sm_forward_chunk_size = int(SMToolParas.get('SMForwardChunkSize', 65536))

        self.logitDensity = paras_structure['logitDensity']
        self.densitySharpness = paras_structure['densitySharpness']
        self.use_sm_cutting_force_structure = bool(
            paras_structure.get('use_sm_cutting_force_structure', False)
        )

        if struc_loss_calculator is not None:
            self.strucLossCalculator = struc_loss_calculator
        else:
            print("[ComprehensiveLoss] Creating StructureLossCalculator...")
            normalizedSize = (aabb_max - aabb_min) / max(aabb_max - aabb_min)
            common_structure_kwargs = dict(
                physical_size=normalizedSize,
                max_resolution=paras_structure['max_resolution'],
                batch_size=paras_structure['time_check_size'],
                cg_iter=paras_structure['cg_iter'],
                cg_tol=paras_structure['cg_tol'],
                E0=paras_structure['e0'],
                grad_clip_threshold=paras_structure['grad_clip_threshold'],
                grad_scale=paras_structure['grad_scale'],
                use_total_mass=paras_structure['use_total_mass'],
                device=device
            )
            if self.use_sm_cutting_force_structure:
                self.strucLossCalculator = StructureLossCalculatorTorchSMForce(
                    **common_structure_kwargs,
                    sm_cutting_force_magnitude=paras_structure.get('sm_cutting_force_magnitude', 0.0),
                    sm_max_displacement_norm=paras_structure['sm_max_displacement_norm']
                )
            else:
                self.strucLossCalculator = StructureLossCalculatorTorch(
                    **common_structure_kwargs,
                    max_displacement=paras_structure['max_displacement']
                )
            print("[ComprehensiveLoss] Compiling _solve_cg with torch.compile (this may take a while)...")
            self.strucLossCalculator._solve_cg = torch.compile(
                self.strucLossCalculator._solve_cg, 
                mode='default'
            )
            print("[ComprehensiveLoss] StructureLossCalculator ready.")
    
    def get_field_values(self, model, positions_inside, target_state_inside, batch_N, batch_mask):
        N, NS, NAMTool = batch_N
        NAll = N + NS + NAMTool
        mask2d = batch_mask.unsqueeze(-1)

        f1_masked, f2_masked = model.forward(positions_inside, field_type='both')
        f1_base, f2_base = f1_masked[:N], f2_masked[:N]

        f1_all = torch.full((NAll,1),65500,device=positions_inside.device, dtype=f1_base.dtype)
        f2_all = torch.full((NAll,1),2, device=positions_inside.device, dtype=f2_base.dtype)
        f1_all = f1_all.masked_scatter(mask2d, f1_masked)
        f2_all = f2_all.masked_scatter(mask2d, f2_masked)

        logits1_raw, logits2_raw = model.forward(positions_inside, field_type='masks')
        logits1 = torch.where(target_state_inside == 1, 100.0, logits1_raw)
        logits2 = torch.where(target_state_inside == 1, -100.0, logits2_raw)

        logits1_all, logits2_all = torch.full((NAll,1), -100.0, device=positions_inside.device, dtype=logits1.dtype), torch.full((NAll,1), -100.0, device=positions_inside.device, dtype=logits2.dtype)
        logits1_all = logits1_all.masked_scatter(mask2d, logits1)
        logits2_all = logits2_all.masked_scatter(mask2d, logits2)

        probs1_all, probs2_all = torch.sigmoid(logits1_all), torch.sigmoid(logits2_all)

        return f1_all, f2_all, logits1_all, logits2_all, probs1_all, probs2_all, logits1, logits2, f1_base, f2_base


    def __call__(self, model, positions_inside, target_state_inside, batch_N, batch_mask, toolPoints_raw, outputSMTool = False, printForTest = False):
        N, NS, NAMTool = batch_N
        f1_all, f2_all, logits1_all, logits2_all, probs1_all, probs2_all, logits1, logits2, f1_base, f2_base = self.get_field_values(model, positions_inside, target_state_inside, batch_N, batch_mask)

        field1_S, field2_S = f1_all[N:N+NS], f2_all[N:N+NS]
        field1_AMTool, field2_AMTool = f1_all[N+NS:N+NS+NAMTool], f2_all[N+NS:N+NS+NAMTool]

        logits1_S = logits1_all[N:N+NS]
        logits2_S = logits2_all[N:N+NS]
        logits1_AMTool = logits1_all[N+NS:N+NS+NAMTool]
        logits2_AMTool = logits2_all[N+NS:N+NS+NAMTool]
        
        probs1_base, probs2_base = probs1_all[:N], probs2_all[:N]
        probs2_S = probs2_all[N:N+NS]
        probs1_AMTool = probs1_all[N+NS:N+NS+NAMTool]
        probs2_AMTool = probs2_all[N+NS:N+NS+NAMTool]

        # Calculate the loss
        # 1. Final state loss
        l_final_state, fs_ratio = self._loss_final_state(logits1, logits2, target_state_inside, printForTest)

        # 2. Self support loss
        l_self_support_avg, l_self_support_max, ss_ratio = self._loss_self_support_logits(positions_inside[:N], f1_base, probs1_base, field1_S, field2_S, logits1_S, logits2_S, printForTest)

        # 3. AM collision free loss
        l_AM_Collision_Free, am_ratio = self._loss_AM_Collision_Free_logits(f1_base, probs1_base, field1_AMTool, field2_AMTool, logits1_AMTool, logits2_AMTool, printForTest)

        # 4. SM collision free loss
        if outputSMTool:
            l_SM_Collision_Free, SMToolInfo, sm_ratio = self._loss_SM_Collision_Free_logits(model, f1_base, f2_base, probs1_base, probs2_base, positions_inside[:N], toolPoints_raw, outputSMTool, printForTest)
        else:
            l_SM_Collision_Free, sm_ratio = self._loss_SM_Collision_Free_logits(model, f1_base, f2_base, probs1_base, probs2_base, positions_inside[:N], toolPoints_raw, outputSMTool, printForTest)

        # 5. Operation volume loss
        l_operation_volume, vol_ratio = self._loss_operation_volume(logits1, 0.1, target_state_inside, printForTest)

        # Store all losses in dictionary
        losses = {
            'final_state': l_final_state,
            'self_support_avg': l_self_support_avg,
            'self_support_max': l_self_support_max,
            'AM_Collision_Free': l_AM_Collision_Free,
            'SM_Collision_Free': l_SM_Collision_Free,
            'operation_volume': l_operation_volume,
        }

        # Physical metrics (ratio-based, computed with no_grad inside each _loss method)
        metrics = {
            'final_state_accuracy': fs_ratio,
            'self_support_ratio': ss_ratio,
            'AM_collision_free_ratio': am_ratio,
            'SM_collision_free_ratio': sm_ratio,
            'operation_volume_ratio': vol_ratio,
        }

        l_self_support = l_self_support_avg + 0.0 * l_self_support_max

        l_operation_volume_new = F.relu(l_operation_volume - self.weights.get('allowed_operation_volume'))
        l_final_state_new = F.relu(l_final_state - self.weights.get('allowed_final_state'))
        l_AM_Collision_Free_new = F.relu(l_AM_Collision_Free - self.weights.get('allowed_AM_Collision_Free'))
        l_self_support_new = F.relu(l_self_support - self.weights.get('allowed_self_support'))

        # Calculate weighted total loss with adaptive weighting
        total_loss = (self.weights.get('final_state', 1.0) * l_final_state_new +
                         self.weights.get('self_support', 1.0) * l_self_support_new +
                         self.weights.get('AM_Collision_Free', 1.0) * l_AM_Collision_Free_new +
                         self.weights.get('SM_Collision_Free', 1.0) * l_SM_Collision_Free +
                         self.weights.get('operation_volume', 1.0) * l_operation_volume_new
                         )

        if outputSMTool:
            return total_loss, losses, metrics, f1_all, f2_all, probs1_all, probs2_all, SMToolInfo
        else:
            return total_loss, losses, metrics, f1_all, f2_all, probs1_all, probs2_all

    def get_loss_structure(self, model, positions, printForTest = False):
        is_inSide, _ = self.check_func(positions)
        f1, f2, logits1_raw, logits2_raw = model.forward(positions, field_type='timesAndMasks')
        logits1 = torch.where(is_inSide == 1, 100.0, logits1_raw)
        logits2 = torch.where(is_inSide == 1, -100.0, logits2_raw)

        probs1, probs2 = torch.sigmoid(logits1), torch.sigmoid(logits2)

        l_structure, l_simp_penalty = self._loss_structure(model, positions, f1, f2, probs1, probs2, logits1, logits2, 0.05, printForTest)

        total_grid = self.strucLossCalculator.TUPLE_RES[0] * self.strucLossCalculator.TUPLE_RES[1] * self.strucLossCalculator.TUPLE_RES[2]
        structure_normalized = l_structure.item() / total_grid

        l_structure_new = F.relu(l_structure - self.weights.get('allowed_structure'))

        return l_structure, l_structure_new, l_simp_penalty, structure_normalized

    def _loss_final_state(self, logits1, logits2, target_state, printForTest = False):
        """Calculate final state loss by combining solid and empty state losses.
        Returns (loss, accuracy_ratio) where accuracy_ratio is the fraction of
        empty-target points correctly classified."""
        
        mask_empty = (target_state.squeeze() == 0)
        if not torch.any(mask_empty):
            return torch.tensor(0.0, device=logits1.device), 1.0
        
        l1 = logits1.squeeze()[mask_empty]
        l2 = logits2.squeeze()[mask_empty]

        loss_empty = self._loss_final_empty_state(l1, l2)
        
        loss = loss_empty.mean()

        with torch.no_grad():
            correct = (l1 < 0) | (l2 > 0)
            accuracy_ratio = correct.float().mean().item()

        if printForTest:
            print(f"Final state loss: {loss}")
            
        return loss, accuracy_ratio

    def _loss_final_empty_state(self, logits1, logits2):
        """Calculate loss for empty state (target_state=0).
        Penalizes when t_end is too close to [time1, time2] interval."""

        loss_M1 = F.binary_cross_entropy_with_logits(logits1, torch.zeros_like(logits1), reduction='none')
        loss_M2 = F.binary_cross_entropy_with_logits(logits2, torch.ones_like(logits2), reduction='none')

        loss = loss_M1*loss_M2

        return loss
        
    def _loss_self_support(self, points, field1_P, probs1_P, field1_S, field2_S, logits1_S, probs2_S, printForTest = False):
        """Calculate loss for self-support points."""
        # 创建mask找出z坐标大于h且field1_P小于t_end的点

        z_mask = (points[:, 2] > self.h).squeeze() # z-coordinate > h
        probs_mask = (probs1_P > 0.5).squeeze() # valid additive manufacturing points
        combined_mask = z_mask & probs_mask # (N,)

        if not torch.any(combined_mask):
            return torch.tensor(0.0, device=points.device)

        nPoints = points.shape[0] # number of points
        time1_S = field1_S.reshape(nPoints, -1)[combined_mask] # (K, nS)
        time2_S = (field1_S + field2_S * (self.max_time - field1_S)).reshape(nPoints, -1)[combined_mask] # (K, nS)
        logits1_S_masked = logits1_S.reshape(nPoints, -1)[combined_mask] # (K, nS)
        probs2_S_masked = probs2_S.reshape(nPoints, -1)[combined_mask] # (K, nS)

        time1 = field1_P[combined_mask] # (K,1)

        h1 = F.relu(time1_S - time1 + self.m)
        # h1 = F.softplus(time1_S - time1 + self.m)
        BCE_M1 = F.binary_cross_entropy_with_logits(logits1_S_masked, torch.ones_like(logits1_S_masked), reduction='none')
        L1 = h1 + self.weights.get('lambda_BCE', 1.0) * BCE_M1

        h2=F.relu(time1-time2_S+self.m)
        # h2=F.softplus(time1-time2_S+self.m)
        p_or = 1.0 - probs2_S_masked * (1.0 - torch.exp(-h2))
        L2 = -torch.log(torch.clamp(p_or, min=1e-7))

        V = L1 + self.weights.get('lambda_L2', 1.0) * L2

        loss = torch.min(V,dim=1).values

        N = loss.shape[0]
        num_hard = int (N*0.3)
        worst_K_values, _ = torch.topk(loss, k=num_hard, dim=0, largest = True)
        avg_loss = torch.mean(worst_K_values)

        # avg_loss = torch.mean(loss)
        max_loss = torch.max(loss)

        if printForTest:
            print(f"Self support loss: {avg_loss}")
            print(f"Self support max loss: {max_loss}")
            # Print average values for debugging
            print(f"Average h1: {torch.mean(h1):.6f}")
            print(f"Average BCE_M1: {torch.mean(BCE_M1):.6f}")
            print(f"Average L1: {torch.mean(L1):.6f}")
            print(f"Average L2: {torch.mean(L2):.6f}")

        return avg_loss, max_loss
    
    def _loss_self_support_logits(self, points, field1_P, probs1_P, field1_S, field2_S, logits1_S, logits2_S, printForTest = False, sharpness=1.0):
        """
        完全在 Logits 空间计算自支撑 Loss。
        更加数值稳定，梯度流更好。
        """
        
        # 1. Masking (保持不变)
        z_mask = (points[:, 2] > self.h).squeeze()
        probs_mask = (probs1_P > 0.5).squeeze() # 注意：这里如果probs1_P也是logits来的，也可以改用logits判断，但通常probs用于硬筛选没问题
        combined_mask = z_mask & probs_mask 

        if not torch.any(combined_mask):
            return torch.tensor(0.0, device=points.device), torch.tensor(0.0, device=points.device), 1.0

        nPoints = points.shape[0]
        
        # 2. 准备数据
        time_P = field1_P[combined_mask] 
        
        time1_S = field1_S.reshape(nPoints, -1)[combined_mask] 
        time2_S = (field1_S + field2_S * (self.max_time - field1_S)).reshape(nPoints, -1)[combined_mask]
        logit1_S = logits1_S.reshape(nPoints, -1)[combined_mask] 
        logit2_S = logits2_S.reshape(nPoints, -1)[combined_mask]

        # 3. 计算 Logits
        
        # 逻辑 A: (Time OK) AND (Exist OK)
        # Time Logit: linear difference
        logit_time_start = sharpness * (time_P - time1_S - self.m)
        logit_prob_exist = sharpness * (logit1_S - 0.5)

        # Created Logit = min(Time, Exist)
        logits_created = torch.minimum(logit_time_start, logit_prob_exist)
        
        # 逻辑 B: (Time Not Cut) OR (Not Remove)
        logit_time_cut = sharpness * (time2_S - time_P - self.m)
        logit_prob_not_remove = sharpness * (-0.5 - logit2_S)
        
        # Remain Logit = max(Time, NotRemove)
        logits_remain = torch.maximum(logit_time_cut, logit_prob_not_remove)
        
        # 4. Final Logit calculation
        # Support = Created AND Remain
        logits_final = torch.minimum(logits_created, logits_remain) # (M, K)
        
        # 5. Loss Calculation (Softplus)
        # 目标：我们希望最大的那个 logit 尽可能大
        best_support_logit, _ = torch.max(logits_final, dim=1) # (M,)
        
        # 原来是: -log(sigmoid(x))
        # 现在是: softplus(-x) = log(1 + exp(-x))
        # 这种写法在 x 很大（>0）时，loss 趋近 0；x 很小（<0）时，loss 线性增加
        # loss_per_point = F.softplus(-best_support_logit)
        loss_per_point = F.relu(-best_support_logit)
        
        avg_loss = torch.mean(loss_per_point)
            
        max_loss = torch.max(loss_per_point)

        with torch.no_grad():
            r_created = torch.minimum(time_P - time1_S, logit1_S)
            r_remain  = torch.maximum(time2_S - time_P, -logit2_S)
            r_final   = torch.minimum(r_created, r_remain)
            r_best, _ = torch.max(r_final, dim=1)
            support_ratio = (r_best > 0).float().mean().item()

        return avg_loss, max_loss, support_ratio
    
    def _loss_AM_Collision_Free(self, field1_P, probs1_P, field1_AMTool, field2_AMTool, probs1_AMTool, probs2_AMTool, printForTest = False):
        time_mask = (probs1_P > 0.5).squeeze()
        if not torch.any(time_mask):
            return torch.tensor(0.0, device=field1_P.device)

        nPoints = field1_P.shape[0]
        time1_AMTool = field1_AMTool.reshape(nPoints, -1)[time_mask]
        time2_AMTool = (field1_AMTool + field2_AMTool * (self.max_time - field1_AMTool)).reshape(nPoints, -1)[time_mask]
        probs1_AMTool_masked = probs1_AMTool.reshape(nPoints, -1)[time_mask]
        probs2_AMTool_masked = probs2_AMTool.reshape(nPoints, -1)[time_mask]
        time1 = field1_P[time_mask]

        # condition 1: probs1_AMTool = 0
        # condition 2: tA1>t1
        h2 = F.relu(time1-time1_AMTool+self.m)
        p2 = torch.exp(-h2)
        # condition 3: tA2<t1 and probs2_AMTool = 1
        h3 = F.relu(time2_AMTool-time1+self.m)
        p3 = probs2_AMTool_masked * torch.exp(-h3)

        p_or = 1.0 - probs1_AMTool_masked * (1.0 - p2) * (1.0 - p3)
        V = -torch.log(torch.clamp(p_or, min=1e-7))

        loss = torch.max(V,dim=1).values
        # tau = 0.1
        # loss = tau * torch.logsumexp(V/tau, dim=1)
        
        avg_loss = torch.mean(loss)

        if printForTest:
            print(f"AM collision free loss: {avg_loss}")

        return avg_loss
    
    def _loss_AM_Collision_Free_logits(self, field1_P, probs1_P, field1_AMTool, field2_AMTool, logits1_AMTool, logits2_AMTool, printForTest = False, sharpness=1.0):
        """
        完全在 Logits 空间计算避障 Loss。
        目标：最小化点 P 打印时间与工具占据时间的重叠。
        Returns (loss, collision_free_ratio).
        """
        # 1. Masking
        time_mask = (probs1_P > 0.5).squeeze()
        if not torch.any(time_mask):
            return torch.tensor(0.0, device=field1_P.device), 1.0

        nPoints = field1_P.shape[0]
        
        # 2. 准备数据
        time_P = field1_P[time_mask] # (M, 1)
        
        # 工具占据的起始和结束时间
        time_A1 = field1_AMTool.reshape(nPoints, -1)[time_mask] # (M, K)
        time_A2 = (field1_AMTool + field2_AMTool * (self.max_time - field1_AMTool)).reshape(nPoints, -1)[time_mask]
        logits_A1 = logits1_AMTool.reshape(nPoints, -1)[time_mask]
        logits_A2 = logits2_AMTool.reshape(nPoints, -1)[time_mask]

        # 3. 计算冲突的 Logit (Collision Logit)
        # 冲突 = (工具存在) AND (A1 在 P 之前) AND (A2 在 P 之后 或 A2 不存在)
        
        logit_exist = sharpness * (logits_A1 - (-0.5))
        logit_start_before = sharpness * (time_P - time_A1 + self.m)
        logit_created = torch.minimum(logit_start_before, logit_exist)

        logit_end_after = sharpness * (time_A2 - time_P + self.m)
        logit_exist_A2 = sharpness * (0.5 - logits_A2)
        logit_remain = torch.maximum(logit_end_after, logit_exist_A2)

        logit_collision = torch.minimum(logit_created, logit_remain)

        # 4. Loss 计算
        # 只要任何一个采样点 S 与 P 发生冲突，P 就不安全
        # 取 max 是为了捕捉最严重的冲突
        worst_collision_logit, _ = torch.max(logit_collision, dim=1) # (M,)

        # 我们希望 worst_collision_logit 尽可能小（负无穷）
        # 使用 softplus(x) = log(1 + exp(x))
        # 当 x 为负很大时，loss 趋近 0；当 x 为正很大时，loss 线性增加
        # loss_per_point = F.softplus(worst_collision_logit)
        loss_per_point = F.relu(worst_collision_logit)

        avg_loss = torch.mean(loss_per_point)

        with torch.no_grad():
            r_created   = torch.minimum(time_P - time_A1, logits_A1)
            r_remain    = torch.maximum(time_A2 - time_P, -logits_A2)
            r_collision = torch.minimum(r_created, r_remain)
            r_worst, _  = torch.max(r_collision, dim=1)
            collision_free_ratio = (r_worst <= 0).float().mean().item()

        if printForTest:
            print(f"AM collision logit loss: {avg_loss.item():.6f}")

        return avg_loss, collision_free_ratio

    def _loss_SM_Collision_Free(self, model, field1_P, field2_P, probs1_P, probs2_P, basePoints, toolPoints_raw_all, outputSMTool = False, printForTest = False):
        mask_M1 = (probs1_P > 0.5).squeeze()
        mask_M2 = (probs2_P > 0.5).squeeze()
        time_mask = mask_M1 & mask_M2
        if not torch.any(time_mask):
            return torch.tensor(0.0, device=field1_P.device)
        
        time2 = (field1_P + field2_P * (self.max_time - field1_P))
        time2_masked = time2[time_mask]

        basePoints_masked = basePoints[time_mask]

        two_tool_vec = model.forward(basePoints_masked, field_type='field3')
        normalVectors_masked = two_tool_vec[...,:3]
        tool_vec = two_tool_vec[...,3:]
        ballCenter = basePoints_masked + normalVectors_masked * self.SMTipDiameter / 2.0

        toolPoints_raw = toolPoints_raw_all[time_mask]

        # Rot = _align_z_to_v_batch(tool_vec) # (m,3,3)
        Rot = _align_z_to_u_upper_half_robust(tool_vec)
        rotated_toolPoints = torch.einsum('ikl,ijl->ijk',Rot, toolPoints_raw) # (m,n,3)
        toolPoints = rotated_toolPoints + ballCenter.unsqueeze(1) # (m,n,3)

        tP_flat = toolPoints.reshape(-1,3)
        mask_inside = ((tP_flat >= self.aabb_min) & (tP_flat <= self.aabb_max)).all(dim=-1)

        isInModel, _ = self.check_func(tP_flat[mask_inside])
        f1_tool_M, f2_tool_M, l1_tool_M_raw, l2_tool_M_raw = model.forward(tP_flat[mask_inside], field_type='timesAndMasks')
        l1_tool_M = torch.where(isInModel == 1, 100.0, l1_tool_M_raw)
        l2_tool_M = torch.where(isInModel == 1, -100.0, l2_tool_M_raw)
        
        f1_tool = torch.full((tP_flat.shape[0],1), 65500, device=tP_flat.device, dtype=f1_tool_M.dtype)
        f2_tool = torch.full((tP_flat.shape[0],1), 2, device=tP_flat.device, dtype=f2_tool_M.dtype)
        f1_tool = f1_tool.masked_scatter(mask_inside.unsqueeze(-1), f1_tool_M)
        f2_tool = f2_tool.masked_scatter(mask_inside.unsqueeze(-1), f2_tool_M)
        l1_tool = torch.full((tP_flat.shape[0],1), -100.0, device=tP_flat.device, dtype=l1_tool_M.dtype)
        l2_tool = torch.full((tP_flat.shape[0],1), -100.0, device=tP_flat.device, dtype=l2_tool_M.dtype)
        l1_tool = l1_tool.masked_scatter(mask_inside.unsqueeze(-1), l1_tool_M)
        l2_tool = l2_tool.masked_scatter(mask_inside.unsqueeze(-1), l2_tool_M)

        time1_tool = f1_tool.reshape(toolPoints.shape[0], -1)
        time2_tool = (f1_tool + f2_tool * (self.max_time - f1_tool)).reshape(toolPoints.shape[0], -1)
        logits1_tool = l1_tool.reshape(toolPoints.shape[0], -1)
        logits2_tool = l2_tool.reshape(toolPoints.shape[0], -1)
        probs1_tool = torch.sigmoid(logits1_tool)
        probs2_tool = torch.sigmoid(logits2_tool)

        if outputSMTool:
            with torch.no_grad():
                tool_vec_all = torch.full((time2.shape[0],3), 0.0, device=time2.device, dtype=tool_vec.dtype)
                tool_vec_all[time_mask] = tool_vec
                toolPoints_all = torch.full((time2.shape[0], toolPoints.shape[1], 3), 0.0, device=time2.device, dtype=toolPoints.dtype)
                toolPoints_all[time_mask] = toolPoints
                time1_tool_all = torch.full((time2.shape[0],time1_tool.shape[1]), -100.0, device=time2.device, dtype=f1_tool.dtype)
                time1_tool_all[time_mask] = time1_tool
                time2_tool_all = torch.full((time2.shape[0],time2_tool.shape[1]), -100.0, device=time2.device, dtype=f2_tool.dtype)
                time2_tool_all[time_mask] = time2_tool
                probs1_tool_all = torch.full((time2.shape[0],probs1_tool.shape[1]), 0.0, device=time2.device, dtype=probs1_tool.dtype)
                probs1_tool_all[time_mask] = probs1_tool
                probs2_tool_all = torch.full((time2.shape[0],probs2_tool.shape[1]), 0.0, device=time2.device, dtype=probs2_tool.dtype)
                probs2_tool_all[time_mask] = probs2_tool
                SM_tool_info = {
                    'tool_vec': tool_vec_all,
                    'toolPoints': toolPoints_all,
                    'time1': time1_tool_all,
                    'time2': time2_tool_all,
                    'probs1': probs1_tool_all,
                    'probs2': probs2_tool_all
                }

        # condition 1: probs1_tool = 0
        # condition 2: tS1>t2
        h2 = F.relu(time2_masked-time1_tool+self.m)
        p2 = torch.exp(-h2)
        # condition 3: tS2<t2 and probs2_tool = 1
        h3 = F.relu(time2_tool-time2_masked+self.m)
        p3 = probs2_tool * torch.exp(-h3)
        
        p_or = 1.0 - probs1_tool * (1.0 - p2) * (1.0 - p3)
        V = -torch.log(torch.clamp(p_or, min=1e-7))

        loss = torch.max(V,dim=1).values
        # tau = 0.1
        # loss = tau * torch.logsumexp(V/tau, dim=1)

        avg_loss = torch.mean(loss)

        if printForTest:
            print(f"SM collision free loss: {avg_loss}")

        if outputSMTool:
            return avg_loss, SM_tool_info
        else:
            return avg_loss
    
    def _loss_SM_Collision_Free_logits(self, model, field1_P, field2_P, probs1_P, probs2_P, basePoints, toolPoints_raw_all, outputSMTool = False, printForTest = False, sharpness=1.0):
        # 1. Masking
        mask_M1 = (probs1_P > 0.5).squeeze()
        mask_M2 = (probs2_P > 0.5).squeeze()
        time_mask = mask_M1 & mask_M2
        if not torch.any(time_mask):
            if outputSMTool:
                return torch.tensor(0.0, device=field1_P.device), None, 1.0
            return torch.tensor(0.0, device=field1_P.device), 1.0
        
        # 工具作用的时间点（P 点完成的时间）
        time2_masked = (field1_P + field2_P * (self.max_time - field1_P))[time_mask]
        basePoints_masked = basePoints[time_mask]

        # 2. 几何计算 (保持原样)
        two_tool_vec = model.forward(basePoints_masked, field_type='field3')
        normalVectors_masked = two_tool_vec[...,:3]
        tool_vec = two_tool_vec[...,3:]
        ballCenter = basePoints_masked + normalVectors_masked * self.SMTipDiameter / 2.0
        toolPoints_raw = toolPoints_raw_all[time_mask]

        # Rot = _align_z_to_v_batch(tool_vec)
        Rot = _align_z_to_u_upper_half_robust(tool_vec)
        rotated_toolPoints = torch.einsum('ikl,ijl->ijk', Rot, toolPoints_raw)
        toolPoints = rotated_toolPoints + ballCenter.unsqueeze(1) # (M, n, 3)

        # 3. 采样点的 Logits 准备
        tP_flat = toolPoints.reshape(-1, 3)
        mask_inside = ((tP_flat >= self.aabb_min) & (tP_flat <= self.aabb_max)).all(dim=-1)
        mask_below = (
            (tP_flat[:, 0] >= self.aabb_min[0]) & (tP_flat[:, 0] <= self.aabb_max[0]) &
            (tP_flat[:, 1] >= self.aabb_min[1]) & (tP_flat[:, 1] <= self.aabb_max[1]) &
            (tP_flat[:, 2] <= self.aabb_min[2])
        )

        tP_inside = tP_flat[mask_inside]
        if tP_inside.numel() == 0:
            empty = tP_inside.new_empty((0, 1))
            isInModel = empty
            f1_tool_M = empty
            f2_tool_M = empty
            l1_tool_M = empty
            l2_tool_M = empty
        else:
            chunk_size = max(1, self.sm_forward_chunk_size)
            is_in_chunks = []
            f1_chunks = []
            f2_chunks = []
            l1_chunks = []
            l2_chunks = []

            for start in range(0, tP_inside.shape[0], chunk_size):
                tP_chunk = tP_inside[start:start + chunk_size]
                with torch.no_grad():
                    isInModel_chunk, _ = self.check_func(tP_chunk)
                f1_chunk, f2_chunk, l1_raw_chunk, l2_raw_chunk = model.forward(tP_chunk, field_type='timesAndMasks')

                is_in_chunks.append(isInModel_chunk)
                f1_chunks.append(f1_chunk)
                f2_chunks.append(f2_chunk)
                l1_chunks.append(torch.where(isInModel_chunk == 1, 100.0, l1_raw_chunk))
                l2_chunks.append(torch.where(isInModel_chunk == 1, -100.0, l2_raw_chunk))

            isInModel = torch.cat(is_in_chunks, dim=0)
            f1_tool_M = torch.cat(f1_chunks, dim=0)
            f2_tool_M = torch.cat(f2_chunks, dim=0)
            l1_tool_M = torch.cat(l1_chunks, dim=0)
            l2_tool_M = torch.cat(l2_chunks, dim=0)
        
        # 填充数据
        def fill_and_reshape(data, mask, fill_val, shape):
            full = torch.full((tP_flat.shape[0], 1), fill_val, device=tP_flat.device, dtype=data.dtype)
            full = full.masked_scatter(mask.unsqueeze(-1), data)
            return full.reshape(shape)

        time1_S = fill_and_reshape(f1_tool_M, mask_inside, 65500, (toolPoints.shape[0], -1))
        time2_S = fill_and_reshape(f1_tool_M + f2_tool_M * (self.max_time - f1_tool_M), mask_inside, -65500, (toolPoints.shape[0], -1))
        logits1_S = fill_and_reshape(l1_tool_M, mask_inside, -100.0, (toolPoints.shape[0], -1))
        logits2_S = fill_and_reshape(l2_tool_M, mask_inside, -100.0, (toolPoints.shape[0], -1))

        if outputSMTool:
            with torch.no_grad():
                probs1_tool = torch.sigmoid(logits1_S)
                probs2_tool = torch.sigmoid(logits2_S)
                tool_vec_all = torch.full((field1_P.shape[0],3), 0.0, device=field1_P.device, dtype=tool_vec.dtype)
                tool_vec_all[time_mask] = tool_vec
                toolPoints_all = torch.full((field1_P.shape[0], toolPoints.shape[1], 3), 0.0, device=field1_P.device, dtype=toolPoints.dtype)
                toolPoints_all[time_mask] = toolPoints
                time1_tool_all = torch.full((field1_P.shape[0],time1_S.shape[1]), -100.0, device=field1_P.device, dtype=time1_S.dtype)
                time1_tool_all[time_mask] = time1_S
                time2_tool_all = torch.full((field1_P.shape[0],time2_S.shape[1]), -100.0, device=field1_P.device, dtype=time2_S.dtype)
                time2_tool_all[time_mask] = time2_S
                probs1_tool_all = torch.full((field1_P.shape[0],probs1_tool.shape[1]), 0.0, device=field1_P.device, dtype=probs1_tool.dtype)
                probs1_tool_all[time_mask] = probs1_tool
                probs2_tool_all = torch.full((field1_P.shape[0],probs2_tool.shape[1]), 0.0, device=field1_P.device, dtype=probs2_tool.dtype)
                probs2_tool_all[time_mask] = probs2_tool
                SM_tool_info = {
                    'tool_vec': tool_vec_all,
                    'toolPoints': toolPoints_all,
                    'time1': time1_tool_all,
                    'time2': time2_tool_all,
                    'probs1': probs1_tool_all,
                    'probs2': probs2_tool_all
                }

        # 4. 计算冲突 Logit (SM 逻辑)
        # 冲突条件：(材料存在) AND (材料出生在工具到达前) AND (材料在工具离开后才被切除或永不切除)
        
        # A. 材料存在 (Mask 1)
        logit_exist = sharpness * (logits1_S - (-0.5))
        # B. 材料出生在工具到达前 (t1_S < t2_P)
        logit_born_before = sharpness * (time2_masked - time1_S + self.m)
        # C. 合并：材料已产生
        logit_material_present = torch.minimum(logit_exist, logit_born_before)

        # D. 材料在工具离开后还没被切除 (t2_S > t2_P 或 Mask 2 为空)
        logit_cut_after = sharpness * (time2_S - time2_masked + self.m)
        logit_not_cut_yet = sharpness * (0.5 - logits2_S) 
        logit_still_there = torch.maximum(logit_cut_after, logit_not_cut_yet)

        # E. 最终冲突 Logit: 既产生了，又还没被切除
        logit_collision = torch.minimum(logit_material_present, logit_still_there)

        # 5. Loss 计算
        # 捕捉每个工具位置上最严重的冲突
        worst_collision_logit, _ = torch.max(logit_collision, dim=1)
        # loss_per_point = F.softplus(worst_collision_logit)
        loss_per_point = F.relu(worst_collision_logit)

        avg_loss = torch.mean(loss_per_point)


        # 6. 新增：AABB 下方禁区惩罚 (Below AABB Penalty)
        # 计算所有点相对于 AABB 底面的侵入深度
        depth = self.aabb_min[2] - tP_flat[:, 2]
        
        # 仅对投影内 (mask_below) 且 确实在下方的点施加惩罚
        # 使用 zeros_like 确保设备和类型一致
        penalty_all = torch.where(mask_below, F.relu(depth), torch.zeros_like(depth))
        
        # Reshape 回 (M, n)，M 是工具位置数，n 是采样点数
        # 使用 -1 自动推导采样点维度，增强鲁棒性
        penalty_per_tool = penalty_all.reshape(toolPoints.shape[0], -1)
        
        # 取每个工具最严重的侵入值。dim=1 对应采样点维度
        worst_penalty_per_tool, _ = torch.max(penalty_per_tool, dim=1)
        
        # 对所有工具位置取平均。即使只有 1 个工具违规，loss 也会有响应
        loss_below_aabb = torch.mean(worst_penalty_per_tool)

        # 最终合并
        self.lambda_below = self.weights['SM_Below_AABB']
        total_loss = avg_loss + self.lambda_below * loss_below_aabb

        with torch.no_grad():
            r_present   = torch.minimum(logits1_S, time2_masked - time1_S)
            r_still     = torch.maximum(time2_S - time2_masked, -logits2_S)
            r_collision = torch.minimum(r_present, r_still)
            r_worst, _  = torch.max(r_collision, dim=1)
            collision_free_ratio = (r_worst <= 0).float().mean().item()

        if printForTest:
            print(f"SM collision logit loss: {avg_loss.item():.6f}")
            print(f"SM below AABB loss: {loss_below_aabb.item():.6f}")

        if outputSMTool:
            return total_loss, SM_tool_info, collision_free_ratio
        
        return total_loss, collision_free_ratio

    def _loss_operation_volume(self, logits1, eps_prob, target_state, printForTest = False):
        mask_0 = (target_state == 0).squeeze()
        if not torch.any(mask_0):
            return torch.tensor(0.0, device=logits1.device), 1.0

        logits1_masked = logits1[mask_0]
        theta = math.log(eps_prob/(1-eps_prob))
        loss = F.relu(logits1_masked - theta)
        ave_loss = torch.mean(loss)

        with torch.no_grad():
            volume_ratio = (logits1_masked < 0).float().mean().item()

        if printForTest:
            print(f"Operation volume loss: {ave_loss}")

        return ave_loss, volume_ratio
    
    def _get_density(self, checkTimes, time1, time2, probs1, probs2, logits1, logits2, rho_min=0.01):
        """
        Calculate density for each point at different time steps (differentiable version).
        
        Args:
            checkTimes: [Batch_size] time steps to check
            time1: [N_pts] first time value for each point
            time2: [N_pts] second time value for each point
            prob1: [N_pts] probability at time1 for each point
            prob2: [N_pts] probability at time2 for each point
        
        Returns:
            density: [N_pts, Batch_size] continuous density value in [0, 1] for each point at each time step
            
        A point is solid (density≈1) at time t if:
            1. time1 < t AND prob1 > 0.5
            2. time2 > t OR prob2 < 0.5
            
        Uses sigmoid functions to create smooth, differentiable approximations of conditions.
        """

        sharpness = self.densitySharpness

        # Reshape for broadcasting: [N_pts, 1] and [1, Batch_size]
        time1_expanded = time1.unsqueeze(1)
        time2_expanded = time2.unsqueeze(1)
        checkTimes_expanded = checkTimes.unsqueeze(0)


        logit_add = sharpness * (checkTimes_expanded - time1_expanded)        # t > t1
        logit_cut = sharpness * (time2_expanded - checkTimes_expanded)        # t < t2

        if self.logitDensity:
            logits1_expanded = logits1.unsqueeze(1)
            logits2_expanded = logits2.unsqueeze(1)
            logit_mask1 = sharpness * (logits1_expanded)    # p1 > 0.5
            logit_mask2 = sharpness * (- logits2_expanded)    # p2 < 0.5
        else:
            probs1_expanded = probs1.unsqueeze(1)
            probs2_expanded = probs2.unsqueeze(1)
            logit_mask1 = sharpness * (probs1_expanded - 0.5)    # p1 > 0.5
            logit_mask2 = sharpness * (0.5 - probs2_expanded)    # p2 < 0.5
        
        # Created = AddTime AND AddMask
        logits_created = torch.minimum(logit_add, logit_mask1)
        
        # Remain = Not(CutTime AND CutMask) = CutTime OR CutMask(Inverse)
        # 只要 t < t2 (logit_cut > 0) 或者 p2 < 0.5 (logit_mask2 > 0) 就是安全的
        logits_remain = torch.maximum(logit_cut, logit_mask2)

        logits_final = torch.minimum(logits_created, logits_remain)
        
        # Final = Created AND Remain
        rho_mfg = torch.sigmoid(logits_final)

        rho_sim = rho_mfg * (1 - rho_min) + rho_min
        
        return rho_sim
    
    def _loss_structure(self, model, positions, field1, field2, probs1, probs2, logits1, logits2, noise_scale = 0.05, printForTest = False):
        mask_AM = (probs1 > 0.5).squeeze()
        mask_SM = (probs2 > 0.5).squeeze()
        mask_SM = mask_AM & mask_SM

        if not torch.any(mask_AM) or not torch.any(mask_SM):
            return torch.tensor(0.0, device=positions.device), torch.tensor(0.0, device=positions.device)

        time1 = field1
        time2 = field1 + field2 * (self.max_time - field1)

        normalized_positions = (positions - self.aabb_min) / max(self.aabb_max - self.aabb_min)

        if self.use_sm_cutting_force_structure:
            with torch.no_grad():
                valid_indices = torch.nonzero(mask_SM, as_tuple=False).flatten()
                sampled_offsets = torch.randint(
                    0,
                    valid_indices.shape[0],
                    (self.strucLossCalculator.BATCH_SIZE,),
                    device=valid_indices.device
                )
                sampled_indices = valid_indices[sampled_offsets]
                check_times = time2[sampled_indices].squeeze()
                sm_force_points = normalized_positions[sampled_indices]

                two_tool_vec = model.forward(positions[sampled_indices], field_type='field3')
                sm_normal_vectors = two_tool_vec[..., :3]
        else:
            with torch.no_grad():
                # Concatenate times satisfying AM and SM masks into a valid set
                valid_time = (torch.cat([time1[mask_AM], time2[mask_SM]], dim=0)).flatten()
                # Randomly select indices with batch size
                indices = torch.randint(0, valid_time.shape[0], (self.strucLossCalculator.BATCH_SIZE,), device=valid_time.device)
                check_times = valid_time[indices]
                # Calculate the standard deviation to use for noise scaling
                std = valid_time.std()
                # Add noise scaled by noise_scale and std to check_times
                noise = torch.randn(self.strucLossCalculator.BATCH_SIZE, device=valid_time.device) * std * noise_scale
                check_times = check_times + noise
                # Clamp the check_times to [0, max_time] range
                check_times = torch.clamp(check_times, min=0.0, max=self.max_time)
                check_times = check_times.squeeze()
        
        density = self._get_density(check_times, time1.squeeze(), time2.squeeze(), probs1.squeeze(), probs2.squeeze(), logits1.squeeze(), logits2.squeeze())

        if self.use_sm_cutting_force_structure:
            structure_loss, simp_penalty = self.strucLossCalculator.structureLoss(
                samplePoints=normalized_positions,
                density=density,
                sm_force_points=sm_force_points,
                sm_normal_vectors=sm_normal_vectors
            )
        else:
            structure_loss, simp_penalty = self.strucLossCalculator.structureLoss(
                samplePoints=normalized_positions,
                density=density
            )

        if printForTest:
            print(f"Structure loss for real: {structure_loss}")
            print(f"Structure loss for simp penalty: {simp_penalty}")
        
        return structure_loss, simp_penalty


class MutualConsistencyLoss(torch.nn.Module):
    def __init__(self, sharpness=10.0, margin = 0.1):
        super().__init__()
        self.k = sharpness
        self.m = margin

    def forward(self, rho_proxy, t, t1, t2, p1, p2):
        """
        t1, t2: 时间场
        p1, p2: Mask 场 (0~1)
        rho_proxy: 目标密度 (0~1)
        """
        
        # =============================================================
        # 1. 物理场要求实心 (Target: Solid)
        # =============================================================
        # 违背 = (未增材) OR (已减材)
        
        # A. 未增材惩罚 (Not Added Penalty)
        # 想要实心，必须同时满足 t > t1 和 p1 > 0.5。
        # 缺任何一个都要惩罚，所以是求和 (AND logic for constraints)
        penalty_not_added = F.relu(t1 - t + self.m) + F.relu(0.6 - p1)
        
        # B. 已减材惩罚 (Removed Penalty)
        # 只有当 t > t2 且 p2 > 0.5 同时发生时，才算被切掉。
        # 只要破坏其中一个就不算被切，所以取最小值 (OR logic for escape)
        # 含义：如果 p2 很小，penalty就是0 (没切)；如果 t 很小，penalty也是0。
        penalty_removed = torch.min(F.relu(t - t2 + self.m), F.relu(p2 - 0.4))
        
        # 只要发生 A 或 B 任意一种情况，都是违背。
        violation_solid = penalty_not_added + penalty_removed
        
        # 加权：只在 Proxy 认为是实体的地方计算
        loss_solid = (violation_solid * rho_proxy).mean()
        
        # =============================================================
        # 2. 物理场要求空心 (Target: Void)
        # =============================================================
        # 违背 = 材料存在
        # 我们计算"逃离当前存在状态"的最短距离
        
        # 路线 1: 变成"未增材" (Make it Not Added)
        # 只要 t < t1 或 p1 < 0.5 即可。
        dist_to_not_added = torch.min(F.relu(t - t1 + self.m), F.relu(p1 - 0.4))
        
        # 路线 2: 变成"已减材" (Make it Removed)
        # 必须 t > t2 且 p2 > 0.5 同时满足。
        dist_to_removed = F.relu(t2 - t + self.m) + F.relu(0.6 - p2)
        
        # 取最短路径逃生
        dist_to_void = torch.min(dist_to_not_added, dist_to_removed)
        
        # 加权：只在 Proxy 认为是空的地方计算
        loss_void = (dist_to_void * (1.0 - rho_proxy)).mean()
        
        # =============================================================
        # 3. 概率对齐 (Probability Alignment)
        # =============================================================
        logit_add = self.k * (t - t1)        # t > t1
        logit_mask1 = self.k * (p1 - 0.5)    # p1 > 0.5
        logit_cut = self.k * (t2 - t)        # t < t2
        logit_mask2 = self.k * (0.5 - p2)    # p2 < 0.5
        
        # Created = AddTime AND AddMask
        logits_created = torch.minimum(logit_add, logit_mask1)
        
        # Remain = Not(CutTime AND CutMask) = CutTime OR CutMask(Inverse)
        # 只要 t < t2 (logit_cut > 0) 或者 p2 < 0.5 (logit_mask2 > 0) 就是安全的
        logits_remain = torch.maximum(logit_cut, logit_mask2)

        logits_final = torch.minimum(logits_created, logits_remain)
        
        # Final = Created AND Remain
        rho_mfg = torch.sigmoid(logits_final)
        
        loss_align = F.mse_loss(rho_mfg, rho_proxy)

        return 10.0 * (loss_solid + loss_void) + 1.0 * loss_align





        
