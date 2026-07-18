"""
Model Evaluation Script
=======================

Loads a trained MultiFieldModel from a .pth file and evaluates it over
multiple batches, reporting mean and std for every loss term and metric.

Usage:
    conda run -n myenv python evaluate_model.py \
        --model_path output/bracket_final_trained.pth \
        --model_name bracket \
        --batch_size 65536 \
        --n_batches 20 \
        --output_json output/eval_results.json
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from fieldopt.models.multi_field import MultiFieldModel
from fieldopt.losses.loss_multi_field import ComprehensiveLoss, _align_z_to_u_upper_half_robust
from fieldopt.geometry.implicit import StreamedGPUTrainingDataGenerator
from fieldopt.geometry.backend import add_geometry_backend_args, load_geometry_function
from fieldopt.geometry.voxel.voxelization import get_normalization_parameters

METRIC_NAMES = [
    'final_state_accuracy',
    'self_support_ratio',
    'AM_collision_free_ratio',
    'SM_collision_free_ratio',
    'operation_volume_ratio',
    'structure_normalized',
]

ANALYSE_NAMES = [
    'support_over_target',
    'extra_over_target',
    'mean_error',
    'max_error',
]


def build_model(cfg, spaceBox):
    model = MultiFieldModel(
        L1=cfg.L1, T1=cfg.T1, F1=cfg.F1, N_min1=cfg.N_min1, N_max1=cfg.N_max1,
        n_neurons1=cfg.N_NEURONS1, n_hidden_layers1=cfg.N_HIDDEN_LAYERS1,
        max_time=cfg.MAX_TIME,
        L2=cfg.L2, T2=cfg.T2, F2=cfg.F2, N_min2=cfg.N_min2, N_max2=cfg.N_max2,
        n_neurons2=cfg.N_NEURONS2, n_hidden_layers2=cfg.N_HIDDEN_LAYERS2,
        L3=cfg.L3, T3=cfg.T3, F3=cfg.F3, N_min3=cfg.N_min3, N_max3=cfg.N_max3,
        n_neurons3=cfg.N_NEURONS3, n_hidden_layers3=cfg.N_HIDDEN_LAYERS3,
        LM1=cfg.LM1, TM1=cfg.TM1, FM1=cfg.FM1, N_minM1=cfg.N_minM1, N_maxM1=cfg.N_maxM1,
        n_neuronsM1=cfg.N_NEURONSM1, n_hidden_layersM1=cfg.N_HIDDEN_LAYERSM1,
        LM2=cfg.LM2, TM2=cfg.TM2, FM2=cfg.FM2, N_minM2=cfg.N_minM2, N_maxM2=cfg.N_maxM2,
        n_neuronsM2=cfg.N_NEURONSM2, n_hidden_layersM2=cfg.N_HIDDEN_LAYERSM2,
        LDT=cfg.LDT, TDT=cfg.TDT, FDT=cfg.FDT, N_minDT=cfg.N_minDT, N_maxDT=cfg.N_maxDT,
        n_neuronsDT=cfg.N_NEURONSDT, n_hidden_layersDT=cfg.N_HIDDEN_LAYERSDT,
        bounding_box=spaceBox,
        device=cfg.DEVICE,
        dropout_rate_field1=cfg.TRAINING_CONFIG['field1_params']['dropout'],
        dropout_rate_field2=cfg.TRAINING_CONFIG['field2_params']['dropout'],
        dropout_rate_field3=cfg.TRAINING_CONFIG['field3_params']['dropout'],
        dropout_rate_fieldM1=cfg.TRAINING_CONFIG['fieldM1_params']['dropout'],
        dropout_rate_fieldM2=cfg.TRAINING_CONFIG['fieldM2_params']['dropout'],
        num_time_frequencies_DT=cfg.NUM_TIME_FREQUENCIES_DT,
        dropout_rate_fieldDT=cfg.TRAINING_CONFIG['fieldDT_params']['dropout'],
    ).to(cfg.DEVICE)
    return model


def _print_table(title, data_mean, data_std):
    """Print a formatted table of mean ± std values."""
    print(f"\n{title}")
    print("-" * 60)
    max_key_len = max(len(k) for k in data_mean.keys()) if data_mean else 20
    for key in data_mean:
        mean_val = data_mean[key]
        std_val = data_std[key]
        # Choose formatting based on magnitude
        if abs(mean_val) < 1e-4 or abs(mean_val) >= 1e4:
            fmt = f"{mean_val:.6e} ± {std_val:.2e}"
        else:
            fmt = f"{mean_val:.6f} ± {std_val:.6f}"
        print(f"  {key:<{max_key_len}} : {fmt}")
    print("-" * 60)


def _print_analyse_table(title, data_mean, data_std):
    """Print analysis metrics with mean ± std."""
    print(f"\n{title}")
    print("-" * 60)
    labels = {
        'support_over_target': 'support/target',
        'extra_over_target': 'extra/target',
        'mean_error': 'mean error',
        'max_error': 'max error',
    }
    ordered_keys = [name for name in ANALYSE_NAMES if name in data_mean]
    max_key_len = max(len(labels.get(k, k)) for k in ordered_keys) if ordered_keys else 20
    for key in ordered_keys:
        mean_val = data_mean[key]
        std_val = data_std[key]
        if abs(mean_val) < 1e-4 or abs(mean_val) >= 1e4:
            fmt = f"{mean_val:.6e} ± {std_val:.2e}"
        else:
            fmt = f"{mean_val:.6f} ± {std_val:.6f}"
        print(f"  {labels.get(key, key):<{max_key_len}} : {fmt}")
    print("-" * 60)


def _ratio_sm_not_cf_near_target1(sm_not_cf_pts, target1_pts, threshold, total_base_n):
    """Return (# sm_not_cf points with min-distance-to-target1 < threshold) / total_base_n."""
    if total_base_n <= 0 or sm_not_cf_pts.numel() == 0 or target1_pts.numel() == 0:
        return 0.0

    # Chunked cdist to avoid creating an overly large distance matrix at once.
    chunk_size = 4096
    near_count = 0
    for i in range(0, sm_not_cf_pts.shape[0], chunk_size):
        a = sm_not_cf_pts[i:i + chunk_size]
        d = torch.cdist(a, target1_pts, p=2)
        min_d = d.min(dim=1).values
        near_count += int((min_d < threshold).sum().item())

    return near_count / max(int(total_base_n), 1)


def _min_distance_to_set(src_pts, dst_pts, chunk_size=4096):
    """Return min distance from each src point to dst point set."""
    if src_pts.numel() == 0:
        return torch.empty((0,), device=src_pts.device, dtype=src_pts.dtype)
    if dst_pts.numel() == 0:
        return torch.full((src_pts.shape[0],), float('inf'), device=src_pts.device, dtype=src_pts.dtype)

    min_dists = []
    for i in range(0, src_pts.shape[0], chunk_size):
        d = torch.cdist(src_pts[i:i + chunk_size], dst_pts, p=2)
        min_dists.append(d.min(dim=1).values)
    return torch.cat(min_dists, dim=0)


def _estimate_surface_threshold(aabb_min, aabb_max, n_points):
    """Estimate a local point spacing and use it as surface threshold."""
    if n_points <= 0:
        return 0.0
    box_extent = aabb_max - aabb_min
    volume = float(torch.prod(box_extent).item())
    if volume <= 0.0:
        return 0.0
    mean_spacing = (volume / float(n_points)) ** (1.0 / 3.0)
    return 1.5 * mean_spacing


def evaluate(args):
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    try:
        cfg = importlib.import_module(f'configs.config_multi_field_{args.model_name}')
    except ImportError as e:
        print(f"Error: cannot load config for model '{args.model_name}': {e}")
        sys.exit(1)

    print(f"\n=== Evaluating model: {args.model_name} ===")
    print(f"  Model path : {args.model_path}")
    print(f"  Device     : {cfg.DEVICE}")

    # ------------------------------------------------------------------
    # 2. Compute bounding box (same as main_optimize.py)
    # ------------------------------------------------------------------
    stl_path = f'stlFiles/{args.model_name}.stl'
    if not os.path.exists(stl_path):
        print(f"Error: STL file not found at '{stl_path}'")
        sys.exit(1)

    _, p_min, spaceBox = get_normalization_parameters(stl_path)
    spaceBox[0] = spaceBox[0] - p_min
    spaceBox[1] = spaceBox[1] - p_min
    print(f"  Space box  : min={spaceBox[0]}, max={spaceBox[1]}")

    cfg_scale = getattr(cfg, 'SCALE', None)
    if cfg_scale is None:
        print("Error: cfg.SCALE is missing; cannot compute default distance threshold (5/scale).")
        sys.exit(1)

    SMTool = getattr(cfg, 'MANU_CONFIG', None)
    toolSize = SMTool['SMToolParas']['SMShankDiameter']

    if args.distance_threshold is None:
        distance_threshold = toolSize*1.5
        threshold_source = "default(5/cfg.SCALE)"
    else:
        distance_threshold = float(args.distance_threshold)
        threshold_source = "cli(--distance_threshold)"
    print(f"  Dist thres : {distance_threshold:.6e} [{threshold_source}]")

    analyse_probs1_threshold = float(args.analyse_probs1_threshold)
    print(f"  Analyse p1 : {analyse_probs1_threshold:.6f} [cli(--analyse_probs1_threshold)]")

    # ------------------------------------------------------------------
    # 3. Build and load model
    # ------------------------------------------------------------------
    model = build_model(cfg, spaceBox)
    if not os.path.exists(args.model_path):
        print(f"Error: model file not found at '{args.model_path}'")
        sys.exit(1)

    state_dict = torch.load(args.model_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.set_active_field('all')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    # ------------------------------------------------------------------
    # 4. Build or load shared H_gpu (SDF implicit function)
    # ------------------------------------------------------------------
    HRes = getattr(cfg, 'HRES', 512)
    H_gpu = load_geometry_function(
        backend=args.geometry_backend,
        stl_path=stl_path,
        model_name=args.model_name,
        artifact_path=args.geometry_artifact_path,
        device=cfg.DEVICE,
        voxel_resolution=HRes,
    )
    print("H_gpu ready.")

    # ------------------------------------------------------------------
    # 5. Data generator and loss function
    # ------------------------------------------------------------------
    effective_batch_size = args.batch_size or cfg.JOINT_BATCH_CONFIG['batch_size']
    surface_threshold_ref = _estimate_surface_threshold(
        torch.as_tensor(spaceBox[0], dtype=torch.float32),
        torch.as_tensor(spaceBox[1], dtype=torch.float32),
        effective_batch_size,
    )
    print(f"  Batch size : {effective_batch_size}")
    print(f"  N batches  : {args.n_batches}")
    print(f"  Surf thres : {surface_threshold_ref:.6e} [auto-estimated from base-point spacing]")

    data_generator = StreamedGPUTrainingDataGenerator(
        aabb_min=spaceBox[0],
        aabb_max=spaceBox[1],
        batch_size=effective_batch_size,
        device=cfg.DEVICE,
        MANU_CONFIG=cfg.MANU_CONFIG,
        MAX_RES_STRUCTURE=cfg.PARAS_STRUCTURE['max_resolution'],
        sample_each_grid=cfg.PARAS_STRUCTURE['sample_each_grid'],
    )

    loss_fn = ComprehensiveLoss(
        weights=cfg.WEIGHTS,
        m=cfg.MARGIN_NEW,
        h=cfg.MANU_CONFIG['hSupport'],
        SMToolParas=cfg.MANU_CONFIG['SMToolParas'],
        aabb_min=spaceBox[0],
        aabb_max=spaceBox[1],
        max_time=cfg.MAX_TIME,
        paras_structure=cfg.PARAS_STRUCTURE,
        device=cfg.DEVICE,
        check_func=H_gpu,
    )

    # ------------------------------------------------------------------
    # 6. Evaluation loop
    # ------------------------------------------------------------------
    all_loss_terms = list(cfg.TERM_SHORT_NAME.keys())
    accumulated_losses = {term: [] for term in all_loss_terms}
    accumulated_metrics = {name: [] for name in METRIC_NAMES}
    accumulated_analyse = {name: [] for name in ANALYSE_NAMES}

    # Track structure-loss zero-cause diagnostics across batches
    structure_zero_causes = {'mask_AM_empty': 0, 'mask_SM_empty': 0, 'fem_below_threshold': 0, 'computed': 0}

    print(f"\nRunning evaluation over {args.n_batches} batches...")

    eval_structure = cfg.WEIGHTS.get('structure', 0.0) > 0.0

    for batch_idx in tqdm(range(args.n_batches), desc="Evaluating"):
        # ---- main losses (no gradient needed) ----
        with torch.no_grad():
            batch_positions_inside, batch_mask, batch_N, batch_toolPoints_raw = next(data_generator)
            batch_targets_inside, _ = H_gpu(batch_positions_inside)

            (
                _loss_no_structure,
                loss_dict,
                metrics,
                _field1_vals,
                _field2_vals,
                _probs1_vals,
                _probs2_vals,
            ) = loss_fn(
                model,
                batch_positions_inside,
                batch_targets_inside,
                batch_N,
                batch_mask,
                batch_toolPoints_raw,
                printForTest=False,
            )

        for term in loss_dict:
            if term in accumulated_losses:
                accumulated_losses[term].append(float(loss_dict[term]))

        for name in metrics:
            if name in accumulated_metrics:
                accumulated_metrics[name].append(float(metrics[name]))

        # Compute analysis items directly in this script.
        with torch.no_grad():
            N = batch_N[0]
            NS = batch_N[1]
            NAMTool = batch_N[2]
            probs1_base = _probs1_vals[:N].squeeze()
            probs2_base = _probs2_vals[:N].squeeze()
            targets_base = batch_targets_inside[:N].squeeze()
            mask_void  = (targets_base == 0)
            mask_solid = (targets_base == 1)
            n_solid  = max(int(mask_solid.sum().item()), 1)
            n_total_base = N

            # per-point AM collision-free：使用各 AMTool 采样点的 logit/time 判断
            # per-point self-support   ：使用各 S    采样点的 logit/time 判断
            _eps = 1e-6
            _max_time = loss_fn.max_time
            f1_base_sq = _field1_vals[:N].squeeze()          # (N,)
            tP_col = f1_base_sq.unsqueeze(1)                 # (N, 1) — base 点打印时间

            # AM collision-free per-point
            if NAMTool > 0 and NAMTool % N == 0:
                _nAM = NAMTool // N
                _f1AM = _field1_vals[N+NS : N+NS+NAMTool].reshape(N, _nAM)    # (N, K)
                _f2AM = _field2_vals[N+NS : N+NS+NAMTool].reshape(N, _nAM)
                _l1AM = torch.logit(_probs1_vals[N+NS : N+NS+NAMTool].reshape(N, _nAM).clamp(_eps, 1-_eps))
                _l2AM = torch.logit(_probs2_vals[N+NS : N+NS+NAMTool].reshape(N, _nAM).clamp(_eps, 1-_eps))
                _t2AM = _f1AM + _f2AM * (_max_time - _f1AM)                   # (N, K)
                _rc = torch.minimum(tP_col - _f1AM, _l1AM)
                _rr = torch.maximum(_t2AM - tP_col, -_l2AM)
                _rw, _ = torch.max(torch.minimum(_rc, _rr), dim=1)            # (N,)
                cf_per_pt = (_rw <= 0)                                         # True = collision-free
            else:
                cf_per_pt = torch.ones(N, dtype=torch.bool, device=probs1_base.device)

            # self-support per-point
            if NS > 0 and NS % N == 0:
                _nS = NS // N
                _f1S = _field1_vals[N : N+NS].reshape(N, _nS)                 # (N, K)
                _f2S = _field2_vals[N : N+NS].reshape(N, _nS)
                _l1S = torch.logit(_probs1_vals[N : N+NS].reshape(N, _nS).clamp(_eps, 1-_eps))
                _l2S = torch.logit(_probs2_vals[N : N+NS].reshape(N, _nS).clamp(_eps, 1-_eps))
                _t2S = _f1S + _f2S * (_max_time - _f1S)                       # (N, K)
                _rc_ss = torch.minimum(tP_col - _f1S, _l1S)
                _rr_ss = torch.maximum(_t2S - tP_col, -_l2S)
                _rb, _ = torch.max(torch.minimum(_rc_ss, _rr_ss), dim=1)      # (N,)
                ss_per_pt = (_rb > 0)                                          # True = self-supported
            else:
                ss_per_pt = torch.ones(N, dtype=torch.bool, device=probs1_base.device)
            _f1_base = _field1_vals[:N].squeeze()          # (N,)
            _f2_base = _field2_vals[:N].squeeze()          # (N,)
            _time2_base = _f1_base + _f2_base * (_max_time - _f1_base)  # (N,)
            _base_pts = batch_positions_inside[:N]         # (N, 3)
            target1_pts = _base_pts[mask_solid]            # (T, 3)
            mask_probs1_support = probs1_base > analyse_probs1_threshold

            mask_sm_candidate = mask_void & (probs1_base > 0.5)
            n_sm_cand = int(mask_sm_candidate.sum().item())
            sm_cf_per_pt = torch.ones(N, dtype=torch.bool, device=probs1_base.device)
            near_target1_per_pt = torch.zeros(N, dtype=torch.bool, device=probs1_base.device)

            if n_sm_cand > 0:
                _time2_sm = _time2_base[mask_sm_candidate]      # (M,)
                _base_pts_sm = _base_pts[mask_sm_candidate]     # (M, 3)

                # 计算 SM 工具几何（field3：法向量 + 工具方向）
                _two_tv = model.forward(_base_pts_sm, field_type='field3')  # (M, 6)
                _norm_v = _two_tv[..., :3]                     # (M, 3)
                _tool_v = _two_tv[..., 3:]                     # (M, 3)
                _ball_c = _base_pts_sm + _norm_v * loss_fn.SMTipDiameter / 2.0  # (M, 3)

                _tp_raw_sm = batch_toolPoints_raw[mask_sm_candidate]  # (M, nTool, 3)
                _Rot = _align_z_to_u_upper_half_robust(_tool_v)       # (M, 3, 3)
                _rot_tp = torch.einsum('ikl,ijl->ijk', _Rot, _tp_raw_sm)  # (M, n, 3)
                _tp_all = _rot_tp + _ball_c.unsqueeze(1)              # (M, n, 3)

                _tp_flat = _tp_all.reshape(-1, 3)                     # (M*n, 3)
                _m_in = ((_tp_flat >= loss_fn.aabb_min) & (_tp_flat <= loss_fn.aabb_max)).all(dim=-1)

                _isin, _ = loss_fn.check_func(_tp_flat[_m_in])
                _f1t, _f2t, _l1t_raw, _l2t_raw = model.forward(_tp_flat[_m_in], field_type='timesAndMasks')
                _l1t = torch.where(_isin == 1, torch.tensor(100.0, device=_tp_flat.device), _l1t_raw)
                _l2t = torch.where(_isin == 1, torch.tensor(-100.0, device=_tp_flat.device), _l2t_raw)

                def _fill_reshape(data, mask, fill_val, n_pts, n_samples):
                    full = torch.full((n_pts * n_samples, 1), fill_val,
                                     device=_tp_flat.device, dtype=data.dtype)
                    full = full.masked_scatter(mask.unsqueeze(-1), data)
                    return full.reshape(n_pts, n_samples)

                _n_tp = _tp_all.shape[1]
                _t1S = _fill_reshape(_f1t,                            _m_in, 65500,   n_sm_cand, _n_tp)
                _t2S = _fill_reshape(_f1t + _f2t * (_max_time - _f1t), _m_in, -65500,  n_sm_cand, _n_tp)
                _l1S = _fill_reshape(_l1t,                            _m_in, -100.0,  n_sm_cand, _n_tp)
                _l2S = _fill_reshape(_l2t,                            _m_in, -100.0,  n_sm_cand, _n_tp)

                _t2_col = _time2_sm.unsqueeze(1)
                _r_pres = torch.minimum(_l1S, _t2_col - _t1S)
                _r_stil = torch.maximum(_t2S - _t2_col, -_l2S)
                _r_wst, _ = torch.max(torch.minimum(_r_pres, _r_stil), dim=1)
                _sm_cf_candidate = (_r_wst <= 0)
                sm_cf_per_pt[mask_sm_candidate] = _sm_cf_candidate

                if target1_pts.numel() > 0:
                    near_mask_candidate = torch.zeros(_base_pts_sm.shape[0], dtype=torch.bool, device=probs1_base.device)
                    chunk_size = 4096
                    for i in range(0, _base_pts_sm.shape[0], chunk_size):
                        d = torch.cdist(_base_pts_sm[i:i + chunk_size], target1_pts, p=2)
                        near_mask_candidate[i:i + chunk_size] = d.min(dim=1).values < distance_threshold
                    near_target1_per_pt[mask_sm_candidate] = near_mask_candidate

            mask_support = mask_void & mask_probs1_support & cf_per_pt & ss_per_pt
            support_ratio_val = mask_support.float().sum().item() / n_solid

            mask_extra_candidate = mask_void & (probs1_base > 0.5) & cf_per_pt & near_target1_per_pt
            mask_extra = mask_extra_candidate & ((probs2_base < 0.5) | (~sm_cf_per_pt))
            extra_ratio_val = mask_extra.float().sum().item() / n_solid

            mask_model_a = mask_solid | mask_extra
            model_a_pts = _base_pts[mask_model_a]
            model_a_non_pts = _base_pts[~mask_model_a]

            surface_threshold = _estimate_surface_threshold(loss_fn.aabb_min, loss_fn.aabb_max, n_total_base)

            if model_a_pts.numel() > 0 and model_a_non_pts.numel() > 0:
                dist_to_non_a = _min_distance_to_set(model_a_pts, model_a_non_pts)
                mask_a_surface = dist_to_non_a < surface_threshold
                a_surface_pts = model_a_pts[mask_a_surface]
            else:
                a_surface_pts = model_a_pts

            if a_surface_pts.numel() > 0 and target1_pts.numel() > 0:
                min_dists = _min_distance_to_set(a_surface_pts, target1_pts)
                min_dists = min_dists * float(cfg.SCALE)
                mean_error_val = float(min_dists.mean().item())
                max_error_val = float(min_dists.max().item())
            else:
                mean_error_val = 0.0
                max_error_val = 0.0

            accumulated_analyse['support_over_target'].append(support_ratio_val)
            accumulated_analyse['extra_over_target'].append(extra_ratio_val)
            accumulated_analyse['mean_error'].append(mean_error_val)
            accumulated_analyse['max_error'].append(max_error_val)

        # ---- structure loss: must be computed OUTSIDE torch.no_grad() ----
        # Reason: StructureFunction.apply() is a custom torch.autograd.Function.
        # Under torch.no_grad() with requires_grad=False inputs, certain PyTorch
        # versions may skip the custom forward path. Computing outside no_grad()
        # guarantees the FEM forward pass runs correctly.
        if eval_structure:
            grid_points = data_generator.sample_grid_points()
            _debug = (batch_idx == 0)  # print internals only on first batch
            l_structure, l_structure_new, l_simp_penalty, structure_norm = loss_fn.get_loss_structure(
                model, grid_points, printForTest=_debug
            )
            l_structure_val = float(l_structure)
            l_simp_val = float(l_simp_penalty)

            # ---- diagnose why structure loss is 0 ----
            if l_structure_val == 0.0:
                with torch.no_grad():
                    is_inSide, _ = loss_fn.check_func(grid_points)
                    f1, f2, l1_raw, l2_raw = model.forward(grid_points, field_type='timesAndMasks')
                    l1 = torch.where(is_inSide == 1, torch.tensor(100.0, device=grid_points.device), l1_raw)
                    l2 = torch.where(is_inSide == 1, torch.tensor(-100.0, device=grid_points.device), l2_raw)
                    p1 = torch.sigmoid(l1).squeeze()
                    p2 = torch.sigmoid(l2).squeeze()
                    mask_am = p1 > 0.5
                    mask_sm = (p2 > 0.5) & mask_am
                    if not torch.any(mask_am):
                        structure_zero_causes['mask_AM_empty'] += 1
                    elif not torch.any(mask_sm):
                        structure_zero_causes['mask_SM_empty'] += 1
                    else:
                        structure_zero_causes['fem_below_threshold'] += 1
                if _debug:
                    with torch.no_grad():
                        n_am = int(mask_am.sum().item())
                        n_sm = int(mask_sm.sum().item())
                        n_total = grid_points.shape[0]
                    print(f"\n[DEBUG batch 0] grid_points={n_total}, mask_AM={n_am}, mask_SM(AM&SM)={n_sm}")
                    if n_am > 0 and n_sm == 0:
                        print("  -> mask_SM is EMPTY: no grid point satisfies probs2 > 0.5 AND probs1 > 0.5")
                        print("     inside-shape points are forced logits2=-100 (probs2≈0, never SM).")
                        print("     Only outside-shape stock points with model-predicted probs2>0.5 count.")
                    elif n_am == 0:
                        print("  -> mask_AM is EMPTY: no grid point satisfies probs1 > 0.5")
                    else:
                        print("  -> FEM displacement is below max_displacement threshold everywhere (structure OK)")
            else:
                structure_zero_causes['computed'] += 1

            accumulated_losses['structure'].append(l_structure_val)
            accumulated_losses['structure_simp_penalty'].append(l_simp_val)
            accumulated_metrics['structure_normalized'].append(float(structure_norm))
        else:
            accumulated_losses['structure'].append(0.0)
            accumulated_losses['structure_simp_penalty'].append(0.0)
            accumulated_metrics['structure_normalized'].append(0.0)

    # Print structure-loss diagnosis summary
    if eval_structure:
        total_zero = sum(v for k, v in structure_zero_causes.items() if k != 'computed')
        if total_zero > 0:
            print(f"\n[Structure Loss Diagnosis] Over {args.n_batches} batches, "
                  f"structure loss was 0 in {total_zero} batch(es):")
            if structure_zero_causes['mask_AM_empty'] > 0:
                print(f"  - mask_AM empty (no AM points in grid): {structure_zero_causes['mask_AM_empty']} batch(es)")
            if structure_zero_causes['mask_SM_empty'] > 0:
                print(f"  - mask_SM empty (no SM points in grid): {structure_zero_causes['mask_SM_empty']} batch(es)")
                print("    Cause: inside-shape points are forced probs2≈0; only outside-shape stock material")
                print("    with model-predicted probs2>0.5 contributes. If stock zone is small or model")
                print("    has not learned SM assignment, mask_SM will be empty → structure loss=0.")
            if structure_zero_causes['fem_below_threshold'] > 0:
                print(f"  - FEM displacement below threshold (structurally sound): "
                      f"{structure_zero_causes['fem_below_threshold']} batch(es)")
        if structure_zero_causes['computed'] > 0:
            print(f"[Structure Loss] Non-zero in {structure_zero_causes['computed']} / {args.n_batches} batches.")

    # ------------------------------------------------------------------
    # 7. Aggregate results
    # ------------------------------------------------------------------
    loss_mean = {}
    loss_std = {}
    for term in all_loss_terms:
        vals = accumulated_losses[term]
        loss_mean[term] = float(np.mean(vals)) if vals else float('nan')
        loss_std[term] = float(np.std(vals)) if vals else float('nan')

    metric_mean = {}
    metric_std = {}
    for name in METRIC_NAMES:
        vals = accumulated_metrics[name]
        metric_mean[name] = float(np.mean(vals)) if vals else float('nan')
        metric_std[name] = float(np.std(vals)) if vals else float('nan')

    # ------------------------------------------------------------------
    # 8. Print results
    # ------------------------------------------------------------------
    n = args.n_batches
    _print_table(
        f"=== Loss Terms (mean ± std over {n} batches) ===",
        loss_mean,
        loss_std,
    )
    _print_table(
        f"=== Metrics   (mean ± std over {n} batches) ===",
        metric_mean,
        metric_std,
    )

    analyse_mean = {}
    analyse_std = {}
    for name in ANALYSE_NAMES:
        vals = accumulated_analyse[name]
        analyse_mean[name] = float(np.mean(vals)) if vals else float('nan')
        analyse_std[name] = float(np.std(vals)) if vals else float('nan')

    _print_analyse_table(
        f"=== Analyse   (mean ± std over {n} batches) ===",
        analyse_mean,
        analyse_std,
    )

    # Summary against BAYESIAN_METRIC_TARGETS if available
    metric_targets = getattr(cfg, 'BAYESIAN_METRIC_TARGETS', None)
    if metric_targets is not None:
        print(f"\n=== Metric vs Target ===")
        print("-" * 70)
        for name, (target, direction) in metric_targets.items():
            actual = metric_mean.get(name, float('nan'))
            if direction == 'higher':
                status = "OK" if actual >= target else "FAIL"
                ratio = actual / target if target > 0 else float('nan')
            else:
                status = "OK" if actual <= target else "FAIL"
                ratio = actual / target if target > 0 else float('nan')
            dir_tag = "higher" if direction == "higher" else "lower"
            print(
                f"  [{status:4s}] {name:<30s}: {actual:.6e}  "
                f"(target={target}, {dir_tag} is better, ratio={ratio:.3f})"
            )
        print("-" * 70)

    # ------------------------------------------------------------------
    # 9. Save to JSON (optional)
    # ------------------------------------------------------------------
    results = {
        'model_path': args.model_path,
        'model_name': args.model_name,
        'batch_size': effective_batch_size,
        'n_batches': args.n_batches,
        'loss_mean': loss_mean,
        'loss_std': loss_std,
        'metric_mean': metric_mean,
        'metric_std': metric_std,
        'analyse_mean': analyse_mean,
        'analyse_std': analyse_std,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MultiFieldModel: report per-batch loss and metric statistics."
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help="Path to the .pth model file (e.g. output/bracket_final_trained.pth)"
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        help="Model/config name (e.g. bracket, GLman). Used to load configs.config_multi_field_{name}"
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help="Batch size for each evaluation step. Defaults to cfg.JOINT_BATCH_CONFIG['batch_size']"
    )
    parser.add_argument(
        '--n_batches', type=int, default=20,
        help="Number of evaluation batches (default: 20)"
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help="Optional path to save results as JSON (e.g. output/eval_results.json)"
    )
    parser.add_argument(
        '--distance_threshold', type=float, default=None,
        help="Distance threshold in normalized coordinates. Default: 5/scale"
    )
    parser.add_argument(
        '--analyse_probs1_threshold', type=float, default=0.5,
        help="Threshold used only for the support/target analysis metric."
    )
    add_geometry_backend_args(parser)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
