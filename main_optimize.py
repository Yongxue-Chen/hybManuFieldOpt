from math import e
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from fieldopt.utils.data_loader import read_initial_field
from fieldopt.models.multi_field import MultiFieldModel
from fieldopt.losses.loss_multi_field import ComprehensiveLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from fieldopt.utils.early_stopping import EarlyStoppingMonitor
from fieldopt.geometry.implicit import StreamedGPUTrainingDataGenerator
import argparse
import sys
from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
from fieldopt.geometry.backend import add_geometry_backend_args, load_geometry_function

import importlib

MODEL_NAME = 'bracket'
DEFAULT_RESOLUTION = 100
BATCH_SIZE = None

def main(weights_dict=None, quiet=False, model_name=None, shared_H_gpu=None,
         shared_struc_loss_calculator=None, joint_batch_size=None):
    """
    Main training function.
    
    Args:
        weights_dict: If provided, override cfg.WEIGHTS with these values and
                      run quietly for Bayesian optimizer trials.
        quiet: If True, suppress console output (tqdm, prints). Used by
               Bayesian optimizer to reduce noise.
        model_name: If provided, use this model name to load the config module
                    (e.g. 'TPMS50', 'GLman'). Defaults to module-level MODEL_NAME.
    
    Returns:
        (loss_dict, metrics_dict, model_state_dict) tuple:
        - loss_dict: {term_name: float} for each loss term at the best epoch
        - metrics_dict: {metric_name: float} physical metrics at the best epoch
        - model_state_dict: best model state from early stopping
        When called normally (not from optimizer), this return value is ignored.
    """
    import builtins
    import copy as _copy
    _original_print = builtins.print
    if quiet:
        # Suppress print output in quiet mode
        builtins.print = lambda *a, **k: None

    parser = argparse.ArgumentParser(description="Train or fine-tune a multi field model from scratch.")
    parser.add_argument(
        '--tmp-model-path',
        '--tmp_model_path',
        dest='tmp_model_path',
        type=str,
        default=None,
        help=(
            "Temporary checkpoint path. Defaults to the final output directory "
            "with filename <model>_tmp.pth."
        ),
    )
    parser.add_argument(
        '--weights-json',
        '--weights_json',
        dest='weights_json',
        type=str,
        default=None,
        help=(
            "Optional JSON file containing fixed WEIGHTS overrides. "
            "Unspecified terms keep their config values."
        ),
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help="Model name to load config (e.g. 'TPMS50', 'GLman'). Overrides module-level MODEL_NAME."
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=DEFAULT_RESOLUTION,
        help=(
            "Resolution used in the default pretrained-checkpoint filename "
            f"(default: {DEFAULT_RESOLUTION})."
        ),
    )
    parser.add_argument(
        '--pretrain-model-path',
        '--pretrain_model_path',
        dest='pretrain_model_path',
        default=None,
        help=(
            "Pretrained checkpoint. Defaults to "
            "model_data/pretrained_fields/<model>/<model>_pretrained_<resolution>.pth."
        ),
    )
    parser.add_argument(
        '--stl-path',
        '--stl_path',
        dest='stl_path',
        default=None,
        help=(
            "Preprocessed body-and-support STL. Defaults to "
            "model_data/preprocessed/<model>/<model>_support.stl."
        ),
    )
    parser.add_argument(
        '--output-path',
        '--output_path',
        dest='output_path',
        default=None,
        help=(
            "Final trained checkpoint. Defaults to "
            "model_data/trained_fields/<model>/<model>_final_trained.pth."
        ),
    )
    parser.add_argument(
        '--joint-epochs',
        '--joint_epochs',
        dest='joint_epochs',
        type=int,
        default=None,
        help="Optional override for JOINT_VIRTUAL_EPOCHS.",
    )
    parser.add_argument(
        '--steps-per-epoch',
        '--steps_per_epoch',
        dest='steps_per_epoch',
        type=int,
        default=None,
        help="Optional override for JOINT_BATCH_CONFIG['steps_per_epoch'].",
    )
    parser.add_argument(
        '--batch-size',
        '--batch_size',
        dest='batch_size',
        type=int,
        default=None,
        help="Optional joint-training batch-size override.",
    )
    add_geometry_backend_args(parser)
    
    args = parser.parse_args()

    # Determine model name and load config
    effective_model_name = model_name or args.model_name or MODEL_NAME
    cfg = importlib.import_module(f'configs.config_multi_field_{effective_model_name}')
    resolution = args.resolution
    if resolution <= 0:
        parser.error("--resolution must be a positive integer")
    for option, value in (
        ('--joint-epochs', args.joint_epochs),
        ('--steps-per-epoch', args.steps_per_epoch),
        ('--batch-size', args.batch_size),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be a positive integer")

    pretrain_model_path = args.pretrain_model_path or os.path.join(
        'model_data',
        'pretrained_fields',
        effective_model_name,
        f'{effective_model_name}_pretrained_{resolution}.pth',
    )
    stl_path = args.stl_path or os.path.join(
        'model_data',
        'preprocessed',
        effective_model_name,
        f'{effective_model_name}_support.stl',
    )
    joint_model_path = args.output_path or os.path.join(
        'model_data',
        'trained_fields',
        effective_model_name,
        f'{effective_model_name}_final_trained.pth',
    )
    tmp_model_path = args.tmp_model_path or os.path.join(
        os.path.dirname(joint_model_path),
        f'{effective_model_name}_tmp.pth',
    )

    for label, path in (
        ('pretrained checkpoint', pretrain_model_path),
        ('preprocessed STL', stl_path),
    ):
        if not os.path.isfile(path):
            parser.error(f"{label} not found: {path}")

    os.makedirs(os.path.dirname(os.path.abspath(joint_model_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(tmp_model_path)), exist_ok=True)

    effective_joint_epochs = args.joint_epochs or cfg.JOINT_VIRTUAL_EPOCHS
    effective_steps_per_epoch = (
        args.steps_per_epoch
        or cfg.JOINT_BATCH_CONFIG.get('steps_per_epoch', 100)
    )
    effective_batch_size = (
        args.batch_size
        or joint_batch_size
        or BATCH_SIZE
        or cfg.JOINT_BATCH_CONFIG['batch_size']
    )

    # Override weights from Bayesian optimizer (in-process call)
    cfg.WEIGHTS = _copy.copy(cfg.WEIGHTS)
    if weights_dict is not None:
        for _k, _v in weights_dict.items():
            cfg.WEIGHTS[_k] = _v
    # Override weights from JSON file (subprocess call)
    elif args.weights_json:
        import json as _json
        if not os.path.isfile(args.weights_json):
            parser.error(f"weights JSON not found: {args.weights_json}")
        with open(args.weights_json) as _f:
            _override_weights = _json.load(_f)
        unknown_weights = sorted(set(_override_weights) - set(cfg.WEIGHTS))
        if unknown_weights:
            parser.error(
                "weights JSON contains unknown terms: "
                + ", ".join(unknown_weights)
            )
        for _k, _v in _override_weights.items():
            cfg.WEIGHTS[_k] = _v

    experiment = None

    print("Optimization inputs:")
    print(f"  model name            : {effective_model_name}")
    print(f"  pretrained checkpoint : {pretrain_model_path}")
    print(f"  support STL           : {stl_path}")
    print(f"  geometry backend      : {args.geometry_backend}")
    print(
        "  geometry artifact     : "
        f"{args.geometry_artifact_path or '[backend default]'}"
    )
    print("Optimization outputs:")
    print(f"  temporary checkpoint  : {tmp_model_path}")
    print(f"  final checkpoint      : {joint_model_path}")
    print("Fixed optimization weights:")
    for _name, _value in cfg.WEIGHTS.items():
        print(f"  {_name}: {_value}")

    # --- Fix random seed for reproducibility ---
    if weights_dict is not None:
        # Bayesian mode: deterministic seed from weights so same config = same result
        _seed = hash(tuple(sorted(weights_dict.items()))) % (2**32)
    else:
        _seed = 42
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    # aabb_min, aabb_max = get_sdf_aabb(stl_path)
    # spaceBox = np.array([aabb_min, aabb_max])
    _, p_min, spaceBox = get_normalization_parameters(stl_path)
    spaceBox[0] = spaceBox[0] - p_min
    spaceBox[1] = spaceBox[1] - p_min

    print("\n--- Space Box ---")
    print(f"Min bounds: [{spaceBox[0][0]:.3f}, {spaceBox[0][1]:.3f}, {spaceBox[0][2]:.3f}]")
    print(f"Max bounds: [{spaceBox[1][0]:.3f}, {spaceBox[1][1]:.3f}, {spaceBox[1][2]:.3f}]")
    print("-----------------------\n")
    
    print(f"\nUsing device: {cfg.DEVICE}")
    model = MultiFieldModel(
        # Parameters for the first field
        L1=cfg.L1, T1=cfg.T1, F1=cfg.F1, N_min1=cfg.N_min1, N_max1=cfg.N_max1,
        n_neurons1=cfg.N_NEURONS1, n_hidden_layers1=cfg.N_HIDDEN_LAYERS1, max_time=cfg.MAX_TIME,
        # Parameters for the second field
        L2=cfg.L2, T2=cfg.T2, F2=cfg.F2, N_min2=cfg.N_min2, N_max2=cfg.N_max2,
        n_neurons2=cfg.N_NEURONS2, n_hidden_layers2=cfg.N_HIDDEN_LAYERS2,
        # Parameters for the third field
        L3=cfg.L3, T3=cfg.T3, F3=cfg.F3, N_min3=cfg.N_min3, N_max3=cfg.N_max3,
        n_neurons3=cfg.N_NEURONS3, n_hidden_layers3=cfg.N_HIDDEN_LAYERS3,
        # Parameters for the mask field
        LM1=cfg.LM1, TM1=cfg.TM1, FM1=cfg.FM1, N_minM1=cfg.N_minM1, N_maxM1=cfg.N_maxM1,
        n_neuronsM1=cfg.N_NEURONSM1, n_hidden_layersM1=cfg.N_HIDDEN_LAYERSM1,
        LM2=cfg.LM2, TM2=cfg.TM2, FM2=cfg.FM2, N_minM2=cfg.N_minM2, N_maxM2=cfg.N_maxM2,
        n_neuronsM2=cfg.N_NEURONSM2, n_hidden_layersM2=cfg.N_HIDDEN_LAYERSM2,

        # Parameters for the spatio-temporal density field
        LDT=cfg.LDT, TDT=cfg.TDT, FDT=cfg.FDT, N_minDT=cfg.N_minDT, N_maxDT=cfg.N_maxDT,
        n_neuronsDT=cfg.N_NEURONSDT, n_hidden_layersDT=cfg.N_HIDDEN_LAYERSDT,
        
        bounding_box=spaceBox,
        device=cfg.DEVICE,
        dropout_rate_field1=cfg.TRAINING_CONFIG['field1_params']['dropout'],  # Use field-specific dropout
        dropout_rate_field2=cfg.TRAINING_CONFIG['field2_params']['dropout'],   # Use field-specific dropout
        dropout_rate_field3=cfg.TRAINING_CONFIG['field3_params']['dropout'],
        dropout_rate_fieldM1=cfg.TRAINING_CONFIG['fieldM1_params']['dropout'],
        dropout_rate_fieldM2=cfg.TRAINING_CONFIG['fieldM2_params']['dropout'],
        num_time_frequencies_DT=cfg.NUM_TIME_FREQUENCIES_DT,
        dropout_rate_fieldDT=cfg.TRAINING_CONFIG['fieldDT_params']['dropout']
    ).to(cfg.DEVICE)

    # tmpPath = f'output/boneTPMS_final_trained-3158.pth'
    # model.load_state_dict(torch.load(tmpPath, map_location=cfg.DEVICE), strict=False)
    # LR = 1e-5

    load_result = model.load_state_dict(
        torch.load(
            pretrain_model_path,
            map_location=cfg.DEVICE,
            weights_only=True,
        ),
        strict=False,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Pretrained checkpoint does not match the current model: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    # model.load_state_dict(torch.load(args.tmp_model_path, map_location=cfg.DEVICE))
    # model.load_state_dict(torch.load(joint_model_path, map_location=cfg.DEVICE))
    LR = cfg.JOINT_TRAIN_LR

    torch.set_float32_matmul_precision('high')
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params:,}")

    # --- Joint training ---
    print("\n" + "="*50)
    print("All fields loaded, starting joint training...")

    if shared_H_gpu is not None:
        H_gpu = shared_H_gpu
    else:
        HRes = getattr(cfg, 'HRES', 512)
        H_gpu = load_geometry_function(
            backend=args.geometry_backend,
            stl_path=stl_path,
            model_name=effective_model_name,
            artifact_path=args.geometry_artifact_path,
            device=cfg.DEVICE,
            voxel_resolution=HRes,
        )
        print("[main] H_gpu ready.")

    data_generator = StreamedGPUTrainingDataGenerator(
        aabb_min=spaceBox[0],
        aabb_max=spaceBox[1],
        batch_size=effective_batch_size,
        device=cfg.DEVICE,
        MANU_CONFIG=cfg.MANU_CONFIG,
        MAX_RES_STRUCTURE=cfg.PARAS_STRUCTURE['max_resolution'],
        sample_each_grid=cfg.PARAS_STRUCTURE['sample_each_grid']
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
        check_func = H_gpu,
        struc_loss_calculator=shared_struc_loss_calculator
    )

    # Set up optimizer for joint training with field-specific learning rates
    model.set_active_field('all')

    # Create parameter groups with different learning rates for field3
    field1_params = list(model.field1.parameters())
    field2_params = list(model.field2.parameters()) 
    field3_params = list(model.field3.parameters())
    fieldM1_params = list(model.fieldM1.parameters())
    fieldM2_params = list(model.fieldM2.parameters())

    # Use different learning rates: field3 gets a smaller learning rate
    field3_lr_ratio = cfg.FIELD3_LR_RATIO
    optimizer = torch.optim.AdamW([
        {'params': field1_params, 'lr': LR, 'weight_decay': cfg.WEIGHT_DECAY},
        {'params': field2_params, 'lr': LR, 'weight_decay': cfg.WEIGHT_DECAY},
        {'params': field3_params, 'lr': LR * field3_lr_ratio, 'weight_decay': cfg.WEIGHT_DECAY},
        {'params': fieldM1_params, 'lr': LR, 'weight_decay': cfg.WEIGHT_DECAY},
        {'params': fieldM2_params, 'lr': LR, 'weight_decay': cfg.WEIGHT_DECAY}
    ], betas=cfg.ADAMW_BETAS, eps=cfg.ADAMW_EPS)

    # Use ReduceLROnPlateau scheduler based on loss reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # minimize loss
        factor=cfg.REDUCE_LRON_PLATEAU_CONFIG['factor'],
        patience=cfg.REDUCE_LRON_PLATEAU_CONFIG['patience'],
        min_lr=cfg.REDUCE_LRON_PLATEAU_CONFIG['min_lr'],
        threshold=cfg.REDUCE_LRON_PLATEAU_CONFIG['threshold'],
        threshold_mode=cfg.REDUCE_LRON_PLATEAU_CONFIG['threshold_mode']  # relative threshold
    )
    
    scaler = GradScaler(device='cuda', enabled=cfg.DEVICE.startswith('cuda'))
    
    # Early stopping mechanism (enhanced)
    early_stopping = EarlyStoppingMonitor(cfg.JOINT_EARLY_STOPPING)
    
    # Joint training loop with improvements
    print("Starting improved joint training...")
    
    # train_losses = []
    epoch_losses = []  # 新增：存储每个epoch的平均loss

    accumulation_steps = cfg.JOINT_BATCH_CONFIG['accumulation_steps']
    
    # 计算每个epoch的steps数量
    # 可以根据数据量和batch size来设置，这里设置为一个合理的默认值
    steps_per_epoch = effective_steps_per_epoch
    total_steps = effective_joint_epochs * steps_per_epoch
    
    print(f"Training configuration:")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Total epochs: {effective_joint_epochs}")
    print(f"- Total training steps: {total_steps}")
    print(f"- Accumulation steps: {accumulation_steps}")

    
    pbar = tqdm(range(effective_joint_epochs), desc="Joint training", disable=quiet)
    global_step = 0
    
    for epoch in pbar:
        model.train()
        epoch_step_losses = []  # 当前epoch内的所有step losses

        epoch_step_each_loss = {term: [] for term in cfg.TERM_SHORT_NAME}

        METRIC_NAMES = ['final_state_accuracy', 'self_support_ratio', 'AM_collision_free_ratio',
                        'SM_collision_free_ratio', 'operation_volume_ratio', 'structure_normalized']
        epoch_step_each_metric = {name: [] for name in METRIC_NAMES}

        epoch_solid_time1_count = 0
        epoch_total_elements = 0

        # 每个epoch进行多个训练步骤
        for step_in_epoch in range(steps_per_epoch):
            if step_in_epoch % 5 == 0:
                batch_positions_inside, batch_mask, batch_N, batch_toolPoints_raw = next(data_generator)
                with torch.no_grad():
                    batch_targets_inside, _ = H_gpu(batch_positions_inside)

            # Loss calculation - improved weight configuration
            with autocast(device_type=cfg.DEVICE, enabled=cfg.DEVICE.startswith('cuda')):
                loss_no_structure, loss_dict, metrics, field1_vals, field2_vals, probs1_vals, probs2_vals = loss_fn(
                    model, 
                    batch_positions_inside, batch_targets_inside, batch_N, batch_mask, batch_toolPoints_raw,
                    printForTest = False
                )
                
                # Count elements greater than tEnd in field1
                with torch.no_grad():
                    field1_solid_time1_count = torch.sum(probs1_vals[:batch_N[0]] > 0.5).item()
                    total_elements = batch_N[0]
                
                for term in loss_dict.keys():
                    epoch_step_each_loss[term].append(loss_dict[term].item())
                for name in metrics:
                    epoch_step_each_metric[name].append(metrics[name])

            loss_grad_1 = loss_no_structure / accumulation_steps
            # Backpropagation
            scaler.scale(loss_grad_1).backward()

            grid_points = data_generator.sample_grid_points()
            with autocast(device_type=cfg.DEVICE, enabled=cfg.DEVICE.startswith('cuda')):
                
                l_structure, l_structure_new, l_simp_penalty, structure_norm = loss_fn.get_loss_structure(
                    model, grid_points, printForTest = False)


                epoch_step_each_loss['structure'].append(l_structure.item())
                epoch_step_each_loss['structure_simp_penalty'].append(l_simp_penalty.item())
                epoch_step_each_metric['structure_normalized'].append(structure_norm)
                
                if cfg.WEIGHTS['structure'] > 0.0:
                    loss_structure_all = cfg.WEIGHTS['structure'] * l_structure_new + cfg.WEIGHTS['structure_simp_penalty'] * l_simp_penalty
                else:
                    loss_structure_all = torch.tensor(0.0, device=grid_points.device)
            
            if loss_structure_all.item() > 0.0:
                loss_grad_2 = loss_structure_all / accumulation_steps
                scaler.scale(loss_grad_2).backward()

            unscaled_real_loss = loss_no_structure.item() + loss_structure_all.item()

            # 记录当前step的统计信息
            epoch_step_losses.append(unscaled_real_loss * 1e2)
            epoch_solid_time1_count += field1_solid_time1_count
            epoch_total_elements += total_elements
            
            # Gradient accumulation and update
            if (global_step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # Improved gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.MAX_CLIP_NORM)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1 
            
            # 更新进度条显示当前step信息
            if step_in_epoch % 10 == 0:  # 每10个step更新一次显示
                current_avg_loss = np.mean(epoch_step_losses[-10:]) if len(epoch_step_losses) >= 10 else np.mean(epoch_step_losses)
                step_percentage = (field1_solid_time1_count / total_elements) * 100 if total_elements > 0 else 0

                step_avg_each_loss = {term: np.mean(epoch_step_each_loss[term][-10:]) if len(epoch_step_each_loss[term]) >= 10 else np.mean(epoch_step_each_loss[term]) for term in cfg.TERM_SHORT_NAME}
                
                # 自动遍历cfg.WEIGHT_TERMS和step_avg_each_loss进行进度条显示
                loss_terms_str = ""
                for term in cfg.TERM_SHORT_NAME:
                    # 用简短别名显示（比如首字母/缩写），可以自定义映射
                    term_short = cfg.TERM_SHORT_NAME[term]
                    val = step_avg_each_loss[term]
                    loss_terms_str += f"{term_short}: {val:.4f} | "
                loss_terms_str += f"solidAM: {step_percentage:.1f}% | "

                pbar.set_description(
                    f"E{epoch} S{step_in_epoch}/{steps_per_epoch} | Loss: {current_avg_loss:.4f} | {loss_terms_str}"
                )

        epoch_avg_loss = np.mean(epoch_step_losses)
        epoch_avg_each_loss = {term: np.mean(epoch_step_each_loss[term]) for term in cfg.TERM_SHORT_NAME}
        epoch_avg_each_metric = {name: np.mean(epoch_step_each_metric[name]) if epoch_step_each_metric[name] else 0.0
                                 for name in epoch_step_each_metric}
        epoch_losses.append(epoch_avg_loss)
        epoch_percentage = (epoch_solid_time1_count / epoch_total_elements) * 100 if epoch_total_elements > 0 else 0

        # Step the scheduler with the epoch average loss  
        # -----------------------------------------------------------
        # Warmup Logic: Skip LR scheduler and Early Stopping during warmup
        # -----------------------------------------------------------
        if epoch < cfg.WARMUP_EPOCHS:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.write(f"[Warmup] Epoch {epoch}/{cfg.WARMUP_EPOCHS}: Skipping scheduler & early stopping. LR={current_lr:.2e}")
            
            should_stop = False
            stop_reason = ""
            best_state = early_stopping.get_best_state()
            
        else:
            scheduler.step(epoch_avg_loss)

            current_lr = optimizer.param_groups[0]['lr']
            combined_each = dict(epoch_avg_each_loss)
            for k, v in epoch_avg_each_metric.items():
                combined_each[f'metric_{k}'] = v
            should_stop, stop_reason, best_state = early_stopping.update(
                epoch_avg_loss, current_lr, model, each_loss=combined_each
            )

        es_status = early_stopping.get_status()

        summary_items = [
            f"[Epoch {epoch}] Epoch Summary",
            f"Avg Loss: {epoch_avg_loss:.4f} (Best: {es_status['best_loss']:.4f})",
        ]
        for term in cfg.TERM_SHORT_NAME:
            avg_val = epoch_avg_each_loss[term]
            summary_items.append(f"{cfg.TERM_SHORT_NAME[term]}: {avg_val:.4f}")
        summary_items.append(f"solid_time1: {epoch_percentage:.1f}%")
        summary_items.append(f"LR: {optimizer.param_groups[0]['lr']:.2e}, LR_F3: {optimizer.param_groups[2]['lr']:.2e}")
        summary_items.append(f"ES Counters: L{es_status['loss_patience']}/{cfg.JOINT_EARLY_STOPPING['loss_patience']}")
        pbar.write(" | ".join(summary_items))
        
            # Check early stopping
        if should_stop:
            pbar.write(f"\n🛑 {stop_reason}")
            if cfg.JOINT_EARLY_STOPPING['restore_best_weights'] and best_state is not None:
                pbar.write(f"Restoring best model weights (loss: {es_status['best_loss']:.6f})")
                model.load_state_dict(best_state)
            break

        # 更新最终的进度条显示
        # summary_items = [
        #     f"Epoch {epoch} completed",
        #     f"Avg Loss: {epoch_avg_loss:.4f} (Best: {es_status['best_loss']:.4f})",
        # ]
        # for term in cfg.TERM_SHORT_NAME:
        #     avg_val = epoch_avg_each_loss[term]
        #     summary_items.append(f"{cfg.TERM_SHORT_NAME[term]}: {avg_val:.4f}")
        # summary_items.append(f"solid_time1: {epoch_percentage:.1f}%")
        # summary_items.append(f"LR: {optimizer.param_groups[0]['lr']:.2e}, LR_F3: {optimizer.param_groups[2]['lr']:.2e}")
        # pbar.write(" | ".join(summary_items))

        if epoch%10==0:
            best_state = early_stopping.get_best_state()
            if best_state is not None:
                torch.save(best_state, tmp_model_path)
                print(f"\nBest model saved to '{tmp_model_path}'")
            else:
                torch.save(model.state_dict(), tmp_model_path)
                print(f"\nBest model not found, saved the last model to '{tmp_model_path}'")
    if epoch < effective_joint_epochs - 1:
        print(f"Joint training stopped early at epoch {epoch}")
    else:
        print("Joint training completed (reached max epochs)")

    # # Save loss curve with both step-wise and epoch-wise losses
    # fig = plt.figure(figsize=(15, 5))
    
    # # Subplot 1: Step-wise loss
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label="Training Loss (per step)", alpha=0.7)
    # plt.xlabel("Training Step")
    # plt.ylabel("Loss")
    # plt.title("Step-wise Training Loss")
    # plt.legend()
    # plt.grid(True)
    
    # # Subplot 2: Epoch-wise loss
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(epoch_losses)), epoch_losses, 'o-', label="Epoch Average Loss", markersize=4)

    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Epoch-wise Training Loss")
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig("output/joint_train_loss_curve.png", dpi=150)

    # if experiment is not None:
    #     experiment.log_figure(figure=fig, figure_name="joint_train_loss_curve")

    # plt.close()

    best_state = early_stopping.get_best_state()
    if best_state is None:
        best_state = model.state_dict()
    
    # Only save the model file in normal (non-optimizer) mode
    if weights_dict is None:
        torch.save(best_state, joint_model_path)
        print(f"\nBest model saved to '{joint_model_path}'")
    if quiet:
        builtins.print = _original_print
    
    # Build loss_dict from best state's per-term losses
    best_each_loss = early_stopping.get_best_each_loss()
    loss_dict = {}
    for term in cfg.TERM_SHORT_NAME:
        loss_dict[term] = best_each_loss[term] if best_each_loss is not None else epoch_avg_each_loss[term]
    
    # Build metrics_dict from best state's per-metric values
    metrics_dict = {}
    if best_each_loss is not None:
        for k, v in best_each_loss.items():
            if k.startswith('metric_'):
                metrics_dict[k[7:]] = v
    else:
        metrics_dict = dict(epoch_avg_each_metric)
    
    return loss_dict, metrics_dict, best_state


if __name__ == '__main__':
    loss_dict, metrics_dict, best_state = main()
    for term in loss_dict.keys():
        print(f"{term}: {loss_dict[term]}")
    for name in metrics_dict.keys():
        print(f"{name}: {metrics_dict[name]}")
