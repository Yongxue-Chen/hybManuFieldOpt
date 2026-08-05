
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from fieldopt.utils.data_loader import read_initial_field

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

from fieldopt.models.multi_field import MultiFieldModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fieldopt.geometry.implicit import get_sdf_aabb
from fieldopt.geometry.voxel.voxelization import get_normalization_parameters

import importlib
DEFAULT_MODEL_NAME = 'bracket'
DEFAULT_RESOLUTION = 100

cfg = None

# Focal Loss for handling hard examples in mask field
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification
    Args:
        pred: predicted probabilities [0, 1]
        target: ground truth labels [0, 1]
        alpha: weighting factor for class imbalance
        gamma: focusing parameter (higher = focus more on hard examples)
    """
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)  # probability of correct class
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def _nearest_distances_to_positive(negative_pts, positive_pts, chunk_size=65536):
    """
    Compute the nearest neighbor distance from each negative point to the positive set.
    Prefers scipy's cKDTree when available, otherwise falls back to chunked torch.cdist.
    """
    negative_cpu = negative_pts.float().cpu()
    positive_cpu = positive_pts.float().cpu()

    if negative_cpu.numel() == 0 or positive_cpu.numel() == 0:
        return torch.zeros(len(negative_cpu))

    if cKDTree is not None:
        tree = cKDTree(positive_cpu.numpy())
        try:
            distances, _ = tree.query(negative_cpu.numpy(), k=1, n_jobs=-1)
        except TypeError:
            distances, _ = tree.query(negative_cpu.numpy(), k=1)
        return torch.from_numpy(distances).float()

    distances = []
    for start in range(0, len(negative_cpu), chunk_size):
        end = min(start + chunk_size, len(negative_cpu))
        chunk = negative_cpu[start:end]
        cdist = torch.cdist(chunk, positive_cpu, p=2)
        min_dist, _ = torch.min(cdist, dim=1)
        distances.append(min_dist)
    return torch.cat(distances, dim=0)


def downsample_far(positive_pts, negative_pts, config):
    """
    Downsample positive or negative samples based on their ratio.
    If either positive:negative or negative:positive exceeds max_ratio,
    downsample the larger group by removing samples that are far from the other group.
    Returns the filtered positive and negative tensors, and a statistics dict.
    """
    stats = {
        'original_positives': len(positive_pts),
        'original_negatives': len(negative_pts),
        'kept_positives': len(positive_pts),
        'kept_negatives': len(negative_pts),
        'dropped_positives': 0,
        'dropped_negatives': 0,
        'far_threshold': None,
        'downsample_target': None,  # 'positive', 'negative', or None
    }

    if config is None or not config.get('enabled', False):
        return positive_pts, negative_pts, stats

    if len(negative_pts) == 0 or len(positive_pts) == 0:
        return positive_pts, negative_pts, stats

    max_ratio = config.get('max_ratio', None)
    if max_ratio is None or max_ratio <= 0:
        return positive_pts, negative_pts, stats

    # Calculate ratios
    pos_to_neg_ratio = len(positive_pts) / len(negative_pts)
    neg_to_pos_ratio = len(negative_pts) / len(positive_pts)

    # Check if downsampling is needed
    if pos_to_neg_ratio < max_ratio and neg_to_pos_ratio < max_ratio:
        stats['downsample_target'] = 'none'
        return positive_pts, negative_pts, stats

    # Determine which group to downsample
    if pos_to_neg_ratio > max_ratio:
        # Too many positives, downsample positive
        target_pts = positive_pts
        reference_pts = negative_pts
        target_count = int(len(negative_pts) * max_ratio)
        downsample_target = 'positive'
    else:
        # Too many negatives, downsample negative
        target_pts = negative_pts
        reference_pts = positive_pts
        target_count = int(len(positive_pts) * max_ratio)
        downsample_target = 'negative'

    stats['downsample_target'] = downsample_target
    target_count = max(1, target_count)  # Keep at least 1 sample

    # Calculate distances from target to reference
    distances = _nearest_distances_to_positive(target_pts, reference_pts)
    if len(distances) == 0:
        return positive_pts, negative_pts, stats

    # Get downsampling parameters
    far_quantile = float(config.get('far_distance_quantile', 0.8))
    far_quantile = min(max(far_quantile, 0.0), 1.0)
    keep_fraction = float(config.get('far_keep_fraction', 0.3))
    keep_fraction = min(max(keep_fraction, 0.0), 1.0)
    random_seed = config.get('random_seed', None)

    # Classify samples as near or far
    threshold = torch.quantile(distances, far_quantile) if far_quantile > 0 else torch.min(distances)
    far_mask = distances >= threshold
    near_mask = ~far_mask

    near_indices = torch.nonzero(near_mask, as_tuple=False).squeeze(1)
    far_indices = torch.nonzero(far_mask, as_tuple=False).squeeze(1)

    generator = torch.Generator()
    if random_seed is not None:
        generator.manual_seed(int(random_seed))

    # Downsample far samples first
    if len(far_indices) > 0 and keep_fraction < 1.0:
        n_keep_far = int(len(far_indices) * keep_fraction)
        if keep_fraction > 0.0:
            n_keep_far = max(1, n_keep_far)
        else:
            n_keep_far = 0
        perm = far_indices[torch.randperm(len(far_indices), generator=generator)]
        far_indices = perm[:n_keep_far]

    # Combine near and (downsampled) far indices
    combined_indices = near_indices
    if len(far_indices) > 0:
        combined_indices = torch.cat([combined_indices, far_indices], dim=0)

    # Further downsample if still exceeds target count
    if len(combined_indices) > target_count:
        total_to_remove = len(combined_indices) - target_count
        far_to_remove = min(total_to_remove, len(far_indices))
        if far_to_remove > 0:
            far_indices = far_indices[:-far_to_remove]
        remaining_to_remove = total_to_remove - far_to_remove
        if remaining_to_remove > 0 and len(near_indices) > 0:
            near_perm = near_indices[torch.randperm(len(near_indices), generator=generator)]
            keep_near = max(0, len(near_indices) - remaining_to_remove)
            near_indices = near_perm[:keep_near]
        combined_parts = []
        if len(near_indices) > 0:
            combined_parts.append(near_indices)
        if len(far_indices) > 0:
            combined_parts.append(far_indices)
        if combined_parts:
            combined_indices = torch.cat(combined_parts, dim=0)
        else:
            combined_indices = torch.arange(min(1, len(target_pts)), dtype=torch.long)

    if combined_indices.numel() == 0:
        # Fallback: keep at least one sample to avoid empty class
        combined_indices = torch.arange(min(1, len(target_pts)), dtype=torch.long)

    combined_indices = combined_indices.unique()
    filtered_target = target_pts[combined_indices]

    # Update stats and return appropriate results
    if downsample_target == 'positive':
        stats.update({
            'kept_positives': len(filtered_target),
            'dropped_positives': stats['original_positives'] - len(filtered_target),
            'far_threshold': threshold.item() if threshold is not None else None,
        })
        return filtered_target, negative_pts, stats
    else:
        stats.update({
            'kept_negatives': len(filtered_target),
            'dropped_negatives': stats['original_negatives'] - len(filtered_target),
            'far_threshold': threshold.item() if threshold is not None else None,
        })
        return positive_pts, filtered_target, stats

# Global variables for normalization parameters
TIME1_MEAN = None
TIME1_STD = None
TIME2_MEAN = None
TIME2_STD = None


def sample_field3_boundary_points(space_box, boundary_threshold, augmentation_config=None):
    """
    Randomly sample points inside boundary slabs for field3 supervision.
    Each slab uses the same constant vector label as the corresponding boundary rule.
    """
    if augmentation_config is None or not augmentation_config.get('enabled', False):
        return torch.empty((0, 3), dtype=torch.float32), torch.empty((0, 6), dtype=torch.float32)

    samples_per_face = augmentation_config.get('samples_per_face', 0)
    if isinstance(samples_per_face, int):
        samples_per_face = {
            'x_min': samples_per_face,
            'x_max': samples_per_face,
            'y_min': samples_per_face,
            'y_max': samples_per_face,
        }
    elif isinstance(samples_per_face, dict):
        samples_per_face = {
            'x_min': int(samples_per_face.get('x_min', 0)),
            'x_max': int(samples_per_face.get('x_max', 0)),
            'y_min': int(samples_per_face.get('y_min', 0)),
            'y_max': int(samples_per_face.get('y_max', 0)),
        }
    else:
        samples_per_face = {
            'x_min': 0,
            'x_max': 0,
            'y_min': 0,
            'y_max': 0,
        }

    bounds = torch.as_tensor(space_box, dtype=torch.float32)
    mins = bounds[0]
    maxs = bounds[1]
    spans = torch.clamp(maxs - mins, min=1e-8)

    generator = torch.Generator()
    random_seed = augmentation_config.get('random_seed', None)
    if random_seed is not None:
        generator.manual_seed(int(random_seed))

    def _rand(count, dims):
        return torch.rand((count, dims), generator=generator, dtype=torch.float32)

    def _sample_face(face_name, axis, is_min_face, label):
        count = samples_per_face.get(face_name, 0)
        if count <= 0:
            return None, None

        points = mins.unsqueeze(0) + _rand(count, 3) * spans.unsqueeze(0)
        slab_width = min(float(boundary_threshold), float(spans[axis].item()))
        slab_noise = _rand(count, 1).squeeze(1) * slab_width

        if is_min_face:
            points[:, axis] = mins[axis] + slab_noise
        else:
            points[:, axis] = maxs[axis] - slab_noise

        normals = torch.tensor(label, dtype=torch.float32).unsqueeze(0).repeat(count, 1)
        return points, normals

    face_specs = [
        ('x_min', 0, True, [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        ('x_max', 0, False, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ('y_min', 1, True, [0.0, -1.0, 0.0, 0.0, -1.0, 0.0]),
        ('y_max', 1, False, [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    ]

    sampled_points = []
    sampled_normals = []
    for face_name, axis, is_min_face, label in face_specs:
        face_points, face_normals = _sample_face(face_name, axis, is_min_face, label)
        if face_points is None:
            continue
        sampled_points.append(face_points)
        sampled_normals.append(face_normals)

    if not sampled_points:
        return torch.empty((0, 3), dtype=torch.float32), torch.empty((0, 6), dtype=torch.float32)

    return torch.cat(sampled_points, dim=0), torch.cat(sampled_normals, dim=0)


def resolve_field3_face_counts(boundary_base_count, augmentation_config=None):
    """
    Resolve how many boundary samples to draw per face.

    Priority:
    1. boundary_sample_multiplier: added boundary samples are boundary_base_count * multiplier.
    2. samples_per_face: explicit per-face counts.
    """
    zero_counts = {
        'x_min': 0,
        'x_max': 0,
        'y_min': 0,
        'y_max': 0,
    }

    if augmentation_config is None or not augmentation_config.get('enabled', False):
        return zero_counts

    boundary_sample_multiplier = augmentation_config.get('boundary_sample_multiplier', None)
    if boundary_sample_multiplier is not None:
        boundary_sample_multiplier = max(float(boundary_sample_multiplier), 0.0)
        added_total = max(0, int(round(boundary_base_count * boundary_sample_multiplier)))

        face_names = ['x_min', 'x_max', 'y_min', 'y_max']
        counts = {}
        base_face_count = added_total // len(face_names)
        remainder = added_total % len(face_names)
        for idx, face_name in enumerate(face_names):
            counts[face_name] = base_face_count + (1 if idx < remainder else 0)
        return counts

    samples_per_face = augmentation_config.get('samples_per_face', 0)
    if isinstance(samples_per_face, int):
        return {
            'x_min': samples_per_face,
            'x_max': samples_per_face,
            'y_min': samples_per_face,
            'y_max': samples_per_face,
        }

    if isinstance(samples_per_face, dict):
        return {
            'x_min': int(samples_per_face.get('x_min', 0)),
            'x_max': int(samples_per_face.get('x_max', 0)),
            'y_min': int(samples_per_face.get('y_min', 0)),
            'y_max': int(samples_per_face.get('y_max', 0)),
        }

    return zero_counts

# In pretrain_field function call
def pretrain_field(model, field_index, train_positions, train_targets, val_positions, val_targets, 
                   train_targets2=None, val_targets2=None, config=None, experiment=None):
    """
    Pretrain the specified field with train/validation split.
    experiment: Optional external logger object.
    Uses per-field optimization strategies and learning rate scheduler based on validation loss.
    
    Args:
        field_index: 1 for field1, 2 for field2, 'M1' for field M1, 'M2' for field M2
        train_positions: training input positions
        train_targets: training target values (for field1) or combined targets (for field2)
        val_positions: validation input positions
        val_targets: validation target values
        train_targets2: second training target values (only used when field_index=2)
        val_targets2: second validation target values (only used when field_index=2)
    """
    # Use the new configuration system or fall back to the old one
    if config is None:
        config = cfg.TRAINING_CONFIG
    print(f"\n" + "-"*20 + f" Starting pretraining for field {field_index} " + "-"*20)
    
    # Set the active field in the model
    if field_index == 1:
        model.set_active_field('field1')
    elif field_index == 2:
        model.set_active_field('field2')
    elif field_index == 3:
        model.set_active_field('field3')
    elif field_index == 'M1':
        model.set_active_field('fieldM1')
    elif field_index == 'M2':
        model.set_active_field('fieldM2')

    # Preload all data to GPU memory
    train_positions = train_positions.to(cfg.DEVICE, non_blocking=True)
    train_targets = train_targets.to(cfg.DEVICE, non_blocking=True)
    val_positions = val_positions.to(cfg.DEVICE, non_blocking=True)
    val_targets = val_targets.to(cfg.DEVICE, non_blocking=True)
    
    if train_targets2 is not None:
        train_targets2 = train_targets2.to(cfg.DEVICE, non_blocking=True)
    if val_targets2 is not None:
        val_targets2 = val_targets2.to(cfg.DEVICE, non_blocking=True)
    
    # Enable more efficient memory format
    train_positions = train_positions.contiguous()
    train_targets = train_targets.contiguous()
    val_positions = val_positions.contiguous()
    val_targets = val_targets.contiguous()
    if train_targets2 is not None:
        train_targets2 = train_targets2.contiguous()
    if val_targets2 is not None:
        val_targets2 = val_targets2.contiguous()
    
    # --- Field-specific optimizer configuration ---
    if field_index == 'M1':
        field_config_key = 'fieldM1_params'
    elif field_index == 'M2':
        field_config_key = 'fieldM2_params'
    else:
        field_config_key = f'field{field_index}_params'
    
    learning_rate = config[field_config_key]['lr']
    weight_decay = config[field_config_key]['weight_decay']
    dropout_rate = config[field_config_key]['dropout']
    print(f"Field {field_index} configuration: LR={learning_rate}, WD={weight_decay}, Dropout={dropout_rate}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay,
                                 betas=(0.9, 0.99),
                                 eps=1e-15)
    scaler = GradScaler(device='cuda', enabled=cfg.DEVICE.startswith('cuda'))

    # Mask fields use plain BCEWithLogitsLoss during pretraining.
    # Keep pos_weight in the return value for logging compatibility.
    pos_weight = None

    batch_size = cfg.PRETRAIN_BATCH_SIZE
    print(f"Using batch size: {batch_size}")
    print(f"Training samples: {len(train_positions)}, Validation samples: {len(val_positions)}")

    is_mask_field = field_index in ('M1', 'M2')
    mask_positive_indices = None
    mask_negative_indices = None
    if is_mask_field:
        train_targets_flat = train_targets.view(-1)
        mask_positive_indices = torch.nonzero(train_targets_flat == 1.0, as_tuple=False).squeeze(1)
        mask_negative_indices = torch.nonzero(train_targets_flat == 0.0, as_tuple=False).squeeze(1)
        print(
            f"Balanced mask batching: positives={len(mask_positive_indices)}, "
            f"negatives={len(mask_negative_indices)}, positive_fraction_per_batch=0.50"
        )
    
    # --- Learning rate scheduler configuration ---
    scheduler_params_key = f'scheduler_params{field_index}'
    scheduler_config = config[scheduler_params_key]

    # Warmup: for mask fields, keep LR constant for warmup_epochs before enabling scheduler
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        print(f"Warmup enabled: keeping constant LR={learning_rate} for first {warmup_epochs} epochs")

    if scheduler_config['type'] == 'ReduceLROnPlateau':
        patience = scheduler_config.get('patience', 10)
        factor = scheduler_config.get('factor', 0.5)
        min_lr = scheduler_config.get('min_lr', 1e-7)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
        
        print(f"Learning rate scheduler: ReduceLROnPlateau (based on validation loss), patience={patience}, factor={factor}, min_lr={min_lr}")
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
    
    train_losses = [] # List to store training loss values
    val_losses = []   # List to store validation loss values

    # --- Early stopping mechanism initialization ---
    early_stop_patience = config['early_stopping']['patience']
    early_stop_min_delta = config['early_stopping']['min_delta']
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    print(f"Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta} (based on validation loss)")
    
    # Helper function to compute loss for a batch
    def compute_loss(positions, targets, targets2=None, add_noise=False):
        use_autocast = cfg.DEVICE.startswith('cuda') and field_index != 'M1' and field_index != 'M2'
        
        # Add spatial noise augmentation during training for better generalization
        if add_noise and model.training:
            # Adaptive noise based on field type
            noise_std = config[field_config_key]['noise_std']
            
            noise = torch.randn_like(positions) * noise_std
            positions = positions + noise
            # Clamp to bounding box to ensure valid inputs
            positions = torch.clamp(positions, 
                                  min=model.bounding_box[0], 
                                  max=model.bounding_box[1])
        
        with autocast(device_type=cfg.DEVICE, enabled=use_autocast):
            if field_index == 1:
                output = model.forward(positions, field_type='field1')
                output_normalized = (output - TIME1_MEAN) / TIME1_STD
                targets_normalized = (targets - TIME1_MEAN) / TIME1_STD
                loss = F.mse_loss(output_normalized, targets_normalized)
            elif field_index == 2:
                with torch.no_grad():
                    field1_output = model.forward(positions, field_type='field1')
                    # Add noise to field1 output during training for robustness
                    if model.training and add_noise:
                        field1_noise = torch.randn_like(field1_output) * (TIME1_STD * 0.02)
                        field1_output = field1_output + field1_noise
                
                field2_output = model.forward(positions, field_type='field2')
                output_time2 = field1_output + field2_output * (cfg.MAX_TIME - field1_output)
                target_time2 = targets + targets2 * (cfg.MAX_TIME - targets)
                output_time2_normalized = (output_time2 - TIME2_MEAN) / TIME2_STD
                target_time2_normalized = (target_time2 - TIME2_MEAN) / TIME2_STD
                loss = F.mse_loss(output_time2_normalized, target_time2_normalized)

            elif field_index == 3:
                normal_output = model.forward(positions, field_type='field3')
                normal_output1 = normal_output[...,:3]
                normal_output2 = normal_output[...,3:]
                target1 = targets[...,:3]
                target2 = targets[...,3:]

                cosine_sim1 = F.cosine_similarity(normal_output1, target1, dim=-1)
                cosine_sim2 = F.cosine_similarity(normal_output2, target2, dim=-1)
                loss = (1.0 - cosine_sim1).mean() + (1.0 - cosine_sim2).mean()

            elif field_index == 'M1':
                # Mask head outputs logits; use unweighted BCEWithLogitsLoss.
                logits = model.forward(positions, field_type='fieldM1')
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            elif field_index == 'M2':
                # Mask head outputs logits; use unweighted BCEWithLogitsLoss.
                logits = model.forward(positions, field_type='fieldM2')
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            else:
                raise ValueError(f"Invalid field index: {field_index}")
        
        return loss
    
    pbar = tqdm(range(config['pretrain_epochs']), desc=f"Pretraining field {field_index}")
    for epoch in pbar:
        # ====== Training Phase ======
        model.train()
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)
        
        if is_mask_field and len(mask_positive_indices) > 0 and len(mask_negative_indices) > 0:
            n_pos = batch_size // 2
            n_neg = batch_size - n_pos
            pos_choice = torch.randint(0, len(mask_positive_indices), (n_pos,), device=cfg.DEVICE)
            neg_choice = torch.randint(0, len(mask_negative_indices), (n_neg,), device=cfg.DEVICE)
            indices = torch.cat([
                mask_positive_indices[pos_choice],
                mask_negative_indices[neg_choice],
            ], dim=0)
            indices = indices[torch.randperm(len(indices), device=cfg.DEVICE)]
        else:
            indices = torch.randint(0, len(train_positions), (batch_size,), device=cfg.DEVICE)
        batch_positions = train_positions[indices]
        batch_targets = train_targets[indices]
        batch_targets2 = train_targets2[indices] if train_targets2 is not None else None
        
        # Forward pass (with data augmentation for training)
        loss = compute_loss(batch_positions, batch_targets, batch_targets2, add_noise=True)
        current_train_loss = loss.item()

        # Backpropagation
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model weights
        scaler.step(optimizer)
        scaler.update()
        
        train_losses.append(current_train_loss)
        
        # ====== Validation Phase (every 5 epochs) ======
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Compute validation loss on all validation data
                val_batch_size = min(batch_size, len(val_positions))
                val_loss_accum = 0.0
                num_val_batches = (len(val_positions) + val_batch_size - 1) // val_batch_size
                
                for i in range(num_val_batches):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, len(val_positions))
                    val_batch_pos = val_positions[start_idx:end_idx]
                    val_batch_tar = val_targets[start_idx:end_idx]
                    val_batch_tar2 = val_targets2[start_idx:end_idx] if val_targets2 is not None else None
                    
                    val_batch_loss = compute_loss(val_batch_pos, val_batch_tar, val_batch_tar2, add_noise=False)
                    val_loss_accum += val_batch_loss.item() * (end_idx - start_idx)
                
                current_val_loss = val_loss_accum / len(val_positions)
                val_losses.append(current_val_loss)
            warmup_tag = " [WARMUP]" if epoch < warmup_epochs else ""
            pbar.set_description(f"Pre-F{field_index} E{epoch} | TL: {current_train_loss:.6f} | VL: {current_val_loss:.6f} | Best VL: {best_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}{warmup_tag}")

            # ====== Early Stopping Check (based on validation loss) ======
            # At the end of warmup, reset early stopping state so accumulated
            # non-improvements during warmup don't trigger an immediate stop.
            if warmup_epochs > 0 and epoch == warmup_epochs:
                pbar.write(f"⚡ Warmup ended at epoch {epoch}. Resetting early stopping counter and best_val_loss.")
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if current_val_loss < best_val_loss - early_stop_min_delta:
                relative_improvement = (best_val_loss - current_val_loss) / best_val_loss if best_val_loss < float('inf') else 0
                absolute_improvement = best_val_loss - current_val_loss
                best_val_loss = current_val_loss
                epochs_no_improve = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pbar.write(f"✓ New best validation loss: {best_val_loss:.6f} (rel_improvement: {relative_improvement:.4f}, abs_improvement: {absolute_improvement:.6f})")
            else:
                epochs_no_improve += 1
            
            # Check if early stopping should be triggered
            # Skip early stopping during warmup (model not expected to improve yet)
            if epoch >= warmup_epochs and epochs_no_improve >= early_stop_patience:
                pbar.write(f"⚠ Early stopping triggered at epoch {epoch} (field {field_index})!")
                pbar.write(f"   Validation loss has not improved for {epochs_no_improve * 5} epochs")
                pbar.write(f"   Best validation loss: {best_val_loss:.6f}")
                break

            # Update learning rate scheduler based on validation loss
            # Skip scheduler during warmup phase (keep LR constant)
            if epoch >= warmup_epochs:
                scheduler.step(current_val_loss)
        else:
            # Update progress bar without validation
            pbar.set_description(f"Pre-F{field_index} E{epoch} | TL: {current_train_loss:.6f} | Best VL: {best_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
    print(f"Field {field_index} pretraining completed.")
    
    # If early stopping was used, restore the best model state
    if best_model_state is not None:
        print(f"✓ Restoring model to best validation loss: {best_val_loss:.6f}")
        model.load_state_dict(best_model_state)

    # Set model back to training all fields after pretraining is done
    model.set_active_field('all')
    
    return train_losses, val_losses, pos_weight

def evaluate_mask_field(model, positions, labels, field_type, batch_size=131072):
    """
    Evaluate mask field metrics (accuracy, recall, F1) on the provided dataset.
    Applies sigmoid + 0.5 threshold to logits.
    """
    total_samples = labels.shape[0]
    if total_samples == 0:
        print(f"[WARN] No samples provided for {field_type} evaluation.")
        return {'accuracy': 0.0, 'recall': 0.0, 'f1': 0.0}

    prev_mode = model.training
    model.eval()

    tp = fp = tn = fn = 0

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_positions = positions[start:end].to(cfg.DEVICE, non_blocking=True)
        batch_labels = labels[start:end].to(cfg.DEVICE, non_blocking=True)

        with torch.no_grad():
            logits = model.forward(batch_positions, field_type=field_type)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

        tp += torch.logical_and(preds == 1.0, batch_labels == 1.0).sum().item()
        tn += torch.logical_and(preds == 0.0, batch_labels == 0.0).sum().item()
        fp += torch.logical_and(preds == 1.0, batch_labels == 0.0).sum().item()
        fn += torch.logical_and(preds == 0.0, batch_labels == 1.0).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if prev_mode:
        model.train()

    print(f"{field_type} validation -> accuracy: {accuracy:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")

    return {'accuracy': accuracy, 'recall': recall, 'f1': f1}

import argparse

def main():

    parser = argparse.ArgumentParser(description="Pretrain a multi field model from scratch.")
    parser.add_argument(
        '--model_name',
        default=DEFAULT_MODEL_NAME,
        help=f"Model/config name to load from configs.config_multi_field_<model_name> (default: {DEFAULT_MODEL_NAME})."
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2 = 20%% validation, 80%% training)."
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=DEFAULT_RESOLUTION,
        help=(
            "Resolution used in the default input and output filenames "
            f"(default: {DEFAULT_RESOLUTION})."
        ),
    )
    parser.add_argument(
        '--initial-field-path',
        '--initial_field_path',
        dest='initial_field_path',
        default=None,
        help=(
            "Initial-field input. Defaults to "
            "model_data/initial_fields/<model>/<model>_res<resolution>_initial_fields.txt."
        ),
    )
    parser.add_argument(
        '--centers-path',
        '--centers_path',
        dest='centers_path',
        default=None,
        help=(
            "Voxel-center input. Defaults to "
            "model_data/initial_fields/<model>/<model>_res<resolution>_centers.txt."
        ),
    )
    parser.add_argument(
        '--stl-path',
        '--stl_path',
        dest='stl_path',
        default=None,
        help=(
            "Preprocessed body-and-support STL used for the normalized AABB. "
            "Defaults to model_data/preprocessed/<model>/<model>_support.stl."
        ),
    )
    parser.add_argument(
        '--output-path',
        '--output_path',
        dest='output_path',
        default=None,
        help=(
            "Pretrained checkpoint output. Defaults to "
            "model_data/pretrained_fields/<model>/<model>_pretrained_<resolution>.pth."
        ),
    )
    parser.add_argument(
        '--pretrain-epochs',
        '--pretrain_epochs',
        dest='pretrain_epochs',
        type=int,
        default=None,
        help=(
            "Optional override for TRAINING_CONFIG['pretrain_epochs']; "
            "use a small value for a quick README test."
        ),
    )
    args = parser.parse_args()

    global cfg
    model_name = args.model_name
    resolution = args.resolution
    if resolution <= 0:
        parser.error("--resolution must be a positive integer")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val_split must be between 0 and 1")
    if args.pretrain_epochs is not None and args.pretrain_epochs <= 0:
        parser.error("--pretrain-epochs must be a positive integer")

    cfg = importlib.import_module(f'configs.config_multi_field_{model_name}')
    if args.pretrain_epochs is not None:
        cfg.TRAINING_CONFIG['pretrain_epochs'] = args.pretrain_epochs

    experiment = None
    # --- 1. Load and prepare data ---
    initial_field_path = args.initial_field_path or os.path.join(
        'model_data',
        'initial_fields',
        model_name,
        f'{model_name}_res{resolution}_initial_fields.txt',
    )
    center_path = args.centers_path or os.path.join(
        'model_data',
        'initial_fields',
        model_name,
        f'{model_name}_res{resolution}_centers.txt',
    )
    stl_path = args.stl_path or os.path.join(
        'model_data',
        'preprocessed',
        model_name,
        f'{model_name}_support.stl',
    )
    pretrain_model_path = args.output_path or os.path.join(
        'model_data',
        'pretrained_fields',
        model_name,
        f'{model_name}_pretrained_{resolution}.pth',
    )

    for label, path in (
        ('initial field', initial_field_path),
        ('voxel centers', center_path),
        ('preprocessed STL', stl_path),
    ):
        if not os.path.isfile(path):
            parser.error(f"{label} file not found: {path}")

    print("Pretraining inputs:")
    print(f"  initial field : {initial_field_path}")
    print(f"  voxel centers : {center_path}")
    print(f"  support STL   : {stl_path}")
    print(f"  resolution    : {resolution}")
    print(f"Pretraining output: {pretrain_model_path}")

    # aabb_min, aabb_max = get_sdf_aabb(stl_path)
    # spaceBox = np.array([aabb_min, aabb_max])
    _, p_min, spaceBox = get_normalization_parameters(stl_path)
    spaceBox[0] = spaceBox[0] - p_min
    spaceBox[1] = spaceBox[1] - p_min

    _, t_end, pts, time1, time2 = read_initial_field(initial_field_path, center_path, get_difference=False)

    t_end = t_end * 2 *cfg.MARGIN
    time1 = time1 * 2 *cfg.MARGIN
    time2 = time2 * 2 *cfg.MARGIN

    valid_M1 = (time1.squeeze(-1) <= t_end)
    valid_M2 = (time2.squeeze(-1) <= t_end)
    empty_mask = ~valid_M1 | valid_M2

    minTime1 = time1.min()
    minTime2 = time2.min()
    maxTime1 = time1[valid_M1].max() if np.any(valid_M1) else None
    maxTime2 = time2[valid_M2].max() if np.any(valid_M2) else None
    print(f"min(time1): {minTime1}, max(time1) < t_end: {maxTime1}")
    print(f"min(time2): {minTime2}, max(time2) < t_end: {maxTime2}")

    alphaT2 = (time2 - time1) / (cfg.MAX_TIME - time1)

    valid_pts_M1 = pts[valid_M1]
    valid_time1_M1 = time1[valid_M1].squeeze(-1)
    
    M1_positive_position = pts[empty_mask & valid_M1]
    M1_negative_position = pts[empty_mask & ~valid_M1]
    # M1_positive_position = pts[valid_M1]
    # M1_negative_position = pts[~valid_M1]

    valid_pts_M2 = pts[valid_M2]
    valid_time1_M2 = time1[valid_M2].squeeze(-1)
    valid_alphaT2_M2 = alphaT2[valid_M2].squeeze(-1)
    
    M2_positive_position = pts[empty_mask & valid_M2]
    M2_negative_position = pts[empty_mask & ~valid_M2]
    # M2_positive_position = pts[valid_M2]
    # M2_negative_position = pts[~valid_M2]

    valid_pts_M1_tensor = torch.tensor(valid_pts_M1, dtype=torch.float32)
    valid_time1_M1_tensor = torch.tensor(valid_time1_M1, dtype=torch.float32).view(-1, 1)

    valid_pts_M2_tensor = torch.tensor(valid_pts_M2, dtype=torch.float32)
    valid_time1_M2_tensor = torch.tensor(valid_time1_M2, dtype=torch.float32).view(-1, 1)
    valid_alphaT2_M2_tensor = torch.tensor(valid_alphaT2_M2, dtype=torch.float32).view(-1, 1)

    M1_positive_position_tensor = torch.tensor(M1_positive_position, dtype=torch.float32)
    M1_negative_position_tensor = torch.tensor(M1_negative_position, dtype=torch.float32)
    M2_positive_position_tensor = torch.tensor(M2_positive_position, dtype=torch.float32)
    M2_negative_position_tensor = torch.tensor(M2_negative_position, dtype=torch.float32)

    # Mask field 2 negative balancing
    para_balance = getattr(cfg, 'PARA_BALANCING', None)
    M1_positive_position_tensor, M1_negative_position_tensor, _ = downsample_far(
        M1_positive_position_tensor, M1_negative_position_tensor, para_balance)
    M2_positive_position_tensor, M2_negative_position_tensor, _ = downsample_far(
        M2_positive_position_tensor, M2_negative_position_tensor, para_balance)

    # generate tensors for field 3
    # Initialize all normals to [0, 0, 1, 0, 0, 1] (default upward direction)
    valid_normal_M2_tensor = torch.zeros(len(valid_pts_M2_tensor), 6)
    valid_normal_M2_tensor[:, 2] = 1.0
    valid_normal_M2_tensor[:, 5] = 1.0
    boundary_threshold = 5.0 / cfg.SCALE

    # Get bounding box boundaries
    x_min, y_min, z_min = spaceBox[0]
    x_max, y_max, z_max = spaceBox[1]

    # Extract x, y coordinates from valid_pts_M2_tensor
    x_coords = valid_pts_M2_tensor[:, 0]
    y_coords = valid_pts_M2_tensor[:, 1]

    # Check distance to x boundaries
    dist_to_x_min = torch.abs(x_coords - x_min)
    dist_to_x_max = torch.abs(x_coords - x_max)

    # Check distance to y boundaries
    dist_to_y_min = torch.abs(y_coords - y_min)
    dist_to_y_max = torch.abs(y_coords - y_max)

    # Create masks for boundary conditions
    mask_x_min = dist_to_x_min <= boundary_threshold
    mask_x_max = dist_to_x_max <= boundary_threshold
    mask_y_min = dist_to_y_min <= boundary_threshold
    mask_y_max = dist_to_y_max <= boundary_threshold

    # Apply boundary normals (overwrite default normals)
    # X lower boundary: normal = [-1, 0, 0]
    valid_normal_M2_tensor[mask_x_min, 0] = -1.0
    valid_normal_M2_tensor[mask_x_min, 1] = 0.0
    valid_normal_M2_tensor[mask_x_min, 2] = 0.0
    valid_normal_M2_tensor[mask_x_min, 3] = -1.0
    valid_normal_M2_tensor[mask_x_min, 4] = 0.0
    valid_normal_M2_tensor[mask_x_min, 5] = 0.0

    # X upper boundary: normal = [1, 0, 0]
    valid_normal_M2_tensor[mask_x_max, 0] = 1.0
    valid_normal_M2_tensor[mask_x_max, 1] = 0.0
    valid_normal_M2_tensor[mask_x_max, 2] = 0.0
    valid_normal_M2_tensor[mask_x_max, 3] = 1.0
    valid_normal_M2_tensor[mask_x_max, 4] = 0.0
    valid_normal_M2_tensor[mask_x_max, 5] = 0.0

    # Y lower boundary: normal = [0, -1, 0]
    valid_normal_M2_tensor[mask_y_min, 0] = 0.0
    valid_normal_M2_tensor[mask_y_min, 1] = -1.0
    valid_normal_M2_tensor[mask_y_min, 2] = 0.0
    valid_normal_M2_tensor[mask_y_min, 3] = 0.0
    valid_normal_M2_tensor[mask_y_min, 4] = -1.0
    valid_normal_M2_tensor[mask_y_min, 5] = 0.0

    # Y upper boundary: normal = [0, 1, 0]
    valid_normal_M2_tensor[mask_y_max, 0] = 0.0
    valid_normal_M2_tensor[mask_y_max, 1] = 1.0
    valid_normal_M2_tensor[mask_y_max, 2] = 0.0
    valid_normal_M2_tensor[mask_y_max, 3] = 0.0
    valid_normal_M2_tensor[mask_y_max, 4] = 1.0
    valid_normal_M2_tensor[mask_y_max, 5] = 0.0

    field3_aug_config = getattr(cfg, 'FIELD3_BOUNDARY_AUG', None)
    original_field3_boundary_mask = mask_x_min | mask_x_max | mask_y_min | mask_y_max
    original_field3_boundary_count = int(original_field3_boundary_mask.sum().item())
    field3_face_counts = resolve_field3_face_counts(original_field3_boundary_count, field3_aug_config)

    field3_pts_tensor = valid_pts_M2_tensor
    field3_normal_tensor = valid_normal_M2_tensor
    aug_pts_3, aug_normals_3 = sample_field3_boundary_points(
        spaceBox,
        boundary_threshold,
        augmentation_config={
            **(field3_aug_config or {}),
            'samples_per_face': field3_face_counts,
        },
    )
    if len(aug_pts_3) > 0:
        field3_pts_tensor = torch.cat([field3_pts_tensor, aug_pts_3], dim=0)
        field3_normal_tensor = torch.cat([field3_normal_tensor, aug_normals_3], dim=0)
        print(
            f"Field 3 boundary augmentation: added {len(aug_pts_3)} samples "
            f"(x_min={field3_face_counts['x_min']}, x_max={field3_face_counts['x_max']}, "
            f"y_min={field3_face_counts['y_min']}, y_max={field3_face_counts['y_max']})"
        )
    print(
        f"Field 3 dataset size: original_total={len(valid_pts_M2_tensor)}, "
        f"original_boundary={original_field3_boundary_count}, augmented_total={len(field3_pts_tensor)}"
    )

    # Standardize Field 1 to improve training stability
    time1_mean = valid_time1_M1_tensor.mean()
    time1_std = valid_time1_M1_tensor.std()
    
    # Standardize Field 2 to improve training stability
    valid_time2_M2_tensor = valid_time1_M2_tensor + valid_alphaT2_M2_tensor * (cfg.MAX_TIME - valid_time1_M2_tensor)
    time2_mean = valid_time2_M2_tensor.mean()
    time2_std = valid_time2_M2_tensor.std()

    # Save normalization parameters globally for later use
    global TIME1_MEAN, TIME1_STD, TIME2_MEAN, TIME2_STD
    TIME1_MEAN = time1_mean.item()
    TIME1_STD = time1_std.item()
    TIME2_MEAN = time2_mean.item()
    TIME2_STD = time2_std.item()

    # --- 2. Initialize model ---
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
        # Parameters for the mask fields
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
        dropout_rate_fieldM2=cfg.TRAINING_CONFIG['fieldM2_params']['dropout']
    ).to(cfg.DEVICE)

    torch.set_float32_matmul_precision('high')
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params:,}")

    print(f"\n--- Mode: Training with validation split ({args.val_split*100:.0f}% validation) ---")

    # Split data into training and validation sets
    n_samples_1 = len(valid_pts_M1_tensor)
    n_val_1 = int(n_samples_1 * args.val_split)
    n_train_1 = n_samples_1 - n_val_1
    indices_1 = torch.randperm(n_samples_1)
    train_indices_1 = indices_1[:n_train_1]
    val_indices_1 = indices_1[n_train_1:]
    train_pts_1 = valid_pts_M1_tensor[train_indices_1]
    train_time1_1 = valid_time1_M1_tensor[train_indices_1]
    val_pts_1 = valid_pts_M1_tensor[val_indices_1]
    val_time1_1 = valid_time1_M1_tensor[val_indices_1]

    n_samples_2 = len(valid_pts_M2_tensor)
    n_val_2 = int(n_samples_2 * args.val_split)
    n_train_2 = n_samples_2 - n_val_2
    indices_2 = torch.randperm(n_samples_2)
    train_indices_2 = indices_2[:n_train_2]
    val_indices_2 = indices_2[n_train_2:]
    train_pts_2 = valid_pts_M2_tensor[train_indices_2]
    train_time1_2 = valid_time1_M2_tensor[train_indices_2]
    train_alphaT2_2 = valid_alphaT2_M2_tensor[train_indices_2]
    val_pts_2 = valid_pts_M2_tensor[val_indices_2]
    val_time1_2 = valid_time1_M2_tensor[val_indices_2]
    val_alphaT2_2 = valid_alphaT2_M2_tensor[val_indices_2]
    
    # --- 3. Pre-training ---

    # 3.1 Pre-train field 1 - fit train_vals1
    print("\n=== Training Field 1 to fit train_vals1 ===")
    pretrain_field(model, 1, train_pts_1, train_time1_1, val_pts_1, val_time1_1, 
                   config=cfg.TRAINING_CONFIG, experiment=experiment)

    # 3.2 Pre-train field 2 - fit train_vals1+train_vals2 with field1+field2
    print("\n=== Training Field 2 so that (field1 + field2) fits (train_vals1 + train_vals2) ===")
    pretrain_field(model, 2, train_pts_2, train_time1_2, val_pts_2, val_time1_2,
                   train_targets2=train_alphaT2_2, val_targets2=val_alphaT2_2,
                   config=cfg.TRAINING_CONFIG, experiment=experiment)

    # 3.3 Pre-train field 3
    n_samples_3 = len(field3_pts_tensor)
    n_val_3 = int(n_samples_3 * args.val_split)
    n_train_3 = n_samples_3 - n_val_3
    indices_3 = torch.randperm(n_samples_3)
    train_indice_3 = indices_3[:n_train_3]
    val_indice_3 = indices_3[n_train_3:]
    train_pts_3 = field3_pts_tensor[train_indice_3]
    train_normal_3 = field3_normal_tensor[train_indice_3]
    val_pts_3 = field3_pts_tensor[val_indice_3]
    val_normal_3 = field3_normal_tensor[val_indice_3]
    pretrain_field(model, 3, train_pts_3, train_normal_3, val_pts_3, val_normal_3,
                   config=cfg.TRAINING_CONFIG, experiment=experiment)
    
    # 3.4 Pre-train field M1 - output 1 for valid points, 0 for invalid points
    print("\n=== Training Mask Field (1 for valid points, 0 for invalid points) ===")
    # Combine valid and invalid points
    all_pts_for_mask = torch.cat([M1_positive_position_tensor, M1_negative_position_tensor], dim=0)
    # Create labels: 1 for valid, 0 for invalid
    mask_labels = torch.cat([
        torch.ones(len(M1_positive_position_tensor), 1, dtype=torch.float32),
        torch.zeros(len(M1_negative_position_tensor), 1, dtype=torch.float32)
    ], dim=0)
    
    # Split mask data
    n_mask_samples = len(all_pts_for_mask)
    n_mask_val = int(n_mask_samples * args.val_split)
    n_mask_train = n_mask_samples - n_mask_val
    
    mask_indices = torch.randperm(n_mask_samples)
    mask_train_indices = mask_indices[:n_mask_train]
    mask_val_indices = mask_indices[n_mask_train:]
    
    M1_train_pts = all_pts_for_mask[mask_train_indices]
    M1_train_labels = mask_labels[mask_train_indices]
    M1_val_pts = all_pts_for_mask[mask_val_indices]
    M1_val_labels = mask_labels[mask_val_indices]
    
    _,_,pos_weight_M1 = pretrain_field(model, 'M1', M1_train_pts, M1_train_labels, M1_val_pts, M1_val_labels,
                   config=cfg.TRAINING_CONFIG, experiment=experiment)


    # 3.5 Field M2.
    #
    # M1 and M2 are trained on the same data: empty_mask = ~valid_M1 | valid_M2
    # and valid_M2 is a subset of valid_M1 (time2 >= time1), so both the
    # positive sets reduce to valid_M2 and both the negative sets to ~valid_M1.
    # When the two mask fields are also architecturally identical, pretraining
    # M2 separately would only reproduce M1 up to the random seed, so the M1
    # weights are copied across instead.
    #
    # The shipped configs keep the two mask fields identical, so this is the
    # normal path. A custom config may still give them different hash encoders
    # (e.g. N_maxM1 != N_maxM2), which makes the tensors shape-incompatible; in
    # that case M2 is pretrained separately as before.
    sd = model.state_dict()
    m1_to_m2 = {k: k.replace('fieldM1', 'fieldM2') for k in sd if 'fieldM1' in k}
    can_copy_M1_to_M2 = bool(m1_to_m2) and all(
        m2_key in sd and sd[m1_key].shape == sd[m2_key].shape
        for m1_key, m2_key in m1_to_m2.items()
    )

    if can_copy_M1_to_M2:
        print("\n=== Copying M1 weights to M2 (same training data) ===")
        for m1_key, m2_key in m1_to_m2.items():
            sd[m2_key] = sd[m1_key].clone()
        model.load_state_dict(sd)
        print("✓ M1 weights copied to M2 successfully.")

        pos_weight_M2 = pos_weight_M1
        M2_val_pts = M1_val_pts
        M2_val_labels = M1_val_labels
        m2_pos_weight_note = " (copied from M1)"
    else:
        print("\n=== fieldM1 and fieldM2 have different shapes, pretraining M2 separately ===")
        print("=== Training Mask Field (1 for valid points, 0 for invalid points) ===")
        all_pts_for_mask = torch.cat([M2_positive_position_tensor, M2_negative_position_tensor], dim=0)
        mask_labels = torch.cat([
            torch.ones(len(M2_positive_position_tensor), 1, dtype=torch.float32),
            torch.zeros(len(M2_negative_position_tensor), 1, dtype=torch.float32)
        ], dim=0)
        n_mask_samples = len(all_pts_for_mask)
        n_mask_val = int(n_mask_samples * args.val_split)
        n_mask_train = n_mask_samples - n_mask_val
        mask_indices = torch.randperm(n_mask_samples)
        mask_train_indices = mask_indices[:n_mask_train]
        mask_val_indices = mask_indices[n_mask_train:]
        M2_train_pts = all_pts_for_mask[mask_train_indices]
        M2_train_labels = mask_labels[mask_train_indices]
        M2_val_pts = all_pts_for_mask[mask_val_indices]
        M2_val_labels = mask_labels[mask_val_indices]

        _,_,pos_weight_M2 = pretrain_field(model, 'M2', M2_train_pts, M2_train_labels, M2_val_pts, M2_val_labels,
                       config=cfg.TRAINING_CONFIG, experiment=experiment)
        m2_pos_weight_note = ""

    print(f"Pos weight M1: {pos_weight_M1}")
    eval_result_M1 = evaluate_mask_field(model, M1_val_pts, M1_val_labels, field_type='fieldM1')
    print(f"Pos weight M2{m2_pos_weight_note}: {pos_weight_M2}")
    eval_result_M2 = evaluate_mask_field(model, M2_val_pts, M2_val_labels, field_type='fieldM2')
    
    # --- 5. Save model and normalization parameters ---
    os.makedirs(os.path.dirname(os.path.abspath(pretrain_model_path)), exist_ok=True)
    torch.save(model.state_dict(), pretrain_model_path)
    print(f"\n✅ Pre-trained model saved to '{pretrain_model_path}'")
if __name__ == '__main__':
    main()
