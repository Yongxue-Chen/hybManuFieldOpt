"""Template configuration for a new FieldOpt-HM model.

Copy this file to ``configs/config_multi_field_<model>.py`` and edit the
model-specific values before running the pipeline. The filename suffix must
match the ``--model_name`` argument used by preprocessing, pretraining,
optimization, and postprocessing scripts.

Required model-specific edits:
- MAX_TIME: planning horizon. Start with an estimate, then update it after the
  voxel-based initial solution is generated.
- SCALE: physical scale used to normalize manufacturing dimensions.
- MANU_CONFIG: AM/SM process parameters, especially SMToolParas.
- WEIGHTS and BAYESIAN_*: loss weights and search ranges for optimization.
"""

import torch
import numpy as np

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# parameters for manufacturing and loss function
MAX_TIME = 200
MARGIN = 0.01
MARGIN_NEW = 0.01

# ==============================================================================
# Model Architecture Parameters
# ==============================================================================

# Multi-resolution hash encoding parameters for the first field
L1 = 16          # Number of levels in the hash table
T1 = 2**16       # Hash table size per level
F1 = 2           # Feature dimension per hash table entry
N_min1 = 16      # Resolution of the coarsest level
N_max1 = 512   # Resolution of the finest level
N_NEURONS1 = 64
N_HIDDEN_LAYERS1 = 2

# Multi-resolution hash encoding parameters for the second field
L2 = 16
T2 = 2**19
F2 = 2
N_min2 = 16
N_max2 = 512   # Resolution of the finest level
N_NEURONS2 = 64
N_HIDDEN_LAYERS2 = 2

# Multi-resolution hash encoding parameters for the third field
L3 = 16
T3 = 2**19
F3 = 2
N_min3 = 16
N_max3 = 4096   # Resolution of the finest level
N_NEURONS3 = 64
N_HIDDEN_LAYERS3 = 2

# Multi-resolution hash encoding parameters for the mask field
# Reduced capacity to prevent overfitting
LM1 = 16  # Reduced from 16 to 12 levels
TM1 = 2**16  # Reduced hash table size
FM1 = 2
N_minM1 = 16
N_maxM1 = 1024
N_NEURONSM1 = 64
N_HIDDEN_LAYERSM1 = 2

LM2 = 16
TM2 = 2**16
FM2 = 2
N_minM2 = 16
N_maxM2 = 2048
N_NEURONSM2 = 64
N_HIDDEN_LAYERSM2 = 2

# Multi-resolution hash encoding parameters for the spatio-temporal density field
LDT = 16           # Hash encoding levels
TDT = 2**19        # Hash table size
FDT = 2            # Features per level
N_minDT = 16       # Minimum resolution
N_maxDT = 4096      # Maximum resolution
N_NEURONSDT = 64   # MLP neurons
N_HIDDEN_LAYERSDT = 2  # MLP hidden layers
NUM_TIME_FREQUENCIES_DT = 10  # Temporal encoding frequencies (output: 2*10=20 dims)
DROPOUT_RATE_FIELDDT = 0.1  # Dropout rate for regularization

# ==============================================================================
# Pre-training Configuration
# ==============================================================================

TRAINING_CONFIG = {
    'pretrain_epochs': 10000,
    'field1_params': {
        'lr': 5e-4,  # Reduced learning rate for better generalization
        'weight_decay': 1e-5,  # Added L2 regularization
        'dropout': 0.0,  # Added dropout for regularization
        'noise_std': 0.001
    },
    'field2_params': {
        'lr': 5e-4,  # Restored: field2 was already performing well
        'weight_decay': 5e-4,  # Restored original value
        'dropout': 0.0,  # Restored original value
        'noise_std': 0.004
    },
    'field3_params': {
        'lr': 5e-4,  # Reduced learning rate
        'weight_decay': 1e-4,  # Added L2 regularization
        'dropout': 0.0,  # Added dropout for regularization
        'noise_std': 0.002
    },
    'fieldM1_params': {
        'lr': 2e-4,
        'weight_decay': 8e-4,
        'dropout': 0.0,
        'noise_std': 0.002
    },
    'fieldM2_params': {
        'lr': 2e-4,
        'weight_decay': 8e-4,
        'dropout': 0.0,
        'noise_std': 0.002
    },
    'fieldDT_params': {
        'lr': 3e-4,
        'weight_decay': 5e-4,
        'dropout': 0.0,
        'noise_std': 0.002
    },
    'early_stopping': {
        'patience': 50,  # Increased patience to allow more training with regularization
        'min_delta': 1e-5  # Relaxed min_delta to account for slower improvement
    },
    'scheduler_params1': {
        'type': 'ReduceLROnPlateau',
        'patience': 15,  # Increased patience
        'factor': 0.5,
        'min_lr': 1e-7,
    },
    'scheduler_params2': {
        'type': 'ReduceLROnPlateau',
        'patience': 15,  # Increased patience
        'factor': 0.5,
        'min_lr': 1e-7
    },
    'scheduler_params3': {
        'type': 'ReduceLROnPlateau',
        'patience': 15,  # Added scheduler for field3
        'factor': 0.5,
        'min_lr': 1e-7
    },
    'scheduler_paramsM1': {
        'type': 'ReduceLROnPlateau',
        'patience': 10,  # Slightly reduced patience for faster adaptation
        'factor': 0.5,
        'min_lr': 1e-7,
        # Optional. Keeps the learning rate constant and suspends both the
        # scheduler and early stopping for this many epochs. Useful when a mask
        # field cold-starts slowly (balanced 50/50 batches can cancel gradients).
        # Omit or set to 0 to disable warmup. Validation runs every 5 epochs, so
        # use a multiple of 5 for the end-of-warmup state reset to take effect.
        'warmup_epochs': 0
    },
    # fieldM2 is only pretrained separately when it is not shape-compatible with
    # fieldM1 (e.g. when N_maxM1 != N_maxM2). Both mask fields see identical
    # training data, so when the two encoders match, main_pre_train.py copies the
    # pretrained fieldM1 weights into fieldM2 and ignores these parameters.
    'scheduler_paramsM2': {
        'type': 'ReduceLROnPlateau',
        'patience': 10,  # Slightly reduced patience for faster adaptation
        'factor': 0.5,
        'min_lr': 1e-7,
        'warmup_epochs': 0
    },
    'scheduler_paramsDT': {
        'type': 'ReduceLROnPlateau',
        'patience': 15,  # Scheduler for spatio-temporal density field
        'factor': 0.5,
        'min_lr': 1e-7
    },
    'accumulation_steps': 1,
    'visualization': {
        'slice_axis': 'z',
        'slice_index': 50
    }
}

PRETRAIN_BATCH_SIZE = int(131072/5)

# ==============================================================================
# Data Balancing Parameters
# ==============================================================================

PARA_BALANCING = {
    'enabled': True,
    # Quantile (0-1) of nearest positive distance above which negatives are considered "far"
    'far_distance_quantile': 0.8,
    # Fraction of the far negatives to keep (rest will be dropped)
    'far_keep_fraction': 0.3,
    # Optional cap ensuring we do not keep more than this multiple of positives
    'max_ratio': 3.0,
    'random_seed': 42
}

FIELD3_BOUNDARY_AUG = {
    'enabled': True,
    # Added boundary samples = original boundary-labeled field3 samples * boundary_sample_multiplier.
    # 1.0 means add one extra copy worth of boundary samples.
    'boundary_sample_multiplier': 100.0,
    # Optional fallback: can be an int or a dict with x_min/x_max/y_min/y_max counts.
    # Only used when boundary_sample_multiplier is None.
    'samples_per_face': None,
    'random_seed': 42,
}
# ==============================================================================
# Basic Training Parameters
# ==============================================================================

DEFAULT_BATCH_SIZE = 524288 * 8

# ==============================================================================
# Joint Training Parameters
# ==============================================================================

JOINT_VIRTUAL_EPOCHS = 200  # Increased training epochs
JOINT_TRAIN_LR = 1e-4      # Increased initial learning rate
FIELD3_LR_RATIO = 2.0
WARMUP_EPOCHS = 2

MAX_CLIP_NORM = 0.5

ADAMW_BETAS = (0.9, 0.99)
ADAMW_EPS = 1e-15

WEIGHT_DECAY = 1e-4

# WEIGHTS = {
#     'final_state': 0.01,
#     'allowed_final_state': 0.065,

#     'self_support': 10.0,

#     'AM_Collision_Free': 10.0,

#     'SM_Collision_Free': 0.9,
#     'SM_Below_AABB': 100.0,

#     'operation_volume': 0.6,
#     'allowed_operation_volume': 0.5,
    
#     'structure': 5.0,
#     'structure_simp_penalty': 0.0001,
# }

WEIGHTS = {
    'final_state': 1.0,
    'allowed_final_state': 0.02,

    'self_support': 35.0,
    'allowed_self_support': 0.001,

    'AM_Collision_Free': 69.0,
    'allowed_AM_Collision_Free': 0.0005,

    'SM_Collision_Free': 3.5,
    'SM_Below_AABB': 100.0,

    'operation_volume': 1.5,
    'allowed_operation_volume': 0.15,
    
    'structure': 2.0,
    'allowed_structure': 0.001,
    'structure_simp_penalty': 0.0001,
}

BAYESIAN_WEIGHT_SEARCH_SPACE = {
    'self_support': {
        'type': 'discrete_range',
        'low': 1.0,
        'high': 100.0,
        'step': 1.0,
    },
    'AM_Collision_Free': {
        'type': 'discrete_range',
        'low': 1.0,
        'high': 100.0,
        'step': 1.0,
    },
    'structure': {
        'type': 'discrete_range',
        'low': 0.1,
        'high': 10.0,
        'step': 0.1,
    },
}

BAYESIAN_METRIC_TARGETS = {
    'final_state_accuracy':      (0.99, 'higher'),   # 98% of empty points correctly classified
    'self_support_ratio':        (0.99, 'higher'),   # 95% of AM points have support
    'AM_collision_free_ratio':   (0.99, 'higher'),   # 99% of AM points collision-free
    'SM_collision_free_ratio':   (0.90, 'higher'),   # 95% of SM points collision-free
    'operation_volume_ratio':    (0.90, 'higher'),   # 90% of empty points not over-added
    'structure_normalized':      (4e-8, 'lower'),  # per-grid structure penalty threshold
}

TERM_SHORT_NAME = {
    'final_state': "FS",
    'self_support_avg': "SS_avg",
    'self_support_max': "SS_max",
    'AM_Collision_Free': "AT",
    'SM_Collision_Free': "ST",
    'operation_volume': "Vol",
    'structure': "Str",
    'structure_simp_penalty': "StrSIMP",
}

SCALE = 100.0
MANU_CONFIG = {
    'hSupport': 0.5/SCALE,
    'supportAngle': np.pi/4,
    'nSupport': 15,

    'AMConeHeight': 50.0/SCALE,
    'AMConeHalfAngle': np.pi/4,
    'AMCollisionMargin': 1.0/SCALE,
    'nAMCone': 9,
    'nAMCylinder': 2,

    'SMToolParas': {
        'SMTipDiameter': 1.0/SCALE,
        'SMShankDiameter': 1.0/SCALE,
        'SMToolLength': 12.0/SCALE,
        'SMHolderDiameter': 100.0/SCALE,
        'SMHolderLength': 70.0/SCALE,
        'SMCollisionMargin': 0.2/SCALE, #only for post-process
        'nSMTip': 2,
        'nSMTool': 4,
        'nSMHolder': 4
    },

    'min_layer_AM': 0.5/SCALE,
    'max_layer_AM': 1.3/SCALE,
    'min_layer_SM': 0.4/SCALE,
    'max_layer_SM': 0.9/SCALE,
    'threshold_SM': 10.0/SCALE,
}

MANU_CONFIG_CHECK = {
    'hSupport': MANU_CONFIG['hSupport'],
    'supportAngle': MANU_CONFIG['supportAngle'],
    'nSupport': 10,

    'AMConeHeight': MANU_CONFIG['AMConeHeight'],
    'AMConeHalfAngle': MANU_CONFIG['AMConeHalfAngle'],
    'AMCollisionMargin': MANU_CONFIG['AMCollisionMargin'],
    'nAMCone': 100,
    'nAMCylinder': 100,

    'SMToolParas': {
        'SMTipDiameter': MANU_CONFIG['SMToolParas']['SMTipDiameter'],
        'SMShankDiameter': MANU_CONFIG['SMToolParas']['SMShankDiameter'],
        'SMToolLength': MANU_CONFIG['SMToolParas']['SMToolLength'],
        'SMHolderDiameter': MANU_CONFIG['SMToolParas']['SMHolderDiameter'],
        'SMHolderLength': MANU_CONFIG['SMToolParas']['SMHolderLength'],
        'SMCollisionMargin': MANU_CONFIG['SMToolParas']['SMCollisionMargin'],
        'nSMTip': 50,
        'nSMTool': 100,
        'nSMHolder': 200
    }
}

# Improved Learning Rate Schedule
JOINT_SCHEDULER_CONFIG = {
    'type': 'CosineAnnealingWarmRestarts',
    'T_0': 1000,        # 以“步”为单位：≈10个epoch（你每epoch=100步）
    'T_mult': 2,
    'eta_min': 1e-7,
    'warmup_epochs': 5
}

REDUCE_LRON_PLATEAU_CONFIG = {
    'factor': 0.5,
    'patience': 4,
    'min_lr': 1e-7,
    'threshold': 1e-3,
    'threshold_mode': 'rel'
}

# Early Stopping Configuration (enhanced for training-only mode)
JOINT_EARLY_STOPPING = {
    'min_delta': 1e-6,        # 最小改进阈值
    'restore_best_weights': True,
    
    # 多种早停策略
    'loss_patience': 20,      # 基于损失的patience
    'lr_threshold': 1e-7,     # 学习率阈值早停
    
    # 趋势分析参数
    'window_size': 10,        # 滑动窗口大小
    'convergence_threshold': 1e-8,  # 收敛阈值
}

# Batch Processing Optimization
JOINT_BATCH_CONFIG = {
    'batch_size': int(2097152/6),
    'accumulation_steps': 1,
    'steps_per_epoch': 50
}

BATCH_SIZE_CHECK = int(50000)

PARAS_STRUCTURE = {
    'max_resolution': 64,
    'sample_each_grid': 3,

    'time_check_size': 32,
    'cg_iter': 100,
    'cg_tol': 1e-6,
    'e0': 10.0,

    'grad_clip_threshold': 1.0,
    'grad_scale': 1.0,

    'logitDensity': True,
    'densitySharpness': 1.0,
    'use_total_mass': 1.0,
    'max_displacement': 0.1,

    'use_sm_cutting_force_structure': True,
    'sm_cutting_force_magnitude': 5.0,
    'sm_max_displacement_norm': 0.05,
}
