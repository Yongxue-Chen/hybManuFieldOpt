import torch
import numpy as np

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# parameters for manufacturing and loss function
MAX_TIME = 200 # max time multiply by 2*MARGIN
MARGIN = 0.01
MARGIN_NEW = 0.01

HRES = 128

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
N_max2 = 4096   # Resolution of the finest level
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
N_maxM1 = 2048
N_NEURONSM1 = 64
N_HIDDEN_LAYERSM1 = 2

LM2 = 16
TM2 = 2**16
FM2 = 2
N_minM2 = 16
N_maxM2 = 4096
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
        'min_lr': 1e-7
    },
    'scheduler_paramsM2': {
        'type': 'ReduceLROnPlateau',
        'patience': 10,  # Slightly reduced patience for faster adaptation
        'factor': 0.5,
        'min_lr': 1e-7
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

PRETRAIN_BATCH_SIZE = 131072

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
# ==============================================================================
# Basic Training Parameters
# ==============================================================================

DEFAULT_BATCH_SIZE = 524288 * 8

# ==============================================================================
# Joint Training Parameters
# ==============================================================================

JOINT_VIRTUAL_EPOCHS = 500  # Increased training epochs
JOINT_TRAIN_LR = 1e-4      # Increased initial learning rate
WARMUP_EPOCHS = 5         # Number of epochs for warmup (no LR adjustment or early stopping)
FIELD3_LR_RATIO = 2.0

MAX_CLIP_NORM = 0.5

ADAMW_BETAS = (0.9, 0.99)
ADAMW_EPS = 1e-15

WEIGHT_DECAY = 1e-4

WEIGHTS = {
    'final_state': 2.0,
    'allowed_final_state': 0.2,

    'self_support': 1.0,

    'AM_Collision_Free': 6.0,
    'SM_Collision_Free': 0.9,

    'operation_volume': 1.0,
    'allowed_operation_volume': 1.5,

    'structure': 0.1,
    'structure_simp_penalty': 0.0001,
}

# ==============================================================================
# Bayesian Optimization Configuration
# ==============================================================================

BAYESIAN_WEIGHT_SEARCH_SPACE = {
    'final_state':              (0.05, 10.0, True),
    # 'allowed_final_state' is fixed (not optimized)
    'self_support':             (0.1, 10.0, True),
    'AM_Collision_Free':        (0.1,  20.0, True),
    'SM_Collision_Free':        (0.1, 10.0, True),
    'operation_volume':         (0.1, 10.0, True),
    # 'allowed_operation_volume' is fixed (not optimized)
    'structure':                (0.01, 5.0, True),
    # 'structure_simp_penalty':   (1e-5,  1e-3, True),
}

BAYESIAN_LOSS_THRESHOLDS = {
    'final_state':         0.25,
    'self_support_avg':    0.5,
    'AM_Collision_Free':   0.005,
    'SM_Collision_Free':   0.25,
    'operation_volume':    1.8,
    'structure':           10.0,

    # 'self_support_max':    2.0,
    # 'structure_simp_penalty': 0.5,
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
    'hSupport': 0.6/SCALE,
    'supportAngle': np.pi/4,
    'nSupport': 15,

    'min_layer_AM': 1.0/SCALE,
    'max_layer_AM': 2.0/SCALE,

    'AMConeHeight': 50.0/SCALE,
    'AMConeHalfAngle': np.pi/4,
    'nAMCone': 5,
    'nAMCylinder': 5,

    'SMToolParas': {
        'SMToolDiameter': 2.0/SCALE,
        'SMToolLength': 20.0/SCALE,
        'SMHolderDiameter': 100.0/SCALE,
        'SMHolderLength': 200.0/SCALE,
        'SMAllowedCollisionHeight': 0.2/SCALE,
        'nSMTip': 6,
        'nSMTool': 5,
        'nSMHolder': 4
    },

    'min_layer_SM': 0.05/SCALE,
    'max_layer_SM': 0.2/SCALE,
    'threshold_SM': 10.0/SCALE,
}

MANU_CONFIG_CHECK = {
    'hSupport': MANU_CONFIG['hSupport'],
    'supportAngle': MANU_CONFIG['supportAngle'],
    'nSupport': 10,

    'AMConeHeight': MANU_CONFIG['AMConeHeight'],
    'AMConeHalfAngle': MANU_CONFIG['AMConeHalfAngle'],
    'nAMCone': 100,
    'nAMCylinder': 100,

    'SMToolParas': {
        'SMToolDiameter': MANU_CONFIG['SMToolParas']['SMToolDiameter'],
        'SMToolLength': MANU_CONFIG['SMToolParas']['SMToolLength'],
        'SMHolderDiameter': MANU_CONFIG['SMToolParas']['SMHolderDiameter'],
        'SMHolderLength': MANU_CONFIG['SMToolParas']['SMHolderLength'],
        'SMAllowedCollisionHeight': MANU_CONFIG['SMToolParas']['SMAllowedCollisionHeight'],
        'nSMTip': 0,
        'nSMTool': 100,
        'nSMHolder': 500
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
    'batch_size': int(2097152/15),
    'accumulation_steps': 1,
    'steps_per_epoch': 50
}

BATCH_SIZE_CHECK = int(50000)

PARAS_STRUCTURE = {
    'max_resolution': 50,
    'sample_each_grid': 2,

    'time_check_size': 16,
    'cg_iter': 100,
    'cg_tol': 1e-6,
    # 'e0': 500.0,
    
    
    'e0': 10.0,

    'grad_clip_threshold': 1.0,
    # 'grad_scale': 1e-5,
    'grad_scale': 1.0,

    'max_displacement': 0.6,
}