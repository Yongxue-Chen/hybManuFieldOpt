import os
import sys
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import numpy as np


@dataclass
class PostprocessContext:
    """Unified context holding everything needed for hybrid manufacturing post-processing."""
    model: Any
    config: Any
    spaceBox: np.ndarray
    check_func: Callable
    scale: float
    device: str
    max_time: float
    manu_config: dict
    stl_path: str


def load_model_and_config(
    config_name: str,
    model_path: str,
    stl_dir: str = 'stlFiles',
    stl_path: Optional[str] = None,
    device: Optional[str] = None,
    voxel_resolution: int = 512,
    geometry_backend: str = 'voxel_artifact',
    geometry_artifact_path: Optional[str] = None,
) -> PostprocessContext:
    """
    Load a trained MultiFieldModel and its associated config.

    Args:
        config_name: Name suffix for config module, e.g. 'bracket' loads
                     ``configs.config_multi_field_bracket``.
        model_path:  Path to the saved ``.pth`` model weights.
        stl_dir:     Directory containing ``<config_name>.stl``.
        stl_path:    Optional explicit STL path. Overrides ``stl_dir`` lookup.
        device:      Override device; defaults to the config's DEVICE.
        voxel_resolution: Resolution used when rebuilding a voxel implicit function.
        geometry_backend: 'voxel_artifact', 'voxel', or 'siren'.
        geometry_artifact_path: Optional saved voxel artifact or neural-SDF checkpoint path.

    Returns:
        PostprocessContext with all fields populated.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    cfg = importlib.import_module(f'configs.config_multi_field_{config_name}')

    if device is None:
        device = cfg.DEVICE

    from fieldopt.geometry.backend import load_geometry_function
    from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
    from fieldopt.models.multi_field import MultiFieldModel

    stl_path = stl_path or os.path.join(stl_dir, f'{config_name}.stl')
    if not os.path.isabs(stl_path):
        stl_path = os.path.join(project_root, stl_path)

    # aabb_min, aabb_max = get_sdf_aabb(stl_path)
    # spaceBox = np.array([aabb_min, aabb_max])
    _, p_min, spaceBox = get_normalization_parameters(stl_path)
    spaceBox[0] = spaceBox[0] - p_min
    spaceBox[1] = spaceBox[1] - p_min
    

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
        bounding_box=spaceBox, device=device,
        dropout_rate_field1=cfg.TRAINING_CONFIG['field1_params']['dropout'],
        dropout_rate_field2=cfg.TRAINING_CONFIG['field2_params']['dropout'],
        dropout_rate_field3=cfg.TRAINING_CONFIG['field3_params']['dropout'],
        dropout_rate_fieldM1=cfg.TRAINING_CONFIG['fieldM1_params']['dropout'],
        dropout_rate_fieldM2=cfg.TRAINING_CONFIG['fieldM2_params']['dropout'],
        num_time_frequencies_DT=cfg.NUM_TIME_FREQUENCIES_DT,
        dropout_rate_fieldDT=cfg.DROPOUT_RATE_FIELDDT,
    ).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found: {model_path}")
    load_result = model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True),
        strict=False,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Trained checkpoint does not match the current model: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()

    check_func = load_geometry_function(
        backend=geometry_backend,
        stl_path=stl_path,
        model_name=config_name,
        artifact_path=geometry_artifact_path,
        device=device,
        voxel_resolution=voxel_resolution,
    )

    manu = getattr(cfg, 'MANU_CONFIG_CHECK', cfg.MANU_CONFIG)
    if isinstance(manu, dict):
        manu = dict(manu)
    # Layer-height keys must come from MANU_CONFIG so that postprocess layer
    # generation is stable and not affected by check-only config variants.
    base_manu = getattr(cfg, 'MANU_CONFIG', manu)
    for k in ('min_layer_AM', 'max_layer_AM', 'min_layer_SM', 'max_layer_SM'):
        if isinstance(base_manu, dict) and k in base_manu:
            manu[k] = base_manu[k]

    return PostprocessContext(
        model=model,
        config=cfg,
        spaceBox=spaceBox,
        check_func=check_func,
        scale=cfg.SCALE,
        device=device,
        max_time=cfg.MAX_TIME,
        manu_config=manu,
        stl_path=stl_path,
    )
