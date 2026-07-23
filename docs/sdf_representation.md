# SDF and Geometry Representation

The codebase now routes training, evaluation, Bayesian optimization, and hybrid postprocessing through a shared geometry backend loader:

- `fieldopt/geometry/backend.py`

The returned callable keeps the existing `H_gpu` / `check_func` interface, so downstream loss and postprocessing code does not need to know whether the geometry comes from a voxel artifact or a neural SDF checkpoint.

## Backends

- `voxel_artifact` (default): loads `model_data/implicit_representations/<model>/<model>_voxel_implicit.pt`.
- `voxel`: rebuilds the voxel-backed implicit function directly from `stlFiles/<model>.stl`.
- `siren` / `neural_sdf`: loads `model_data/implicit_representations/<model>/<model>_siren_sdf.pt`, or an explicit checkpoint can be passed with `--geometry_artifact_path`.

## Command Examples

Default saved voxel artifact:

```bash
conda run -n myenv python main_optimize.py --model_name bracket
```

Rebuild voxel geometry from STL:

```bash
conda run -n myenv python main_optimize.py \
    --model_name bracket \
    --geometry_backend voxel
```

Use neural/SIREN SDF checkpoint:

```bash
conda run -n myenv python main_optimize.py \
    --model_name bracket \
    --geometry_backend siren \
    --geometry_artifact_path model_data/implicit_representations/bracket/bracket_sdf.pt
```

The same flags are available in `evaluate_model.py`, `bayesian_weight_optimizer.py`, `bayesian_weight_optimizer_constrained.py`, and `fieldopt.postprocessing`.

## Release Notes

The current `siren` backend uses the existing neural SDF loader in `fieldopt/geometry/implicit.py`. If a stricter SIREN checkpoint format is introduced later, only `fieldopt/geometry/backend.py` and the SDF loader need to be extended; the main optimization and postprocessing entry points should remain unchanged.
