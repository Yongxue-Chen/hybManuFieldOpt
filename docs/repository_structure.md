# Repository Structure

## User-Facing Entry Points

These scripts remain at the repository root because users run them directly:

- `preprocess.py` - run support generation and preprocessing for a selected model.
- `main_pre_train.py` - pretrain the neural fields from an initial field.
- `main_optimize.py` - run the main continuous optimization.
- `bayesian_weight_optimizer.py` - search loss weights and call the optimizer.
- `bayesian_weight_optimizer_constrained.py` - constrained variant of Bayesian weight search.
- `build_voxel_implicit.py` - build and save a voxel implicit representation.
- `evaluate_model.py` - evaluate a trained checkpoint.
- `postprocess.py` - generate layers and tool paths from a trained checkpoint.

Configuration modules live in:

- `configs/config_multi_field_*.py`

## Internal Package

Most implementation code lives under `fieldopt/`:

- `fieldopt/preprocessing/` - support generation, overhang extraction, and preprocessing geometry utilities.
- `fieldopt/geometry/` - geometry representations and mesh utilities.
  - `fieldopt/geometry/voxel/` - voxelization code.
  - `fieldopt/geometry/sdf/` - neural/SIREN SDF and inclusion training code.
  - `fieldopt/geometry/mesh/` - STL repair, scaling, TPMS/lattice generation, and mesh helpers.
  - `fieldopt/geometry/smoothing/` - voxel-to-STL and mesh smoothing utilities.
  - `fieldopt/geometry/backend.py` - shared geometry backend loader used by training, evaluation, Bayesian optimization, and postprocessing.
  - `fieldopt/geometry/implicit.py` - voxel-backed implicit geometry and streamed training data.
- `fieldopt/models/` - instant-NGP-style hash-grid neural fields and MLP components.
- `fieldopt/losses/` - optimization losses and structural loss utilities.
- `fieldopt/postprocessing/` - layer/path generation, tool orientation, collision checking, and visualization.
- `fieldopt/utils/` - shared data loading and training utility code.

## Data and Output Placeholders

- `model_data/` - model inputs and generated artifacts organized by pipeline stage; the public bracket STL lives under `model_data/target_shapes/`.
- `initialFields/` - local initial-field files produced by the companion voxel planner.
- `stlFiles/` - local STL files expected by default commands.
- `output/` - generated checkpoints, geometry artifacts, evaluation files, paths, and related outputs.
- `examples/` - runnable examples and example-specific assets.
- `assets/` - lightweight README and documentation assets.

## External Tool

Initial-field generation is provided by the companion repository:

- https://github.com/Yongxue-Chen/hybManuAccEro
