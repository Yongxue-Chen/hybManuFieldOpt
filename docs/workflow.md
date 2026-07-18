# Workflow

This project follows a hybrid manufacturing optimization pipeline.

## 1. Support Generation

Input: a raw STL model.

The preprocessing stage adds support geometry to the input STL. Relevant code lives in:

- `preprocess.py`
- `fieldopt/preprocessing/`

Expected outputs are support/body STL variants and removal-plan metadata. Large generated geometry is not committed to the source release.

## 2. Voxelization and SDF Preparation

The support-augmented STL is used in two parallel preparation steps:

- Voxelization, used to generate voxel data for an initial field.
- SDF training, used to build an implicit geometry representation.

Relevant code lives in:

- `fieldopt/geometry/voxel/voxelization.py`
- `fieldopt/geometry/sdf/train_sdf.py`
- `fieldopt/geometry/sdf/train_sdf_v2.py`
- `fieldopt/geometry/sdf/train_sdf_mesh_to_sdf.py`
- `fieldopt/geometry/implicit.py`

The current codebase supports both voxel-backed implicit functions and neural/SIREN SDF checkpoints through `fieldopt/geometry/backend.py`. Use `--geometry_backend voxel_artifact` for saved voxel artifacts, `--geometry_backend voxel` to rebuild from STL, or `--geometry_backend siren` with `--geometry_artifact_path` for a trained SDF checkpoint.

## 3. External Voxel-Based Initial Field Generation

The voxelized result is passed to an external voxel-based method to generate the initial field.

The external implementation is not included in this source tree. Use the companion repository:

- https://github.com/Yongxue-Chen/hybManuAccEro

Expected outputs should be placed under `model_data/initial_fields/<model>/`.

## 4. Configuration

Each model/task needs a `configs/config_multi_field_<name>.py` file, loaded as `configs.config_multi_field_<name>`.

These config files stay in the `configs/` package because the training scripts load them dynamically with `importlib`.

## 5. Pretraining

The voxelized result and generated initial field are used to pretrain an initial network.

Relevant entry point:

- `main_pre_train.py`

## 6. Bayesian Optimization and Final Training

The pretrained network is refined through Bayesian optimization, which calls `main_optimize.py` to train candidate models and select a final network. Geometry representation is selected with the shared `--geometry_backend` / `--geometry_artifact_path` flags.

Relevant entry points:

- `bayesian_weight_optimizer.py`
- `bayesian_weight_optimizer_constrained.py`
- `main_optimize.py`

## 7. Postprocessing

The final network is passed to postprocessing to generate layers and tool paths.

Relevant entry point and code live in:

- `postprocess.py`
- `fieldopt/postprocessing/`

Generated layers, paths, checkpoints, and related output artifacts belong under `output/` and are ignored by Git.
