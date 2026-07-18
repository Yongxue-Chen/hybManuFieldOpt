#!/usr/bin/env python3
"""Verify the FieldOpt-HM runtime, CUDA hash grids, and all five field networks."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


REQUIRED_MODULES = (
    "numpy",
    "tqdm",
    "PIL",
    "matplotlib",
    "trimesh",
    "manifold3d",
    "pyvista",
    "scipy",
    "skimage",
    "pyglet",
    "optuna",
    "pyvistaqt",
    "mesh_to_sdf",
    "PySide6",
    "open3d",
    "siren_pytorch",
    "igl",
    "stl",
    "embreex",
    "rtree",
)

PROJECT_MODULES = (
    "fieldopt.models",
    "fieldopt.losses.loss_multi_field",
    "fieldopt.geometry.backend",
    "fieldopt.geometry.implicit",
    "fieldopt.geometry.voxel.voxelization",
    "fieldopt.preprocessing",
    "fieldopt.postprocessing",
)


def check_imports() -> None:
    for name in REQUIRED_MODULES + PROJECT_MODULES:
        importlib.import_module(name)
        print(f"[OK] import {name}")

    for unnecessary in ("taichi", "PyQt5"):
        if importlib.util.find_spec(unnecessary) is not None:
            raise AssertionError(f"unnecessary core dependency is installed: {unnecessary}")
        print(f"[OK] unnecessary dependency absent: {unnecessary}")


def check_geometry_packages() -> None:
    import trimesh
    from rtree import index

    if not trimesh.ray.has_embree:
        raise AssertionError("trimesh did not detect Embree acceleration")

    left = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    right = left.copy()
    right.apply_translation((0.5, 0.0, 0.0))
    merged = trimesh.boolean.union([left, right], engine="manifold")
    if merged is None or not merged.is_volume:
        raise AssertionError("manifold3d boolean union failed")

    tree = index.Index()
    tree.insert(0, (0.0, 0.0, 1.0, 1.0))
    if list(tree.intersection((0.5, 0.5, 0.6, 0.6))) != [0]:
        raise AssertionError("rtree query failed")
    print("[OK] Embree, manifold3d, and rtree runtime checks")


def check_cuda_and_tcnn():
    import torch
    import tinycudann as tcnn

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise AssertionError(
            f"expected compute capability 12.0, got {torch.cuda.get_device_capability(0)}"
        )

    device = torch.device("cuda:0")
    encoding = tcnn.Encoding(
        n_input_dims=3,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": 4,
            "n_features_per_level": 2,
            "log2_hashmap_size": 10,
            "base_resolution": 4,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
        },
    ).to(device)
    points = torch.rand(256, 3, device=device, requires_grad=True)
    encoded = encoding(points)
    encoded.float().square().mean().backward()
    if points.grad is None or not torch.isfinite(points.grad).all():
        raise AssertionError("tiny-cuda-nn input gradient is missing or non-finite")
    parameter_grads = [p.grad for p in encoding.parameters() if p.grad is not None]
    if not parameter_grads or not all(torch.isfinite(g).all() for g in parameter_grads):
        raise AssertionError("tiny-cuda-nn parameter gradients are missing or non-finite")

    print(
        "[OK] CUDA/tiny-cuda-nn forward+backward:",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
        tuple(encoded.shape),
    )
    return torch, device


def check_five_field_networks(torch, device) -> None:
    from fieldopt.models.multi_field import MultiFieldModel

    shared = dict(L=4, T=2**10, F=2, N_min=4, N_max=32, neurons=16, layers=1)
    model = MultiFieldModel(
        L1=shared["L"], T1=shared["T"], F1=shared["F"],
        N_min1=shared["N_min"], N_max1=shared["N_max"],
        n_neurons1=shared["neurons"], n_hidden_layers1=shared["layers"], max_time=10.0,
        L2=shared["L"], T2=shared["T"], F2=shared["F"],
        N_min2=shared["N_min"], N_max2=shared["N_max"],
        n_neurons2=shared["neurons"], n_hidden_layers2=shared["layers"],
        L3=shared["L"], T3=shared["T"], F3=shared["F"],
        N_min3=shared["N_min"], N_max3=shared["N_max"],
        n_neurons3=shared["neurons"], n_hidden_layers3=shared["layers"],
        LM1=shared["L"], TM1=shared["T"], FM1=shared["F"],
        N_minM1=shared["N_min"], N_maxM1=shared["N_max"],
        n_neuronsM1=shared["neurons"], n_hidden_layersM1=shared["layers"],
        LM2=shared["L"], TM2=shared["T"], FM2=shared["F"],
        N_minM2=shared["N_min"], N_maxM2=shared["N_max"],
        n_neuronsM2=shared["neurons"], n_hidden_layersM2=shared["layers"],
        LDT=shared["L"], TDT=shared["T"], FDT=shared["F"],
        N_minDT=shared["N_min"], N_maxDT=shared["N_max"],
        n_neuronsDT=shared["neurons"], n_hidden_layersDT=shared["layers"],
        bounding_box=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        device=str(device),
        dropout_rate_field1=0.0,
        dropout_rate_field2=0.0,
        dropout_rate_field3=0.0,
        dropout_rate_fieldM1=0.0,
        dropout_rate_fieldM2=0.0,
        dropout_rate_fieldDT=0.0,
    ).train()

    expected_shapes = {
        "field1": (256, 1),
        "field2": (256, 1),
        "field3": (256, 6),
        "fieldM1": (256, 1),
        "fieldM2": (256, 1),
    }
    field_modules = {
        "field1": model.field1,
        "field2": model.field2,
        "field3": model.field3,
        "fieldM1": model.fieldM1,
        "fieldM2": model.fieldM2,
    }

    for field_name, expected_shape in expected_shapes.items():
        model.zero_grad(set_to_none=True)
        points = torch.rand(256, 3, device=device, requires_grad=True)
        output = model(points, field_type=field_name)
        if tuple(output.shape) != expected_shape:
            raise AssertionError(f"{field_name}: expected {expected_shape}, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{field_name}: non-finite forward output")
        output.float().square().mean().backward()
        grads = [p.grad for p in field_modules[field_name].parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError(f"{field_name}: missing or non-finite parameter gradient")
        print(f"[OK] {field_name} forward+backward {tuple(output.shape)}")

    all_outputs = model(torch.rand(256, 3, device=device), field_type="all")
    if tuple(tuple(value.shape) for value in all_outputs) != tuple(expected_shapes.values()):
        raise AssertionError("combined five-field forward returned unexpected shapes")
    torch.cuda.synchronize(device)
    print("[OK] combined five-field CUDA forward")


def main() -> int:
    check_imports()
    check_geometry_packages()
    torch, device = check_cuda_and_tcnn()
    check_five_field_networks(torch, device)
    print("ALL ENVIRONMENT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
