"""Unified implicit geometry backend loading.

All training, evaluation, and postprocessing entry points should use this module
instead of directly constructing H_gpu. The callable returned by
``load_geometry_function`` always follows the existing H_gpu/check_func
interface:

    states, aux = check_func(points)

where ``states`` is an ``(N, 1)`` float tensor with 1.0 for inside and 0.0 for
outside. ``aux`` is backend-specific: voxel grid indices for voxel backends and
raw SDF values for neural/SIREN SDF backends.
"""

from __future__ import annotations

import os
from typing import Callable, Literal

GeometryBackend = Literal["voxel_artifact", "voxel", "siren", "neural_sdf"]

BACKEND_ALIASES = {
    "saved": "voxel_artifact",
    "saved_voxel": "voxel_artifact",
    "voxel_saved": "voxel_artifact",
    "voxel_artifact": "voxel_artifact",
    "voxel": "voxel",
    "sdf": "siren",
    "siren": "siren",
    "neural": "siren",
    "neural_sdf": "siren",
}


def normalize_geometry_backend(backend: str | None) -> str:
    key = (backend or "voxel_artifact").strip().lower().replace("-", "_")
    if key not in BACKEND_ALIASES:
        valid = ", ".join(sorted(BACKEND_ALIASES))
        raise ValueError(f"Unsupported geometry backend {backend!r}. Valid values: {valid}")
    return BACKEND_ALIASES[key]


def default_geometry_artifact_path(model_name: str, backend: str) -> str | None:
    backend = normalize_geometry_backend(backend)
    if backend == "voxel_artifact":
        return (
            "model_data/implicit_representations/"
            f"{model_name}/{model_name}_voxel_implicit.pt"
        )
    if backend == "siren":
        return (
            "model_data/implicit_representations/"
            f"{model_name}/{model_name}_siren_sdf.pt"
        )
    return None


def load_geometry_function(
    *,
    backend: str | None,
    stl_path: str,
    model_name: str | None = None,
    artifact_path: str | None = None,
    device: str = "cuda",
    voxel_resolution: int = 512,
) -> Callable:
    """Load or build an implicit geometry query function.

    Args:
        backend: ``voxel_artifact`` to load a saved pure-PyTorch voxel artifact,
            ``voxel`` to rebuild a voxel implicit function from STL, or ``siren``
            / ``neural_sdf`` to load a trained neural SDF checkpoint.
        stl_path: STL path used for voxelization and default SDF checkpoint
            resolution.
        model_name: Optional model name used for default artifact paths.
        artifact_path: Optional explicit artifact/checkpoint path.
        device: Torch device string.
        voxel_resolution: Resolution used when ``backend='voxel'``.

    Returns:
        Callable with the existing H_gpu/check_func interface.
    """
    from fieldopt.geometry.implicit import (
        create_pure_pytorch_implicit_function,
        load_pure_pytorch_implicit_function,
        load_sdf_implicit_function,
    )

    backend = normalize_geometry_backend(backend)
    if artifact_path is None and model_name:
        artifact_path = default_geometry_artifact_path(model_name, backend)

    if backend == "voxel_artifact":
        if not artifact_path:
            raise ValueError("geometry artifact path is required for voxel_artifact backend")
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(
                f"Saved voxel H_gpu artifact not found: {artifact_path}\n"
                "Build it first with build_voxel_implicit.py, or use --geometry_backend voxel/siren."
            )
        print(f"[geometry] Loading voxel artifact: {artifact_path}")
        return load_pure_pytorch_implicit_function(artifact_path, device=device)

    if backend == "voxel":
        print(f"[geometry] Building voxel implicit function from STL: {stl_path} (resolution={voxel_resolution})")
        return create_pure_pytorch_implicit_function(
            stl_path,
            voxel_resolution=voxel_resolution,
            device=device,
        )

    if backend == "siren":
        print(f"[geometry] Loading neural/SIREN SDF for STL: {stl_path}")
        return load_sdf_implicit_function(
            stl_path,
            device=device,
            checkpoint_path=artifact_path,
        )

    raise AssertionError(f"Unhandled geometry backend: {backend}")


def add_geometry_backend_args(parser, *, default_backend: str = "voxel_artifact"):
    """Add standard geometry-backend CLI flags to an argparse parser."""
    parser.add_argument(
        "--geometry_backend",
        choices=["voxel_artifact", "voxel", "siren", "neural_sdf"],
        default=default_backend,
        help=(
            "Implicit geometry backend. 'voxel_artifact' loads "
            "model_data/implicit_representations/<model>/<model>_voxel_implicit.pt; "
            "'voxel' rebuilds from STL; 'siren'/'neural_sdf' loads a trained SDF checkpoint."
        ),
    )
    parser.add_argument(
        "--geometry_artifact_path",
        default=None,
        help=(
            "Optional explicit geometry artifact/checkpoint path. For voxel_artifact this is "
            "the saved H_gpu .pt file; for siren/neural_sdf this is the trained SDF checkpoint."
        ),
    )
