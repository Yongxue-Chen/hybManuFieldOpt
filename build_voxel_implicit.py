import argparse
import os

from fieldopt.geometry.implicit import (
    create_pure_pytorch_implicit_function,
    save_pure_pytorch_implicit_function,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build and save a voxel-backed pure-PyTorch implicit representation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="boneTPMS",
        help="Model name used only to resolve default STL/output paths.",
    )
    parser.add_argument(
        "--stl_path",
        type=str,
        default=None,
        help=(
            "Optional STL path. Default: "
            "model_data/preprocessed/<model_name>/<model_name>_support.stl"
        ),
    )
    parser.add_argument(
        "--voxel_resolution",
        type=int,
        default=512,
        help="Voxel resolution used to build the implicit function.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string used to build the implicit function.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Optional output artifact path. Default: "
            "demo_runs/implicit_representations/<model_name>/"
            "<model_name>_voxel_implicit.pt"
        ),
    )
    args = parser.parse_args()

    stl_path = (
        args.stl_path
        or (
            "model_data/preprocessed/"
            f"{args.model_name}/{args.model_name}_support.stl"
        )
    )
    hres = args.voxel_resolution
    device = args.device
    output_path = (
        args.output_path
        or (
            "demo_runs/implicit_representations/"
            f"{args.model_name}/{args.model_name}_voxel_implicit.pt"
        )
    )

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    print(f"[build_voxel_implicit] model_name: {args.model_name}")
    print(f"[build_voxel_implicit] stl_path: {stl_path}")
    print(f"[build_voxel_implicit] voxel_resolution: {hres}")
    print(f"[build_voxel_implicit] device: {device}")
    print(f"[build_voxel_implicit] output_path: {output_path}")

    H_gpu = create_pure_pytorch_implicit_function(
        stl_path,
        voxel_resolution=hres,
        device=device,
    )
    saved_path = save_pure_pytorch_implicit_function(H_gpu, output_path)
    print(f"[build_voxel_implicit] Saved voxel implicit artifact to: {saved_path}")


if __name__ == "__main__":
    main()
