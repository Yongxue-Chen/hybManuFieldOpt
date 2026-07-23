import argparse
import os
import time
import torch

from .pipeline import run_pipeline
from .model_loader import load_model_and_config
from .visualizer import launch_visualizer
from fieldopt.geometry.backend import add_geometry_backend_args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hybrid post-processing pipeline."
    )
    parser.add_argument(
        "--config-name",
        "--model_name",
        dest="config_name",
        default="bracket",
        help="Configuration suffix (default: bracket).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Trained model checkpoint. Defaults to "
            "model_data/trained_fields/<model>/<model>_final_trained.pth."
        ),
    )
    parser.add_argument(
        "--stl-path",
        default=None,
        help=(
            "Preprocessed support STL. Defaults to "
            "model_data/preprocessed/<model>/<model>_support.stl."
        ),
    )
    parser.add_argument(
        "--cached",
        choices=["save", "load"],
        default="save",
        help="'save': run pipeline then save result to disk. "
             "'load': skip pipeline and load the saved result. "
             "Default file is model_data/postprocessed/<model>/"
             "<model>_layers_paths.pt.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path for the cached pipeline result used by --cached save/load. "
             "Default follows --cached help.",
    )
    parser.add_argument(
        "--layer-only",
        action="store_true",
        help="Skip path generation and collision detection. "
             "Only generate layers and launch the visualizer.",
    )
    parser.add_argument(
        "--sm-only",
        action="store_true",
        help="Only keep SM layers and only generate SM paths.",
    )
    parser.add_argument(
        "--skip-collision-check",
        action="store_true",
        help="Generate raw paths for every layer, but skip collision detection "
             "and collision avoidance/correction.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not launch the visualizer.",
    )
    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=100,
        help="Resolution used only when --geometry_backend voxel (default: 100).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=256,
        help="Layer-extraction grid resolution (default: 256).",
    )
    parser.add_argument(
        "--am-coarse-count",
        type=int,
        default=100,
        help="Number of candidate AM time steps (default: 100).",
    )
    parser.add_argument(
        "--sm-coarse-count",
        type=int,
        default=100,
        help="Number of candidate SM time steps (default: 100).",
    )
    add_geometry_backend_args(parser)

    return parser


def _default_model_path(config_name: str) -> str:
    return os.path.join(
        "model_data",
        "trained_fields",
        config_name,
        f"{config_name}_final_trained.pth",
    )


def _default_stl_path(config_name: str) -> str:
    return os.path.join(
        "model_data",
        "preprocessed",
        config_name,
        f"{config_name}_support.stl",
    )


def _cache_path(
    config_name: str,
    output_path: str | None = None,
    sm_only: bool = False,
) -> str:
    if output_path is not None:
        return output_path
    mode_suffix = "_sm_only" if sm_only else ""
    return os.path.join(
        "model_data",
        "postprocessed",
        config_name,
        f"{config_name}{mode_suffix}_layers_paths.pt",
    )


def _filter_sm_only_result(result: dict) -> dict:
    layers = result.get("layers", [])
    paths = result.get("paths", [])
    if len(paths) != len(layers):
        return result

    sm_pairs = [
        (layer, path)
        for layer, path in zip(layers, paths)
        if getattr(layer, "layer_type", None) == "SM"
    ]
    result["layers"] = [layer for layer, _ in sm_pairs]
    result["paths"] = [path for _, path in sm_pairs]
    return result


def main() -> None:
    start_time = time.perf_counter()
    parser = _build_parser()
    args = parser.parse_args()
    for option, value in (
        ("--voxel-resolution", args.voxel_resolution),
        ("--grid-resolution", args.grid_resolution),
        ("--am-coarse-count", args.am_coarse_count),
        ("--sm-coarse-count", args.sm_coarse_count),
    ):
        if value <= 0:
            parser.error(f"{option} must be a positive integer")
    try:
        cache_file = _cache_path(
            args.config_name,
            args.output_path,
            args.sm_only,
        )
        model_path = args.model_path or _default_model_path(args.config_name)
        stl_path = args.stl_path or _default_stl_path(args.config_name)

        if args.cached == "load":
            print(f"Loading cached pipeline result from {cache_file} ...")
            result = torch.load(cache_file, weights_only=False)
            print("Rebuilding model context (needed for material-state viz) ...")
            ctx = load_model_and_config(
                args.config_name, model_path,
                stl_path=stl_path,
                voxel_resolution=args.voxel_resolution,
                geometry_backend=args.geometry_backend,
                geometry_artifact_path=args.geometry_artifact_path,
            )
            result['ctx'] = ctx
            if args.sm_only:
                result = _filter_sm_only_result(result)
        else:
            result = run_pipeline(
                config_name=args.config_name,
                model_path=model_path,
                stl_path=stl_path,
                voxel_resolution=args.voxel_resolution,
                skip_paths=args.layer_only,
                sm_only=args.sm_only,
                skip_collision_check=args.skip_collision_check,
                geometry_backend=args.geometry_backend,
                geometry_artifact_path=args.geometry_artifact_path,
                grid_resolution=args.grid_resolution,
                am_coarse_count=args.am_coarse_count,
                sm_coarse_count=args.sm_coarse_count,
            )
            if args.cached == "save":
                save_data = {k: v for k, v in result.items() if k != 'ctx'}
                cache_dir = os.path.dirname(cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                torch.save(save_data, cache_file)
                print(f"Pipeline result saved to {cache_file}")

        if not args.no_visualize:
            launch_visualizer(result)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
