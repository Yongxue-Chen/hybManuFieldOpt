"""User-facing preprocessing entry point."""

import argparse

from fieldopt.preprocessing import pre_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run support generation and preprocessing.")
    parser.add_argument(
        "--model_name",
        default=pre_main.MODEL_NAME,
        help=f"Model/config name to preprocess (default: {pre_main.MODEL_NAME}).",
    )
    parser.add_argument(
        "--input-stl",
        default=None,
        help=(
            "Input target-shape STL. Defaults to "
            f"{pre_main.DEFAULT_STL_DIR}/<model_name>.stl."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=pre_main.DEFAULT_OUTPUT_ROOT,
        help=(
            "Parent folder for preprocessing results from multiple models. "
            "The model name is appended automatically as a subfolder "
            f"(default: {pre_main.DEFAULT_OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Exact output folder for this model. When provided, it overrides "
            "the automatically constructed <output-root>/<model_name> path."
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pre_main.main(
        model_name=args.model_name,
        input_stl=args.input_stl,
        output_root=args.output_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
