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
        help="Input STL path. Defaults to stlFiles/<model_name>.stl.",
    )
    parser.add_argument(
        "--output-root",
        default=pre_main.DEFAULT_OUTPUT_ROOT,
        help=(
            "Root folder for preprocessing outputs. A <model_name> subfolder is "
            f"created by default (default: {pre_main.DEFAULT_OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output folder. Overrides --output-root/<model_name>.",
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
