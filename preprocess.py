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
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pre_main.MODEL_NAME = args.model_name
    pre_main.main()


if __name__ == "__main__":
    main()
