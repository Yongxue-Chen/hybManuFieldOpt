"""
Scale an STL file so that the longest side of its bounding box equals a target length,
then move the minimum bounding-box point of the scaled mesh to the origin by default.

Usage:
    python scale_stl.py input.stl output.stl --target 100.0
    python scale_stl.py input.stl output.stl --target 100.0 --center
    python scale_stl.py input.stl output.stl --target 100.0 --center-minz-to-zero
"""

import argparse
import numpy as np
import trimesh


def scale_stl(
    input_path: str,
    output_path: str,
    target_size: float,
    center: bool = False,
    center_minz_to_zero: bool = False,
) -> None:
    mesh = trimesh.load(input_path, force="mesh")

    bounds = mesh.bounds          # shape (2, 3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents = bounds[1] - bounds[0]  # [dx, dy, dz]
    max_extent = extents.max()

    if max_extent == 0:
        raise ValueError("Mesh has zero extent — cannot scale.")

    scale_factor = target_size / max_extent
    mesh.apply_scale(scale_factor)

    if center or center_minz_to_zero:
        mesh.apply_translation(-mesh.centroid)

    scaled_bounds = mesh.bounds
    if center_minz_to_zero:
        mesh.apply_translation([0, 0, -scaled_bounds[0][2]])
    elif not center:
        mesh.apply_translation(-scaled_bounds[0])

    mesh.export(output_path)

    new_bounds = mesh.bounds
    new_extents = new_bounds[1] - new_bounds[0]
    print(f"Input  : {input_path}")
    print(f"Original extents : {extents}")
    print(f"Scale factor     : {scale_factor:.6f}")
    print(f"Placement mode   : {'center' if center else 'center then min-z to zero' if center_minz_to_zero else 'min bound to origin'}")
    print(f"Scaled extents   : {new_extents}")
    print(f"Max extent after : {new_extents.max():.6f}  (target: {target_size})")
    print(f"Min bound after  : {new_bounds[0]}")
    print(f"Output : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scale an STL so its bounding box longest side equals --target, "
            "then move the minimum bounding-box point to the origin by default."
        )
    )
    parser.add_argument("input", help="Path to the input STL file")
    parser.add_argument("output", help="Path for the output (scaled) STL file")
    parser.add_argument(
        "--target", type=float, required=True,
        help="Desired length for the longest bounding-box edge"
    )
    parser.add_argument(
        "--center", action="store_true",
        help="Translate the mesh centroid to the origin after scaling"
    )
    parser.add_argument(
        "--center-minz-to-zero", action="store_true",
        help="Translate the mesh centroid to the origin after scaling, then move minimum Z bound to 0"
    )
    args = parser.parse_args()

    if args.center and args.center_minz_to_zero:
        parser.error("--center and --center-minz-to-zero cannot be used together")

    scale_stl(
        args.input,
        args.output,
        args.target,
        center=args.center,
        center_minz_to_zero=args.center_minz_to_zero,
    )


if __name__ == "__main__":
    main()
