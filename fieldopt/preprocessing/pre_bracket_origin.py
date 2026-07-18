import importlib
import json
import math
import os
import sys

import numpy as np
import trimesh


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from .collision_detector import CollisionDetector
from .geometry_utils import create_cylinder_mesh
from .pre_main import (
    _boolean_union_meshes,
    _component_count,
    _export_stl,
    _mesh_quality_score,
    _validate_body_mesh,
)
from .preprocessor import extract_overhang_points, normalize_mesh_to_unit
from .support_generator import SupportGenerator


MODEL_NAME = "fertility_origin"
CONFIG_MODEL_NAME = "fertility"
SAMPLE_DENSITY = 1500
R_SUPP = 1.5
STEP_LEN = 5.0
SUPPORT_FLOOR_OFFSET = 0.003
SUPPORT_RADIUS_SCALE = 1.0
SUPPORT_TOP_EMBED_FACTOR = 0.3
BOOLEAN_ENGINE = "manifold"
SUPPORT_RANDOM_SEED = 24
MAX_GENERATION_ATTEMPTS = 3
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "stlFiles", "fertility_origin_support_generation")
SUPPORT_PROGRESS_STAGES = (0.25, 0.50, 0.75)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, "stlFiles", f"{MODEL_NAME}.stl")
    cfg = importlib.import_module(f"configs.config_multi_field_{CONFIG_MODEL_NAME}")

    raw_mesh = trimesh.load(model_path)
    mesh = normalize_mesh_to_unit(raw_mesh)
    _validate_body_mesh(mesh, model_path)

    scale_factor = cfg.SCALE
    r_supp = R_SUPP / scale_factor
    step_len = STEP_LEN / scale_factor

    r_tip = cfg.MANU_CONFIG["SMToolParas"]["SMTipDiameter"] / 2.0
    r_shank = cfg.MANU_CONFIG["SMToolParas"]["SMShankDiameter"] / 2.0
    l_tool = cfg.MANU_CONFIG["SMToolParas"]["SMToolLength"] * 0.9
    r_holder = cfg.MANU_CONFIG["SMToolParas"]["SMHolderDiameter"] / 2.0
    l_shank = cfg.MANU_CONFIG["SMToolParas"]["SMHolderLength"]

    tool_params = {
        "r_tip": r_tip,
        "r_shank": r_shank,
        "l_tool": l_tool,
        "r_holder": r_holder,
        "l_shank": l_shank,
    }

    mesh_bounds = mesh.bounds
    support_floor_z = float(mesh_bounds[0][2] + SUPPORT_FLOOR_OFFSET)
    if support_floor_z >= float(mesh_bounds[1][2]):
        raise RuntimeError(
            f"Support floor z={support_floor_z:.6f} exceeds mesh top z={mesh_bounds[1][2]:.6f}."
        )

    logical_aabb = [
        mesh_bounds[0][0] + r_supp,
        mesh_bounds[1][0] - r_supp,
        mesh_bounds[0][1] + r_supp,
        mesh_bounds[1][1] - r_supp,
        support_floor_z,
    ]
    aabb = [
        mesh_bounds[0][0],
        mesh_bounds[1][0],
        mesh_bounds[0][1],
        mesh_bounds[1][1],
        mesh_bounds[0][2],
        mesh_bounds[1][2],
    ]
    best_result = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        attempt_seed = SUPPORT_RANDOM_SEED + attempt
        np.random.seed(attempt_seed)
        print(
            f"开始生成支撑... model={MODEL_NAME}, config={CONFIG_MODEL_NAME}, "
            f"floor_z={support_floor_z:.6f}, "
            f"attempt={attempt + 1}/{MAX_GENERATION_ATTEMPTS}, seed={attempt_seed}"
        )

        detector = CollisionDetector(mesh, tool_params, device="cuda")
        detector.floor_z = support_floor_z
        seeds = extract_overhang_points(
            mesh,
            sample_density=SAMPLE_DENSITY,
            z_min=5.0 / scale_factor,
        )
        generator = SupportGenerator(detector, seeds, r_supp, step_len, logical_aabb, aabb)
        generator.run()

        support_mesh = _create_support_mesh(generator.finished_paths, r_supp, support_floor_z)
        combined_mesh = _boolean_union_meshes(mesh, support_mesh, engine=BOOLEAN_ENGINE)
        candidate = {
            "score": _mesh_quality_score(combined_mesh),
            "body_mesh": mesh.copy(),
            "support_mesh": support_mesh.copy(),
            "combined_mesh": combined_mesh,
            "finished_paths": generator.finished_paths,
            "seed": attempt_seed,
        }

        if best_result is None or candidate["score"] < best_result["score"]:
            best_result = candidate

        if combined_mesh.is_watertight and combined_mesh.is_volume and _component_count(combined_mesh) == 1:
            break

        print(
            f"  Attempt {attempt + 1} did not yield a clean volume; "
            f"score={candidate['score']}"
        )

    final_combined = best_result["combined_mesh"]
    if final_combined.is_watertight and final_combined.is_volume:
        print(f"采用 seed={best_result['seed']} 的结果进行导出。")
    else:
        print(
            f"Warning: best result is still not watertight after {MAX_GENERATION_ATTEMPTS} attempts. "
            f"Using seed={best_result['seed']} with score={best_result['score']}."
        )

    body_output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_body_only.stl")
    support_output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_support_only.stl")
    combined_output_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_support.stl")

    _export_stl(best_result["body_mesh"], body_output_path, "主体 STL")
    _export_stl(best_result["support_mesh"], support_output_path, "支撑 STL")
    _export_stl(final_combined, combined_output_path, "主体+支撑 union STL")
    export_support_progress_stls(best_result["finished_paths"], r_supp, support_floor_z, OUTPUT_DIR)
    export_removal_plan(best_result["finished_paths"][::-1], OUTPUT_DIR)


def _create_support_mesh(paths, r_supp, support_floor_z):
    return create_cylinder_mesh(
        paths,
        r_supp,
        z_min=support_floor_z,
        radius_scale=SUPPORT_RADIUS_SCALE,
        top_embed_length=r_supp * SUPPORT_TOP_EMBED_FACTOR,
    )


def export_support_progress_stls(paths, r_supp, support_floor_z, output_dir):
    total_paths = len(paths)
    if total_paths == 0:
        print("Warning: no support paths generated; skipping progress STL exports.")
        return

    for ratio in SUPPORT_PROGRESS_STAGES:
        stage_percent = int(round(ratio * 100))
        stage_count = min(total_paths, max(1, int(math.ceil(total_paths * ratio))))
        stage_mesh = _create_support_mesh(paths[:stage_count], r_supp, support_floor_z)
        output_path = os.path.join(
            output_dir,
            f"{MODEL_NAME}_support_only_stage_{stage_percent}.stl",
        )
        _export_stl(
            stage_mesh,
            output_path,
            f"支撑中间过程 {stage_percent}% STL ({stage_count}/{total_paths} paths)",
        )


def export_removal_plan(paths, output_dir):
    plan = []

    for path_data in paths:
        for pos, axis in reversed(path_data[1:-1]):
            safe_axis = axis.tolist() if hasattr(axis, "tolist") else axis

            plan.append(
                {
                    "pos": pos.tolist() if hasattr(pos, "tolist") else pos,
                    "axis": safe_axis,
                }
            )

    output_path = os.path.join(output_dir, f"{MODEL_NAME}_removal_plan.json")
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=4)
    print(f"拆除计划已导出: path={output_path}, steps={len(plan)}")


if __name__ == "__main__":
    main()
