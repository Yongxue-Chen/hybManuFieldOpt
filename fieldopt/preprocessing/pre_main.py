import trimesh
import numpy as np
import json
import importlib
from .preprocessor import normalize_mesh_to_unit, extract_overhang_points
from .collision_detector import CollisionDetector
from .support_generator import SupportGenerator
from .geometry_utils import create_cylinder_mesh
import os
import sys

MODEL_NAME = 'MBBSmooth'
DEFAULT_STL_DIR = 'stlFiles'
DEFAULT_OUTPUT_ROOT = 'outputs/preprocess'
SAMPLE_DENSITY = 3000
R_SUPP = 1.5
STEP_LEN = 5.0
SUPPORT_FLOOR_OFFSET = 0.003
SUPPORT_RADIUS_SCALE = 1.0
SUPPORT_TOP_EMBED_FACTOR = 0.3
BOOLEAN_ENGINE = 'manifold'
SUPPORT_RANDOM_SEED = 24
MAX_GENERATION_ATTEMPTS = 3
UNION_ARTIFACT_MAX_FACES = 64
UNION_ARTIFACT_ABS_VOLUME = 1e-7
UNION_ARTIFACT_REL_VOLUME = 1e-6

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main(model_name=None, input_stl=None, output_root=DEFAULT_OUTPUT_ROOT, output_dir=None):
    # --- 参数设置与模型加载 ---
    active_model_name = model_name or MODEL_NAME
    model_path = input_stl or os.path.join(DEFAULT_STL_DIR, f'{active_model_name}.stl')
    output_dir = output_dir or os.path.join(output_root, active_model_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg = importlib.import_module(f'configs.config_multi_field_{active_model_name}')

    print(f"Preprocess model: {active_model_name}")
    print(f"Input STL: {model_path}")
    print(f"Output directory: {output_dir}")

    # 归一化
    raw_mesh = trimesh.load(model_path)
    mesh = normalize_mesh_to_unit(raw_mesh)
    _validate_body_mesh(mesh, model_path)

    # 参数设置
    scale_factor = cfg.SCALE
    r_supp = R_SUPP / scale_factor
    step_len = STEP_LEN / scale_factor

    r_tip = cfg.MANU_CONFIG['SMToolParas']['SMTipDiameter'] / 2.0
    r_shank = cfg.MANU_CONFIG['SMToolParas']['SMShankDiameter'] / 2.0
    l_tool = cfg.MANU_CONFIG['SMToolParas']['SMToolLength'] * 0.8
    r_holder = cfg.MANU_CONFIG['SMToolParas']['SMHolderDiameter'] / 2.0
    l_shank = cfg.MANU_CONFIG['SMToolParas']['SMHolderLength']

    tool_params = {
        'r_tip': r_tip,
        'r_shank': r_shank,
        'l_tool': l_tool,
        'r_holder': r_holder,
        'l_shank': l_shank,
    }
    
    mesh_bounds = mesh.bounds
    support_floor_z = float(mesh_bounds[0][2] + SUPPORT_FLOOR_OFFSET)
    if support_floor_z >= float(mesh_bounds[1][2]):
        raise RuntimeError(
            f"Support floor z={support_floor_z:.6f} exceeds mesh top z={mesh_bounds[1][2]:.6f}."
        )

    logical_aabb = [mesh_bounds[0][0]+r_supp, mesh_bounds[1][0]-r_supp, 
                    mesh_bounds[0][1]+r_supp, mesh_bounds[1][1]-r_supp, 
                    support_floor_z]
    aabb = [mesh_bounds[0][0], mesh_bounds[1][0], mesh_bounds[0][1], mesh_bounds[1][1], mesh_bounds[0][2], mesh_bounds[1][2]]
    best_result = None

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        attempt_seed = SUPPORT_RANDOM_SEED + attempt
        np.random.seed(attempt_seed)
        print(
            f"开始生成支撑... floor_z={support_floor_z:.6f}, "
            f"attempt={attempt + 1}/{MAX_GENERATION_ATTEMPTS}, seed={attempt_seed}"
        )

        detector = CollisionDetector(mesh, tool_params, device='cuda')
        detector.floor_z = support_floor_z
        seeds = extract_overhang_points(mesh, sample_density=SAMPLE_DENSITY, z_min=5.0/scale_factor)
        generator = SupportGenerator(detector, seeds, r_supp, step_len, logical_aabb, aabb)
        generator.run()

        support_mesh = create_cylinder_mesh(
            generator.finished_paths,
            r_supp,
            z_min=support_floor_z,
            radius_scale=SUPPORT_RADIUS_SCALE,
            top_embed_length=r_supp * SUPPORT_TOP_EMBED_FACTOR,
        )
        combined_mesh = _boolean_union_meshes(mesh, support_mesh, engine=BOOLEAN_ENGINE)
        candidate = {
            'score': _mesh_quality_score(combined_mesh),
            'body_mesh': mesh.copy(),
            'support_mesh': support_mesh.copy(),
            'combined_mesh': combined_mesh,
            'finished_paths': generator.finished_paths,
            'seed': attempt_seed,
        }

        if best_result is None or candidate['score'] < best_result['score']:
            best_result = candidate

        if combined_mesh.is_watertight and combined_mesh.is_volume and _component_count(combined_mesh) == 1:
            break

        print(
            f"  Attempt {attempt + 1} did not yield a clean volume; "
            f"score={candidate['score']}"
        )

    final_combined = best_result['combined_mesh']
    if final_combined.is_watertight and final_combined.is_volume:
        print(f"采用 seed={best_result['seed']} 的结果进行导出。")
    else:
        print(
            f"Warning: best result is still not watertight after {MAX_GENERATION_ATTEMPTS} attempts. "
            f"Using seed={best_result['seed']} with score={best_result['score']}."
        )

    # --- 5. 导出结果 (保持归一化空间) ---
    body_output_path = os.path.join(output_dir, f'{active_model_name}_body_only.stl')
    support_output_path = os.path.join(output_dir, f'{active_model_name}_support_only.stl')
    combined_output_path = os.path.join(output_dir, f'{active_model_name}_support.stl')
    removal_plan_path = os.path.join(output_dir, f'{active_model_name}_removal_plan.json')

    _export_stl(best_result['body_mesh'], body_output_path, "主体 STL")
    _export_stl(best_result['support_mesh'], support_output_path, "支撑 STL")
    _export_stl(final_combined, combined_output_path, "主体+支撑 union STL")

    # B. 导出拆除计划 (绝对逆序逻辑)
    # 全局逆序：最后生成的(外侧)先拆；局部逆序：从地面向上拆
    export_removal_plan(best_result['finished_paths'][::-1], removal_plan_path)

def export_removal_plan(paths, output_path):
    """
    保存拆除路径：
    1. paths: 全局倒序的路径列表 (List of paths)
    2. 每个 path: [(p1, None), (p2, v2), ..., (pn, None)]
    """
    plan = []
    
    # 1. 全局倒序：从外侧支撑开始拆
    for path_data in paths:
        # 2. 局部倒序：从地面/碰撞点 逆向回到 悬垂点
        # reversed(path_data[1:-1]) 选取中间那些经过 SM 校验的点
        for pos, axis in reversed(path_data[1:-1]):
            # 确保 axis 是列表格式以便 JSON 序列化
            safe_axis = axis.tolist() if hasattr(axis, 'tolist') else axis
            
            plan.append({
                "pos": pos.tolist() if hasattr(pos, 'tolist') else pos,
                "axis": safe_axis # 这里现在存储的是真正校验成功的姿态
            })
    
    with open(output_path, 'w') as f:
        json.dump(plan, f, indent=4)
    print(f"拆除计划已导出: path={output_path}, steps={len(plan)}")


def _export_stl(mesh, output_path, label):
    exported = _repair_mesh(mesh)
    exported.export(output_path)
    print(
        f"{label} 已导出: path={output_path}, "
        f"faces={len(exported.faces)}, "
        f"components={_component_count(exported)}, "
        f"watertight={exported.is_watertight}, "
        f"is_volume={exported.is_volume}"
    )


def _repair_mesh(mesh):
    mesh = mesh.copy()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process()
    mesh.merge_vertices()
    mesh.fix_normals()
    return mesh


def _component_count(mesh):
    return len(list(mesh.split(only_watertight=False)))


def _edge_topology_stats(mesh):
    unique_edges = mesh.edges_unique
    inverse = mesh.edges_unique_inverse
    counts = np.bincount(inverse, minlength=len(unique_edges))
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())
    return boundary_edges, nonmanifold_edges


def _mesh_quality_score(mesh):
    boundary_edges, nonmanifold_edges = _edge_topology_stats(mesh)
    component_count = _component_count(mesh)
    return (
        int(not mesh.is_watertight),
        int(not mesh.is_volume),
        nonmanifold_edges,
        boundary_edges,
        max(component_count - 1, 0),
        len(mesh.faces),
    )


def _cleanup_union_artifacts(mesh):
    """
    清理 boolean union 后偶发的极小伪组件（常见于 manifold 数值边界）。
    """
    components = sorted(
        mesh.split(only_watertight=False),
        key=lambda m: len(m.faces),
        reverse=True,
    )
    if len(components) <= 1:
        return mesh

    volumes = [abs(c.volume) if c.is_volume else 0.0 for c in components]
    max_volume = max(volumes) if volumes else 0.0
    keep_components = []
    dropped_stats = []

    for comp in components:
        abs_volume = abs(comp.volume) if comp.is_volume else 0.0
        rel_volume = (abs_volume / max_volume) if max_volume > 0.0 else 0.0
        tiny_volume = (
            abs_volume < UNION_ARTIFACT_ABS_VOLUME
            or rel_volume < UNION_ARTIFACT_REL_VOLUME
        )
        tiny_faces = len(comp.faces) <= UNION_ARTIFACT_MAX_FACES

        if tiny_faces and (tiny_volume or not comp.is_volume):
            dropped_stats.append((len(comp.faces), abs_volume, comp.is_volume))
            continue
        keep_components.append(comp)

    if not dropped_stats or not keep_components:
        return mesh

    cleaned = (
        keep_components[0].copy()
        if len(keep_components) == 1
        else trimesh.util.concatenate(keep_components)
    )
    print(
        f"  Cleanup union artifacts: dropped={len(dropped_stats)} tiny components, "
        f"remaining_components={_component_count(cleaned)}"
    )
    for idx, (faces, abs_volume, is_volume) in enumerate(dropped_stats, start=1):
        print(
            f"    dropped[{idx}]: faces={faces}, abs_volume={abs_volume:.6e}, "
            f"is_volume={is_volume}"
        )
    return cleaned


def _seal_post_union_holes(mesh):
    """
    对 union 结果做一次轻量补洞，仅处理小开口场景。
    """
    boundary_before, nonmanifold_before = _edge_topology_stats(mesh)
    if boundary_before == 0:
        return mesh

    repaired = mesh.copy()
    trimesh.repair.fill_holes(repaired)
    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()
    repaired.fix_normals()

    boundary_after, nonmanifold_after = _edge_topology_stats(repaired)
    print(
        "  Post-union hole repair: "
        f"boundary_edges {boundary_before}->{boundary_after}, "
        f"nonmanifold_edges {nonmanifold_before}->{nonmanifold_after}, "
        f"watertight {mesh.is_watertight}->{repaired.is_watertight}"
    )
    return repaired


def _validate_body_mesh(mesh, model_path):
    """
    这个生成链假设输入的是“干净的主体 STL”。
    如果把已经带支撑/带碎块的 STL 重新当主体输入，后续 boolean union
    很容易产生严重错误几何，因此这里直接 fail-fast。
    """
    repaired = _repair_mesh(mesh)
    component_count = _component_count(repaired)
    print(
        f"主体检查: watertight={repaired.is_watertight}, "
        f"is_volume={repaired.is_volume}, components={component_count}, "
        f"faces={len(repaired.faces)}"
    )

    if component_count > 1 or not repaired.is_volume:
        print(
            "Warning: input body STL is not a single clean volume. "
            f"path={model_path}, components={component_count}, "
            f"is_volume={repaired.is_volume}. "
            "Boolean union may become unstable."
        )


def _boolean_union_meshes(body_mesh, support_mesh, engine='manifold'):
    """
    对主体和支撑执行 mesh boolean union。
    输入主体 STL 文件不会被修改，只有导出的组合结果会是一体化后的新网格。
    """
    body_mesh = _repair_mesh(body_mesh)
    support_mesh = _repair_mesh(support_mesh)

    if support_mesh.faces is None or len(support_mesh.faces) == 0:
        print("Warning: support mesh is empty, exporting body mesh only.")
        return body_mesh

    available_engines = trimesh.boolean.engines_available
    if engine not in available_engines:
        raise RuntimeError(
            f"Boolean engine '{engine}' is not available. "
            f"Available engines: {available_engines}"
        )

    body_components = _component_count(body_mesh)
    support_components = sorted(
        support_mesh.split(only_watertight=False),
        key=lambda m: len(m.faces),
        reverse=True,
    )
    print(
        f"导出前执行 boolean union: engine={engine}, "
        f"body_components={body_components}, "
        f"support_components={len(support_components)}, "
        f"support_faces={len(support_mesh.faces)}"
    )

    union_inputs = [body_mesh]
    skipped_components = []

    for idx, component in enumerate(support_components, start=1):
        candidate = _repair_mesh(component)
        if len(candidate.faces) == 0:
            continue

        if not candidate.is_volume:
            print(
                f"  Skip support component {idx}/{len(support_components)}: "
                f"not a valid volume after repair."
            )
            skipped_components.append(candidate)
            continue

        union_inputs.append(candidate)

    print(f"  Participating in boolean union: {len(union_inputs)} meshes")
    merged = trimesh.boolean.union(
        union_inputs,
        engine=engine,
        check_volume=False,
    )
    merged = _repair_mesh(merged)
    merged = _cleanup_union_artifacts(merged)
    merged = _seal_post_union_holes(merged)

    if skipped_components:
        print(f"  Warning: {len(skipped_components)} support components were skipped from union.")
        merged = trimesh.util.concatenate([merged] + skipped_components)
        merged = _repair_mesh(merged)

    component_count = _component_count(merged)
    print(
        f"Boolean union 完成: watertight={merged.is_watertight}, "
        f"components={component_count}, faces={len(merged.faces)}"
    )
    return merged

if __name__ == "__main__":
    main()
