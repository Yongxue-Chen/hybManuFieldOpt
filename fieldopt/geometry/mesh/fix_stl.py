import trimesh
import numpy as np
import argparse
import os
import sys

def _repair_component(mesh):
    mesh = mesh.copy()

    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process()
    mesh.merge_vertices()

    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as e:
        print(f"Warning: fill_holes failed: {e}")

    mesh.fix_normals()
    return mesh


def _repair_components(mesh, keep_largest_component=False):
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        repaired = _repair_component(mesh)
        return repaired, 1

    print(f"Found {len(components)} connected components.")

    if keep_largest_component:
        largest_component = max(components, key=lambda m: len(m.faces))
        print(f"Keeping largest component with {len(largest_component.faces)} faces.")
        repaired = _repair_component(largest_component)
        return repaired, 1

    repaired_components = []
    watertight_count = 0
    for idx, component in enumerate(sorted(components, key=lambda m: len(m.faces), reverse=True), start=1):
        repaired = _repair_component(component)
        repaired_components.append(repaired)
        watertight_count += int(repaired.is_watertight)
        print(
            f"  Component {idx:03d}: faces={len(component.faces)} "
            f"-> repaired_watertight={repaired.is_watertight}"
        )

    print(f"Repaired {len(repaired_components)} components; watertight components: {watertight_count}/{len(repaired_components)}")
    return trimesh.util.concatenate(repaired_components), len(repaired_components)


def _count_mesh_components(mesh):
    return len(list(mesh.split(only_watertight=False)))


def _boolean_union_components(mesh, engine=None):
    components_before = _count_mesh_components(mesh)
    if components_before <= 1:
        print("Mesh already has a single connected component. Skipping boolean union.")
        return mesh, 0, components_before, components_before

    engines = trimesh.boolean.engines_available
    if engine is not None and engine not in engines:
        raise RuntimeError(
            f"Requested boolean engine '{engine}' is not available. "
            f"Available engines: {sorted(engines) if engines else 'none'}"
        )
    if engine is None:
        if not engines:
            raise RuntimeError(
                "No mesh boolean engine is available. Install one supported by trimesh "
                "(for example manifold3d or blender) to use --boolean-union."
            )
        engine = sorted(engines)[0]

    print("\n--- Step 2: Boolean Union Components ---")
    print(
        f"Attempting mesh boolean union on {components_before} components "
        f"using engine: {engine}"
    )

    components = sorted(
        mesh.split(only_watertight=False),
        key=lambda m: len(m.faces),
        reverse=True,
    )
    merged = _repair_component(components[0])
    merged_count = 1

    for idx, component in enumerate(components[1:], start=2):
        candidate = _repair_component(component)
        try:
            merged = trimesh.boolean.union([merged, candidate], engine=engine)
            merged = _repair_component(merged)
            merged_count = idx
            print(f"  Boolean-unioned component {idx}/{components_before}.")
        except Exception as e:
            print(f"  Warning: boolean union failed on component {idx}: {e}")
            merged = trimesh.util.concatenate([merged, candidate])

    components_after = _count_mesh_components(merged)
    print(
        f"After boolean union - Watertight: {merged.is_watertight}, "
        f"Components: {components_after}"
    )
    return merged, merged_count, components_before, components_after


def fix_mesh(
    input_path,
    output_path=None,
    keep_largest_component=False,
    boolean_union=False,
    boolean_engine=None,
):
    """
    Load an STL file and attempt to repair it using mesh-only operations.
    """
    print(f"Loading mesh from: {input_path}")
    try:
        # force='mesh' tries to force a single mesh
        mesh = trimesh.load(input_path, force='mesh')
    except Exception as e:
        print(f"Error loading mesh directly: {e}")
        try:
            # Fallback for Scenes
            scene = trimesh.load(input_path)
            if isinstance(scene, trimesh.Scene):
                if len(scene.geometry) == 0:
                    print("Error: Empty scene loaded.")
                    return
                print("Loaded a Scene, extracting geometry...")
                mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
            else:
                mesh = scene
        except Exception as e2:
            print(f"Failed to load as scene: {e2}")
            return

    print(f"Original Mesh - Watertight: {mesh.is_watertight}, Euler Number: {mesh.euler_number}, Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    if mesh.is_watertight:
        print("Mesh is already watertight. No repair needed.")
    else:
        print("--- Step 1: Standard Repair ---")
        mesh = _repair_component(mesh)
        
        print(f"After Basic Repair - Watertight: {mesh.is_watertight}")
        
        if not mesh.is_watertight:
            print("Basic repair insufficient. Repairing connected components separately...")
            mesh, component_count = _repair_components(
                mesh,
                keep_largest_component=keep_largest_component,
            )
            print(
                f"After Component Repair - Watertight: {mesh.is_watertight}, "
                f"Components kept: {component_count}"
            )

    if boolean_union:
        current_components = _count_mesh_components(mesh)
        if current_components > 1:
            mesh, _, _, _ = _boolean_union_components(
                mesh,
                engine=boolean_engine,
            )
        else:
            print("Single connected component detected. Skipping boolean union step.")

    # Check remaining issues
    if not mesh.is_watertight:
        try:
            edges = mesh.outline()
            if edges:
                print(f"Warning: Remaining open edges count: {len(edges.entities)}")
        except:
            pass

    # Save
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fixed{ext}"
    
    print(f"Saving repaired mesh to: {output_path}")
    try:
        mesh.export(output_path)
        print("Save successful.")
    except Exception as e:
        print(f"Error saving mesh: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair STL files using mesh-only operations.")
    parser.add_argument("input_file", help="Path to input STL file")
    parser.add_argument("--output", "-o", help="Path to output STL file (optional)", default=None)
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="Only keep the largest connected component. Off by default so support structures are preserved.",
    )
    parser.add_argument(
        "--boolean-union",
        action="store_true",
        help="Perform mesh boolean union across connected components without voxelization.",
    )
    parser.add_argument(
        "--boolean-engine",
        default=None,
        help="Optional trimesh boolean engine name, e.g. blender or manifold.",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    fix_mesh(
        args.input_file,
        args.output,
        keep_largest_component=args.keep_largest,
        boolean_union=args.boolean_union,
        boolean_engine=args.boolean_engine,
    )
