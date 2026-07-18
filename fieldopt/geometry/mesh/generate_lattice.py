import numpy as np
import trimesh
from itertools import product


BOOLEAN_ENGINE = "manifold"


def beam_between(p0, p1, thickness):
    """
    在两点之间创建一个方截面梁
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-8:
        raise ValueError("Zero length beam")

    # 先创建沿 Z 轴的 box
    beam = trimesh.creation.box(extents=(thickness, thickness, length))

    # 将 Z 轴对齐到 vec
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length

    T = trimesh.geometry.align_vectors(z_axis, direction)
    beam.apply_transform(T)

    # 平移到中点
    midpoint = (p0 + p1) / 2.0
    beam.apply_translation(midpoint)

    return beam


def boolean_union_meshes(meshes, engine=BOOLEAN_ENGINE, batch_size=30):
    """
    分批做布尔并集，避免一次性把所有梁丢给布尔引擎导致内存/时间过大。
    """
    if engine not in trimesh.boolean.engines_available:
        available = ", ".join(str(item) for item in trimesh.boolean.engines_available)
        raise RuntimeError(
            f"Boolean engine '{engine}' is not available. Available engines: {available}"
        )

    meshes = list(meshes)
    if not meshes:
        raise ValueError("No meshes to union")

    while len(meshes) > 1:
        merged = []
        total_batches = (len(meshes) + batch_size - 1) // batch_size

        for batch_index, start in enumerate(range(0, len(meshes), batch_size), start=1):
            batch = meshes[start:start + batch_size]
            print(
                f"Boolean union batch {batch_index}/{total_batches}: "
                f"{len(batch)} mesh(es)"
            )

            if len(batch) == 1:
                merged.append(batch[0])
                continue

            unioned = trimesh.boolean.union(batch, engine=engine)
            if isinstance(unioned, list):
                unioned = trimesh.util.concatenate(unioned)

            merged.append(unioned)

        meshes = merged

    mesh = meshes[0]
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    return mesh


def create_clean_cubic_lattice(
    cube_size=100.0,
    cells=8,
    thickness=2.0,
    use_boolean=True,
):
    """
    生成干净的立方格构：
    每条完整网格线只建立一次，并可用布尔并集融合相交梁。
    """
    half = cube_size / 2.0

    coords = np.linspace(-half, half, cells + 1)

    meshes = []

    # X方向整根梁
    for y, z in product(coords, coords):
        meshes.append(
            beam_between((-half, y, z), (half, y, z), thickness)
        )

    # Y方向整根梁
    for x, z in product(coords, coords):
        meshes.append(
            beam_between((x, -half, z), (x, half, z), thickness)
        )

    # Z方向整根梁
    for x, y in product(coords, coords):
        meshes.append(
            beam_between((x, y, -half), (x, y, half), thickness)
        )

    print(f"beams: {len(meshes)}")

    if use_boolean:
        return boolean_union_meshes(meshes)

    mesh = trimesh.util.concatenate(meshes)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    return mesh


def main():
    cube_size = 100.0
    cells = 9
    thickness = 1.8
    output = "clean_lattice_cube.stl"

    mesh = create_clean_cubic_lattice(
        cube_size=cube_size,
        cells=cells,
        thickness=thickness,
        use_boolean=True,
    )

    print("is_watertight:", mesh.is_watertight)
    print("is_volume:", mesh.is_volume)
    print("faces:", len(mesh.faces))
    print("bounds:", mesh.bounds)

    mesh.export(output)
    print(f"导出完成: {output}")


if __name__ == "__main__":
    main()
