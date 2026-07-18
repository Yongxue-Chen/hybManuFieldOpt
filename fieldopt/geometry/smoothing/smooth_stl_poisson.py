import argparse
from pathlib import Path

import open3d as o3d
import numpy as np


def smooth_stl_poisson(
    input_stl: Path,
    output_stl: Path,
    num_points: int = 200000,
    poisson_depth: int = 9,
    linear_fit: bool = False,
    simplify_target_faces: int = 0
) -> None:
    """
    使用 Open3D 的泊松表面重建 (Screened Poisson Surface Reconstruction) 对 STL 进行平滑。

    这种算法对满是阶梯（Staircase）的体素提取模型特别有效。
    它的原理是将阶梯模型表面当做点云采样，然后基于点云和法线重新构建一个光滑的水密曲面。

    Args:
        input_stl (Path): 输入 STL 文件路径（带阶梯的原始模型）。
        output_stl (Path): 输出 STL 文件路径。
        num_points (int): 
            从原始网格上采样的点云数量。对于细节较多的模型，建议 100000~500000。
            采样越多，重建越贴合原文理，但如果包含微小阶梯也可能被还原。默认 200000。
        poisson_depth (int):
            泊松重建的八叉树深度（Depth）。
            这决定了重建网格的精细度。值越大（如 10、11），表面细节越丰富，但也越容易还原阶梯；
            值为 8、9 时能将高频阶梯“平滑掉”而保留结构。默认 9。
        linear_fit (bool):
            如果设置为 True，重建网格的表面会使用线性插值，对于机械零件/直角结构可能有帮助。
        simplify_target_faces (int):
            因为泊松重建往往会生成大量面片（往往达上百万面），
            如果大于0，将在重建后使用二次误差度量（Quadric Error Metric）减面到指定面数。
    """
    print(f"[1/4] 读取模型: {input_stl}")
    mesh = o3d.io.read_triangle_mesh(str(input_stl))
    
    if not mesh.has_triangles():
        raise ValueError(f"读取的文件没有包含有效的三角面: {input_stl}")

    # 1. 在原始表面上均匀撒点，生成点云
    print(f"[2/4] 在表面均匀采样点云: {num_points} 点...")
    # Poisson disk 撒点比均匀随机撒点（Uniform）更整齐均匀
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # 2. 估计点云的法向
    print("[3/4] 估计并对齐点云法向...")
    # 基于周围点建立凸包并计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    # 尽可能让法线朝外，保持一致性
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # 3. 运行泊松表面重建
    print(f"[4/4] 运行泊松表面重建 (Depth: {poisson_depth})...")
    recon_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=linear_fit
    )

    # 后处理：裁剪掉超出原始原始点云盒子之外的“飞边”（泊松重建特性）
    print("      >> 裁剪冗余重建边缘并计算顶点法线...")
    bbox = pcd.get_axis_aligned_bounding_box()
    # 可以稍微扩大一点 bbox 防止切到正常的表面
    bbox_max = bbox.get_max_bound() + np.array([0.5, 0.5, 0.5])
    bbox_min = bbox.get_min_bound() - np.array([0.5, 0.5, 0.5])
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    recon_mesh = recon_mesh.crop(bbox)

    # 如果需要减面
    if simplify_target_faces > 0 and len(recon_mesh.triangles) > simplify_target_faces:
        print(f"      >> 网格减面: {len(recon_mesh.triangles)} -> {simplify_target_faces} 面")
        recon_mesh = recon_mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_target_faces)

    recon_mesh.compute_vertex_normals()

    # 4. 导出平滑结果
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_stl), recon_mesh)
    
    print("\n✅ STL 泊松平滑完成！")
    print(f"输出文件: {output_stl}")
    print(f"最终面数: {len(recon_mesh.triangles)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对体素化生成的 STL 模型进行泊松表面重建平滑处理。"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="[必填] 输入带有阶梯锯齿的 STL 文件路径",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="[必填] 输出平滑后的 STL 文件路径",
    )
    parser.add_argument(
        "--num-points", type=int, default=200000,
        help="采样点数量。越多越精细，默认 200000。",
    )
    parser.add_argument(
        "--poisson-depth", type=int, default=9,
        help="泊松重建八叉树深度。值太大可能还原锯齿，太小会丢失宏观细节，建议 8~10，默认 9。",
    )
    parser.add_argument(
        "--simplify-faces", type=int, default=0,
        help="重建完成后减面到的目标面数。默认 0 (不减面，这通常会产生一个很致密的大网格)。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    smooth_stl_poisson(
        input_stl=args.input,
        output_stl=args.output,
        num_points=args.num_points,
        poisson_depth=args.poisson_depth,
        simplify_target_faces=args.simplify_faces
    )


if __name__ == "__main__":
    main()
