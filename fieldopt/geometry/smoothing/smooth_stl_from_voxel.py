import argparse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.smoothing import filter_humphrey, filter_laplacian, filter_taubin


def _load_mesh(stl_path: Path) -> trimesh.Trimesh:
    """
    载入 STL 文件并统一返回 trimesh.Trimesh 对象。

    体素化生成的 STL 有时会被解析为包含多个子几何体的 Scene，
    此函数会将所有子几何体合并为单一网格，确保后续处理的一致性。

    Args:
        stl_path: 输入 STL 文件的路径。

    Returns:
        trimesh.Trimesh: 合并后的三角网格对象。
    """
    # force="mesh" 强制将文件解析为网格，而非场景或点云
    loaded = trimesh.load(str(stl_path), force="mesh")

    if isinstance(loaded, trimesh.Scene):
        # 若解析结果是 Scene（含多个子几何体），则合并为单一网格
        if not loaded.geometry:
            raise ValueError(f"No geometry found in: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")

    return mesh


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    对网格进行基础修复，清除无效数据，确保网格拓扑干净。

    体素化 STL 通常存在孤立顶点、退化三角面、重复面片等问题，
    平滑和细分算法对网格质量敏感，修复后可防止这些操作产生异常。

    Args:
        mesh: 待修复的三角网格。

    Returns:
        trimesh.Trimesh: 修复后的网格（原地修改后返回）。
    """
    mesh.remove_unreferenced_vertices()  # 删除没有被任何面片引用的孤立顶点
    mesh.remove_degenerate_faces()       # 删除面积为零的退化三角面（三点共线或重合）
    mesh.remove_duplicate_faces()        # 删除完全重叠的重复面片
    mesh.fill_holes()                    # 填补网格上的孔洞，使曲面尽量封闭
    mesh.remove_infinite_values()        # 清除顶点坐标中出现的 NaN / Inf 值
    return mesh


def smooth_voxel_stl(
    input_stl: Path,
    output_stl: Path,
    method: str = "taubin",
    smooth_iterations: int = 30,
    subdivision_iterations: int = 1,
    laplacian_lambda: float = 0.45,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.53,
    humphrey_alpha: float = 0.1,
    humphrey_beta: float = 0.6,
) -> None:
    """
    对体素化生成的 STL 模型进行平滑处理，并将结果导出为新的 STL 文件。

    体素化 STL 的表面呈明显的阶梯锯齿状，本函数通过先细分再平滑的流程
    消除这些锯齿，得到视觉上光滑的曲面模型。

    处理流程：载入 → 修复 → Loop细分 → 平滑滤波 → 修复 → 导出

    Args:
        input_stl (Path):
            输入 STL 文件路径，即体素化后导出的原始模型。

        output_stl (Path):
            输出 STL 文件路径，平滑后的结果将保存到此路径。
            若目录不存在会自动创建。

        method (str):
            平滑算法类型，可选值：
            - "taubin"    （默认）Taubin 平滑：交替使用正负步长抑制体积收缩，
                          是体素 STL 平滑的最佳选择，平滑效果好且形状保持好。
            - "laplacian" 拉普拉斯平滑：将每个顶点移向邻居的平均位置，
                          速度快但多次迭代后会导致网格整体收缩变小。
            - "humphrey" Humphrey 平滑：在拉普拉斯的基础上加入反馈修正，
                          比拉普拉斯更好地保留原始形状特征。

        smooth_iterations (int):
            平滑迭代次数，默认 30。
            数值越大，表面越光滑，但几何细节损失也越多；
            通常 10~50 之间效果较好。

        subdivision_iterations (int):
            Loop 细分迭代次数，默认 1。
            细分会在平滑前增加网格的三角面片密度（每次迭代面数约增加 4 倍），
            面片越密，平滑后的曲面越细腻；设为 0 表示跳过细分直接平滑。
            注意：细分次数过多会显著增加内存和计算时间。

        laplacian_lambda (float):
            拉普拉斯平滑的步长系数，默认 0.45，范围建议 (0, 1)。
            控制每次迭代顶点向邻居均值移动的幅度，越大移动越激进，
            仅在 method="laplacian" 时生效。

        taubin_lambda (float):
            Taubin 平滑的正向步长系数，默认 0.5，范围建议 (0, 1)。
            正向步长使顶点向邻居均值靠拢（平滑），
            仅在 method="taubin" 时生效。

        taubin_nu (float):
            Taubin 平滑的反向步长系数，默认 -0.53，范围建议 (-1, 0)。
            反向步长与 lambda 方向相反，用于抵消体积收缩效应；
            |nu| 应略大于 lambda 以保证收敛，
            仅在 method="taubin" 时生效。

        humphrey_alpha (float):
            Humphrey 平滑的原始位置保留权重，默认 0.1，范围 [0, 1]。
            值越大，顶点越倾向于保持原始位置（平滑效果减弱）；
            仅在 method="humphrey" 时生效。

        humphrey_beta (float):
            Humphrey 平滑的反馈修正强度，默认 0.6，范围 [0, 1]。
            值越大，对拉普拉斯收缩的修正越强，形状保留越好，
            但过大可能导致震荡；仅在 method="humphrey" 时生效。
    """
    mesh = _load_mesh(input_stl)
    mesh = _repair_mesh(mesh)  # 平滑前先修复，避免坏面干扰细分

    if subdivision_iterations > 0:
        # Loop 细分：增加面片密度，使平滑后曲面更细腻
        # 每迭代一次，面片数量约变为原来的 4 倍
        mesh = mesh.subdivide_loop(iterations=subdivision_iterations)

    method = method.lower().strip()
    if method == "taubin":
        # Taubin 平滑：正向步长(lamb)平滑 + 反向步长(nu)补偿，抑制体积缩小
        filter_taubin(mesh, lamb=taubin_lambda, nu=taubin_nu, iterations=smooth_iterations)
    elif method == "laplacian":
        # 拉普拉斯平滑：每个顶点移向其相邻顶点的加权平均位置
        filter_laplacian(mesh, lamb=laplacian_lambda, iterations=smooth_iterations)
    elif method == "humphrey":
        # Humphrey 平滑：带反馈修正的拉普拉斯，更好地保留原始形状
        filter_humphrey(
            mesh,
            alpha=humphrey_alpha,
            beta=humphrey_beta,
            iterations=smooth_iterations,
        )
    else:
        raise ValueError(
            f"Unsupported method: {method}. Choose from: taubin, laplacian, humphrey"
        )

    mesh = _repair_mesh(mesh)  # 平滑后再次修复，清除可能引入的无效数据
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_stl))

    print("STL smoothing complete")
    print(f"input  : {input_stl}")
    print(f"output : {output_stl}")
    print(f"method : {method}")
    print(f"faces  : {len(mesh.faces)}")
    print(f"verts  : {len(mesh.vertices)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对体素化生成的 STL 模型进行平滑处理，输出光滑的 STL 文件。"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="[必填] 输入 STL 文件路径（体素化后的原始模型）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="[必填] 输出 STL 文件路径（平滑后的结果）",
    )
    parser.add_argument(
        "--method",
        default="taubin",
        choices=["taubin", "laplacian", "humphrey"],
        help=(
            "平滑算法类型，默认 taubin。"
            " taubin: 保形平滑，体积收缩小，推荐首选；"
            " laplacian: 速度快，但多次迭代后网格会收缩；"
            " humphrey: 带修正的拉普拉斯，特征保留较好。"
        ),
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=30,
        help="平滑迭代次数，默认 30。越大越光滑，但几何细节损失也越多，建议范围 10~50。",
    )
    parser.add_argument(
        "--subdivision-iterations",
        type=int,
        default=1,
        help=(
            "Loop 细分迭代次数，默认 1。"
            " 细分在平滑前增加面片密度（每次约 ×4），使平滑曲面更细腻。"
            " 设为 0 可跳过细分；注意次数过多会大幅增加内存和耗时。"
        ),
    )
    parser.add_argument(
        "--laplacian-lambda",
        type=float,
        default=0.45,
        help="拉普拉斯平滑步长系数，默认 0.45，范围 (0,1)。越大移动越激进，仅 method=laplacian 时生效。",
    )
    parser.add_argument(
        "--taubin-lambda",
        type=float,
        default=0.5,
        help="Taubin 正向步长系数，默认 0.5，范围 (0,1)。控制平滑幅度，仅 method=taubin 时生效。",
    )
    parser.add_argument(
        "--taubin-nu",
        type=float,
        default=-0.53,
        help=(
            "Taubin 反向步长系数，默认 -0.53，范围 (-1,0)。"
            " 与 lambda 方向相反，补偿体积收缩；|nu| 应略大于 lambda，仅 method=taubin 时生效。"
        ),
    )
    parser.add_argument(
        "--humphrey-alpha",
        type=float,
        default=0.1,
        help="Humphrey 原始位置保留权重，默认 0.1，范围 [0,1]。越大越保留原形，仅 method=humphrey 时生效。",
    )
    parser.add_argument(
        "--humphrey-beta",
        type=float,
        default=0.6,
        help=(
            "Humphrey 反馈修正强度，默认 0.6，范围 [0,1]。"
            " 越大对收缩的修正越强，但过大可能震荡，仅 method=humphrey 时生效。"
        ),
    )
    return parser


def main() -> None:
    """命令行入口：解析参数并调用 smooth_voxel_stl 完成平滑处理。"""
    parser = build_parser()
    args = parser.parse_args()

    # 参数合法性校验
    if args.smooth_iterations < 1:
        raise ValueError("--smooth-iterations must be >= 1")
    if args.subdivision_iterations < 0:
        raise ValueError("--subdivision-iterations must be >= 0")

    smooth_voxel_stl(
        input_stl=args.input,
        output_stl=args.output,
        method=args.method,
        smooth_iterations=args.smooth_iterations,
        subdivision_iterations=args.subdivision_iterations,
        laplacian_lambda=args.laplacian_lambda,
        taubin_lambda=args.taubin_lambda,
        taubin_nu=args.taubin_nu,
        humphrey_alpha=args.humphrey_alpha,
        humphrey_beta=args.humphrey_beta,
    )


if __name__ == "__main__":
    main()
