"""
voxel_txt_to_stl.py
====================
将 voxelization.py 输出的体素 .txt 文件转换为完全光顺（无阶梯）的 STL 网格。

核心思路：为什么不直接对二值场做高斯模糊？
-------------------------------------------
对 0/1 二值场做高斯模糊得到的是"糊开的边界"，Marching Cubes 提取的等值面
仍然受体素栅格约束，sigma 不够大时阶梯感依然明显。

本脚本改用**符号距离场（SDF）**作为 Marching Cubes 的输入：
  - SDF(p) = 点 p 到最近表面的欧氏距离（内部为正，外部为负）
  - SDF 是真正连续光滑的标量场，MC 在 SDF=0 处的等值面
    不受体素栅格影响，几何上最接近真实曲面，阶梯感最小
  - 再配合少量 Taubin 网格平滑，可做到视觉上完全无阶梯

处理流程
--------
1. 读取体素文件（writeVoxelMatrix 格式）
2. 用 Euclidean Distance Transform 计算 SDF
3. 可选：对 SDF 做轻微高斯平滑（进一步软化边界）
4. Marching Cubes 在 SDF=0 处提取等值面
5. Taubin 网格平滑（消除残余锯齿，体积收缩极小）
6. 导出 STL

体素文件格式（writeVoxelMatrix 生成）
--------------------------------------
第 1 行：nx,ny,nz       （体素矩阵的 x/y/z 方向尺寸）
第 2 行：0,0,0           （原点占位，留作扩展）
第 3 行：flattened 体素值（0/1，逗号分隔，按 z>y>x 顺序展开）

用法示例
--------
# 最简单的用法（全用默认参数，推荐）
python fieldopt/geometry/voxel_txt_to_stl.py \\
    --input  initialFields/boneTPMS/boneTPMS_res80_voxels.txt \\
    --output output/boneTPMS_smooth.stl

# 更光滑（增加 Taubin 迭代次数）
python fieldopt/geometry/voxel_txt_to_stl.py \\
    --input  initialFields/fertility/fertility_res80_voxels.txt \\
    --output output/fertility_smooth.stl \\
    --smooth-iterations 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.measure import marching_cubes
import trimesh
from trimesh.smoothing import filter_taubin


# ---------------------------------------------------------------------------
# 体素文件读取
# ---------------------------------------------------------------------------

def read_voxel_txt(filepath: Path) -> np.ndarray:
    """
    读取 voxelization.py / writeVoxelMatrix 格式的体素文件。

    文件格式：
        第 1 行： nx,ny,nz
        第 2 行： 0,0,0  （原点占位）
        第 3 行起：逗号分隔的 0/1 值，按 voxel_matrix.flatten() 展开
                  （展开顺序：z 最慢变，y 次之，x 最快）

    Returns
    -------
    np.ndarray
        shape = (nz, ny, nx)，dtype = float32，值为 0.0 或 1.0
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Voxel file not found: {filepath}")

    with filepath.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 3:
        raise ValueError(
            f"Voxel file must have at least 3 lines, got {len(lines)}: {filepath}"
        )

    # 第 1 行：nx, ny, nz（writeVoxelMatrix 写入顺序为 boardSize[2],boardSize[1],boardSize[0]）
    dims = list(map(int, lines[0].split(",")))
    if len(dims) != 3:
        raise ValueError(f"Unexpected dimension line: '{lines[0]}'")
    nx, ny, nz = dims  # boardSize[2]=nx, boardSize[1]=ny, boardSize[0]=nz

    # 第 3 行：展开的体素值（跳过第 2 行的原点占位）
    voxel_flat = np.fromstring(lines[2], dtype=np.float32, sep=",")

    expected = nz * ny * nx
    if voxel_flat.size != expected:
        raise ValueError(
            f"Expected {expected} voxel values ({nz}×{ny}×{nx}), "
            f"got {voxel_flat.size}"
        )

    # 重塑为 (nz, ny, nx)：与 voxel_matrix.flatten() 的 C-order 对应
    voxel_matrix = voxel_flat.reshape(nz, ny, nx)
    return voxel_matrix


# ---------------------------------------------------------------------------
# 转换主函数
# ---------------------------------------------------------------------------

def _compute_sdf(voxel: np.ndarray) -> np.ndarray:
    """
    从二值体素场计算符号距离场（SDF）。

    SDF(p) 定义：
        内部点（voxel=1）：  到最近表面的欧氏距离（正值）
        外部点（voxel=0）：  到最近表面的欧氏距离的负值（负值）

    使用 scipy 的 Euclidean Distance Transform（精确欧氏距离变换），
    比高斯模糊更准确地描述真实几何曲面，Marching Cubes 在 SDF=0
    处的等值面几何上最接近原始曲面，阶梯感最小。

    Returns
    -------
    np.ndarray（float32，与 voxel 同形状）
        符号距离场，单位为体素
    """
    interior = voxel > 0.5
    # edt_in：内部各点到最近边界（从内向外）的距离
    edt_in  = distance_transform_edt(interior).astype(np.float32)
    # edt_out：外部各点到最近边界（从外向内）的距离
    edt_out = distance_transform_edt(~interior).astype(np.float32)
    sdf = edt_in - edt_out   # 内部为正，外部为负，表面为 0
    return sdf


def voxel_to_smooth_stl(
    input_voxel: Path,
    output_stl: Path,
    sigma: float = 0.5,
    mc_level: float = -1.0,
    mesh_smooth_iterations: int = 30,
    taubin_lambda: float = 0.5,
    taubin_nu: float = -0.56,
) -> trimesh.Trimesh:
    """
    将体素 .txt 文件转换为完全光顺（无阶梯）的 STL 文件。

    流程：读取体素 → SDF → 可选高斯平滑 SDF → Marching Cubes → Taubin 平滑 → 导出

    Parameters
    ----------
    input_voxel : Path
        输入体素文件路径（voxelization.py 输出的 _voxels.txt）。
    output_stl : Path
        输出 STL 文件路径。
    sigma : float
        对 SDF 施加的高斯平滑标准差（单位：体素），默认 0.5。
        SDF 本身已是连续光滑场，此处只需轻微平滑；设为 0 跳过。
    mc_level : float
        Marching Cubes 提取等值面时使用的 SDF 阈值，默认 -0.3。
        高斯模糊（sigma>0）会使 SDF 零点向内漂移（等值面收缩体积偏小）；
        通过设为负值可将等值面外推回来，补偿体积损失。
        - 设为 0.0：精确跟踪 SDF=0 的表面（sigma>0 时体积偏小）
        - 设为 -0.3~-0.5：补偿 sigma=0.5 引起的收缩（推荐）
        - 值越负，提取的体积越大
    mesh_smooth_iterations : int
        Marching Cubes 后的 Taubin 网格平滑迭代数，默认 30。
        Taubin 平滑是消除残余锯齿的关键步骤，设为 0 可跳过。
    taubin_lambda : float
        Taubin 正向步长，默认 0.5，范围 (0, 1)。
    taubin_nu : float
        Taubin 反向步长，默认 -0.56，范围 (-1, 0)。
        |nu| 应明显大于 lambda 以补偿多次迭代后的体积净收缩。
        （lambda=0.5, nu=-0.56 → 每对迭代净收缩仅约 1.2%）

    Returns
    -------
    trimesh.Trimesh
        平滑后的三角网格对象。
    """
    # ---- 1. 读取体素 ----
    print(f"[1/4] 读取体素文件: {input_voxel}")
    voxel = read_voxel_txt(input_voxel)
    nz, ny, nx = voxel.shape
    fill_ratio = voxel.mean()
    print(f"      体素尺寸: nz={nz}, ny={ny}, nx={nx}  填充率: {fill_ratio:.3f}")

    # 填充：为了使得落在边界上的体素也能形成封闭的面，我们在所有维度填充一圈0(外部)
    # 注意：mc_level 为负时等值面会往外扩张 |mc_level| 个体素，padding 必须足够容纳这个扩张，
    # 否则外边界的面片无法生成，导致大片缺失。
    # 安全 padding = max(1, ceil(|mc_level|)) + 1（额外 +1 给高斯平滑留余量）
    import math
    pad_width = max(1, math.ceil(abs(min(mc_level, 0.0)))) + 1
    voxel = np.pad(voxel, pad_width=pad_width, mode='constant', constant_values=0)
    print(f"      自动 padding: {pad_width} 层（mc_level={mc_level}）")

    # ---- 2. 计算 SDF（符号距离场）----
    # 相比直接高斯模糊二值场：SDF 是真正连续的几何距离场，
    # MC 在 level=0 处的等值面不受体素栅格约束，阶梯感最小
    print("[2/4] 计算符号距离场 (SDF)...")
    sdf = _compute_sdf(voxel)
    if sigma > 0:
        print(f"      对 SDF 施加高斯平滑 (sigma={sigma})...")
        sdf = gaussian_filter(sdf, sigma=sigma)

    # ---- 3. Marching Cubes 在 SDF=mc_level 处提取等值面 ----
    # 注意：对 SDF 施加高斯平滑后，零点会向内漂移（等值面对应的体积变小）。
    # 通过将 level 设为负值（如 -1.0），可以补偿这一收缩，使提取的体积更接近原始体素体积。
    print(f"[3/4] Marching Cubes (level={mc_level}, sigma={sigma} 的收缩补偿)...")
    verts, faces, normals, _ = marching_cubes(sdf, level=mc_level, step_size=1)
    
    # 修复 0: 去除由于 np.pad(pad_width) 所带来的网格坐标偏移（偏移量 = pad_width）
    verts = verts - float(pad_width)

    # 修复 1: marching_cubes 返回的顶点坐标是 (z, y, x) 顺序，需要调换为 (x, y, z)
    # 因为 SDF (edt_in - edt_out) 的梯度指向内部，marching_cubes 提取的面法向默认向内。
    # 调换 X 和 Z 轴相当于进行了一次镜像翻转，使得原本向内的法向变成了向外！
    # 因此我们不需要（也不应该）再反转面片的顶点连接顺序。
    verts = verts[:, ::-1]

    # 修复 3: 按网格最大尺寸进行统一缩放，以保留模型的原始宽高比（而非缩放为 1x1x1 的正方体）
    max_dim = float(max(max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)))
    verts_norm = verts / max_dim
    
    # 忽略可能不正确的 normals，让 trimesh 自动重新计算顶点法向
    mesh = trimesh.Trimesh(vertices=verts_norm, faces=faces)
    trimesh.repair.fix_normals(mesh)
    print(f"      提取网格: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")

    # ---- 4. Taubin 网格平滑（消除残余锯齿，体积收缩极小）----
    if mesh_smooth_iterations > 0:
        print(
            f"[4/4] Taubin 网格平滑 "
            f"(iterations={mesh_smooth_iterations}, "
            f"lambda={taubin_lambda}, nu={taubin_nu})..."
        )
        filter_taubin(
            mesh,
            lamb=taubin_lambda,
            nu=taubin_nu,
            iterations=mesh_smooth_iterations,
        )
    else:
        print("[4/4] 跳过网格平滑（mesh_smooth_iterations=0）")

    # ---- 导出 ----
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_stl))
    print(f"\n完成！输出 STL: {output_stl}")
    print(f"最终网格: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")
    return mesh


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将 voxelization.py 输出的体素 .txt 文件转换为完全光顺（无阶梯）的 STL。\n"
            "流程：读取体素 → 符号距离场(SDF) → 可选高斯平滑 → Marching Cubes → Taubin 网格平滑 → 导出 STL"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="VOXEL_TXT",
        help="[必填] 输入体素文件路径（*_voxels.txt）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="STL_FILE",
        help="[必填] 输出 STL 文件路径",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help=(
            "对 SDF 施加的高斯平滑标准差（单位：体素），默认 0.5。"
            " SDF 本身已是连续光滑场，只需轻微平滑；设为 0 可跳过。"
        ),
    )
    parser.add_argument(
        "--mc-level",
        type=float,
        default=-0.5,
        metavar="FLOAT",
        help=(
            "Marching Cubes 等值面阈值，默认 -1.0。"
            " 高斯模糊(sigma>0)会使 SDF 零点向内漂移导致体积偏小；"
            " 设为负值可将等值面外推补偿体积损失（越负体积越大）。"
            " sigma=0 时建议设为 0.0。"
        ),
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=2,
        metavar="INT",
        help="Marching Cubes 后的 Taubin 网格平滑迭代数，默认 30；设为 0 跳过。",
    )
    parser.add_argument(
        "--taubin-lambda",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Taubin 正向步长，默认 0.5，范围 (0,1)。",
    )
    parser.add_argument(
        "--taubin-nu",
        type=float,
        default=-0.56,
        metavar="FLOAT",
        help="Taubin 反向步长，默认 -0.56，范围 (-1,0)；|nu| 应明显大于 lambda 以补偿多次迭代的净收缩。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.sigma < 0:
        parser.error("--sigma must be >= 0")
    if args.smooth_iterations < 0:
        parser.error("--smooth-iterations must be >= 0")

    voxel_to_smooth_stl(
        input_voxel=args.input,
        output_stl=args.output,
        sigma=args.sigma,
        mc_level=args.mc_level,
        mesh_smooth_iterations=args.smooth_iterations,
        taubin_lambda=args.taubin_lambda,
        taubin_nu=args.taubin_nu,
    )


if __name__ == "__main__":
    main()
