"""
Offline training script: trains a small MLP to represent the Signed Distance
Field (SDF) of a given STL mesh and saves the result to
    stlFiles/<model_name>_sdf.pt

A Signed Distance Field convention:
    SDF < 0  →  point is inside the model
    SDF > 0  →  point is outside the model

Usage (run from the project root):
    conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf --stl stlFiles/bracket.stl
    conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf \
        --stl stlFiles/bracket.stl --epochs 3000 --device cuda

Public symbols used by other modules:
    SDFNet            – the MLP architecture
    FourierEncoding   – sinusoidal positional encoding
    normalise_pts     – maps coords to [-1, 1] given stored AABB
    load_trained_sdf  – load a saved neural-SDF checkpoint
    predict_sdf_points
    query_point_labels
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import trimesh
from tqdm import tqdm

# Support both "python -m fieldopt.geometry.sdf.train_sdf" (script) and
# "from .train_sdf import SDFNet" (package import).
try:
    from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fieldopt.geometry.voxel.voxelization import get_normalization_parameters


# ─────────────────────────── Network Architecture ────────────────────────────

class FourierEncoding(nn.Module):
    """Maps 3-D coords to sinusoidal features of dimension ``3 + 3*2*L``."""

    def __init__(self, L: int = 6) -> None:
        super().__init__()
        self.L = L
        freqs = (2.0 ** torch.arange(L, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 3 + 3 * 2 * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.view(*([1] * (x.ndim - 1)), self.L)
        x_proj = x.unsqueeze(-1) * freqs
        sin_cos = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return torch.cat([x, sin_cos.flatten(start_dim=-2)], dim=-1)


class SDFNet(nn.Module):
    """MLP with Fourier positional encoding for signed distance regression."""

    def __init__(
        self,
        fourier_L: int = 6,
        hidden_dim: int = 256,
        n_hidden: int = 4,
    ) -> None:
        super().__init__()
        self.encoding = FourierEncoding(L=fourier_L)
        in_dim = self.encoding.out_dim

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Softplus(beta=100)]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Softplus(beta=100)]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.encoding(x))


# ─────────────────────────── Coordinate Normalisation ────────────────────────

def normalise_pts(
    pts: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
) -> torch.Tensor:
    """Normalise coordinates from AABB space to [-1, 1]."""
    return 2.0 * (pts - aabb_min) / (aabb_max - aabb_min).clamp(min=1e-8) - 1.0


# ─────────────────────────── Mesh Utilities ──────────────────────────────────

def _repair_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.process()
    mesh.merge_vertices()
    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()
    return mesh


def _load_normalised_mesh(stl_file: str) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Load an STL, apply the project's standard normalisation, attempt repairs
    on non-watertight components, and return
        (mesh, aabb_min [float32], aabb_max [float32]).
    """
    scale, p_min, _ = get_normalization_parameters(stl_file)
    mesh = trimesh.load_mesh(stl_file)
    mesh.apply_translation(-p_min)
    mesh.apply_scale(scale)

    if not mesh.is_watertight:
        components = list(mesh.split(only_watertight=False))
        repaired = [_repair_component(c) for c in components]
        mesh = trimesh.util.concatenate(repaired)

    aabb_min = mesh.bounds[0].astype(np.float32)
    aabb_max = mesh.bounds[1].astype(np.float32)
    return mesh, aabb_min, aabb_max


# ─────────────────────────── Ground-truth SDF ────────────────────────────────

def compute_sdf_gt(mesh: trimesh.Trimesh, pts: np.ndarray) -> np.ndarray:
    """
    Compute signed distance for an array of query points.
    Convention: negative inside the mesh, positive outside.
    """
    _, dists, _ = trimesh.proximity.closest_point(mesh, pts)
    inside = mesh.contains(pts)
    signs = np.where(inside, -1.0, 1.0)
    return (dists * signs).astype(np.float32)


# ─────────────────────────── Dataset Generation ──────────────────────────────

def sample_training_points(
    mesh: trimesh.Trimesh,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    n_surf: int,
    n_vol: int,
    surface_sigma_ratio: float = 0.01,
    surface_pad_ratio: float = 0.05,
    volume_pad_ratio: float = 0.10,
    balance_volume_signs: bool = False,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
    verbose: bool = True,
) -> np.ndarray:
    """
    Sample a mixed set of near-surface and volume points in normalised space.
    """
    diag = float(np.linalg.norm(aabb_max - aabb_min))
    sigma = surface_sigma_ratio * diag

    if verbose:
        print(f"  Sampling {n_surf:,} near-surface points (sigma={sigma:.4f})...")
    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surf)
    surf_pts = (surf_pts + np.random.randn(*surf_pts.shape) * sigma).astype(np.float32)

    # Allow slight out-of-AABB to cover the surface shell.
    aabb_pad = surface_pad_ratio * diag
    surf_pts = np.clip(surf_pts, aabb_min - aabb_pad, aabb_max + aabb_pad)

    # Extend volume sampling beyond the AABB so the network also learns the
    # outside region (positive SDF).
    vol_pad = volume_pad_ratio * (aabb_max - aabb_min)
    vol_min = aabb_min - vol_pad
    vol_max = aabb_max + vol_pad
    if verbose:
        mode = "balanced" if balance_volume_signs else "random"
        print(
            f"  Sampling {n_vol:,} {mode} volume points "
            f"(AABB ± {100 * volume_pad_ratio:.0f}%)..."
        )
    if balance_volume_signs and n_vol > 0:
        vol_pts = sample_balanced_volume_points(
            mesh=mesh,
            vol_min=vol_min,
            vol_max=vol_max,
            n_vol=n_vol,
            target_inside_ratio=volume_balance_ratio,
            oversample_factor=volume_balance_oversample,
        )
    else:
        vol_pts = (
            vol_min
            + np.random.rand(n_vol, 3).astype(np.float32) * (vol_max - vol_min)
        )

    return np.vstack([surf_pts, vol_pts]).astype(np.float32)


def sample_balanced_volume_points(
    mesh: trimesh.Trimesh,
    vol_min: np.ndarray,
    vol_max: np.ndarray,
    n_vol: int,
    target_inside_ratio: float = 0.5,
    oversample_factor: int = 6,
    max_rounds: int = 12,
) -> np.ndarray:
    """
    Rejection-sample volume points until inside/outside counts are closer to
    the requested ratio than naive uniform sampling would provide.
    """
    if n_vol <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    target_inside_ratio = float(np.clip(target_inside_ratio, 0.0, 1.0))
    target_inside = int(round(n_vol * target_inside_ratio))
    target_outside = n_vol - target_inside
    oversample_factor = max(int(oversample_factor), 2)
    inside_chunks: list[np.ndarray] = []
    outside_chunks: list[np.ndarray] = []
    n_inside = 0
    n_outside = 0

    for _ in range(max(int(max_rounds), 1)):
        need_inside = max(target_inside - n_inside, 0)
        need_outside = max(target_outside - n_outside, 0)
        if need_inside <= 0 and need_outside <= 0:
            break

        n_candidates = max((need_inside + need_outside) * oversample_factor, 4096)
        candidates = (
            vol_min
            + np.random.rand(n_candidates, 3).astype(np.float32) * (vol_max - vol_min)
        )
        inside_mask = mesh.contains(candidates)

        if need_inside > 0:
            inside_pts = candidates[inside_mask]
            if len(inside_pts) > 0:
                take = min(len(inside_pts), need_inside)
                inside_chunks.append(inside_pts[:take])
                n_inside += take

        if need_outside > 0:
            outside_pts = candidates[~inside_mask]
            if len(outside_pts) > 0:
                take = min(len(outside_pts), need_outside)
                outside_chunks.append(outside_pts[:take])
                n_outside += take

    inside_pts = (
        np.concatenate(inside_chunks, axis=0)[:target_inside]
        if inside_chunks else np.zeros((0, 3), dtype=np.float32)
    )
    outside_pts = (
        np.concatenate(outside_chunks, axis=0)[:target_outside]
        if outside_chunks else np.zeros((0, 3), dtype=np.float32)
    )

    shortfall = n_vol - len(inside_pts) - len(outside_pts)
    if shortfall > 0:
        fallback = (
            vol_min
            + np.random.rand(shortfall, 3).astype(np.float32) * (vol_max - vol_min)
        )
        return np.vstack([inside_pts, outside_pts, fallback]).astype(np.float32)

    return np.vstack([inside_pts, outside_pts]).astype(np.float32)


def _to_device_tensors(
    pts_np: np.ndarray,
    sdf_np: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pts_t = torch.tensor(pts_np, dtype=torch.float32, device=device)
    sdf_t = torch.tensor(sdf_np, dtype=torch.float32, device=device).unsqueeze(-1)
    return pts_t, sdf_t


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _configure_torch_runtime(device: str) -> None:
    if not _is_cuda_device(device):
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _resolve_amp_settings(device: str) -> tuple[bool, torch.dtype | None, str, bool]:
    if not _is_cuda_device(device):
        return False, None, "disabled", False

    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bf16", False
    return True, torch.float16, "fp16", True


def _autocast_context(
    device: str,
    enabled: bool,
    amp_dtype: torch.dtype | None,
):
    if not enabled or amp_dtype is None:
        return contextlib.nullcontext()
    device_type = "cuda" if _is_cuda_device(device) else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _autotune_batch_size(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    aabb_min_t: torch.Tensor,
    aabb_max_t: torch.Tensor,
    lr: float,
    lambda_eikonal: float,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    starting_batch_size: int = 8192,
    max_batch_size: int = 262_144,
) -> int:
    if not _is_cuda_device(device):
        return max(int(starting_batch_size), 1)

    batch_size = max(int(starting_batch_size), 1024)
    max_batch_size = max(batch_size, int(max_batch_size))
    candidate = batch_size
    best = batch_size

    while candidate <= max_batch_size:
        pts_in = None
        pred = None
        grads = None
        loss = None
        mse = None
        eikonal = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)
            optimizer.zero_grad(set_to_none=True)

            pts_in = torch.empty(candidate, 3, dtype=torch.float32, device=device)
            pts_in.uniform_(0.0, 1.0)
            pts_in = pts_in.requires_grad_(True)
            sdf_target = torch.zeros(candidate, 1, dtype=torch.float32, device=device)

            with _autocast_context(device, amp_enabled, amp_dtype):
                pred = net(normalise_pts(pts_in, aabb_min_t, aabb_max_t))

            pred_fp32 = pred.float()
            mse = nn.functional.mse_loss(pred_fp32, sdf_target)
            grads = torch.autograd.grad(pred_fp32.sum(), pts_in, create_graph=True)[0]
            eikonal = ((grads.float().norm(dim=-1) - 1.0) ** 2).mean()
            loss = mse + lambda_eikonal * eikonal
            loss.backward()
            best = candidate
            candidate *= 2
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            break
        finally:
            optimizer.zero_grad(set_to_none=True)
            del pts_in, pred, grads, loss, mse, eikonal
            torch.cuda.empty_cache()

    optimizer.param_groups[0]["lr"] = lr
    return best


def _summarize_sdf_samples(sdf_np: np.ndarray) -> str:
    n_total = len(sdf_np)
    n_inside = int((sdf_np < 0).sum())
    return (
        f"Total: {n_total:,}  |  inside: {n_inside:,} "
        f"({100 * n_inside / max(n_total, 1):.1f}%)  |  outside: {n_total - n_inside:,}"
    )


def refresh_training_pool(
    mesh: trimesh.Trimesh,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    pts_t: torch.Tensor,
    sdf_t: torch.Tensor,
    device: str,
    refresh_fraction: float,
    n_surf: int,
    n_vol: int,
    balance_volume_signs: bool,
    volume_balance_ratio: float,
    volume_balance_oversample: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Replace a subset of the current sample pool with freshly sampled points.
    """
    refresh_fraction = float(np.clip(refresh_fraction, 0.0, 1.0))
    pool_size = len(pts_t)
    n_refresh = int(round(pool_size * refresh_fraction))
    if n_refresh <= 0:
        return pts_t, sdf_t, 0

    total_requested = max(n_surf + n_vol, 1)
    n_refresh_surf = int(round(n_refresh * n_surf / total_requested))
    n_refresh_surf = min(n_refresh_surf, n_refresh)
    n_refresh_vol = n_refresh - n_refresh_surf

    fresh_pts = sample_training_points(
        mesh,
        aabb_min,
        aabb_max,
        n_surf=n_refresh_surf,
        n_vol=n_refresh_vol,
        balance_volume_signs=balance_volume_signs,
        volume_balance_ratio=volume_balance_ratio,
        volume_balance_oversample=volume_balance_oversample,
        verbose=False,
    )
    fresh_sdf = compute_sdf_gt(mesh, fresh_pts)
    fresh_pts_t, fresh_sdf_t = _to_device_tensors(fresh_pts, fresh_sdf, device=device)

    replace_idx = torch.randperm(pool_size, device=device)[:n_refresh]
    pts_t[replace_idx] = fresh_pts_t
    sdf_t[replace_idx] = fresh_sdf_t
    return pts_t, sdf_t, n_refresh


def generate_dataset(
    mesh: trimesh.Trimesh,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    n_surf: int = 150_000,
    n_vol: int = 150_000,
    balance_volume_signs: bool = False,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample near-surface and volume points, compute ground-truth SDF.
    Returns (pts [N, 3], sdf [N]) in normalised mesh space.
    """
    pts = sample_training_points(
        mesh,
        aabb_min,
        aabb_max,
        n_surf,
        n_vol,
        balance_volume_signs=balance_volume_signs,
        volume_balance_ratio=volume_balance_ratio,
        volume_balance_oversample=volume_balance_oversample,
    )
    print(f"  Computing ground-truth SDF for {len(pts):,} points...")
    sdf = compute_sdf_gt(mesh, pts)
    return pts, sdf


def sample_validation_dataset(
    mesh: trimesh.Trimesh,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    n_eval: int,
    volume_pad_ratio: float = 0.10,
    balance_volume_signs: bool = False,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a validation set once so repeated evaluations compare against the
    exact same points and ground-truth SDF values.
    """
    diag = float(np.linalg.norm(aabb_max_np - aabb_min_np))
    n_half = n_eval // 2

    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_half)
    surf_pts = (
        surf_pts + np.random.randn(*surf_pts.shape) * 0.005 * diag
    ).astype(np.float32)
    aabb_pad = 0.02 * diag
    surf_pts = np.clip(surf_pts, aabb_min_np - aabb_pad, aabb_max_np + aabb_pad)

    vol_pad = volume_pad_ratio * (aabb_max_np - aabb_min_np)
    vol_min = aabb_min_np - vol_pad
    vol_max = aabb_max_np + vol_pad
    if balance_volume_signs and n_half > 0:
        vol_pts = sample_balanced_volume_points(
            mesh=mesh,
            vol_min=vol_min,
            vol_max=vol_max,
            n_vol=n_half,
            target_inside_ratio=volume_balance_ratio,
            oversample_factor=volume_balance_oversample,
        )
    else:
        vol_pts = (
            vol_min
            + np.random.rand(n_half, 3).astype(np.float32) * (vol_max - vol_min)
        )
    pts_eval = np.vstack([surf_pts, vol_pts]).astype(np.float32)
    sdf_gt = compute_sdf_gt(mesh, pts_eval).flatten()
    return pts_eval, sdf_gt


def sample_volume_validation_dataset(
    mesh: trimesh.Trimesh,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    n_eval: int,
    volume_pad_ratio: float = 0.10,
    balance_volume_signs: bool = True,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a volume-only validation set, which is more representative for
    point-inside / point-outside queries than a surface-heavy validation set.
    """
    vol_pad = volume_pad_ratio * (aabb_max_np - aabb_min_np)
    vol_min = aabb_min_np - vol_pad
    vol_max = aabb_max_np + vol_pad

    if balance_volume_signs and n_eval > 0:
        pts_eval = sample_balanced_volume_points(
            mesh=mesh,
            vol_min=vol_min,
            vol_max=vol_max,
            n_vol=n_eval,
            target_inside_ratio=volume_balance_ratio,
            oversample_factor=volume_balance_oversample,
        )
    else:
        pts_eval = (
            vol_min
            + np.random.rand(n_eval, 3).astype(np.float32) * (vol_max - vol_min)
        )
    sdf_gt = compute_sdf_gt(mesh, pts_eval).flatten()
    return pts_eval.astype(np.float32), sdf_gt


def _predict_sdf_numpy(
    net: SDFNet,
    pts_np: np.ndarray,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    device: str,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the network on a numpy point cloud and return numpy SDFs."""
    _configure_torch_runtime(device)
    amp_enabled, amp_dtype, _, _ = _resolve_amp_settings(device)
    aabb_min_t = torch.tensor(aabb_min_np, dtype=torch.float32, device=device)
    aabb_max_t = torch.tensor(aabb_max_np, dtype=torch.float32, device=device)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(pts_np), chunk_size):
            pts_t = torch.tensor(
                pts_np[start : start + chunk_size],
                dtype=torch.float32,
                device=device,
            )
            with _autocast_context(device, amp_enabled, amp_dtype):
                preds.append(
                    net(normalise_pts(pts_t, aabb_min_t, aabb_max_t))
                    .float()
                    .cpu()
                    .numpy()
                    .flatten()
                )

    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)


def _sign_accuracy_from_sdf(pred_sdf: np.ndarray, target_sdf: np.ndarray) -> float:
    """Compute inside/outside sign accuracy, ignoring exact zeros in GT."""
    nonzero = target_sdf != 0.0
    return float(
        ((pred_sdf[nonzero] < 0) == (target_sdf[nonzero] < 0)).mean()
    ) if nonzero.any() else float("nan")


def _confident_sign_metrics(
    pred_sdf: np.ndarray,
    target_sdf: np.ndarray,
    decision_margin: float,
) -> tuple[float, float]:
    """
    Evaluate sign accuracy only on points whose predicted magnitude exceeds a
    safety margin. Returns (accuracy, coverage).
    """
    if decision_margin <= 0.0:
        return _sign_accuracy_from_sdf(pred_sdf, target_sdf), 1.0

    confident_mask = np.abs(pred_sdf) > decision_margin
    coverage = float(confident_mask.mean())
    if not confident_mask.any():
        return float("nan"), coverage
    return (
        _sign_accuracy_from_sdf(pred_sdf[confident_mask], target_sdf[confident_mask]),
        coverage,
    )


def _recommend_decision_margin(
    pred_sdf: np.ndarray,
    target_sdf: np.ndarray,
    default_margin: float,
    target_confident_accuracy: float = 0.95,
) -> dict[str, float]:
    """
    Pick a practical boundary-band half-width tau for three-way
    inside/outside/boundary queries.

    We sweep candidate margins over |pred_sdf|. If any candidate achieves the
    requested confident-sign accuracy, we keep the one with the highest
    coverage. Otherwise we fall back to a balanced score that still prefers
    good coverage.
    """
    abs_pred = np.abs(np.asarray(pred_sdf, dtype=np.float32).flatten())
    target_sdf = np.asarray(target_sdf, dtype=np.float32).flatten()
    if abs_pred.size == 0:
        return {
            "decision_margin": float(default_margin),
            "confident_accuracy": float("nan"),
            "confident_coverage": 0.0,
            "boundary_fraction": 1.0,
        }

    quantiles = np.linspace(0.0, 0.98, 99, dtype=np.float32)
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, float(default_margin)], dtype=np.float32),
                np.quantile(abs_pred, quantiles).astype(np.float32),
            ]
        )
    )

    best_meeting_target: tuple[float, float, float] | None = None
    best_fallback: tuple[float, float, float, float] | None = None

    for margin in candidates:
        acc, cov = _confident_sign_metrics(pred_sdf, target_sdf, float(margin))
        if not np.isfinite(acc):
            continue

        if acc >= target_confident_accuracy:
            candidate = (cov, -float(margin), acc)
            if best_meeting_target is None or candidate > best_meeting_target:
                best_meeting_target = candidate

        score = float(acc) - 0.25 * (1.0 - float(cov))
        fallback = (score, cov, -float(margin), acc)
        if best_fallback is None or fallback > best_fallback:
            best_fallback = fallback

    if best_meeting_target is not None:
        cov, neg_margin, acc = best_meeting_target
        margin = -neg_margin
    elif best_fallback is not None:
        _, cov, neg_margin, acc = best_fallback
        margin = -neg_margin
    else:
        margin = float(default_margin)
        acc, cov = _confident_sign_metrics(pred_sdf, target_sdf, margin)

    return {
        "decision_margin": float(margin),
        "confident_accuracy": float(acc),
        "confident_coverage": float(cov),
        "boundary_fraction": float(1.0 - cov),
    }


def classify_sdf_values(
    sdf_values: np.ndarray,
    decision_margin: float,
) -> np.ndarray:
    """
    Convert predicted SDF values into {"inside", "outside", "boundary"}.
    """
    sdf_values = np.asarray(sdf_values, dtype=np.float32)
    labels = np.full(sdf_values.shape, "boundary", dtype=object)
    labels[sdf_values < -decision_margin] = "inside"
    labels[sdf_values > decision_margin] = "outside"
    return labels


def load_trained_sdf(
    stl_file: str | None = None,
    checkpoint_path: str | None = None,
    device: str = "cuda",
) -> tuple[SDFNet, dict]:
    """
    Load a trained SDF checkpoint and return ``(network, checkpoint_dict)``.
    Provide either ``stl_file`` or ``checkpoint_path``.
    """
    if checkpoint_path is None:
        if stl_file is None:
            raise ValueError("Either stl_file or checkpoint_path must be provided.")
        abs_stl = os.path.abspath(stl_file)
        stl_dir = os.path.dirname(abs_stl)
        stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
        checkpoint_path = os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = SDFNet(
        fourier_L=checkpoint["fourier_L"],
        hidden_dim=checkpoint["hidden_dim"],
        n_hidden=checkpoint["n_hidden"],
    ).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint


def predict_sdf_points(
    net: SDFNet,
    pts_np: np.ndarray,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    device: str = "cuda",
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Public wrapper for batched SDF evaluation on numpy points."""
    return _predict_sdf_numpy(
        net=net,
        pts_np=np.asarray(pts_np, dtype=np.float32),
        aabb_min_np=np.asarray(aabb_min_np, dtype=np.float32),
        aabb_max_np=np.asarray(aabb_max_np, dtype=np.float32),
        device=device,
        chunk_size=chunk_size,
    )


def query_point_labels(
    net: SDFNet,
    pts_np: np.ndarray,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    device: str = "cuda",
    decision_margin: float = 0.0,
    chunk_size: int = 65_536,
) -> dict[str, np.ndarray | float]:
    """
    Fast three-way geometric query using the trained neural SDF.

    Returns a dict containing:
        sdf_values      – raw predicted signed distances
        labels          – {"inside", "outside", "boundary"} for each point
        decision_margin – the boundary-band half-width used for classification
    """
    sdf_values = predict_sdf_points(
        net=net,
        pts_np=pts_np,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        device=device,
        chunk_size=chunk_size,
    )
    labels = classify_sdf_values(sdf_values, decision_margin=float(decision_margin))
    return {
        "sdf_values": sdf_values,
        "labels": labels,
        "decision_margin": float(decision_margin),
    }


def _apply_training_profile(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply higher-level training presets.

    The profile only overrides knobs that are most tied to the intended
    downstream use. Architecture and epoch count are left to the caller.
    """
    profile = str(getattr(args, "optimize_for", "balanced")).strip().lower()
    if profile == "balanced":
        return args

    if profile == "geometry_query":
        args.n_surf = 150_000
        args.n_vol = 300_000
        args.refresh_fraction = 0.20
        args.refresh_every = 10
        args.val_every = 10
        args.near_surface_weight = 2.0
        args.lambda_sign = 0.03
        args.sign_margin_ratio = 0.01
        args.sign_near_surface_weight = 0.50
        args.sign_conflict_band_ratio = 0.05
        args.sign_start_epoch = 1
        args.sign_ramp_epochs = 20
        args.early_stopping_metric = "hybrid"
        args.decision_margin_ratio = 0.005
        args.balance_volume_signs = True
        args.validation_balance_volume_signs = True
        return args

    if profile == "inclusion":
        args.n_surf = 50_000
        args.n_vol = 500_000
        args.refresh_fraction = 0.20
        args.refresh_every = 10
        args.val_every = 10
        args.near_surface_weight = 1.0
        args.lambda_sign = 0.05
        args.sign_margin_ratio = 0.015
        args.sign_near_surface_weight = 0.25
        args.sign_conflict_band_ratio = 0.05
        args.sign_start_epoch = 1
        args.sign_ramp_epochs = 15
        args.early_stopping_metric = "volume_sign_error"
        args.decision_margin_ratio = 0.0075
        args.balance_volume_signs = True
        args.validation_balance_volume_signs = True
        return args

    raise ValueError(f"Unknown optimize_for profile: {profile}")


def _resolve_epoch_sign_weight(
    epoch: int,
    lambda_sign: float,
    sign_start_epoch: int,
    sign_ramp_epochs: int,
) -> float:
    """Warm up sign supervision so early training focuses on SDF regression."""
    if lambda_sign <= 0.0:
        return 0.0
    if epoch < sign_start_epoch:
        return 0.0
    if sign_ramp_epochs <= 0:
        return lambda_sign

    progress = min(max(epoch - sign_start_epoch + 1, 0) / sign_ramp_epochs, 1.0)
    return lambda_sign * progress


def _compute_validation_metric(
    val_metrics: dict,
    metric_name: str,
    near_surface_band: float,
) -> float:
    """
    Convert a validation summary into a scalar early-stopping objective.
    Lower is better.
    """
    near_surface_mae = val_metrics["near_surface_mae"]
    global_mae = val_metrics["mae"]
    volume_sign_acc = val_metrics["volume_sign_accuracy"]

    if metric_name == "near_surface_mae":
        return near_surface_mae if np.isfinite(near_surface_mae) else global_mae
    if metric_name == "mae":
        return global_mae
    if metric_name == "volume_sign_error":
        return 1.0 - volume_sign_acc if np.isfinite(volume_sign_acc) else float("inf")
    if metric_name == "hybrid":
        near_term = near_surface_mae if np.isfinite(near_surface_mae) else global_mae
        volume_sign_err = 1.0 - volume_sign_acc if np.isfinite(volume_sign_acc) else 1.0
        return near_term + near_surface_band * volume_sign_err

    raise ValueError(f"Unknown early stopping metric: {metric_name}")


def weighted_sdf_mse_loss(
    pred_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    near_surface_band: float,
    near_surface_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted MSE that emphasises samples close to the zero level set.
    Returns (weighted_mse, mean_weight).
    """
    if near_surface_weight <= 1.0 or near_surface_band <= 0.0:
        unit_weight = torch.ones_like(target_sdf)
        return nn.functional.mse_loss(pred_sdf, target_sdf), unit_weight.mean()

    weights = torch.ones_like(target_sdf)
    near_mask = target_sdf.abs() < near_surface_band
    weights = torch.where(
        near_mask,
        torch.full_like(weights, float(near_surface_weight)),
        weights,
    )
    sq_err = (pred_sdf - target_sdf) ** 2
    weighted_mse = (weights * sq_err).sum() / weights.sum().clamp_min(1e-8)
    return weighted_mse, weights.mean()


def sdf_sign_loss(
    pred_sdf: torch.Tensor,
    target_sdf: torch.Tensor,
    sign_margin: float,
    conflict_band: float,
    near_surface_band: float,
    sign_near_surface_weight: float,
    balance_classes: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encourage the predicted SDF sign to match the ground truth sign with a
    small margin around zero. Near-surface samples can be up-weighted further
    because they are the most important for the zero level set.
    Returns (loss, sign_accuracy_in_batch).
    """
    target_sign = torch.where(target_sdf < 0, -1.0, 1.0)
    margin_term = sign_margin - target_sign * pred_sdf
    per_point_loss = torch.nn.functional.softplus(margin_term)

    weights = torch.ones_like(per_point_loss)
    if conflict_band > 0.0:
        # Very close to the zero level-set, exact SDF regression and a fixed
        # sign margin fight each other. Mask this band out from sign loss.
        conflict_mask = target_sdf.abs() < conflict_band
        weights = torch.where(
            conflict_mask,
            torch.zeros_like(weights),
            weights,
        )
    if sign_near_surface_weight != 1.0 and near_surface_band > 0.0:
        near_mask = target_sdf.abs() < near_surface_band
        weights = torch.where(
            near_mask,
            weights * float(sign_near_surface_weight),
            weights,
        )
    if balance_classes:
        inside_mask = target_sdf < 0
        outside_mask = ~inside_mask
        n_inside = inside_mask.sum().item()
        n_outside = outside_mask.sum().item()
        if n_inside > 0 and n_outside > 0:
            total = n_inside + n_outside
            inside_w = total / (2.0 * n_inside)
            outside_w = total / (2.0 * n_outside)
            class_weights = torch.where(
                inside_mask,
                torch.full_like(weights, float(inside_w)),
                torch.full_like(weights, float(outside_w)),
            )
            weights = weights * class_weights

    loss = (weights * per_point_loss).sum() / weights.sum().clamp_min(1e-8)

    batch_sign_acc = ((pred_sdf < 0) == (target_sdf < 0)).float().mean()
    return loss, batch_sign_acc


# ─────────────────────────── Training ────────────────────────────────────────

def train(
    stl_file: str,
    epochs: int = 800,
    batch_size: int = 0,
    lr: float = 1e-4,
    lambda_eikonal: float = 0.02,
    fourier_L: int = 6,
    hidden_dim: int = 256,
    n_hidden: int = 4,
    device: str = "cuda",
    n_surf: int = 80_000,
    n_vol: int = 40_000,
    refresh_fraction: float = 0.50,
    refresh_every: int = 2,
    log_every: int = 1,
    val_every: int = 20,
    val_samples: int = 20_000,
    early_stopping_patience: int = 24,
    early_stopping_min_delta: float = 1e-4,
    near_surface_band_ratio: float = 0.05,
    near_surface_weight: float = 2.0,
    lambda_sign: float = 0.05,
    sign_margin_ratio: float = 0.01,
    sign_conflict_band_ratio: float | None = None,
    sign_near_surface_weight: float = 1.0,
    sign_start_epoch: int = 1,
    sign_ramp_epochs: int = 25,
    early_stopping_metric: str = "volume_sign_error",
    decision_margin_ratio: float = 0.005,
    balance_volume_signs: bool = True,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
    validation_balance_volume_signs: bool | None = None,
) -> str:
    """
    Train SDFNet for the given STL and save the checkpoint.
    Returns the path of the saved ``.pt`` file.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    _configure_torch_runtime(device)

    # ── Derive save path ──────────────────────────────────────────────────────
    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    save_path = os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    print(f"\n{'='*60}")
    print(f"Training SDF network for : {stl_file}")
    print(f"Checkpoint will be saved  : {save_path}")
    print(f"Device: {device}  |  Epochs: {epochs}  |  Requested batch: {batch_size:,}")
    print("=" * 60)

    # ── Load and normalise mesh ───────────────────────────────────────────────
    print("\n[1/4] Loading and normalising mesh...")
    mesh, aabb_min_np, aabb_max_np = _load_normalised_mesh(stl_file)
    aabb_min_t = torch.tensor(aabb_min_np, dtype=torch.float32, device=device)
    aabb_max_t = torch.tensor(aabb_max_np, dtype=torch.float32, device=device)
    print(f"  AABB min: {aabb_min_np}  max: {aabb_max_np}")
    print("  Training space: normalised mesh space  |  network input: AABB -> [-1, 1]")
    print("  SDF target units: normalised length units")

    # ── Generate training dataset ─────────────────────────────────────────────
    print("\n[2/4] Generating initial training dataset...")
    pts_np, sdf_np = generate_dataset(
        mesh,
        aabb_min_np,
        aabb_max_np,
        n_surf,
        n_vol,
        balance_volume_signs=balance_volume_signs,
        volume_balance_ratio=volume_balance_ratio,
        volume_balance_oversample=volume_balance_oversample,
    )
    pts_t, sdf_t = _to_device_tensors(pts_np, sdf_np, device=device)
    N = len(pts_t)
    refresh_fraction = float(np.clip(refresh_fraction, 0.0, 1.0))
    refresh_every = max(int(refresh_every), 1)
    log_every = max(int(log_every), 1)
    val_every = max(int(val_every), 1)
    val_samples = max(int(val_samples), 2)
    early_stopping_patience = max(int(early_stopping_patience), 1)
    early_stopping_min_delta = max(float(early_stopping_min_delta), 0.0)
    near_surface_band_ratio = max(float(near_surface_band_ratio), 0.0)
    near_surface_weight = max(float(near_surface_weight), 1.0)
    lambda_sign = max(float(lambda_sign), 0.0)
    sign_margin_ratio = max(float(sign_margin_ratio), 0.0)
    sign_near_surface_weight = float(sign_near_surface_weight)
    sign_start_epoch = max(int(sign_start_epoch), 1)
    sign_ramp_epochs = max(int(sign_ramp_epochs), 0)
    decision_margin_ratio = max(float(decision_margin_ratio), 0.0)
    volume_balance_ratio = float(np.clip(volume_balance_ratio, 0.0, 1.0))
    volume_balance_oversample = max(int(volume_balance_oversample), 2)
    early_stopping_metric = str(early_stopping_metric).strip().lower()
    if early_stopping_metric not in {"hybrid", "near_surface_mae", "mae", "volume_sign_error"}:
        raise ValueError(
            "early_stopping_metric must be one of: "
            "'hybrid', 'near_surface_mae', 'mae', 'volume_sign_error'"
        )
    if validation_balance_volume_signs is None:
        validation_balance_volume_signs = balance_volume_signs
    diag = float(np.linalg.norm(aabb_max_np - aabb_min_np))
    near_surface_band = near_surface_band_ratio * diag
    sign_margin = sign_margin_ratio * diag
    if sign_conflict_band_ratio is None:
        sign_conflict_band_ratio = near_surface_band_ratio
    sign_conflict_band_ratio = max(float(sign_conflict_band_ratio), 0.0)
    sign_conflict_band = sign_conflict_band_ratio * diag
    decision_margin = decision_margin_ratio * diag
    print(f"  {_summarize_sdf_samples(sdf_np)}")
    if refresh_fraction > 0.0:
        n_refresh = int(round(N * refresh_fraction))
        print(
            f"  Dynamic refresh enabled: replace ~{n_refresh:,} samples "
            f"every {refresh_every} epoch(s)."
        )
    else:
        print("  Dynamic refresh disabled: training pool remains fixed.")
    if balance_volume_signs and n_vol > 0:
        print(
            f"  Volume sign balancing: enabled "
            f"(target inside={100 * volume_balance_ratio:.1f}% of volume samples)"
        )
    else:
        print("  Volume sign balancing: disabled")
    print(
        f"  Logging every {log_every} epoch(s)  |  Validation every {val_every} epoch(s) "
        f"on {val_samples:,} fixed points"
    )
    print(
        f"  Early stopping patience={early_stopping_patience}  "
        f"min_delta={early_stopping_min_delta:.1e}"
    )
    if near_surface_weight > 1.0 and near_surface_band > 0.0:
        print(
            f"  Near-surface loss weighting: x{near_surface_weight:.2f} "
            f"for |SDF| < {near_surface_band:.4f} ({100 * near_surface_band_ratio:.1f}% diag)"
        )
    else:
        print("  Near-surface loss weighting: disabled")
    if lambda_sign > 0.0:
        print(
            f"  Sign supervision: lambda={lambda_sign:.3f}  "
            f"margin={sign_margin:.4f} ({100 * sign_margin_ratio:.1f}% diag)"
        )
        print(
            f"  Sign schedule: start@epoch {sign_start_epoch}  "
            f"ramp={sign_ramp_epochs} epoch(s)  "
            f"conflict-mask |SDF|<{sign_conflict_band:.4f}  "
            f"sign-near-w={sign_near_surface_weight:.2f}"
        )
    else:
        print("  Sign supervision: disabled")
    val_pts_np, val_sdf_gt = sample_validation_dataset(
        mesh=mesh,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        n_eval=val_samples,
        volume_pad_ratio=0.10,
        balance_volume_signs=validation_balance_volume_signs,
        volume_balance_ratio=volume_balance_ratio,
        volume_balance_oversample=volume_balance_oversample,
    )
    vol_val_pts_np, vol_val_sdf_gt = sample_volume_validation_dataset(
        mesh=mesh,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        n_eval=val_samples,
        volume_pad_ratio=0.10,
        balance_volume_signs=True,
        volume_balance_ratio=volume_balance_ratio,
        volume_balance_oversample=volume_balance_oversample,
    )
    print(
        f"  Fixed validation sets prepared: mixed={len(val_pts_np):,}  "
        f"volume-only={len(vol_val_pts_np):,}  "
        f"decision margin={decision_margin:.4f} ({100 * decision_margin_ratio:.1f}% diag)"
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[3/4] Building SDFNet...")
    net = SDFNet(fourier_L=fourier_L, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Architecture: Fourier(L={fourier_L}) → {n_hidden}×Linear({hidden_dim}) → 1")
    print(f"  Total parameters: {n_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    amp_enabled, amp_dtype, amp_name, use_grad_scaler = _resolve_amp_settings(device)
    scaler = (
        torch.amp.GradScaler(device="cuda", enabled=True)
        if use_grad_scaler else None
    )
    if batch_size <= 0:
        if _is_cuda_device(device):
            print("  Auto-tuning batch size for current GPU...")
            batch_size = _autotune_batch_size(
                net=net,
                optimizer=optimizer,
                aabb_min_t=aabb_min_t,
                aabb_max_t=aabb_max_t,
                lr=lr,
                lambda_eikonal=lambda_eikonal,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        else:
            batch_size = 8192
    batch_size = max(int(batch_size), 1)
    print(f"  AMP: {amp_name}  |  Effective batch size: {batch_size:,}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy(net.state_dict())

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Training for {epochs} epochs...")
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        epoch_lambda_sign = _resolve_epoch_sign_weight(
            epoch=epoch,
            lambda_sign=lambda_sign,
            sign_start_epoch=sign_start_epoch,
            sign_ramp_epochs=sign_ramp_epochs,
        )
        if epoch > 1 and refresh_fraction > 0.0 and (epoch - 1) % refresh_every == 0:
            pts_t, sdf_t, n_refreshed = refresh_training_pool(
                mesh=mesh,
                aabb_min=aabb_min_np,
                aabb_max=aabb_max_np,
                pts_t=pts_t,
                sdf_t=sdf_t,
                device=device,
                refresh_fraction=refresh_fraction,
                n_surf=n_surf,
                n_vol=n_vol,
                balance_volume_signs=balance_volume_signs,
                volume_balance_ratio=volume_balance_ratio,
                volume_balance_oversample=volume_balance_oversample,
            )
            tqdm.write(
                f"  Refreshed {n_refreshed:,} / {N:,} training samples "
                f"before epoch {epoch}."
            )

        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_reg_mse = 0.0
        epoch_sign = 0.0
        epoch_eik = 0.0
        epoch_weight_mean = 0.0
        epoch_batch_sign_acc = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            pts_batch = pts_t[idx]
            sdf_batch = sdf_t[idx]

            # Require grad on input for Eikonal term
            optimizer.zero_grad(set_to_none=True)
            pts_in = pts_batch.detach().requires_grad_(True)
            with _autocast_context(device, amp_enabled, amp_dtype):
                x_norm = normalise_pts(pts_in, aabb_min_t, aabb_max_t)
                pred = net(x_norm)

            pred_fp32 = pred.float()
            mse, mean_weight = weighted_sdf_mse_loss(
                pred_sdf=pred_fp32,
                target_sdf=sdf_batch,
                near_surface_band=near_surface_band,
                near_surface_weight=near_surface_weight,
            )
            reg_mse = nn.functional.mse_loss(pred_fp32, sdf_batch)
            if epoch_lambda_sign > 0.0:
                sign_loss, batch_sign_acc = sdf_sign_loss(
                    pred_sdf=pred_fp32,
                    target_sdf=sdf_batch,
                    sign_margin=sign_margin,
                    conflict_band=sign_conflict_band,
                    near_surface_band=near_surface_band,
                    sign_near_surface_weight=sign_near_surface_weight,
                )
            else:
                sign_loss = pred_fp32.new_zeros(())
                batch_sign_acc = ((pred_fp32 < 0) == (sdf_batch < 0)).float().mean()

            # Eikonal regularisation: ||∇SDF|| should be ≈ 1
            grads = torch.autograd.grad(
                pred_fp32.sum(), pts_in, create_graph=True
            )[0]
            eikonal = ((grads.float().norm(dim=-1) - 1.0) ** 2).mean()

            loss = mse + epoch_lambda_sign * sign_loss + lambda_eikonal * eikonal
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_reg_mse += reg_mse.item()
            epoch_sign += sign_loss.item()
            epoch_eik += eikonal.item()
            epoch_weight_mean += mean_weight.item()
            epoch_batch_sign_acc += batch_sign_acc.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_reg_mse = epoch_reg_mse / max(n_batches, 1)
        avg_sign = epoch_sign / max(n_batches, 1)
        avg_eik = epoch_eik / max(n_batches, 1)
        avg_weight_mean = epoch_weight_mean / max(n_batches, 1)
        avg_batch_sign_acc = epoch_batch_sign_acc / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        ran_validation = (epoch == 1) or (epoch % val_every == 0) or (epoch == epochs)
        val_metric = None
        if ran_validation:
            val_metrics = evaluate_sdf_accuracy(
                net,
                mesh,
                aabb_min_np,
                aabb_max_np,
                device=device,
                n_eval=val_samples,
                pts_eval=val_pts_np,
                sdf_gt=val_sdf_gt,
                volume_pts_eval=vol_val_pts_np,
                volume_sdf_gt=vol_val_sdf_gt,
                near_surface_band_ratio=near_surface_band_ratio,
                decision_margin=decision_margin,
                verbose=False,
            )
            val_metric = _compute_validation_metric(
                val_metrics=val_metrics,
                metric_name=early_stopping_metric,
                near_surface_band=near_surface_band,
            )

            improved = val_metric < (best_metric - early_stopping_min_delta)
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = copy.deepcopy(net.state_dict())
            else:
                epochs_without_improvement += 1

        if (epoch % log_every == 0) or (epoch == 1) or (epoch == epochs):
            msg = (
                f"  Epoch {epoch:4d}/{epochs}"
                f"  Loss={avg_loss:.6f}"
                f"  MSE={avg_mse:.6f}"
                f"  BaseMSE={avg_reg_mse:.6f}"
                f"  SignLoss={avg_sign:.6f}"
                f"  SignW={epoch_lambda_sign:.3f}"
                f"  Eikonal={avg_eik:.6f}"
                f"  TrainSign={100 * avg_batch_sign_acc:.2f}%"
                f"  MeanW={avg_weight_mean:.2f}"
                f"  LR={lr_now:.2e}"
            )
            if val_metric is not None:
                msg += (
                    f"  ValMAE={val_metrics['mae']:.6f}"
                    f"  ValGlobalRMSE={val_metrics['rmse']:.6f}"
                    f"  ValNearMAE={val_metrics['near_surface_mae']:.6f}"
                    f"  ValNearRMSE={val_metrics['near_surface_rmse']:.6f}"
                    f"  ValSign={100 * val_metrics['sign_accuracy']:.2f}%"
                    f"  ValVolSign={100 * val_metrics['volume_sign_accuracy']:.2f}%"
                    f"  ValConfSign={100 * val_metrics['volume_confident_sign_accuracy']:.2f}%"
                    f"  ConfCov={100 * val_metrics['volume_confident_coverage']:.2f}%"
                    f"  ValMetric={val_metric:.6f}"
                    f"  Best={best_metric:.6f}@{best_epoch}"
                    f"  Patience={epochs_without_improvement}/{early_stopping_patience}"
                )
            tqdm.write(
                msg
            )

        if ran_validation and epochs_without_improvement >= early_stopping_patience:
            tqdm.write(
                f"  Early stopping triggered at epoch {epoch}. "
                f"Best validation metric={best_metric:.6f} from epoch {best_epoch}."
            )
            break

    # ── Post-training accuracy evaluation ────────────────────────────────────
    if best_epoch > 0:
        print(f"\n  Restoring best model from epoch {best_epoch} (metric={best_metric:.6f})...")
        net.load_state_dict(best_state_dict)

    net.eval()
    final_metrics = evaluate_sdf_accuracy(
        net,
        mesh,
        aabb_min_np,
        aabb_max_np,
        device=device,
        near_surface_band_ratio=near_surface_band_ratio,
        decision_margin=decision_margin,
    )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    checkpoint = {
        "state_dict": net.state_dict(),
        "aabb_min": aabb_min_np,
        "aabb_max": aabb_max_np,
        "fourier_L": fourier_L,
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_metric_name": early_stopping_metric,
        "refresh_fraction": refresh_fraction,
        "refresh_every": refresh_every,
        "near_surface_band_ratio": near_surface_band_ratio,
        "near_surface_weight": near_surface_weight,
        "lambda_sign": lambda_sign,
        "sign_margin_ratio": sign_margin_ratio,
        "sign_conflict_band_ratio": sign_conflict_band_ratio,
        "sign_near_surface_weight": sign_near_surface_weight,
        "sign_start_epoch": sign_start_epoch,
        "sign_ramp_epochs": sign_ramp_epochs,
        "decision_margin_ratio": decision_margin_ratio,
        "recommended_decision_margin": final_metrics["recommended_decision_margin"],
        "recommended_decision_margin_ratio": (
            final_metrics["recommended_decision_margin"] / max(diag, 1e-8)
        ),
        "recommended_confident_sign_accuracy": final_metrics["recommended_confident_sign_accuracy"],
        "recommended_confident_coverage": final_metrics["recommended_confident_coverage"],
        "recommended_boundary_fraction": final_metrics["recommended_boundary_fraction"],
        "model_space": "normalised_mesh_space",
        "input_normalization": "aabb_to_minus1_1",
        "sdf_units": "normalised_length",
    }
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint saved to: {save_path}")
    return save_path


# ─────────────────────────── Accuracy Evaluation ─────────────────────────────

def evaluate_sdf_accuracy(
    net: SDFNet,
    mesh: trimesh.Trimesh,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    device: str = "cuda",
    n_eval: int = 50_000,
    pts_eval: np.ndarray | None = None,
    sdf_gt: np.ndarray | None = None,
    volume_pts_eval: np.ndarray | None = None,
    volume_sdf_gt: np.ndarray | None = None,
    volume_pad_ratio: float = 0.10,
    balance_volume_signs: bool = False,
    volume_balance_ratio: float = 0.5,
    volume_balance_oversample: int = 6,
    near_surface_band_ratio: float = 0.05,
    decision_margin: float = 0.0,
    calibrate_decision_margin: bool = True,
    target_confident_accuracy: float = 0.95,
    verbose: bool = True,
) -> dict:
    """
    Evaluate trained SDF accuracy on a *fresh* set of points (not used during
    training) and print a summary report.

    Metrics reported
    ----------------
    Global MAE / RMSE
        Average absolute / root-mean-square SDF error over the whole AABB.
    Near-surface MAE / RMSE
        Same metrics restricted to points with |SDF_gt| < 5 % of the diagonal.
        This is the most important region for inside/outside classification.
    Inside/outside sign accuracy
        Percentage of points where the predicted sign matches the GT sign
        (i.e., correctly classified as inside or outside).

    Units
    -----
    All distances are in *normalised space* where the longest mesh dimension
    equals 1.0. Multiply by the original mesh size to get physical units.

    Returns
    -------
    dict with keys: mae, rmse, near_surface_mae, near_surface_rmse,
                    sign_accuracy, volume_sign_accuracy,
                    volume_confident_sign_accuracy, volume_confident_coverage,
                    n_near_surface.
    """
    using_fixed_set = pts_eval is not None
    if verbose:
        label = "fixed validation set" if using_fixed_set else "fresh samples"
        print(f"\n  Running accuracy evaluation on {label}...")
    diag = float(np.linalg.norm(aabb_max_np - aabb_min_np))
    if pts_eval is None:
        pts_eval, sdf_gt = sample_validation_dataset(
            mesh=mesh,
            aabb_min_np=aabb_min_np,
            aabb_max_np=aabb_max_np,
            n_eval=n_eval,
            volume_pad_ratio=volume_pad_ratio,
            balance_volume_signs=balance_volume_signs,
            volume_balance_ratio=volume_balance_ratio,
            volume_balance_oversample=volume_balance_oversample,
        )
    elif sdf_gt is None:
        sdf_gt = compute_sdf_gt(mesh, pts_eval).flatten()
    else:
        sdf_gt = np.asarray(sdf_gt, dtype=np.float32).flatten()

    if volume_pts_eval is None:
        volume_pts_eval, volume_sdf_gt = sample_volume_validation_dataset(
            mesh=mesh,
            aabb_min_np=aabb_min_np,
            aabb_max_np=aabb_max_np,
            n_eval=n_eval,
            volume_pad_ratio=volume_pad_ratio,
            balance_volume_signs=True,
            volume_balance_ratio=volume_balance_ratio,
            volume_balance_oversample=volume_balance_oversample,
        )
    elif volume_sdf_gt is None:
        volume_sdf_gt = compute_sdf_gt(mesh, volume_pts_eval).flatten()
    else:
        volume_sdf_gt = np.asarray(volume_sdf_gt, dtype=np.float32).flatten()

    sdf_pred = _predict_sdf_numpy(
        net=net,
        pts_np=pts_eval,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        device=device,
    )
    volume_sdf_pred = _predict_sdf_numpy(
        net=net,
        pts_np=volume_pts_eval,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        device=device,
    )

    err = sdf_pred - sdf_gt
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    near_mask = np.abs(sdf_gt) < near_surface_band_ratio * diag
    n_near = int(near_mask.sum())
    if n_near > 0:
        near_err = sdf_pred[near_mask] - sdf_gt[near_mask]
        near_mae  = float(np.mean(np.abs(near_err)))
        near_rmse = float(np.sqrt(np.mean(near_err ** 2)))
    else:
        near_mae = near_rmse = float('nan')

    sign_acc = _sign_accuracy_from_sdf(sdf_pred, sdf_gt)
    volume_sign_acc = _sign_accuracy_from_sdf(volume_sdf_pred, volume_sdf_gt)
    volume_conf_acc, volume_conf_cov = _confident_sign_metrics(
        pred_sdf=volume_sdf_pred,
        target_sdf=volume_sdf_gt,
        decision_margin=decision_margin,
    )
    recommended_margin = float(decision_margin)
    recommended_conf_acc = float(volume_conf_acc)
    recommended_conf_cov = float(volume_conf_cov)
    recommended_boundary_fraction = float(1.0 - volume_conf_cov)
    if calibrate_decision_margin:
        margin_stats = _recommend_decision_margin(
            pred_sdf=volume_sdf_pred,
            target_sdf=volume_sdf_gt,
            default_margin=decision_margin,
            target_confident_accuracy=target_confident_accuracy,
        )
        recommended_margin = float(margin_stats["decision_margin"])
        recommended_conf_acc = float(margin_stats["confident_accuracy"])
        recommended_conf_cov = float(margin_stats["confident_coverage"])
        recommended_boundary_fraction = float(margin_stats["boundary_fraction"])

    if verbose:
        print(f"\n{'='*58}")
        report_n = len(pts_eval)
        source_label = "fixed" if using_fixed_set else "fresh"
        print(f"  SDF Accuracy Report  ({source_label} {report_n:,} validation points)")
        print(f"  {'─'*54}")
        print(f"  Global  MAE  : {mae:.6f}   RMSE : {rmse:.6f}")
        print(f"  Near-surf MAE: {near_mae:.6f}   RMSE : {near_rmse:.6f}"
              f"  (n={n_near:,}, |SDF|<{100 * near_surface_band_ratio:.1f}% diag)")
        print(f"  Sign accuracy: {100*sign_acc:.2f}%  (inside/outside classification)")
        print(f"  Volume sign  : {100*volume_sign_acc:.2f}%  (balanced volume-only set)")
        print(
            f"  Confident vol: {100*volume_conf_acc:.2f}%  "
            f"(coverage={100*volume_conf_cov:.2f}%, |pred|>{decision_margin:.4f})"
        )
        print(
            f"  Recommended tau: {recommended_margin:.6f}  "
            f"(conf={100*recommended_conf_acc:.2f}%, "
            f"coverage={100*recommended_conf_cov:.2f}%, "
            f"boundary={100*recommended_boundary_fraction:.2f}%)"
        )
        print(f"  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
        print(f"  Distances are in normalised space  (longest_dim = 1.0)")
        print(f"  Diagonal of AABB = {diag:.4f}  (normalised units)")
        print('=' * 58 + '\n')

    return {
        'mae': mae,
        'rmse': rmse,
        'near_surface_mae': near_mae,
        'near_surface_rmse': near_rmse,
        'sign_accuracy': sign_acc,
        'volume_sign_accuracy': volume_sign_acc,
        'volume_confident_sign_accuracy': volume_conf_acc,
        'volume_confident_coverage': volume_conf_cov,
        'recommended_decision_margin': recommended_margin,
        'recommended_confident_sign_accuracy': recommended_conf_acc,
        'recommended_confident_coverage': recommended_conf_cov,
        'recommended_boundary_fraction': recommended_boundary_fraction,
        'n_near_surface': n_near,
    }


# ─────────────────────────── Visualisation ───────────────────────────────────

def visualise_sdf(
    stl_file: str,
    device: str = "cuda",
    grid_resolution: int = 128,
    save_figures: bool = True,
) -> None:
    """
    Load a saved SDF checkpoint and visualise it two ways.

    1. **2-D slice plots** (matplotlib, saved as PNG + shown interactively).
       Three orthogonal mid-plane slices are colour-mapped with the SDF value.
       The zero-level contour (= model surface) is drawn in black.

    2. **3-D isosurface comparison** (trimesh viewer, requires scikit-image).
       Marching cubes extracts the zero-level-set from the SDF volume.
       The reconstructed surface (red) is shown overlaid on the original STL
       (blue) so you can visually judge shape accuracy.
       A surface-proximity report is also printed (pseudo-Hausdorff distance).

    Args:
        stl_file:        Path to the original STL (used to derive checkpoint
                         path and to load the reference mesh for comparison).
        device:          PyTorch device string.
        grid_resolution: Side length of the evaluation grid (default 128).
                         Use 64 for a fast preview, 256 for high fidelity.
        save_figures:    Save 2-D PNG next to the checkpoint file.
    """
    import matplotlib.pyplot as plt
    _configure_torch_runtime(device)
    amp_enabled, amp_dtype, _, _ = _resolve_amp_settings(device)

    # ── Locate and load checkpoint ────────────────────────────────────────────
    abs_stl   = os.path.abspath(stl_file)
    stl_dir   = os.path.dirname(abs_stl)
    stl_name  = os.path.splitext(os.path.basename(abs_stl))[0]
    ckpt_path = os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first:  conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf"
            f" --stl {stl_file}"
        )

    print(f"\n{'='*60}")
    print(f"Visualising: {ckpt_path}")
    print('=' * 60)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = SDFNet(
        fourier_L=ckpt['fourier_L'],
        hidden_dim=ckpt['hidden_dim'],
        n_hidden=ckpt['n_hidden'],
    ).to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()

    aabb_min = ckpt['aabb_min'].astype(np.float32)
    aabb_max = ckpt['aabb_max'].astype(np.float32)
    aabb_min_t = torch.tensor(aabb_min, dtype=torch.float32, device=device)
    aabb_max_t = torch.tensor(aabb_max, dtype=torch.float32, device=device)

    # ── Evaluate SDF on a dense 3-D grid ─────────────────────────────────────
    # Extend 10% beyond AABB so outside-mesh (SDF > 0) points are included.
    # Without padding, a solid model fills its own bounding box and all grid
    # points are inside → no zero-crossing for marching cubes.
    R = grid_resolution
    vis_pad = 0.10 * (aabb_max - aabb_min)
    vis_min = aabb_min - vis_pad
    vis_max = aabb_max + vis_pad
    print(f"\n[1/3] Evaluating SDF on {R}³ = {R**3:,} grid points (AABB ± 10%)...")
    xs = np.linspace(vis_min[0], vis_max[0], R, dtype=np.float32)
    ys = np.linspace(vis_min[1], vis_max[1], R, dtype=np.float32)
    zs = np.linspace(vis_min[2], vis_max[2], R, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    sdf_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(pts_grid), 50_000):
            p = torch.tensor(pts_grid[i : i + 50_000], dtype=torch.float32, device=device)
            with _autocast_context(device, amp_enabled, amp_dtype):
                sdf_chunks.append(
                    net(normalise_pts(p, aabb_min_t, aabb_max_t)).float().cpu().numpy()
                )
    sdf_vol = np.concatenate(sdf_chunks).reshape(R, R, R)
    print(f"  SDF range on grid: [{sdf_vol.min():.4f}, {sdf_vol.max():.4f}]")
    print(f"  Inside fraction  : {(sdf_vol < 0).mean() * 100:.1f}%")

    # ── 2-D slice colour maps ─────────────────────────────────────────────────
    print("\n[2/3] Generating 2-D slice plots...")
    mid = R // 2

    # sdf_vol indexing: [ix, iy, iz].  We build (data, x_coords, y_coords, title)
    # for each of three orthogonal planes so the axes are physically correct.
    slice_specs = [
        (sdf_vol[:, :, mid],   xs,  ys,  f"XY plane  (z = {zs[mid]:.3f})",  "x", "y"),
        (sdf_vol[:, mid, :],   xs,  zs,  f"XZ plane  (y = {ys[mid]:.3f})",  "x", "z"),
        (sdf_vol[mid, :, :],   ys,  zs,  f"YZ plane  (x = {xs[mid]:.3f})",  "y", "z"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Neural SDF  ·  {stl_name}  ·  grid {R}³\n"
        f"Black contour = zero level-set (model surface)   "
        f"|   blue = inside, red = outside",
        fontsize=11, fontweight='bold',
    )

    for ax, (sl, xc, yc, title, xlabel, ylabel) in zip(axes, slice_specs):
        vmax = float(np.percentile(np.abs(sl), 97))
        im = ax.imshow(
            sl.T,                  # transpose so x→cols, y→rows
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
            aspect='auto',
        )
        # Zero contour = model surface boundary
        ax.contour(xc, yc, sl.T, levels=[0.0], colors='k', linewidths=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{xlabel} (normalised)", fontsize=9)
        ax.set_ylabel(f"{ylabel} (normalised)", fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SDF', fontsize=8)

    plt.tight_layout()

    if save_figures:
        fig_path = os.path.join(stl_dir, f"{stl_name}_sdf_slices.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Saved:  {fig_path}")

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    # ── 3-D isosurface vs original STL ───────────────────────────────────────
    print("\n[3/3] Extracting zero-isosurface via marching cubes...")
    try:
        from skimage.measure import marching_cubes  # type: ignore
    except ImportError:
        print(
            "  scikit-image is not installed — skipping 3-D visualisation.\n"
            "  Install it with:  conda run -n myenv pip install scikit-image"
        )
        return

    if sdf_vol.min() >= 0.0 or sdf_vol.max() <= 0.0:
        print(
            f"  WARNING: SDF range [{sdf_vol.min():.4f}, {sdf_vol.max():.4f}] "
            f"does not cross 0 — cannot extract isosurface.\n"
            f"  This usually means the network has not converged yet.\n"
            f"  Try retraining with more epochs (--epochs 1000) or more samples "
            f"(--n_surf 300000 --n_vol 300000)."
        )
        return

    spacing = ((vis_max - vis_min) / (R - 1)).tolist()
    verts, faces, _, _ = marching_cubes(sdf_vol, level=0.0, spacing=spacing)
    verts = verts + vis_min           # shift into normalised world space

    recon_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    recon_mesh.fix_normals()
    print(f"  Reconstructed: {len(verts):,} vertices, {len(faces):,} faces")

    orig_mesh, _, _ = _load_normalised_mesh(stl_file)
    print(f"  Original STL : {len(orig_mesh.vertices):,} vertices, {len(orig_mesh.faces):,} faces")

    # ── Surface proximity report ──────────────────────────────────────────────
    n_samp = min(20_000, len(recon_mesh.vertices))
    samp_idx = np.random.choice(len(recon_mesh.vertices), n_samp, replace=False)
    _, dists_h, _ = trimesh.proximity.closest_point(orig_mesh, recon_mesh.vertices[samp_idx])
    diag = float(np.linalg.norm(aabb_max - aabb_min))

    print(f"\n  ┌─ Surface accuracy: isosurface ↔ original mesh ────────────")
    print(f"  │  Sample size          : {n_samp:,}")
    print(f"  │  Mean dist (normalised): {dists_h.mean():.6f}"
          f"  ({100*dists_h.mean()/diag:.3f}% of diag)")
    print(f"  │  Median dist          : {np.median(dists_h):.6f}")
    print(f"  │  95th-pct dist        : {np.percentile(dists_h, 95):.6f}")
    print(f"  │  Max dist (pseudo-H.) : {dists_h.max():.6f}")
    print(f"  └──────────────────────────────────────────────────────────\n")

    # ── Save reconstructed mesh ───────────────────────────────────────────────
    recon_path = os.path.join(stl_dir, f"{stl_name}_sdf_recon.stl")
    recon_mesh.export(recon_path)
    print(f"  Reconstructed mesh saved : {recon_path}")
    print(f"  Original STL             : {abs_stl}")
    print("  Open both files in MeshLab / ParaView to compare visually.\n")

    # ── Open trimesh viewer (only when a display is available) ────────────────
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        orig_vis  = orig_mesh.copy()
        recon_vis = recon_mesh.copy()
        orig_vis.visual.face_colors  = np.array([ 80, 120, 220, 130], dtype=np.uint8)
        recon_vis.visual.face_colors = np.array([220,  80,  80, 130], dtype=np.uint8)
        print("  Opening trimesh 3-D viewer  (close window to exit):")
        print("    Blue  = original STL mesh")
        print("    Red   = neural SDF zero-isosurface\n")
        try:
            trimesh.Scene([orig_vis, recon_vis]).show()
        except Exception as e:
            print(f"  Could not open viewer ({e}).")
            print(f"  Load the saved STL files instead.")
    else:
        print("  No display detected (headless / SSH session).")
        print("  Interactive viewer skipped — load the saved STL files to inspect.")


# ─────────────────────────── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a neural SDF for an STL file, then optionally visualise the result.\n\n"
            "Examples:\n"
            "  # Train only:\n"
            "  conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf --stl stlFiles/bracket.stl\n\n"
            "  # Train for STL replacement / point queries:\n"
            "  conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf "
            "--stl stlFiles/bracket.stl --optimize_for geometry_query\n\n"
            "  # Train + visualise (opens matplotlib slices + trimesh viewer):\n"
            "  conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf "
            "--stl stlFiles/bracket.stl --visualise\n\n"
            "  # Visualise a previously trained checkpoint (no retraining):\n"
            "  conda run -n myenv python -m fieldopt.geometry.sdf.train_sdf "
            "--stl stlFiles/bracket.stl --visualise --skip_train"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stl",             required=True,            help="Path to the input STL file")
    parser.add_argument(
        "--optimize_for",
        default="balanced",
        choices=["balanced", "geometry_query", "inclusion"],
        help=(
            "Training preset. 'geometry_query' is recommended when the network "
            "should replace the STL for fast inside/outside/boundary queries."
        ),
    )
    parser.add_argument("--epochs",          type=int,   default=800,  help="Training epochs (default: 800)")
    parser.add_argument(
        "--batch_size", type=int, default=0,
        help="Training batch size. Use 0 to auto-tune on CUDA (default: 0).",
    )
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--lambda_eikonal",  type=float, default=0.02)
    parser.add_argument("--fourier_L",       type=int,   default=6)
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    parser.add_argument("--n_hidden",        type=int,   default=4)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--n_surf",          type=int,   default=80_000)
    parser.add_argument("--n_vol",           type=int,   default=40_000)
    parser.add_argument(
        "--refresh_fraction", type=float, default=0.50,
        help=(
            "Fraction of the training sample pool to replace with fresh samples "
            "during training (default: 0.50). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--refresh_every", type=int, default=2,
        help="Refresh the sample pool every N epochs (default: 2).",
    )
    parser.add_argument(
        "--log_every", type=int, default=1,
        help="Print training metrics every N epochs (default: 1).",
    )
    parser.add_argument(
        "--val_every", type=int, default=20,
        help="Run fresh-sample validation every N epochs (default: 20).",
    )
    parser.add_argument(
        "--val_samples", type=int, default=20_000,
        help="Number of fresh validation points for each validation run (default: 20000).",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=24,
        help="Stop after this many validation runs without improvement (default: 24).",
    )
    parser.add_argument(
        "--early_stopping_min_delta", type=float, default=1e-4,
        help="Minimum validation improvement required to reset patience (default: 1e-4).",
    )
    parser.add_argument(
        "--near_surface_band_ratio", type=float, default=0.05,
        help=(
            "Treat samples with |SDF| smaller than this fraction of the AABB diagonal "
            "as near-surface for loss weighting (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--near_surface_weight", type=float, default=2.0,
        help="Extra loss weight for near-surface samples (default: 2.0). Set 1.0 to disable.",
    )
    parser.add_argument(
        "--lambda_sign", type=float, default=0.05,
        help="Target weight for the explicit sign loss after warmup (default: 0.05). Set 0 to disable.",
    )
    parser.add_argument(
        "--sign_margin_ratio", type=float, default=0.01,
        help=(
            "Desired sign margin as a fraction of the AABB diagonal. "
            "Inside points are encouraged below -margin and outside above +margin."
        ),
    )
    parser.add_argument(
        "--sign_conflict_band_ratio", type=float, default=None,
        help=(
            "Mask sign supervision inside this |SDF| band to avoid fighting exact "
            "distance regression. Default: same as near_surface_band_ratio."
        ),
    )
    parser.add_argument(
        "--sign_near_surface_weight", type=float, default=1.0,
        help=(
            "Extra weight for sign supervision inside the near-surface band. "
            "Use 1.0 to leave sign loss neutral there; values below 1 de-emphasise it."
        ),
    )
    parser.add_argument(
        "--sign_start_epoch", type=int, default=1,
        help="Epoch at which sign supervision starts ramping in (default: 1).",
    )
    parser.add_argument(
        "--sign_ramp_epochs", type=int, default=25,
        help="Number of epochs used to ramp sign supervision to full strength (default: 25).",
    )
    parser.add_argument(
        "--early_stopping_metric",
        default="volume_sign_error",
        choices=["hybrid", "near_surface_mae", "mae", "volume_sign_error"],
        help=(
            "Metric used for early stopping. 'volume_sign_error' prioritises "
            "balanced volume-only inside/outside accuracy (default: volume_sign_error)."
        ),
    )
    parser.add_argument(
        "--decision_margin_ratio", type=float, default=0.005,
        help=(
            "Safety margin used for confident inside/outside reporting during validation "
            "(default: 0.005 of AABB diagonal)."
        ),
    )
    parser.add_argument(
        "--balance_volume_signs", action="store_true",
        help="Use rejection sampling so volume points are closer to inside/outside balance.",
    )
    parser.add_argument(
        "--volume_balance_ratio", type=float, default=0.5,
        help="Target inside fraction among sampled volume points when balancing is enabled.",
    )
    parser.add_argument(
        "--volume_balance_oversample", type=int, default=6,
        help="Oversampling factor used by balanced volume rejection sampling (default: 6).",
    )
    parser.add_argument(
        "--validation_balance_volume_signs", action="store_true",
        help=(
            "Balance inside/outside volume samples in validation too. "
            "If omitted, validation follows the training setting."
        ),
    )
    parser.add_argument(
        "--visualise", action="store_true",
        help="After training, open the 2-D slice plot and 3-D isosurface viewer.",
    )
    parser.add_argument(
        "--grid_res", type=int, default=128,
        help="Grid resolution for visualisation (default: 128). Use 64 for speed, 256 for quality.",
    )
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training and go straight to visualisation of an existing checkpoint.",
    )
    args = parser.parse_args()
    args = _apply_training_profile(args)

    if not args.skip_train:
        train(
            stl_file=args.stl,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_eikonal=args.lambda_eikonal,
            fourier_L=args.fourier_L,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            device=args.device,
            n_surf=args.n_surf,
            n_vol=args.n_vol,
            refresh_fraction=args.refresh_fraction,
            refresh_every=args.refresh_every,
            log_every=args.log_every,
            val_every=args.val_every,
            val_samples=args.val_samples,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            near_surface_band_ratio=args.near_surface_band_ratio,
            near_surface_weight=args.near_surface_weight,
            lambda_sign=args.lambda_sign,
            sign_margin_ratio=args.sign_margin_ratio,
            sign_conflict_band_ratio=args.sign_conflict_band_ratio,
            sign_near_surface_weight=args.sign_near_surface_weight,
            sign_start_epoch=args.sign_start_epoch,
            sign_ramp_epochs=args.sign_ramp_epochs,
            early_stopping_metric=args.early_stopping_metric,
            decision_margin_ratio=args.decision_margin_ratio,
            balance_volume_signs=args.balance_volume_signs,
            volume_balance_ratio=args.volume_balance_ratio,
            volume_balance_oversample=args.volume_balance_oversample,
            validation_balance_volume_signs=(
                True if args.validation_balance_volume_signs else None
            ),
        )

    if args.visualise or args.skip_train:
        visualise_sdf(
            stl_file=args.stl,
            device=args.device,
            grid_resolution=args.grid_res,
        )
