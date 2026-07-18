import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .train_sdf import (
        _configure_torch_runtime,
        _resolve_amp_settings,
        _autocast_context,
        _to_device_tensors,
        _load_normalised_mesh,
        generate_dataset,
        sample_validation_dataset,
    )
    from fieldopt.models.hash_encoder import MultiResHashEncoder
    from fieldopt.models.mlp import MLP
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fieldopt.geometry.sdf.train_sdf import (
        _configure_torch_runtime,
        _resolve_amp_settings,
        _autocast_context,
        _to_device_tensors,
        _load_normalised_mesh,
        generate_dataset,
        sample_validation_dataset,
    )
    from fieldopt.models.hash_encoder import MultiResHashEncoder
    from fieldopt.models.mlp import MLP


# ─────────────────────── instant-NGP Network ─────────────────────────────────

class NGPInclusionNet(nn.Module):
    """
    Instant-NGP style inside/outside classifier:
    raw coords -> multi-resolution hash encoder -> MLP -> logit
    """

    def __init__(
        self,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        device: str,
        n_levels: int = 12,
        log2_hashmap_size: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 256,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        hashmap_size = 2 ** int(log2_hashmap_size)
        self.encoder = MultiResHashEncoder(
            L=int(n_levels),
            T=int(hashmap_size),
            F=int(n_features_per_level),
            N_min=int(base_resolution),
            N_max=int(finest_resolution),
            device=device,
            bounding_box=[
                aabb_min.astype(np.float32),
                aabb_max.astype(np.float32),
            ],
        )
        self.mlp = MLP(
            input_dim=self.encoder.output_dim,
            output_dim=1,
            n_neurons=int(hidden_dim),
            n_hidden_layers=int(n_hidden_layers),
            activation="relu",
            output_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(x).float())


# ─────────────────────── Training ────────────────────────────────────────────

def train_inclusion(
    stl_file: str,
    epochs: int = 1000,
    batch_size: int = 131072,
    lr: float = 5e-4,
    device: str = "cuda",
    n_surf: int = 250_000,
    n_vol: int = 250_000,
    val_samples: int = 50_000,
    n_levels: int = 12,
    log2_hashmap_size: int = 16,
    n_features_per_level: int = 2,
    base_resolution: int = 16,
    finest_resolution: int = 256,
    hidden_dim: int = 128,
    n_hidden_layers: int = 3,
    lambda_smooth: float = 0.1,
    early_stopping_patience: int = 10,
):
    """
    Train an instant-NGP network for high-precision inside/outside classification.
    Uses BCE with steep soft labels, boundary-focused weighting, and smoothness
    regularization to prevent hash-grid overfitting.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    _configure_torch_runtime(device)

    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]
    save_path = os.path.join(stl_dir, f"{stl_name}_sdf.pt")

    print(f"\n{'='*60}")
    print(f"Training instant-NGP inclusion classifier for : {stl_file}")
    print(f"Checkpoint will be saved to : {save_path}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size:,}")
    print(f"Network: instant-NGP ({n_levels} levels, hashmap 2^{log2_hashmap_size}, MLP {n_hidden_layers}x{hidden_dim})")
    print(f"Loss: Boundary-Focused BCE + Smoothness(λ={lambda_smooth})")
    print(f"Early stopping patience: {early_stopping_patience} val rounds")
    print("=" * 60)

    print("\n[1/4] Loading and normalising mesh...")
    mesh, aabb_min_np, aabb_max_np = _load_normalised_mesh(stl_file)
    diag = float(np.linalg.norm(aabb_max_np - aabb_min_np))

    # Pad AABB for the hash encoder to cover volume points (sampled ±10% beyond AABB)
    pad = 0.12 * (aabb_max_np - aabb_min_np)
    encoder_aabb_min = (aabb_min_np - pad).astype(np.float32)
    encoder_aabb_max = (aabb_max_np + pad).astype(np.float32)

    print("\n[2/4] Generating initial training dataset...")
    pts_np, sdf_np = generate_dataset(
        mesh,
        aabb_min_np,
        aabb_max_np,
        n_surf,
        n_vol,
        balance_volume_signs=True,
        volume_balance_ratio=0.5,
    )

    pts_t, sdf_t = _to_device_tensors(pts_np, sdf_np, device=device)
    N = len(pts_t)

    transition_band = 0.02 * diag
    soft_labels_t = torch.clamp((sdf_t / (2.0 * transition_band)) + 0.5, 0.0, 1.0)

    val_pts_np, val_sdf_gt = sample_validation_dataset(
        mesh=mesh,
        aabb_min_np=aabb_min_np,
        aabb_max_np=aabb_max_np,
        n_eval=val_samples,
    )
    val_label_np = (val_sdf_gt >= 0.0).astype(np.float32)

    print("\n[3/4] Building instant-NGP classifier network...")
    ngp_config = dict(
        n_levels=n_levels,
        log2_hashmap_size=log2_hashmap_size,
        n_features_per_level=n_features_per_level,
        base_resolution=base_resolution,
        finest_resolution=finest_resolution,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
    )
    net = NGPInclusionNet(
        aabb_min=encoder_aabb_min,
        aabb_max=encoder_aabb_max,
        device=device,
        **ngp_config,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Hash levels: {n_levels}  |  Hashmap: 2^{log2_hashmap_size}  |  MLP: {n_hidden_layers}x{hidden_dim}")
    print(f"  Resolution: {base_resolution} -> {finest_resolution}")
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    amp_enabled, amp_dtype, amp_name, use_grad_scaler = _resolve_amp_settings(device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True) if use_grad_scaler else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_acc = 0.0
    best_state_dict = copy.deepcopy(net.state_dict())
    best_epoch = 0
    patience_counter = 0
    # Perturbation scale for smoothness regularization (fraction of AABB diagonal)
    smooth_sigma = 0.005 * diag

    def evaluate(pts, labels):
        net.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(pts), 65536):
                p = torch.tensor(pts[i:i+65536], dtype=torch.float32, device=device)
                with _autocast_context(device, amp_enabled, amp_dtype):
                    pred = net(p)
                preds.append(pred.float().cpu().numpy())
            preds = np.concatenate(preds)
            pred_labels = (preds >= 0).astype(np.float32)
            acc = (pred_labels.flatten() == labels.flatten()).mean()
        net.train()
        return acc

    print(f"\n[4/4] Training for up to {epochs} epochs...")
    near_surface_band = 0.05 * diag

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch"):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        epoch_smooth_loss = 0.0
        n_batches = 0
        epoch_acc = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            pts_batch = pts_t[idx]
            sdf_batch = sdf_t[idx]
            soft_label_batch = soft_labels_t[idx]

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled, amp_dtype):
                pred_logits = net(pts_batch)

                raw_loss = criterion(pred_logits, soft_label_batch)

                weights = torch.ones_like(raw_loss)
                near_mask = sdf_batch.abs() < near_surface_band
                weights[near_mask] = 10.0

                bce_loss = (weights * raw_loss).mean()

                # Smoothness regularization on a random 25% subset to stay fast
                if lambda_smooth > 0:
                    n_smooth = max(len(pts_batch) // 4, 256)
                    smooth_idx = torch.randint(0, len(pts_batch), (n_smooth,), device=device)
                    pts_sub = pts_batch[smooth_idx]
                    logits_sub = pred_logits[smooth_idx]
                    noise = torch.randn_like(pts_sub) * smooth_sigma
                    pred_perturbed = net(pts_sub + noise)
                    s_loss = ((logits_sub - pred_perturbed) ** 2).mean()
                else:
                    s_loss = torch.zeros((), device=device)

                loss = bce_loss + lambda_smooth * s_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += bce_loss.item()
            epoch_smooth_loss += s_loss.item()
            pred_bin = (pred_logits > 0.0).float()
            target_bin = (sdf_batch >= 0.0).float()
            epoch_acc += (pred_bin == target_bin).float().mean().item()
            n_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            val_acc = evaluate(val_pts_np, val_label_np)

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(
                f"  Epoch {epoch:4d}/{epochs}  BCE={epoch_fieldopt/losses/max(n_batches,1):.4f}  "
                f"Smooth={epoch_smooth_fieldopt/losses/max(n_batches,1):.4f}  "
                f"TrainAcc={100*epoch_acc/max(n_batches,1):.2f}%  "
                f"ValAcc={100*val_acc:.2f}%  Best={100*best_acc:.2f}%@{best_epoch}  "
                f"P={patience_counter}/{early_stopping_patience}  LR={current_lr:.1e}"
            )

            if patience_counter >= early_stopping_patience:
                tqdm.write(
                    f"  Early stopping at epoch {epoch}. "
                    f"Best ValAcc={100*best_acc:.2f}% from epoch {best_epoch}."
                )
                break

    print(f"\n  Restoring best model (ValAcc={100*best_acc:.2f}% from epoch {best_epoch})...")
    net.load_state_dict(best_state_dict)

    checkpoint = {
        "state_dict": net.state_dict(),
        "aabb_min": aabb_min_np,
        "aabb_max": aabb_max_np,
        "encoder_aabb_min": encoder_aabb_min,
        "encoder_aabb_max": encoder_aabb_max,
        "network_type": "ngp",
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
    }
    checkpoint.update({f"ngp_{k}": v for k, v in ngp_config.items()})
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")

    return save_path, net, aabb_min_np, aabb_max_np, encoder_aabb_min, encoder_aabb_max, ngp_config


# ─────────────────────── Visualisation ───────────────────────────────────────

def visualise_ngp_inclusion(
    net: NGPInclusionNet,
    aabb_min_np: np.ndarray,
    aabb_max_np: np.ndarray,
    stl_file: str,
    device: str = "cuda",
    grid_resolution: int = 128,
):
    """Visualise the trained NGP inclusion classifier with slice plots and isosurface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_torch_runtime(device)
    amp_enabled, amp_dtype, _, _ = _resolve_amp_settings(device)

    abs_stl = os.path.abspath(stl_file)
    stl_dir = os.path.dirname(abs_stl)
    stl_name = os.path.splitext(os.path.basename(abs_stl))[0]

    R = grid_resolution
    vis_pad = 0.10 * (aabb_max_np - aabb_min_np)
    vis_min = aabb_min_np - vis_pad
    vis_max = aabb_max_np + vis_pad

    print(f"\n{'='*60}")
    print(f"Visualising NGP inclusion: {stl_name}")
    print("=" * 60)

    print(f"\n[1/3] Evaluating on {R}³ = {R**3:,} grid points (AABB ± 10%)...")
    xs = np.linspace(vis_min[0], vis_max[0], R, dtype=np.float32)
    ys = np.linspace(vis_min[1], vis_max[1], R, dtype=np.float32)
    zs = np.linspace(vis_min[2], vis_max[2], R, dtype=np.float32)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    sdf_chunks: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(pts_grid), 65536):
            p = torch.tensor(pts_grid[i:i+65536], dtype=torch.float32, device=device)
            with _autocast_context(device, amp_enabled, amp_dtype):
                sdf_chunks.append(net(p).float().cpu().numpy())

    sdf_vol = np.concatenate(sdf_chunks).reshape(R, R, R)
    print(f"  Logit range: [{sdf_vol.min():.4f}, {sdf_vol.max():.4f}]")
    print(f"  Inside fraction: {(sdf_vol < 0).mean() * 100:.1f}%")

    print("\n[2/3] Generating 2-D slice plots...")
    mid = R // 2
    slice_specs = [
        (sdf_vol[:, :, mid], xs, ys, f"XY plane (z={zs[mid]:.3f})", "x", "y"),
        (sdf_vol[:, mid, :], xs, zs, f"XZ plane (y={ys[mid]:.3f})", "x", "z"),
        (sdf_vol[mid, :, :], ys, zs, f"YZ plane (x={xs[mid]:.3f})", "y", "z"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"NGP Inclusion — {stl_name} — grid {R}³\n"
        f"Black contour = decision boundary  |  blue = inside, red = outside",
        fontsize=11, fontweight='bold',
    )
    for ax, (sl, xc, yc, title, xlabel, ylabel) in zip(axes, slice_specs):
        vmax = max(float(np.percentile(np.abs(sl), 97)), 1e-4)
        im = ax.imshow(
            sl.T, origin='lower', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
            aspect='auto',
        )
        ax.contour(xc, yc, sl.T, levels=[0.0], colors='k', linewidths=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{xlabel} (normalised)", fontsize=9)
        ax.set_ylabel(f"{ylabel} (normalised)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(stl_dir, f"{stl_name}_ngp_inclusion_slices.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    print("\n[3/3] Extracting zero-isosurface via marching cubes...")
    try:
        from skimage.measure import marching_cubes
        import trimesh
    except ImportError:
        print("  scikit-image or trimesh not installed — skipping 3-D reconstruction.")
        return

    if sdf_vol.min() >= 0.0 or sdf_vol.max() <= 0.0:
        print(
            f"  WARNING: Logit range [{sdf_vol.min():.4f}, {sdf_vol.max():.4f}] "
            f"does not cross 0 — cannot extract isosurface."
        )
        return

    spacing = ((vis_max - vis_min) / (R - 1)).tolist()
    verts, faces, _, _ = marching_cubes(sdf_vol, level=0.0, spacing=spacing)
    verts = verts + vis_min

    recon_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    recon_mesh.fix_normals()
    print(f"  Reconstructed: {len(verts):,} vertices, {len(faces):,} faces")

    orig_mesh, _, _ = _load_normalised_mesh(stl_file)
    print(f"  Original STL : {len(orig_mesh.vertices):,} vertices, {len(orig_mesh.faces):,} faces")

    n_samp = min(20_000, len(recon_mesh.vertices))
    samp_idx = np.random.choice(len(recon_mesh.vertices), n_samp, replace=False)
    _, dists_h, _ = trimesh.proximity.closest_point(orig_mesh, recon_mesh.vertices[samp_idx])
    diag = float(np.linalg.norm(aabb_max_np - aabb_min_np))

    print(f"\n  ┌─ Surface accuracy: isosurface ↔ original mesh ────────────")
    print(f"  │  Sample size          : {n_samp:,}")
    print(f"  │  Mean dist (normalised): {dists_h.mean():.6f}"
          f"  ({100*dists_h.mean()/diag:.3f}% of diag)")
    print(f"  │  Median dist          : {np.median(dists_h):.6f}")
    print(f"  │  95th-pct dist        : {np.percentile(dists_h, 95):.6f}")
    print(f"  │  Max dist (pseudo-H.) : {dists_h.max():.6f}")
    print(f"  └──────────────────────────────────────────────────────────\n")

    recon_path = os.path.join(stl_dir, f"{stl_name}_ngp_inclusion_recon.stl")
    recon_mesh.export(recon_path)
    print(f"  Reconstructed mesh saved : {recon_path}")
    print(f"  Original STL             : {abs_stl}")

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        orig_vis = orig_mesh.copy()
        recon_vis = recon_mesh.copy()
        orig_vis.visual.face_colors = np.array([80, 120, 220, 130], dtype=np.uint8)
        recon_vis.visual.face_colors = np.array([220, 80, 80, 130], dtype=np.uint8)
        print("  Opening trimesh 3-D viewer  (close window to exit):")
        print("    Blue  = original STL mesh")
        print("    Red   = NGP inclusion zero-isosurface\n")
        try:
            trimesh.Scene([orig_vis, recon_vis]).show()
        except Exception as e:
            print(f"  Could not open viewer ({e}).")


# ─────────────────────── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Instant-NGP inclusion classifier training (BCE + smoothness)",
    )
    parser.add_argument("--stl", required=True, help="Path to STL file")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--visualise", action="store_true", help="Visualize result after training")
    parser.add_argument("--n_levels", type=int, default=12, help="Number of hash grid levels (default: 12)")
    parser.add_argument("--log2_hashmap_size", type=int, default=16, help="log2 of hash table size per level (default: 16)")
    parser.add_argument("--finest_resolution", type=int, default=256, help="Resolution of finest hash level (default: 256)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="MLP hidden layer width (default: 128)")
    parser.add_argument("--n_hidden_layers", type=int, default=3, help="Number of MLP hidden layers (default: 3)")
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness regularization weight (default: 0.1)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Stop after N val rounds without improvement (default: 10)")
    args = parser.parse_args()

    result = train_inclusion(
        args.stl, args.epochs, args.batch_size, args.lr, device="cuda",
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        finest_resolution=args.finest_resolution,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        lambda_smooth=args.lambda_smooth,
        early_stopping_patience=args.early_stopping_patience,
    )
    save_path, net, aabb_min_np, aabb_max_np = result[0], result[1], result[2], result[3]

    if args.visualise:
        print("\nOpening visualizer...")
        visualise_ngp_inclusion(
            net, aabb_min_np, aabb_max_np,
            args.stl, device="cuda", grid_resolution=128,
        )
