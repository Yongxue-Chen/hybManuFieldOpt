"""
Bayesian Optimization Framework for Weight Tuning
===================================================

Uses Optuna (TPE sampler) to find optimal WEIGHTS parameters by minimizing
the maximum normalized loss: max(loss_i / threshold_i) for evaluated terms.

When the objective < 1.0, ALL evaluated losses are below their thresholds.
The optimizer naturally steers toward balanced, feasible solutions.

Features:
- Minimax objective: minimize the worst-performing normalized loss
- Best model saving (pth file)
- Configurable loss exclusion from evaluation
- Search space and thresholds loaded dynamically from config

Usage:
    conda run -n myenv python bayesian_weight_optimizer.py
    conda run -n myenv python bayesian_weight_optimizer.py --dry_run --n_trials 5 --model_name GLman
"""

import torch
import optuna
from optuna.samplers import TPESampler
import json
import csv
import os
import gc
import time
import copy
import math
import importlib
from fieldopt.geometry.backend import add_geometry_backend_args
from datetime import datetime

# ==============================================================================
# Configuration: General
# ==============================================================================

N_TRIALS = 100
MODEL_NAME = 'spiralBall'  # Default, can be overridden by --model_name

EXCLUDE_FROM_EVAL = [
    'structure_simp_penalty',
    'self_support_max'
]



def run_training(weights_dict: dict, shared_H_gpu=None, shared_struc_loss_calculator=None,
                 joint_batch_size=None) -> tuple:
    """
    Run one full training session with the given weights and return
    (loss_dict, metrics_dict, model_state_dict).
    """
    from main_optimize import main
    import sys
    
    old_argv = sys.argv
    sys.argv = ['main_optimize.py']
    
    try:
            return main(
                weights_dict=weights_dict,
                quiet=True,
                model_name=MODEL_NAME,
                shared_H_gpu=shared_H_gpu,
                shared_struc_loss_calculator=shared_struc_loss_calculator,
                joint_batch_size=joint_batch_size,
            )
    finally:
        sys.argv = old_argv


def _run_training_subprocess(weights_dict: dict) -> tuple:
    """
    Alternative: run training as a subprocess.
    """
    raise NotImplementedError(
        "Subprocess-based training is not implemented. Use run_training() instead."
    )


# ==============================================================================
# Dummy training function for testing the framework
# ==============================================================================

def _dummy_run_training(weights_dict: dict, metric_targets: dict) -> tuple:
    """
    Dummy training function for testing the optimization framework.
    Returns (loss_dict, metrics_dict, fake_model_state_dict).
    """
    import random

    losses = {
        'final_state': random.uniform(0.01, 0.1),
        'self_support_avg': random.uniform(0.001, 0.01),
        'self_support_max': random.uniform(0.01, 0.1),
        'AM_Collision_Free': random.uniform(0.0001, 0.01),
        'SM_Collision_Free': random.uniform(0.001, 0.05),
        'operation_volume': random.uniform(0.1, 0.5),
        'structure': random.uniform(0.0001, 0.01),
        'structure_simp_penalty': random.uniform(0.0, 0.01),
    }

    metrics = {}
    for name, (target, direction) in metric_targets.items():
        if direction == 'higher':
            base = target * random.uniform(0.85, 1.15)
            metrics[name] = min(1.0, max(0.0, base))
        else:
            base = target * random.uniform(0.3, 2.0)
            metrics[name] = max(0.0, base)
    
    fake_model_state = {"dummy_param": "trial_model"}
    
    return losses, metrics, fake_model_state


# ==============================================================================
# Core Optimization Logic
# ==============================================================================

def get_evaluated_thresholds(thresholds: dict):
    """Return only the thresholds that participate in minimax evaluation."""
    return {
        term: threshold
        for term, threshold in thresholds.items()
        if term not in EXCLUDE_FROM_EVAL
    }


def compute_minimax_objective(loss_dict: dict, thresholds: dict) -> tuple:
    """
    Compute minimax objective: max(loss_i / threshold_i) for evaluated terms.
    
    Returns:
        (max_normalized_loss, worst_term, normalized_losses_dict, feasible)
    """
    eval_thresholds = get_evaluated_thresholds(thresholds)
    
    normalized_losses = {}
    for term, threshold in eval_thresholds.items():
        loss_val = loss_dict.get(term, 0.0)
        if threshold == float('inf') or threshold == 0:
            normalized_losses[term] = 0.0
        else:
            normalized_losses[term] = loss_val / threshold
        
        if normalized_losses[term] < 1.01:
            if term == 'final_state' or term == 'operation_volume':
                normalized_losses[term] = 0.0
    
    # Also compute normalized values for excluded terms (for logging only)
    all_normalized = dict(normalized_losses)
    for term in EXCLUDE_FROM_EVAL:
        if term in thresholds and term in loss_dict:
            threshold = thresholds[term]
            if threshold != float('inf') and threshold != 0:
                all_normalized[term] = loss_dict[term] / threshold
            else:
                all_normalized[term] = 0.0
    
    max_normalized_loss = max(normalized_losses.values()) if normalized_losses else 0.0
    worst_term = max(normalized_losses, key=normalized_losses.get) if normalized_losses else "N/A"
    feasible = max_normalized_loss <= 1.0
    
    return max_normalized_loss, worst_term, all_normalized, feasible


def compute_metric_objective(metrics_dict: dict, metric_targets: dict) -> tuple:
    """
    Compute minimax objective from physical metrics.
    
    For 'higher is better': normalized = target / max(actual, eps)
    For 'lower is better':  normalized = actual / target
    
    Feasible when all normalized <= 1.0.
    
    Returns:
        (max_normalized, worst_term, normalized_dict, feasible)
    """
    normalized = {}
    for name, (target, direction) in metric_targets.items():
        if name in EXCLUDE_FROM_EVAL:
            continue
        actual = metrics_dict.get(name, 0.0)
        if direction == 'higher':
            if name != "operation_volume_ratio" and actual > 0.99:
                normalized[name] = 0.0
            else:
                normalized[name] = target / max(actual, 1e-6)
        else:
            normalized[name] = actual / target if target > 0 else 0.0
        
        if (name == "self_support_ratio" or name == "final_state_accuracy") and normalized[name] > 1.0:
            normalized[name] = normalized[name] * 5.0
    
    max_norm = max(normalized.values()) if normalized else 0.0
    worst_term = max(normalized, key=normalized.get) if normalized else "N/A"
    feasible = max_norm <= 1.0
    return max_norm, worst_term, normalized, feasible


def _is_continuous_space(space_def) -> bool:
    """Continuous space format: (low, high, log_scale_bool)."""
    return (
        isinstance(space_def, (list, tuple))
        and len(space_def) == 3
        and isinstance(space_def[2], bool)
    )


def _is_discrete_range_space(space_def) -> bool:
    """Discrete ranged space format: {'type': 'discrete_range', 'low': x, 'high': y, 'step': z}."""
    return (
        isinstance(space_def, dict)
        and space_def.get("type") == "discrete_range"
        and "low" in space_def
        and "high" in space_def
        and "step" in space_def
    )


def _is_discrete_space(space_def) -> bool:
    """Discrete space format: [v1, v2, ...] (or tuple with >=1 values)."""
    return isinstance(space_def, (list, tuple)) and not _is_continuous_space(space_def) and len(space_def) > 0


def _decimal_places(value) -> int:
    """Infer a stable rounding precision from a numeric step or boundary."""
    text = format(float(value), ".12f").rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def _expand_discrete_range(space_def) -> list:
    """Expand a discrete range definition into an explicit candidate list."""
    low = float(space_def["low"])
    high = float(space_def["high"])
    step = float(space_def["step"])

    if step <= 0:
        raise ValueError(f"Discrete range step must be positive, got {step}.")
    if high < low:
        raise ValueError(f"Discrete range high must be >= low, got low={low}, high={high}.")

    precision = max(_decimal_places(low), _decimal_places(high), _decimal_places(step), 8)
    n_steps_float = (high - low) / step
    n_steps = int(round(n_steps_float))
    if not math.isclose(low + n_steps * step, high, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "Discrete range requires (high - low) to be divisible by step. "
            f"Got low={low}, high={high}, step={step}."
        )

    return [round(low + i * step, precision) for i in range(n_steps + 1)]


def _sample_weight(trial: optuna.Trial, name: str, space_def):
    """Sample one parameter from either continuous or discrete search space."""
    if _is_continuous_space(space_def):
        low, high, log_scale = space_def
        if log_scale:
            return trial.suggest_float(name, low, high, log=True)
        return trial.suggest_float(name, low, high)
    if _is_discrete_range_space(space_def):
        return trial.suggest_categorical(name, _expand_discrete_range(space_def))
    if _is_discrete_space(space_def):
        return trial.suggest_categorical(name, list(space_def))
    raise ValueError(
        f"Invalid search space for '{name}': {space_def}. "
        "Use (low, high, log_bool) for continuous, "
        "{'type': 'discrete_range', 'low': ..., 'high': ..., 'step': ...} for stepped discrete, "
        "or [choices...] for discrete."
    )


def _serialize_space_def(space_def):
    """Serialize search space definition for logging/output."""
    if _is_continuous_space(space_def):
        return {
            "type": "continuous",
            "low": space_def[0],
            "high": space_def[1],
            "log": space_def[2],
        }
    if _is_discrete_range_space(space_def):
        choices = _expand_discrete_range(space_def)
        return {
            "type": "discrete_range",
            "low": space_def["low"],
            "high": space_def["high"],
            "step": space_def["step"],
            "n_choices": len(choices),
            "choices": choices,
        }
    if _is_discrete_space(space_def):
        return {
            "type": "discrete",
            "choices": list(space_def),
        }
    return {
        "type": "invalid",
        "raw": repr(space_def),
    }


def _is_better_metric(candidate_value, best_value, direction: str) -> bool:
    """Return True if candidate is better than best according to direction."""
    if best_value is None:
        return True
    if direction == "higher":
        return candidate_value > best_value
    return candidate_value < best_value


def _initialize_metric_best_state(metric_targets: dict, study=None) -> dict:
    """Build metric best tracker, including historical completed trials when resuming."""
    metric_best_state = {
        name: {"value": None, "trial": None, "direction": direction}
        for name, (_, direction) in metric_targets.items()
    }

    if study is None:
        return metric_best_state

    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        for name, (_, direction) in metric_targets.items():
            metric_key = f"metric_{name}"
            if metric_key not in t.user_attrs:
                continue
            candidate = t.user_attrs[metric_key]
            if candidate is None:
                continue
            try:
                candidate = float(candidate)
            except Exception:
                continue
            if math.isnan(candidate):
                continue
            if _is_better_metric(candidate, metric_best_state[name]["value"], direction):
                metric_best_state[name]["value"] = candidate
                metric_best_state[name]["trial"] = t.number
    return metric_best_state


def _save_current_best_snapshot(
    output_dir,
    trial_number,
    best_model_state,
    metric_best_state,
    metric_targets,
):
    """Persist and optionally upload current best weights + per-metric best values."""
    if output_dir is None:
        return None

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "current_trial": trial_number,
        "best_weights_so_far": {
            "trial_number": best_model_state.get("trial"),
            "objective_value": best_model_state.get("objective"),
            "weights": best_model_state.get("weights", {}),
        },
        "best_metric_values_so_far": {},
        "metric_targets": {
            name: {"target": target, "direction": direction}
            for name, (target, direction) in metric_targets.items()
        },
    }

    for name in metric_targets.keys():
        info = metric_best_state.get(name, {})
        snapshot["best_metric_values_so_far"][name] = {
            "best_value": info.get("value"),
            "trial_number": info.get("trial"),
            "direction": info.get("direction"),
        }

    snapshot_path = os.path.join(output_dir, "current_best_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"  📁 Saved current best snapshot to: {snapshot_path}")

    return snapshot_path


def create_objective(training_fn, search_space, metric_targets, is_dry_run=False, study=None, initial_model_state=None, output_dir=None, global_start_time=None):
    """
    Create an Optuna objective function using Minimax strategy on physical metrics.
    
    Objective: minimize max(normalized_metric_i) where normalized <= 1.0 means feasible.
    Terms in EXCLUDE_FROM_EVAL are logged but don't affect the objective.
    """
    
    initial_best_obj = float('inf')
    initial_best_trial = -1
    initial_best_weights = {}
    if study is not None and len(study.trials) > 0:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            initial_best_obj = study.best_value
            initial_best_trial = study.best_trial.number
            initial_best_weights = dict(study.best_trial.params)
            print(f"Resuming with historical best objective: {initial_best_obj:.6f}")

    best_model_state = {
        "state": initial_model_state,
        "objective": initial_best_obj,
        "trial": initial_best_trial,
        "weights": initial_best_weights,
    }
    metric_best_state = _initialize_metric_best_state(metric_targets, study=study)
    
    def objective(trial: optuna.Trial) -> float:
        # 1. Sample weights from search space
        weights_dict = {}
        for name, space_def in search_space.items():
            weights_dict[name] = _sample_weight(trial, name, space_def)
        
        # 2. Run training
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: Starting training...")
        print(f"Weights: {json.dumps({k: round(v, 6) for k, v in weights_dict.items()}, indent=2)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if is_dry_run:
                 loss_dict, metrics_dict, model_state = training_fn(weights_dict, metric_targets)
            else:
                 loss_dict, metrics_dict, model_state = training_fn(weights_dict)
                 
                 if torch.cuda.is_available():
                     mem_before = torch.cuda.memory_allocated() / 1024**3
                     print(f"  [GPU Memory] Before cleanup: {mem_before:.2f} GB allocated")
                 
                 model_state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v 
                                for k, v in model_state.items()}

                 loss_dict = {k: float(v) if hasattr(v, 'item') else v for k, v in loss_dict.items()}
                 metrics_dict = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics_dict.items()}

                 gc.collect()
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                     mem_after = torch.cuda.memory_allocated() / 1024**3
                     print(f"  [GPU Memory] After cleanup:  {mem_after:.2f} GB allocated (freed {mem_before - mem_after:.2f} GB)")
                 
        except Exception as e:
            print(f"Trial {trial.number} FAILED: {e}")
            trial.set_user_attr("failed", True)
            trial.set_user_attr("error", str(e))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('inf')
        
        elapsed = time.time() - start_time
        total_elapsed = time.time() - global_start_time if global_start_time else elapsed
        
        # 3. Compute minimax objective from physical metrics
        max_norm, worst_term, all_normalized, feasible = compute_metric_objective(metrics_dict, metric_targets)
        
        # 4. Log to console
        print(f"\nTrial {trial.number} completed in {elapsed:.1f}s")
        print(f"  Trial running time: {elapsed:.1f}s")
        print(f"  Total elapsed time: {total_elapsed:.1f}s")
        print(f"  Minimax objective: {max_norm:.6f} {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
        print(f"  Worst term: {worst_term} ({all_normalized.get(worst_term, 0):.4f}x target)")
        for name, (target, direction) in metric_targets.items():
            actual = metrics_dict.get(name, float('nan'))
            ratio = all_normalized.get(name, float('nan'))
            excluded_tag = " [EXCLUDED]" if name in EXCLUDE_FROM_EVAL else ""
            status = "✅" if ratio <= 1.0 else "❌"
            dir_tag = "higher is better" if direction == 'higher' else "lower is better"
            print(f"  {status} {name}: {actual:.6f} / {target} target = {ratio:.4f}x  ({dir_tag}){excluded_tag}")
        
        # Also print raw losses for reference
        print(f"  Raw losses:")
        for term, val in loss_dict.items():
            print(f"    {term}: {val:.6f}")
        
        # 5. Check for new best
        is_new_best = False
        if max_norm < best_model_state["objective"]:
            best_model_state["state"] = copy.deepcopy(model_state)
            best_model_state["objective"] = max_norm
            best_model_state["trial"] = trial.number
            best_model_state["weights"] = copy.deepcopy(weights_dict)
            is_new_best = True
            print(f"  🏆 New best model! (trial {trial.number}, objective {max_norm:.6f})")
            
            if output_dir is not None and best_model_state["state"] is not None:
                try:
                    model_path = os.path.join(output_dir, "best_model.pth")
                    torch.save(best_model_state["state"], model_path)
                    print(f"  💾 Saved intermediate best model to {model_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to save intermediate best model: {e}")

        # Track best metric value so far for each metric
        new_metric_bests = []
        for name, (_, direction) in metric_targets.items():
            if name not in metrics_dict:
                continue
            try:
                candidate = float(metrics_dict[name])
            except Exception:
                continue
            if math.isnan(candidate):
                continue
            if _is_better_metric(candidate, metric_best_state[name]["value"], direction):
                metric_best_state[name]["value"] = candidate
                metric_best_state[name]["trial"] = trial.number
                metric_best_state[name]["direction"] = direction
                new_metric_bests.append(name)

        current_best_weights = best_model_state.get("weights", {}) or {}
        print("  Current best weights so far:")
        if current_best_weights:
            print(f"    best trial: {best_model_state.get('trial')} | objective: {best_model_state.get('objective'):.6f}")
            for name in search_space.keys():
                if name in current_best_weights:
                    print(f"    {name}: {float(current_best_weights[name]):.6f}")
        else:
            print("    (not available yet)")

        print("  Current best metric values so far:")
        for name, (_, direction) in metric_targets.items():
            best_info = metric_best_state.get(name, {})
            best_val = best_info.get("value")
            best_trial = best_info.get("trial")
            marker = " 🆕" if name in new_metric_bests else ""
            if best_val is None:
                print(f"    {name}: N/A ({direction} is better){marker}")
            else:
                print(f"    {name}: {best_val:.6f} (trial {best_trial}, {direction} is better){marker}")

        _save_current_best_snapshot(
            output_dir=output_dir,
            trial_number=trial.number,
            best_model_state=best_model_state,
            metric_best_state=metric_best_state,
            metric_targets=metric_targets,
        )

        # 6. Store trial metadata
        for key, val in loss_dict.items():
            trial.set_user_attr(f"loss_{key}", val)
        for key, val in metrics_dict.items():
            trial.set_user_attr(f"metric_{key}", val)
        for key, val in all_normalized.items():
            trial.set_user_attr(f"norm_{key}", val)
        trial.set_user_attr("feasible", bool(feasible))
        trial.set_user_attr("max_normalized_loss", float(max_norm))
        trial.set_user_attr("worst_term", worst_term)
        trial.set_user_attr("running_time", elapsed)
        trial.set_user_attr("total_elapsed_time", total_elapsed)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return max_norm
    
    return objective, best_model_state


def run_optimization(
    training_fn=None,
    search_space=None,
    metric_targets=None,
    n_trials=N_TRIALS,
    study_name=None,
    output_dir=None,
    resume=False,
    is_dry_run=False,
    joint_batch_size=None,
    geometry_backend='voxel_artifact',
    geometry_artifact_path=None,
):
    """
    Run the Bayesian optimization loop.
    """
    global_start_time = time.time()
    if training_fn is None:
        training_fn = run_training
    
    if search_space is None or metric_targets is None:
        raise ValueError("search_space and metric_targets must be provided!")

    os.makedirs(output_dir, exist_ok=True)
    
    # --- Pre-create expensive shared objects (only for real training) ---
    shared_H_gpu = None
    shared_struc_loss_calculator = None
    if not is_dry_run:
        import torch
        import importlib as _imp
        _cfg = _imp.import_module(f'configs.config_multi_field_{MODEL_NAME}')
        from fieldopt.geometry.backend import load_geometry_function
        from fieldopt.geometry.voxel.voxelization import get_normalization_parameters
        from fieldopt.losses.structure_loss_torch import StructureLossCalculatorTorch as StructureLossCalculator

        stl_path = f'stlFiles/{MODEL_NAME}.stl'
        # aabb_min, aabb_max = get_sdf_aabb(stl_path)
        # spaceBox = np.array([aabb_min, aabb_max])
        _, p_min, spaceBox = get_normalization_parameters(stl_path)
        spaceBox[0] = spaceBox[0] - p_min
        spaceBox[1] = spaceBox[1] - p_min
        aabb_min = spaceBox[0]
        aabb_max = spaceBox[1]

        # 1. Load the geometry function once, matching main_optimize.py behavior.
        shared_H_gpu = load_geometry_function(
            backend=geometry_backend,
            stl_path=stl_path,
            model_name=MODEL_NAME,
            artifact_path=geometry_artifact_path,
            device=_cfg.DEVICE,
            voxel_resolution=getattr(_cfg, 'HRES', 512),
        )
        print("[BayesOpt] Shared H_gpu ready.")

        # 2. Create StructureLossCalculator + torch.compile once
        print("[BayesOpt] Creating shared StructureLossCalculator...")
        normalizedSize = (aabb_max - aabb_min) / max(aabb_max - aabb_min)
        shared_struc_loss_calculator = StructureLossCalculator(
            physical_size=normalizedSize,
            max_resolution=_cfg.PARAS_STRUCTURE['max_resolution'],
            batch_size=_cfg.PARAS_STRUCTURE['time_check_size'],
            cg_iter=_cfg.PARAS_STRUCTURE['cg_iter'],
            cg_tol=_cfg.PARAS_STRUCTURE['cg_tol'],
            E0=_cfg.PARAS_STRUCTURE['e0'],
            grad_clip_threshold=_cfg.PARAS_STRUCTURE['grad_clip_threshold'],
            grad_scale=_cfg.PARAS_STRUCTURE['grad_scale'],
            use_total_mass=_cfg.PARAS_STRUCTURE['use_total_mass'],
            max_displacement=_cfg.PARAS_STRUCTURE['max_displacement']
        )
        print("[BayesOpt] Compiling _solve_cg with torch.compile (this may take a while)...")
        shared_struc_loss_calculator._solve_cg = torch.compile(
            shared_struc_loss_calculator._solve_cg,
            mode='default'
        )
        print("[BayesOpt] Shared StructureLossCalculator ready.")

        # Wrap training_fn to inject shared objects
        _original_training_fn = training_fn

        def training_fn(weights_dict):
            return _original_training_fn(
                weights_dict,
                shared_H_gpu=shared_H_gpu,
                shared_struc_loss_calculator=shared_struc_loss_calculator,
                joint_batch_size=joint_batch_size,
            )
    
    # --- Optuna Study Setup ---
    db_path = os.path.join(output_dir, f'{study_name}.db')
    storage = f"sqlite:///{db_path}"
    
    sampler = TPESampler(
        seed=1234,
        n_startup_trials=20,
        multivariate=True,
    )
    
    if resume:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
        )
        print(f"Resuming study '{study_name}' with {len(study.trials)} existing trials")
    else:
        # Fresh start: delete existing DB so everything begins from scratch
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"🗑️  Removed existing study DB for fresh start: {db_path}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            direction="minimize",
            load_if_exists=False,
        )

    # --- Load existing best model when resuming ---
    initial_model_state = None
    if resume:
        existing_model_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(existing_model_path):
            try:
                import torch as _torch
                initial_model_state = _torch.load(existing_model_path, map_location='cpu', weights_only=False)
                print(f"📂 Loaded existing best model from: {existing_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load existing best model: {e}")
        else:
            print(f"ℹ️  No existing best model found at: {existing_model_path}")

    # --- Enqueue default weights from config as the initial trial (only for fresh studies) ---
    if len(study.trials) == 0:
        try:
            _cfg = importlib.import_module(f'configs.config_multi_field_{MODEL_NAME}')
            default_weights = _cfg.WEIGHTS
            # Only enqueue weights that are in the search space
            initial_params = {k: v for k, v in default_weights.items() if k in search_space}
            if initial_params:
                study.enqueue_trial(initial_params)
                print(f"📌 Enqueued config default WEIGHTS as initial trial:")
                for k, v in initial_params.items():
                    print(f"    {k}: {v}")
        except Exception as e:
            print(f"⚠️  Could not enqueue default weights: {e}")
    else:
        print(f"ℹ️  Skipping default weights enqueue (study already has {len(study.trials)} trials)")

    # --- Run Optimization ---
    print(f"\n{'#'*60}")
    print(f"Starting Bayesian Optimization")
    print(f"  Trials: {n_trials}")
    print(f"  Study: {study_name}")
    print(f"  Output: {output_dir}")
    print(f"  DB File: {db_path} (use --resume to continue, omit to start fresh)")
    print(f"  Excluded from eval: {EXCLUDE_FROM_EVAL}")
    print(f"  Metric targets:")
    for name, (target, direction) in metric_targets.items():
        print(f"    {name}: {target} ({direction} is better)")
    print(f"{'#'*60}\n")
    
    objective_fn, best_model_state = create_objective(
        training_fn, search_space, metric_targets, is_dry_run, study=study,
        initial_model_state=initial_model_state, output_dir=output_dir, global_start_time=global_start_time
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    
    # ==============================================================================
    # Results Analysis
    # ==============================================================================
    
    print(f"\n{'#'*60}")
    print(f"Optimization Complete!")
    print(f"{'#'*60}\n")
    
    # Find best feasible trial (lowest minimax objective among feasible ones)
    best_feasible = None
    best_feasible_obj = float('inf')
    
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.user_attrs.get("feasible", False):
            if trial.value < best_feasible_obj:
                best_feasible_obj = trial.value
                best_feasible = trial
    
    # Also get overall best (may be infeasible)
    best_overall = study.best_trial
    
    # Print results
    print("=" * 60)
    print("BEST OVERALL TRIAL (may violate constraints)")
    print("=" * 60)
    _print_trial_summary(best_overall, metric_targets)
    
    if best_feasible is not None:
        print("\n" + "=" * 60)
        print("BEST FEASIBLE TRIAL (all evaluated constraints satisfied)")
        print("=" * 60)
        _print_trial_summary(best_feasible, metric_targets)
    else:
        print("\n⚠️  No feasible trial found! Consider:")
        print("  - Relaxing BAYESIAN_METRIC_TARGETS")
        print("  - Expanding WEIGHT_SEARCH_SPACE")
        print("  - Increasing N_TRIALS")
    
    # Save results
    _save_results(study, output_dir, best_feasible, best_model_state, search_space, metric_targets)
    
    
    return study


def _print_trial_summary(trial, metric_targets):
    """Print a summary of a trial."""
    feasible = trial.user_attrs.get('feasible', False)
    print(f"  Trial #{trial.number}")
    print(f"  Minimax objective: {trial.value:.6f} {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
    print(f"  Worst term: {trial.user_attrs.get('worst_term', 'N/A')}")
    print(f"  Trial running time: {trial.user_attrs.get('running_time', 0.0):.1f}s")
    print(f"  Total elapsed time: {trial.user_attrs.get('total_elapsed_time', 0.0):.1f}s")
    print(f"\n  Weights:")
    for name, value in trial.params.items():
        print(f"    {name}: {value:.6f}")
    print(f"\n  Metrics (actual / target = normalized):")
    for name, (target, direction) in metric_targets.items():
        metric_key = f"metric_{name}"
        norm_key = f"norm_{name}"
        actual = trial.user_attrs.get(metric_key, float('nan'))
        norm_val = trial.user_attrs.get(norm_key, float('nan'))
        excluded_tag = " [EXCLUDED]" if name in EXCLUDE_FROM_EVAL else ""
        status = "✅" if norm_val <= 1.0 else "❌"
        dir_tag = "higher" if direction == 'higher' else "lower"
        print(f"    {status} {name}: {actual:.6f} / {target} = {norm_val:.4f}x  ({dir_tag} is better){excluded_tag}")


def _save_results(study, output_dir, best_feasible, best_model_state, search_space, metric_targets):
    """Save optimization results to files."""
    
    # 1. Save best weights to JSON
    results = {
        "best_overall": {
            "trial_number": study.best_trial.number,
            "objective_value": study.best_trial.value,
            "trial_running_time": study.best_trial.user_attrs.get("running_time", 0.0),
            "total_elapsed_time": study.best_trial.user_attrs.get("total_elapsed_time", 0.0),
            "weights": study.best_trial.params,
            "feasible": study.best_trial.user_attrs.get("feasible", False),
            "losses": {k[5:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("loss_")},
            "metrics": {k[7:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("metric_")},
            "normalized": {k[5:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("norm_")},
        },
        "exclude_from_eval": EXCLUDE_FROM_EVAL,
        "metric_targets": {k: {"target": v[0], "direction": v[1]} for k, v in metric_targets.items()},
        "search_space": {k: _serialize_space_def(v) for k, v in search_space.items()},
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }
    
    if best_feasible is not None:
        results["best_feasible"] = {
            "trial_number": best_feasible.number,
            "objective_value": best_feasible.value,
            "trial_running_time": best_feasible.user_attrs.get("running_time", 0.0),
            "total_elapsed_time": best_feasible.user_attrs.get("total_elapsed_time", 0.0),
            "weights": best_feasible.params,
            "losses": {k[5:]: v for k, v in best_feasible.user_attrs.items() if k.startswith("loss_")},
            "metrics": {k[7:]: v for k, v in best_feasible.user_attrs.items() if k.startswith("metric_")},
            "normalized": {k[5:]: v for k, v in best_feasible.user_attrs.items() if k.startswith("norm_")},
        }
    
    results_path = os.path.join(output_dir, "best_weights.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Best weights saved to: {results_path}")
    if best_model_state["state"] is not None:
        import torch
        model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(best_model_state["state"], model_path)
        print(f"📁 Best model saved to: {model_path} (trial {best_model_state['trial']}, objective {best_model_state['objective']:.6f})")
    else:
        print("⚠️  No model state was saved (all trials may have failed)")
    
    # 3. Save trial history to CSV
    csv_path = os.path.join(output_dir, "trial_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = None
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            row = {
                "trial": trial.number,
                "objective_value": trial.value,
                "feasible": trial.user_attrs.get("feasible", False),
                "worst_term": trial.user_attrs.get("worst_term", ""),
                "trial_running_time": trial.user_attrs.get("running_time", 0.0),
                "total_elapsed_time": trial.user_attrs.get("total_elapsed_time", 0.0),
            }
            for name in search_space:
                row[f"w_{name}"] = trial.params.get(name, None)
            for key, val in trial.user_attrs.items():
                if key.startswith("loss_") or key.startswith("norm_"):
                    row[key] = val
            
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
    
    print(f"📁 Trial history saved to: {csv_path}")
    
    # 4. Print copy-paste ready weights dict
    best = best_feasible if best_feasible is not None else study.best_trial
    print(f"\n{'='*60}")
    print("Copy-paste ready WEIGHTS dict for your config:")
    print("=" * 60)
    print("WEIGHTS = {")
    for name in search_space:
        val = best.params.get(name, 0.0)
        print(f"    '{name}': {val:.6f},")
    print("}")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Bayesian optimization for weight tuning")
    parser.add_argument(
        "--n_trials", type=int, default=N_TRIALS,
        help=f"Number of optimization trials (default: {N_TRIALS})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing study database"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run with dummy training function for testing"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: output/bo_results_<MODEL_NAME>)"
    )
    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAME,
        help=f"Model name to use (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=180000,
        help="Override JOINT_BATCH_CONFIG['batch_size'] when running Bayesian optimization."
    )
    add_geometry_backend_args(parser)
    
    args = parser.parse_args()
    
    # Apply model name and update dependent configs
    MODEL_NAME = args.model_name
    
    # Load configuration dynamically based on MODEL_NAME
    try:
        cfg = importlib.import_module(f'configs.config_multi_field_{MODEL_NAME}')
        search_space = cfg.BAYESIAN_WEIGHT_SEARCH_SPACE
        metric_targets = cfg.BAYESIAN_METRIC_TARGETS
    except ImportError as e:
        print(f"❌ Error loading config for model '{MODEL_NAME}': {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"❌ Config for model '{MODEL_NAME}' is missing BAYESIAN_WEIGHT_SEARCH_SPACE or BAYESIAN_METRIC_TARGETS: {e}")
        sys.exit(1)
    
    # Update paths based on model name
    study_name = f"weight_opt_{MODEL_NAME}"
    
    if args.output_dir is None:
        args.output_dir = f"output/bo_results_{MODEL_NAME}"
    
    if args.dry_run:
        print("🧪 DRY RUN MODE: Using dummy training function")
        training_fn = _dummy_run_training
    else:
        training_fn = run_training
    print(f"Using joint batch size: {args.batch_size}")
    
    study = run_optimization(
        training_fn=training_fn,
        search_space=search_space,
        metric_targets=metric_targets,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        study_name=study_name,
        resume=args.resume,
        is_dry_run=args.dry_run,
        joint_batch_size=args.batch_size,
        geometry_backend=args.geometry_backend,
        geometry_artifact_path=args.geometry_artifact_path,
    )
