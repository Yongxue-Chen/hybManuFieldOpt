"""
Constrained Bayesian Optimization Framework for Weight Tuning
=============================================================

使用 Optuna TPE 搜索 loss weights，并施加以下硬约束（不修改任何 config 文件）：

  - final_state 的 weight 固定为 1.0，不进入搜索空间
  - operation_volume 的 weight 固定为 1.5，不进入搜索空间
  - SM_Collision_Free 的 weight 固定为 self_support weight 的 1/10（派生值，不独立搜索）
  - 自由变量（3个）：self_support、AM_Collision_Free、structure
  - BAYESIAN_METRIC_TARGETS 与 config 保持一致，动态加载

Usage:
    conda run -n fieldopt_hm_rtx5090 python bayesian_weight_optimizer_constrained.py --model_name bracket
    conda run -n fieldopt_hm_rtx5090 python bayesian_weight_optimizer_constrained.py --dry_run --n_trials 5 --model_name bracket
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

N_TRIALS = 50
MODEL_NAME = 'bracket'

EXCLUDE_FROM_EVAL = [
    'structure_simp_penalty',
    'self_support_max'
]

# 固定约束常量
FIXED_FINAL_STATE_WEIGHT = 1.0
FIXED_OPERATION_VOLUME_WEIGHT = 1.5
SM_TO_SELF_SUPPORT_RATIO = 0.1  # SM_Collision_Free = self_support * SM_TO_SELF_SUPPORT_RATIO



# ==============================================================================
# Constraint Helpers
# ==============================================================================

def build_constrained_search_space(cfg_search_space: dict) -> dict:
    """
    从 config 的 BAYESIAN_WEIGHT_SEARCH_SPACE 中构建约束搜索空间：
    - 移除 'SM_Collision_Free'（它将由 self_support 自动派生）
    - 移除 'operation_volume'（固定为 FIXED_OPERATION_VOLUME_WEIGHT = 1.5）
    - 保留 self_support、AM_Collision_Free、structure 共 3 个自由变量

    Args:
        cfg_search_space: config 中定义的原始搜索空间字典

    Returns:
        过滤后的搜索空间字典（仅含 3 个自由变量）
    """
    excluded_from_search = {'SM_Collision_Free', 'operation_volume'}
    constrained = {
        k: v for k, v in cfg_search_space.items()
        if k not in excluded_from_search
    }
    if 'self_support' not in constrained:
        raise ValueError(
            "Config 的 BAYESIAN_WEIGHT_SEARCH_SPACE 中必须包含 'self_support' 键，"
            "因为 SM_Collision_Free 由 self_support * {:.2f} 派生。".format(SM_TO_SELF_SUPPORT_RATIO)
        )
    return constrained


def apply_hard_constraints(weights_dict: dict) -> dict:
    """
    在 Optuna 采样的权重基础上，注入固定和派生权重值：
    - final_state     = FIXED_FINAL_STATE_WEIGHT (1.0)  [固定]
    - operation_volume = FIXED_OPERATION_VOLUME_WEIGHT (1.5)  [固定]
    - SM_Collision_Free = self_support * SM_TO_SELF_SUPPORT_RATIO (1/10)  [派生]

    Args:
        weights_dict: Optuna 采样的权重字典（仅含 3 个自由变量）

    Returns:
        包含全部权重（含固定和派生值）的字典，用于传给 training
    """
    full_weights = dict(weights_dict)
    full_weights['final_state'] = FIXED_FINAL_STATE_WEIGHT
    full_weights['operation_volume'] = FIXED_OPERATION_VOLUME_WEIGHT
    full_weights['SM_Collision_Free'] = weights_dict['self_support'] * SM_TO_SELF_SUPPORT_RATIO
    return full_weights


# ==============================================================================
# Training Function
# ==============================================================================

def run_training(weights_dict: dict, shared_H_gpu=None, shared_struc_loss_calculator=None,
                 joint_batch_size=None, main_optimize_args=None) -> tuple:
    """
    Run one full training session with the given weights and return
    (loss_dict, metrics_dict, model_state_dict).
    """
    from main_optimize import main
    import builtins
    import sys

    old_argv = sys.argv
    original_print = builtins.print
    sys.argv = ['main_optimize.py', *(main_optimize_args or [])]

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
        builtins.print = original_print


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

        if name in {
            "self_support_ratio",
            "final_state_accuracy",
            "AM_collision_free_ratio",
        } and normalized[name] > 1.002:
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
    return (
        isinstance(space_def, (list, tuple))
        and not _is_continuous_space(space_def)
        and len(space_def) > 0
    )


def _decimal_places(value) -> int:
    """Infer a stable rounding precision from a numeric step or boundary."""
    text = format(float(value), ".12f").rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def _expand_discrete_range(space_def) -> list:
    """
    Expand a discrete range definition into an explicit candidate list.

    The generated values include both ends when they lie on the grid.
    """
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

    values = [round(low + i * step, precision) for i in range(n_steps + 1)]
    return values


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


def _validate_enqueued_params_against_space(params: dict, search_space: dict):
    """Ensure enqueued/default parameters fall exactly on the declared search grid."""
    for name, value in params.items():
        if name not in search_space:
            continue
        space_def = search_space[name]
        if _is_continuous_space(space_def):
            low, high, _ = space_def
            if not (low <= value <= high):
                raise ValueError(
                    f"Default weight '{name}={value}' is outside continuous range [{low}, {high}]."
                )
            continue
        if _is_discrete_range_space(space_def):
            choices = _expand_discrete_range(space_def)
            if not any(math.isclose(float(value), float(choice), rel_tol=1e-9, abs_tol=1e-9) for choice in choices):
                raise ValueError(
                    f"Default weight '{name}={value}' is not on the discrete grid {choices}."
                )
            continue
        if _is_discrete_space(space_def):
            if value not in space_def:
                raise ValueError(
                    f"Default weight '{name}={value}' is not in discrete choices {list(space_def)}."
                )
            continue
        raise ValueError(f"Unsupported search space for '{name}': {space_def}")


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
        # 记录硬约束规则，方便复查
        "constraints": {
            "final_state": FIXED_FINAL_STATE_WEIGHT,
            "operation_volume": FIXED_OPERATION_VOLUME_WEIGHT,
            "SM_Collision_Free": f"self_support * {SM_TO_SELF_SUPPORT_RATIO}",
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

    注意：search_space 已经是过滤后的约束搜索空间（不含 SM_Collision_Free）。
    训练时会通过 apply_hard_constraints() 自动添加固定/派生权重。
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
        # 1. 从搜索空间采样自由权重
        sampled_weights = {}
        for name, space_def in search_space.items():
            sampled_weights[name] = _sample_weight(trial, name, space_def)

        # 2. 施加硬约束，生成完整的 weights_dict
        weights_dict = apply_hard_constraints(sampled_weights)

        # 3. 打印本次 trial 信息（含固定/派生值，方便核查）
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: Starting training...")
        print(f"  [搜索变量] {json.dumps({k: round(v, 6) for k, v in sampled_weights.items()}, indent=4)}")
        print(f"  [固定约束] final_state = {weights_dict['final_state']}")
        print(f"  [固定约束] operation_volume = {weights_dict['operation_volume']}")
        print(f"  [派生约束] SM_Collision_Free = self_support({sampled_weights['self_support']:.4f}) × {SM_TO_SELF_SUPPORT_RATIO} = {weights_dict['SM_Collision_Free']:.6f}")
        print(f"  [完整权重] {json.dumps({k: round(v, 6) for k, v in weights_dict.items()}, indent=4)}")
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

        # 4. 计算 minimax 目标
        max_norm, worst_term, all_normalized, feasible = compute_metric_objective(metrics_dict, metric_targets)

        # 5. 打印结果
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

        print(f"  Raw losses:")
        for term, val in loss_dict.items():
            print(f"    {term}: {val:.6f}")

        # 6. 更新最优模型
        is_new_best = False
        if max_norm < best_model_state["objective"]:
            best_model_state["state"] = copy.deepcopy(model_state)
            best_model_state["objective"] = max_norm
            best_model_state["trial"] = trial.number
            best_model_state["weights"] = copy.deepcopy(weights_dict)  # 保存完整权重（含固定/派生）
            is_new_best = True
            print(f"  🏆 New best model! (trial {trial.number}, objective {max_norm:.6f})")

            if output_dir is not None and best_model_state["state"] is not None:
                try:
                    model_path = os.path.join(output_dir, "best_model.pth")
                    torch.save(best_model_state["state"], model_path)
                    print(f"  💾 Saved intermediate best model to {model_path}")
                except Exception as e:
                    print(f"  ⚠️ Failed to save intermediate best model: {e}")

        # 7. 追踪每个 metric 的最优值
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
            # 打印完整权重（含固定/派生）
            fixed_names = ['final_state', 'operation_volume', 'SM_Collision_Free']
            for name in list(search_space.keys()) + fixed_names:
                if name in current_best_weights:
                    if name == 'final_state':
                        marker = " [fixed]"
                    elif name == 'operation_volume':
                        marker = " [fixed]"
                    elif name == 'SM_Collision_Free':
                        marker = " [derived=self_support×{:.2f}]".format(SM_TO_SELF_SUPPORT_RATIO)
                    else:
                        marker = ""
                    print(f"    {name}: {float(current_best_weights[name]):.6f}{marker}")
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

        # 8. 记录 trial 元数据
        for key, val in loss_dict.items():
            trial.set_user_attr(f"loss_{key}", val)
        for key, val in metrics_dict.items():
            trial.set_user_attr(f"metric_{key}", val)
        for key, val in all_normalized.items():
            trial.set_user_attr(f"norm_{key}", val)
        # 记录完整权重（含固定/派生）
        for key, val in weights_dict.items():
            trial.set_user_attr(f"weight_{key}", val)
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
    resolution=100,
    pretrain_model_path=None,
    stl_path=None,
    joint_epochs=None,
    steps_per_epoch=None,
):
    """
    Run the constrained Bayesian optimization loop.
    search_space 已是过滤后的约束搜索空间。
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
        from fieldopt.losses.structure_loss_torch import StructureLossCalculatorTorch
        from fieldopt.losses.structure_loss_torch_sm_force import (
            StructureLossCalculatorTorchSMForce,
        )

        effective_stl_path = stl_path or os.path.join(
            'model_data', 'preprocessed', MODEL_NAME, f'{MODEL_NAME}_support.stl'
        )
        effective_pretrain_model_path = pretrain_model_path or os.path.join(
            'model_data',
            'pretrained_fields',
            MODEL_NAME,
            f'{MODEL_NAME}_pretrained_{resolution}.pth',
        )
        for label, path in (
            ('preprocessed STL', effective_stl_path),
            ('pretrained checkpoint', effective_pretrain_model_path),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} not found: {path}")

        _, p_min, spaceBox = get_normalization_parameters(effective_stl_path)
        spaceBox[0] = spaceBox[0] - p_min
        spaceBox[1] = spaceBox[1] - p_min
        aabb_min = spaceBox[0]
        aabb_max = spaceBox[1]

        shared_H_gpu = load_geometry_function(
            backend=geometry_backend,
            stl_path=effective_stl_path,
            model_name=MODEL_NAME,
            artifact_path=geometry_artifact_path,
            device=_cfg.DEVICE,
            voxel_resolution=resolution,
        )
        print(f"[BayesOpt-Constrained] Shared geometry backend ready: {geometry_backend}")

        print("[BayesOpt-Constrained] Creating shared StructureLossCalculator...")
        normalizedSize = (aabb_max - aabb_min) / max(aabb_max - aabb_min)
        common_structure_kwargs = dict(
            physical_size=normalizedSize,
            max_resolution=_cfg.PARAS_STRUCTURE['max_resolution'],
            batch_size=_cfg.PARAS_STRUCTURE['time_check_size'],
            cg_iter=_cfg.PARAS_STRUCTURE['cg_iter'],
            cg_tol=_cfg.PARAS_STRUCTURE['cg_tol'],
            E0=_cfg.PARAS_STRUCTURE['e0'],
            grad_clip_threshold=_cfg.PARAS_STRUCTURE['grad_clip_threshold'],
            grad_scale=_cfg.PARAS_STRUCTURE['grad_scale'],
            use_total_mass=_cfg.PARAS_STRUCTURE['use_total_mass'],
            device=_cfg.DEVICE,
        )
        if _cfg.PARAS_STRUCTURE.get(
            'use_sm_cutting_force_structure', False
        ):
            shared_struc_loss_calculator = StructureLossCalculatorTorchSMForce(
                **common_structure_kwargs,
                sm_cutting_force_magnitude=_cfg.PARAS_STRUCTURE.get(
                    'sm_cutting_force_magnitude', 0.0
                ),
                sm_max_displacement_norm=_cfg.PARAS_STRUCTURE[
                    'sm_max_displacement_norm'
                ],
            )
        else:
            shared_struc_loss_calculator = StructureLossCalculatorTorch(
                **common_structure_kwargs,
                max_displacement=_cfg.PARAS_STRUCTURE['max_displacement'],
            )
        print("[BayesOpt-Constrained] Compiling _solve_cg with torch.compile (this may take a while)...")
        shared_struc_loss_calculator._solve_cg = torch.compile(
            shared_struc_loss_calculator._solve_cg,
            mode='default'
        )
        print("[BayesOpt-Constrained] Shared StructureLossCalculator ready.")

        _original_training_fn = training_fn
        main_optimize_args = [
            '--resolution', str(resolution),
            '--stl-path', effective_stl_path,
            '--pretrain-model-path', effective_pretrain_model_path,
            '--tmp-model-path', os.path.join(output_dir, 'current_trial_tmp.pth'),
            '--geometry_backend', geometry_backend,
        ]
        if geometry_artifact_path:
            main_optimize_args.extend(
                ['--geometry_artifact_path', geometry_artifact_path]
            )
        if joint_epochs is not None:
            main_optimize_args.extend(['--joint-epochs', str(joint_epochs)])
        if steps_per_epoch is not None:
            main_optimize_args.extend(
                ['--steps-per-epoch', str(steps_per_epoch)]
            )

        def training_fn(weights_dict):
            return _original_training_fn(
                weights_dict,
                shared_H_gpu=shared_H_gpu,
                shared_struc_loss_calculator=shared_struc_loss_calculator,
                joint_batch_size=joint_batch_size,
                main_optimize_args=main_optimize_args,
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
    # 只 enqueue 搜索空间中有的自由变量，固定/派生值不进 enqueue
    if len(study.trials) == 0:
        try:
            _cfg = importlib.import_module(f'configs.config_multi_field_{MODEL_NAME}')
            default_weights = _cfg.WEIGHTS
            initial_params = {k: v for k, v in default_weights.items() if k in search_space}
            if initial_params:
                _validate_enqueued_params_against_space(initial_params, search_space)
                study.enqueue_trial(initial_params)
                print(f"📌 Enqueued config default WEIGHTS as initial trial (自由变量):")
                for k, v in initial_params.items():
                    print(f"    {k}: {v}")
                print(f"  (固定约束) final_state = {FIXED_FINAL_STATE_WEIGHT}")
                print(f"  (固定约束) operation_volume = {FIXED_OPERATION_VOLUME_WEIGHT}")
                print(f"  (派生约束) SM_Collision_Free = self_support × {SM_TO_SELF_SUPPORT_RATIO}")
        except Exception as e:
            print(f"⚠️  Could not enqueue default weights: {e}")
    else:
        print(f"ℹ️  Skipping default weights enqueue (study already has {len(study.trials)} trials)")

    # --- Run Optimization ---
    print(f"\n{'#'*60}")
    print(f"Starting Constrained Bayesian Optimization")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Trials:  {n_trials}")
    print(f"  Study:   {study_name}")
    print(f"  Output:  {output_dir}")
    print(f"  DB File: {db_path}")
    print(f"  硬约束:")
    print(f"    final_state        = {FIXED_FINAL_STATE_WEIGHT} (fixed)")
    print(f"    operation_volume   = {FIXED_OPERATION_VOLUME_WEIGHT} (fixed)")
    print(f"    SM_Collision_Free  = self_support × {SM_TO_SELF_SUPPORT_RATIO} (derived)")
    print(f"  搜索自由变量 ({len(search_space)}个): {list(search_space.keys())}")
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

    best_feasible = None
    best_feasible_obj = float('inf')

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.user_attrs.get("feasible", False):
            if trial.value < best_feasible_obj:
                best_feasible_obj = trial.value
                best_feasible = trial

    best_overall = study.best_trial

    print("=" * 60)
    print("BEST OVERALL TRIAL (may violate constraints)")
    print("=" * 60)
    _print_trial_summary(best_overall, metric_targets, search_space)

    if best_feasible is not None:
        print("\n" + "=" * 60)
        print("BEST FEASIBLE TRIAL (all evaluated constraints satisfied)")
        print("=" * 60)
        _print_trial_summary(best_feasible, metric_targets, search_space)
    else:
        print("\n⚠️  No feasible trial found! Consider:")
        print("  - Relaxing BAYESIAN_METRIC_TARGETS")
        print("  - Expanding WEIGHT_SEARCH_SPACE")
        print("  - Increasing N_TRIALS")

    _save_results(study, output_dir, best_feasible, best_model_state, search_space, metric_targets)

    return study


def _print_trial_summary(trial, metric_targets, search_space):
    """Print a summary of a trial."""
    feasible = trial.user_attrs.get('feasible', False)
    print(f"  Trial #{trial.number}")
    print(f"  Minimax objective: {trial.value:.6f} {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
    print(f"  Worst term: {trial.user_attrs.get('worst_term', 'N/A')}")
    print(f"  Trial running time: {trial.user_attrs.get('running_time', 0.0):.1f}s")
    print(f"  Total elapsed time: {trial.user_attrs.get('total_elapsed_time', 0.0):.1f}s")
    print(f"\n  搜索变量 (Optuna sampled):")
    for name, value in trial.params.items():
        print(f"    {name}: {value:.6f}")
    print(f"  固定/派生权重:")
    fs_val = trial.user_attrs.get("weight_final_state", FIXED_FINAL_STATE_WEIGHT)
    ov_val = trial.user_attrs.get("weight_operation_volume", FIXED_OPERATION_VOLUME_WEIGHT)
    sm_val = trial.user_attrs.get("weight_SM_Collision_Free", None)
    print(f"    final_state: {fs_val} [fixed]")
    print(f"    operation_volume: {ov_val} [fixed]")
    if sm_val is not None:
        print(f"    SM_Collision_Free: {sm_val:.6f} [derived = self_support × {SM_TO_SELF_SUPPORT_RATIO}]")
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

    results = {
        "best_overall": {
            "trial_number": study.best_trial.number,
            "objective_value": study.best_trial.value,
            "trial_running_time": study.best_trial.user_attrs.get("running_time", 0.0),
            "total_elapsed_time": study.best_trial.user_attrs.get("total_elapsed_time", 0.0),
            "sampled_weights": study.best_trial.params,
            "full_weights": {
                k[7:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("weight_")
            },
            "feasible": study.best_trial.user_attrs.get("feasible", False),
            "losses": {k[5:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("loss_")},
            "metrics": {k[7:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("metric_")},
            "normalized": {k[5:]: v for k, v in study.best_trial.user_attrs.items() if k.startswith("norm_")},
        },
        "constraints": {
            "final_state": FIXED_FINAL_STATE_WEIGHT,
            "SM_Collision_Free": f"self_support * {SM_TO_SELF_SUPPORT_RATIO}",
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
            "sampled_weights": best_feasible.params,
            "full_weights": {
                k[7:]: v for k, v in best_feasible.user_attrs.items() if k.startswith("weight_")
            },
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
            # 搜索变量
            for name in search_space:
                row[f"w_{name}"] = trial.params.get(name, None)
            # 固定/派生权重
            row["w_final_state"] = trial.user_attrs.get("weight_final_state", FIXED_FINAL_STATE_WEIGHT)
            row["w_operation_volume"] = trial.user_attrs.get("weight_operation_volume", FIXED_OPERATION_VOLUME_WEIGHT)
            row["w_SM_Collision_Free"] = trial.user_attrs.get("weight_SM_Collision_Free", None)
            for key, val in trial.user_attrs.items():
                if key.startswith("loss_") or key.startswith("norm_"):
                    row[key] = val

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)

    print(f"📁 Trial history saved to: {csv_path}")

    best = best_feasible if best_feasible is not None else study.best_trial
    print(f"\n{'='*60}")
    print("Copy-paste ready WEIGHTS dict for your config:")
    print("=" * 60)
    print("WEIGHTS = {")
    print(f"    'final_state': {FIXED_FINAL_STATE_WEIGHT},  # [fixed by constrained optimizer]")
    print(f"    'operation_volume': {FIXED_OPERATION_VOLUME_WEIGHT},  # [fixed by constrained optimizer]")
    for name in search_space:
        val = best.params.get(name, 0.0)
        print(f"    '{name}': {val:.6f},")
    sm_val = best.user_attrs.get("weight_SM_Collision_Free", None)
    if sm_val is not None:
        print(f"    'SM_Collision_Free': {sm_val:.6f},  # [derived = self_support × {SM_TO_SELF_SUPPORT_RATIO}]")
    print("}")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Constrained Bayesian optimization for weight tuning.\n"
                    f"硬约束: final_state={FIXED_FINAL_STATE_WEIGHT} (fixed), "
                    f"operation_volume={FIXED_OPERATION_VOLUME_WEIGHT} (fixed), "
                    f"SM_Collision_Free = self_support × {SM_TO_SELF_SUPPORT_RATIO} (derived)\n"
                    f"自由搜索变量 (3个): self_support, AM_Collision_Free, structure"
    )
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
        help=(
            "Output directory. Defaults to "
            "model_data/trained_fields/<model>/bayesian_optimization."
        ),
    )
    parser.add_argument(
        "--model_name", type=str, default=MODEL_NAME,
        help=f"Model name to use (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=(
            "Optional JOINT_BATCH_CONFIG['batch_size'] override. "
            "By default, use the model config."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=100,
        help="Voxel resolution used by the geometry representation (default: 100)."
    )
    parser.add_argument(
        "--pretrain-model-path", "--pretrain_model_path",
        dest="pretrain_model_path", default=None,
        help=(
            "Pretrained checkpoint passed to main_optimize.py. Defaults to "
            "model_data/pretrained_fields/<model>/<model>_pretrained_<resolution>.pth."
        ),
    )
    parser.add_argument(
        "--stl-path", "--stl_path", dest="stl_path", default=None,
        help=(
            "Preprocessed support STL. Defaults to "
            "model_data/preprocessed/<model>/<model>_support.stl."
        ),
    )
    parser.add_argument(
        "--joint-epochs", "--joint_epochs", dest="joint_epochs",
        type=int, default=None,
        help="Optional main_optimize.py epoch override; default uses the model config.",
    )
    parser.add_argument(
        "--steps-per-epoch", "--steps_per_epoch", dest="steps_per_epoch",
        type=int, default=None,
        help="Optional main_optimize.py steps-per-epoch override; default uses the model config.",
    )
    add_geometry_backend_args(parser)

    args = parser.parse_args()
    for option, value in (
        ('--n_trials', args.n_trials),
        ('--resolution', args.resolution),
        ('--batch_size', args.batch_size),
        ('--joint-epochs', args.joint_epochs),
        ('--steps-per-epoch', args.steps_per_epoch),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be a positive integer")

    # Apply model name and update dependent configs
    MODEL_NAME = args.model_name

    # Load configuration dynamically based on MODEL_NAME
    try:
        cfg = importlib.import_module(f'configs.config_multi_field_{MODEL_NAME}')
        raw_search_space = cfg.BAYESIAN_WEIGHT_SEARCH_SPACE
        metric_targets = cfg.BAYESIAN_METRIC_TARGETS
    except ImportError as e:
        print(f"❌ Error loading config for model '{MODEL_NAME}': {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"❌ Config for model '{MODEL_NAME}' is missing BAYESIAN_WEIGHT_SEARCH_SPACE or BAYESIAN_METRIC_TARGETS: {e}")
        sys.exit(1)

    # 构建约束搜索空间（过滤掉 SM_Collision_Free）
    try:
        search_space = build_constrained_search_space(raw_search_space)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 命名（加 _constrained 后缀与原脚本区分）
    study_name = f"weight_opt_{MODEL_NAME}_constrained"

    if args.output_dir is None:
        args.output_dir = os.path.join(
            'model_data',
            'trained_fields',
            MODEL_NAME,
            'bayesian_optimization',
        )

    if args.dry_run:
        print("🧪 DRY RUN MODE: Using dummy training function")
        training_fn = _dummy_run_training
    else:
        training_fn = run_training
    print(f"Using joint batch size: {args.batch_size}")
    print(f"搜索空间（约束后，{len(search_space)}个自由变量）: {list(search_space.keys())}")
    print(f"固定约束: final_state = {FIXED_FINAL_STATE_WEIGHT}")
    print(f"固定约束: operation_volume = {FIXED_OPERATION_VOLUME_WEIGHT}")
    print(f"派生约束: SM_Collision_Free = self_support × {SM_TO_SELF_SUPPORT_RATIO}")

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
        resolution=args.resolution,
        pretrain_model_path=args.pretrain_model_path,
        stl_path=args.stl_path,
        joint_epochs=args.joint_epochs,
        steps_per_epoch=args.steps_per_epoch,
    )
