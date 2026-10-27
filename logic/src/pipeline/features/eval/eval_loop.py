"""Inner evaluation loop and dataset/eval kwargs for the eval engine.

Moved from ``logic.src.pipeline.features.eval.engine``.
"""

import contextlib
from typing import Any, Dict, List, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.features.eval.evaluate import evaluate_policy, get_automatic_batch_size
from logic.src.utils.functions import move_to


def _build_dataset_kwargs(cfg: Config) -> Dict[str, Any]:
    """Build keyword arguments for make_dataset from Config.

    Args:
        cfg: Configuration object containing evaluation parameters.

    Returns:
        Dictionary of keyword arguments for make_dataset.
    """
    ev = cfg.eval
    env_cfg = getattr(cfg, cfg.task, cfg.eval).env
    graph = (
        env_cfg.curriculum_graphs[0]
        if env_cfg.curriculum_graphs
        else (env_cfg.eval_graphs[0] if env_cfg.eval_graphs else None)
    )
    if graph is None:
        raise ValueError("No graph configuration found in environment config.")
    return {
        "num_samples": ev.val_size,
        "offset": ev.offset,
        "distribution": ev.data_distribution,
        "vertex_strat": graph.vertex_method,
        "size": graph.num_loc,
        "focus_graph": graph.focus_graph,
        "focus_size": graph.focus_size,
        "area": graph.area,
        "dist_matrix_path": graph.dm_filepath,
        "number_edges": graph.edge_threshold,
        "waste_weight": graph.reward.waste_weight if graph and getattr(graph, "reward", None) else 1.0,
        "dist_strat": graph.distance_method,
        "edge_strat": graph.edge_method,
    }


def _build_eval_kwargs(cfg: Config) -> Dict[str, Any]:
    """Build keyword arguments for evaluate_policy from Config.

    Args:
        cfg: Configuration object containing evaluation parameters.

    Returns:
        Dictionary of keyword arguments for evaluate_policy.
    """
    ev = cfg.eval
    result: Dict[str, Any] = {
        "eval_batch_size": ev.eval_batch_size,
        "no_progress_bar": cfg.tracking.no_progress_bar,
        "compress_mask": ev.compress_mask,
        "max_calc_batch_size": ev.max_calc_batch_size,
    }
    if ev.decoding:
        result["decoding"] = {
            "strategy": ev.decoding.strategy,
            "temperature": ev.decoding.temperature,
            "beam_width": ev.decoding.beam_width,
        }
    return result


def _eval_dataset(
    model: Any,
    dataset: Dataset,
    beam_width: int,
    softmax_temp: float,
    cfg: Config,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Inner evaluation loop for a single batch/process.

    Args:
        model: Trained model to evaluate.
        dataset: Dataset to evaluate on.
        beam_width: Beam width for decoding.
        softmax_temp: Softmax temperature for decoding.
        cfg: Configuration object.
        device: Device to evaluate on.

    Returns:
        List of evaluation results.
    """
    ev = cfg.eval
    strategy = ev.decoding.strategy if ev.decoding else "greedy"

    model.to(device)
    model.eval()
    if hasattr(model, "set_strategy"):
        model.set_strategy(strategy, temp=softmax_temp)

    eval_kwargs = _build_eval_kwargs(cfg)
    eval_batch_size = eval_kwargs.get("eval_batch_size", 1024)
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    method = strategy
    if method == "sample":
        method = "sampling"

    # Automatic batch size tuning
    auto_batch = False  # Could be added to EvalConfig in the future
    if auto_batch:
        eval_batch_size = get_automatic_batch_size(
            model,
            model.problem,
            dataloader,
            method=method,
            initial_batch_size=eval_batch_size,
            samples=beam_width,
        )
        dataloader = DataLoader(dataset, batch_size=eval_batch_size, pin_memory=True)

    eval_results = evaluate_policy(
        model,
        model.problem,
        dataloader,
        method=method,
        return_results=True,
        samples=beam_width,
        softmax_temperature=softmax_temp,
        device=device,
        **eval_kwargs,
    )

    with contextlib.suppress(Exception):
        from logic.src.tracking.logging.visualization.heatmaps import (
            maybe_log_eval_attention_heatmaps,
        )

        first_batch = move_to(next(iter(dataloader)), device)
        maybe_log_eval_attention_heatmaps(model, first_batch, cfg, output_subdir="eval_attention")

    # Evaluators return rewards (PTP: profit, higher is better); problem.get_costs and
    # the reported "cost" use the opposite sign.
    rewards_best = eval_results["rewards"]
    sequences_best = eval_results["sequences"].cpu().numpy()
    duration_per_batch = eval_results["duration"] / len(dataloader)

    results: List[Dict[str, Any]] = []
    for i, (seq, reward) in enumerate(zip(sequences_best, rewards_best, strict=False)):
        batch_i: Any = {}
        if seq is not None:
            if model.problem.NAME in ("cptp", "cptptc"):
                # Multi-trip problems: keep the trailing depot-return marker
                # (cptptc is single-vehicle-multi-trip, same shape as cptp's
                # per-trip capacity resets -- see logic/src/envs/cptptc.py)
                seq = np.trim_zeros(seq).tolist() + [0]
            elif model.problem.NAME == "ptp":
                seq = np.trim_zeros(seq).tolist()
            else:
                seq = None
                raise AssertionError("Unknown problem: {}".format(model.problem.NAME))

        if seq is not None:
            seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
            instance = dataset[i]
            instance_obj: object = instance
            if isinstance(instance_obj, (list, tuple)):
                batch_i = {k: v.unsqueeze(0).to(device) for k, v in zip(["locs"], instance_obj, strict=False)}
            elif isinstance(instance_obj, ITraversable):
                batch_i = {}
                for k, v in instance_obj.items():
                    if torch.is_tensor(v):
                        batch_i[k] = v.unsqueeze(0).to(device)
                    elif isinstance(v, np.ndarray):
                        batch_i[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                    else:
                        batch_i[k] = v
            else:
                batch_i = cast(torch.Tensor, instance).unsqueeze(0).to(device)

            dm_input = (
                batch_i.get("dist_matrix", batch_i.get("dm", batch_i.get("dist", None)))
                if isinstance(batch_i, dict)
                else None
            )
            if hasattr(model.problem, "evaluate_tour"):
                c_dict = model.problem.evaluate_tour(batch_i, seq_tensor, dist_matrix=dm_input)
            else:
                _, c_dict, _ = model.problem.get_costs(batch_i, seq_tensor, None, dm_input)
        else:
            c_dict = {
                "length": torch.tensor(0.0),
                "waste": torch.tensor(0.0),
                "overflows": torch.tensor(0.0),
            }

        bin_cap_kg = float(batch_i.get("bin_capacity_kg", 47.5)) if isinstance(batch_i, dict) else 47.5
        result = {
            "cost": -float(reward),
            "reward": float(reward),
            "seq": seq,
            "duration": duration_per_batch,
            "km": c_dict["length"].item(),
            "kg": c_dict["waste"].item() * bin_cap_kg,
            "overflows": c_dict["overflows"].item(),
        }
        # Additive KPI: only present when the problem tracks it (e.g. cptptc's
        # "time" key from get_costs), so this stays plug-and-play for future
        # problem-specific KPIs without a fixed schema here.
        if "time" in c_dict:
            result["time"] = c_dict["time"].item()
        results.append(result)
    return results
