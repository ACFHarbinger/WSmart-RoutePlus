"""
Evaluation engine for WSmart-Route.

Attributes:
    get_best: Get the best sequence and cost from a list of sequences and costs.
    _build_dataset_kwargs: Build keyword arguments for make_dataset from Config.
    _build_eval_kwargs: Build keyword arguments for evaluate_policy from Config.
    eval_dataset_mp: Worker function for multiprocessing evaluation.
    _eval_dataset: Inner evaluation loop for a single batch/process.

Example:
    >>> from logic.src.configs import Config
    >>> from logic.src.pipeline.features.eval.engine import get_best, _build_dataset_kwargs, _build_eval_kwargs, eval_dataset_mp, _eval_dataset
    >>>
    >>> # Example for get_best
    >>> sequences = np.array([[1, 2, 3], [1, 3, 2]])
    >>> cost = np.array([10, 5])
    >>> best_sequences, best_costs = get_best(sequences, cost)
    >>> print(f"Best sequence: {best_sequences}")
    >>> print(f"Best cost: {best_costs}")
    >>>
    >>> # Example for _build_dataset_kwargs
    >>> cfg = Config()
    >>> dataset_kwargs = _build_dataset_kwargs(cfg)
    >>> print(f"Dataset kwargs: {dataset_kwargs}")
    >>>
    >>> # Example for _build_eval_kwargs
    >>> eval_kwargs = _build_eval_kwargs(cfg)
    >>> print(f"Evaluation kwargs: {eval_kwargs}")
    >>>
    >>> # Example for eval_dataset_mp (simplified - requires actual dataset and model)
    >>> # This would typically be run in a multiprocessing context
    >>> # dataset_path = "path/to/dataset.pkl"
    >>> # beam_width = 5
    >>> # softmax_temp = 1.0
    >>> # results = eval_dataset_mp((dataset_path, beam_width, softmax_temp, cfg, 0, 1))
    >>> # print(f"Evaluation results: {results}")
    >>>
    >>> # Example for _eval_dataset (simplified - requires actual dataset, model, and device)
    >>> # This would typically be run in a multiprocessing context
    >>> # model = load_model("path/to/model.pth")
    >>> # dataset = model.problem.make_dataset(filename="path/to/dataset.pkl", num_samples=100)
    >>> # results = _eval_dataset(model, dataset, 5, 1.0, cfg, torch.device("cuda:0"))
    >>> # print(f"Evaluation results: {results}")
"""

import contextlib
import datetime
import itertools
import math
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from logic.src.configs import Config
from logic.src.interfaces import ITraversable
from logic.src.pipeline.features.eval.evaluate import evaluate_policy, get_automatic_batch_size
from logic.src.utils.data.loader import save_dataset
from logic.src.utils.functions import move_to
from logic.src.utils.model.loader import load_model

mp = torch.multiprocessing.get_context("spawn")


def get_best(
    sequences: np.ndarray,
    cost: np.ndarray,
    ids: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc

    Args:
        sequences: All found solutions.
        cost: Costs of all found solutions.
        ids: IDs of the solutions (should be [0, 0, ..., 1, 1, ...]).
        batch_size: Expected batch size (number of instances).

    Returns:
        Tuple of (best_sequences, best_costs).
    """
    if ids is None:
        idx = cost.argmin()
        return [sequences[idx]], [float(cost[idx])]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)
    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]
    return [sequences[i] if i >= 0 else None for i in result], [float(cost[i]) if i >= 0 else math.inf for i in result]


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


def eval_dataset_mp(
    args: Tuple[str, int, float, Config, int, int],
) -> List[Dict[str, Any]]:
    """Worker function for multiprocessing evaluation.

    Args:
        args: Tuple of (dataset_path, beam_width, softmax_temp, cfg, process_id, num_processes).

    Returns:
        List of evaluation results.
    """
    (dataset_path, beam_width, softmax_temp, cfg, i, num_processes) = args
    ev = cfg.eval
    model_path = ""
    if ev.policy and ev.policy.model:
        model_path = ev.policy.model.load_path or ""

    model_raw, _ = load_model(model_path)
    model = cast(Any, model_raw)
    val_size = ev.val_size // num_processes

    ds_kwargs = _build_dataset_kwargs(cfg)
    ds_kwargs["num_samples"] = val_size
    ds_kwargs["offset"] = ev.offset + val_size * i

    dataset = model.problem.make_dataset(filename=dataset_path, **ds_kwargs)
    device = torch.device("cuda:{}".format(i))
    return _eval_dataset(model, dataset, beam_width, softmax_temp, cfg, device)


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
        **eval_kwargs,
    )

    with contextlib.suppress(Exception):
        from logic.src.tracking.logging.visualization.heatmaps import (
            maybe_log_eval_attention_heatmaps,
        )

        first_batch = move_to(next(iter(dataloader)), device)
        maybe_log_eval_attention_heatmaps(model, first_batch, cfg, output_subdir="eval_attention")

    costs_best = eval_results["rewards"]
    sequences_best = eval_results["sequences"].cpu().numpy()
    duration_per_batch = eval_results["duration"] / len(dataloader)

    results: List[Dict[str, Any]] = []
    for i, (seq, cost) in enumerate(zip(sequences_best, costs_best, strict=False)):
        if seq is not None:
            if model.problem.NAME in ("cvrpp", "cwcvrp", "sdwcvrp", "ttop"):
                # Multi-trip problems: keep the trailing depot-return marker
                # (ttop is single-vehicle-multi-trip, same shape as cvrpp's
                # per-trip capacity resets -- see logic/src/envs/tasks/ttop.py)
                seq = np.trim_zeros(seq).tolist() + [0]
            elif model.problem.NAME in ("vrpp", "wcvrp"):
                seq = np.trim_zeros(seq).tolist()
            else:
                seq = None
                raise AssertionError("Unknown problem: {}".format(model.problem.NAME))

        if seq is not None:
            seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
            instance = dataset[i]
            instance_obj: object = instance
            batch_i: Any
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

            _, c_dict, _ = model.problem.get_costs(
                batch_i, seq_tensor, None, batch_i.get("dist_matrix") if isinstance(batch_i, dict) else None
            )
        else:
            c_dict = {
                "length": torch.tensor(0.0),
                "waste": torch.tensor(0.0),
                "overflows": torch.tensor(0.0),
            }

        result = {
            "cost": float(cost),
            "seq": seq,
            "duration": duration_per_batch,
            "km": c_dict["length"].item(),
            "kg": c_dict["waste"].item() * 100,
            "overflows": c_dict["overflows"].item(),
        }
        # Additive KPI: only present when the problem tracks it (e.g. ttop's
        # "time" key from get_costs), so this stays plug-and-play for future
        # problem-specific KPIs without a fixed schema here.
        if "time" in c_dict:
            result["time"] = c_dict["time"].item()
        results.append(result)
    return results


def eval_dataset(
    dataset_path: str,
    beam_width: int,
    softmax_temp: float,
    cfg: Config,
    method: Optional[str] = None,
    strategy: Optional[str] = None,
    run: Optional[Any] = None,
) -> Tuple[List[float], List[Optional[List[int]]], List[float]]:
    """Evaluates a model on a given dataset.

    Args:
        dataset_path: Path to the evaluation dataset.
        beam_width: Beam width for beam-search decoding.
        softmax_temp: Softmax temperature for sampling.
        cfg: Root Hydra configuration.
        method: Decoding method override.
        strategy: Decoding strategy override.
        run: Optional WSTracker :class:`Run` for metric/artifact logging.

    Returns:
        Tuple of (costs, tours, durations).
    """
    ev = cfg.eval
    model_path = ""
    if ev.policy:
        if hasattr(ev.policy, "load_path") and ev.policy.load_path:
            model_path = ev.policy.load_path
        elif hasattr(ev.policy, "model") and ev.policy.model and ev.policy.model.load_path:
            model_path = ev.policy.model.load_path or ""

    model_raw, _ = load_model(str(model_path))
    model = cast(Any, model_raw)
    use_cuda = torch.cuda.is_available() and not getattr(ev, "no_cuda", False)
    if getattr(cfg, "device", None) == "cpu":
        use_cuda = False

    # Resolve strategy
    # resolved_strategy = strategy or method or (ev.decoding.strategy if ev.decoding else "greedy")

    if ev.multiprocessing:
        results = _eval_multiprocessing(dataset_path, beam_width, softmax_temp, cfg)
    else:
        results = _eval_singleprocess(model, dataset_path, beam_width, softmax_temp, cfg, use_cuda)

    parallelism: int = ev.eval_batch_size
    avg_metrics = {
        "cost": np.mean([r["cost"] for r in results]),
        "std": np.std([r["cost"] for r in results]),
        "km": np.mean([r["km"] for r in results]),
        "kg": np.mean([r["kg"] for r in results]),
        "overflows": np.mean([r["overflows"] for r in results]),
    }

    logger.info("Average cost: {} +- {}".format(avg_metrics["cost"], avg_metrics["std"]))
    logger.info(
        "Average KM: {}, Average KG: {}, Average Overflows: {}".format(
            avg_metrics["km"], avg_metrics["kg"], avg_metrics["overflows"]
        )
    )

    costs: List[float] = [float(r["cost"]) for r in results]
    tours: List[Optional[List[int]]] = [r["seq"] for r in results]
    durations: List[float] = [float(r["duration"]) for r in results]
    logger.info(
        "Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations)))
    )
    logger.info("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    logger.info(
        "Calculated total duration: {}".format(datetime.timedelta(seconds=int(np.sum(durations) / parallelism)))
    )

    model_name = "_".join(os.path.normpath(os.path.splitext(model_path or "model")[0]).split(os.sep)[-2:])
    out_file = ev.output_filename or _get_eval_output_path(
        model, dataset_path, cfg, model_name, beam_width, softmax_temp, len(costs)
    )

    assert ev.overwrite or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    # Log evaluation metrics and artifact to WSTracker run
    if run is not None:
        ds_name = os.path.splitext(os.path.basename(dataset_path))[0]
        prefix = f"eval/{ds_name}/bw{beam_width}"
        with contextlib.suppress(Exception):
            run.log_metric(f"{prefix}/avg_cost", float(avg_metrics["cost"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/std_cost", float(avg_metrics["std"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_km", float(avg_metrics["km"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_kg", float(avg_metrics["kg"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_overflows", float(avg_metrics["overflows"]))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/avg_duration", float(np.mean(durations)))  # type: ignore[arg-type]
            run.log_metric(f"{prefix}/n_samples", float(len(costs)))  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            run.log_artifact(out_file, artifact_type="eval_results")
        with contextlib.suppress(Exception):
            run.log_dataset_event(  # type: ignore[union-attr]
                "load",
                file_path=dataset_path,
                num_samples=len(costs),
                metadata={
                    "event": "eval_dataset",
                    "beam_width": beam_width,
                    "strategy": strategy or method or "",
                    "variable_name": "dataset_path",
                    "source_file": "features/eval/engine.py",
                    "source_line": 299,
                },
            )

    return costs, tours, durations


def _eval_multiprocessing(dataset_path: str, beam_width: int, softmax_temp: float, cfg: Config) -> List[Dict[str, Any]]:
    """Helper for multiprocessing evaluation.

    Args:
        dataset_path: Path to dataset.
        beam_width: Beam width for decoding.
        softmax_temp: Softmax temperature for decoding.
        cfg: Configuration object.

    Returns:
        List of evaluation results.
    """
    num_processes = torch.cuda.device_count()
    assert cfg.eval.val_size % num_processes == 0, "val_size must be divisible by num_processes"

    with mp.Pool(num_processes) as pool:
        return list(
            itertools.chain.from_iterable(
                pool.map(
                    eval_dataset_mp,
                    [(dataset_path, beam_width, softmax_temp, cfg, i, num_processes) for i in range(num_processes)],
                )
            )
        )


def _eval_singleprocess(
    model: Any, dataset_path: str, beam_width: int, softmax_temp: float, cfg: Config, use_cuda: bool
) -> List[Dict[str, Any]]:
    """Helper for single-process evaluation.

    Args:
        model: Model to evaluate.
        dataset_path: Path to dataset.
        beam_width: Beam width for decoding.
        softmax_temp: Softmax temperature for decoding.
        cfg: Configuration object.
        use_cuda: Whether to use CUDA.

    Returns:
        List of evaluation results.
    """
    device = torch.device("cpu" if not use_cuda else "cuda:0")

    ds_kwargs = _build_dataset_kwargs(cfg)
    dataset = model.problem.make_dataset(filename=dataset_path, **ds_kwargs)
    return _eval_dataset(model, dataset, beam_width, softmax_temp, cfg, device)


def _get_eval_output_path(
    model: Any,
    dataset_path: str,
    cfg: Config,
    model_name: str,
    beam_width: int,
    softmax_temp: float,
    num_costs: int,
) -> str:
    """Generate evaluation result output path.

    Args:
        model: Model used for evaluation.
        dataset_path: Path to dataset.
        cfg: Configuration object.
        model_name: Name of the model.
        beam_width: Beam width for decoding.
        softmax_temp: Softmax temperature for decoding.
        num_costs: Number of costs to evaluate.

    Returns:
        Path to evaluation results file.
    """
    ev = cfg.eval
    strategy = ev.decoding.strategy if ev.decoding else "greedy"
    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    results_dir = os.path.join(ev.results_dir, model.problem.NAME, dataset_basename)
    os.makedirs(results_dir, exist_ok=True)

    return os.path.join(
        results_dir,
        "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename,
            model_name,
            strategy,
            beam_width if strategy != "greedy" else "",
            softmax_temp,
            ev.offset,
            ev.offset + num_costs,
            ext,
        ),
    )
