"""Optuna objective, worker entry, and metric helpers for simulation HPO.

Moved from ``logic.src.pipeline.simulations.hpo.hpo_handler``.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import optuna
import torch
from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.constants import ROOT_DIR, SIM_METRICS
from logic.src.pipeline.features.test.config import expand_policy_configs
from logic.src.pipeline.simulations.hpo.base import PolicyHPOBase
from logic.src.pipeline.simulations.manager import sequential_simulations
from logic.src.pipeline.simulations.repository import (
    load_indices,
)
from logic.src.utils.infrastructure.setup_sims import get_graph_config

logger = logging.getLogger("HPO_Handler")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def _metric_direction(metric_name: str) -> str:
    """Return the Optuna direction string for a given metric name.

    Args:
        metric_name: Metric identifier (e.g. 'profit', 'overflows').

    Returns:
        str: 'minimize' or 'maximize'.
    """
    return "minimize" if metric_name in _MINIMISE_METRICS else "maximize"


def _extract_metric(log: Dict, metric_name: str) -> float:
    """Extract a scalar metric value from a simulation log dict.

    Args:
        log: Output of sequential_simulations keyed by policy name.
        metric_name: Metric to extract (must be in SIM_METRICS).

    Returns:
        float: Metric value, or -inf / +inf depending on direction when
            the log is empty or the metric is not found.
    """
    direction = _metric_direction(metric_name)
    fallback = float("inf") if direction == "minimize" else float("-inf")

    if metric_name in SIM_METRICS:
        idx = SIM_METRICS.index(metric_name)
    elif "profit" in SIM_METRICS:
        idx = SIM_METRICS.index("profit")
    else:
        return fallback

    pol_name = list(log.keys())[0] if log else None
    if pol_name and log[pol_name]:
        return float(log[pol_name][idx])

    return fallback


def objective(  # noqa: C901
    trial: optuna.Trial,
    base_cfg: Config,
    data_size: int,
    lock: Any,
) -> Union[float, Tuple[float, ...]]:
    """Optuna objective function for simulation policy HPO.

    Supports both single-objective and multi-objective modes, controlled by
    ``base_cfg.hpo_sim.metrics`` (a list of metric names).  When the list has
    exactly one entry the function returns a scalar; otherwise it returns a
    tuple so Optuna can compute the Pareto front.

    Parameter names passed to ``trial.suggest_*`` are always the *short* names
    from the search space definition (without the full config path prefix).
    The mapping to config paths is stored in ``trial.set_user_attr`` for
    auditability and study resumption safety.

    Args:
        trial: The Optuna trial object for parameter sampling.
        base_cfg: The base configuration; cloned per trial.
        data_size: Number of available bins for the selected area.
        lock: Multiprocessing lock for repository access.

    Returns:
        Union[float, Tuple[float, ...]]: Metric value(s) for this trial.
    """
    hpo_sim = base_cfg.hpo_sim
    policy_name = hpo_sim.policy_name

    # Resolve metric list (backwards-compatible with scalar 'metric' field).
    metrics: List[str] = list(getattr(hpo_sim, "metrics", None) or [getattr(hpo_sim, "metric", "profit")])
    is_multi_objective = len(metrics) > 1
    search_space: Dict[str, Any] = dict(hpo_sim.search_space)

    # -----------------------------------------------------------------
    # 1. Build param→config-path mapping and sample from search space.
    # -----------------------------------------------------------------
    # Short names are used with trial.suggest_* so Optuna's internal state
    # (which drives TPE/CMA-ES surrogate models) is consistent across study
    # resumptions.  The full config path is stored as a trial attribute.
    param_path_map: Dict[str, str] = {}
    params_by_path: Dict[str, Any] = {}

    for short_name, spec in search_space.items():
        full_path = short_name if "." in short_name else f"sim.full_policies.0.{policy_name}.{short_name}"
        param_path_map[short_name] = full_path

        value = PolicyHPOBase.suggest_param(trial, short_name, spec)
        params_by_path[full_path] = value

    # Store the mapping for auditability (visible in Optuna dashboard / DB).
    trial.set_user_attr("param_path_map", param_path_map)

    # -----------------------------------------------------------------
    # 2. Clone config and apply sampled params.
    # -----------------------------------------------------------------
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    trial_cfg = OmegaConf.create(cfg_dict)

    # Ensure full_policies[0] contains the target policy as a dict.
    if not trial_cfg.sim.full_policies:
        trial_cfg.sim.full_policies = [{policy_name: {}}]
    elif (
        isinstance(trial_cfg.sim.full_policies[0], str)
        or isinstance(trial_cfg.sim.full_policies[0], dict)
        and policy_name not in trial_cfg.sim.full_policies[0]
    ):
        trial_cfg.sim.full_policies[0] = {policy_name: {}}

    for path, val in params_by_path.items():
        OmegaConf.update(trial_cfg, path, val)

    # -----------------------------------------------------------------
    # 3. Synchronise graph / simulation settings.
    # -----------------------------------------------------------------
    graph = get_graph_config(trial_cfg.sim)
    graph.num_loc = hpo_sim.graph.num_loc
    graph.area = hpo_sim.graph.area
    graph.waste_type = hpo_sim.graph.waste_type
    graph.n_days = hpo_sim.graph.n_days
    graph.n_samples = hpo_sim.graph.n_samples

    if hasattr(graph, "edge_threshold"):
        try:
            graph.edge_threshold = int(graph.edge_threshold)
        except (ValueError, TypeError):
            graph.edge_threshold = 0

    trial_cfg.sim.graph = graph
    trial_cfg.sim.policies = [policy_name]
    expand_policy_configs(trial_cfg)  # type: ignore[arg-type]

    # -----------------------------------------------------------------
    # 4. Resolve instance indices and device.
    # -----------------------------------------------------------------
    n_samples = hpo_sim.graph.n_samples
    num_loc = hpo_sim.graph.num_loc

    if data_size != num_loc:
        focus_graph = trial_cfg.sim.graph.focus_graph
        if not focus_graph:
            wtype_suffix = f"_{hpo_sim.graph.waste_type}" if hpo_sim.graph.waste_type else ""
            focus_graph = f"graphs_{hpo_sim.graph.area}_{num_loc}V_{n_samples}N{wtype_suffix}.json"

        indices = load_indices(focus_graph, n_samples, num_loc, data_size, lock)
        if len(indices) == 1:
            indices = [indices[0]] * n_samples
    else:
        indices = [None] * n_samples  # type: ignore[assignment]

    sample_idx_ls = [list(range(n_samples))]
    device = torch.device(getattr(trial_cfg, "device", "cpu"))

    # -----------------------------------------------------------------
    # 5. Execute simulation (iterative or black-box).
    # -----------------------------------------------------------------
    try:
        # Check for iterative support if a specialized handler is available.
        iterative_callback = None
        if hpo_sim.method.lower() in ("nsgaii", "tpe", "cmaes") and not is_multi_objective:
            # Only enable iterative pruning for single-objective for now to avoid complexity
            def _iterative_callback(day: int, cum_metrics: Dict[str, float], s_id: int) -> None:
                # To maintain step consistency across trials, we only report for the first sample
                if s_id != sample_idx_ls[0][0]:
                    return

                # Report the primary metric
                val = cum_metrics.get(metrics[0], 0.0)
                trial.report(val, step=day)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            iterative_callback = _iterative_callback

        # Standard execution via sequential_simulations.
        # Note: sequential_simulations is currently a black-box. To support
        # Hyperband pruning effectively, it would need to yield metrics per day.
        log, _, _ = sequential_simulations(
            cfg=trial_cfg,  # type: ignore[arg-type]
            device=device,
            indices_ls=indices,
            sample_idx_ls=sample_idx_ls,
            model_weights_path="",
            lock=lock,
            callback=cast(Any, iterative_callback),  # sequential_simulations callback stub omits the day
        )

        if not log:
            raise ValueError("Simulation returned no logs.")

        # -----------------------------------------------------------------
        # 6. Extract results.
        # -----------------------------------------------------------------
        values = tuple(_extract_metric(log, m) for m in metrics)

        return values[0] if len(values) == 1 else values

    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned() from e


def worker(
    study_name: str,
    storage_url: str,
    base_cfg_yaml: str,
    data_size: int,
    n_trials: int,
    lock: Any,
) -> None:
    """Entry point for each HPO worker process.

    Each process loads the shared Optuna study from the persistent storage
    backend and independently contributes trials.  This avoids GIL contention
    entirely because each process runs its own Python interpreter.

    Args:
        study_name: Name of the Optuna study to load.
        storage_url: SQLAlchemy URL for the shared storage backend.
        base_cfg_yaml: Serialised OmegaConf YAML of the base Config.
            Passed as a string so the Config can be reconstructed inside the
            forked process without pickling issues.
        data_size: Number of available bins for the selected area.
        n_trials: Number of trials this worker should contribute.
        lock: Shared multiprocessing lock for repository access.
    """
    # Silence per-worker Optuna INFO logs to avoid interleaved console noise.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"[DEBUG] worker: study_name={study_name}, n_trials={n_trials}")
    base_cfg = OmegaConf.create(base_cfg_yaml)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        lambda trial: objective_debug(trial, base_cfg, data_size, lock),
        n_trials=n_trials,
    )


def objective_debug(trial, base_cfg, data_size, lock):
    print(f"[DEBUG] objective start: trial={trial.number}")
    try:
        res = objective(trial, base_cfg, data_size, lock)
        print(f"[DEBUG] objective end: trial={trial.number}, res={res}")
        return res
    except Exception as e:
        print(f"[DEBUG] objective error: trial={trial.number}, error={e}")
        import traceback

        traceback.print_exc()
        raise


def _export_hpo_visualization_reports(study: optuna.Study) -> None:
    """Write Optuna Plotly reports to ``assets/hpo_reports/`` when possible."""
    from logic.src.pipeline.simulations.hpo.hpo_reports import export_optuna_study_reports

    reports_parent = os.path.join(str(ROOT_DIR), "assets", "hpo_reports")
    report_dir = export_optuna_study_reports(study, output_dir=reports_parent)
    if report_dir:
        logger.info(f"HPO visualization reports: {report_dir}")


_MINIMISE_METRICS = {"overflows", "kg_lost", "length"}


def _select_pareto_representative(
    pareto_trials: List[optuna.trial.FrozenTrial],
    directions: List[str],
) -> Optional[optuna.trial.FrozenTrial]:
    """Select a single representative trial from a Pareto front.

    Uses a normalised L1 distance to the ideal point (best value per
    objective across all Pareto trials).  Minimisation objectives are
    negated before distance computation so that 'closer to ideal'
    always means 'smaller L1 distance'.

    Args:
        pareto_trials: Trials on the Pareto front.
        directions: Per-objective directions ('minimize'/'maximize').

    Returns:
        Optional[optuna.trial.FrozenTrial]: The representative trial, or None
            if pareto_trials is empty.
    """
    if not pareto_trials:
        return None
    if len(pareto_trials) == 1:
        return pareto_trials[0]

    n_obj = len(directions)

    # Collect objective values, sign-flipped so that 'more positive = better'.
    signed: List[List[float]] = []
    for t in pareto_trials:
        row = []
        for _j, (v, d) in enumerate(zip(t.values, directions, strict=False)):  # pyrefly: ignore [bad-argument-type]
            row.append(-v if d == "minimize" else v)
        signed.append(row)

    # Ideal point: max per objective in signed space.
    ideal = [max(row[j] for row in signed) for j in range(n_obj)]
    nadir = [min(row[j] for row in signed) for j in range(n_obj)]
    ranges = [max(ideal[j] - nadir[j], 1e-9) for j in range(n_obj)]

    # Normalised L1 distance from each trial to the ideal point.
    distances = [sum(abs(ideal[j] - row[j]) / ranges[j] for j in range(n_obj)) for row in signed]

    best_idx = distances.index(min(distances))
    return pareto_trials[best_idx]
