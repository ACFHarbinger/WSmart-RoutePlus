"""Post-processing and aggregation of simulation results.

Attributes:
    _log_sim_metrics: Log simulation metrics to WSTracker.
    aggregate_final_results: Aggregate final results from all simulation samples.

Example:
    >>> from logic.src.pipeline.features.test.orchestrator.results_handler import aggregate_final_results
    >>> log, log_std = aggregate_final_results(log_tmp, cfg, lock)
"""

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import logic.src.constants as udef
from logic.src.configs import Config
from logic.src.tracking.logging.log_utils import output_stats
from logic.src.pipeline.simulations.day_context import policy_result_key

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]


def _log_sim_metrics(log: Dict[str, Any], log_std: Optional[Dict[str, Any]] = None) -> None:
    """Forward aggregated per-policy metrics to the active WSTracker run.

    Args:
        log: Description of log.
        log_std: Description of log_std.
    """
    run = get_active_run() if get_active_run is not None else None
    if run is None:
        return
    for pol_name_k, metrics in log.items():
        metrics_obj: object = metrics
        if isinstance(metrics_obj, (list, tuple)):
            for metric_name, val in zip(udef.SIM_METRICS, metrics_obj, strict=False):
                run.log_metric(f"sim/{pol_name_k}/{metric_name}", float(val))
    if log_std is not None:
        for pol_name_k, std_metrics in log_std.items():
            std_metrics_obj: object = std_metrics
            if isinstance(std_metrics_obj, (list, tuple)):
                for metric_name, val in zip(udef.SIM_METRICS, std_metrics_obj, strict=False):
                    run.log_metric(f"sim/{pol_name_k}/{metric_name}_std", float(val))


def aggregate_final_results(log_tmp: Any, cfg: Config, lock: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Aggregate results from all finished simulation samples.

    Args:
        log_tmp: Manager dict of policy -> list of metric lists.
        cfg: Root configuration.
        lock: Multiprocessing lock.

    Returns:
        Tuple containing:
            - log (Dict[str, Any]): Aggregated simulation results.
            - log_std (Optional[Dict[str, Any]]): Standard deviation of results.
    """
    sim = cfg.sim
    policies = sim.full_policies
    policy_names = [policy_result_key(p, sim) for p in policies]

    if sim.graph.n_samples > 1:
        if sim.resume:
            output_stats_any: Any = output_stats
            return output_stats_any(
                home_dir=str(udef.ROOT_DIR),
                ndays=sim.graph.n_days,
                nbins=sim.graph.num_loc,
                output_dir=sim.output_dir,
                area=sim.graph.area,
                waste_type=sim.graph.waste_type,
                data_distribution=sim.data_distribution,
                run_name=getattr(sim, "run_name", "") or "",
                nsamples=sim.graph.n_samples,
                policies=policy_names,
                keys=udef.SIM_METRICS,
                lock=lock,
            )
        else:
            log: Dict[str, Any] = {}
            log_std: Dict[str, Any] = {}
            log_full: Dict[str, List[List[float]]] = defaultdict(list)

            # Values are multiprocessing ListProxy objects in parallel runs, not lists.
            for key, val in log_tmp.items():
                log_full[key].extend(list(val))

            for pol_name in policy_names:
                # A policy with no successful sample is omitted. An all-zero mean
                # would be written as if the policy had run (DS-18).
                if not log_full[pol_name]:
                    continue
                log[pol_name] = [statistics.mean(v) for v in zip(*log_full[pol_name], strict=False)]
                log_std[pol_name] = [
                    statistics.stdev(v) if len(log_full[pol_name]) > 1 else 0.0
                    for v in zip(*log_full[pol_name], strict=False)
                ]

            _log_sim_metrics(log, log_std)
            return log, log_std
    else:
        log = {pol: res[0] for pol, res in log_tmp.items() if res}
        _log_sim_metrics(log)
        return log, None
