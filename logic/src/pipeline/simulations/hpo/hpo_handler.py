"""

The Optuna objective and worker live in ``hpo_objective``; this module keeps the handler/orchestrator and re-exports those names.
Core handler for simulation policy hyperparameter optimization (HPO).

This module implements the Optuna objective function for simulation policies,
including parameter sampling, simulation execution, and metric calculation.
It supports process-based parallel optimization via a shared SQLite storage
backend, multi-objective Pareto-front search, Hyperband pruning, and
integrates with the MLflow tracking system.

Key improvements over the previous version:
  - Sampler method is actually used (was previously ignored).
  - Study is persisted to SQLite so crashed runs can be resumed.
  - Parallel execution uses separate OS processes + shared storage,
    avoiding GIL contention on CPU-bound simulations.
  - Multi-objective mode (e.g. profit ↑ + overflows ↓) supported via NSGA-II.
  - Hyperband pruner for iterative solvers.
  - Parameter suggestion is fully delegated to PolicyHPOBase.suggest_param
    (no duplicated inline logic).
  - Param names passed to Optuna are always the short name; the full config
    path is tracked in a separate PARAM_PATH_MAP per trial to keep the study
    DB coherent across restarts.
  - Search space is validated before any trial runs.

Attributes:
    logger: Logger instance for HPO operations.

Functions:
    objective:     Optuna objective for a single trial (single or multi-objective).
    worker:        Entry point for each worker process.
    run_hpo_sim:   Orchestrates the parallel HPO process.

Example:
    >>> # from logic.src.policies.helpers.hpo import run_hpo_sim
    >>> # best_metric = run_hpo_sim(cfg)
    >>> # print(f"Best metric: {best_metric}")
"""

import logging
import multiprocessing as mp
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import optuna
from omegaconf import OmegaConf
from optuna.importance import FanovaImportanceEvaluator
from optuna.pruners import HyperbandPruner
from optuna.trial import FrozenTrial

from logic.src import tracking as wst
from logic.src.configs import Config
from logic.src.constants import ROOT_DIR
from logic.src.pipeline.simulations.hpo.base import PolicyHPOBase
from logic.src.pipeline.simulations.hpo.search_spaces import validate_search_space
from logic.src.pipeline.simulations.repository import (
    load_simulator_data,
    set_repository_from_path,
)

from .hpo_objective import (
    _MINIMISE_METRICS as _MINIMISE_METRICS,
)
from .hpo_objective import (
    _export_hpo_visualization_reports as _export_hpo_visualization_reports,
)
from .hpo_objective import (
    _extract_metric as _extract_metric,
)
from .hpo_objective import (
    _metric_direction as _metric_direction,
)
from .hpo_objective import (
    _select_pareto_representative as _select_pareto_representative,
)
from .hpo_objective import (
    objective as objective,
)
from .hpo_objective import (
    objective_debug as objective_debug,
)
from .hpo_objective import (
    worker as worker,
)

# Suppress verbose warnings from BoTorch for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("HPO_Handler")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum recommended number of simulation samples per trial.
# A single sample has high variance for stochastic solvers (ALNS, ACO, etc.).
MIN_RECOMMENDED_SAMPLES = 5

# Metrics that are minimised (lower is better); all others are maximised.


# ---------------------------------------------------------------------------
# Handler Class
# ---------------------------------------------------------------------------


class HPOSimulationHandler:
    """State-of-the-art HPO Engine for routing policies.

    Integrates BoTorch (Gaussian Processes), Hyperband (Pruning), and fANOVA.

    Attributes:
        cfg (Config): Root application configuration.
        study_name (str): Name of the Optuna study.
        storage_url (str): URL of the Optuna storage backend.
        directions (List[str]): List of optimization directions.
        metric_names (List[str]): List of metric names.
        max_budget (int): Maximum budget for the HPO process.
        sampler (PolicyHPOBase): Sampler for Optuna.
        pruner (HyperbandPruner): Pruner for Optuna.
        study (optuna.study.Study): Optuna study object.
    """

    def __init__(
        self,
        cfg: Config,
        study_name: str,
        storage_url: str,
        directions: List[str],
        metric_names: Optional[List[str]] = None,
        max_budget: int = 100,
    ):
        """Initialize HPOSimulationHandler.

        Args:
            cfg: Root application configuration.
            study_name: Name of the Optuna study.
            storage_url: URL of the Optuna storage backend.
            directions: List of optimization directions.
            metric_names: List of metric names.
            max_budget: Maximum budget for the HPO process.

        """
        self.cfg = cfg
        self.study_name = study_name
        self.storage_url = storage_url
        self.directions = directions
        self.metric_names = metric_names or [f"Obj_{i}" for i in range(len(directions))]
        self.max_budget = max_budget

        # Advanced Sampler: BoTorch (native covariance modeling)
        self.sampler = PolicyHPOBase.build_sampler(
            method=cfg.hpo_sim.method,
            seed=cfg.seed,
            search_space=cfg.hpo_sim.search_space,
        )

        # Multi-Fidelity Pruner: Hyperband
        self.pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.max_budget,
            reduction_factor=3,
        )

        self._init_storage()
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            directions=self.directions,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,
        )

    def _init_storage(self) -> None:
        """Ensure SQLite storage is ready.

        Args:
            self: HPOSimulationHandler instance.

        """
        if self.storage_url.startswith("sqlite:///"):
            db_path = self.storage_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            if not os.path.exists(db_path):
                with open(db_path, "a"):
                    pass

    def get_objective(self, lock: Any, data_size: int) -> Callable[[optuna.Trial], Union[float, List[float], Tuple[float]]]:
        """Creates the Optuna objective closure.

        Args:
            self: HPOSimulationHandler instance.
            lock: Lock for synchronization.
            data_size: Size of the data.

        Returns:
            Callable[[optuna.Trial], Union[float, List[float]]]: Optuna objective closure.
        """

        def objective_fn(trial: optuna.Trial) -> Union[float, List[float]]:
            # Use the existing functional objective but wrapped in the class state.
            return objective(trial, self.cfg, data_size, lock) # pyrefly: ignore [bad-return]

        return objective_fn

    def run_fanova_analysis(self, target_idx: int = 0) -> None:
        """Execute functional Analysis of Variance (fANOVA).

        Args:
            self: HPOSimulationHandler instance.
            target_idx: Index of the target metric.

        """
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 30:
            logger.info(f"fANOVA requires ~30 trials (found {len(completed)}). Skipping.")
            return

        logger.info(f"--- fANOVA Analysis (Target: {self.metric_names[target_idx]}) ---")
        try:
            evaluator = FanovaImportanceEvaluator(n_trees=64, max_depth=64)

            def target_fn(t: FrozenTrial) -> float:
                return float(t.values[target_idx] if t.values else t.value) # pyrefly: ignore [bad-argument-type]

            importances = optuna.importance.get_param_importances(self.study, evaluator=evaluator, target=target_fn)

            for param, importance in importances.items():
                logger.info(f"{param:>30}: {importance:.4f} ({importance * 100:>5.2f}%)")
        except Exception as e:
            logger.error(f"fANOVA failed: {e}")

    def log_pareto_front(self) -> None:
        """Report the Pareto-optimal frontier.

        Args:
            self: HPOSimulationHandler instance.

        """
        try:
            best_trials = self.study.best_trials
            logger.info(f"--- Pareto Front ({len(best_trials)} optimal configs) ---")
            for _i, trial in enumerate(best_trials):
                metrics_str = " | ".join(f"{n}: {v:.4f}" for n, v in zip(self.metric_names, trial.values, strict=False)) # pyrefly: ignore [bad-argument-type]
                logger.info(f"Trial {trial.number} -> {metrics_str}")
        except Exception:
            if self.study.best_trial:
                logger.info(f"Best Value: {self.study.best_value:.4f}")
                logger.info(f"Best Params: {self.study.best_params}")




# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Worker (process-based parallelism)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------



def run_hpo_sim(cfg: Config) -> Union[float, List[float]]:
    """Run Hyperparameter Optimization for simulation policies.

    Orchestrates the full HPO pipeline:
      1. Validates the search space before committing any resources.
      2. Creates or resumes a persistent Optuna study.
      3. Launches worker processes (one per ``num_workers``), each contributing
         an equal share of trials via the shared storage backend.
      4. Logs the best result(s) to the experiment tracker.

    In multi-objective mode (``hpo_sim.metrics`` has >1 entry), returns the
    list of objective values for the Pareto-optimal trial closest to the
    ideal point; in single-objective mode, returns the best scalar value.

    Args:
        cfg: Root application configuration containing HPO settings.

    Returns:
        Union[float, List[float]]: Best metric value (single-objective) or
            best Pareto-front objective values (multi-objective).
    """
    hpo_sim = cfg.hpo_sim
    sim = cfg.sim

    # -----------------------------------------------------------------
    # 0. Resolve and validate search space eagerly.
    # -----------------------------------------------------------------
    search_space = dict(hpo_sim.search_space)

    if not search_space:
        from logic.src.pipeline.simulations.hpo.search_spaces import compose_search_space

        search_space = compose_search_space(
            routing=hpo_sim.policy_name,
            packing=getattr(hpo_sim, "selection_name", None),
            route_improvement=getattr(hpo_sim, "improver_name", None),
            acceptance_criteria=getattr(hpo_sim, "acceptance_name", None),
            routing_keywords=getattr(hpo_sim, "policy_keywords", None),
            packing_keywords=getattr(hpo_sim, "selection_keywords", None),
            route_improvement_keywords=getattr(hpo_sim, "improver_keywords", None),
            acceptance_criteria_keywords=getattr(hpo_sim, "acceptance_keywords", None),
        )

        if not search_space:
            logger.warning(
                f"No search space provided in config and no default found for "
                f"policy='{hpo_sim.policy_name}'. HPO will perform no parameter changes."
            )
        else:
            # Save the composed search space back to the config for worker access.
            hpo_sim.search_space = search_space

    # Validate the final search space.
    if search_space:
        validate_search_space(search_space, hpo_sim.policy_name)

    # Warn if n_samples is too low for reliable HPO.
    n_samples = hpo_sim.graph.n_samples
    if n_samples < MIN_RECOMMENDED_SAMPLES:
        logger.warning(
            f"hpo_sim.graph.n_samples={n_samples} is below the recommended minimum of "
            f"{MIN_RECOMMENDED_SAMPLES}. Results may be noisy. Consider increasing n_samples."
        )

    # -----------------------------------------------------------------
    # 1. Resolve metric list (supports legacy scalar 'metric' field).
    # -----------------------------------------------------------------
    metrics: List[str] = list(getattr(hpo_sim, "metrics", None) or [getattr(hpo_sim, "metric", "profit")])
    directions = [_metric_direction(m) for m in metrics]
    is_multi_objective = len(metrics) > 1

    logger.info(f"Metrics: {metrics}  |  Directions: {directions}")

    if is_multi_objective and hpo_sim.method.lower() not in ("nsgaii", "random"):
        logger.warning(
            f"Multi-objective HPO with method='{hpo_sim.method}' may not work correctly. "
            f"Consider switching to method='nsgaii'."
        )

    # -----------------------------------------------------------------
    # 2. Initialise experiment tracking.
    # -----------------------------------------------------------------
    experiment_name = cfg.experiment_name or f"hpo_sim_{hpo_sim.policy_name}"
    wst.init(experiment_name=experiment_name)

    # -----------------------------------------------------------------
    # 3. Resolve data repository and data_size.
    # -----------------------------------------------------------------
    load_ds = getattr(sim.graph, "load_dataset", None)
    if load_ds is not None and set_repository_from_path(
        str(load_ds), area=sim.graph.area, waste_type=sim.graph.waste_type
    ):
        data_size = sim.graph.num_loc
    else:
        set_repository_from_path(str(ROOT_DIR), area=sim.graph.area, waste_type=sim.graph.waste_type)
        try:
            data_tmp, _ = load_simulator_data(sim.data_dir, sim.graph.num_loc, sim.graph.area, sim.graph.waste_type)
            data_size = len(data_tmp)
        except Exception:
            data_size = sim.graph.num_loc

    output_dir = getattr(cfg, "output_dir", str(ROOT_DIR))
    os.makedirs(output_dir, exist_ok=True)
    n_workers = max(1, hpo_sim.num_workers)
    n_trials = hpo_sim.n_trials

    handler = HPOSimulationHandler(
        cfg=cfg,
        study_name=f"{hpo_sim.policy_name}_seed{cfg.seed}",
        storage_url=f"sqlite:///{os.path.join(output_dir, f'hpo_{hpo_sim.policy_name}.db')}",
        directions=directions,
        metric_names=metrics,
        max_budget=hpo_sim.graph.n_days,
    )

    logger.info(
        f"Starting HPO  policy={hpo_sim.policy_name}  method={hpo_sim.method}  trials={n_trials}  workers={n_workers}"
    )

    base_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    manager = mp.Manager()
    lock = manager.Lock()

    if n_workers == 1:
        worker(handler.study_name, handler.storage_url, base_cfg_yaml, data_size, n_trials, lock)
    else:
        processes = []
        trials_per_worker = max(1, n_trials // n_workers)
        trials_last_worker = n_trials - trials_per_worker * (n_workers - 1)

        for i in range(n_workers):
            worker_trials = trials_last_worker if i == n_workers - 1 else trials_per_worker
            p = mp.Process(
                target=worker,
                args=(handler.study_name, handler.storage_url, base_cfg_yaml, data_size, worker_trials, lock),
                daemon=False,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # -----------------------------------------------------------------
    # 6. Post-optimization analysis.
    # -----------------------------------------------------------------
    handler.log_pareto_front()
    handler.run_fanova_analysis()

    _export_hpo_visualization_reports(handler.study)

    run = wst.get_active_run()

    if is_multi_objective:
        pareto_trials = handler.study.best_trials
        logger.info(f"HPO complete. Pareto front size: {len(pareto_trials)}")

        # Report the trial closest to the ideal point (normalised L1 distance).
        best_trial = _select_pareto_representative(pareto_trials, directions)
        best_values = list(best_trial.values) if best_trial else [] # pyrefly: ignore [bad-argument-type]

        if run is not None and best_trial:
            run.log_params({f"hpo/best/{k}": v for k, v in best_trial.params.items()})
            for m, v in zip(metrics, best_values, strict=False):
                run.log_metric(f"hpo/best/{m}", v)
            run.set_tag("task", "hpo_sim")
            run.set_tag("policy", hpo_sim.policy_name)
            run.set_tag("mode", "multi_objective")
            run.flush()

        return best_values

    else:
        best_trial = handler.study.best_trial
        best_value = best_trial.value if best_trial else float("-inf")

        logger.info(f"HPO complete. Best {metrics[0]}: {best_value}")
        logger.info(f"Best params: {best_trial.params if best_trial else {}}")

        if run is not None and best_trial:
            run.log_params({f"hpo/best/{k}": v for k, v in best_trial.params.items()})
            run.log_metric(f"hpo/best_{metrics[0]}", best_value)
            run.set_tag("task", "hpo_sim")
            run.set_tag("policy", hpo_sim.policy_name)
            run.set_tag("mode", "single_objective")
            run.flush()

        return best_value # pyrefly: ignore [bad-return]


# ---------------------------------------------------------------------------
# Pareto selection helper
# ---------------------------------------------------------------------------

