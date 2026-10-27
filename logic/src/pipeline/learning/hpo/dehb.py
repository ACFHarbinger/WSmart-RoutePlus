"""
Differential Evolution Hyperband (DEHB) wrapper.

Extends :class:`BaseHPO` to optimise hyperparameters via the DEHB algorithm,
supporting float, int, and categorical parameter types through the shared
typed search-space specification.

Attributes:
    DifferentialEvolutionHyperband: DEHB HPO algorithm.
    CS: ConfigSpace library.
    ParamSpec: Type alias for parameter specification.
    BaseHPO: Abstract base class for HPO algorithms.
    normalise_search_space: Normalise search space.
    apply_params: Apply parameters to config.

Example:
    >>> from logic.src.pipeline.learning.hpo import DifferentialEvolutionHyperband
    >>> dehb_hpo = DifferentialEvolutionHyperband(cfg, objective_fn)
    >>> dehb_hpo
    <logic.src.pipeline.learning.hpo.dehb.DifferentialEvolutionHyperband object at 0x...>
"""

import time
from typing import Any, Callable, Dict, Optional, cast

from dehb import DEHB

from logic.src.configs import Config

try:
    from logic.src.tracking.core.run import get_active_run
except ImportError:
    get_active_run = None  # type: ignore[assignment]

from .base import BaseHPO, ParamSpec


class DifferentialEvolutionHyperband(BaseHPO):
    """
    HPO backend using Differential Evolution Hyperband.

    This wrapper builds a ``ConfigSpace.ConfigurationSpace`` from the typed
    search-space dict and delegates the optimisation loop to the upstream
    :class:`dehb.DEHB` solver.
    Attributes:
        None
    """

    def __init__(
        self,
        cfg: Config,
        objective_fn: Callable,
        search_space: Optional[Dict[str, ParamSpec]] = None,
        min_fidelity: int = 1,
        max_fidelity: int = 10,
        eta: int = 3,
        n_workers: int = 1,
        output_path: str = "./dehb_output",
        **kwargs: Any,
    ):
        """Initialize DEHB wrapper.

        Args:
            cfg: Root application configuration.
            objective_fn: Callable ``(config_dict, fidelity) -> dict``
                returning at least ``{"fitness": float}`` (lower is better).
            search_space: Optional pre-normalised search space.
            min_fidelity: Minimum fidelity (e.g. epochs).
            max_fidelity: Maximum fidelity.
            eta: Halving rate.
            n_workers: Number of workers.
            output_path: Path for logs and results.
            kwargs: Extra arguments forwarded to :class:`dehb.DEHB`.
        """
        super().__init__(cfg, objective_fn, search_space)

        # Build ConfigSpace from typed specs
        config_space = self.build_configspace(self.search_space)

        # Store DEHB solver as an attribute rather than inheriting from it
        self._dehb = DEHB(
            cs=config_space,
            f=objective_fn,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            eta=eta,
            n_workers=n_workers,
            output_path=output_path,
            **kwargs,
        )

    def run(self) -> float:
        """Run DEHB optimization.

        Returns:
            Best metric value found (as maximisation target, i.e. negated fitness).
        """
        fevals = self.cfg.hpo.fevals if hasattr(self.cfg.hpo, "fevals") else self.cfg.hpo.n_trials
        start = time.time()

        self._dehb.run(fevals=fevals)

        elapsed = time.time() - start

        # Extract best result
        best_config, best_score = self._dehb.get_incumbents()

        if hasattr(best_config, "get_dictionary"):
            # ConfigSpace Configuration is a Mapping; dict() replaces deprecated get_dictionary().
            best_config = dict(cast(Any, best_config))

        # Store for external inspection
        self.best_config = best_config
        self.best_score = best_score
        self.runtime = elapsed
        self.history = self._dehb.history

        # Log to WSTracker
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            run.log_params({f"hpo/best/{k}": v for k, v in best_config.items()})
            run.log_metric("hpo/best_score", float(-best_score) if best_score is not None else 0.0)
            run.log_metric("hpo/runtime_s", elapsed)
            run.log_metric("hpo/n_evals", fevals)
            run.set_tag("hpo_backend", "dehb")

        # DEHB minimises fitness; we negate to return a maximisation value
        return float(-best_score) if best_score is not None else 0.0
