"""Population-based search template for routing meta-heuristics (WP-11 P-34).

:class:`PopulationSearch` is the shared skeleton of every population routing
engine: initialise a population, repeat (vary -> educate -> replace -> track
best) until the iteration budget, the wall-clock budget or stagnation says
stop, then return the incumbent as the ``(routes, profit, cost)`` triple.
The first engine ported onto the template is PSOMA
(:class:`~logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver`);
the other population meta-heuristics (HGS, GA, plain PSO, ...) are deferred
(RQ-claude-3) but are expected to map onto the same hooks:

- **HGS** (dual subpopulations + penalty adaptation): ``_initial_population``
  builds both the feasible and the infeasible subpopulations,
  ``_vary_population`` runs selection/crossover within each subpopulation,
  ``_educate`` is the HGS local-search (SISR-style) repair/improvement,
  ``_replace`` applies the distance-diversity survivor selection and
  ``_track_best`` folds both subpopulations into the incumbent while adapting
  the capacity-penalty multiplier. ``_should_stop`` additionally honours the
  minimum-population-diversity trigger.
- **GA** (elitist carry-forward + educate-best): ``_vary_population`` is
  tournament selection + crossover + mutation, ``_educate`` improves the best
  offspring (memetic GA), ``_replace`` performs the elitist carry-forward of
  the incumbent into the next generation and ``_track_best`` tracks the best
  individual seen so far.
- **Plain PSO** (no memetic step): ``_vary_population`` is the velocity update
  + decode + personal/global best tracking, while ``_educate`` and
  ``_replace`` stay no-ops; the template degenerates to the canonical PSO loop.

Engines keep full ownership of their population representation, decoder and
RNG streams; the template never draws from an engine's generators, so seeded
runs stay bit-identical as long as each hook consumes its streams in a fixed
order.

Example:
    >>> class MyEngine(PopulationSearch):
    ...     def _initial_population(self) -> None: ...
    ...     def _vary_population(self) -> None: ...
"""

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Type

from logic.src.policies.routing.base.sub_problem import SubProblem

__all__ = ["PopulationSearch"]


class PopulationSearch(ABC):
    """Template base class for population-based routing meta-heuristics.

    Constructor contract (matches the ``Solver`` protocol from P-30):

        solver = MyEngine(problem: SubProblem, cfg: MyConfig)
        routes, profit, cost = solver.solve()

    Template hooks (called in this order by :meth:`solve`):

    - :meth:`_setup_search` — one-time setup; derives the wall-clock
      ``_deadline`` from ``cfg.time_limit``.
    - :meth:`_initial_population` (abstract) — build the starting population
      and fold it into the incumbent best.
    - :meth:`_prepare_intensification` — one-shot pre-loop warmup (acceptance
      criterion setup, operator calibration, ...).
    - :meth:`_vary_population` (abstract) — advance the population by one
      variation step (PSO velocity update, GA crossover, ...).
    - :meth:`_educate` — per-iteration intensification of the incumbent
      (memetic local search; no-op for swarm-only engines).
    - :meth:`_replace` — survivor selection at the end of a round (default:
      the population persists unchanged).
    - :meth:`_track_best` — merge the round's discoveries into the incumbent
      best; returns True when the best improved (drives stagnation counting).
    - :meth:`_should_stop` — two-level stopping rule (wall-clock budget and
      stagnation patience by default; engines may extend it).
    - :meth:`_result` — assemble the final ``(routes, profit, cost)`` triple
      from the incumbent.

    Attributes:
        problem (SubProblem): The extracted routing sub-problem.
        cfg (Any): Typed engine configuration (coercion is engine-specific).
        best_routes (List[List[int]]): Search-level incumbent routes.
        best_profit (float): Search-level incumbent profit.
        _deadline (float): ``time.perf_counter()`` timestamp after which the
            search must stop (``inf`` when no time limit is configured).
    """

    def __init__(self, problem: SubProblem, cfg: Any = None) -> None:
        """Initializes the search with a sub-problem and its typed config.

        Args:
            problem (SubProblem): Extracted routing sub-problem.
            cfg (Any): Engine configuration; engines coerce it via
                :meth:`_coerce_config` before calling ``super().__init__``.
        """
        self.problem = problem
        self.cfg = cfg
        self.best_routes: List[List[int]] = []
        self.best_profit = -float("inf")
        self._deadline = float("inf")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_config(cfg: Any, config_cls: Type) -> Any:
        """Lenient engine-level config coercion: dataclass instances pass
        through, mappings and plain objects are filtered onto the dataclass
        fields (policy-level loads stay strict; engines stay lenient).
        """
        if cfg is None:
            return config_cls()
        if isinstance(cfg, config_cls):
            return cfg
        valid_keys = {f.name for f in dataclasses.fields(config_cls)}
        if isinstance(cfg, dict):
            return config_cls(**{k: v for k, v in cfg.items() if k in valid_keys})
        if hasattr(cfg, "__dict__"):
            return config_cls(**{k: v for k, v in vars(cfg).items() if k in valid_keys})
        return cfg

    # ------------------------------------------------------------------
    # Template hooks
    # ------------------------------------------------------------------

    def _setup_search(self, start_time: float) -> None:
        """One-time search setup: derive ``_deadline`` from ``cfg.time_limit``."""
        time_limit = float(getattr(self.cfg, "time_limit", 0.0) or 0.0)
        self._deadline = start_time + time_limit if time_limit > 0 else float("inf")

    @abstractmethod
    def _initial_population(self) -> None:
        """Build the starting population and fold it into the incumbent best."""

    def _prepare_intensification(self) -> None:  # noqa: B027 (intentional overridable no-op hook)
        """One-shot warmup before the main loop (criterion setup, operator calibration)."""
        ...

    @abstractmethod
    def _vary_population(self) -> None:
        """Advance the population by one generation/variation step."""

    def _educate(self) -> None:  # noqa: B027 (intentional overridable no-op hook)
        """Per-iteration intensification of the incumbent (memetic step)."""
        ...

    def _replace(self) -> None:  # noqa: B027 (intentional overridable no-op hook)
        """Survivor selection at the end of a round (default: population persists)."""
        ...

    def _track_best(self, previous_best: float) -> bool:
        """Fold the round's discoveries into the incumbent best.

        Args:
            previous_best (float): Incumbent profit captured before this round.

        Returns:
            bool: True when the incumbent improved during this round.
        """
        return self.best_profit > previous_best

    def _should_stop(self, elapsed: float, iteration: int, stagnation: int) -> bool:
        """Two-level stopping rule: wall-clock budget or stagnation patience.

        Engines without a stagnation parameter (``cfg.L``) only honour the
        time budget; engines with extra triggers (diversity, ...) override.
        """
        time_limit = float(getattr(self.cfg, "time_limit", 0.0) or 0.0)
        if time_limit > 0 and elapsed > time_limit:
            return True
        patience = getattr(self.cfg, "L", None)
        return patience is not None and stagnation >= patience

    def _result(self) -> Tuple[List[List[int]], float, float]:
        """Assemble the final ``(routes, profit, cost)`` triple from the incumbent."""
        return self.best_routes, self.best_profit, self._solution_cost(self.best_routes)

    def _solution_cost(self, routes: List[List[int]]) -> float:
        """Total travel cost of a route set at the sub-problem's per-unit cost."""
        dist_matrix = self.problem.dist_matrix
        cost_unit = self.problem.cost_unit
        cost = 0.0
        for route in routes:
            if not route:
                continue
            cost += dist_matrix[0, route[0]]
            for k in range(len(route) - 1):
                cost += dist_matrix[route[k], route[k + 1]]
            cost += dist_matrix[route[-1], 0]
        return cost * cost_unit

    # ------------------------------------------------------------------
    # Time budget
    # ------------------------------------------------------------------

    def _out_of_time(self) -> bool:
        """True once the solve's wall-clock budget is spent."""
        return time.perf_counter() > self._deadline

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Execute the population search.

        Returns:
            Tuple[List[List[int]], float, float]: A tuple containing:
                - The best routing solution found (list of routes).
                - The total profit of the best solution.
                - The total travel cost (distance) of the best solution.
        """
        start_time = time.perf_counter()
        self._setup_search(start_time)
        self._initial_population()
        self._prepare_intensification()

        stagnation_counter = 0
        max_iterations = int(getattr(self.cfg, "max_iterations", 0) or 0)
        for iteration in range(max_iterations):
            if self._should_stop(time.perf_counter() - start_time, iteration, stagnation_counter):
                break
            previous_best = self.best_profit
            self._vary_population()
            self._educate()
            self._replace()
            stagnation_counter = 0 if self._track_best(previous_best) else stagnation_counter + 1

        return self._result()
