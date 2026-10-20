"""
ARCO (Adaptive Routing Algorithm Orchestrator) — Adaptive-Weight Meta-Constructor.

The ARCO is a meta-policy that coordinates multiple routing algorithms in a
dynamically ordered sequence.  Unlike the Sequential Routing Algorithm (SRC),
which executes routing algorithms in a fixed, configuration-defined order, the ARCO
maintains two sets of adaptive weights that are updated online:

    w_first[i]      Desirability of routing algorithm i as the *first* in the chain.
    W[i, j]         Conditional desirability of routing algorithm j *immediately after*
                    routing algorithm i, given i has already been placed in the chain.

At each call the ARCO builds a full permutation of its routing algorithm pool by
sequentially eliminating already-selected constructors:

    Step 1 — First position:
        Select c_1 = argmax* w_first  (with exploration noise).

    Step 2 — Second position:
        Mask out c_1 from W[c_1, :], select c_2 from the remaining candidates.

    Step k — k-th position:
        Use W[c_{k-1}, :] as scores, mask already-chosen routing algorithms, select c_k.

Each routing algorithm in the resulting ordered chain receives the tour produced by
its predecessor as its warm start, giving ARCO the same state-threading
semantics as SRC.

After the full chain executes, ARCO observes per-step marginal improvement
(Δ profit or Δ cost) and updates W[c_{k-1}, c_k] via EMA:

    W[i, j] ← (1 − α) · W[i, j] + α · marginal_reward(j)

This allows the orchestrator to learn, across simulation days, which routing algorithm
orderings are most effective for the current problem instance and distribution.

Selection strategies:
    - ``"epsilon_greedy"``: greedily follow weights with probability (1 − ε),
      explore uniformly with probability ε.
    - ``"greedy"``:         always pick the maximum-weight successor.
    - ``"softmax"``:        Boltzmann-proportional sampling; temperature τ
                            controls the sharpness of the distribution.

Registry key: ``"arco"``

References:
    Cowling, P., Kendall, G., & Soubeiga, E. (2001). A hyper-heuristic approach
    to scheduling a sales summit. PATAT III, LNCS 2079, 176–190.

    Burke, E. K., et al. (2013). Hyper-heuristics: A survey of the state of
    the art. Journal of the Operational Research Society, 64(12), 1695–1724.

Attributes:
    AdaptiveRoutingAlgorithmOrchestrator: Meta-constructor that learns optimal
        ordering of routing heuristics through online experience.

Example:
    >>> from logic.src.policies.routing.other_algorithms.adaptive_routing_algorithm_orchestrator import AdaptiveRoutingAlgorithmOrchestrator
    >>> params = ARCOConfig()
    >>> arco = AdaptiveRoutingAlgorithmOrchestrator(params)
    >>> routes = arco.build_routes(context)

The adaptive sequence selection (greedy / epsilon-greedy / softmax),
the EMA weight update, the weight decay and the sigmoid reward helper
live in ``arco_weights.py`` as the ``ARCOWeightsMixin``, mixed into the
orchestrator here.
"""

from __future__ import annotations

import logging
import time
from random import Random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from logic.src.interfaces.context.search_context import SearchContext

from logic.src.configs.policies.arco import ARCOConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .arco_weights import ARCOWeightsMixin

logger = logging.getLogger(__name__)


@GlobalRegistry.register(
    PolicyTag.HEURISTIC,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.ADAPTIVE_ALGORITHM,
)
@RoutingAlgorithmRegistry.register("arco")
class AdaptiveRoutingAlgorithmOrchestrator(ARCOWeightsMixin, BaseRoutingPolicy):
    """
    Adaptive Routing Algorithm Orchestrator (ARCO).

    A meta-constructor that maintains pair-wise transition weights between
    routing algorithm names and uses them to dynamically order the execution chain
    at each call.  Weights are updated online from observed marginal
    improvements, allowing ARCO to learn which ordering performs best for the
    current problem instance.

    Architecture
    ------------
    Weight storage (persists across calls):
        w_first   : np.ndarray, shape (n,)       — first-position scores.
        W         : np.ndarray, shape (n, n)      — pair-wise transition matrix.
                    W[i, j] is the desirability of running routing algorithm j
                    immediately after routing algorithm i.  Diagonal is always 0.

    Selection at call time:
        1. Pick c_0  via  w_first  (using the configured selection strategy).
        2. Pick c_1  via  W[c_0, :]  after masking out c_0.
        3. Pick c_k  via  W[c_{k-1}, :]  after masking out {c_0, …, c_{k-1}}.

    Weight update after each call:
        For each consecutive pair (c_{k-1}, c_k) in the selected sequence:
            W[c_{k-1}, c_k] ← (1 − α) · W[c_{k-1}, c_k] + α · r_k
        where r_k is the normalised marginal reward from routing algorithm c_k.
        The first-position weight is updated similarly:
            w_first[c_0] ← (1 − α) · w_first[c_0] + α · r_0

    Registry key: ``"arco"``
    Attributes:
        None
    """

    def __init__(self, config: Any = None) -> None:
        """Initializes the ARCO orchestrator.

        Args:
            config: Configuration dictionary or Hydra config object.
                If None, default ARCOConfig are used.
        """
        super().__init__(config)
        self.constructors: List[IRoutingAlgorithm] = []
        self._constructor_names: List[str] = []
        self._initialized: bool = False

        # Build params from config
        if self.config is not None:
            self.params = ARCOConfig.from_config(self.config)
        else:
            self.params = ARCOConfig()

        # Adaptive weight state — initialised lazily once routing algorithm names are known
        self._n: int = 0
        self._idx: Dict[str, int] = {}
        self._w_first: Optional[np.ndarray] = None  # shape (n,)
        self._W: Optional[np.ndarray] = None  # shape (n, n), diagonal = 0
        self._call_counts: Optional[np.ndarray] = None  # per-constructor call count
        self._rng: Random = Random(self.params.seed)

    # ------------------------------------------------------------------
    # BaseRoutingPolicy interface
    # ------------------------------------------------------------------

    @classmethod
    def _config_class(cls):
        """Docstring.

        Returns:
            Description of return value.
        """
        from logic.src.configs.policies.arco import ARCOConfig

        return ARCOConfig

    def _get_config_key(self) -> str:
        """Get the config key for the ARCO orchestrator.

        Returns:
            str: The config key for the Adaptive Routing Algorithm Orchestrator.
        """
        return "arco"

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _initialize_constructors(self) -> None:
        """Lazily initialise sub-constructors and weight matrices."""
        if self._initialized:
            return

        from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory

        names: List[str] = self.params.routing_algorithms
        self.constructors = [RoutingAlgorithmFactory.get_adapter(name) for name in names]
        self._constructor_names = list(names)
        self._n = len(names)
        self._idx = {name: i for i, name in enumerate(names)}
        self._init_weights()
        self._initialized = True

    def _init_weights(self) -> None:
        """Initialise weight matrices to uniform values."""
        n = self._n
        init = self.params.weight_init

        self._w_first = np.full(n, init, dtype=float)
        self._W = np.full((n, n), init, dtype=float)
        np.fill_diagonal(self._W, 0.0)  # Cannot follow yourself

        self._call_counts = np.zeros(n, dtype=int)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute(
        self,
        **kwargs: Any,
    ) -> Tuple[
        Union[List[int], List[List[int]]],
        float,
        float,
        Optional[SearchContext],
        Optional[Any],
    ]:
        """
        Execute the adaptive routing algorithm chain for one routing call.

        1. Uses current weights to order the routing algorithm pool.
        2. Runs each routing algorithm in sequence, threading state forward.
        3. Observes marginal improvement per step and updates weights.

        Args:
            kwargs: Description of kwargs.
            **kwargs: Simulation context (tour, cost, profit, search_context,
                multi_day_context, mandatory, bins, distance_matrix, …).

        Returns:
            5-tuple: (tour, cost, profit, search_context, multi_day_context).
        Args:
            kwargs: Description of kwargs.
        """
        self._initialize_constructors()

        if not self.constructors:
            tour = kwargs.get("tour", [0, 0])
            cost = kwargs.get("cost", 0.0)
            profit = kwargs.get("profit", 0.0)
            return tour, cost, profit, kwargs.get("search_context"), kwargs.get("multi_day_context")

        # Apply per-call weight decay
        if self.params.decay < 1.0:
            self._apply_decay()

        start_time = time.perf_counter()

        # ---- Step 1: build the ordered sequence via adaptive selection ----
        sequence = self._select_sequence()
        logger.debug("ARCO selected sequence: %s", [self._constructor_names[i] for i in sequence])

        # ---- Step 2: execute the chain, tracking per-step performance ----
        current_tour: Union[List[int], List[List[int]]] = kwargs.get("tour", [0, 0])
        current_cost: float = kwargs.get("cost", 0.0)
        current_profit: float = kwargs.get("profit", 0.0)
        current_ctx: Optional[SearchContext] = kwargs.get("search_context")
        current_mdc: Optional[Any] = kwargs.get("multi_day_context")

        step_profits: List[float] = []
        step_costs: List[float] = []

        for step_idx, c_idx in enumerate(sequence):
            elapsed = time.perf_counter() - start_time
            if elapsed > self.params.time_limit:
                logger.warning(
                    "ARCO time limit reached (%.2fs > %.2fs). Stopping at step %d/%d.",
                    elapsed,
                    self.params.time_limit,
                    step_idx,
                    len(sequence),
                )
                break

            kwargs["tour"] = current_tour
            kwargs["cost"] = current_cost
            kwargs["profit"] = current_profit
            kwargs["search_context"] = current_ctx
            kwargs["multi_day_context"] = current_mdc

            prev_profit = current_profit
            prev_cost = current_cost

            constructor = self.constructors[c_idx]
            current_tour, current_cost, current_profit, current_ctx, current_mdc = constructor.execute(**kwargs)

            step_profits.append(current_profit - prev_profit)
            step_costs.append(prev_cost - current_cost)  # positive = improvement

        # ---- Step 3: online weight update ----
        self._update_weights(sequence[: len(step_profits)], step_profits, step_costs)

        return current_tour, current_cost, current_profit, current_ctx, current_mdc

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def best_sequence(self) -> List[str]:
        """
        Return the greedy-optimal routing algorithm sequence given current weights.

        Useful for logging or inspection after online learning has converged.

        Returns:
            List[str]: Routing algorithm names in the optimal learned order.
        """
        self._initialize_constructors()
        assert self._w_first is not None
        assert self._W is not None

        # Temporarily override strategy with greedy for read-only extraction
        original_strategy = self.params.selection_strategy
        original_epsilon = self.params.epsilon
        self.params.selection_strategy = "greedy"
        self.params.epsilon = 0.0

        idx_seq = self._select_sequence()

        self.params.selection_strategy = original_strategy
        self.params.epsilon = original_epsilon

        return [self._constructor_names[i] for i in idx_seq]

    # ------------------------------------------------------------------
    # Required abstract method — not used (execute() is overridden)
    # ------------------------------------------------------------------

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Any,
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Not used — execute() is overridden directly.

        Args:
            sub_dist_matrix: Description of sub_dist_matrix.
            sub_wastes: Description of sub_wastes.
            capacity: Description of capacity.
            revenue: Description of revenue.
            cost_unit: Description of cost_unit.
            values: Description of values.
            mandatory_nodes: Description of mandatory_nodes.
            kwargs: Description of kwargs.

        Returns:
            Description of return value.
        """
        return [], 0.0, 0.0
