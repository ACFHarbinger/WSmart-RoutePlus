"""
Hyper-Heuristic Ant Colony Optimization (Hyper-ACO).

This module implements a Hyper-Heuristic ACO where ants construct
sequences of local search operators rather than node sequences.
The pheromone matrix encodes transition probabilities between operators.

Key Features:
- Operator graph: Nodes represent local search operators (2-opt, swap, etc.)
- Pheromone trail: tau[i][j] = favorability of applying operator j after operator i
- Heuristic information: Dynamic based on operator success rate / execution time
- Solution evaluation: Apply operator sequence to a base solution

Paper reference: Chen, Kendall & Vanden Berghe (2007),
"An Ant Based Hyper-heuristic for the Travelling Tournament Problem".

Adaptations from TTP to VRPP:
- Objective = total routing cost (× C) − collected revenue (× R), subject to
  capacity constraints; infeasible solutions are penalised via Strategic Oscillation.
- `use_dynamic_lambda=True` (default): replaces the paper's λ=1.0001 base with a
  scale-invariant exp(I/Z) using an EMA of recent |improvement|, improving
  robustness across different problem scales.
- `elitism_ratio < 1.0` (optional): hedged elitism for diversity preservation;
  paper default is 1.0 (all ants sync to global best on improvement).

Attributes:
    HyperHeuristicACO: HyperHeuristicACO class.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, cfg)
    >>> best_solution = solver.solve(initial_solution)

The class is split into mixins: the HyperHeuristicSearch hooks (with ``_MoveTelemetry``
and ``_elitism_sync_count``, re-exported here) live in ``hyper_aco_hooks.py``; sequence
construction, the pheromone interface and the evaluation helpers live in
``hyper_aco_pheromones.py``.
"""

import random
from typing import Any, Dict, List, Optional, Union

import numpy as np

from logic.src.configs.policies.aco_hh import HyperHeuristicACOConfig
from logic.src.policies.routing.base.base_routing_policy import _structured_load, _unwrap_engine_section
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.bootstrap import (
    ACOBootstrapMixin,
)
from logic.src.policies.routing.hyper_heuristics.hyper_heuristic_base import HyperHeuristicSearch

from .hyper_aco_hooks import HyperACOSearchHooksMixin
from .hyper_aco_pheromones import HyperACOPheromoneMixin
from .hyper_operators import (
    HYPER_OPERATORS,
)


class HyperHeuristicACO(HyperACOSearchHooksMixin, HyperACOPheromoneMixin, ACOBootstrapMixin, HyperHeuristicSearch):
    """Ant Colony Optimization hyper-heuristic (Chen et al. 2007) on the shared
    route-level HH template (P-35).

    The search state that must persist across ``solve()`` calls (operator
    pheromones ``tau``, visibility ``eta``, traversal counts, strategic
    oscillation pressure ``pv``) is built in ``__init__``; ``solve()`` resets
    only the per-run incumbent/candidate state, so external consumers (HVPL,
    RL-HVPL) can keep coaching with one instance.
    """

    """
    Pure Ant System (AS) Hyper-Heuristic ACO solver.

    Ants construct sequences of local search operators.
    The best sequences are reinforced via pheromone updates.

    Attributes:
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        cfg: Hyper-heuristics parameters.
        initial_solution: Initial routes.
        mandatory_nodes: Mandatory nodes.
        random: Random number generator.
        np_rng: NumPy random number generator.
        operator_names: List of operator names.
        n_operators: Number of operators.
        op_to_idx: Mapping from operator names to indices.
        tau: Pheromone matrix. Row n_operators is the virtual start row used
            only when an ant has no prior position (first journey step after
            uniform placement on real vertices already seeds the real rows).
        eta: Heuristic visibility matrix (per directed edge i→j).
        edge_count: Cumulative edge traversal count (num(i,j) in the paper).
        sequence_length: Length of operator sequences (= n_operators, per paper).
        pheromone: Node-level sparse pheromone matrix (city graph, i→j edges).
            Maintained separately from ``tau`` so that external callers (e.g.
            HVPL) can reinforce node transitions via ``deposit_edge`` without
            coupling into the operator-graph internals.
    """

    def __init__(
        self,
        problem: SubProblem,
        cfg: Optional[Union[HyperHeuristicACOConfig, Dict[str, Any]]] = None,
        initial_solution: Optional[List[List[int]]] = None,
    ):
        """
        Initialize HyperHeuristicACO.

        Args:
            problem: The routing sub-problem (distances, wastes, capacity,
                revenue, cost unit, mandatory nodes).
            cfg: HyperHeuristicACOConfig dataclass or raw parameter mapping.
            initial_solution: Optional starting routes.
        """
        dist_matrix = problem.dist_matrix
        wastes = problem.wastes
        capacity = problem.capacity
        R = problem.revenue
        C = problem.cost_unit
        params = cfg
        mandatory_nodes = problem.mandatory_nodes

        self.problem = problem
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        # Standardize to HyperHeuristicACOConfig (P-10: the typed config is the
        # only parameter layer). The parameter name stays ``params`` because
        # external callers (e.g. HVPL) pass it by keyword.
        if params is None:
            params = HyperHeuristicACOConfig()
        elif isinstance(params, dict):
            params = _structured_load(HyperHeuristicACOConfig, _unwrap_engine_section(params))
        self.cfg = params
        # Value validation that lived in the deleted params layer's __post_init__
        # (P-10): the typed config stays declarative, so the solver enforces it.
        cfg = self.cfg
        if not (0 < cfg.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {cfg.rho}")
        if cfg.Q < 0:
            raise ValueError(f"Pheromone floor constant Q must be >= 0, got {cfg.Q}")
        if not (0 < cfg.elitism_ratio <= 1):
            raise ValueError(f"elitism_ratio must be in (0, 1], got {cfg.elitism_ratio}")
        if cfg.operators is not None:
            unknown = [op for op in cfg.operators if op not in HYPER_OPERATORS]
            if unknown or not cfg.operators:
                raise ValueError(f"operators must be a non-empty subset of {list(HYPER_OPERATORS)}; unknown: {unknown}")
        if cfg.sequence_length is not None and cfg.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {cfg.sequence_length}")
        self.initial_solution = initial_solution or []
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(self.cfg.seed) if self.cfg.seed is not None else random.Random()
        self.np_rng = np.random.default_rng(self.cfg.seed) if self.cfg.seed is not None else np.random.default_rng()

        self.operator_names = list(self.cfg.operators) if self.cfg.operators else list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)
        self.op_to_idx = {name: i for i, name in enumerate(self.operator_names)}

        # Journey length defaults to n (number of heuristics/vertices), per Chen et al. (2007) §III.B
        self.sequence_length = self.cfg.sequence_length or self.n_operators

        # tau[i][j]: pheromone on directed edge i→j.
        # Row n_operators is a virtual "no prior position" row — only used on the very
        # first selection step of an ant that somehow lacks a recorded position.
        # After uniform placement (Bug #2 fix) the real rows [0..n_operators-1] are
        # seeded immediately by the initial operator application, so this row is
        # primarily a safety fallback.
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.cfg.tau_0)

        # eta[i][j]: per-edge visibility (heuristic information).
        # Initialised to 1 per paper: "For each vertex j: set ηj = 1".
        self.eta = np.ones((self.n_operators + 1, self.n_operators))

        # edge_count[i][j]: num(i,j) — cumulative count of traversals, never reset.
        # Grows monotonically from the start of the run (per paper §III.B).
        self.edge_count = np.zeros((self.n_operators + 1, self.n_operators))

        # Strategic Oscillation state (per paper §III.B)
        self.initial_best_cost = self._calculate_routing_cost(self.initial_solution)

        # Node-level (city-graph) pheromone structure.
        # Used by external callers (e.g. HVPL) to deposit on route edges via
        # deposit_edge(i, j, delta), keeping city-graph reinforcement decoupled
        # from the operator-sequence pheromone matrix self.tau.
        # Local import to avoid a circular dependency: hyper_aco.py is loaded
        # inside hyper_heuristics/__init__.py, which is imported before
        # meta_heuristics/__init__.py finishes — a module-level import of
        # SparsePheromoneTau would trigger meta_heuristics/__init__.py and
        # create a cycle.  Deferring to instantiation time is safe because
        # both packages are fully initialized by then.
        from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.pheromones import (  # noqa: PLC0415
            SparsePheromoneTau,
        )

        n_nodes = len(dist_matrix)  # includes depot (index 0)
        self.pheromone = SparsePheromoneTau(
            n_nodes=n_nodes,
            tau_0=self.cfg.tau_0,
            scale=5.0,
            tau_min=self.cfg.tau_0 / 100.0,
            tau_max=self.cfg.tau_0 * 10.0,
        )
        self.initial_pv = self.initial_best_cost / 100.0 if self.initial_best_cost > 0 else 10.0
        self.pv = self.initial_pv

        # Dynamic Normalization state for scale-invariant visibility (VRPP extension)
        self.Z = self.initial_best_cost * 0.01 if self.initial_best_cost > 0 else 1.0
        self.Z_alpha = 0.1  # EMA smoothing factor

    def construct(
        self,
        nodes: List[int],
        mandatory_nodes: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Construct a solution from scratch using pure pheromone-guided exploration.

        This method is analogous to ``SolutionConstructor.construct`` in the
        k-sparse ACO: it builds a complete solution from an empty slate rather
        than improving an existing one.  The construction proceeds in two phases:

        1. **Bootstrap** — A bootstrap insertion operator is chosen **uniformly
           at random** from the full portfolio of ``recreate_repair`` heuristics
           (pure exploration — equivalent to flat tau_0 pheromone, matching MMAS
           semantics).  All nodes are treated as the unvisited pool and inserted
           into an initially-empty route list by the selected heuristic.  When
           ``cfg.profit_aware_operators`` is enabled the profit-maximising
           variant of each heuristic is used; otherwise the cost-minimising
           variant is used.  The available bootstrap operators are:

           Standard (cost-minimising): greedy, greedy-with-blinks, nearest,
           farthest, savings, regret-2, regret-3, regret-4, deep, GENI,
           branch-and-bound LDS.

           Profit-aware (revenue - cost): greedy_profit, blinks_profit,
           nearest_profit, farthest_profit, savings_profit, regret-2_profit,
           regret-3_profit, regret-4_profit, deep_profit, GENI_profit,
           bb_profit.

        2. **Exploration** — A single ant is placed on a uniformly-random
           operator vertex (mirroring the uniform placement in ``solve``) and a
           full operator sequence of length ``sequence_length`` is selected via
           the pure AS proportional (roulette-wheel) rule through
           ``_select_sequence``.  The sequence is then applied to the bootstrapped
           solution via ``build_solution``.  No local pheromone updates are
           performed inside this method — all reinforcement is expected to happen
           globally after evaluation, consistent with MMAS semantics.

        Args:
            nodes: List of all customer node indices (excluding depot 0) that
                the ant should attempt to route.
            mandatory_nodes: Nodes that must be visited regardless of
                profitability.  Overrides ``self.mandatory_nodes`` when provided;
                falls back to ``self.mandatory_nodes`` when ``None``.

        Returns:
            List of routes, each a list of node indices (excluding depot 0),
            representing the constructed solution.
        """
        effective_mandatory = mandatory_nodes if mandatory_nodes is not None else (self.mandatory_nodes or [])

        # ----------------------------------------------------------------
        # Phase 1 – Bootstrap: choose an insertion heuristic uniformly at
        # random from the full recreate_repair portfolio (pure exploration).
        # All nodes play the role of "removed" nodes; starting from [] mirrors
        # the k-sparse ant beginning with all nodes in the unvisited set.
        #
        # Uniform selection is semantically identical to roulette-wheel
        # selection with a flat tau_0 matrix — maximum entropy, no bias.
        # ----------------------------------------------------------------
        bootstrapped: List[List[int]] = self._bootstrap(
            nodes=nodes,
            effective_mandatory=effective_mandatory,
        )

        # ----------------------------------------------------------------
        # Phase 2 – Exploration: place one ant uniformly on the operator
        # graph and select an operator sequence via pure roulette-wheel
        # (no exploitation bias, matching MMAS / k-sparse construct semantics).
        # Then apply the sequence to the bootstrapped solution.
        # ----------------------------------------------------------------
        start_op_idx: int = self.random.randrange(self.n_operators)
        routes, _sequence, _eta_updates, _end_op_idx = self.build_solution(
            base_solution=bootstrapped,
            start_op_idx=start_op_idx,
        )

        return routes
