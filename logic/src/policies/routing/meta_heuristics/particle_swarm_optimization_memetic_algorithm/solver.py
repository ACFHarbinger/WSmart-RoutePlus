"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Particles navigate the discrete routing space via swap-based velocity.
A low inertia weight (ω≈0.4) forces intensive local exploitation.

Attributes:
    PSOMASolver (Type): Core solver class for PSOMA, built on PopulationSearch.
    PSOMAConfig (Type): Parameter dataclass for the solver.

Example:
    >>> problem = build_subproblem(dist_matrix, wastes, capacity, R, C)
    >>> solver = PSOMASolver(problem, params)
    >>> routes, profit, cost = solver.solve()

Reference:
    Liu, B., Wang, L., Jin, Y., & Huang, D. (2006).
    "An Effective PSO-Based Memetic Algorithm for TSP"

The swarm, evaluation and SA local-search helpers live in ``swarm_search.py``
(:class:`PSOMASwarmSearchMixin`).
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.psoma import PSOMAConfig
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.routing.meta_heuristics.population_search import PopulationSearch

from .particle import PSOMAParticle
from .swarm_search import PSOMASwarmSearchMixin


class PSOMASolver(PSOMASwarmSearchMixin, PopulationSearch):
    """
    PSOMASolver class for Vehicle Routing Problem with Pickup and Delivery (VRPP).
    Implements a Particle Swarm Optimization (PSO) algorithm with a Memetic Algorithm (MA).

    Ported onto the :class:`PopulationSearch` template (WP-11 P-34): the swarm is
    the population, the PSO velocity/ROV/Split step is ``_vary_population`` and the
    operator-reward SA phase is ``_educate``. The incumbent is stored in the
    ``gbest_*`` attributes and mirrored onto ``best_profit``/``best_routes`` so the
    template loop can track stagnation.

    Attributes:
        problem (SubProblem): Extracted routing sub-problem.
        cfg (PSOMAConfig): Algorithm-specific parameters.
        params (PSOMAConfig): Alias of ``cfg`` kept for the SA configuration write-back.
        dist_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
        capacity (float): Maximum vehicle collection capacity.
        R (float): Revenue per kg of waste.
        C (float): Cost per km traveled.
        mandatory_nodes (Optional[List[int]]): Nodes that must be visited.
        n_nodes (int): Number of nodes (excluding depot).
        clients (List[int]): List of client IDs.
        random (random.Random): Random number generator.
        np_rng (np.random.Generator): Seeded numpy stream for vector draws.
        split_solver (LinearSplit): Solver for partitioning tours into routes.
        acceptance_criterion (AcceptanceCriterion): Criterion for accepting worse solutions.
        metropolis_steps (int): Number of metropolis steps per iteration.
        operators (List[Callable]): List of mutation operators.
        probabilities (np.ndarray): Probabilities for selecting mutation operators.
        rewards (np.ndarray): Rewards for each mutation operator.
        swarm (List[PSOMAParticle]): The swarm of particles.
        gbest_X (np.ndarray): Global best position vector.
        gbest_giant_tour (np.ndarray): Global best giant tour.
        gbest_routes (List[List[int]]): Global best routes.
        gbest_profit (float): Global best profit.
        gbest_mapping (np.ndarray): Global best mapping.
    """

    def __init__(self, problem: SubProblem, cfg: Optional[Union[PSOMAConfig, Dict[str, Any]]] = None):
        """Initializes the PSOMA solver.

        Args:
            problem: Extracted routing sub-problem (distance matrix,
                wastes, capacity, revenue, cost unit, mandatory nodes).
            cfg: PSOMA configuration
                dataclass, dictionary, or any object with a ``__dict__``; keys are
                filtered onto the PSOMAConfig fields.
        """
        coerced_cfg = self._coerce_config(cfg, PSOMAConfig)
        super().__init__(problem, coerced_cfg)

        self.cfg: PSOMAConfig = coerced_cfg
        self.params = coerced_cfg

        self.dist_matrix = problem.dist_matrix
        self.wastes = problem.wastes
        self.capacity = problem.capacity
        self.R = problem.revenue
        self.C = problem.cost_unit

        self.mandatory_nodes = list(problem.mandatory_nodes) if problem.mandatory_nodes else []
        self.n_nodes = len(self.dist_matrix) - 1
        self.clients = list(range(1, self.n_nodes + 1))
        self.random = random.Random(coerced_cfg.seed) if coerced_cfg.seed is not None else random.Random()
        # One seeded numpy stream for every vector draw, so a run is repeatable outside the simulator's global reseed.
        self.np_rng = np.random.default_rng(coerced_cfg.seed)
        self.split_solver = LinearSplit(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            max_vehicles=0,
            mandatory_nodes=self.mandatory_nodes,
        )
        self.split_solver.profit_aware = bool(getattr(coerced_cfg, "profit_aware_operators", True))

        # Fresh per-solve acceptance criterion instance (P-11)
        from logic.src.policies.acceptance_criteria.base.factory import (
            AcceptanceCriterionFactory,
        )

        acc_val = getattr(self.cfg, "acceptance_criterion", None)
        if hasattr(acc_val, "accept"):
            self.acceptance_criterion = acc_val
        elif acc_val is not None and hasattr(acc_val, "method"):
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val.method,
                config=acc_val.params,
            )
        elif isinstance(acc_val, dict) and "method" in acc_val:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val["method"],
                config=acc_val.get("params"),
            )
        else:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=self.cfg.T0,
                alpha=self.cfg.lambda_cooling,
                seed=self.cfg.seed,
            )
        self.params.acceptance_criterion = self.acceptance_criterion

        # Steps per SA search: capped so one search cannot consume the whole time budget.
        self.metropolis_steps = max(1, min(self.n_nodes * (self.n_nodes - 1), getattr(self.cfg, "sa_steps", 5000)))
        self._deadline = float("inf")
        self.operators = [self._swap, self._insert, self._inverse]
        self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3])
        self.rewards = np.zeros(3)

        self.swarm: List[PSOMAParticle] = []
        self.gbest_X = np.zeros(self.n_nodes)
        self.gbest_giant_tour = np.zeros(self.n_nodes, dtype=int)
        self.gbest_routes: List[List[int]] = []
        self.gbest_profit = -float("inf")
        self.gbest_mapping = np.zeros(self.n_nodes, dtype=int)

        # Construct the Profit-Biased Distance Matrix for O(1) Surrogate Evaluation
        self.mu = 10.0  # Tuning parameter: higher values pull profitable nodes closer
        self.biased_dist_matrix = np.copy(self.dist_matrix)

        # Calculate profit-to-cost ratio (rho) for each node
        rho = np.zeros(self.n_nodes + 1)
        for i in self.clients:
            marginal_cost = self.dist_matrix[0, i] + self.dist_matrix[i, 0]
            if marginal_cost > 0:
                rho[i] = (self.wastes[i] * self.R) / (marginal_cost * self.C)

        # Transform the matrix
        for i in range(1, self.n_nodes + 1):
            for j in range(1, self.n_nodes + 1):
                if i != j:
                    self.biased_dist_matrix[i, j] -= self.mu * ((rho[i] + rho[j]) / 2.0)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Executes the PSO Memetic Algorithm to find an optimal routing solution.

        The algorithm iterates between a global exploration phase (PSO) and
        a local exploitation phase (SA) for a fixed number of iterations or until a time limit is reached.

        Returns:
            Tuple[List[List[int]], float, float]: A tuple containing:
                - The best routing solution found (list of routes).
                - The total profit of the best solution.
                - The total travel cost (distance) of the best solution.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0
        return super().solve()

    # ------------------------------------------------------------------
    # PopulationSearch template hooks
    # ------------------------------------------------------------------

    def _initial_population(self) -> None:
        """Build the swarm and fold the initial best particles into the global best."""
        self._init_swarm()

    def _prepare_intensification(self) -> None:
        """Warm up the acceptance criterion and calibrate operator rewards (training phase)."""
        self.acceptance_criterion.setup(self.gbest_profit)
        self._training_phase()

    def _vary_population(self) -> None:
        """One PSO step over the swarm: velocity update, ROV decode, Split, pbest/gbest tracking."""
        for p in self.swarm:
            r1, r2 = self.np_rng.random(self.n_nodes), self.np_rng.random(self.n_nodes)
            p.V = (
                self.params.omega * p.V
                + self.params.c1 * r1 * (p.pbest_X - p.X)
                + self.params.c2 * r2 * (self.gbest_X - p.X)
            )
            p.V = np.clip(p.V, self.params.v_min, self.params.v_max)
            p.X = np.clip(p.X + p.V, self.params.x_min, self.params.x_max)

            p.giant_tour, p.mapping_indices = p._rov_rule(p.X)
            p.routes, p.profit = self.split_solver.split(p.giant_tour.tolist())
            if p.profit > p.pbest_profit:
                p.pbest_X = np.copy(p.X)
                p.pbest_giant_tour = np.copy(p.giant_tour)
                p.pbest_mapping_indices = np.copy(p.mapping_indices)
                p.pbest_profit = p.profit

            if p.profit > self.gbest_profit:
                self._set_gbest(p.X, p.giant_tour, p.mapping_indices, p.routes, p.profit)

    def _educate(self) -> None:
        """Memetic step: operator-reward SA on the incumbent + criterion cooling step."""
        self._non_training_phase()
        self.acceptance_criterion.step(
            current_obj=self.gbest_profit, candidate_obj=self.gbest_profit, accepted=True
        )

    def _update_probabilities(self) -> None:
        """Updates the probabilities for operator selection based on rewards.

        """
        epsilon = 0.05  # 5% minimum probability for any operator

        # Apply epsilon bound to raw rewards to ensure exploration
        bounded_rewards = np.maximum(epsilon, self.rewards)
        total_reward = np.sum(bounded_rewards)

        self.probabilities = bounded_rewards / total_reward
