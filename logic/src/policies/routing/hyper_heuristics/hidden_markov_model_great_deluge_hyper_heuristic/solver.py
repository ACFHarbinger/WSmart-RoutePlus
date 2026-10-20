"""
Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH) for VRPP.

This online-learning approach treats the sequence of applied Low-Level
Heuristics (LLHs) as a Markov chain. The system maintains a *belief*
distribution over three hidden states — improving, stagnating, escaping —
using the Forward Algorithm, rather than assigning a single deterministic
state.

At each iteration the belief vector P(S_t | O_{1:t}) is updated via the
observation likelihood of the normalised profit change Δ_norm.  The LLH
selection probabilities are then the belief-weighted mixture of the
per-state emission matrices.

The Great Deluge acceptance criterion accepts candidate solutions whose
profit exceeds a linearly rising water level, providing a deterministic
escape from local optima without requiring temperature tuning.

Reference:
    Onsem, W. V., Demoen, B., & Causmaecker, P. D. "HHaaHMM:
    A Hyper-Heuristic as a Hidden Markov Model", 2014
    McMullan, P. "An Extended Implementation of the Great
    Deluge Algorithm for Course Timetabling", 2007

Attributes:
    _OBS_IMPROVE (int): Observation for improved solution.
    _OBS_WORSE (int): Observation for worse solution with same cost.
    _OBS_WORSE_COST (int): Observation for worse solution with higher cost.
    _OBS_SAME (int): Observation for same solution.
    _OBS_SAME_COST (int): Observation for same solution with same cost.
    _N_OBS (int): Number of observations.
    HMMGDHHSolver: Solves VRPP using Hidden Markov Model + Great Deluge.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.hidden_markov_model_great_deluge_hyper_heuristic import HMMGDHHSolver, HMMGDHHConfig, run_hmmgd_hh
    >>> solver = HMMGDHHSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> best_solution, best_profit, best_cost = solver.solve()
    >>> print(best_solution, best_profit, best_cost)

Belief updates live in ``hmm_belief`` and the low-level heuristics live in ``hmm_low_level``. Both are mixed into ``HMMGDHHSolver``. The observation alphabet is defined in ``hmm_belief`` and re-exported here.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.hmm_gd_hh import HMMGDHHConfig
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes

from .hmm_belief import (
    _N_OBS as _N_OBS,
)
from .hmm_belief import (
    _OBS_SAME as _OBS_SAME,
)
from .hmm_belief import (
    _OBS_SAME_COST as _OBS_SAME_COST,
)
from .hmm_belief import (
    HMMBeliefMixin as HMMBeliefMixin,
)
from .hmm_low_level import HMMLowLevelMixin as HMMLowLevelMixin


class HMMGDHHSolver(HMMBeliefMixin, HMMLowLevelMixin):
    r"""
    Hidden Markov Model + Great Deluge Hyper-Heuristic solver (HHaaHMM).

    Aligned with: Onsem, W. V., Demoen, B., & Causmaecker, P. D. "HHaaHMM:
    A Hyper-Heuristic as a Hidden Markov Model", 2014.

    This implementation treats the hidden states as representing the **local topology
    of the search space** evaluated by the heuristics. This topology is modeled as
    an Input-Output Hidden Markov Model (IOHMM), where:
    - **Inputs ($u_t$)**: Low-Level Heuristics (LLHs).
    - **Outputs ($o_t$)**: Discrete categories of solution change ($\Delta$ Profit, $\Delta$ Cost).
    - **Hidden States ($s_t$)**: Structural search phases (e.g., local optima regions).

    Bi-Objective Transformation (Fix #R-Final-1):
    This implementation adapts the original IOHMM observation alphabet. Instead of
    tracking computational time, we repurpose the 'time' dimension to track
    'routing cost' (distance). This transforms the HMM into a bi-objective model
    that learns the trade-off between financial profit and physical routing efficiency.

    Note on Great Deluge Integration:
    The IOHMM learns and updates its belief based on *every* proposed move (exploration),
    mapping the immediate neighborhood dynamics. The Great Deluge (GD) criterion
    independently controls the *acceptance trajectory*.

    Attributes:
        dist_matrix (np.ndarray): Distance matrix.
        wastes (Dict[int, float]): Mapping from node index to waste volume.
        capacity (float): Vehicle capacity.
        R (float): Revenue parameter.
        C (float): Cost parameter.
        params (HMMGDHHConfig): HMM-GD-HH parameters.
        mandatory_nodes (Optional[List[int]]): Optional list of mandatory nodes.
        n_nodes (int): Number of nodes.
        n_llh (int): Number of Low-Level Heuristics.
        n_states (int): Number of hidden states.
        nodes: List of nodes.
        random: Random state.
        _llh_pool (List[Callable]): Pool of Low-Level Heuristics.
        alpha (float): Weight for entropy in action selection.
        gamma (float): Decay factor for online EM counts.
        t (int): Observation counter.
        _counts_A (np.ndarray): Count matrix for transition matrix.
        _counts_B (np.ndarray): Count matrix for emission matrix.
        _A (np.ndarray): Transition matrix.
        _B (np.ndarray): Emission matrix.
        _belief (np.ndarray): Belief matrix.
        _expected_profit (np.ndarray): Expected profit matrix.
        _profit_counts (np.ndarray): Profit counts matrix.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HMMGDHHConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initializes the HMM-GD-HH solver.

        Args:
            dist_matrix: Distance matrix.
            wastes: Mapping from node index to waste volume.
            capacity: Vehicle capacity.
            R: Revenue parameter.
            C: Cost parameter.
            params: HMM-GD-HH parameters.
            mandatory_nodes: Optional list of mandatory nodes.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Dynamic LLH Pool (Fix #R-Final-2: Ablation Blocker)
        # Sliced according to params.n_llh to enable proper research evaluation
        full_pool = [self._llh0, self._llh1, self._llh2, self._llh3, self._llh4, self._llh5, self._llh6]
        self._llh_pool = full_pool[: params.n_llh]
        self.n_llh = len(self._llh_pool)

        # Static parameters
        self.alpha = 0.5  # Weight for entropy in action selection
        self.gamma = 0.9  # Decay factor for online EM counts (optional smoothing)

        # --- Dynamic HHaaHMM State ---

        # Number of hidden states starts small (Fix #3)
        self.n_states = 2
        self.t = 0  # Observation counter

        # Transition matrix A[s_t][s_{t+1}] | input u_t (HHaaHMM uses P(s' | s, u))
        # Initialised with small random noise to break symmetry (Fix #R-Final-2)
        self._A = np.ones((self.n_llh, self.n_states, self.n_states)) / self.n_states
        self._A += np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(self.n_states)] for _ in range(self.n_states)]
                for _ in range(self.n_llh)
            ]
        )
        self._A = np.clip(self._A, 1e-12, 1.0)
        self._A /= self._A.sum(axis=2, keepdims=True)

        # Emission matrix B[s_t][o_t] | input u_t (HHaaHMM uses P(o | s, u))
        # Initialised with small random noise to break symmetry (Fix #R-Final-2)
        self._B = np.ones((self.n_llh, self.n_states, _N_OBS)) / _N_OBS
        self._B += np.array(
            [
                [[self.random.uniform(-0.01, 0.01) for _ in range(_N_OBS)] for _ in range(self.n_states)]
                for _ in range(self.n_llh)
            ]
        )
        self._B = np.clip(self._B, 1e-12, 1.0)
        self._B /= self._B.sum(axis=2, keepdims=True)

        # State belief: P(state | history)
        self._belief = np.ones(self.n_states) / self.n_states

        # EM counters (for stochastic online EM estimation)
        self._counts_A = np.zeros((self.n_llh, self.n_states, self.n_states))
        self._counts_B = np.zeros((self.n_llh, self.n_states, _N_OBS))

        # --- Refinement 1: State-Aware Expected Profit (Fix #1) ---
        # Matrix shape: (n_states, n_llh)
        self._expected_profit = np.zeros((self.n_states, self.n_llh))
        self._profit_counts = np.zeros((self.n_states, self.n_llh))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, initial_routes: List[List[int]]) -> Tuple[List[List[int]], float, float]:
        """
        Main HH-GD loop for VRPP.

        Following the principle of "calculating a good heuristic at any point in time
        given the old data" (Onsem et al. 2014, Section 6), the IOHMM learns the
        local neighborhood topology from every proposed move. The Great Deluge
        threshold then filters these proposals to manage the global search trajectory.

        Args:
            initial_routes: Initial routes for the VRPP.

        Returns:
            Tuple[List[List[int]], float, float]: Tuple of (best_routes, best_profit, best_cost).
        """
        routes = copy.deepcopy(initial_routes)
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit
        best_cost = self._cost(routes)

        start_time = time.perf_counter()

        # Great Deluge: water level starts linearly rising
        water_level = (
            best_profit * (1.0 - self.params.flood_margin) if best_profit > 0 else -abs(self.params.flood_margin)
        )

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            # 1. Action Selection: Entropy-Maximizing (Fix #5, #R1, #R-Final-1 & #Final-3)
            # score(u) = NormalizedProfit(u) + annealed_alpha * Entropy(P(s'|u))
            u_idx = self._select_action(iteration, self.params.max_iterations)
            llh = self._llh_pool[u_idx]

            # 2. Dynamic State Scaling (Fix #3)
            self.t += 1
            self._check_state_scaling()

            # Store belief prior to action (for state-aware profit update)
            previous_belief = self._belief.copy()

            # 3. Apply LLH and observe outcomes
            # The HMM learns from all moves (exploration), uncoupled from Great Deluge acceptance (Fix #R3)
            old_cost = self._cost(routes)
            try:
                new_routes = llh(routes, self.params.n_removal)
                new_profit = self._evaluate(new_routes)
                new_cost = self._cost(new_routes)
            except Exception:
                new_routes = routes
                new_profit = profit
                new_cost = old_cost

            # 4. Map to discrete observation alphabet (Fix #R-Final-1)
            o_idx = self._map_observation(new_profit - profit, new_cost - old_cost)

            # 5. Online EM update (Fix #R-Final-3: Stochastic Online EM)
            self._online_em_update(u_idx, o_idx)

            # 6. Refinement 1: State-Aware Profit Update (Fix #R1 & #R-Final-2)
            # update E[p | s, u] for all s weighted by P(s | history)
            delta_profit = new_profit - profit

            # --- Fix #R-Final-2: Wasted Effort Penalty ---
            # If the move is idempotent (SAME or SAME_COST), apply a small penalty
            # even if delta_profit is 0, to encourage search progression.
            if o_idx in [_OBS_SAME, _OBS_SAME_COST]:
                delta_profit -= 1e-3

            for s_idx in range(self.n_states):
                self._profit_counts[s_idx, u_idx] += previous_belief[s_idx]
                # Incremental online average of profit weighted by state belief
                step = previous_belief[s_idx] / (self._profit_counts[s_idx, u_idx] + 1e-12)
                self._expected_profit[s_idx, u_idx] += step * (delta_profit - self._expected_profit[s_idx, u_idx])

            # Great Deluge acceptance
            if new_profit >= water_level:
                routes = new_routes
                profit = new_profit
                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit
                    best_cost = self._cost(best_routes)

            # Increase water level
            water_level += self.params.rain_speed * abs(best_profit + 1e-9)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                water_level=water_level,
                hmm_state=int(np.argmax(self._belief)),
                llh_selected=u_idx,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().

        Returns:
            List[List[int]]: List of routes.
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return optimized_routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes.

        Args:
            routes: List of routes.

        Returns:
            Net profit.
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total routing distance.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
