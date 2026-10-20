"""
LKH-2 Meta-Heuristic — Full Implementation (Helsgaun 2009 / 2012).

LKH-2 extends LKH-1 with a **population-based** outer loop that diversifies
the search beyond what a single-trajectory ILS can achieve.  The key additions
over LKH-1 are:

1. **Population Management** (Section 2, Helsgaun 2009):
   Maintains a fixed-size population of locally-optimal tours.  New members
   replace the worst current member when they improve upon it or when the new
   tour is sufficiently distinct from all existing members (diversity
   preservation).

2. **Iterative Partial Transcription (IPT) Crossover** (Section 3):
   Generates offspring tours by combining two parent tours:
   - Identify edges shared by both parents (the "backbone").
   - Copy backbone edges to the offspring.
   - Fill remaining gaps with the LKH-1 local search, treating each gap as
     a sub-tour to optimise.
   This is the key operator that makes LKH-2 a *meta*-heuristic rather than
   merely a better-parameterised LKH-1.

3. **Shared Penalty Vector** (Section 2.2):
   The subgradient-derived penalty vector π is computed once from the first
   LKH-1 solution and then shared across all population members.  It defines
   the candidate set for every crossover offspring and local-search call.
   This avoids redundant subgradient runs.

4. **Population-Informed Restarts** (Section 4):
   Rather than always perturbing the best tour (as in ILS), LKH-2 selects
   parent pairs from the population probabilistically (fitness-proportionate
   with diversity bonus) to seed crossover.  Pure ILS restarts (double-bridge)
   are applied only when the population has fully converged.

Architecture
------------
This module owns:

- :class:`TourPopulation` — population container with insertion / selection.
- :func:`ipt_crossover`   — Iterative Partial Transcription operator.
- :func:`solve_lkh2`      — top-level LKH-2 meta-heuristic.

Heavy computation is delegated to:

- :func:`solve_lkh1` (from ``lkh1.py``) for Phase 1 (subgradient + penalized
  candidates + local search of each new solution).
- :func:`_improve_tour` (from ``lin_kernighan_helsgaun.py``) for the inner
  k-opt loop within IPT gap-filling.
- :func:`compute_alpha_measures`, :func:`get_candidate_set` (from
  ``lin_kernighan_helsgaun.py``) for penalized candidates.
- :func:`_double_bridge_kick` (from ``_tour_construction.py``) for pure ILS
  restarts when population diversity collapses.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two import solve_lkh
    >>> tour, cost = solve_lkh(dist_matrix)

References:
----------
Helsgaun, K. (2009). General k-opt submoves for the Lin-Kernighan-Helsgaun
  TSP heuristic. Mathematical Programming Computation, 1(2-3), 119–163.

Helsgaun, K. (2012). LKH-2: A new version of LKH.  Technical Report,
  Roskilde University.
"""

from __future__ import annotations

from random import Random
from typing import List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
)
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import (
    _improve_tour,
    compute_alpha_measures,
    get_candidate_set,
    run_subgradient,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .lin_kernighan_helsgaun_two_crossover import (
    ipt_crossover as ipt_crossover,
)
from .lin_kernighan_helsgaun_two_population import (
    TourPopulation as TourPopulation,
)

__all__ = [
    "solve_lkh",
    "solve_lkh2",
    "TourPopulation",
    "ipt_crossover",
]

# ---------------------------------------------------------------------------
# LKH-2 Main Solver
# ---------------------------------------------------------------------------


def solve_lkh2(  # noqa: C901
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 200,
    max_k: int = 5,
    n_candidates: int = 5,
    population_size: int = 10,
    diversity_threshold: float = 0.05,
    crossover_prob: float = 0.6,
    sg_max_iter: int = 100,
    sg_mu_init: float = 1.0,
    sg_patience: int = 20,
    local_search_iter_per_offspring: int = 50,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve a TSP instance using the LKH-2 meta-heuristic (Helsgaun 2009).

    Phase 1 — Penalty Initialisation (shared across all restarts):
        Run subgradient optimisation (:func:`run_subgradient`) once to compute
        π* and the penalized candidate sets that guide ALL subsequent searches.

    Phase 2 — Population Seeding:
        Generate ``population_size`` initial locally-optimal tours by running
        LKH-1 local search from different starting points (nearest-neighbour
        from random seeds).  All use the shared penalized candidates.

    Phase 3 — Crossover-based Diversification loop:
        For each iteration:
        a. With probability ``crossover_prob``: select two parent tours from
           the population; apply IPT crossover to generate an offspring; locally
           optimise the offspring.
        b. Otherwise: apply double-bridge to the best tour and locally optimise.
        c. Try to insert the new tour into the population.
        d. Update the global best.

    Args:
        distance_matrix: (n × n) raw symmetric cost matrix.
        initial_tour: Optional starting tour.
        max_iterations: Total iteration budget (crossover + ILS).
        max_k: k-opt depth (2–5).
        n_candidates: Candidate list size per node.
        population_size: Size of the tour population.
        diversity_threshold: Minimum fractional Hamming distance for
            diversity-driven population insertion.
        crossover_prob: Probability of applying IPT crossover vs. pure
            ILS double-bridge restart.
        sg_max_iter: Subgradient iteration budget.
        sg_mu_init: Initial Polyak step multiplier.
        sg_patience: μ-halving patience.
        local_search_iter_per_offspring: k-opt passes per new solution.
        recorder: Optional telemetry recorder.
        np_rng: NumPy Generator.
        seed: Alternative seed.

    Returns:
        Tuple[List[int], float, float]: (best_tour, best_cost, hk_bound).
    """
    n = len(distance_matrix)
    if n < 3:
        t = list(range(n)) + [0]
        return t, float(get_cost(t, distance_matrix)), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed if seed is not None else 42)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # -----------------------------------------------------------------
    # Phase 1: One-time subgradient to get π* and penalized candidates
    # -----------------------------------------------------------------
    pi, hk_bound, penalized_dm = run_subgradient(
        distance_matrix,
        max_iter=sg_max_iter,
        mu_init=sg_mu_init,
        patience=sg_patience,
    )

    alpha_pen = compute_alpha_measures(penalized_dm)
    candidates = get_candidate_set(penalized_dm, alpha_pen, max_candidates=n_candidates)

    if recorder is not None:
        recorder.record(engine="lkh2_subgradient", hk_bound=hk_bound)

    # -----------------------------------------------------------------
    # Phase 2: Seed the population with locally-optimal tours
    # -----------------------------------------------------------------
    population = TourPopulation(max_size=population_size, diversity_threshold=diversity_threshold)

    # Seed 0: from the initial tour / nearest-neighbour
    seed_tour = _initialize_tour(distance_matrix, initial_tour)
    seed_cost = get_cost(seed_tour, distance_matrix)
    dont_look_seed: Optional[np.ndarray] = np.zeros(n, dtype=bool)
    for _ in range(local_search_iter_per_offspring):
        seed_tour, seed_cost, improved, dont_look_seed = _improve_tour(
            seed_tour,
            seed_cost,
            candidates,
            distance_matrix,
            stdlib_rng,
            dont_look_seed,
            max_k,
        )
        if not improved:
            break
    population.try_insert(seed_tour, seed_cost)

    # Additional seeds: random double-bridge perturbations from seed 0
    for _k in range(population_size - 1):
        alt_tour = _double_bridge_kick(seed_tour, distance_matrix, stdlib_rng)
        alt_cost = get_cost(alt_tour, distance_matrix)
        alt_dont_look: Optional[np.ndarray] = np.zeros(n, dtype=bool)
        for _ in range(local_search_iter_per_offspring):
            alt_tour, alt_cost, improved, alt_dont_look = _improve_tour(
                alt_tour,
                alt_cost,
                candidates,
                distance_matrix,
                stdlib_rng,
                alt_dont_look,
                max_k,
            )
            if not improved:
                break
        population.try_insert(alt_tour, alt_cost)

    best_tour = population.best_tour[:]
    best_cost = population.best_cost

    if recorder is not None:
        recorder.record(
            engine="lkh2_population_init",
            pop_size=population.size,
            pop_diversity=population.diversity,
            best_cost=best_cost,
        )

    # -----------------------------------------------------------------
    # Phase 3: Crossover + ILS meta-loop
    # -----------------------------------------------------------------
    for iteration in range(max_iterations):
        # Decide: crossover or pure ILS
        use_crossover = float(np_rng.random()) < crossover_prob and population.size >= 2

        if use_crossover:
            # Select two parent tours from population
            parents = population.sample_parents(np_rng, n_parents=2)
            p1_tour, _ = parents[0]
            p2_tour, _ = parents[1]

            # Generate offspring via IPT crossover + local search
            offspring, offspring_cost = ipt_crossover(
                p1_tour,
                p2_tour,
                distance_matrix,
                candidates,
                stdlib_rng,
                max_k=max_k,
                max_gap_iter=local_search_iter_per_offspring,
            )
        else:
            # Pure ILS: double-bridge from best tour
            offspring = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            offspring_cost = get_cost(offspring, distance_matrix)
            dont_look_ils: Optional[np.ndarray] = np.zeros(n, dtype=bool)
            for _ in range(local_search_iter_per_offspring):
                offspring, offspring_cost, improved, dont_look_ils = _improve_tour(
                    offspring,
                    offspring_cost,
                    candidates,
                    distance_matrix,
                    stdlib_rng,
                    dont_look_ils,
                    max_k,
                )
                if not improved:
                    break

        # Try to add to population
        population.try_insert(offspring, offspring_cost)

        # Update global best
        if is_better(offspring_cost, best_cost):
            best_cost = offspring_cost
            best_tour = offspring[:]

        if recorder is not None:
            recorder.record(
                engine="lkh2",
                iteration=iteration,
                best_cost=best_cost,
                hk_bound=hk_bound,
                gap=(best_cost - hk_bound) / max(abs(hk_bound), 1.0),
                pop_size=population.size,
                pop_diversity=population.diversity,
                used_crossover=use_crossover,
            )

        # Early termination: optimality gap closed
        if hk_bound > 0 and (best_cost - hk_bound) / hk_bound < 1e-4:
            break

    return best_tour, best_cost, hk_bound


# ---------------------------------------------------------------------------
# Convenience wrapper matching the solve_lkh interface
# ---------------------------------------------------------------------------


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 200,
    max_k: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Thin wrapper for :func:`solve_lkh2` matching the two-return interface
    of ``lin_kernighan_helsgaun.solve_lkh`` and ``lkh1.solve_lkh``.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).
        max_iterations: Maximum LKH-2 main loop iterations.
        max_k: Maximum k-opt move size (3, 4, or 5).
        recorder: Optional telemetry recorder.
        np_rng: NumPy Generator.
        seed: Alternative seed.

    Returns:
        Tuple[List[int], float]: (best_tour, best_cost) without the Held-Karp bound.
    """
    tour, cost, _ = solve_lkh2(
        distance_matrix,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        max_k=max_k,
        recorder=recorder,
        np_rng=np_rng,
        seed=seed,
    )
    return tour, cost
