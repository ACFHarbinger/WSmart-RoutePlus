# Comprehensive Analysis: Acceptance Criteria in Route Construction

This document evaluates the `logic/src/policies/route_construction` package of the `WSmart-Route` codebase, comprehensively indexing **every single algorithm**. We assess whether popular meta-heuristics, hyper-heuristics, and matheuristics seamlessly leverage the newly established `acceptance_criteria/` structure, or if they rely on hardcoded and localized conditionals to accept iterative solver moves.

Our core finding validates the initial thesis: Despite the existence of 17 completely decoupled, typed, and mathematically sound criteria classes (e.g., `FitnessProportionalAcceptance`, `MonteCarloAcceptance`, `GreatDelugeAcceptance`), **zero algorithms actually orchestrate or depend upon them**. Currently, acceptance execution is entirely hardcoded deeply into specific routines, redundant across multiple modules, or evaluated deterministically as part of an exact execution layer.

---

## 1. Modular Acceptance Criteria Modules

The `logic/src/policies/route_construction/acceptance_criteria/` defines the following 17 distinct, interface-conforming (`IAcceptanceCriterion`) mathematical standards:

| Acceptance Criterion | Reference Paper | Small Description | Key Equation(s) |
| :--- | :--- | :--- | :--- |
| **All Moves** | Baseline / Random Walk | Accepts all incoming proposed solutions regardless of objective degradation. | $P(accept) = 1$ |
| **Aspiration Criterion** | Glover (1989) | Reverses a Tabu status if the move strictly beats the global known best. | $f\_{cand} > f\_{global\_best}$ |
| **Boltzmann Metropolis** | Metropolis et al. (1953) | Probabilistically accepts worsening moves dependent on an exponential temperature matrix. | $P(A) = e^{-\Delta f / T}$ |
| **Ensemble Move** | Ozcan et al. (2008) | Votes across multiple criteria objects and returns an aggregate mathematical decision. | $V = \sum w_i \cdot I_i(accept)$ |
| **Fitness Proportional** | Holland (1975) | Roulette-Wheel selection. Acceptance probabilistically scaled relative to total fitness. | $P(A) = \frac{f\_{cand}}{f\_{cur} + f\_{cand}}$ |
| **Great Deluge** | Dueck (1993) | Rejects solutions dropping below a mathematical "water level" constraint that steadily rises. | $f\_{cand} \ge Level$ |
| **Improving & Equal** | Classic Hill Climbing | Accepts moves that are strictly better or completely equivalent to current cost. | $f\_{cand} \ge f\_{cur}$ |
| **Late Acceptance** | Burke & Bykov (2017) | Compares candidate against a historical incumbent `L` trajectory steps ago. | $f\_{cand} \ge f\_{history}[i \pmod L]$ |
| **Monte Carlo** | Random Choice | Fixed-probability acceptance metric applied identically against any worsening move. | $P(A \mid worsening) = p$ |
| **Old Bachelor** | Hu et al. (1995) | Dynamic thresholding strategy that contracts on success and dilates upon failure. | $f\_{cand} > f\_{cur} - \tau\_{dynamic}$ |
| **Only Improving** | Classic Greedy descent | Strictest elitist logic. Instantly disregards equivalent or worsening limits. | $f\_{cand} > f\_{cur}$ |
| **Pareto Dominance** | Multiobjective Opt | Requires strict metric vector superiority (no value degradation permitted across the vector array). | $f\_{cand} \succ_{pareto} f\_{cur}$ |
| **Probabilistic Transition**| Dorigo (1992) | Ant Colony proportional evaluation rule (scaled aggressively by parameterized $\alpha$ factor). | $P(A) = \frac{f\_{cand}^\alpha}{f\_{cur}^\alpha + f\_{cand}^\alpha}$ |
| **Record-to-Record** | Dueck (1993) | Accepts any moves deviating by at most `tolerance`% strictly from the global best benchmark. | $f\_{cand} \ge f\_{best} - \text{dev}$ |
| **Step Counting Hill** | Bykov (2003) | Locks in a fixed comparative bound for `K` strict iteration steps before explicitly resetting. | $f\_{cand} \ge bound\_{stagnant}$ |
| **Threshold Accepting** | Dueck & Scheuer (1990) | Deterministic SA derivative. Strictly decays acceptable mathematical worsening linearly toward 0. | $f\_{cand} \ge f\_{cur} - T\_{decay}$ |
| **Tournament** | Goldberg (1991) | Candidate mathematically wins/loses a loaded stochastic dice roll against the current solution. | $P(cand\_wins) = p$ |

---

## 2. Meta-Heuristics (39 Algorithms)

Meta-heuristics govern the iterative stochastic exploration of trajectory and population-based spaces.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **adaptive_large_neighborhood_search** | Simulated Annealing | **Hardcoded**. Utilizes string flags inside `if` conditionals (`"sa"`). |
| **ant_colony_optimization_k_sparse** | Probabilistic Transition | **Hardcoded**. Transition rule mathematics directly executed via matrix probabilities in `solver.py`. |
| **artificial_bee_colony** | Greedy / Nectar | **Hardcoded**. Worker roles individually test `if trial < current` inline. |
| **augmented_hybrid_volleyball_premier_league** | Substitution Policy | **Hardcoded**. Rank-based list selection applied iteratively. |
| **differential_evolution** | Only Improving (Greedy) | **Hardcoded**. Vector mutations enforce hard dominance logic during generations. |
| **evolution_strategy_mu_comma_lambda** | Fitness Selection / Elitism | **Hardcoded**. Custom selection mechanism. |
| **evolution_strategy_mu_kappa_lambda** | Fitness Selection | **Hardcoded**. |
| **evolution_strategy_mu_plus_lambda** | Fitness Selection | **Hardcoded**. |
| **fast_iterative_localized_optimization** | Only Improving | **Hardcoded**. Simple greedy ascent conditions limit iterations. |
| **firefly_algorithm** | Only Improving (Attractiveness) | **Hardcoded**. Formula is $e^{-\gamma r^2}$. |
| **genetic_algorithm** | Tournament / Fitness Proportional | **Hardcoded**. Internal helper toggles parameters. |
| **genius** | Only Improving | **Hardcoded**. Explicit checks for unstringing/stringing heuristics. |
| **guided_local_search** | Only Improving (Penalty Map) | **Hardcoded**. Augmented targets handled via implicit dictionary calls. |
| **harmony_search** | Acceptance Pitch Adjustment | **Hardcoded**. |
| **hybrid_genetic_search** | Biased Fitness / Diversity | **Hardcoded**. |
| **hybrid_genetic_search_adaptive_large_neighborhood_search** | Biased Fitness + SA | **Hardcoded**. |
| **hybrid_genetic_search_ruin_and_recreate** | Biased Fitness | **Hardcoded**. |
| **hybrid_memetic_search** | Best Reinsertion | **Hardcoded**. |
| **hybrid_volleyball_premier_league** | Rank-Based | **Hardcoded**. |
| **iterated_local_search** | Threshold / Only Improving | **Hardcoded**. |
| **knowledge_guided_local_search** | Only Improving | **Hardcoded**. |
| **league_championship_algorithm** | Match-Based | **Hardcoded**. |
| **memetic_algorithm** | Greedy Survivor | **Hardcoded**. |
| **memetic_algorithm_dual_population** | Greedy Survivor | **Hardcoded**. |
| **memetic_algorithm_island_model** | Greedy Survivor / Migration | **Hardcoded**. |
| **memetic_algorithm_tolerance_based_selection** | Tolerance Thresholds | **Hardcoded**. |
| **particle_swarm_optimization** | Global/Local Best Updating | **Hardcoded**. |
| **particle_swarm_optimization_distance_based_algorithm** | Local Best Updating | **Hardcoded**. |
| **particle_swarm_optimization_memetic_algorithm** | Local Best Updating | **Hardcoded**. |
| **quantum_differential_evolution** | Greedy Selection | **Hardcoded**. |
| **reactive_tabu_search** | Tabu List + Dynamics | **Hardcoded**. Checks iterations inline. |
| **simulated_annealing** | Boltzmann Metropolis | **Hardcoded**. |
| **simulated_annealing_neighborhood_search** | Boltzmann Metropolis | **Hardcoded**. |
| **sine_cosine_algorithm** | Sine/Cosine Scale Updating | **Hardcoded**. |
| **slack_induction_by_string_removal** | Only Improving | **Hardcoded**. |
| **soccer_league_competition** | League Rank Updating | **Hardcoded**. |
| **tabu_search** | Only Improving / Aspiration | **Hardcoded**. |
| **variable_neighborhood_search** | Only Improving | **Hardcoded**. |
| **volleyball_premier_league** | Rank Base | **Hardcoded**. |

---

## 3. Hyper-Heuristics (6 Algorithms)

Hyper-heuristics search across the space of heuristic operators.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **ant_colony_optimization_hyper_heuristic** | ACO Transition | **Hardcoded**. Evaluates operator probabilities. |
| **genetic_programming_hyper_heuristic** | Evolutionary Elitist | **Hardcoded**. |
| **guided_indicators_hyper_heuristic** | Selection Probability Updating | **Hardcoded**. |
| **hidden_markov_model_great_deluge_hyper_heuristic** | Moving Lower/Upper Bound | **Hardcoded**. Even though Great Deluge logic exists, it is evaluated locally instead of globally referenced. |
| **hyper_heuristic_us_lk** | Lin-Kernighan Improvement | **Hardcoded**. |
| **sequence_based_selection_hyper_heuristic** | Markov Chain Based | **Hardcoded**. |

---

## 4. Matheuristics (9 Algorithms)

Matheuristics fuse heuristic architectures with exact mathematical programming components.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **adaptive_kernel_search** | MIP Bounds Improvement | **N/A**. Managed entirely by Gurobi tolerances. |
| **cluster_first_route_second** | Set Partitioning Objective | **N/A**. Exact evaluation phase strictly accepts optimal results. |
| **iterated_local_search_randomized_vns_set_partitioning**| Only Improving | **Hardcoded**. Hybridized local-search conditional boundaries. |
| **kernel_search** | Restricted MIP Optimality | **N/A**. |
| **lin_kernighan_helsgaun_three** | Ascending Penalty | **Hardcoded**. Deeply embedded conditional evaluations for LKH. |
| **local_branching** | MIP K-Neighborhoods | **N/A**. |
| **local_branching_variable_neighborhood_search** | Relaxed Bound Iteration | **Hardcoded**. |
| **partial_optimization_metaheuristic** | Re-Optimization Improving | **Hardcoded**. |
| **relaxation_enforced_neighborhood_search** | LP Relaxation Rounding | **N/A**. |

---

## 5. Exact & Decomposition Solvers (12 Algorithms)

Exact Solvers yield optimally verified solutions via mathematical formulations. Since acceptance criteria specifically operate over isolated stochastic local perturbations, *acceptance criteria mechanics traditionally do not apply here*. They act on mathematical constraints and optimality guarantees directly.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **branch_and_bound** | Absolute LB/UB Pruning | **N/A (Exact)** |
| **branch_and_cut** | Absolute LB/UB Pruning | **N/A (Exact)** |
| **branch_and_price** | Column Generation Costs | **N/A (Exact)** |
| **branch_and_price_and_cut** | Farkas/Reduced Costs | **N/A (Exact)** |
| **constraint_programming_with_boolean_satisfiability**| Boolean True/False | **N/A (Exact)** |
| **exact_stochastic_dynamic_programming** | Value Function Iteration | **N/A (Exact)** |
| **integer_l_shaped_benders_decomposition** | Benders Cuts Optimality | **N/A (Exact)** |
| **logic_based_benders_decomposition** | Logic Cuts Optimality | **N/A (Exact)** |
| **progressive_hedging** | Consensus Variables | **N/A (Heuristic-Exact)** |
| **scenario_tree_extensive_form** | Deterministic Equivalent | **N/A (Exact)** |
| **smart_waste_collection_two_commodity_flow** | Formulation Bounds | **N/A (Exact)** |

---

## 6. Learning & Heuristic Learning Models (5 Algorithms)

Learning algorithms train network layers or explicit policy tables across multiple epochs.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **neural_agent** (*learning_algorithms*) | Policy Gradient / Max-Reward | **Modular via RL APIs**. The objective function natively replaces trajectory acceptance methods in favor of continuous optimization losses. |
| **reinforcement_learning_adaptive_large_neighborhood_search** | RL Operator Q-Learning + SA | **Hardcoded / Modular Hybrid**. |
| **reinforcement_learning_augmented_hybrid_volleyball_premier_league** | RL Exploration / Action Values | **Hardcoded / Modular Hybrid**. |
| **reinforcement_learning_great_deluge_hyper_heuristic** | Bandits + Deluge Decay | **Hardcoded / Modular Hybrid**. |
| **reinforcement_learning_hybrid_volleyball_premier_league** | Bandits + Strategy Action Values | **Hardcoded / Modular Hybrid**. |

---

## 7. Other Algorithms (2 Algorithms)

These exist largely as standard or historical base implementations.

| Algorithm Directive | Acceptance Concept | Implementation Status |
| :--- | :--- | :--- |
| **capacitated_vehicle_routing_problem** | Problem Base Def | **N/A**. |
| **travelling_salesman_problem** | Problem Base Def | **N/A**. |

---

### Conclusion
A major architectural refactor is highly required if the goal is system-wide interoperability. `WSmart-Route`'s `route_construction` pipeline currently defines 74 distinct computational routing mechanics, nearly all of which isolate and hard-code their iteration and survival boundaries. Linking the `acceptance_criteria/` directory as a factory-driven injection parameter to each heuristic's `solver.py` loop would drastically cut code redundancy and explode cross-configuration capability.
