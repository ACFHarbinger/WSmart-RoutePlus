# Comprehensive Analysis: Acceptance Criteria in Route Construction

**Project**: WSmart+ Route
**Date**: August 27, 2026
**Purpose**: Systematic review of acceptance criteria implementations in `logic/src/policies/acceptance_criteria/` and their integration across route construction algorithms (`logic/src/policies/route_construction/`).
**Total Criteria Modules**: 23 modular criteria (implementing `IAcceptanceCriterion`)
**Total Algorithms Evaluated**: 80+ routing algorithms across exact, metaheuristic, matheuristic, hyper-heuristic, and learning paradigms

---

## Executive Summary

This document evaluates the acceptance criteria architecture in the `WSmart+ Route` codebase. We assess whether metaheuristics, hyper-heuristics, and matheuristics leverage the decoupled `logic/src/policies/acceptance_criteria/` infrastructure or rely on hardcoded local conditionals.

### Key Findings

1. **Modular Architecture**: 23 decoupled, typed, and mathematically sound acceptance criteria classes implementing `IAcceptanceCriterion` are defined in `logic/src/policies/acceptance_criteria/`.
2. **Acceptance Spectrum**: Coverage spans classical greedy descent (`OnlyImproving`), thermal/stochastic (`BoltzmannMetropolis`, `AdaptiveBoltzmannMetropolis`, `GeneralizedTsallisSA`), historical/threshold (`LateAcceptance`, `RecordToRecord`, `GreatDeluge`, `NonLinearGreatDeluge`, `OldBachelor`, `ThresholdAccepting`, `StepCountingHillClimbing`), game/tournament (`BinaryTournament`, `FitnessProportional`, `ProbabilisticTransition`), physics/energy (`DemonAlgorithm`), multiobjective (`ParetoDominance`, `EpsilonDominance`), and structural search (`SkewedVNS`).
3. **Integration Status**: While modular criteria are fully tested in unit suites (`logic/test/unit/policies/test_acceptance_criteria.py`), several legacy metaheuristics continue to embed inline `if candidate_cost < current_cost` or local cooling loops. Standardized factory injection via `BaseRoutingPolicy` and Hydra configuration parameters provides the target unification pathway.

---

## 1. Modular Acceptance Criteria Modules (23 Standards)

The `logic/src/policies/acceptance_criteria/` package defines 23 distinct mathematical acceptance criteria conforming to the `IAcceptanceCriterion` protocol:

| Acceptance Criterion | Reference Paper | Description | Mathematical Rule / Formulation |
| :--- | :--- | :--- | :--- |
| **All Moves** | Baseline / Random Walk | Accepts all incoming proposed solutions unconditionally. | $P(\text{accept}) = 1.0$ |
| **Aspiration Criterion** | Glover (1989) | Reverses Tabu status if candidate strictly improves upon global best. | $f_{\text{cand}} > f_{\text{best}}$ |
| **Boltzmann Metropolis** | Metropolis et al. (1953) | Standard SA exponential temperature-decay acceptance for worsening moves. | $P(\text{accept}) = \exp(-\Delta f / T)$ |
| **Adaptive Boltzmann** | Kirkpatrick (1983) | Temperature adjusted dynamically based on recent acceptance rate $\chi$. | $T_{k+1} = T_k \cdot (1 + \frac{\chi - \chi_{\text{target}}}{\chi_{\text{target}}})$ |
| **Binary Tournament** | Goldberg (1991) | Stochastic selection between candidate and incumbent with winning probability $p$. | $P(\text{cand wins}) = p$ |
| **Demon Algorithm** | Kuske (1997) | Maintains energy credits (demon) from improvements to fund future worsening moves. | Accept if $\Delta E \le E_{\text{demon}}$; $E_{\text{demon}} \leftarrow E_{\text{demon}} - \Delta E$ |
| **Ensemble Move** | Özcan et al. (2008) | Weighted voting ensemble across multiple constituent criteria. | $\sum w_i \cdot \mathbb{I}_i(\text{accept}) \ge \theta_{\text{threshold}}$ |
| **Epsilon Dominance** | Laumanns et al. (2002) | Approximate Pareto dominance with grid granularity $\varepsilon$. | $(1+\varepsilon) \mathbf{f}_{\text{cand}} \succeq \mathbf{f}_{\text{cur}}$ |
| **Exponential Monte Carlo** | Ayob & Kendall (2003) | Exponential counter scaling acceptance probability over successive non-improvements. | $P(\text{accept}) = \exp(-\Delta f \cdot c / T)$ |
| **Fitness Proportional** | Holland (1975) | Roulette-wheel proportional probability relative to combined candidate fitness. | $P(\text{accept}) = \frac{f_{\text{cand}}}{f_{\text{cur}} + f_{\text{cand}}}$ |
| **Generalized Tsallis SA** | Tsallis & Stariolo (1996) | Non-extensive statistical mechanics acceptance with entropic index $q$. | $P(\text{accept}) = [1 - (1-q)\Delta f / T]^{1/(1-q)}$ |
| **Great Deluge** | Dueck (1993) | Rejects solutions dropping below a linearly/exponentially rising water level $B$. | $f_{\text{cand}} \ge B$; $B_{k+1} = B_k + \Delta B$ |
| **Non-Linear Great Deluge**| Landa-Silva & Mbititi (2010)| Non-linear adaptive water level decay based on stagnation and search velocity. | $B_{k+1} = B_k + \gamma \cdot \exp(-\beta k) \cdot \Delta f$ |
| **Improving & Equal** | Classic Hill Climbing | Accepts moves that are strictly better or completely equivalent in objective value. | $f_{\text{cand}} \ge f_{\text{cur}}$ |
| **Late Acceptance** | Burke & Bykov (2017) | Compares candidate against a historical incumbent $L$ iterations in the past. | $f_{\text{cand}} \ge f_{\text{history}}[i \pmod L]$ |
| **Monte Carlo** | Classic Random Search | Fixed-probability acceptance for any worsening move. | $P(\text{accept} \mid \text{worsening}) = p_0$ |
| **Old Bachelor** | Hu, Kahng & Tsao (1995) | Dynamic threshold that lowers upon successful moves and raises on stagnation. | $f_{\text{cand}} \ge f_{\text{cur}} - \tau_k$; $\tau_{k+1} = \tau_k \pm \Delta \tau$ |
| **Only Improving** | Pure Greedy Descent | Strictest elitist criterion; rejects all equivalent or worsening transitions. | $f_{\text{cand}} > f_{\text{cur}}$ |
| **Pareto Dominance** | Multiobjective VRP | Requires strict multi-attribute vector non-inferiority. | $\mathbf{f}_{\text{cand}} \succ \mathbf{f}_{\text{cur}}$ |
| **Probabilistic Transition**| Dorigo et al. (1996) | Power-scaled Ant Colony proportional transition rule. | $P(\text{accept}) = \frac{f_{\text{cand}}^\alpha}{f_{\text{cur}}^\alpha + f_{\text{cand}}^\alpha}$ |
| **Record-to-Record** | Dueck (1993) | Accepts any moves deviating by at most tolerance $\delta$ from global best incumbent. | $f_{\text{cand}} \ge f_{\text{best}} - \delta$ |
| **Skewed VNS** | Hansen et al. (2000) | Distance-skewed acceptance penalizing moves that explore too close to current basin. | $f_{\text{cand}} - \alpha \cdot d(\text{cand}, \text{cur}) \ge f_{\text{cur}}$ |
| **Step Counting Hill** | Bykov & Petrovic (2016) | Maintains fixed comparison benchmark for $K$ consecutive steps before updating. | $f_{\text{cand}} \ge f_{\text{step}}$; update $f_{\text{step}}$ every $K$ steps |
| **Threshold Accepting** | Dueck & Scheuer (1990) | Deterministic annealing derivative; worsening allowed within decaying tolerance $T$.| $f_{\text{cand}} \ge f_{\text{cur}} - T_k$; $T_{k+1} = \alpha T_k$ |

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
