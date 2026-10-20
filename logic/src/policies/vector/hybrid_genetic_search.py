"""Vectorized Hybrid Genetic Search (HGS) algorithm.

This module implements the HGS algorithm for the Vehicle Routing Problem,
following the principles of diversity-aware population management and
intensified local search (Education). It is fully vectorized to allow
parallel evolution of multiple batch instances on the GPU.

Attributes:
    VectorizedHGS: Vectorized version of the Hybrid Genetic Search algorithm.

Example:
    None
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch

from logic.src.policies.vector.core.crossover import vectorized_ordered_crossover
from logic.src.policies.vector.core.population import VectorizedPopulation
from logic.src.policies.vector.operators import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from logic.src.policies.vector.shared import vectorized_linear_split
from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedHGS(PolicyVizMixin):
    """Vectorized Hybrid Genetic Search (HGS) Solver.

    Orchestrates the HGS meta-heuristic by managing a population of giant tours,
    applying genetic recombination (OX crossover), and intensifying offspring
    quality via multiple parallel local search operators (Education).

    Attributes:
        dist_matrix: pairwise distances [B, N, N] or [N, N].
        wastes: node-wise demand/waste values.
        vehicle_capacity: maximum capacity per route.
        max_iterations: internal loop limit for local search operators.
        time_limit: wall-clock search budget.
        device: hardware target.
        seed: RNG initialization constant.
        generator: torch device-side RNG.
        rng: CPU-side python RNG for high-level branching.
    """

    def __init__(
        self,
        dist_matrix: torch.Tensor,
        wastes: torch.Tensor,
        vehicle_capacity: Union[float, torch.Tensor],
        max_iterations: int = 50,
        time_limit: float = 1.0,
        device: str = "cuda",
        seed: int = 42,
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HGS solver.

        Args:
            dist_matrix: node-to-node cost tensor.
            wastes: node filling rates or demands.
            vehicle_capacity: constraint for the split algorithm.
            max_iterations: local search iteration limit.
            time_limit: total runtime budget.
            device: hardware identifier.
            seed: random seed for both CPU and GPU RNGs.
            generator: existing torch generator (optional).
            rng: existing python Random instance (optional).
            kwargs: Additional keyword arguments.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.vehicle_capacity = vehicle_capacity
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.device = torch.device(device)
        self.seed = seed
        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = random.Random(self.seed)

    def __getstate__(self) -> Dict[str, Any]:
        """Prepares state for serialization, extracting generator state.

        Returns:
            Dict[str, Any]: Object map suitable for pickling.
        """
        state = self.__dict__.copy()
        if "generator" in state and state["generator"] is not None:
            state["generator_state"] = state["generator"].get_state()
            state["generator_device"] = str(state["generator"].device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores search state and re-syncs device RNGs.

        Args:
            state: Serialized attribute dictionary.
        """
        gen_state = state.pop("generator_state", None)
        gen_device = state.pop("generator_device", None)
        self.__dict__.update(state)
        if gen_state is not None:
            self.generator = torch.Generator(device=gen_device)
            self.generator.set_state(gen_state)
        else:
            self.generator = torch.Generator(device="cpu")

    def solve(
        self,
        initial_solutions: torch.Tensor,
        n_generations: int = 50,
        population_size: int = 10,
        elite_size: int = 5,
        time_limit: Optional[float] = None,
        max_vehicles: int = 0,
        crossover_rate: float = 0.7,
    ) -> Tuple[Union[torch.Tensor, List[List[int]]], torch.Tensor]:
        """Run the evolutionary search starting from provided base solutions.

        Implements the main HGS pipeline: Selection -> Crossover -> Education ->
        Survivor Selection -> Diversity Monitoring -> Optional Restart.

        Args:
            initial_solutions: baseline giant tours [B, N].
            n_generations: maximum generation cycles.
            population_size: size of the genetic pool.
            elite_size: number of top individuals kept for the next generation.
            time_limit: runtime override in seconds.
            max_vehicles: fleet size constraint.
            crossover_rate: probability of recombination per child.

        Returns:
            Tuple[Union[torch.Tensor, List[List[int]]], torch.Tensor]:
                - best_routes: optimized node sequences.
                - best_final_costs: corresponding costs [B].
        """
        B, N = initial_solutions.size()
        start_time = time.time()

        _, costs = vectorized_linear_split(
            initial_solutions,
            self.dist_matrix,
            self.wastes,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )

        pop = VectorizedPopulation(population_size, self.device, self.generator)
        pop.initialize(initial_solutions, costs, elite_size)

        no_improv = 0
        best_cost_tracker = pop.costs.min().item()

        for _gen in range(n_generations):
            if time_limit is not None and (time.time() - start_time > time_limit):
                break

            # 1. Selection & Recombination
            p1, p2 = pop.get_parents(n_offspring=1)
            p1 = p1.squeeze(1)
            p2 = p2.squeeze(1)

            offspring_giant = (
                vectorized_ordered_crossover(p1, p2, self.device, self.generator)
                if self.rng.random() < crossover_rate
                else p1.clone()
            )

            # 2. Linear Split (Giant Tour -> CVRP Routes)
            routes_list, split_costs = vectorized_linear_split(
                offspring_giant,
                self.dist_matrix,
                self.wastes,
                self.vehicle_capacity,
                max_vehicles=max_vehicles,
            )

            # 3. Education Phase (Intensified Local Search)
            improved_routes, improved_costs = self.educate(routes_list, split_costs, max_vehicles)

            # 4. Reconstruction (Routes -> Giant Tour)
            giant_candidates = torch.zeros((B, N), dtype=torch.long, device=self.device)
            improved_routes_cpu = improved_routes.cpu().numpy()

            for b in range(B):
                r_full = improved_routes_cpu[b]
                g_tour = r_full[r_full != 0]
                if len(g_tour) == N:
                    giant_candidates[b, :] = torch.tensor(g_tour, device=self.device)
                else:
                    giant_candidates[b, :] = offspring_giant[b, :]

            # 5. Survivor Selection (Diversity-Aware Pruning)
            pop.add_individuals(giant_candidates, improved_costs, elite_size)

            current_best = pop.costs.min().item()
            if current_best < best_cost_tracker - 1e-4:
                best_cost_tracker = current_best
                no_improv = 0
            else:
                no_improv += 1

            restarted = False
            if no_improv > 50:
                # 6. Diversification / Restart
                k = elite_size
                sorted_idx = torch.argsort(pop.biased_fitness, dim=1)[:, :k]
                elite_pop = torch.gather(pop.population, 1, sorted_idx.unsqueeze(2).expand(-1, -1, N))
                elite_cost = torch.gather(pop.costs, 1, sorted_idx)

                n_rest = pop.max_size - k
                new_pop = torch.zeros((B, n_rest, N), dtype=torch.long, device=self.device)
                for b in range(B):
                    for i in range(n_rest):
                        new_pop[b, i] = torch.randperm(N, device=self.device, generator=self.generator) + 1

                b_sz, n_rst, n_nds = new_pop.shape
                flat_pop = new_pop.view(b_sz * n_rst, n_nds)

                flat_dist = self.dist_matrix.repeat_interleave(n_rst, dim=0)
                flat_wastes = self.wastes.repeat_interleave(n_rst, dim=0)
                flat_cap = self.vehicle_capacity
                if isinstance(flat_cap, torch.Tensor) and flat_cap.dim() > 0:
                    flat_cap = flat_cap.repeat_interleave(n_rst, dim=0)

                _, new_costs_flat = vectorized_linear_split(
                    flat_pop,
                    flat_dist,
                    flat_wastes,
                    cast(float, flat_cap),
                    max_vehicles=max_vehicles,
                )
                new_costs = new_costs_flat.view(b_sz, n_rst)

                pop.population = torch.cat([elite_pop, new_pop], dim=1)
                pop.costs = torch.cat([elite_cost, new_costs], dim=1)
                pop.compute_biased_fitness(elite_size)
                no_improv = 0
                restarted = True

            self._viz_record(
                generation=_gen,
                best_cost=float(pop.costs.min().item()),
                mean_cost=float(pop.costs.mean().item()),
                worst_cost=float(pop.costs.max().item()),
                no_improv=no_improv,
                restarted=restarted,
            )

        best_cost_pop, best_idx = torch.min(pop.costs, dim=1)
        best_giant = pop.population[torch.arange(B), best_idx]
        best_routes, best_final_costs = vectorized_linear_split(
            best_giant,
            self.dist_matrix,
            self.wastes,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )
        return best_routes, best_final_costs

    def educate(
        self,
        routes_list: List[List[int]],
        split_costs: torch.Tensor,
        max_vehicles: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Education Phase: Intensified parallel local search.

        Iteratively applies neighborhood operators (Relocate, Swap, Opt) to improve
        solution quality. This constitutes the 'Hybrid' component of the algorithm.

        Args:
            routes_list: base node sequences from the split operator.
            split_costs: initial costs before optimization [B].
            max_vehicles: fleet constraint (passed to split if needed).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - improved_routes: sequences after education [B, max_L].
                - improved_costs: updated costs after neighborhood search [B].
        """
        B = len(routes_list)
        max_l = max(len(r) for r in routes_list)
        offspring_routes = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
        for b in range(B):
            r = routes_list[b]
            offspring_routes[b, : len(r)] = torch.tensor(r, device=self.device)

        if max_l > 2:
            improved_routes = vectorized_two_opt(
                offspring_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )
            improved_routes = vectorized_three_opt(
                improved_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )
            improved_routes = vectorized_swap(
                improved_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )
            improved_routes = vectorized_relocate(
                improved_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )
            improved_routes = vectorized_two_opt_star(
                improved_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )
            improved_routes = vectorized_swap_star(
                improved_routes,
                self.dist_matrix,
                max_iterations=self.max_iterations,
                generator=self.generator,
            )

            from_n = improved_routes[:, :-1]
            to_n = improved_routes[:, 1:]

            if self.dist_matrix.dim() == 3 and self.dist_matrix.size(0) == B:
                batch_ids = torch.arange(B, device=self.device).view(B, 1)
                dists = self.dist_matrix[batch_ids, from_n, to_n]
            else:
                expanded_dm = self.dist_matrix
                if expanded_dm.dim() == 2:
                    expanded_dm = expanded_dm.unsqueeze(0)
                if expanded_dm.size(0) == 1 and B > 1:
                    expanded_dm = expanded_dm.expand(B, -1, -1)
                batch_ids = torch.arange(B, device=self.device).view(B, 1)
                dists = expanded_dm[batch_ids, from_n, to_n]

            improved_costs = dists.sum(dim=1)
        else:
            improved_routes = offspring_routes
            improved_costs = split_costs

        return improved_routes, improved_costs
