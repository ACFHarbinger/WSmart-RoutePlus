"""HGS Population Management.

Provides specialized data structures and management logic for maintaining
a diverse pool of high-quality solutions throughout the evolutionary search.

Attributes:
    VectorizedPopulation: Class managing a diverse pool of elite solutions.

Example:
    >>> from logic.src.policies.vectorized.routing.core import VectorizedPopulation
    >>> pop = VectorizedPopulation(size=50, device="cuda")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .evaluation import calc_broken_pairs_distance


class VectorizedPopulation:
    """Manages a diverse population of solutions for the HGS algorithm.

    Coordinates the storage, evaluation, and selection of solutions across a
    batch of independent search instances. Implements biased fitness functions
    that balance objective costs with solution diversity.

    Attributes:
        max_size: Maximum number of individuals allowed in the population.
        device: Hardware device where tensors are stored.
        generator: Torch RNG for stochastic operations (e.g., parent selection).
        population: Tensor of candidate solutions (giant tours) of shape [B, P, N].
        costs: Objective cost for each individual of shape [B, P].
        biased_fitness: Calculated biased fitness ranking of shape [B, P].
        diversity_scores: Diversity metrics for each solution of shape [B, P].
    """

    def __init__(self, size: int, device: Any, generator: Optional[torch.Generator] = None) -> None:
        """Initializes the population management state.

        Args:
            size: Maximum allowed population size (P).
            device: Computing device for tensor operations.
            generator: Optional Torch random number generator.
        """
        self.max_size = size
        self.device = device
        self.generator = generator
        self.population: torch.Tensor = torch.empty(0)  # (B, P, N)
        self.costs: torch.Tensor = torch.empty(0)  # (B, P)
        self.biased_fitness: torch.Tensor = torch.empty(0)  # (B, P)
        self.diversity_scores: torch.Tensor = torch.empty(0)  # (B, P)

    def __getstate__(self) -> Dict[str, Any]:
        """Prepares the population state for pickling.

        Returns:
            Dict[str, Any]: State dictionary with generator state metadata.
        """
        state = self.__dict__.copy()
        if "generator" in state and state["generator"] is not None:
            state["generator_state"] = state["generator"].get_state()
            state["generator_device"] = str(state["generator"].device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores the population state from a pickle bundle.

        Args:
            state: Dictionary containing the persisted state.
        """
        gen_state = state.pop("generator_state", None)
        gen_device = state.pop("generator_device", None)
        self.__dict__.update(state)
        if gen_state is not None:
            self.generator = torch.Generator(device=gen_device)
            self.generator.set_state(gen_state)
        else:
            self.generator = torch.Generator(device="cpu")

    def initialize(self, initial_pop: torch.Tensor, initial_costs: torch.Tensor, nb_elite: int) -> None:
        """Initializes the population with a set of starting solutions.

        Args:
            initial_pop: Starting solutions of shape [B, N] or [B, P0, N].
            initial_costs: Costs of initial solutions of shape [B] or [B, P0].
            nb_elite: Number of elite individuals to protect in biased fitness.
        """
        if initial_pop.dim() == 2:
            initial_pop = initial_pop.unsqueeze(1)  # (B, 1, N)

        # Consistent costs shape (B, P0)
        if initial_costs.dim() == 1:
            initial_costs = initial_costs.unsqueeze(1)  # (B, 1)

        self.population = initial_pop
        self.costs = initial_costs
        self.compute_biased_fitness(nb_elite)

    def add_individuals(self, candidates: torch.Tensor, costs: torch.Tensor, nb_elite: int) -> None:
        """Merges new individuals into the pool and performs survivor selection.

        Maintains the population size by ranking all individuals (current pool
        plus candidates) via biased fitness and keeping only the top `max_size`.

        Args:
            candidates: New solutions to evaluate of shape [B, C, N].
            costs: Costs of new candidates of shape [B, C].
            nb_elite: Number of elite individuals to protect in rankings.
        """
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
            costs = costs.unsqueeze(1)

        # 1. Concatenate
        combined_pop = torch.cat([self.population, candidates], dim=1)
        combined_costs = torch.cat([self.costs, costs], dim=1)

        # 2. Update state temporarily
        self.population = combined_pop
        self.costs = combined_costs

        # 3. Compute Fitness & Survivor Selection
        self.compute_biased_fitness(nb_elite)

        if self.population.size(1) > self.max_size:
            # Sort by fitness (ascending, smaller is better)
            _, indices = torch.sort(self.biased_fitness, dim=1)
            survivors = indices[:, : self.max_size]

            # Gather survivors
            B, _, N = self.population.size()
            surv_expanded = survivors.unsqueeze(2).expand(-1, -1, N)
            self.population = torch.gather(self.population, 1, surv_expanded)
            self.costs = torch.gather(self.costs, 1, survivors)
            self.biased_fitness = torch.gather(self.biased_fitness, 1, survivors)

            if self.diversity_scores is not None and self.diversity_scores.numel() > 0:
                self.diversity_scores = torch.gather(self.diversity_scores, 1, survivors)

    def compute_biased_fitness(self, nb_elite: int) -> None:
        """Calculates biased fitness ranking for all individuals.

        BF(I) = Rank_C(I) + (1 - N_elite/|Pop|) * Rank_D(I)
        where Rank_C is cost rank and Rank_D is diversity rank. Lower is better.

        Args:
            nb_elite: Number of elite individuals defined in the search config.
        """
        B, P, N = self.population.size()

        # 1. Rank by Cost (Ascending: lower cost is better, Rank 0)
        cost_indices = torch.argsort(self.costs, dim=1)
        cost_ranks = torch.argsort(cost_indices, dim=1).float()

        # 2. Diversity
        # Calculate diversity (higher is better)
        self.diversity_scores = calc_broken_pairs_distance(self.population)

        # Rank by Diversity (Descending: higher diversity is better, Rank 0)
        div_indices = torch.argsort(self.diversity_scores, dim=1, descending=True)
        div_ranks = torch.argsort(div_indices, dim=1).float()

        # 3. Combine
        diversity_weight = 1.0 - (nb_elite / P)
        self.biased_fitness = cost_ranks + diversity_weight * div_ranks

    def get_parents(self, n_offspring: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects parent pairs for recombination via binary tournament.

        Args:
            n_offspring: Number of parent pairs to select for the next generation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A pair of parent tensors each
                of shape [B, n_offspring, N].
        """
        B, P, N = self.population.size()

        def tournament() -> torch.Tensor:
            """Selects indices using binary tournament based on biased fitness."""
            idx_a = torch.randint(0, P, (B, n_offspring), device=self.device, generator=self.generator)
            idx_b = torch.randint(0, P, (B, n_offspring), device=self.device, generator=self.generator)
            fit_a = torch.gather(self.biased_fitness, 1, idx_a)
            fit_b = torch.gather(self.biased_fitness, 1, idx_b)
            return torch.where(fit_a < fit_b, idx_a, idx_b)

        parent1_idx = tournament()
        parent2_idx = tournament()

        p1 = torch.gather(self.population, 1, parent1_idx.unsqueeze(2).expand(-1, -1, N))
        p2 = torch.gather(self.population, 1, parent2_idx.unsqueeze(2).expand(-1, -1, N))

        return p1, p2
