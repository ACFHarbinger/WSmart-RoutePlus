"""Vectorized Ant Colony Optimization (ACO) Policy.

This module implements a high-performance, GPU-accelerated Ant Colony
System (ACS). It processes multiple problem instances and multiple ants per
instance in parallel using tensor operations, drastically reducing the
computational overhead of pheromone updates and local updates.

Attributes:
    VectorizedACOPolicy: High-performance parallel ACO policy.

Example:
    >>> from logic.src.policies.vector.ant_colony_system import VectorizedACOPolicy
    >>> policy = VectorizedACOPolicy(env_name="tsp")
    >>> out = policy(td, env=env)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict

from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.pipeline.rl.envs.base.base import RL4COEnvBase
from logic.src.tracking.viz_mixin import PolicyVizMixin


class VectorizedACOPolicy(AutoregressivePolicy, PolicyVizMixin):
    """Vectorized ACS Policy for Combinatorial Optimization.

    Maintains a global pheromone matrix (tau) per batch and utilizes heuristic
    information (eta) to guide ants through a stochastic decision process.
    Supports ACS-specific elitism and weight-based pheromone updates.

    Attributes:
        n_ants: Number of parallel agents per batch instance.
        n_iterations: Number of pheromone update cycles.
        alpha: Pheromone influence coefficient.
        beta: Heuristic/distance influence coefficient.
        decay: Pheromone evaporation rate.
        elitism: Number of top ants contributing to global updates.
        q0: ACS exploitation probability (vs exploration).
        min_pheromone: Floor value for pheromone trails to prevent stagnation.
    """

    def __init__(
        self,
        env_name: str,
        n_ants: int = 20,
        n_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        decay: float = 0.1,
        elitism: int = 1,
        q0: float = 0.9,
        min_pheromone: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the ACO policy.

        Args:
            env_name: Name of the environment (e.g., "tsp").
            n_ants: Number of parallel ants to use per batch instance.
            n_iterations: Number of ACO iterations/generations.
            alpha: Pheromone importance exponent.
            beta: Heuristic importance exponent.
            decay: Pheromone evaporation rate (rho).
            elitism: Number of top ants to use for global pheromone update.
            q0: Probability of choosing the best move (exploitation) in ACS.
            min_pheromone: Minimum pheromone level (tau_min).
            seed: Random seed for reproducibility.
            device: Computing device ("cpu", "cuda").
            kwargs: Additional arguments for AutoregressivePolicy.
        """
        super().__init__(env_name=env_name, seed=seed, device=device, **kwargs)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.seed = seed
        self.beta = beta
        self.decay = decay
        self.elitism = elitism
        self.q0 = q0
        self.min_pheromone = min_pheromone

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for serialization, extracting generator state.

        Returns:
            Dict[str, Any]: Attribute map safe for pickling.
        """
        state = self.__dict__.copy()
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and initialize RNG from serialized data.

        Args:
            state: Serialized dictionary of attributes.
        """
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def _get_heuristic(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Compute inverse distance heuristic.

        Args:
            dist_matrix: Pairwise node distances of shape [B, N, N].

        Returns:
            torch.Tensor: Heuristic visibility information (eta) of shape [B, N, N].
        """
        dist_safe = dist_matrix + 1e-6
        return 1.0 / dist_safe

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the parallel ACO search.

        Constructs solutions for all ants, updates pheromones based on iteration
        quality, and tracks the global best solution across the batch.

        Args:
            td: TensorDict containing the problem instance data (e.g., node locations).
            env: Optional environment for step transitions and masking.
            strategy: Decoding strategy (e.g., "sampling", "greedy").
            num_starts: Number of starting nodes/tours to evaluate.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - actions: The best tour found for each instance.
                - reward: Negative tour cost.
                - cost: Absolute tour length.
                - log_likelihood: Dummy probabilities (ACO is not gradient-based).
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device

        locs = td["locs"]
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        num_nodes = dist_matrix.shape[1]

        eta = self._get_heuristic(dist_matrix)
        tau = torch.ones_like(dist_matrix) * self.min_pheromone

        best_tours = torch.zeros((batch_size, num_nodes), dtype=torch.long, device=device)
        best_costs = torch.full((batch_size,), float("inf"), device=device)

        for _aco_iter in range(self.n_iterations):
            iter_ant_tours, _ = self._construct_solutions(dist_matrix, tau, eta, env)
            costs = self._evaluate_batch(iter_ant_tours, dist_matrix)

            min_costs, min_idx = costs.min(dim=1)
            improved = min_costs < best_costs
            best_costs[improved] = min_costs[improved]

            best_ants_indices = min_idx.view(batch_size, 1, 1).expand(-1, 1, num_nodes)
            best_tours_iter = iter_ant_tours.gather(1, best_ants_indices).squeeze(1)
            best_tours[improved] = best_tours_iter[improved]

            tau = self._update_pheromones(tau, iter_ant_tours, costs)

            self._viz_record(
                iteration=_aco_iter,
                global_best_cost=float(best_costs.min().item()),
                iter_best_cost=float(min_costs.min().item()),
                tau_mean=float(tau.mean().item()),
                tau_max=float(tau.max().item()),
            )

        return {
            "actions": best_tours,
            "reward": -best_costs,
            "cost": best_costs,
            "log_likelihood": torch.zeros(batch_size, device=device),
        }

    def _construct_solutions(
        self,
        dist_matrix: torch.Tensor,
        tau: torch.Tensor,
        eta: torch.Tensor,
        env: Optional[RL4COEnvBase],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel path construction for all ants.

        Utilizes multinomial sampling guided by (tau^alpha * eta^beta) with masking
        for previously visited nodes.

        Args:
            dist_matrix: Distance tensor of shape [B, N, N].
            tau: Pheromone matrix of shape [B, N, N].
            eta: Heuristic matrix of shape [B, N, N].
            env: Environment for masking/validations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - final_ant_tours: Node sequences of shape [B, K, N].
                - log_probs_sum: Tour probabilities of shape [B, K].
        """
        batch_size, num_nodes, _ = dist_matrix.shape
        n_ants = self.n_ants
        device = dist_matrix.device

        # [B, K, N, N]
        tau_k = tau.unsqueeze(1).expand(-1, n_ants, -1, -1)
        eta_k = eta.unsqueeze(1).expand(-1, n_ants, -1, -1)

        current_node = torch.zeros((batch_size, n_ants), dtype=torch.long, device=device)
        visited = torch.zeros((batch_size, n_ants, num_nodes), dtype=torch.bool, device=device)
        visited.scatter_(2, current_node.unsqueeze(2), 1)

        tours_accumulator: List[torch.Tensor] = [current_node]
        log_probs_sum = torch.zeros((batch_size, n_ants), device=device)

        for _ in range(num_nodes - 1):
            gather_idx = current_node.view(batch_size, n_ants, 1, 1).expand(-1, -1, 1, num_nodes)
            tau_step = tau_k.gather(2, gather_idx).squeeze(2)
            eta_step = eta_k.gather(2, gather_idx).squeeze(2)

            scores = (tau_step**self.alpha) * (eta_step**self.beta)
            scores.masked_fill_(visited, 0)

            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-10)

            rand_val = torch.rand((batch_size, n_ants), generator=self.generator, device=device)
            greedy_mask = rand_val < self.q0
            greedy_action = probs.argmax(dim=-1)
            sample_probs = probs + 1e-10
            sample_action = torch.multinomial(sample_probs.view(-1, num_nodes), 1, generator=self.generator).view(
                batch_size, n_ants
            )

            next_node = torch.where(greedy_mask, greedy_action, sample_action)

            current_node = next_node
            visited.scatter_(2, current_node.unsqueeze(2), 1)
            tours_accumulator.append(current_node)

            selected_probs = probs.gather(2, next_node.unsqueeze(2)).squeeze(2)
            log_probs_sum += torch.log(selected_probs + 1e-10)

        final_ant_tours = torch.stack(tours_accumulator, dim=2)
        return final_ant_tours, log_probs_sum

    def _evaluate_batch(self, ant_tours: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """Compute total lengths for all constructed ant tours.

        Args:
            ant_tours: Node sequences of shape [B, K, N].
            dist_matrix: Distances of shape [B, N, N].

        Returns:
            torch.Tensor: Absolute tour lengths of shape [B, K].
        """
        batch_size, n_ants, num_nodes = ant_tours.shape

        from_node = ant_tours
        to_node = torch.roll(ant_tours, -1, dims=2)

        batch_idx = (
            torch.arange(batch_size, device=ant_tours.device)
            .view(batch_size, 1, 1)
            .expand(batch_size, n_ants, num_nodes)
        )
        dists = dist_matrix[batch_idx, from_node, to_node]

        return dists.sum(dim=2)

    def _update_pheromones(self, tau: torch.Tensor, ant_tours: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        """Perform pheromone evaporation and global update from the best ants.

        Args:
            tau: Current pheromone matrix of shape [B, N, N].
            ant_tours: Iteration tours of shape [B, K, N].
            costs: Iteration costs of shape [B, K].

        Returns:
            torch.Tensor: Evaporated and updated pheromone matrix.
        """
        tau = (1 - self.decay) * tau
        batch_size, n_ants, num_nodes = ant_tours.shape

        min_costs, min_idx = costs.min(dim=1)
        idx_expanded = min_idx.view(batch_size, 1, 1).expand(-1, 1, num_nodes)
        best_tours = ant_tours.gather(1, idx_expanded).squeeze(1)

        delta = 1.0 / (min_costs.view(batch_size, 1, 1) + 1e-10)

        from_node = best_tours
        to_node = torch.roll(best_tours, -1, dims=1)
        batch_idx = torch.arange(batch_size, device=tau.device).view(batch_size, 1)

        b_idx_flat = batch_idx.expand(-1, num_nodes).reshape(-1)
        from_flat = from_node.reshape(-1)
        to_flat = to_node.reshape(-1)
        delta_flat = delta.expand(-1, -1, num_nodes).reshape(-1)

        tau.index_put_((b_idx_flat, from_flat, to_flat), delta_flat, accumulate=True)
        tau.index_put_((b_idx_flat, to_flat, from_flat), delta_flat, accumulate=True)

        return torch.clamp(tau, min=self.min_pheromone, max=10.0)
