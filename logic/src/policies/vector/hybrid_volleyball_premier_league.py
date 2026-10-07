"""Vectorized Hybrid Volleyball Premier League (HVPL) Policy.

This module implements the HVPL algorithm, a population-based meta-heuristic
that mimics the competition and coaching dynamics of a professional sports
league. It fuses constructionist capabilities (ACO) with improvement
heuristics (ALNS) using a parallelized population management system on the GPU.

Attributes:
    VectorizedHVPL: High-performance population-based hybrid policy.

Example:
    >>> from logic.src.policies.vector.hybrid_volleyball_premier_league import VectorizedHVPL
    >>> policy = VectorizedHVPL(env_name="ptp")
    >>> out = policy(td)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.policies.vector.adaptive_large_neighborhood_search import (
    VectorizedALNS,
)
from logic.src.policies.vector.hgs import VectorizedHGS
from logic.src.policies.vector.shared.linear import vectorized_linear_split


class VectorizedHVPL(AutoregressivePolicy):
    """Vectorized HVPL Policy for high-performance routing.

    Maintains a league of 'teams' (giant tours) which undergo a coaching phase
    (ALNS refinement), global competition (best-tour tracking), and a substitution
    phase (replacing low-performers via ACO construction).

    Attributes:
        n_teams: Population size (teams) per problem instance.
        max_iterations: Number of seasons/cycles to simulate.
        sub_rate: Portion of teams to replace per cycle.
        time_limit: Wall-clock runtime constraint in seconds.
        aco_iterations: Iterations for the construction phase.
        alns_iterations: Internal iterations for the coaching phase.
    """

    def __init__(
        self,
        env_name: str,
        n_teams: int = 10,
        max_iterations: int = 20,
        sub_rate: float = 0.2,
        time_limit: float = 60.0,
        aco_iterations: int = 1,
        alns_iterations: int = 100,
        device: str = "cuda",
        generator: Optional[torch.Generator] = None,
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HVPL policy.

        Args:
            env_name: Name of the environment identifier.
            n_teams: Number of parallel teams (individuals) in the league.
            max_iterations: Number of seasons (generations) to simulate.
            sub_rate: Probability/ratio of replacing underperforming teams.
            time_limit: Maximum execution time in seconds.
            aco_iterations: Number of construction iterations for new teams.
            alns_iterations: Number of refinement iterations in the coaching phase.
            device: Computing device for tensor operations.
            generator: Torch RNG generator for sampling.
            rng: Python random instance for heuristics.
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name=env_name, device=device, generator=generator, rng=rng, **kwargs)
        self.n_teams = n_teams
        self.max_iterations = max_iterations
        self.sub_rate = sub_rate
        self.time_limit = time_limit
        self.aco_iterations = aco_iterations
        self.alns_iterations = alns_iterations

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for serialization, stripping hardware-bound RNGs.

        Returns:
            Dict[str, Any]: Attribute map suitable for pickling.
        """
        state = self.__dict__.copy()
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore search parameters and re-initialize device RNGs.

        Args:
            state: Serialized attribute dictionary.
        """
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the HVPL simulation loop.

        Args:
            td: TensorDict containing instance data (locations, demands).
            env: Optional environment for reward calculation.
            strategy: Solver strategy (e.g., "greedy").
            num_starts: Unused in HVPL.
            max_steps: Maximum step constraint.
            phase: Current phase ("train", "val", "test").
            return_actions: Whether to return full action sequences.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Optimized results dictionary including:
                - actions (torch.Tensor): Padded action sequences.
                - reward (torch.Tensor): Calculated reward for the solution.
                - cost (torch.Tensor): Raw objective value from the engine.
                - log_likelihood (torch.Tensor): Zero vector (HVPL is non-pi).
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device
        start_time = time.time()

        # 1. Setup Data
        dist_matrix, waste, capacity = self._setup_data(td)
        num_nodes = dist_matrix.shape[1]

        # 2. League Initialization (ACO Construction)
        expanded_dist = dist_matrix.repeat_interleave(self.n_teams, dim=0)
        expanded_waste = waste.repeat_interleave(self.n_teams, dim=0)
        expanded_capacity = capacity.repeat_interleave(self.n_teams, dim=0)

        tau = torch.ones((batch_size, num_nodes, num_nodes), device=device) * 0.1
        eta = 1.0 / (dist_matrix + 1e-6)

        population_tours = self._aco_construct(dist_matrix, tau, eta, self.n_teams)

        best_tours = population_tours[:, 0].clone()
        best_costs = torch.full((batch_size,), float("inf"), device=device)

        # 3. League Season
        for _iter in range(self.max_iterations):
            if time.time() - start_time > self.time_limit:
                break

            # 4. Coaching Phase (Vectorized ALNS refinement)
            instance_costs, coached_routes_list = self._coaching_phase(
                population_tours, expanded_dist, expanded_waste, expanded_capacity
            )

            # 5. Global Competition (Select iteration best per batch)
            improved = self._global_competition(
                instance_costs,
                mentored_routes_list=coached_routes_list,
                best_tours=best_tours,
                best_costs=best_costs,
                num_nodes=num_nodes,
            )

            # 6. Pheromone Update (Stochastic reinforcement of best edges)
            if improved.any():
                tau = self._update_pheromones(tau, best_tours, best_costs)

            # 7. Substitution Phase (Replace weakest solutions)
            if self.sub_rate > 0:
                self._substitution_phase(population_tours, instance_costs, dist_matrix, tau, eta)

        # 8. Final packaging
        return self._format_output(td, best_tours, dist_matrix, waste, capacity, env=env)

    def _setup_data(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and compute necessary tensors from the TensorDict state.

        Args:
            td: Input state bundle.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - dist_matrix: Euclidean cost tensor of shape [B, N, N].
                - waste: Falling levels of shape [B, N].
                - capacity: Scalar or per-route limits of shape [B].
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device
        locs = td["locs"]
        num_nodes = locs.shape[1]

        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        waste_at_nodes = td.get("waste", torch.zeros(batch_size, num_nodes - 1, device=device))
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:
            capacity = capacity.expand(batch_size)

        return dist_matrix, waste, capacity

    def _coaching_phase(
        self,
        population_tours: torch.Tensor,
        dist: torch.Tensor,
        waste: torch.Tensor,
        capacity: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """Apply ALNS improvement search to the entire population.

        Args:
            population_tours: Candidate giant tours of shape [B, K, N].
            dist: Expanded distances of shape [B*K, N, N].
            waste: Expanded demands of shape [B*K, N].
            capacity: Expanded limits of shape [B*K].

        Returns:
            Tuple[torch.Tensor, List[List[int]]]: A tuple containing:
                - instance_costs: Refinement costs per team of shape [B, K].
                - coached_routes_list: Nested route collections from ALNS.
        """
        batch_size = population_tours.shape[0]
        device = population_tours.device

        alns_engine = VectorizedALNS(dist_matrix=dist, wastes=waste, vehicle_capacity=capacity, device=str(device))

        flat_tours = population_tours.view(batch_size * self.n_teams, -1)
        coached_routes_list, coached_costs = alns_engine.solve(
            initial_solutions=flat_tours, n_iterations=self.alns_iterations
        )

        instance_costs = coached_costs.view(batch_size, self.n_teams)
        return instance_costs, coached_routes_list

    def _global_competition(
        self,
        instance_costs: torch.Tensor,
        mentored_routes_list: Union[List[List[int]], List[torch.Tensor]],
        best_tours: torch.Tensor,
        best_costs: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Find batch improvements and update the elite giant tours.

        Args:
            instance_costs: Costs for all teams of shape [B, K].
            mentored_routes_list: Refined routes from ALNS.
            best_tours: Persistent elite tours of shape [B, N-1].
            best_costs: Persistent elite costs of shape [B].
            num_nodes: Total node count.

        Returns:
            torch.Tensor: Boolean improvement flag per batch instance of shape [B].
        """
        batch_size = instance_costs.shape[0]
        device = best_tours.device

        min_costs, min_idx = instance_costs.min(dim=1)
        improved = min_costs < best_costs
        best_costs[improved] = min_costs[improved]

        for b in range(batch_size):
            best_idx = int(min_idx[b].item())
            flat_idx = b * self.n_teams + best_idx
            routes = mentored_routes_list[flat_idx]

            if isinstance(routes, torch.Tensor):
                nodes = routes[routes != 0]
            else:
                nodes = torch.tensor([n for n in routes if n != 0], device=device)

            if nodes.size(0) < num_nodes - 1:
                nodes = torch.cat(
                    [
                        nodes,
                        torch.zeros(
                            num_nodes - 1 - nodes.size(0),
                            dtype=torch.long,
                            device=device,
                        ),
                    ]
                )
            elif nodes.size(0) > num_nodes - 1:
                nodes = nodes[: num_nodes - 1]
            best_tours[b] = nodes
        return improved

    def _substitution_phase(
        self,
        population_tours: torch.Tensor,
        instance_costs: torch.Tensor,
        dist_matrix: torch.Tensor,
        tau: torch.Tensor,
        eta: torch.Tensor,
    ) -> None:
        """Identify worst performers and replace them via re-construction.

        Args:
            population_tours: Current giant tour pool of shape [B, K, N-1].
            instance_costs: Recorded team performance of shape [B, K].
            dist_matrix: Global problem distances of shape [B, N, N].
            tau: Pheromone trails for guided construction of shape [B, N, N].
            eta: Heuristic visibility of shape [B, N, N].
        """
        batch_size = population_tours.shape[0]
        n_sub = int(self.n_teams * self.sub_rate)
        if n_sub > 0:
            worst_indices = instance_costs.argsort(dim=1, descending=True)[:, :n_sub]
            sub_tours = self._aco_construct(dist_matrix, tau, eta, n_sub)
            for b in range(batch_size):
                for s in range(n_sub):
                    idx = int(worst_indices[b, s].item())
                    population_tours[b, idx] = sub_tours[b, s]

    def _format_output(
        self,
        td: TensorDict,
        best_tours: torch.Tensor,
        dist_matrix: torch.Tensor,
        waste: torch.Tensor,
        capacity: torch.Tensor,
        env: Optional[RL4COEnvBase] = None,
    ) -> Dict[str, Any]:
        """Final conversion from giant tours to split routes and output dicts.

        Args:
            td: Input state for metadata access.
            best_tours: Final elite tours of shape [B, N-1].
            dist_matrix: Problem scaling of shape [B, N, N].
            waste: Demands of shape [B, N].
            capacity: Constraints of shape [B].
            env: Environment instance for reward sizing.

        Returns:
            Dict[str, Any]: Final results bundle dictionary with:
                - actions (torch.Tensor): Padded action sequences.
                - reward (torch.Tensor): Calculated reward/cost for the solution.
                - cost (torch.Tensor): Raw objective value from the engine.
                - log_likelihood (torch.Tensor): Zero vector (HVPL is non-pi).
        """
        batch_size = best_tours.shape[0]
        device = best_tours.device

        final_routes, final_costs = vectorized_linear_split(best_tours, dist_matrix, waste, capacity)

        max_len = max([len(r) for r in final_routes] + [2])
        all_actions = []
        for r in final_routes:
            a = torch.tensor(r, device=device, dtype=torch.long)
            if len(a) < max_len:
                a = torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)])
            all_actions.append(a)

        padded_actions = torch.stack(all_actions)

        reward = VectorizedHGS._compute_reward(td, env, padded_actions)

        return {
            "actions": padded_actions,
            "reward": reward,
            "cost": final_costs.to(device),
            "log_likelihood": torch.zeros(batch_size, device=device),
        }

    def _aco_construct(
        self,
        dist_matrix: torch.Tensor,
        tau: torch.Tensor,
        eta: torch.Tensor,
        n_ants_per_instance: int,
    ) -> torch.Tensor:
        """Parallelized construction utilizing iteration pheromone state.

        Args:
            dist_matrix: Distances of shape [B, N, N].
            tau: Shared pheromones of shape [B, N, N].
            eta: Heuristic data of shape [B, N, N].
            n_ants_per_instance: Concurrent walkers per instance.

        Returns:
            torch.Tensor: Constructed giant tours of shape [B, K, N-1].
        """
        batch_size, num_nodes, _ = dist_matrix.shape
        device = dist_matrix.device

        tau_k = tau.unsqueeze(1).expand(-1, n_ants_per_instance, -1, -1)
        eta_k = eta.unsqueeze(1).expand(-1, n_ants_per_instance, -1, -1)

        current_node = torch.zeros((batch_size, n_ants_per_instance), dtype=torch.long, device=device)
        visited = torch.zeros(
            (batch_size, n_ants_per_instance, num_nodes),
            dtype=torch.bool,
            device=device,
        )
        visited.scatter_(2, current_node.unsqueeze(2), 1)

        tours = []
        for _ in range(num_nodes - 1):
            gather_idx = current_node.view(batch_size, n_ants_per_instance, 1, 1).expand(-1, -1, 1, num_nodes)
            tau_step = tau_k.gather(2, gather_idx).squeeze(2)
            eta_step = eta_k.gather(2, gather_idx).squeeze(2)

            scores = (tau_step**1.0) * (eta_step**2.0)
            scores.masked_fill_(visited, 0)

            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-10)
            next_node = torch.multinomial(probs.view(-1, num_nodes), 1, generator=self.generator).view(
                batch_size, n_ants_per_instance
            )

            current_node = next_node
            visited.scatter_(2, current_node.unsqueeze(2), 1)
            tours.append(current_node)

        return torch.stack(tours, dim=2)

    def _update_pheromones(self, tau: torch.Tensor, best_tours: torch.Tensor, best_costs: torch.Tensor) -> torch.Tensor:
        """Update pheromone intensities based on global batch performance.

        Args:
            tau: Current trails [B, N, N].
            best_tours: Elite sequences [B, N-1].
            best_costs: Elite costs [B].

        Returns:
            torch.Tensor: Reinforced pheromone matrix [B, N, N].
        """
        batch_size, N_minus_1 = best_tours.shape
        num_nodes = N_minus_1 + 1
        device = tau.device

        tau = 0.9 * tau
        delta = 1.0 / (best_costs + 1e-10)

        padded = torch.zeros((batch_size, num_nodes + 1), dtype=torch.long, device=device)
        padded[:, 1:-1] = best_tours

        from_nodes = padded[:, :-1]
        to_nodes = padded[:, 1:]

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_nodes)

        tau.index_put_(
            (batch_idx.flatten(), from_nodes.flatten(), to_nodes.flatten()),
            delta.repeat_interleave(num_nodes).to(tau.dtype),
            accumulate=True,
        )
        tau.index_put_(
            (batch_idx.flatten(), to_nodes.flatten(), from_nodes.flatten()),
            delta.repeat_interleave(num_nodes).to(tau.dtype),
            accumulate=True,
        )

        return torch.clamp(tau, 0.01, 10.0)
