"""Vectorized HGS-ALNS Hybrid Policy for RL.

This module implements a deep fusion of Hybrid Genetic Search (HGS) and
Adaptive Large Neighborhood Search (ALNS). In this hybrid architecture,
the HGS evolutionary loop provides global search coverage, while the
ALNS engine serves as a high-intensity 'Education' phase for offspring
refinement.

Attributes:
    VectorizedHGSALNSEngine: Internal solver engine combining HGS and ALNS.
    VectorizedHGSALNS: RL4CO policy wrapper for the hybrid solver.

Example:
    >>> from logic.src.policies.vector.hgs_alns import VectorizedHGSALNS
    >>> policy = VectorizedHGSALNS(env_name="vrpp")
    >>> out = policy(td)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.policies.vector.adaptive_large_neighborhood_search import (
    VectorizedALNS,
)
from logic.src.policies.vector.hgs import (
    VectorizedHGS as VectorizedHGSPolicy,
)
from logic.src.policies.vector.hybrid_genetic_search import (
    VectorizedHGS as VectorizedHGSEngine,
)


class VectorizedHGSALNSEngine(VectorizedHGSEngine):
    """HGS Engine with ALNS-based Education.

    Overrides the default HGS local search 'Education' phase with a full ALNS
    pass, providing deeper intensification of offspring solution quality.

    Attributes:
        alns_education_iterations: Iterations for the ALNS sub-pass.
        alns_start_temp: Initial temperature for ALNS simulated annealing.
        alns_cooling_rate: Cooling schedule factor.
        alns_engine: Internal refinement module.
    """

    def __init__(
        self,
        dist_matrix: torch.Tensor,
        wastes: torch.Tensor,
        vehicle_capacity: Union[float, torch.Tensor],
        time_limit: float = 1.0,
        device: str = "cpu",
        rng: Optional[random.Random] = None,
        generator: Optional[torch.Generator] = None,
        alns_education_iterations: int = 50,
        alns_start_temp: float = 0.5,
        alns_cooling_rate: float = 0.9995,
    ) -> None:
        """Initialize the HGS-ALNS hybrid engine.

        Args:
            dist_matrix: Problem distances of shape [B, N, N].
            wastes: Node demands of shape [B, N].
            vehicle_capacity: Scalar or per-route capacity.
            time_limit: Global timeout per generation.
            device: Hardware identifier.
            rng: Python Random instance.
            generator: Torch device-side RNG.
            alns_education_iterations: Refinement intensity.
            alns_start_temp: ALNS heat metadata.
            alns_cooling_rate: ALNS annealing metadata.
        """
        super().__init__(
            dist_matrix=dist_matrix,
            wastes=wastes,
            vehicle_capacity=vehicle_capacity,
            time_limit=time_limit,
            device=device,
            generator=generator,
            rng=rng,
        )
        self.alns_education_iterations = alns_education_iterations
        self.alns_start_temp = alns_start_temp
        self.alns_cooling_rate = alns_cooling_rate
        self.alns_engine = VectorizedALNS(
            dist_matrix=dist_matrix,
            wastes=wastes,
            vehicle_capacity=vehicle_capacity,
            device=device,
            time_limit=self.time_limit,
        )

    def educate(
        self,
        routes_list: List[List[int]],
        split_costs: torch.Tensor,
        max_vehicles: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Education Phase utilizing a full Vectorized ALNS pass.

        Converts linear-split routes into giant tours, applies complex ALNS
        destruction/repair moves, and reconstructs the improved solution.

        Args:
            routes_list: Current node sequences from split.
            split_costs: Baseline costs of shape [B].
            max_vehicles: Fleet constraint.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - improved_routes_tensor: Refined node sequences of shape [B, max_L].
                - improved_costs: Updated fitness values after ALNS refinement of shape [B].
        """
        B = len(routes_list)
        N = self.dist_matrix.shape[1] - 1

        initial_solutions = torch.zeros((B, N), dtype=torch.long, device=self.device)
        for b in range(B):
            r = routes_list[b]
            nodes = torch.tensor([n for n in r if n != 0], device=self.device, dtype=torch.long)
            if nodes.size(0) > N:
                nodes = nodes[:N]
            elif nodes.size(0) < N:
                nodes = torch.cat(
                    [
                        nodes,
                        torch.zeros(N - nodes.size(0), device=self.device, dtype=torch.long),
                    ]
                )
            initial_solutions[b] = nodes

        improved_routes_list, improved_costs = self.alns_engine.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.alns_education_iterations,
            max_vehicles=max_vehicles,
            start_temp=self.alns_start_temp,
            cooling_rate=self.alns_cooling_rate,
        )

        max_l = max(len(r) for r in improved_routes_list)
        improved_routes_tensor = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
        for b in range(B):
            r = improved_routes_list[b]
            improved_routes_tensor[b, : len(r)] = torch.tensor(r, device=self.device)

        return improved_routes_tensor, improved_costs


class VectorizedHGSALNS(VectorizedHGSPolicy):
    """Vectorized HGS-ALNS Policy wrapper for the RL4CO ecosystem.

    Integrates the high-intensity HGS-ALNS engine into the standard neural
    policy forward pass, enabling expert-level optimization for VRP instances.

    Attributes:
        alns_education_iterations: Internal refinement epochs.
        alns_start_temp: Initial bias for ALNS search.
        alns_cooling_rate: Intensity of search narrowing.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        population_size: int = 20,
        n_generations: int = 10,
        elite_size: int = 5,
        crossover_rate: float = 0.5,
        max_vehicles: int = 0,
        alns_education_iterations: int = 50,
        alns_start_temp: float = 0.5,
        alns_cooling_rate: float = 0.9995,
        seed: int = 42,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the RL policy wrapper.

        Args:
            env_name: Name of the environment identifier.
            time_limit: Total wall clock time allowed for search.
            population_size: Number of individuals in the HGS population.
            n_generations: Number of evolutionary generations.
            elite_size: Number of top individuals kept between generations.
            crossover_rate: Probability of performing crossover vs cloning.
            max_vehicles: Constraint on fleet size; 0 implies unconstrained.
            alns_education_iterations: Refinement intensity during education phase.
            alns_start_temp: Starting temperature for ALNS local search.
            alns_cooling_rate: Speed of temperature reduction in ALNS.
            seed: RNG seed for reproducible experiments.
            device: Computing device for tensor operations.
            kwargs: Additional arguments for VectorizedHGSPolicy.
        """
        super().__init__(
            env_name=env_name,
            time_limit=time_limit,
            population_size=population_size,
            n_generations=n_generations,
            elite_size=elite_size,
            max_vehicles=max_vehicles,
            crossover_rate=crossover_rate,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.alns_education_iterations = alns_education_iterations
        self.alns_start_temp = alns_start_temp
        self.alns_cooling_rate = alns_cooling_rate

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
        """Resolve instances using the vectorized HGS-ALNS engine.

        Args:
            td: TensorDict containing instance data (locations, wastes, capacity).
            env: Optional environment for reward calculation.
            strategy: Solver strategy (default: "greedy").
            num_starts: Unused in HGS.
            max_steps: Termination criterion based on step count.
            phase: Current execution phase ("train", "val", "test").
            return_actions: Whether to return the full tour tensor.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results dictionary containing:
                - actions (torch.Tensor): Padded action sequences.
                - reward (torch.Tensor): Calculated reward for the solution.
                - cost (torch.Tensor): Raw objective value from the engine.
                - log_likelihood (torch.Tensor): Zero vector (hybrid is non-pi).
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device

        # 1. Setup Data
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

        # 2. Initialization: Random permutations
        initial_solutions = torch.stack(
            [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
        )

        # 3. Solver Execution using HGS-ALNS Engine
        solver = VectorizedHGSALNSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=str(device),
            rng=self.rng,
            generator=self.generator,
            alns_education_iterations=self.alns_education_iterations,
            alns_start_temp=self.alns_start_temp,
            alns_cooling_rate=self.alns_cooling_rate,
        )

        best_routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_generations=self.n_generations,
            population_size=self.population_size,
            elite_size=self.elite_size,
            max_vehicles=self.max_vehicles,
            time_limit=self.time_limit,
            crossover_rate=self.crossover_rate,
        )

        # 4. Format Actions
        all_actions = []
        for b in range(batch_size):
            routes = best_routes_list[b]
            a = routes if isinstance(routes, torch.Tensor) else torch.tensor(routes, device=device, dtype=torch.long)
            all_actions.append(a)

        max_len = max([len(a) for a in all_actions] + [2])
        padded_actions = torch.stack(
            [
                torch.cat(
                    [
                        a,
                        torch.zeros(max_len - len(a), device=device, dtype=torch.long),
                    ]
                )
                for a in all_actions
            ]
        )

        reward = VectorizedHGSPolicy._compute_reward(td, env, padded_actions)

        return {
            "actions": padded_actions,
            "reward": reward,
            "cost": costs.to(device),
            "log_likelihood": torch.zeros(batch_size, device=device),
        }
