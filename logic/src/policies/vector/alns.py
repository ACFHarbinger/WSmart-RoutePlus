"""ALNS Policy wrapper for RL4CO integration.

This module provides the `VectorizedALNS` class, which wraps the high-performance
ALNS engine to satisfy the `AutoregressivePolicy` interface used in training
and evaluation pipelines.

Attributes:
    VectorizedALNS: ALNS solver wrapper for the RL4CO pipeline.

Example:
    >>> policy = VectorizedALNS(env_name="vrpp")
    >>> out = policy(td)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.policies.vector.hgs import VectorizedHGS

from .adaptive_large_neighborhood_search import (
    VectorizedALNS as VectorizedALNSEngine,
)


class VectorizedALNS(AutoregressivePolicy):
    """RL4CO-compatible ALNS Policy wrapper.

    Orchestrates the data extraction from environment states (TensorDict),
    execution of the vectorized ALNS engine, and conversion of the resulting
    routes back into a standardized action format.

    Attributes:
        time_limit: Execution timeout for the solver in seconds.
        max_iterations: Internal search loop limit.
        max_vehicles: Fleet size limit for discretization.
        start_temp: Initial temperature for Simulated Annealing.
        cooling_rate: SA temperature decay factor.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        max_iterations: int = 500,
        max_vehicles: int = 0,
        start_temp: float = 0.5,
        cooling_rate: float = 0.9995,
        device: str = "cpu",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the ALNS policy wrapper.

        Args:
            env_name: Name of the environment identifier.
            time_limit: Maximum search time in seconds.
            max_iterations: Maximum number of improvement iterations.
            max_vehicles: Fleet size limit (0 for unlimited).
            start_temp: Initial temperature for Simulated Annealing.
            cooling_rate: Decay rate for the temperature.
            device: Computing device for operations.
            seed: Random seed for search reproducibility.
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name=env_name, device=device, seed=seed, **kwargs)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.max_vehicles = max_vehicles
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate

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
        """Execute the ALNS solver on a batch of environment instances.

        Extracts locations, wastes, and capacities from the TensorDict, executes
        the vectorized ALNS optimization, and computes rewards for the resulting
        action sequences.

        Args:
            td: TensorDict containing instance data.
            env: Optional environment for reward calculation.
            strategy: Solver strategy (e.g., "greedy").
            num_starts: Unused in ALNS.
            max_steps: Maximum step constraint.
            phase: Current phase ("train", "val", "test").
            return_actions: Whether to return full action sequences.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results including:
                - reward (torch.Tensor): Calculated reward/cost for the solution.
                - actions (torch.Tensor): The sequence of applied moves.
                - log_likelihood (torch.Tensor): Zero vector (ALNS is non-pi).
                - cost (torch.Tensor): Raw objective values from the engine.
        """
        batch_size = td.batch_size[0]
        device = td.device if td.device is not None else td["locs"].device

        # 1. Data Parsing
        customers = td["locs"]  # (batch, num_nodes, 2)
        depot = td["depot"].unsqueeze(1)  # (batch, 1, 2)
        locs = torch.cat([depot, customers], dim=1)  # (batch, num_nodes + 1, 2)

        num_nodes = locs.shape[1]

        # Compute Euclidean distance matrix for the engine
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        waste_at_nodes = td.get("waste", torch.zeros(batch_size, num_nodes - 1, device=device))
        # Prepend 0/None waste for the depot node
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)  # pyrefly: ignore [no-matching-overload]

        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:  # pyrefly: ignore [missing-attribute]
            capacity = capacity.expand(batch_size)  # pyrefly: ignore [missing-attribute]

        # 2. Engine Execution
        initial_solutions = torch.stack(
            [torch.randperm(num_nodes - 1, device=device, generator=self.generator) + 1 for _ in range(batch_size)]
        )

        solver = VectorizedALNSEngine(
            dist_matrix=dist_matrix,
            wastes=waste,
            vehicle_capacity=capacity,  # pyrefly: ignore [bad-argument-type]
            time_limit=self.time_limit,
            device=str(device),
            generator=self.generator,
        )

        routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.max_iterations,
            time_limit=self.time_limit,
            max_vehicles=kwargs.get("max_vehicles", self.max_vehicles),
            start_temp=self.start_temp,
            cooling_rate=self.cooling_rate,
        )

        # 3. Output Packaging
        all_actions = []
        for b in range(batch_size):
            route_nodes_raw = routes_list[b]
            if isinstance(route_nodes_raw, torch.Tensor):
                route_tensor = route_nodes_raw
            else:
                route_tensor = torch.tensor(route_nodes_raw, device=device)

            actions = route_tensor if route_tensor.size(0) > 0 else torch.tensor([0, 0], device=device)
            all_actions.append(actions)

        # Pad variable-length action sequences for batch reward calculation
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        reward = VectorizedHGS._compute_reward(td, env, padded_actions)

        return {
            "reward": reward,
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }
