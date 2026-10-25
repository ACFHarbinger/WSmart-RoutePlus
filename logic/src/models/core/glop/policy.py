"""GLOP Policy: Global-Local Optimization Policy.

This module implements `GLOPPolicy`, a two-stage hierarchical model that first
partitions a large problem into sub-clusters using a NAR encoder, and then
refines each cluster using dedicated local solvers (e.g., greedy TSP).

Attributes:
    SubProblemSolverType: Type alias for local solver callables.
    GLOPPolicy: Hierarchical partitioning policy.

Example:
    >>> from logic.src.models.core.glop.policy import GLOPPolicy
    >>> policy = GLOPPolicy(env_name="cvrp", subprob_solver='greedy')
    >>> out = policy(td, my_env)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.non_autoregressive.policy import NonAutoregressivePolicy
from logic.src.models.subnets.modules.glop_factory import get_adapter

# Type for subproblem solvers
SubProblemSolverType = Callable[[torch.Tensor], torch.Tensor]


class GLOPPolicy(NonAutoregressivePolicy):
    """GLOP: Global-Local Optimization Policy.

    Decomposes large-scale optimization into:
    1. Global Partitioning: Discrete node clustering via Non-Autoregressive model.
    2. Local Optimization: Fine-grained routing within each cluster.

    Attributes:
        n_samples (int): parallel partition variants to explore.
        subprob_solver (Union[SubProblemSolverType, str]): Logic for sub-problem routing.
        subprob_batch_size (int): Max sub-problems processed in one local pass.
        adapter_class (Type): Environment-specific adapter for partition integration.
    """

    def __init__(
        self,
        env_name: str = "cvrp",
        n_samples: int = 10,
        temperature: float = 1.0,
        embed_dim: int = 64,
        subprob_solver: Union[SubProblemSolverType, str] = "greedy",
        subprob_batch_size: int = 2000,
        **encoder_kwargs: Any,
    ) -> None:
        """Initializes the GLOP policy.

        Args:
            env_name: Name of the environment identifier.
            n_samples: Number of parallel partition samples.
            temperature: Softmax temperature for partition exploration.
            embed_dim: Dimensionality of latent embeddings.
            subprob_solver: Local solver identifier or callable.
            subprob_batch_size: Max sub-problems processed per local pass.
            encoder_kwargs: Additional keyword arguments for the NAR encoder.
        """
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            temperature=temperature,
            train_strategy="multistart_sampling",
            val_strategy="multistart_greedy",
            test_strategy="multistart_greedy",
            **encoder_kwargs,
        )

        self.n_samples = n_samples
        self.subprob_solver = subprob_solver
        self.subprob_batch_size = subprob_batch_size

        self.adapter_class = get_adapter(env_name)

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: Literal["train", "val", "test"] = "test",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        strategy: Optional[str] = None,
        **decoding_kwargs: Any,
    ) -> Dict[str, Any]:
        """Encodes partitions and refines them via local construction.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            calc_reward: Whether to calculate environment rewards.
            return_actions: Whether to include refined actions in results.
            return_entropy: Whether to calculate partition entropy.
            strategy: Optional decoding strategy override.
            decoding_kwargs: Additional keyword arguments for NAR decoding.

        Returns:
            Dict[str, Any]: Results including 'reward' and optionally 'actions'.
        """
        # Phase 1: Partition nodes using the NAR encoder
        par_out = super().forward(
            td=td,
            env=env,  # type: ignore[arg-type]
            phase=phase,
            calc_reward=False,
            return_actions=True,
            return_entropy=return_entropy,
            num_starts=self.n_samples,
            strategy=strategy,  # type: ignore[arg-type]
            **decoding_kwargs,
        )

        partition_actions = par_out["actions"]

        # Phase 2: Solve each partition locally
        local_out = self._local_policy(td, partition_actions)
        final_actions = local_out["actions"]

        # Harmonize outputs
        out = par_out
        if return_actions:
            out["actions"] = final_actions
        else:
            out.pop("actions", None)

        if calc_reward and env is not None:
            from logic.src.utils.decoding import batchify

            td_repeated = batchify(td, self.n_samples)
            reward = env.get_reward(td_repeated, final_actions)
            out["reward"] = reward.detach()

        return out

    @torch.no_grad()
    def _local_policy(
        self,
        td: TensorDict,
        partition_actions: torch.Tensor,
    ) -> Dict[str, Any]:
        """Orchestrates local solver application across all partitions.

        Args:
            td: Original problem state.
            partition_actions: Cluster assignments [B*Samples, N].

        Returns:
            Dict[str, Any]: Refined actions merged back into full tours.
        """
        from logic.src.utils.decoding import unbatchify

        # Transform to internal shape for adapter processing
        partition_actions_reshaped = (
            unbatchify(partition_actions, self.n_samples)
            .transpose(0, 1)
            .contiguous()
            .view(-1, partition_actions.size(-1))
        )

        # Initialize environment-aware partition adapter
        adapter = self.adapter_class(
            td,
            partition_actions_reshaped,
            subprob_batch_size=self.subprob_batch_size,
        )

        # Iterate through sub-problem batches
        solver = self._get_solver()
        for mapping in adapter.get_batched_subprobs():
            subprob_actions = solver(mapping.subprob_coordinates)
            adapter.update_actions(mapping, subprob_actions)

        # Extract and restore original batch/sample ordering
        actions = adapter.get_actions().to(td.device)
        batch_size = td.batch_size[0] if td.batch_size else partition_actions.size(0) // self.n_samples
        actions = actions.view(batch_size, self.n_samples, -1).transpose(0, 1).contiguous()
        actions = actions.view(-1, actions.size(-1))

        return {"actions": actions}

    def _get_solver(self) -> SubProblemSolverType:
        """Retrieves the concrete solver implementation based on config.

        Returns:
            SubProblemSolverType: The resolved local solver function.
        """
        if isinstance(self.subprob_solver, str):
            if self.subprob_solver == "greedy":
                return self._greedy_tsp_solver
            elif self.subprob_solver == "nearest":
                return self._nearest_neighbor_solver
            else:
                raise ValueError(f"Unknown local solver: {self.subprob_solver}")
        return self.subprob_solver

    @staticmethod
    def _greedy_tsp_solver(coords: torch.Tensor) -> torch.Tensor:
        """Baseline greedy/nearest neighbor solver for local segments.

        Args:
            coords: Spatial coordinates [N_sub, 2].

        Returns:
            torch.Tensor: Node sequence trial [N_sub].
        """
        n = coords.size(0)
        if n <= 1:
            return torch.arange(n)

        visited = torch.zeros(n, dtype=torch.bool, device=coords.device)
        tour = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = tour[-1]
            dists = torch.cdist(coords[current : current + 1], coords).squeeze(0)
            dists[visited] = float("inf")
            nearest = dists.argmin().item()
            tour.append(nearest)  # type: ignore
            visited[nearest] = True

        return torch.tensor(tour, device=coords.device)

    @staticmethod
    def _nearest_neighbor_solver(coords: torch.Tensor) -> torch.Tensor:
        """Alias for greedy TSP construction.

        Args:
            coords: Spatial coordinates [N_sub, 2].

        Returns:
            torch.Tensor: Node sequence trial [N_sub].
        """
        return GLOPPolicy._greedy_tsp_solver(coords)
