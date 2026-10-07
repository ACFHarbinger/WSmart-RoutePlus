"""Hybrid Two-Stage Policy for routing optimization.

This module implements `HybridTwoStagePolicy`, which decomposes the routing
task into two phases:
1. Initialization: Selecting a high-level algorithm (HGS, ALNS, ACO).
2. Refinement: Iteratively selecting and applying vectorized local search
   operators to improve the initial solution.

Attributes:
    HybridTwoStagePolicy: Combined neural-heuristic optimization model.

Example:
    >>> policy = HybridTwoStagePolicy(env_name="tsp")
    >>> out = policy(td, env)
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
from logic.src.policies.vector.adaptive_large_neighborhood_search import VectorizedALNS
from logic.src.policies.vector.ant_colony_system import VectorizedACOPolicy
from logic.src.policies.vector.hybrid_genetic_search import VectorizedHGS
from logic.src.policies.vector.operators import (
    vectorized_cluster_removal,
    vectorized_cross_exchange,
    vectorized_ejection_chain,
    vectorized_greedy_insertion,
    vectorized_lambda_interchange,
    vectorized_lkh,
    vectorized_or_opt,
    vectorized_random_removal,
    vectorized_regret_k_insertion,
    vectorized_relocate,
    vectorized_shaw_removal,
    vectorized_string_removal,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
    vectorized_type_i_unstringing,
    vectorized_type_ii_unstringing,
    vectorized_type_iii_unstringing,
    vectorized_type_iv_unstringing,
    vectorized_worst_removal,
)

from .improvement_step_decoder import ImprovementStepDecoder


class HybridTwoStagePolicy(AutoregressivePolicy):
    """Hybrid Meta-Heuristic Policy with Iterative Refinement.

    Combines neural selection of global initialization solvers with neural
    scheduling of local search operators. Effectively acts as a learned
    controller for classic Operations Research heuristics.

    Attributes:
        refine_steps (int): count of improvement iterations.
        seed (int): RNG seed for operator sampling.
        generator (torch.Generator): Local random state.
        init_embedding (nn.Module): Node feature projection.
        encoder (GraphAttentionEncoder): Context transformer.
        init_router (nn.Sequential): Classifier for Stage 1 (HGS/ALNS/ACO).
        INIT_ALGOS (List[str]): Registry of initialization choices.
        OPERATORS (List[Callable]): Registry of vectorized refinement moves.
        improvement_decoder (ImprovementStepDecoder): Operator scheduler.
        aco_model (Optional[VectorizedACOPolicy]): Neural ACO instance (lazy).
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        refine_steps: int = 10,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the hybrid two-stage policy.

        Args:
            env_name: Name of the environment (e.g., "tsp", "ptp").
            embed_dim: Dimension of node embeddings.
            hidden_dim: MLP hidden layer dimension.
            n_encode_layers: Number of Graph Attention Encoder layers.
            n_heads: Number of attention heads in the GAT.
            refine_steps: Number of refinement iterations to perform.
            seed: Random seed for reproducibility.
            kwargs: Additional keyword arguments for the encoder.
        """
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.refine_steps = refine_steps
        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        self.init_embedding = get_init_embedding(env_name, embed_dim)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=hidden_dim,
            n_layers=n_encode_layers,
            **kwargs,
        )

        # Stage 1: Init Selector
        self.init_router = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.INIT_ALGOS = ["hgs", "alns", "aco"]

        # Stage 2: Improvement Registry
        self.OPERATORS = [
            vectorized_cluster_removal,
            vectorized_random_removal,
            vectorized_shaw_removal,
            vectorized_string_removal,
            vectorized_worst_removal,
            vectorized_cross_exchange,
            vectorized_ejection_chain,
            vectorized_lambda_interchange,
            vectorized_or_opt,
            vectorized_relocate,
            vectorized_swap,
            vectorized_greedy_insertion,
            vectorized_regret_k_insertion,
            vectorized_lkh,
            vectorized_swap_star,
            vectorized_three_opt,
            vectorized_two_opt,
            vectorized_two_opt_star,
            vectorized_type_i_unstringing,
            vectorized_type_ii_unstringing,
            vectorized_type_iii_unstringing,
            vectorized_type_iv_unstringing,
        ]
        self.improvement_decoder = ImprovementStepDecoder(
            embed_dim=embed_dim, n_operators=len(self.OPERATORS), hidden_dim=hidden_dim
        )
        self.aco_model: Optional[VectorizedACOPolicy] = None

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes current model state, handling PRNG portability.

        Returns:
            Dict[str, Any]: Attribute map including PRNG byte-state.
        """
        state = self.__dict__.copy()
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores model state from serialized dictionary.

        Args:
            state: Pre-loaded attribute map.
        """
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def _initialize_tours(
        self,
        td: TensorDict,
        strategy: str,
        embeddings: torch.Tensor,
        env: Optional[RL4COEnvBase] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Generates a batch of initial solutions using hybrid solvers.

        Args:
            td: TensorDict containing instance data.
            strategy: Choice strategy ("greedy" or "sampling").
            embeddings: Contextual node embeddings.
            env: The problem environment.
            kwargs: Additional keyword arguments for the initial solvers.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - tours: Initial node sequences [B, N].
                - init_choice_idx: Algorithm category selected per instance [B].

        """
        batch_size = td.size(0)
        device = td["locs"].device

        graph_context = embeddings.mean(dim=1)
        init_logits = self.init_router(graph_context)

        if strategy == "greedy":
            init_choice_idx = init_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(init_logits, dim=-1)
            init_choice_idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        tours = torch.zeros(batch_size, td["locs"].size(1), dtype=torch.long, device=device)

        if self.aco_model is None:
            self.aco_model = VectorizedACOPolicy(
                env_name=self.env_name,
                embed_dim=self.encoder.embed_dim,
                **kwargs,  # type: ignore[arg-type]
            )

        for algo_idx, algo_name in enumerate(self.INIT_ALGOS):
            mask = init_choice_idx == algo_idx
            if not mask.any():
                continue

            sub_td = td[mask]
            dist_matrix = self._get_dist_matrix(sub_td)
            wastes = sub_td.get("waste", None)

            if algo_name == "hgs":
                solver = VectorizedHGS(dist_matrix, wastes, vehicle_capacity=1.0, device=device)
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_generations=5, population_size=10)
            elif algo_name == "alns":
                solver = VectorizedALNS(dist_matrix, wastes, vehicle_capacity=1.0, device=device)
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_iterations=10)  # type: ignore
            elif algo_name == "aco":
                out = self.aco_model(sub_td, env, **kwargs)
                sub_tours = out["actions"]
            else:
                raise ValueError(f"Unknown initialization algorithm: {algo_name}")

            if not isinstance(sub_tours, list):
                tours[mask] = sub_tours.to(dtype=torch.long)

        return tours, init_choice_idx

    def _apply_operator_step(
        self,
        td: TensorDict,
        embeddings: torch.Tensor,
        current_tours: torch.Tensor,
        dist_matrix_all: torch.Tensor,
        removed_nodes_state: torch.Tensor,
        strategy: str,
        env: Optional[RL4COEnvBase] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 2 Step: Selects and executes a single improvement operator.

        Args:
            td: problem state.
            embeddings: graph features.
            current_tours: existing sequences [B, N_max].
            dist_matrix_all: pairwise distances [B, N, N].
            removed_nodes_state: buffer for partially destroyed solutions.
            strategy: selection mode.
            env: problem dynamics.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - next_tours: improved sequences.
                - updated_removed_nodes: new buffer state.
                - log_prob: selection likelihood per instance.
        """
        batch_size = td.size(0)
        device = td.device
        op_logits, _ = self.improvement_decoder(td, embeddings, env)

        log_p = torch.zeros(batch_size, device=device)
        if strategy == "greedy":
            op_idx = op_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(op_logits, dim=-1)
            op_idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            log_p = torch.log(probs.gather(1, op_idx.unsqueeze(-1)) + 1e-10).squeeze(-1)

        next_tours = current_tours.clone()
        updated_removed_nodes_list = [None] * batch_size

        for o_i, operator_fn in enumerate(self.OPERATORS):
            mask = op_idx == o_i
            if not mask.any():
                continue

            sub_tours, sub_dist = current_tours[mask], dist_matrix_all[mask]
            sub_removed = (
                removed_nodes_state[mask]
                if removed_nodes_state.size(0) == batch_size
                else torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)
            )

            new_sub_tours, new_sub_removed = self._execute_refinement_operator(
                operator_fn, sub_tours, sub_dist, sub_removed, device
            )

            # Re-pad if operator altered tour length
            if new_sub_tours.size(1) > next_tours.size(1):
                diff = new_sub_tours.size(1) - next_tours.size(1)
                padding = torch.zeros((batch_size, diff), dtype=next_tours.dtype, device=device)
                next_tours = torch.cat([next_tours, padding], dim=1)

            if new_sub_tours.size(1) < next_tours.size(1):
                diff = next_tours.size(1) - new_sub_tours.size(1)
                padding = torch.zeros(
                    (new_sub_tours.size(0), diff),
                    dtype=new_sub_tours.dtype,
                    device=device,
                )
                new_sub_tours = torch.cat([new_sub_tours, padding], dim=1)

            next_tours[mask] = new_sub_tours
            sub_indices = torch.nonzero(mask).squeeze(-1)
            if sub_indices.dim() == 0:
                sub_indices = sub_indices.unsqueeze(0)
            for i, idx in enumerate(sub_indices):
                updated_removed_nodes_list[idx.item()] = new_sub_removed[i]

        return (
            next_tours,
            self._assemble_removed_state(updated_removed_nodes_list, device),
            log_p,
        )

    def _execute_refinement_operator(
        self,
        operator_fn: Any,
        sub_tours: torch.Tensor,
        sub_dist: torch.Tensor,
        sub_removed: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Safely invokes a vectorized heuristic operator.

        Args:
            operator_fn: callable vectorized function.
            sub_tours: input tour segment.
            sub_dist: distance matrix segment.
            sub_removed: removed nodes buffer.
            device: target hardware.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (modified tours, new removed nodes).
        """
        sig = inspect.signature(operator_fn)
        valid_kwargs = {}
        if "max_iterations" in sig.parameters:
            valid_kwargs["max_iterations"] = 5
        if "n_remove" in sig.parameters:
            valid_kwargs["n_remove"] = int(sub_tours.size(1) * 0.1) + 1
        if "removed_nodes" in sig.parameters:
            valid_kwargs["removed_nodes"] = sub_removed

        try:
            result = operator_fn(sub_tours, sub_dist, **valid_kwargs)
        except Exception:
            result = sub_tours

        if isinstance(result, tuple):
            return result

        return result, (
            torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)
            if "removed_nodes" in sig.parameters
            else sub_removed
        )

    def _assemble_removed_state(self, removed_list: list, device: torch.device) -> torch.Tensor:
        """Pads and stacks variable-length removed node lists into a tensor.

        Args:
            removed_list: batch of tensors [N_rem].
            device: target hardware.

        Returns:
            torch.Tensor: padded removed node tensor [B, Max_Rem].
        """
        batch_size = len(removed_list)
        max_len = max([r.size(0) if r is not None else 0 for r in removed_list])
        if max_len == 0:
            return torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        new_state = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        for b, r in enumerate(removed_list):
            if r is not None:
                new_state[b, : r.size(0)] = r
        return new_state

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Main execution logic: Init -> iterative refinement.

        Args:
            td: TensorDict containing instance data.
            env: The problem environment.
            strategy: Selection strategy for both stages.
            kwargs: Additional keyword arguments for the policies.

        Returns:
            Dict[str, Any]: Result map with final reward, actions, and audit data.
        """
        # Phase 1: Contextual Encoding
        init_embeds = self.init_embedding(td)
        embeddings = self.encoder(init_embeds)  # type: ignore[misc]

        # Phase 2: Solve via meta-algorithm (HGS/ALNS/ACO)
        assert env is not None, "Environment must be provided for two-stage solving."
        current_tours, init_choice_idx = self._initialize_tours(td, strategy, embeddings, env, **kwargs)

        # Phase 3: Selection and application of low-level heuristics
        dist_matrix_all = self._get_dist_matrix(td)
        removed_nodes_state = torch.zeros((td.size(0), 0), dtype=torch.long, device=td.device)
        log_likelihood = torch.zeros(td.size(0), device=td.device)

        for _step in range(self.refine_steps):
            current_tours, removed_nodes_state, step_log_p = self._apply_operator_step(
                td,
                embeddings,
                current_tours,
                dist_matrix_all,
                removed_nodes_state,
                strategy,
                env,
            )
            log_likelihood = log_likelihood + step_log_p

        # Evaluation
        reward = env.get_reward(td, current_tours)

        return {
            "actions": current_tours,
            "reward": reward,
            "log_likelihood": log_likelihood,
            "init_choice": init_choice_idx,
        }

    def _get_dist_matrix(self, td: TensorDict) -> torch.Tensor:
        """Utility to calculate graph distance adjacency.

        Args:
            td: TensorDict containing "locs".

        Returns:
            torch.Tensor: Pairwise distance matrix of shape [B, N, N].
        """
        locs = td["locs"]
        return torch.cdist(locs, locs)

    def _get_random_tours(self, td: TensorDict) -> torch.Tensor:
        """Utility for naive initial permutations (excluding the depot).

        Args:
            td: TensorDict containing the problem context.

        Returns:
            torch.Tensor: Batch of random permutations [B, N-1].
        """
        B, N, _ = td["locs"].shape
        perms = torch.stack([torch.randperm(N - 1, device=td.device) + 1 for _ in range(B)])
        return perms
