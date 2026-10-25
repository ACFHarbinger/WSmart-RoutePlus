"""Pointer Network Policy: RNN-based Construction Policy.

This module provides `PointerNetworkPolicy`, an adapter that wraps the classic
LSTM encoder-decoder architecture of `PointerNetwork` into the standard
`AutoregressivePolicy` interface used by the RL4CO pipeline.

Attributes:
    PointerNetworkPolicy: Adapter for LSTM-based constructive search.

Example:
    >>> policy = PointerNetworkPolicy(env_name="vrp", embed_dim=128)
    >>> out = policy(td, env, strategy="greedy")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.core.pointer_network import PointerNetwork
from logic.src.utils.tasks.dummy_problem import DummyProblem


class PointerNetworkPolicy(AutoregressivePolicy):
    """RNN-based Autoregressive Policy.

    Legacy-style policy that maintains internal recurrent states (LSTM) to
    sequentially select nodes. Used primarily as a baseline or for sequential
    graph processing without multi-head attention.

    Attributes:
        model (PointerNetwork): Standalone LSTM pointer architecture.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initializes the Pointer Network Policy adapter.

        Args:
            env_name: Name of the environment identifier.
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of LSTM hidden states.
            kwargs: Additional keyword arguments for the PointerNetwork model.
        """
        super().__init__(env_name=env_name, embed_dim=embed_dim)
        self.model = PointerNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=DummyProblem(env_name),
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "sampling",
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculates construction solution using the recurrent pointer loop.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            strategy: Decoding strategy identifier (e.g., "sampling").
            num_starts: Number of parallel construction starts.
            actions: Optional pre-selected actions for teacher forcing.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Result mapping containing:
                - reward: calculated scores [B].
                - log_likelihood: cumulative path log probabilities [B].
                - actions: selected node indices [B, SeqLen].
        """
        batch_size, graph_size, _ = td["locs"].size()
        inputs = td["locs"]

        # 1. Custom recurrent embedding projection
        embedded_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, inputs.size(-1)),
            self.model.embedding,
        ).view(graph_size, batch_size, -1)

        # 2. Strategy configuration
        self.model.set_strategy(strategy)

        # 3. Execution (Sampling vs Teacher Forcing)
        if actions is not None:
            # Evaluate log-probs for existing actions
            log_p_output, out_actions = self.model._inner(embedded_inputs, eval_tours=actions)
        else:
            # Generate new actions via recurrent construction
            log_p_output, out_actions = self.model._inner(embedded_inputs)

        # 4. Probabilities extraction: gather [B, Steps, Nodes] -> [B, Steps]
        log_p = log_p_output.gather(2, out_actions.unsqueeze(-1)).squeeze(-1)
        log_likelihood = log_p.sum(dim=1)

        # 5. Env-based reward calculation
        assert env is not None, "Environment must be provided for reward calculation."
        reward = env.get_reward(td, out_actions)

        return {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "actions": out_actions,
        }
