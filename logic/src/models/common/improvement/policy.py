"""Improvement Policy module.

This module provides the implementation of the `ImprovementPolicy`, which
recursively applies local search moves or refinement actions to improve
an initial solution.

Attributes:
    ImprovementPolicy: Base class for improvement policies.

Example:
    >>> encoder = ImprovementEncoder(embed_dim=128)
    >>> decoder = ImprovementDecoder(embed_dim=128)
    >>> policy = ImprovementPolicy(encoder, decoder, env_name="tsp_kopt")
    >>> out = policy(td, env)
"""

from __future__ import annotations

import random
from abc import ABC
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs import get_env
from logic.src.envs.base.base import RL4COEnvBase

from .decoder import ImprovementDecoder
from .encoder import ImprovementEncoder


class ImprovementPolicy(nn.Module, ABC):
    """Base class for improvement policies.

    Improvement policies take an instance and an existing solution as input,
    calculating embeddings for both, and then iteratively predicting and
    applying moves to minimize the cost (or maximize reward).

    Attributes:
        encoder: Encodes problem+solution state.
        decoder: Predicts improvement operations.
        env_name: Name of the refinement environment.
        embed_dim: Dimensionality of feature vectors.
        seed: Seed for reproducibility of stochastic actions.
        device: Computing device.
        generator: Seeded generator for sampling.
        rng: Standard library random generator.
    """

    def __init__(
        self,
        encoder: Optional[ImprovementEncoder] = None,
        decoder: Optional[ImprovementDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        seed: int = 42,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialize the ImprovementPolicy.



        Args:
            encoder: Encoder module for problem/solution state.
            decoder: Decoder module for move selection.
            env_name: Optional environment identifier for refinement.
            embed_dim: Dimensionality of latent embeddings.
            seed: Random seed for reproducibility.
            device: Target computing device.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.seed = seed
        self.device = torch.device(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for serialization handling non-picklable components.

        Returns:
            Dict[str, Any]: Policy state dictionary.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state from a serialized dictionary.

        Args:
            state: Serialized policy attributes.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

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
        """Iteratively apply refinement moves to the current solution.

        The policy performs a fixed or dynamic number of improvement steps,
        collecting the log-likelihood of each decision and updating the solution
        in the provided or default environment.



        Args:
            td: TensorDict containing problem instance and current solution.
            env: Refinement environment managing local search moves.
            strategy: Move selection strategy (e.g., "greedy", "sampling").
            num_starts: Number of parallel trajectories per instance.
            max_steps: Maximum number of improvement iterations.
            phase: Current phase ("train", "val", "test").
            return_actions: Whether to include the move history in output.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results including:
                - reward (torch.Tensor): Final reward for the improved solution.
                - log_likelihood (torch.Tensor): Cumulative log probability of moves.
                - actions (torch.Tensor, optional): The sequence of applied moves.

        Raises:
            ValueError: If either the encoder or decoder is not initialized.
        """
        if env is None:
            # Try to get from name or default to base
            env = get_env(self.env_name or "tsp_kopt")

        # Initial solution generation (done by env.reset)
        from logic.src.utils.decoding import batchify, unbatchify

        td = env.reset(td)

        # Batch for multiple starts if requested
        if num_starts > 1:
            td = batchify(td, num_starts)

        # Iterative improvement loop
        log_probs = []
        actions = []

        # Default steps from td or config
        if max_steps is None:
            max_steps_td = td.get("max_steps", None)
            max_steps = int(max_steps_td.item()) if max_steps_td is not None else 10

        assert isinstance(max_steps, int), f"max_steps must be an int, got {type(max_steps)}"

        for _i in range(max_steps):
            # 1. Encode current state
            if self.encoder is None:
                raise ValueError("Encoder must be provided for ImprovementPolicy")
            embeddings = self.encoder(td)

            # 2. Predict move via decoder
            if self.decoder is None:
                raise ValueError("Decoder must be provided for ImprovementPolicy")
            log_p, move = self.decoder(td, embeddings, env, strategy=strategy, **kwargs)

            # 3. Apply move
            td.set("action", move)
            td = env.step(td)["next"]

            log_probs.append(log_p)
            actions.append(move)

            if td["done"].all():
                break

        # Collect results
        out = {
            "reward": env.get_reward(td),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
        }

        if return_actions:
            out["actions"] = torch.stack(actions, dim=1)

        # Unbatch if multiple starts
        if num_starts > 1:
            out["reward"] = unbatchify(out["reward"], num_starts)
            out["log_likelihood"] = unbatchify(out["log_likelihood"], num_starts)
            if return_actions:
                out["actions"] = unbatchify(out["actions"], num_starts)

        return out
