"""Neural Optimizer (NeuOpt) Policy.

This module implements the `NeuOptPolicy`, which integrates the NeuOpt-specific
Transformer encoder and pairwise decoder for iterative solution improvement.

Attributes:
    NeuOptPolicy: Policy for guided iterative search in combinatorial spaces.

Example:
    >>> policy = NeuOptPolicy(embed_dim=128)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import NeuOptDecoder
from .encoder import NeuOptEncoder


def execute_neuopt_basis_sequence(solution: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Executes a paper-faithful NeuOpt basis-move sequence (Ma et al. NeurIPS 2023, §4.1, Appendix A).

    Maintains and transforms an evolving open Hamiltonian path P across sequential basis moves:
    1. Step 1 (S-move): S(xa) removes tour edge e_out(xa -> succ(xa)), creating an open Hamiltonian path P
       with endpoints xi = xa (rank 0) and xj = succ(xa) (rank 1).
    2. Steps 2..K (I-moves / E-moves):
       - An I-move I(xv) introduces e_in(xi -> xv), removes e_out(xv -> succ(xv)), and reverses the
         path segment between xj and xv. The endpoints evolve to xi = xj and xj = succ(xv).
       - An E-move E(xj) or early cycle closure introduces e_in(xi -> xj) to close the Hamiltonian path,
         after which subsequent actions are null-padded.
    3. Automatic closure: if the sequence terminates after an I-move without an explicit E-move,
       an automatic E-move connects the current endpoints xi -> xj to close the Hamiltonian cycle.

    Args:
        solution: Tensor of shape [B, N] containing current tours.
        actions: Tensor of shape [B, K] containing basis-move node IDs [x1, x2, ..., xK].

    Returns:
        torch.Tensor: Modified solutions [B, N] representing valid Hamiltonian cycles.
    """
    bs, n = solution.shape
    if n <= 2:
        return solution.clone()
    new_solution = solution.clone()

    for b in range(bs):
        sol_b = solution[b].tolist()
        act_b = actions[b].tolist()

        xa = act_b[0]
        if xa not in sol_b:
            continue
        idx_a = sol_b.index(xa)
        xb = sol_b[(idx_a + 1) % n]

        ranks = {sol_b[(idx_a + r) % n]: r for r in range(n)}

        # Initial Hamiltonian path P after S-move S(xa) removes edge (xa, xb).
        # We maintain the directed path P starting at xj (initially xb, rank 1) and ending at xa (rank 0).
        P = [sol_b[(idx_a + 1 + r) % n] for r in range(n)]
        xj = xb

        for k in range(1, len(act_b)):
            xv = act_b[k]
            # Null action padding after cycle closure or termination
            if xv == xa:
                continue
            # E-move: explicitly connects to xj, closing the cycle
            if xv == xj:
                break

            # I-move: xv must be an internal node on path P
            if xv not in P:
                break
            idx_v = P.index(xv)
            if idx_v == len(P) - 1:
                continue
            if idx_v == 0:
                break

            # Path P starts at xj (P[0]) and ends at xa (P[-1]).
            # Adding edge (xi, xv) creates a cycle between xi and xv.
            # Part 1: xj ... xv
            # Part 2: xw ... xi (where xw is the successor of xv along P)
            part1 = P[: idx_v + 1]
            part2 = P[idx_v + 1 :]

            # New path: part2 + reversed(part1), starts at xw and ends at xj
            P_new = part2 + list(reversed(part1))
            xw = P_new[0]

            if ranks[xw] < ranks[xj]:
                P = list(reversed(P_new))
            else:
                P = P_new
                xj = xw

        # Final closure: re-align to sol_b[0]
        if sol_b[0] in P:
            idx_0 = P.index(sol_b[0])
            tour = P[idx_0:] + P[:idx_0]
        else:
            tour = P
        new_solution[b] = torch.tensor(tour, dtype=solution.dtype, device=solution.device)

    return new_solution


class NeuOptPolicy(ImprovementPolicy):
    """NeuOpt Policy for iterative improvement.

    Leverages a deep Transformer encoder to capture global problem context
    and a move-selection decoder to identify the most beneficial local
    refinements.

    Attributes:
        encoder (NeuOptEncoder): Deep Transformer encoder for state representation.
        decoder (NeuOptDecoder): Pairwise move selection decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initializes the NeuOpt policy.

        Args:
            embed_dim: Latent dimension for all internal representations.
            num_heads: Number of attention heads in the transformer layers.
            num_layers: Depth of the Transformer encoder.
            kwargs: Additional keyword arguments for the base policy.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = NeuOptEncoder(embed_dim, num_heads, num_layers, **kwargs)
        self.decoder = NeuOptDecoder(embed_dim, **kwargs)

    def forward(
        self,
        td: TensorDict,
        env: Any = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: int | None = None,
        phase: str = "train",
        return_actions: bool = True,
        k_basis: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Iteratively applies NeuOpt k-opt refinement moves with full basis-sequence execution.

        Args:
            td: TensorDict containing problem instance and current solution.
            env: Refinement environment managing local search moves.
            strategy: Move selection strategy ("greedy", "sampling").
            num_starts: Number of parallel trajectories per instance.
            max_steps: Maximum number of improvement iterations.
            phase: Current phase ("train", "val", "test").
            return_actions: Whether to include move history in output.
            k_basis: Maximum number of basis moves per k-opt exchange.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Results including reward, log_likelihood, and actions.
        """
        from logic.src.envs import get_env
        from logic.src.utils.decoding import batchify, unbatchify

        if env is None:
            env = get_env(self.env_name or "tsp_kopt")

        td = env.reset(td)

        if num_starts > 1:
            td = batchify(td, num_starts)

        log_probs = []
        actions = []

        if max_steps is None:
            max_steps_td = td.get("max_steps", None)
            max_steps = int(max_steps_td.item()) if max_steps_td is not None else 10

        assert isinstance(max_steps, int), f"max_steps must be an int, got {type(max_steps)}"

        effective_k = k_basis if k_basis is not None else getattr(self.decoder, "k_basis", 2)

        for _i in range(max_steps):
            if self.encoder is None:
                raise ValueError("Encoder must be provided for ImprovementPolicy")
            embeddings = self.encoder(td)

            if self.decoder is None:
                raise ValueError("Decoder must be provided for ImprovementPolicy")
            log_p, move = self.decoder(td, embeddings, env, strategy=strategy, k_basis=effective_k, **kwargs)

            # Execute paper-faithful NeuOpt basis sequence on td["solution"]
            td["solution"] = execute_neuopt_basis_sequence(td["solution"], move)
            td.set("action", move)

            if "i" in td.keys():
                td["i"] = td["i"] + 1

            log_probs.append(log_p)
            actions.append(move)

            if td.get("done", torch.zeros(1)).all():
                break

        out: dict[str, Any] = {
            "reward": env.get_reward(td),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
        }

        if return_actions:
            out["actions"] = torch.stack(actions, dim=1)

        if num_starts > 1:
            out["reward"] = unbatchify(out["reward"], num_starts)
            out["log_likelihood"] = unbatchify(out["log_likelihood"], num_starts)
            if return_actions:
                out["actions"] = unbatchify(out["actions"], num_starts)

        return out
