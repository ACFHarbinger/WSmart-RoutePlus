"""Neural Neighborhood Search (N2S) Policy.

This module implements the `N2SPolicy`, which integrates the N2S-specific
encoder and decoder for iterative improvement of combinatorial solutions.

Attributes:
    N2SPolicy: Collaborative Transformer policy for neighborhood search.

Example:
    >>> policy = N2SPolicy(embed_dim=128, k_neighbors=20)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from logic.src.models.common.improvement.policy import ImprovementPolicy

from .decoder import N2SDecoder
from .encoder import N2SEncoder


def execute_n2s_request_move(solution: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Executes paper-faithful N2S request removal and precedence-correct reinsertion (Li et al. 2021).

    With action a_t = {(i+, i-), (j, k)}:
    1. Removes request node pair (i+, i-) from the tour.
    2. Reinserts pickup node i+ after node j, and delivery node i- after node k
       in the residual tour, strictly maintaining pickup-before-delivery precedence.

    Args:
        solution: Current tour tensor [B, N].
        actions: Action tensor [B, 4] with rows [i+, i-, j, k]. Malformed actions raise ValueError.

    Returns:
        torch.Tensor: Modified tour tensor [B, N].
    """
    bs, n = solution.shape
    if actions.ndim != 2 or actions.shape != (bs, 4):
        raise ValueError("N2S request actions must have shape [batch, 4]")
    if actions.is_floating_point() or actions.is_complex() or actions.dtype == torch.bool:
        raise ValueError("N2S request actions must contain integer node IDs")

    new_solution = solution.clone()
    for b in range(bs):
        sol_b = solution[b].tolist()
        i_plus = int(actions[b, 0].item())
        i_minus = int(actions[b, 1].item())
        j_node = int(actions[b, 2].item())
        k_node = int(actions[b, 3].item())

        residual = [x for x in sol_b if x != i_plus and x != i_minus]
        if i_plus == i_minus or sol_b.count(i_plus) != 1 or sol_b.count(i_minus) != 1:
            raise ValueError("Request nodes must be distinct and each occur once in the tour")
        if j_node not in residual or k_node not in residual:
            raise ValueError("Insertion anchors must belong to the residual tour")
        pos_j = residual.index(j_node)
        pos_k = residual.index(k_node)
        if pos_j > pos_k:
            raise ValueError("Pickup insertion must not follow delivery insertion")
        p1, n1 = pos_j, i_plus
        p2, n2 = pos_k, i_minus

        if p1 == p2:
            updated = residual[: p1 + 1] + [n1, n2] + residual[p1 + 1 :]
        else:
            updated = residual[: p1 + 1] + [n1] + residual[p1 + 1 : p2 + 1] + [n2] + residual[p2 + 1 :]

        new_solution[b] = torch.tensor(updated, dtype=solution.dtype, device=solution.device)

    return new_solution


def validate_pdp_solution(
    solution: torch.Tensor,
    partner_ids: torch.Tensor,
    is_pickup: torch.Tensor,
    is_delivery: torch.Tensor,
) -> None:
    """Validates that all pickups precede their respective deliveries in the tour."""
    bs, n = solution.shape
    expected = torch.arange(n, device=solution.device).unsqueeze(0).expand(bs, -1)
    if solution.is_floating_point() or solution.is_complex() or solution.dtype == torch.bool:
        raise ValueError("PDP solution must be an integer permutation")
    if not torch.equal(solution.sort(dim=1).values, expected):
        raise ValueError("PDP solution must be a permutation of all node IDs")
    sol_pos = torch.zeros(bs, n, dtype=torch.long, device=solution.device)
    positions = torch.arange(n, device=solution.device).unsqueeze(0).expand(bs, -1)
    sol_pos.scatter_(1, solution, positions)
    pos_pickup = sol_pos
    pos_delivery = sol_pos.gather(1, partner_ids)
    violating = is_pickup & (pos_pickup >= pos_delivery)
    if violating.any():
        viol_b, viol_node = torch.where(violating)
        b0, p0 = int(viol_b[0]), int(viol_node[0])
        d0 = int(partner_ids[b0, p0])
        raise ValueError(
            f"Initial PDP solution violates precedence constraint: "
            f"pickup {p0} (pos {int(pos_pickup[b0, p0])}) appears after "
            f"delivery {d0} (pos {int(pos_delivery[b0, p0])})"
        )


def create_feasible_pdp_solution(
    bs: int,
    n: int,
    partner_ids: torch.Tensor,
    is_pickup: torch.Tensor,
    is_delivery: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generates a canonical precedence-feasible initial PDP solution for each instance in the batch.

    In the resulting tour, all pickups precede their respective deliveries (e.g. [0, p1, ..., pm, d1, ..., dm]).
    """
    solution = torch.zeros(bs, n, dtype=torch.long, device=device)
    node_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
    is_depot = (partner_ids == node_indices) | (node_indices == 0)

    for b in range(bs):
        depots = [i for i in range(n) if bool(is_depot[b, i])]
        pickups = [i for i in range(n) if bool(is_pickup[b, i])]
        deliveries = [int(partner_ids[b, p].item()) for p in pickups]
        tour = depots + pickups + deliveries
        solution[b] = torch.tensor(tour, dtype=torch.long, device=device)
    return solution


def _update_n2s_history(
    history: torch.Tensor,
    move: torch.Tensor,
    window_queue: list[tuple[torch.Tensor, torch.Tensor]],
    window_size: int,
) -> None:
    """Updates N2S removal history buffer and recency indicators (Li et al. 2021, Eq. 13)."""
    x_rem = move[:, 0]
    x_pair = move[:, 1]
    window_queue.append((x_rem, x_pair))
    if len(window_queue) > window_size:
        x_old_rem, x_old_pair = window_queue.pop(0)
        history[:, :, 0].scatter_add_(
            1, x_old_rem.unsqueeze(-1), -torch.ones_like(x_old_rem.unsqueeze(-1), dtype=torch.float32)
        )
        history[:, :, 0].scatter_add_(
            1, x_old_pair.unsqueeze(-1), -torch.ones_like(x_old_pair.unsqueeze(-1), dtype=torch.float32)
        )
        history[:, :, 0].clamp_(min=0.0)

    history[:, :, 0].scatter_add_(
        1, x_rem.unsqueeze(-1), torch.ones_like(x_rem.unsqueeze(-1), dtype=torch.float32)
    )
    history[:, :, 0].scatter_add_(
        1, x_pair.unsqueeze(-1), torch.ones_like(x_pair.unsqueeze(-1), dtype=torch.float32)
    )
    history[:, :, 3] = history[:, :, 2]
    history[:, :, 2] = history[:, :, 1]
    history[:, :, 1] = 0.0
    history[:, :, 1].scatter_(1, x_rem.unsqueeze(-1), 1.0)
    history[:, :, 1].scatter_(1, x_pair.unsqueeze(-1), 1.0)


def _sync_n2s_caller(orig_td: Optional[TensorDict], out: Dict[str, Any], bs: int, num_starts: int) -> None:
    """Copy the best start back without changing the caller's batch dimensions."""
    if orig_td is not None:
        if num_starts > 1:
            best = out["reward"].argmax(dim=1)
            rows = torch.arange(bs, device=best.device)
            orig_td["solution"] = out["solution"][rows, best]
            orig_td["partner_ids"] = out["partner_ids"][rows, best]
        else:
            orig_td["solution"] = out["solution"]
            orig_td["partner_ids"] = out["partner_ids"]


class N2SPolicy(ImprovementPolicy):
    """N2S Policy for iterative routing improvement.

    Leverages a spatial-aware encoder and a move-selection decoder to
    iteratively refine a solution by exploring and selecting local
    neighborhood moves.

    Attributes:
        encoder (N2SEncoder): Transformer encoder for problem and solution state.
        decoder (N2SDecoder): Pairwise node-selection decoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initializes the N2S policy.

        Args:
            embed_dim: Dimensionality of the node features.
            num_heads: Number of attention heads.
            k_neighbors: Number of candidate neighbors per node.
            kwargs: Additional keyword arguments.
        """
        super().__init__(env_name="tsp_kopt", embed_dim=embed_dim)
        self.encoder = N2SEncoder(embed_dim, num_heads, k_neighbors, **kwargs)
        self.decoder = N2SDecoder(embed_dim, **kwargs)

    def _setup_initial_pdp_solution(
        self,
        td: TensorDict,
        caller_had_solution: bool,
        caller_solution: Any,
        has_explicit_pdp: bool,
        bs: int,
        n: int,
        device: torch.device,
        kwargs: dict[str, Any],
    ) -> None:
        """Sets up and validates the initial precedence-feasible PDP tour."""
        partner_ids = self.decoder._resolve_partner_ids(td, bs, n, device, kwargs)
        is_delivery, _ = self.decoder._identify_delivery_nodes(
            None, partner_ids, bs, n, device, td=td, validate_solution=False
        )
        node_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
        is_depot = (partner_ids == node_indices) | (node_indices == 0)
        is_pickup = ~is_delivery & ~is_depot

        td["partner_ids"] = partner_ids

        if caller_had_solution and caller_solution is not None:
            try:
                validate_pdp_solution(caller_solution, partner_ids, is_pickup, is_delivery)
            except ValueError:
                if has_explicit_pdp or kwargs.get("reject_infeasible", False):
                    raise
                td["solution"] = create_feasible_pdp_solution(bs, n, partner_ids, is_pickup, is_delivery, device)
            else:
                td["solution"] = caller_solution
        else:
            td["solution"] = create_feasible_pdp_solution(bs, n, partner_ids, is_pickup, is_delivery, device)

    def forward(
        self,
        td: TensorDict,
        env: Any = None,
        strategy: str = "greedy",
        num_starts: int = 1,
        max_steps: int | None = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Iteratively applies N2S refinement moves with evolving request-removal history (Eq. 13).

        Args:
            td: TensorDict containing problem instance and current solution.
            env: Refinement environment managing local search moves.
            strategy: Move selection strategy ("greedy", "sampling").
            num_starts: Number of parallel trajectories per instance.
            max_steps: Maximum number of improvement iterations.
            phase: Current phase ("train", "val", "test").
            return_actions: Whether to include move history in output.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Results including reward, log_likelihood, and actions.
        """
        from logic.src.envs import get_env
        from logic.src.utils.decoding import batchify, unbatchify

        # Track if caller supplied a solution explicitly in td
        caller_had_solution = td is not None and "solution" in td.keys()
        caller_solution = td["solution"].clone() if caller_had_solution else None
        has_explicit_pdp = td is not None and any(
            k in td.keys() for k in ("partner_ids", "partners", "delivery", "is_delivery", "is_pickup", "pickup_mask", "delivery_mask")
        )

        has_explicit_pdp = has_explicit_pdp or kwargs.get("partner_ids") is not None
        orig_td = td
        if env is None:
            env = get_env(self.env_name or "tsp_kopt")

        td = env.reset(td)

        bs = td.batch_size[0]
        device = td.device if hasattr(td, "device") and td.device is not None else "cpu"
        if "solution" in td.keys():
            n = td["solution"].shape[-1]
        elif "locs" in td.keys():
            n = td["locs"].shape[-2] + 1
        else:
            n = 21

        self._setup_initial_pdp_solution(
            td, caller_had_solution, caller_solution, has_explicit_pdp, bs, n, device, kwargs
        )

        if num_starts > 1:
            td = batchify(td, num_starts)

        log_probs = []
        actions = []

        if max_steps is None:
            max_steps_td = td.get("max_steps", None)
            max_steps = int(max_steps_td.item()) if max_steps_td is not None else 10

        assert isinstance(max_steps, int), f"max_steps must be an int, got {type(max_steps)}"

        # Initialize history [B, N, 4]: c(i), 1_last1, 1_last2, 1_last3 per Li et al. (2021) Eq. (13)
        history = torch.zeros(td.batch_size[0], n, 4, device=device)
        window_size = kwargs.get("window_size", 10)
        window_queue: list[tuple[torch.Tensor, torch.Tensor]] = []

        for _i in range(max_steps):
            if self.encoder is None:
                raise ValueError("Encoder must be provided for ImprovementPolicy")
            embeddings = self.encoder(td)

            if self.decoder is None:
                raise ValueError("Decoder must be provided for ImprovementPolicy")
            log_p, move = self.decoder(td, embeddings, env, strategy=strategy, history=history, **kwargs)

            # Execute paper-faithful request removal and precedence-correct reinsertion
            td["solution"] = execute_n2s_request_move(td["solution"], move)
            td.set("action", move)

            # Update history by request identity (both x_rem and x_pair)
            _update_n2s_history(history, move, window_queue, window_size)

            if "i" in td.keys():
                td["i"] = td["i"] + 1

            log_probs.append(log_p)
            actions.append(move)

            if td.get("done", torch.zeros(1)).all():
                break

        out: dict[str, Any] = {
            "reward": env.get_reward(td),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
            "solution": td["solution"],
            "partner_ids": td["partner_ids"],
        }

        if return_actions:
            out["actions"] = torch.stack(actions, dim=1)

        if num_starts > 1:
            out["reward"] = unbatchify(out["reward"], num_starts)
            out["log_likelihood"] = unbatchify(out["log_likelihood"], num_starts)
            out["solution"] = unbatchify(out["solution"], num_starts)
            out["partner_ids"] = unbatchify(out["partner_ids"], num_starts)
            if return_actions:
                out["actions"] = unbatchify(out["actions"], num_starts)

        _sync_n2s_caller(orig_td, out, bs, num_starts)

        return out
