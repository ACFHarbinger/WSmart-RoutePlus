"""Tour-partner helpers for N2SDecoder.

Moved from ``logic.src.models.core.n2s.decoder``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import torch
from tensordict import TensorDict


class N2SDecoderPartnersMixin:
    """Moved methods of :class:`N2SDecoder`."""

    def _get_tour_neighbors(
        self,
        td: TensorDict,
        bs: int,
        n: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolves predecessor and successor indices for each node in current solution.

        Args:
            td: Environment state TensorDict.
            bs: Batch size.
            n: Number of nodes.
            device: Computation device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (pred_idx [B, N], succ_idx [B, N]).
        """
        if "solution" in td.keys():
            sol = cast(torch.Tensor, td["solution"])  # [B, N] TensorDict values are a tensor union
            if sol.size(1) == n:
                prev_nodes = sol.roll(shifts=1, dims=1)
                next_nodes = sol.roll(shifts=-1, dims=1)
                pred_idx = torch.empty(bs, n, dtype=torch.long, device=device)
                succ_idx = torch.empty(bs, n, dtype=torch.long, device=device)
                pred_idx.scatter_(1, sol, prev_nodes)
                succ_idx.scatter_(1, sol, next_nodes)
                return pred_idx, succ_idx

        base_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
        pred_idx = (base_indices - 1) % n
        succ_idx = (base_indices + 1) % n
        return pred_idx, succ_idx

    @staticmethod
    def _validate_partner_ids(p: torch.Tensor, bs: int, n: int, device: torch.device) -> torch.Tensor:
        """Validate a symmetric request map with depot 0 and paired customers."""
        if p.is_floating_point() or p.is_complex() or p.dtype == torch.bool:
            raise ValueError("PDP partner IDs must be integers")
        if p.dim() == 1:
            p = p.unsqueeze(0).expand(bs, -1)
        if p.shape != (bs, n) or n < 3:
            raise ValueError("PDP partner map must have shape [batch, nodes] with at least one request")
        p = p.to(device=device, dtype=torch.long)
        if bool(((p < 0) | (p >= n)).any()):
            raise ValueError("PDP partner IDs must be in node range")
        nodes = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
        if bool((p[:, 0] != 0).any()) or bool((p[:, 1:] == nodes[:, 1:]).any()):
            raise ValueError("Only depot 0 may be self-paired, and it must be self-paired")
        if not torch.equal(p.gather(1, p), nodes):
            raise ValueError("PDP partner map must contain symmetric disjoint pairs")
        return p

    def _resolve_partner_ids(
        self,
        td: TensorDict,
        bs: int,
        n: int,
        device: torch.device,
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Resolves request partner mapping across requests in PDP or TSP."""
        if "partner_ids" in td.keys():
            p = cast(torch.Tensor, td["partner_ids"])  # TensorDict values are a tensor union
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(bs, -1)
            return self._validate_partner_ids(p, bs, n, device)
        if "partners" in td.keys():
            p = cast(torch.Tensor, td["partners"])  # TensorDict values are a tensor union
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(bs, -1)
            return self._validate_partner_ids(p, bs, n, device)
        if "delivery" in td.keys():
            p = cast(torch.Tensor, td["delivery"])  # TensorDict values are a tensor union
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(bs, -1)
            return self._validate_partner_ids(p, bs, n, device)
        if kwargs.get("partner_ids") is not None:
            p = kwargs["partner_ids"]
            if isinstance(p, torch.Tensor):
                if p.dim() == 1:
                    p = p.unsqueeze(0).expand(bs, -1)
                return self._validate_partner_ids(p, bs, n, device)

        partner_ids = torch.empty(bs, n, dtype=torch.long, device=device)
        if n > 1 and n % 2 == 1:
            half = (n - 1) // 2
            p = torch.arange(n, device=device)
            p[0] = 0
            p[1 : half + 1] = p[1 : half + 1] + half
            p[half + 1 :] = p[half + 1 :] - half
            partner_ids[:] = p
        else:
            half = max(1, n // 2)
            p = torch.arange(n, device=device)
            p[:half] = (p[:half] + half) % n
            p[half:] = (p[half:] - half) % n
            partner_ids[:] = p
        return self._validate_partner_ids(partner_ids, bs, n, device)

    def _identify_delivery_nodes(
        self,
        sol_tensor: Optional[torch.Tensor],
        partner_ids: torch.Tensor,
        bs: int,
        n: int,
        device: torch.device,
        td: Optional[TensorDict] = None,
        validate_solution: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Identifies delivery nodes based on immutable request schema and validates solution precedence.

        In PDP (Li et al., 2021, §3.1 Footnote 3), requests are immutable: each request
        is a pair (i+, i-) where i+ is the pickup and i- is the delivery. This identity
        is an intrinsic property of the problem instance and never changes with tour order.
        """
        node_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
        is_depot = (partner_ids == node_indices) | (node_indices == 0)

        state = cast(Any, td)  # TensorDict values are a tensor union
        if td is not None and ("is_delivery" in td.keys() or "delivery_mask" in td.keys()):
            deliv_key = "is_delivery" if "is_delivery" in td.keys() else "delivery_mask"
            is_delivery = cast(torch.Tensor, state[deliv_key]).to(dtype=torch.bool, device=device)
            if is_delivery.dim() == 1:
                is_delivery = is_delivery.unsqueeze(0).expand(bs, -1)
        elif td is not None and ("is_pickup" in td.keys() or "pickup_mask" in td.keys()):
            pick_key = "is_pickup" if "is_pickup" in td.keys() else "pickup_mask"
            is_pickup = cast(torch.Tensor, state[pick_key]).to(dtype=torch.bool, device=device)
            if is_pickup.dim() == 1:
                is_pickup = is_pickup.unsqueeze(0).expand(bs, -1)
            is_delivery = ~is_pickup & ~is_depot
        else:
            # Canonical PDP numbering (Li et al. 2021, §3.1 Footnote 3):
            # min(u, partner(u)) is pickup, max(u, partner(u)) is delivery
            is_delivery = (partner_ids < node_indices) & ~is_depot

        if is_delivery.shape != (bs, n) or bool(is_delivery[:, 0].any()):
            raise ValueError("PDP delivery roles must have shape [batch, nodes] and exclude depot")
        if bool((is_delivery[:, 1:] == is_delivery.gather(1, partner_ids)[:, 1:]).any()):
            raise ValueError("Each PDP request must have exactly one pickup and one delivery")

        sol_pos = None
        if sol_tensor is not None and sol_tensor.size(1) == n:
            sol_pos = torch.zeros(bs, n, dtype=torch.long, device=device)
            positions = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
            sol_pos.scatter_(1, sol_tensor, positions)

            if validate_solution:
                is_pickup = ~is_delivery & ~is_depot
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

        return is_delivery, sol_pos
