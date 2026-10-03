"""N2S Decoder implementation.

This module implements the paper-faithful `N2SDecoder` (Li et al., IJCAI 2021),
which factorizes neighborhood search moves into:
1. Node-Pair / Request Removal Decoder (§4.2, Eq. 12-13):
   - Computes tour-closeness scores lambda_i (Eq. 12) measuring connection strength to tour neighbors.
   - Evaluates removal distribution via MLP_lambda (Eq. 13, 2m + 4 -> 32 -> 32 -> 1) taking
     multi-head closeness and historical frequency/recency features.
   - Samples the node / request to remove with log P_remove.
2. Conditional Reinsertion Decoder (§4.2, Eq. 14-15):
   - Computes directional preference matrices mu_p and mu_s (Eq. 14).
   - Evaluates conditional reinsertion position distribution via MLP_mu (Eq. 15, 4m -> 32 -> 32 -> 1)
     conditioned on the removed node and its tour partner.
   - Enforces precedence and validity constraints (masking immediate predecessor and identity).
   - Samples reinsertion target with log P_reinsert|remove.
3. Joint action [x_rem, x_target] and factorized log-likelihood log P_rem + log P_reinsert.

Attributes:
    N2SDecoder: Paper-faithful removal and conditional reinsertion decoder.

Example:
    >>> decoder = N2SDecoder(embed_dim=128, num_heads=4)
    >>> log_p, actions = decoder(td, h, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class N2SDecoder(ImprovementDecoder):
    """N2S Decoder with Request-Removal and Conditional Reinsertion (Li et al. IJCAI 2021, §4.2).

    Conditioned on the incumbent tour permutation, it explicitly factorizes the neighborhood move:
    1. Evaluates candidate request/node removal using neighbor closeness and history via MLP_lambda.
    2. Conditions joint reinsertion positions on the selected removal request via MLP_mu with
       precedence and topological validity masking.

    Attributes:
        embed_dim (int): Dimensionality of node embeddings.
        num_heads (int): Count of attention heads (default: 4 per paper).
        tanh_clipping (float): Entropy control constant C (default: 6.0).
        seed (int): RNG seed for sampling.
        generator (torch.Generator): PRNG instance for reproducible sampling.
        W_h_local (nn.Linear): Local projection for max-pooling layer (Eq. 11).
        W_h_global (nn.Linear): Global projection for max-pooling layer (Eq. 11).
        project_q_lambda (nn.Linear): Query projection for neighbor closeness (Eq. 12).
        project_k_lambda (nn.Linear): Key projection for neighbor closeness (Eq. 12).
        mlp_lambda (nn.Sequential): 3-layer MLP for removal distribution (Eq. 13, 2m + 4 -> 32 -> 32 -> 1).
        project_qp (nn.Linear): Predecessor query projection (Eq. 14).
        project_kp (nn.Linear): Predecessor key projection (Eq. 14).
        project_qs (nn.Linear): Successor query projection (Eq. 14).
        project_ks (nn.Linear): Successor key projection (Eq. 14).
        mlp_mu (nn.Sequential): 4-layer MLP for conditional reinsertion (Eq. 15, 4m -> 32 -> 32 -> 1).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        tanh_clipping: float = 6.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the paper-faithful N2S decoder.

        Args:
            embed_dim: Dimensionality of the node features.
            num_heads: Number of attention heads (paper default: 4).
            tanh_clipping: Clipping factor C for entropy control (paper default: 6.0).
            seed: Random seed for action sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            divisors = [d for d in range(1, num_heads + 1) if embed_dim % d == 0]
            num_heads = max(divisors) if divisors else 1
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.seed = seed
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim**0.5)

        # Max-pooling sub-layer (Eq. 11)
        self.W_h_local = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_h_global = nn.Linear(embed_dim, embed_dim, bias=False)

        # Closeness scoring projections (Eq. 12)
        self.project_q_lambda = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_k_lambda = nn.Linear(embed_dim, embed_dim, bias=False)

        # Node-Pair Removal MLP_lambda (Eq. 13): (2m + 4 -> 32 -> 32 -> 1)
        mlp_lambda_in = 2 * num_heads + 4
        self.mlp_lambda = nn.Sequential(
            nn.Linear(mlp_lambda_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Directional insertion preference projections (Eq. 14)
        self.project_qp = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_kp = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_qs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_ks = nn.Linear(embed_dim, embed_dim, bias=False)

        # Conditional Reinsertion MLP_mu (Eq. 15): (4m -> 32 -> 32 -> 1)
        mlp_mu_in = 4 * num_heads
        self.mlp_mu = nn.Sequential(
            nn.Linear(mlp_mu_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @property
    def device(self) -> torch.device:
        """Determines the current hardware placement of parameters.

        Returns:
            torch.device: The active device (CPU/CUDA).
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes the state, handling non-picklable components.

        Returns:
            Dict[str, Any]: Attribute map with generator state recorded.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores policy state including the RNG generator.

        Args:
            state: Serialized dictionary of attributes.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Validates checkpoint keys against legacy generic decoder weights."""
        legacy_keys = {prefix + "project_q.weight", prefix + "project_k.weight"}
        present_legacy = [k for k in legacy_keys if k in state_dict]
        has_new_keys = any(k.startswith(prefix + "W_h_") or k.startswith(prefix + "mlp_lambda") for k in state_dict)
        if present_legacy and not has_new_keys:
            raise RuntimeError(
                f"Detected legacy generic pairwise decoder checkpoint with keys {present_legacy} "
                f"for {self.__class__.__name__}. N2SDecoder has been upgraded to the paper-faithful "
                f"tour-aware request-removal and conditional reinsertion architecture (Li et al. IJCAI 2021). "
                f"Legacy checkpoints cannot be loaded silently and must be retrained."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

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
            sol = td["solution"]  # [B, N]
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
            p = td["partner_ids"]
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(bs, -1)
            return self._validate_partner_ids(p, bs, n, device)
        if "partners" in td.keys():
            p = td["partners"]
            if p.dim() == 1:
                p = p.unsqueeze(0).expand(bs, -1)
            return self._validate_partner_ids(p, bs, n, device)
        if "delivery" in td.keys():
            p = td["delivery"]
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

        if td is not None and ("is_delivery" in td.keys() or "delivery_mask" in td.keys()):
            deliv_key = "is_delivery" if "is_delivery" in td.keys() else "delivery_mask"
            is_delivery = td[deliv_key].to(dtype=torch.bool, device=device)
            if is_delivery.dim() == 1:
                is_delivery = is_delivery.unsqueeze(0).expand(bs, -1)
        elif td is not None and ("is_pickup" in td.keys() or "pickup_mask" in td.keys()):
            pick_key = "is_pickup" if "is_pickup" in td.keys() else "pickup_mask"
            is_pickup = td[pick_key].to(dtype=torch.bool, device=device)
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

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts an improvement move [x_rem, x_target] via paper factorized removal and reinsertion.

        Args:
            td: TensorDict containing problem and solution state.
            embeddings: Encoded node features [B, N, D].
            env: Environment managing the problem physics.
            kwargs: Additional keyword arguments including 'strategy' and 'history'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Joint log-likelihood of the selected move [B].
                - actions: Pair of node indices [x_rem, x_target] [B, 2].
        """
        h = embeddings[0] if isinstance(embeddings, (tuple, list)) else embeddings
        bs, n, d = h.shape
        m = self.num_heads
        dk = self.head_dim
        strategy = kwargs.get("strategy", "greedy")

        # 1. Max-pooling context augmentation (Eq. 11)
        h_max = h.max(dim=1, keepdim=True).values  # [B, 1, d]
        h_hat = self.W_h_local(h) + self.W_h_global(h_max)  # [B, N, d]

        # 2. Extract predecessor and successor tour context
        pred_idx, succ_idx = self._get_tour_neighbors(td, bs, n, h.device)
        h_pred = h_hat.gather(1, pred_idx.unsqueeze(-1).expand(-1, -1, d))
        h_succ = h_hat.gather(1, succ_idx.unsqueeze(-1).expand(-1, -1, d))

        # 3. Candidate node closeness scoring lambda_i (Eq. 12)
        q_pred = self.project_q_lambda(h_pred).view(bs, n, m, dk)
        k_curr = self.project_k_lambda(h_hat).view(bs, n, m, dk)
        q_curr = self.project_q_lambda(h_hat).view(bs, n, m, dk)
        k_succ = self.project_k_lambda(h_succ).view(bs, n, m, dk)

        term1 = (q_pred * k_curr).sum(dim=-1) * self.scale
        term2 = (q_curr * k_succ).sum(dim=-1) * self.scale
        term3 = (q_pred * k_succ).sum(dim=-1) * self.scale
        lambda_scores = term1 + term2 - term3  # [B, N, m]

        # --- Stage 1: Request / Node Removal Decoder (Eq. 13) ---
        partner_ids = self._resolve_partner_ids(td, bs, n, h.device, kwargs)
        lambda_pair = lambda_scores.gather(1, partner_ids.unsqueeze(-1).expand(-1, -1, m))

        # History features: c(i), 1_last1, 1_last2, 1_last3 (Eq. 13)
        history = kwargs.get("history")
        if history is not None and isinstance(history, torch.Tensor) and history.shape == (bs, n, 4):
            hist_feat = history
        else:
            hist_feat = torch.zeros(bs, n, 4, device=h.device)

        removal_features = torch.cat([lambda_scores, lambda_pair, hist_feat], dim=-1)  # [B, N, 2m + 4]
        lambda_tilde = self.mlp_lambda(removal_features).squeeze(-1)  # [B, N]
        logits_remove = self.tanh_clipping * torch.tanh(lambda_tilde)

        device = h.device
        sol_tensor = td.get("solution", None)
        validate_sol = kwargs.get("validate_solution", False)
        is_delivery, sol_pos = self._identify_delivery_nodes(
            sol_tensor, partner_ids, bs, n, device, td=td, validate_solution=validate_sol
        )

        node_indices = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
        is_depot = (partner_ids == node_indices) | (node_indices == 0)
        is_pickup = ~is_delivery & ~is_depot

        # In standard PDP, depot is not a request; delivery nodes are also masked from removal
        # so the removal distribution is strictly over requests (indexed by their pickup node)
        if n > 2:
            logits_remove[:, 0] = float("-inf")
            logits_remove[is_delivery] = float("-inf")
            all_masked = torch.isneginf(logits_remove).all(dim=-1)
            if all_masked.any():
                logits_remove[all_masked, 1:] = 0.0

        if strategy == "greedy":
            x_rem = logits_remove.argmax(dim=-1)  # [B]
        else:
            probs_rem = F.softmax(logits_remove, dim=-1)
            x_rem = torch.multinomial(probs_rem, 1, generator=self.generator).squeeze(-1)

        log_p_rem = F.log_softmax(logits_remove, dim=-1).gather(1, x_rem.unsqueeze(-1)).squeeze(-1)
        x_pair = partner_ids.gather(1, x_rem.unsqueeze(-1)).squeeze(-1)  # [B]

        # Determine canonical i_plus (pickup) and i_minus (delivery) for the chosen request
        rem_is_pickup = is_pickup.gather(1, x_rem.unsqueeze(-1)).squeeze(-1)
        i_plus = torch.where(rem_is_pickup, x_rem, x_pair)
        i_minus = torch.where(rem_is_pickup, x_pair, x_rem)

        # --- Stage 2: Conditional Reinsertion Decoder (Eq. 14, 15) ---
        # Note that here pred(·) and succ(·) should be considered in the new solution
        # where nodes i+, i- have already been removed (§4.2, Eq. 15).
        # Compute residual tour successors and positions after removing i_plus and i_minus
        if sol_tensor is None or sol_tensor.size(1) != n:
            sol_tensor = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)

        succ_res = torch.zeros(bs, n, dtype=torch.long, device=device)
        pos_res = torch.zeros(bs, n, dtype=torch.long, device=device)

        for b in range(bs):
            sb = sol_tensor[b].tolist()
            rem_b = {i_plus[b].item(), i_minus[b].item()}
            res_b = [x for x in sb if x not in rem_b]
            len_res = len(res_b)
            if len_res > 0:
                for p, node in enumerate(res_b):
                    succ_res[b, node] = res_b[(p + 1) % len_res]
                    pos_res[b, node] = p

        # Predecessor and successor preference matrices across heads (Eq. 14)
        qp = self.project_qp(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        kp = self.project_kp(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        mu_p = torch.matmul(qp, kp.transpose(-2, -1)) * self.scale  # [B, m, N, N]

        qs = self.project_qs(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        ks = self.project_ks(h_hat).view(bs, n, m, dk).permute(0, 2, 1, 3)
        mu_s = torch.matmul(qs, ks.transpose(-2, -1)) * self.scale  # [B, m, N, N]

        # Condition on canonical pickup i_plus and delivery i_minus
        i_plus_exp = i_plus.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, m, 1, n)
        i_minus_exp = i_minus.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, m, 1, n)

        # In residual tour: succ_res gives the node that will follow the newly inserted node!
        # mu_p[i_plus, succ_res(j)] (gathered at succ_res(j)) and mu_s[i_plus, j] (at j)
        mu_p_plus_raw = mu_p.gather(2, i_plus_exp).squeeze(2).permute(0, 2, 1)  # [B, N, m]
        mu_p_plus = mu_p_plus_raw.gather(1, succ_res.unsqueeze(-1).expand(-1, -1, m))  # [B, N, m] at succ_res(j)
        mu_s_plus = mu_s.gather(2, i_plus_exp).squeeze(2).permute(0, 2, 1)  # [B, N, m] at j

        # mu_p[i_minus, succ_res(k)] (at succ_res(k)) and mu_s[i_minus, k] (at k)
        mu_p_minus_raw = mu_p.gather(2, i_minus_exp).squeeze(2).permute(0, 2, 1)  # [B, N, m]
        mu_p_minus = mu_p_minus_raw.gather(1, succ_res.unsqueeze(-1).expand(-1, -1, m))  # [B, N, m] at succ_res(k)
        mu_s_minus = mu_s.gather(2, i_minus_exp).squeeze(2).permute(0, 2, 1)  # [B, N, m] at k

        # Assemble 2m features for candidate pickup position j and candidate delivery position k
        f_j = torch.cat([mu_p_plus, mu_s_plus], dim=-1)  # [B, N, 2m]
        f_k = torch.cat([mu_p_minus, mu_s_minus], dim=-1)  # [B, N, 2m]

        # Evaluate MLP_mu (4m -> 32 -> 32 -> 1) over all candidate position pairs (j, k) per Eq. (15)
        # Using factored first-layer linear projection for efficiency
        w1 = self.mlp_mu[0].weight[:, : 2 * m]
        w2 = self.mlp_mu[0].weight[:, 2 * m :]
        b = self.mlp_mu[0].bias

        h1_j = F.linear(f_j, w1)  # [B, N, 32]
        h1_k = F.linear(f_k, w2)  # [B, N, 32]
        h1 = F.relu(h1_j.unsqueeze(2) + h1_k.unsqueeze(1) + b)  # [B, N, N, 32]
        mu_tilde = self.mlp_mu[4](F.relu(self.mlp_mu[2](h1))).squeeze(-1)  # [B, N, N]
        logits_reinsert = self.tanh_clipping * torch.tanh(mu_tilde)  # [B, N, N]

        # Mask invalid target positions per Eq. (15):
        # 1. Neither j nor k can be the removed nodes (x_rem, x_pair)
        # 2. Precedence constraint in residual tour: pos_res(j) <= pos_res(k)
        mask_rem_nodes = torch.zeros(bs, n, device=device, dtype=torch.bool)
        mask_rem_nodes.scatter_(1, x_rem.unsqueeze(-1), True)
        mask_rem_nodes.scatter_(1, x_pair.unsqueeze(-1), True)

        mask_j = mask_rem_nodes.unsqueeze(2).expand(-1, -1, n)
        mask_k = mask_rem_nodes.unsqueeze(1).expand(-1, n, -1)
        invalid_mask = mask_j | mask_k

        # Tour-order precedence constraint: pos_res(j) <= pos_res(k)
        pos_j = pos_res.unsqueeze(2).expand(-1, -1, n)
        pos_k = pos_res.unsqueeze(1).expand(-1, n, -1)
        precedence_violation = pos_j > pos_k
        invalid_mask = invalid_mask | precedence_violation

        logits_reinsert = logits_reinsert.masked_fill(invalid_mask, float("-inf"))

        # Flatten for joint categorical distribution over candidate pairs (j, k)
        logits_flat = logits_reinsert.view(bs, n * n)
        all_reinsert_masked = torch.isneginf(logits_flat).all(dim=-1)
        if all_reinsert_masked.any():
            logits_flat[all_reinsert_masked, 0] = 0.0

        if strategy == "greedy":
            pair_idx = logits_flat.argmax(dim=-1)  # [B]
        else:
            probs_reinsert = F.softmax(logits_flat, dim=-1)
            pair_idx = torch.multinomial(probs_reinsert, 1, generator=self.generator).squeeze(-1)

        log_p_reinsert = F.log_softmax(logits_flat, dim=-1).gather(1, pair_idx.unsqueeze(-1)).squeeze(-1)
        j_sel = pair_idx // n
        k_sel = pair_idx % n

        # Assemble action [i_plus, i_minus, j_sel, k_sel] and joint factorized log probability
        actions = torch.stack([i_plus, i_minus, j_sel, k_sel], dim=-1)
        log_p = log_p_rem + log_p_reinsert

        return log_p, actions
