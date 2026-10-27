"""Forward pass of N2SDecoder.

Moved from ``logic.src.models.core.n2s.decoder``.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.envs.base.base import RL4COEnvBase


class N2SDecoderForwardMixin:
    """Moved methods of :class:`N2SDecoder`."""

    num_heads: int
    head_dim: int
    scale: float
    tanh_clipping: float
    W_h_local: Any
    W_h_global: Any
    project_q_lambda: Any
    project_k_lambda: Any
    project_qp: Any
    project_kp: Any
    project_qs: Any
    project_ks: Any
    mlp_lambda: Any
    mlp_mu: Any
    generator: Any
    _get_tour_neighbors: Callable[..., Any]
    _resolve_partner_ids: Callable[..., Any]
    _identify_delivery_nodes: Callable[..., Any]

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts a request move [i_plus, i_minus, j, k] via removal and reinsertion.

        Args:
            td: TensorDict containing problem and solution state.
            embeddings: Encoded node features [B, N, D].
            env: Environment managing the problem physics.
            kwargs: Additional keyword arguments including 'strategy' and 'history'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Joint log-likelihood of the selected move [B].
                - actions: Node IDs [i_plus, i_minus, j, k] [B, 4]. The pickup i_plus
                  and delivery i_minus are inserted after residual-tour anchors j and k;
                  when j == k, the pickup is inserted before the delivery.
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
