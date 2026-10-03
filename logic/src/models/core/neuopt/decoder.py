"""NeuOpt Decoder implementation.

This module implements the paper-faithful `NeuOptDecoder` (Ma et al., NeurIPS 2023),
which parameterizes a sequential basis-move factorized search move using a Recurrent
Dual-Stream (RDS) architecture.

Architecture (§4.1-4.2, Eq. 1, 2, 3):
- Recurrent Dual-Stream (RDS) factorization:
  - Move stream (mu): models historical move decisions.
  - Edge stream (lambda): models edge proposals.
- Hidden states initialized via mean-pooled graph embedding: q0 = mean(h).
- First step inputs driven by learnable vectors o_hat, o_tilde.
- Second step inputs:
  - Move stream: o2_mu = h[x1] (last selected move / anchor node).
  - Edge stream: o2_lambda = h[x1] (introduced-edge lower-ranked endpoint = xa at step 2).
- Dual-stream additive + multiplicative (Hadamard) attention with distinct primed and
  unprimed projections (Eq. 3):
  score_mu = tanh((q W^Q + h W^K) + (q W'^Q) * (h W'^K)) W^O
  score_lambda = tanh((q W^Q + h W^K) + (q W'^Q) * (h W'^K)) W^O
- Cyclic-rank feasibility constraint (Gamma[a, j] <= Gamma[a, v], §4.1-4.2):
  Enforces sequential basis moves (S-move, I-move, E-move) and prevents invalid moves.

Attributes:
    NeuOptDecoder: Recurrent Dual-Stream action decoder for k-opt moves.

Example:
    >>> decoder = NeuOptDecoder(embed_dim=128)
    >>> log_p, actions = decoder(td, embeddings, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class NeuOptDecoder(ImprovementDecoder):
    """NeuOpt Recurrent Dual-Stream (RDS) Decoder (Ma et al. NeurIPS 2023, §4.1-4.2).

    Factorizes local search moves into sequential basis moves (S-move, I-move, E-move)
    conditioned on dual recurrent streams (move stream mu and edge stream lambda) with
    distinct primed/unprimed additive and Hadamard attention scoring.

    Attributes:
        embed_dim (int): Dimensionality of latent embeddings.
        tanh_clipping (float): Entropy control constant C (default: 6.0).
        seed (int): RNG seed for sampling.
        generator (torch.Generator): PRNG instance for reproducible sampling.
        gru_mu (nn.GRUCell): Recurrent cell for move stream mu.
        gru_lambda (nn.GRUCell): Recurrent cell for edge stream lambda.
        init_o_mu (nn.Parameter): Learnable initial input vector o_hat for stream mu.
        init_o_lambda (nn.Parameter): Learnable initial input vector o_tilde for stream lambda.
        W_mu_q (nn.Linear): Unprimed query projection for stream mu.
        W_mu_k (nn.Linear): Unprimed key projection for stream mu.
        W_mu_q_prime (nn.Linear): Primed query projection for stream mu (Hadamard).
        W_mu_k_prime (nn.Linear): Primed key projection for stream mu (Hadamard).
        W_mu_o (nn.Linear): Output scoring projection for stream mu.
        W_lambda_q (nn.Linear): Unprimed query projection for stream lambda.
        W_lambda_k (nn.Linear): Unprimed key projection for stream lambda.
        W_lambda_q_prime (nn.Linear): Primed query projection for stream lambda (Hadamard).
        W_lambda_k_prime (nn.Linear): Primed key projection for stream lambda (Hadamard).
        W_lambda_o (nn.Linear): Output scoring projection for stream lambda.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        tanh_clipping: float = 6.0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes the paper-faithful NeuOpt RDS decoder.

        Args:
            embed_dim: Feature dimension of the input embeddings.
            tanh_clipping: Clipping factor C for entropy control (paper default: 6.0).
            seed: Random seed for move sampling.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.tanh_clipping = tanh_clipping
        self.seed = kwargs.get("seed", seed)
        self.k_basis = kwargs.get("k_basis", 2)
        init_device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=init_device).manual_seed(self.seed)

        # Recurrent dual streams (Eq. 2)
        self.gru_mu = nn.GRUCell(embed_dim, embed_dim)
        self.gru_lambda = nn.GRUCell(embed_dim, embed_dim)

        # Learnable initial inputs o_hat and o_tilde (Eq. 2 context)
        self.init_o_mu = nn.Parameter(torch.empty(embed_dim))
        self.init_o_lambda = nn.Parameter(torch.empty(embed_dim))
        nn.init.uniform_(self.init_o_mu, -0.1, 0.1)
        nn.init.uniform_(self.init_o_lambda, -0.1, 0.1)

        # Move stream mu: distinct primed and unprimed attention projections (Eq. 3)
        self.W_mu_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_mu_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_mu_q_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_mu_k_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_mu_o = nn.Linear(embed_dim, 1, bias=False)

        # Edge stream lambda: distinct primed and unprimed attention projections (Eq. 3)
        self.W_lambda_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_lambda_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_lambda_q_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_lambda_k_prime = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_lambda_o = nn.Linear(embed_dim, 1, bias=False)

    @property
    def device(self) -> torch.device:
        """Determines the current execution device.

        Returns:
            torch.device: Current placement of solver parameters.
        """
        return next(self.parameters()).device

    def __getstate__(self) -> Dict[str, Any]:
        """Serializes current state for persistence.

        Returns:
            Dict[str, Any]: Attribute map with generator state extracted.
        """
        state = self.__dict__.copy()
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores policy parameters and RNG state.

        Args:
            state: Serialized dictionary of solver attributes.
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
        has_new_keys = any(k.startswith(prefix + "gru_") or k.startswith(prefix + "W_mu_") for k in state_dict)
        if present_legacy and not has_new_keys:
            raise RuntimeError(
                f"Detected legacy generic pairwise decoder checkpoint with keys {present_legacy} "
                f"for {self.__class__.__name__}. NeuOptDecoder has been upgraded to the paper-faithful "
                f"Recurrent Dual-Stream (RDS) architecture (Ma et al. NeurIPS 2023). "
                f"Legacy checkpoints cannot be loaded silently and must be retrained."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _compute_stream_scores(
        self,
        q_mu: torch.Tensor,
        q_lambda: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes additive + Hadamard attention scores per Eq. (3).

        Args:
            q_mu: Hidden state of move stream [B, d].
            q_lambda: Hidden state of edge stream [B, d].
            h: Node embeddings [B, N, d].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (score_mu [B, N], score_lambda [B, N]).
        """
        # Stream mu with distinct primed and unprimed projections (Eq. 3)
        q_mu_add = self.W_mu_q(q_mu).unsqueeze(1)  # [B, 1, d]
        h_mu_add = self.W_mu_k(h)  # [B, N, d]
        q_mu_had = self.W_mu_q_prime(q_mu).unsqueeze(1)  # [B, 1, d]
        h_mu_had = self.W_mu_k_prime(h)  # [B, N, d]
        mu_feature = torch.tanh((q_mu_add + h_mu_add) + (q_mu_had * h_mu_had))  # [B, N, d]
        score_mu = self.W_mu_o(mu_feature).squeeze(-1)  # [B, N]

        # Stream lambda with distinct primed and unprimed projections (Eq. 3)
        q_lambda_add = self.W_lambda_q(q_lambda).unsqueeze(1)  # [B, 1, d]
        h_lambda_add = self.W_lambda_k(h)  # [B, N, d]
        q_lambda_had = self.W_lambda_q_prime(q_lambda).unsqueeze(1)  # [B, 1, d]
        h_lambda_had = self.W_lambda_k_prime(h)  # [B, N, d]
        lambda_feature = torch.tanh((q_lambda_add + h_lambda_add) + (q_lambda_had * h_lambda_had))  # [B, N, d]
        score_lambda = self.W_lambda_o(lambda_feature).squeeze(-1)  # [B, N]

        return score_mu, score_lambda

    def _get_tour_successor(self, td: TensorDict, x1: torch.Tensor, bs: int, n: int) -> torch.Tensor:
        """Finds the tour successor node x_b = succ(x1) for the removed edge e_out(xa -> xb).

        Args:
            td: TensorDict containing 'solution' [B, N].
            x1: Anchor node indices [B].
            bs: Batch size.
            n: Number of nodes.

        Returns:
            torch.Tensor: Successor node indices [B].
        """
        if "solution" in td.keys():
            sol = td["solution"]
            if sol.size(1) == n:
                # Find position of x1 in sol
                sol_pos = torch.zeros(bs, n, dtype=torch.long, device=x1.device)
                positions = torch.arange(n, device=x1.device).unsqueeze(0).expand(bs, -1)
                sol_pos.scatter_(1, sol, positions)
                pos_x1 = sol_pos.gather(1, x1.unsqueeze(-1)).squeeze(-1)  # [B]
                pos_succ = (pos_x1 + 1) % n
                xb = sol.gather(1, pos_succ.unsqueeze(-1)).squeeze(-1)
                return xb
        return (x1 + 1) % n

    def _compute_cyclic_rank_mask(
        self,
        td: TensorDict,
        x1: torch.Tensor,
        bs: int,
        n: int,
        xj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes feasibility mask based on cyclic node rank Gamma[a, v] >= Gamma[a, xj] (§4.1-4.2).

        Args:
            td: TensorDict containing problem state.
            x1: Selected anchor node xa [B].
            bs: Batch size.
            n: Number of nodes.
            xj: Upper-ranked endpoint xj of current Hamiltonian path [B]. Defaults to succ(x1).

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates invalid nodes (rank < rank(xj)).
        """
        device = x1.device
        if xj is None:
            xj = self._get_tour_successor(td, x1, bs, n)

        if "solution" in td.keys():
            sol = td["solution"]
            if sol.size(1) == n:
                sol_pos = torch.zeros(bs, n, dtype=torch.long, device=device)
                positions = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
                sol_pos.scatter_(1, sol, positions)
                pos_a = sol_pos.gather(1, x1.unsqueeze(-1)).squeeze(-1)  # [B]
                pos_j = sol_pos.gather(1, xj.unsqueeze(-1)).squeeze(-1)  # [B]

                # Node rank w.r.t. anchor: Gamma[a, u] = (pos_u - pos_a) % n
                ranks = (sol_pos - pos_a.unsqueeze(-1)) % n  # [B, N]
                rank_j = ((pos_j - pos_a) % n).unsqueeze(-1)  # [B, 1]

                # Valid choices require Gamma[a, v] >= Gamma[a, xj]
                return ranks < rank_j

        # Fallback mask: prevent selecting self (x2 == x1)
        mask = torch.zeros(bs, n, device=device, dtype=torch.bool)
        mask.scatter_(1, x1.unsqueeze(-1), True)
        return mask

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts an improvement move via K-step RDS autoregressive basis-move decoding (§4.1-4.2).

        Args:
            td: TensorDict containing problem and current solution state.
            embeddings: Node embeddings from encoder [B, N, d].
            env: The environment defining move validity and reward.
            kwargs: Additional keyword arguments including 'strategy' and 'k_basis'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_p: Joint log-likelihood of the selected move [B].
                - actions: Indices of basis-move nodes [B, K].
        """
        h = embeddings[0] if isinstance(embeddings, (tuple, list)) else embeddings
        bs, n, d = h.shape
        strategy = kwargs.get("strategy", "greedy")
        K = kwargs.get("k_basis", self.k_basis)

        # 1. Initialize hidden states with mean-pooled graph embedding (Eq. 2)
        q0 = h.mean(dim=1)  # [B, d]
        q_mu = q0
        q_lambda = q0

        # --- Step kappa = 1 (S-move: select anchor node xa = x1) ---
        o1_mu = self.init_o_mu.unsqueeze(0).expand(bs, -1)
        o1_lambda = self.init_o_lambda.unsqueeze(0).expand(bs, -1)

        q_mu = self.gru_mu(o1_mu, q_mu)
        q_lambda = self.gru_lambda(o1_lambda, q_lambda)

        score_mu_1, score_lambda_1 = self._compute_stream_scores(q_mu, q_lambda, h)
        logits_1 = self.tanh_clipping * torch.tanh(score_mu_1 + score_lambda_1)  # [B, N]

        if strategy == "greedy":
            x1 = logits_1.argmax(dim=-1)
        else:
            probs_1 = F.softmax(logits_1, dim=-1)
            x1 = torch.multinomial(probs_1, 1, generator=self.generator).squeeze(-1)

        log_p1 = F.log_softmax(logits_1, dim=-1).gather(1, x1.unsqueeze(-1)).squeeze(-1)

        actions_list = [x1]
        log_p_list = [log_p1]

        # Initial Hamiltonian path endpoints after S-move S(xa):
        # xi = xa (rank 0), xj = xb = succ(xa) (rank 1)
        xi = x1
        xj = self._get_tour_successor(td, x1, bs, n)
        prev_x = x1

        # Keep the same evolving open path as the basis-sequence executor.
        open_paths = []
        node_ranks = []
        if "solution" in td.keys() and td["solution"].size(1) == n:
            for batch_idx, row in enumerate(td["solution"].tolist()):
                anchor = int(x1[batch_idx])
                anchor_pos = row.index(anchor)
                ordered = row[anchor_pos:] + row[:anchor_pos]
                node_ranks.append({node: rank for rank, node in enumerate(ordered)})
                open_paths.append(ordered[1:] + ordered[:1])

        # Track early-stopped instances (E-move closed cycle)
        terminated = torch.zeros(bs, dtype=torch.bool, device=h.device)

        # --- Steps kappa = 2 .. K (I-moves and closing E-move) ---
        for _kappa in range(2, K + 1):
            # Recurrent inputs per §4.2:
            # Move stream updates with last selected node: h[prev_x]
            # Edge stream updates with lower-ranked endpoint of Hamiltonian path: h[xi]
            prev_x_exp = prev_x.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, d)
            h_prev = h.gather(1, prev_x_exp).squeeze(1)

            xi_exp = xi.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, d)
            h_xi = h.gather(1, xi_exp).squeeze(1)

            q_mu = self.gru_mu(h_prev, q_mu)
            q_lambda = self.gru_lambda(h_xi, q_lambda)

            score_mu, score_lambda = self._compute_stream_scores(q_mu, q_lambda, h)
            logits = self.tanh_clipping * torch.tanh(score_mu + score_lambda)  # [B, N]

            # Dynamic cyclic-rank feasibility condition: Gamma[a, v] >= Gamma[a, xj] (§4.1-4.2)
            rank_mask = self._compute_cyclic_rank_mask(td, x1, bs, n, xj=xj)
            logits = logits.masked_fill(rank_mask, float("-inf"))

            # If all nodes masked, enforce E-move (select xj, closing cycle)
            all_masked = torch.isneginf(logits).all(dim=-1)
            if all_masked.any():
                logits[all_masked, xj[all_masked]] = 0.0

            if strategy == "greedy":
                x_k = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                x_k = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

            step_log_p = F.log_softmax(logits, dim=-1).gather(1, x_k.unsqueeze(-1)).squeeze(-1)

            # If already terminated by prior E-move, pad with null action (xa) and log_prob 0
            x_k = torch.where(terminated, x1, x_k)
            step_log_p = torch.where(terminated, torch.zeros_like(step_log_p), step_log_p)

            # Check if this step is an E-move (x_k == xj or all masked)
            is_e_move = (x_k == xj) | all_masked
            terminated = terminated | is_e_move

            # Successors change after path reversals; original-tour successors
            # no longer identify the edge removed by a later I-move.
            if open_paths:
                next_xi = xi.clone()
                next_xj = xj.clone()
                for batch_idx, path in enumerate(open_paths):
                    if bool(terminated[batch_idx]):
                        continue
                    selected = int(x_k[batch_idx])
                    selected_pos = path.index(selected)
                    updated = path[selected_pos + 1 :] + list(reversed(path[: selected_pos + 1]))
                    ranks = node_ranks[batch_idx]
                    if ranks[updated[0]] < ranks[updated[-1]]:
                        updated.reverse()
                    open_paths[batch_idx] = updated
                    next_xi[batch_idx] = updated[-1]
                    next_xj[batch_idx] = updated[0]
                xi, xj = next_xi, next_xj

            prev_x = x_k
            actions_list.append(x_k)
            log_p_list.append(step_log_p)

        # 3. Assemble actions and cumulative log probability
        actions = torch.stack(actions_list, dim=-1)  # [B, K]
        log_p = torch.stack(log_p_list, dim=-1).sum(dim=-1)  # [B]

        return log_p, actions
