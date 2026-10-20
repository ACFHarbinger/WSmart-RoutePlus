"""Vectorised K-sparse ACO: the simulator's KS-ACO (``KSparseACOConfig``) on a batch.

Mirrors :class:`~logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.solver.KSparseACOSolver`
(MMAS, Stützle & Hoos 2000; Hale 2021), with all ants of all instances built at once:

- candidate lists: the ``k_sparse`` nearest nodes of every node (the depot is always a
  candidate of a bin);
- construction: from the depot, the next bin is drawn among the feasible bins in the
  candidate list (or among all feasible bins when none is a candidate) with probability
  ``tau^alpha * eta^beta``, ``eta = 1 / d``; a bin is feasible when it fits the
  remaining capacity and, with ``profit_aware_operators``, when its revenue beats its
  marginal detour; a route closes when no bin is feasible and a new one starts while
  some unvisited bin pays its depot round trip;
- optional local search (``local_search``, ``local_search_iterations`` moves);
- the iteration-best and best-so-far solutions are the cheapest ones; on a new best,
  ``tau_max = 1 / (rho * best_cost)``; MMAS update: deposit ``1 / best_cost`` on the
  best-so-far edges, then evaporate everything by ``(1 - rho)`` within
  ``[tau_min, tau_max]``;
- ``tau_0`` defaults to ``rho / nearest-neighbour cost`` when None;
- stop after ``max_iterations`` or ``time_limit`` seconds.

Attributes:
    construct_ants: Batched ant construction.
    VectorizedACOPolicy: Batched KS-ACO expert configured by ``KSparseACOConfig``.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.policies.vector.shared import routes as rs
from logic.src.policies.vector.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vector.shared.vrpp import VRPPBatch

from .hybrid_genetic_search import HGS_LS_OPERATORS


def candidate_mask(inst: VRPPBatch, k: int) -> torch.Tensor:
    """``[B, M, M]`` mask of the ``k`` nearest nodes of every node (depot forced in for bins)."""
    B, M = inst.dist.size(0), inst.dist.size(1)
    d = inst.dist.clone()
    eye = torch.eye(M, dtype=torch.bool, device=d.device).unsqueeze(0)
    d = torch.where(eye, torch.full_like(d, float("inf")), d)
    k = max(0, min(k, M - 1))
    mask = torch.zeros(B, M, M, dtype=torch.bool, device=d.device)
    if k > 0:
        idx = torch.topk(d, k, dim=2, largest=False).indices
        mask.scatter_(2, idx, True)
        # Guarantee depot access: replace the furthest candidate by the depot.
        missing = ~mask[:, 1:, 0]
        far = idx[:, 1:, -1]
        rows = missing.nonzero(as_tuple=True)
        mask[rows[0], rows[1] + 1, far[rows]] = False
        mask[:, 1:, 0] = True
    return mask


def construct_ants(
    inst: VRPPBatch,
    tau: torch.Tensor,
    cand: torch.Tensor,
    n_ants: int,
    alpha: float,
    beta: float,
    profit_aware: bool,
    gen: Optional[torch.Generator],
    expand_pool: bool = True,
) -> torch.Tensor:
    """Build ``n_ants`` solutions per instance.

    Args:
        inst: Instances.
        tau: Pheromones ``[B, M, M]``.
        cand: Candidate masks ``[B, M, M]``.
        n_ants: Ants per instance.
        alpha: Pheromone exponent.
        beta: Heuristic exponent.
        profit_aware: Only bins whose revenue beats their detour are feasible.
        gen: Torch generator.
        expand_pool: VRPP mode (new routes only while some bin pays its round trip).

    Returns:
        torch.Tensor: Route sequences ``[B, n_ants, L]``.
    """
    B, M = inst.dist.size(0), inst.dist.size(1)
    N = M - 1
    dev = inst.device
    R = B * n_ants
    rep = torch.arange(B, device=dev).repeat_interleave(n_ants)
    D = inst.dist[rep]
    W = inst.waste[rep]
    cap = inst.capacity[rep]
    T = tau[rep]
    CM = cand[rep]
    eta = torch.where(D > 0, 1.0 / D.clamp(min=1e-12), torch.zeros_like(D))
    L = rs.width(N)
    S = torch.zeros(R, L, dtype=torch.long, device=dev)
    pos = torch.zeros(R, dtype=torch.long, device=dev)
    cur = torch.zeros(R, dtype=torch.long, device=dev)
    load = torch.zeros(R, device=dev)
    unv = torch.ones(R, M, dtype=torch.bool, device=dev)
    unv[:, 0] = False
    unv = unv & (cap.unsqueeze(1) + 1e-9 >= W)  # bins that can never fit are dropped
    done = torch.zeros(R, dtype=torch.bool, device=dev)
    r = torch.arange(R, device=dev)
    rev = W * inst.R
    pays_round = rev > 2 * D[:, 0, :] * inst.C
    # Before the first route: stop when nothing pays (VRPP).
    if expand_pool:
        done = done | ~(unv & pays_round).any(dim=1)
    for _ in range(2 * N + 2):
        if bool(done.all()):
            break
        feas = unv & (load.unsqueeze(1) + W <= cap.unsqueeze(1) + 1e-9)
        if profit_aware and expand_pool:
            marg = (D[r, cur] + D[:, :, 0] - D[r, cur, 0].unsqueeze(1)) * inst.C
            feas = feas & (rev > marg)
        in_c = feas & CM[r, cur]
        pool = torch.where(in_c.any(dim=1, keepdim=True), in_c, feas)
        has = pool.any(dim=1) & ~done
        weight = (T[r, cur].clamp(min=0) ** alpha) * (eta[r, cur] ** beta)
        weight = torch.where(pool, weight, torch.zeros_like(weight))
        zero_tot = weight.sum(dim=1) <= 0
        weight = torch.where(zero_tot.unsqueeze(1), pool.float(), weight)
        weight = torch.where(has.unsqueeze(1), weight, torch.ones_like(weight))
        nxt = torch.multinomial(weight, 1, generator=gen).squeeze(1)
        # Move.
        pos = torch.where(has, pos + 1, pos)
        S[r[has], pos[has]] = nxt[has]
        load = torch.where(has, load + W[r, nxt], load)
        unv[r[has], nxt[has]] = False
        cur = torch.where(has, nxt, cur)
        # Close the route of ants with nothing feasible.
        close = ~has & ~done & (cur != 0)
        pos = torch.where(close, pos + 1, pos)
        cur = torch.where(close, torch.zeros_like(cur), cur)
        load = torch.where(close, torch.zeros_like(load), load)
        # Ants at the depot with nothing feasible, or (VRPP) nothing paying, are done.
        stuck = ~has & ~done & ~close
        cont = unv.any(dim=1)
        if expand_pool:
            cont = cont & (unv & pays_round).any(dim=1)
        done = done | stuck | (close & ~cont) | (~unv.any(dim=1) & (cur == 0))
    return rs.normalize(S).view(B, n_ants, L)


def nearest_neighbour_cost(inst: VRPPBatch) -> torch.Tensor:
    """Cost of a capacity-feasible nearest-neighbour solution over all bins ``[B]``."""
    B, M = inst.dist.size(0), inst.dist.size(1)
    dev = inst.device
    r = torch.arange(B, device=dev)
    unv = torch.ones(B, M, dtype=torch.bool, device=dev)
    unv[:, 0] = False
    unv = unv & (inst.waste <= inst.capacity.unsqueeze(1) + 1e-9)
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    load = torch.zeros(B, device=dev)
    total = torch.zeros(B, device=dev)
    for _ in range(2 * M):
        if not bool(unv.any()):
            break
        feas = unv & (load.unsqueeze(1) + inst.waste <= inst.capacity.unsqueeze(1) + 1e-9)
        d = torch.where(feas, inst.dist[r, cur], torch.full_like(inst.waste, float("inf")))
        best, nxt = d.min(dim=1)
        has = torch.isfinite(best)
        total = total + torch.where(has, best, inst.dist[r, cur, 0])
        load = torch.where(has, load + inst.waste[r, nxt], torch.zeros_like(load))
        unv[r[has], nxt[has]] = False
        cur = torch.where(has, nxt, torch.zeros_like(cur))
    return total + inst.dist[r, cur, 0]


class VectorizedACOPolicy(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``KSparseACOPolicy`` / ``KSparseACOSolver``.

    Takes exactly the ``KSparseACOConfig`` fields; ``packing``, ``route_improvement`` and
    ``acceptance_criterion`` (unused by the simulator's MMAS loop) are ignored. ``q0``
    is accepted but, as in the simulator's MMAS construction, not used.
    """

    config_cls = KSparseACOConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched KS-ACO.

        Args:
            inst: Instances.
            initial: Ignored.

        Returns:
            torch.Tensor: Best route sequences.
        """
        cfg = self.cfg
        B, M = inst.dist.size(0), inst.dist.size(1)
        dev = inst.device
        if M <= 1:
            return rs.empty_routes(B, 0, dev)
        if cfg.tau_0 is None:
            nn = nearest_neighbour_cost(inst)
            tau0 = torch.where(nn > 0, cfg.rho / nn.clamp(min=1e-12), torch.full_like(nn, cfg.tau_max))
        else:
            tau0 = torch.full((B,), float(cfg.tau_0), device=dev)
        tau = tau0.view(B, 1, 1).expand(B, M, M).clone()
        tau_max = torch.full((B,), float(cfg.tau_max), device=dev)
        cand = candidate_mask(inst, int(cfg.k_sparse))
        best_S = rs.empty_routes(B, M - 1, dev)
        best_cost = torch.full((B,), float("inf"), device=dev)
        start = time.perf_counter()
        b = torch.arange(B, device=dev)
        for it in range(int(cfg.max_iterations)):
            if cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit:
                break
            ants = construct_ants(
                inst,
                tau,
                cand,
                int(cfg.n_ants),
                cfg.alpha,
                cfg.beta,
                bool(cfg.profit_aware_operators),
                self.generator,
                self.expand_pool,
            )
            A, L = ants.size(1), ants.size(2)
            flat = ants.reshape(B * A, L)
            rep = b.repeat_interleave(A)
            sub = inst.subset(rep)
            if cfg.local_search:
                flat = rs.local_search(flat, sub, HGS_LS_OPERATORS, int(cfg.local_search_iterations))
            costs = rs.seq_cost(flat, sub).view(B, A)
            ib_cost, ib = costs.min(dim=1)
            ib_S = flat.view(B, A, L)[b, ib]
            improved = ib_cost < best_cost
            best_cost = torch.where(improved, ib_cost, best_cost)
            best_S = torch.where(improved.unsqueeze(1), ib_S, best_S)
            new_tau_max = 1.0 / (cfg.rho * best_cost.clamp(min=1e-12))
            tau_max = torch.where(improved & (best_cost > 0), new_tau_max, tau_max)
            # Deposit on best-so-far edges, then evaporate and bound.
            ok = torch.isfinite(best_cost) & (best_cost > 0) & (rs.n_visited(best_S) > 0)
            delta = torch.where(ok, 1.0 / best_cost.clamp(min=1e-12), torch.zeros_like(best_cost))
            u, v = best_S[:, :-1], best_S[:, 1:]
            edge = (u != 0) | (v != 0)
            tau.index_put_(
                (b.unsqueeze(1).expand_as(u)[edge], u[edge], v[edge]),
                delta.unsqueeze(1).expand_as(u)[edge],
                accumulate=True,
            )
            tau = torch.where(ok.view(B, 1, 1), tau * (1 - cfg.rho), tau)
            tau = torch.maximum(torch.minimum(tau, tau_max.view(B, 1, 1)), torch.full_like(tau, cfg.tau_min))
            self._viz_record(iteration=it, best_cost=float(best_cost.mean()))
        return best_S
