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
- optional local search (``local_search``): the granular local search of
  ``ACOLocalSearch`` with the candidate lists as neighbourhoods
  (:func:`~logic.src.policies.vectorized.routing.shared.pair_ls.pair_local_search`);
- the iteration-best and best-so-far solutions are the cheapest ones; on a new best,
  ``tau_max = 1 / (rho * best_cost)`` and the default pheromone is capped by it; MMAS
  update on the sparse pheromones of ``SparsePheromoneTau``: deposit ``1 / best_cost``
  on the best-so-far edges (stored as ``get + delta``), then evaporate the default and
  the stored edges by ``(1 - rho)`` within ``[tau_min, tau_max]`` and drop the stored
  edges within ``10^-scale`` of the default;
- ``tau_0`` defaults to ``rho / nearest-neighbour cost`` when None, the nearest
  neighbour being taken in the candidate list or, when no candidate fits, among
  ``max(1, int(log2(n_nodes)))`` random fitting bins;
- stop after ``max_iterations`` or ``time_limit`` seconds.

Attributes:
    construct_ants: Batched ant construction.
    nearest_neighbour_cost: Nearest-neighbour cost used for ``tau_0``.
    SparsePheromones: Batched sparse pheromones with a default value.
    VectorizedACOPolicy: Batched KS-ACO expert configured by ``KSparseACOConfig``.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.pair_ls import pair_local_search
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch


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


def candidate_lists(cand: torch.Tensor) -> torch.Tensor:
    """Candidate lists ``[B, M, K]`` (-1 padded) from candidate masks ``[B, M, M]``."""
    k = int(cand.sum(dim=2).max()) if cand.numel() else 0
    k = max(k, 1)
    vals, idx = torch.topk(cand.float(), k, dim=2)
    return torch.where(vals > 0, idx, torch.full_like(idx, -1))


def nearest_neighbour_cost(inst: VRPPBatch, cand: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    """``KSparseACOSolver._nearest_neighbor_cost`` on a batch ``[B]``.

    From the current node, go to the nearest unvisited bin of its candidate list that
    fits the remaining capacity; when there is none, sample ``min(max(1,
    int(log2(n_nodes))), |valid|)`` random capacity-fitting unvisited bins and go to the
    nearest of them; with no fitting bin at all, return to the depot and open a new
    route. Every route ends with its return leg.
    """
    B, M = inst.dist.size(0), inst.dist.size(1)
    dev = inst.device
    r = torch.arange(B, device=dev)
    unv = torch.ones(B, M, dtype=torch.bool, device=dev)
    unv[:, 0] = False
    # A bin that never fits is left out, as in the simulator.
    unv = unv & (inst.waste <= inst.capacity.unsqueeze(1))
    log_n = max(1, int(math.log2(M)))
    cur = torch.zeros(B, dtype=torch.long, device=dev)
    load = torch.zeros(B, device=dev)
    total = torch.zeros(B, device=dev)
    inf = torch.full_like(inst.waste, float("inf"))
    for _ in range(2 * M + 2):
        if not bool(unv.any()):
            break
        valid = unv & (load.unsqueeze(1) + inst.waste <= inst.capacity.unsqueeze(1))
        drow = inst.dist[r, cur]
        in_c = valid & cand[r, cur]
        d_c = torch.where(in_c, drow, inf)
        key = torch.where(valid, torch.rand(B, M, generator=gen, device=dev), torch.full_like(drow, 2.0))
        n_valid = valid.sum(dim=1, keepdim=True)
        kth = torch.sort(key, dim=1).values.gather(1, (n_valid.clamp(max=log_n) - 1).clamp(min=0))
        sampled = valid & (key <= kth)
        d_s = torch.where(sampled, drow, inf)
        use_c = in_c.any(dim=1)
        d = torch.where(use_c.unsqueeze(1), d_c, d_s)
        best, nxt = d.min(dim=1)
        has = torch.isfinite(best) & unv.any(dim=1)
        close = ~has & unv.any(dim=1)
        total = total + torch.where(has, best, torch.where(close, inst.dist[r, cur, 0], torch.zeros_like(best)))
        load = torch.where(has, load + inst.waste[r, nxt], torch.where(close, torch.zeros_like(load), load))
        unv[r[has], nxt[has]] = False
        cur = torch.where(has, nxt, torch.where(close, torch.zeros_like(cur), cur))
    return total + inst.dist[r, cur, 0]


class SparsePheromones:
    """Batched ``SparsePheromoneTau``: a default value per instance plus stored edges.

    ``get`` returns the stored value of an edge or the default; a deposit stores
    ``get + delta``; ``evaporate_all`` scales and clamps the default and every stored
    edge to ``[tau_min, tau_max]`` and drops the stored edges within ``10^-scale`` of the
    default.
    """

    def __init__(self, B: int, M: int, tau_0: torch.Tensor, scale: float, tau_min: float, tau_max: float) -> None:
        dev = tau_0.device
        self.default = tau_0.clone()
        self.val = torch.zeros(B, M, M, device=dev)
        self.stored = torch.zeros(B, M, M, dtype=torch.bool, device=dev)
        self.tau_min = float(tau_min)
        self.tau_max = torch.full((B,), float(tau_max), device=dev)
        self.precision = 10 ** (-float(scale))

    def dense(self) -> torch.Tensor:
        """``get`` for every edge ``[B, M, M]``."""
        return torch.where(self.stored, self.val, self.default.view(-1, 1, 1))

    def deposit(self, S: torch.Tensor, delta: torch.Tensor, rows: torch.Tensor) -> None:
        """Deposit ``delta`` on the (directed) edges of the routes of ``S`` for ``rows``."""
        B = S.size(0)
        u, v = S[:, :-1], S[:, 1:]
        edge = ((u != 0) | (v != 0)) & rows.unsqueeze(1)
        bi = torch.arange(B, device=S.device).unsqueeze(1).expand_as(u)
        cur = self.dense()[bi[edge], u[edge], v[edge]]
        self.val[bi[edge], u[edge], v[edge]] = cur + delta.unsqueeze(1).expand_as(u)[edge]
        self.stored[bi[edge], u[edge], v[edge]] = True

    def evaporate(self, rho: float, rows: torch.Tensor) -> None:
        """``evaporate_all`` for ``rows``."""
        tmax = self.tau_max
        new_def = torch.maximum(torch.minimum(self.default * (1 - rho), tmax), torch.full_like(tmax, self.tau_min))
        self.default = torch.where(rows, new_def, self.default)
        nv = self.val * (1 - rho)
        nv = torch.maximum(torch.minimum(nv, tmax.view(-1, 1, 1)), torch.full_like(nv, self.tau_min))
        r3 = rows.view(-1, 1, 1)
        self.val = torch.where(r3 & self.stored, nv, self.val)
        prune = r3 & self.stored & ((self.val - self.default.view(-1, 1, 1)).abs() <= self.precision)
        self.stored = self.stored & ~prune


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
        cand = candidate_mask(inst, int(cfg.k_sparse))
        if cfg.tau_0 is None:
            nn = nearest_neighbour_cost(inst, cand, self.generator)
            tau0 = torch.where(nn > 0, cfg.rho / nn.clamp(min=1e-12), torch.full_like(nn, cfg.tau_max))
        else:
            tau0 = torch.full((B,), float(cfg.tau_0), device=dev)
        ph = SparsePheromones(B, M, tau0, cfg.scale, cfg.tau_min, cfg.tau_max)
        nb = candidate_lists(cand)
        best_S = rs.empty_routes(B, M - 1, dev)
        best_cost = torch.full((B,), float("inf"), device=dev)
        start = time.perf_counter()
        b = torch.arange(B, device=dev)
        for it in range(int(cfg.max_iterations)):
            if cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit:
                break
            ants = construct_ants(
                inst,
                ph.dense(),
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
                flat = pair_local_search(flat, sub, nb[rep], int(cfg.local_search_iterations), self.generator)
            costs = rs.seq_cost(flat, sub).view(B, A)
            ib_cost, ib = costs.min(dim=1)
            ib_S = flat.view(B, A, L)[b, ib]
            improved = ib_cost < best_cost
            best_cost = torch.where(improved, ib_cost, best_cost)
            best_S = torch.where(improved.unsqueeze(1), ib_S, best_S)
            # New best: tau_max = 1 / (rho * best_cost) and the default is capped by it.
            # (A best cost of 0 - no bin pays its round trip - divides by zero in the
            # simulator; the empty solution is kept here.)
            upd = improved & (best_cost > 0)
            new_tau_max = 1.0 / (cfg.rho * best_cost.clamp(min=1e-12))
            ph.tau_max = torch.where(upd, new_tau_max, ph.tau_max)
            ph.default = torch.where(upd, torch.minimum(ph.default, new_tau_max), ph.default)
            ok = torch.isfinite(best_cost) & (best_cost > 0) & (rs.n_visited(best_S) > 0)
            delta = torch.where(ok, 1.0 / best_cost.clamp(min=1e-12), torch.zeros_like(best_cost))
            ph.deposit(best_S, delta, ok)
            ph.evaporate(cfg.rho, ok)
            self._viz_record(iteration=it, best_cost=float(best_cost.mean()))
        return best_S
