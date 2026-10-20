"""Batched Hybrid Genetic Search engine mirroring the simulator's ``HGSSolver``.

Same algorithm as :class:`~logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver`
(Vidal 2022, VRPP adaptation), run on every instance of a batch at once:

- an individual is a giant tour over all bins; it is decoded by the profit Split (bins
  that do not pay may be skipped; bins whose revenue does not cover the depot round
  trip are filtered out first, as in ``evaluate``) and educated by local search
  (``local_search_iterations`` moves of relocate, swap, 2-opt, 2-opt*, swap*), after
  which the giant tour is rebuilt from the routes followed by the unvisited bins;
- the population starts with ``4 * mu`` educated random individuals;
- each iteration: binary tournament on biased fitness, crossover with probability
  ``crossover_rate`` (otherwise a clone of the first parent), education, insertion;
- survivors: when the population reaches ``mu + n_offspring`` it is trimmed to ``mu`` by
  removing clones first, then the worst biased fitness
  ``rank_profit + (1 - nb_elite / P) * rank_diversity`` with diversity the mean
  broken-pairs distance to the ``nb_close`` closest individuals;
- stop: ``time_limit`` seconds when positive, otherwise ``n_iterations_no_improvement``
  iterations without improvement; with a time limit the population restarts after
  ``restart_timer`` iterations without improvement.

In VRPP mode the Split never produces capacity violations, so the feasible and
infeasible subpopulations of the simulator collapse into one and the capacity penalty
(``initial_penalty_capacity``, ``penalty_increase``, ``penalty_decrease``,
``target_feasible``, ``repair_probability``) has no effect, as in the simulator.

Attributes:
    HGSPopulation: Batched population with broken-pairs diversity.
    educate: Decode, educate and re-encode giant tours.
    VectorizedHGSEngine: The batched HGS search.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

import torch

from logic.src.configs.policies.hgs import HGSConfig
from logic.src.policies.vector.core.crossover import vectorized_ordered_crossover
from logic.src.policies.vector.shared import routes as rs
from logic.src.policies.vector.shared.vrpp import VRPPBatch, compact, profit_split

HGS_LS_OPERATORS = ("relocate", "swap", "two_opt", "two_opt_star", "swap_star")


def decode(tours: torch.Tensor, inst: VRPPBatch, max_vehicles: int = 0) -> torch.Tensor:
    """Giant tours ``[B, N]`` -> route sequences via the break-even filter and profit Split."""
    b = torch.arange(tours.size(0), device=tours.device).unsqueeze(1)
    rev = torch.gather(inst.waste, 1, tours) * inst.R
    rt = (inst.dist[b, 0, tours] + inst.dist[b, tours, 0]) * inst.C
    keep = (tours != 0) & (rev >= rt)
    active = compact(torch.where(keep, tours, torch.zeros_like(tours)))
    routes, _, _ = profit_split(active, inst, max_vehicles=max_vehicles, allow_skip=True)
    assert routes is not None
    return rs.from_route_lists(routes, inst.n_bins, tours.device)


def encode(S: torch.Tensor, old_tours: torch.Tensor) -> torch.Tensor:
    """Giant tours: the bins of ``S`` in route order, then the unvisited bins in ``old_tours`` order."""
    N = old_tours.size(1)
    M = N + 1
    vis = rs.visited(S, M)
    head = rs.to_giant_tours(S, N)
    tail_keep = ~torch.gather(vis, 1, old_tours)
    tail = compact(torch.where(tail_keep, old_tours, torch.zeros_like(old_tours)))
    n_head = (head != 0).sum(dim=1, keepdim=True)
    j = torch.arange(N, device=S.device).unsqueeze(0)
    src = (j - n_head).clamp(min=0)
    out = torch.where(j < n_head, head, torch.gather(tail, 1, src))
    return out


def educate(
    tours: torch.Tensor, inst: VRPPBatch, ls_moves: int, max_vehicles: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode, educate with local search and re-encode giant tours.

    Returns:
        Tuple of (giant tours, route sequences, profit).
    """
    S = decode(tours, inst, max_vehicles)
    if ls_moves > 0:
        S = rs.local_search(S, inst, HGS_LS_OPERATORS, ls_moves)
    return encode(S, tours), S, rs.seq_profit(S, inst)


def _edge_indicator(S: torch.Tensor, M: int) -> torch.Tensor:
    """Undirected edge indicator ``[..., M*M]`` of route sequences (depot edges included)."""
    a, b = S[..., :-1], S[..., 1:]
    u, v = torch.minimum(a, b), torch.maximum(a, b)
    valid = (a != 0) | (b != 0)
    h = torch.where(valid, u * M + v, torch.zeros_like(u))
    ind = torch.zeros(*S.shape[:-1], M * M, dtype=torch.float32, device=S.device)
    ind.scatter_(-1, h, valid.float())
    ind[..., 0] = 0.0
    return ind


class HGSPopulation:
    """A batched population of ``[B, P]`` individuals with liveness masks.

    Attributes:
        tours: Giant tours ``[B, P, N]``.
        seqs: Route sequences ``[B, P, L]``.
        profit: Profits ``[B, P]`` (-inf for empty slots).
        alive: Liveness ``[B, P]``.
        edges: Edge indicators ``[B, P, M*M]``.
        dist: Pairwise broken-pairs distances ``[B, P, P]``.
    """

    def __init__(self, B: int, P: int, N: int, L: int, M: int, device: torch.device) -> None:
        self.tours = torch.zeros(B, P, N, dtype=torch.long, device=device)
        self.seqs = torch.zeros(B, P, L, dtype=torch.long, device=device)
        self.profit = torch.full((B, P), float("-inf"), device=device)
        self.alive = torch.zeros(B, P, dtype=torch.bool, device=device)
        self.edges = torch.zeros(B, P, M * M, device=device)
        self.n_edges = torch.zeros(B, P, device=device)
        self.dist = torch.zeros(B, P, P, device=device)
        self.M = M

    def insert(
        self, tours: torch.Tensor, seqs: torch.Tensor, profit: torch.Tensor, rows: Optional[torch.Tensor] = None
    ) -> None:
        """Put one individual per row in the first free slot (rows must have one)."""
        B = tours.size(0)
        dev = tours.device
        rows = torch.arange(B, device=dev) if rows is None else rows
        free = (~self.alive[rows]).float().argmax(dim=1)
        self.tours[rows, free] = tours
        self.seqs[rows, free] = seqs
        self.profit[rows, free] = profit
        self.alive[rows, free] = True
        e = _edge_indicator(seqs, self.M)
        self.edges[rows, free] = e
        ne = e.sum(dim=1)
        self.n_edges[rows, free] = ne
        inter = torch.einsum("be,bpe->bp", e, self.edges[rows])
        mx = torch.maximum(ne.unsqueeze(1), self.n_edges[rows]).clamp(min=1)
        d = 1.0 - inter / mx
        both_empty = (ne.unsqueeze(1) == 0) & (self.n_edges[rows] == 0)
        d = torch.where(both_empty, torch.zeros_like(d), d)
        self.dist[rows, free] = d
        self.dist[rows, :, free] = d
        self.dist[rows, free, free] = 0.0

    def remove(self, idx: torch.Tensor, rows: torch.Tensor) -> None:
        """Free slot ``idx[k]`` of row ``rows[k]``."""
        self.alive[rows, idx] = False
        self.profit[rows, idx] = float("-inf")

    def size(self) -> torch.Tensor:
        """Population size per row ``[B]``."""
        return self.alive.sum(dim=1)

    def biased_fitness(self, nb_elite: int, nb_close: int) -> torch.Tensor:
        """Biased fitness ``[B, P]`` (lower is better, +inf for empty slots)."""
        B, P = self.alive.shape
        dev = self.alive.device
        alive = self.alive
        n = alive.sum(dim=1, keepdim=True).float()
        prof = torch.where(alive, self.profit, torch.full_like(self.profit, float("-inf")))
        order = torch.argsort(-prof, dim=1, stable=True)
        rank_p = torch.argsort(order, dim=1).float() + 1
        pair_ok = alive.unsqueeze(1) & alive.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=dev).unsqueeze(0)
        d = torch.where(pair_ok, self.dist, torch.full_like(self.dist, float("inf")))
        k = max(1, min(int(nb_close), P - 1))
        near = torch.topk(d, k, dim=2, largest=False).values
        finite = torch.isfinite(near)
        cnt = finite.sum(dim=2).clamp(min=1)
        div = torch.where(finite, near, torch.zeros_like(near)).sum(dim=2) / cnt
        div = torch.where(finite.any(dim=2), div, torch.zeros_like(div))
        div = torch.where(alive, div, torch.full_like(div, float("-inf")))
        order_d = torch.argsort(-div, dim=1, stable=True)
        rank_d = torch.argsort(order_d, dim=1).float() + 1
        w = (1.0 - nb_elite / n.clamp(min=1)).clamp(min=0.0)
        fit = rank_p + w * rank_d
        return torch.where(alive, fit, torch.full_like(fit, float("inf")))

    def clones(self) -> torch.Tensor:
        """``[B, P]`` mask of individuals whose edge set equals an earlier alive one."""
        B, P = self.alive.shape
        dev = self.alive.device
        same = (self.dist < 1e-9) & (self.n_edges.unsqueeze(2) == self.n_edges.unsqueeze(1))
        same = same & self.alive.unsqueeze(1) & self.alive.unsqueeze(2)
        earlier = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
        return (same & earlier).any(dim=2) & self.alive

    def trim(self, target: int, nb_elite: int, nb_close: int, rows: torch.Tensor) -> None:
        """Remove clones, then the worst biased fitness, until ``rows`` have ``target`` individuals."""
        while True:
            over = rows & (self.size() > target)
            if not bool(over.any()):
                break
            idx_rows = over.nonzero(as_tuple=True)[0]
            cl = self.clones()[idx_rows]
            fit = self.biased_fitness(nb_elite, nb_close)[idx_rows]
            has_clone = cl.any(dim=1)
            clone_idx = cl.float().argmax(dim=1)
            fit = torch.where(self.alive[idx_rows], fit, torch.full_like(fit, float("-inf")))
            worst_idx = fit.argmax(dim=1)
            self.remove(torch.where(has_clone, clone_idx, worst_idx), idx_rows)

    def tournament(self, fit: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Binary tournament: index ``[B]`` of the fitter of two random alive individuals."""
        w = self.alive.float()
        a = torch.multinomial(w, 1, replacement=True, generator=gen).squeeze(1)
        b = torch.multinomial(w, 1, replacement=True, generator=gen).squeeze(1)
        r = torch.arange(w.size(0), device=w.device)
        return torch.where(fit[r, a] <= fit[r, b], a, b)

    def best(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Best (giant tour, route sequence, profit) per row."""
        i = self.profit.argmax(dim=1)
        r = torch.arange(self.profit.size(0), device=i.device)
        return self.tours[r, i], self.seqs[r, i], self.profit[r, i]


class VectorizedHGSEngine:
    """Batched HGS for the VRPP configured by ``HGSConfig``."""

    def __init__(
        self, cfg: HGSConfig, generator: torch.Generator, recorder: Optional[Callable[..., Any]] = None
    ) -> None:
        """Store the configuration.

        Args:
            cfg: HGS configuration.
            generator: Torch generator.
            recorder: Optional telemetry callback.
        """
        self.cfg = cfg
        self.gen = generator
        self._record = recorder or (lambda **k: None)

    def _init_population(self, inst: VRPPBatch) -> HGSPopulation:
        cfg = self.cfg
        B, N, M = inst.batch_size, inst.n_bins, inst.n_bins + 1
        mu = int(cfg.mu)
        cap = mu + int(cfg.n_offspring) + 1
        size0 = 4 * mu
        pop = HGSPopulation(B, max(cap, size0 + 1), N, rs.width(N), M, inst.device)
        for _ in range(size0):
            tours = rs.random_giant_tours(B, N, self.gen, inst.device)
            t, S, p = educate(tours, inst, int(cfg.local_search_iterations), int(cfg.max_vehicles))
            pop.insert(t, S, p)
        return pop

    def solve(self, inst: VRPPBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run HGS on the batch.

        Returns:
            Tuple of (best route sequences, best profit ``[B]``).
        """
        cfg = self.cfg
        B, N = inst.batch_size, inst.n_bins
        dev = inst.device
        if N == 0:
            return rs.empty_routes(B, N, dev), torch.zeros(B, device=dev)
        start = time.perf_counter()
        use_time = cfg.time_limit > 0
        restart_threshold = cfg.restart_timer if cfg.restart_timer > 0 else cfg.n_iterations_no_improvement
        pop = self._init_population(inst)
        _, best_S, best_p = pop.best()
        no_imp = torch.zeros(B, dtype=torch.long, device=dev)
        running = torch.ones(B, dtype=torch.bool, device=dev)
        it = 0
        mu, lam = int(cfg.mu), int(cfg.n_offspring)
        while True:
            if use_time and time.perf_counter() - start >= cfg.time_limit:
                break
            if not use_time:
                running = running & (no_imp < int(cfg.n_iterations_no_improvement))
            if not bool(running.any()):
                break
            if use_time:
                restart = (no_imp >= restart_threshold).nonzero(as_tuple=True)[0]
                if restart.numel():
                    sub = inst.subset(restart)
                    fresh = self._init_population(sub)
                    for name in ("tours", "seqs", "profit", "alive", "edges", "n_edges", "dist"):
                        t = getattr(pop, name)
                        f = getattr(fresh, name)
                        if t.shape[1:] == f.shape[1:]:
                            t[restart] = f
                    no_imp[restart] = 0
            it += 1
            no_imp = no_imp + running.long()
            fit = pop.biased_fitness(int(cfg.nb_elite), int(cfg.nb_close))
            r = torch.arange(B, device=dev)
            p1 = pop.tours[r, pop.tournament(fit, self.gen)]
            p2 = pop.tours[r, pop.tournament(fit, self.gen)]
            do_x = torch.rand(B, generator=self.gen, device=dev) < cfg.crossover_rate
            child = torch.where(do_x.unsqueeze(1), vectorized_ordered_crossover(p1, p2, generator=self.gen), p1)
            rows = running.nonzero(as_tuple=True)[0]
            t, S, p = educate(child[rows], inst.subset(rows), int(cfg.local_search_iterations), int(cfg.max_vehicles))
            pop.insert(t, S, p, rows)
            imp = torch.zeros(B, dtype=torch.bool, device=dev)
            imp[rows] = p > best_p[rows] + 1e-9
            best_S = torch.where(imp.unsqueeze(1), _scatter_rows(best_S, rows, S), best_S)
            best_p = torch.where(imp, _scatter_vals(best_p, rows, p), best_p)
            no_imp = torch.where(imp, torch.zeros_like(no_imp), no_imp)
            full = pop.size() >= mu + lam
            if bool(full.any()):
                pop.trim(mu, int(cfg.nb_elite), int(cfg.nb_close), full)
            self._record(iteration=it, best_profit=float(best_p.mean()))
        return best_S, best_p


def _scatter_rows(base: torch.Tensor, rows: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    out[rows] = vals
    return out


def _scatter_vals(base: torch.Tensor, rows: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    out = base.clone()
    out[rows] = vals
    return out
