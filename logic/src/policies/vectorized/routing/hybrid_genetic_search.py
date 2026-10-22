"""Batched Hybrid Genetic Search engine mirroring the simulator's ``HGSSolver``.

Same algorithm as :class:`~logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver`
(Vidal 2022, VRPP adaptation), run on every instance of a batch at once:

- an individual is a giant tour over all bins. ``evaluate`` without routes filters out
  the bins whose revenue does not cover the depot round trip and splits the rest with
  the profit Split; the education is the simulator's granular local search
  (:func:`~logic.src.policies.vectorized.routing.shared.pair_ls.pair_local_search`, ``nb_granular``
  nearest neighbours, ``local_search_iterations`` iterations), after which the giant
  tour is rewritten as in ``HGSLocalSearch.optimize`` (routes ordered by the old
  position of their first bin replace the visited bins in place) and re-split
  (``evaluate`` with routes, no break-even filter);
- the population starts with ``4 * mu`` educated random individuals;
- each iteration: two binary tournaments on two distinct individuals (the biased
  fitness stored by the last update; offspring inserted since have ``inf``), the
  route-profit GPX crossover with probability ``crossover_rate`` (otherwise a clone of
  the first parent), education, insertion at the end of the population list;
- survivors: when the population reaches ``mu + n_offspring`` it is trimmed to ``mu`` by
  removing the first clone in list order, else the worst biased fitness
  ``rank_profit + max(0, 1 - nb_elite / P) * rank_diversity`` with diversity the mean
  broken-pairs distance to the ``nb_close`` closest individuals, updating the fitness
  after every removal;
- stop: ``time_limit`` seconds when positive, otherwise ``n_iterations_no_improvement``
  iterations without improvement; with a time limit the population restarts after
  ``restart_timer`` iterations without improvement; a population below two stops.

In VRPP mode the Split never produces capacity violations, so the feasible and
infeasible subpopulations of the simulator collapse into one and the capacity penalty
(``initial_penalty_capacity``, ``penalty_increase``, ``penalty_decrease``,
``target_feasible``, ``repair_probability``) has no effect, as in the simulator.

Attributes:
    HGSPopulation: Batched population with broken-pairs diversity.
    educate: Decode, educate and re-encode giant tours.
    route_profit_gpx: The simulator's route-profit GPX crossover.
    VectorizedHGSEngine: The batched HGS search.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

import torch

from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.pair_ls import granular_neighbors, pair_local_search
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch, compact, profit_split


def decode(tours: torch.Tensor, inst: VRPPBatch, max_vehicles: int = 0) -> torch.Tensor:
    """``evaluate`` of an individual without routes: break-even filter, then profit Split."""
    b = torch.arange(tours.size(0), device=tours.device).unsqueeze(1)
    rev = torch.gather(inst.waste, 1, tours) * inst.R
    rt = (inst.dist[b, 0, tours] + inst.dist[b, tours, 0]) * inst.C
    keep = (tours != 0) & (rev >= rt)
    return split_active(tours, keep, inst, max_vehicles)


def split_active(tours: torch.Tensor, keep: torch.Tensor, inst: VRPPBatch, max_vehicles: int = 0) -> torch.Tensor:
    """Profit Split of the kept bins of ``tours`` in giant-tour order (``evaluate`` with routes)."""
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


def _tour_positions(tours: torch.Tensor, M: int) -> torch.Tensor:
    """Position of every node in the giant tours ``[B, M]`` (depot: ``N``)."""
    B, N = tours.shape
    pos = torch.full((B, M), N, dtype=torch.long, device=tours.device)
    pos.scatter_(1, tours, torch.arange(N, device=tours.device).unsqueeze(0).expand(B, N))
    return pos


def replace_visited(tours: torch.Tensor, seq: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Write the bins of ``seq`` (packed) into the positions of ``tours`` whose bin is in ``vis``.

    The genotype rewrite of ``HGSLocalSearch.optimize`` and ``route_profit_gpx_crossover``.
    """
    hit = torch.gather(vis, 1, tours)
    k = (torch.cumsum(hit.long(), dim=1) - 1).clamp(min=0)
    return torch.where(hit, torch.gather(seq, 1, k), tours)


def rewrite_giant_tours(S: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """``HGSLocalSearch.optimize`` genotype update.

    Routes are ordered by the old giant-tour position of their first bin and their bins
    replace, in that order, the visited bins of the old giant tour; unvisited bins keep
    their positions.
    """
    B, L = S.shape
    N = tours.size(1)
    M = N + 1
    pos = _tour_positions(tours, M)
    rid = rs.route_ids(S).clamp(min=0)
    prev = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
    first = (S != 0) & (prev == 0)
    rkey = torch.full((B, L + 1), M, dtype=torch.long, device=S.device)
    rkey.scatter_(1, torch.where(first, rid, torch.full_like(rid, L)), torch.gather(pos, 1, S))
    j = torch.arange(L, device=S.device).unsqueeze(0)
    key = torch.where(S != 0, torch.gather(rkey, 1, rid) * (L + 1) + j, torch.full_like(rid, (M + 1) * (L + 1)))
    seq = torch.gather(S, 1, torch.argsort(key, dim=1))[:, :N]
    return replace_visited(tours, seq, rs.visited(S, M))


def educate(
    tours: torch.Tensor,
    inst: VRPPBatch,
    neighbors: torch.Tensor,
    ls_iterations: int,
    gen: Optional[torch.Generator],
    max_vehicles: int = 0,
    S: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``evaluate`` -> ``HGSLocalSearch.optimize`` -> ``evaluate`` on giant tours.

    Individuals whose Split leaves no route skip the local search, as in the simulator.

    Returns:
        Tuple of (giant tours, route sequences, profit).
    """
    M = inst.n_bins + 1
    if S is None:
        S = decode(tours, inst, max_vehicles)
    has = rs.n_visited(S) > 0
    if ls_iterations > 0 and bool(has.any()):
        S_ls = pair_local_search(S, inst, neighbors, ls_iterations, gen, active=has)
        new_tours = rewrite_giant_tours(S_ls, tours)
        S_new = split_active(new_tours, torch.gather(rs.visited(S_ls, M), 1, new_tours), inst, max_vehicles)
        tours = torch.where(has.unsqueeze(1), new_tours, tours)
        S = torch.where(has.unsqueeze(1), S_new, S)
    return tours, S, rs.seq_profit(S, inst)


def _directed_edges(S: torch.Tensor, M: int) -> torch.Tensor:
    """Directed edge indicator ``[B, M, M]`` of route sequences (depot legs included)."""
    B = S.size(0)
    a, b = S[:, :-1], S[:, 1:]
    valid = (a != 0) | (b != 0)
    E = torch.zeros(B, M * M, dtype=torch.bool, device=S.device)
    E.scatter_(1, torch.where(valid, a * M + b, torch.zeros_like(a)), valid)
    E[:, 0] = False
    return E.view(B, M, M)


def route_profit_gpx(
    t1: torch.Tensor,
    S1: torch.Tensor,
    t2: torch.Tensor,
    S2: torch.Tensor,
    inst: VRPPBatch,
    gen: Optional[torch.Generator],
) -> torch.Tensor:
    """Batched ``route_profit_gpx_crossover``: child giant tours ``[B, N]``.

    1. Components of the graph of uncommon physical edges (depot legs excluded) over
       all bins, in increasing order of their smallest bin (the simulator iterates a
       Python ``set`` of small ints, which is ascending).
    2. Every component whose load fits the capacity becomes a child route, ordered as
       in the routes of a parent drawn with probability 1/2 (giant-tour order for its
       unrouted bins).
    3. The other bins, in random order, are packed greedily: best-profit insertion into
       a capacity-feasible child route (profit >= -1e-4), or a new route when that pays
       more and is not loss-making.
    4. The child giant tour is the first parent's with the visited bins replaced by the
       child routes' bins; its routes are discarded and re-built by ``evaluate``.

    Rows where a parent has no route return the first parent's giant tour.
    """
    B, N = t1.shape
    M, L = N + 1, rs.width(N)
    dev = t1.device
    w = inst.waste
    E1, E2 = _directed_edges(S1, M), _directed_edges(S2, M)
    Uc = E1 ^ E2
    A = Uc | Uc.transpose(1, 2)
    A[:, 0, :] = False
    A[:, :, 0] = False
    big = M + 1
    lab = torch.arange(M, device=dev).unsqueeze(0).expand(B, M).clone()
    while True:
        nbr = torch.where(A, lab.unsqueeze(1).expand(B, M, M), torch.full((B, M, M), big, device=dev)).min(dim=2).values
        new = torch.minimum(lab, nbr)
        if torch.equal(new, lab):
            break
        lab = new
    comp_load = torch.zeros(B, M, dtype=w.dtype, device=dev).scatter_add_(1, lab, w)
    fits = torch.gather(comp_load <= inst.capacity.view(B, 1), 1, lab)
    coin = torch.rand(B, M, generator=gen, device=dev) < 0.5
    use1 = torch.gather(coin, 1, lab)

    # Order inside a component: parent route order, then parent giant-tour order.
    def order_key(S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = torch.full((B, M), -1, dtype=torch.long, device=dev)
        j = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        p.scatter_(1, torch.where(S != 0, S, torch.zeros_like(S)), torch.where(S != 0, j, torch.full_like(j, -1)))
        p[:, 0] = -1
        return torch.where(p >= 0, p, L + _tour_positions(t, M))

    k1, k2 = order_key(S1, t1), order_key(S2, t2)
    span = 2 * L + N + 2
    nodes = torch.arange(M, device=dev).unsqueeze(0).expand(B, M)
    inherit = fits & (nodes != 0)
    key = torch.where(inherit, lab * span + torch.where(use1, k1, k2), torch.full_like(lab, big * span))
    order = torch.argsort(key, dim=1)
    seq = torch.gather(nodes, 1, order)
    slab = torch.gather(lab, 1, order)
    sin = torch.gather(inherit, 1, order)
    prev_lab = torch.cat([torch.full_like(slab[:, :1], -1), slab[:, :-1]], dim=1)
    boundary = sin & (slab != prev_lab)
    tgt = torch.arange(M, device=dev).unsqueeze(0) + torch.cumsum(boundary.long(), dim=1)
    S = torch.zeros(B, L, dtype=torch.long, device=dev)
    S.scatter_(1, torch.where(sin, tgt, torch.full_like(tgt, L - 1)), torch.where(sin, seq, torch.zeros_like(seq)))
    S[:, L - 1] = 0
    # Pool: bins of over-capacity components, shuffled.
    pool = (~fits) & (nodes != 0)
    pkey = torch.where(pool, torch.rand(B, M, generator=gen, device=dev), torch.full((B, M), 2.0, device=dev))
    porder = torch.argsort(pkey, dim=1)
    n_pool = int(pool.sum(dim=1).max())
    bi = torch.arange(B, device=dev).unsqueeze(1)
    C = float(inst.C)
    for t in range(n_pool):
        x = porder[:, t]
        act = torch.gather(pool, 1, x.unsqueeze(1)).squeeze(1)
        if not bool(act.any()):
            continue
        wx = torch.gather(w, 1, x.unsqueeze(1))
        rev = wx * inst.R
        a, b = S[:, :-1], S[:, 1:]
        xe = x.unsqueeze(1)
        cost = inst.dist[bi, a, xe] + inst.dist[bi, xe, b] - inst.dist[bi, a, b]
        prof = rev - cost * C
        rid, loads = rs.route_loads(S, inst)
        gl = torch.gather(loads, 1, rid[:, :-1].clamp(min=0))
        real = ~((a == 0) & (b == 0))
        ok = real & (gl + wx <= inst.capacity.view(B, 1)) & (prof >= -1e-4)
        prof_m = torch.where(ok, prof, torch.full_like(prof, float("-inf")))
        best = prof_m.max(dim=1).values
        gap = prof_m.argmax(dim=1)
        newp = (rev - (inst.dist[bi, 0, xe] + inst.dist[bi, xe, 0]) * C).squeeze(1)
        new_route = act & (newp >= 0) & (newp > best)
        ins = act & ~new_route & torch.isfinite(best)
        zz = (a == 0) & (b == 0)
        first_zz = zz.float().argmax(dim=1)
        g = torch.where(new_route, first_zz, gap)
        S = rs.insert_nodes(S, x, g, new_route | ins)
    child = replace_visited(t1, rs.to_giant_tours(S, N), rs.visited(S, M))
    no_routes = (rs.n_visited(S1) == 0) | (rs.n_visited(S2) == 0)
    return torch.where(no_routes.unsqueeze(1), t1, child)


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

    Mirrors the simulator's population list: ``order`` is the position of every
    individual in the list (new individuals are appended; ``update_biased_fitness``
    re-sorts the list by diversity), ``fit`` the biased fitness stored by the last
    ``update_biased_fitness`` (``inf`` for individuals inserted since, as in
    ``Individual``).

    Attributes:
        tours: Giant tours ``[B, P, N]``.
        seqs: Route sequences ``[B, P, L]``.
        profit: Profits ``[B, P]`` (-inf for empty slots).
        alive: Liveness ``[B, P]``.
        edges: Edge indicators ``[B, P, M*M]``.
        dist: Pairwise broken-pairs distances ``[B, P, P]``.
        order: List position ``[B, P]``.
        fit: Stored biased fitness ``[B, P]``.
    """

    def __init__(self, B: int, P: int, N: int, L: int, M: int, device: torch.device) -> None:
        self.tours = torch.zeros(B, P, N, dtype=torch.long, device=device)
        self.seqs = torch.zeros(B, P, L, dtype=torch.long, device=device)
        self.profit = torch.full((B, P), float("-inf"), device=device)
        self.alive = torch.zeros(B, P, dtype=torch.bool, device=device)
        self.edges = torch.zeros(B, P, M * M, device=device)
        self.n_edges = torch.zeros(B, P, device=device)
        self.dist = torch.zeros(B, P, P, device=device)
        self.order = torch.zeros(B, P, dtype=torch.long, device=device)
        self.fit = torch.full((B, P), float("inf"), device=device)
        self.M = M

    def insert(
        self, tours: torch.Tensor, seqs: torch.Tensor, profit: torch.Tensor, rows: Optional[torch.Tensor] = None
    ) -> None:
        """Append one individual per row (first free slot; rows must have one)."""
        B = tours.size(0)
        dev = tours.device
        rows = torch.arange(B, device=dev) if rows is None else rows
        free = (~self.alive[rows]).float().argmax(dim=1)
        last = torch.where(self.alive[rows], self.order[rows], torch.full_like(self.order[rows], -1)).max(dim=1).values
        self.order[rows, free] = last + 1
        self.fit[rows, free] = float("inf")
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
        self.fit[rows, idx] = float("inf")

    def size(self) -> torch.Tensor:
        """Population size per row ``[B]``."""
        return self.alive.sum(dim=1)

    def _list_key(self) -> torch.Tensor:
        big = self.order.size(1) * 4 + 4
        return torch.where(self.alive, self.order, torch.full_like(self.order, big))

    def biased_fitness(self, nb_elite: int, nb_close: int, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``update_biased_fitness``: store and return the biased fitness (lower is better).

        Profit ranks break ties by list order, diversity ranks by profit rank (two stable
        sorts, as in the simulator); the list is then in diversity order.
        """
        B, P = self.alive.shape
        dev = self.alive.device
        rows = torch.ones(B, dtype=torch.bool, device=dev) if rows is None else rows
        alive = self.alive
        n = alive.sum(dim=1, keepdim=True).float()
        prof = torch.where(alive, self.profit, torch.full_like(self.profit, float("-inf")))
        lk = torch.argsort(self._list_key(), dim=1, stable=True)
        o1 = torch.gather(lk, 1, torch.argsort(-torch.gather(prof, 1, lk), dim=1, stable=True))
        rank_p = torch.argsort(o1, dim=1).float() + 1
        pair_ok = alive.unsqueeze(1) & alive.unsqueeze(2) & ~torch.eye(P, dtype=torch.bool, device=dev).unsqueeze(0)
        d = torch.where(pair_ok, self.dist, torch.full_like(self.dist, float("inf")))
        k = max(1, min(int(nb_close), P - 1))
        near = torch.topk(d, k, dim=2, largest=False).values
        finite = torch.isfinite(near)
        cnt = finite.sum(dim=2).clamp(min=1)
        div = torch.where(finite, near, torch.zeros_like(near)).sum(dim=2) / cnt
        div = torch.where(finite.any(dim=2), div, torch.zeros_like(div))
        div = torch.where(alive, div, torch.full_like(div, float("-inf")))
        o2 = torch.gather(o1, 1, torch.argsort(-torch.gather(div, 1, o1), dim=1, stable=True))
        rank_d = torch.argsort(o2, dim=1)
        w = (1.0 - nb_elite / n.clamp(min=1)).clamp(min=0.0)
        fit = rank_p + w * (rank_d.float() + 1)
        fit = torch.where(alive, fit, torch.full_like(fit, float("inf")))
        r = rows.unsqueeze(1)
        self.fit = torch.where(r, fit, self.fit)
        self.order = torch.where(r & alive, rank_d, self.order)
        return fit

    def clones(self) -> torch.Tensor:
        """``[B, P]`` mask of individuals whose edge set equals one earlier in the list."""
        same = (self.dist < 1e-9) & (self.n_edges.unsqueeze(2) == self.n_edges.unsqueeze(1))
        same = same & self.alive.unsqueeze(1) & self.alive.unsqueeze(2)
        lk = self._list_key()
        earlier = lk.unsqueeze(2) > lk.unsqueeze(1)
        return (same & earlier).any(dim=2) & self.alive

    def trim(self, target: int, nb_elite: int, nb_close: int, rows: torch.Tensor) -> None:
        """``_trim_pop``: remove the first clone in list order, else the worst stored fitness."""
        self.biased_fitness(nb_elite, nb_close, rows)
        lk_big = float(self.order.size(1) * 4 + 8)
        while True:
            over = rows & (self.size() > target)
            if not bool(over.any()):
                break
            idx_rows = over.nonzero(as_tuple=True)[0]
            lk = self._list_key()[idx_rows].float()
            cl = self.clones()[idx_rows]
            has_clone = cl.any(dim=1)
            clone_idx = torch.where(cl, lk, torch.full_like(lk, lk_big)).argmin(dim=1)
            fit = torch.where(self.alive[idx_rows], self.fit[idx_rows], torch.full_like(lk, float("-inf")))
            worst = fit.max(dim=1, keepdim=True).values
            worst_idx = torch.where(fit == worst, lk, torch.full_like(lk, lk_big)).argmin(dim=1)
            self.remove(torch.where(has_clone, clone_idx, worst_idx), idx_rows)
            self.biased_fitness(nb_elite, nb_close, over)

    def tournament(self, fit: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Binary tournament on two distinct random individuals (``random.sample``); ties go to the second."""
        w = self.alive.float()
        ab = torch.multinomial(w, 2, replacement=False, generator=gen)
        a, b = ab[:, 0], ab[:, 1]
        r = torch.arange(w.size(0), device=w.device)
        return torch.where(fit[r, a] < fit[r, b], a, b)

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

    def _educate(self, tours: torch.Tensor, inst: VRPPBatch, nb: torch.Tensor):
        cfg = self.cfg
        return educate(tours, inst, nb, int(cfg.local_search_iterations), self.gen, int(cfg.max_vehicles))

    def _init_population(self, inst: VRPPBatch, nb: torch.Tensor) -> HGSPopulation:
        cfg = self.cfg
        B, N, M = inst.batch_size, inst.n_bins, inst.n_bins + 1
        mu = int(cfg.mu)
        cap = mu + int(cfg.n_offspring) + 1
        size0 = 4 * mu
        pop = HGSPopulation(B, max(cap, size0 + 1), N, rs.width(N), M, inst.device)
        for _ in range(size0):
            tours = rs.random_giant_tours(B, N, self.gen, inst.device)
            t, S, p = self._educate(tours, inst, nb)
            pop.insert(t, S, p)
        pop.biased_fitness(int(cfg.nb_elite), int(cfg.nb_close))
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
        nb = granular_neighbors(inst, int(cfg.nb_granular))
        start = time.perf_counter()
        use_time = cfg.time_limit > 0
        restart_threshold = cfg.restart_timer if cfg.restart_timer > 0 else cfg.n_iterations_no_improvement
        pop = self._init_population(inst, nb)
        _, best_S, best_p = pop.best()
        no_imp = torch.zeros(B, dtype=torch.long, device=dev)
        running = torch.ones(B, dtype=torch.bool, device=dev)
        it = 0
        mu, lam = int(cfg.mu), int(cfg.n_offspring)
        r = torch.arange(B, device=dev)
        while True:
            if use_time and time.perf_counter() - start >= cfg.time_limit:
                break
            if not use_time:
                running = running & (no_imp < int(cfg.n_iterations_no_improvement))
            if use_time:
                restart = (running & (no_imp >= restart_threshold)).nonzero(as_tuple=True)[0]
                if restart.numel():
                    sub = inst.subset(restart)
                    fresh = self._init_population(sub, nb[restart])
                    for name in ("tours", "seqs", "profit", "alive", "edges", "n_edges", "dist", "order", "fit"):
                        t = getattr(pop, name)
                        f = getattr(fresh, name)
                        if t.shape[1:] == f.shape[1:]:
                            t[restart] = f
                    no_imp[restart] = 0
            running = running & (pop.size() >= 2)
            if not bool(running.any()):
                break
            it += 1
            no_imp = no_imp + running.long()
            rows = running.nonzero(as_tuple=True)[0]
            fit = pop.fit
            i1 = pop.tournament(fit, self.gen)
            i2 = pop.tournament(fit, self.gen)
            do_x = torch.rand(B, generator=self.gen, device=dev) < cfg.crossover_rate
            child = pop.tours[r, i1]
            xr = (do_x & running).nonzero(as_tuple=True)[0]
            if xr.numel():
                child[xr] = route_profit_gpx(
                    pop.tours[xr, i1[xr]],
                    pop.seqs[xr, i1[xr]],
                    pop.tours[xr, i2[xr]],
                    pop.seqs[xr, i2[xr]],
                    inst.subset(xr),
                    self.gen,
                )
            t, S, p = self._educate(child[rows], inst.subset(rows), nb[rows])
            pop.insert(t, S, p, rows)
            imp = torch.zeros(B, dtype=torch.bool, device=dev)
            imp[rows] = p > best_p[rows]
            best_S = torch.where(imp.unsqueeze(1), _scatter_rows(best_S, rows, S), best_S)
            best_p = torch.where(imp, _scatter_vals(best_p, rows, p), best_p)
            no_imp = torch.where(imp, torch.zeros_like(no_imp), no_imp)
            full = running & (pop.size() >= mu + lam)
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
