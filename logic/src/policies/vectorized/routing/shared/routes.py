"""Batched route-sequence operators shared by the vectorised policies.

A batch of VRPP solutions is a **route sequence** tensor ``S`` of shape ``[B, L]``:
``0 r1 0 r2 0 ... rk 0`` padded with zeros, where every ``ri`` is a depot-to-depot route
and ``L = 2 N + 2`` (``N`` bins) so a new route always fits. Depot zeros sit at both
ends of every route, so the detour, insertion and capacity rules below are exactly the
route-based rules of the simulator's operators
(:mod:`logic.src.policies.helpers.operators`): removal savings use the route
neighbours, an insertion between two consecutive depot zeros opens a new route, and a
route's load is checked against the vehicle capacity.

Attributes:
    empty_routes: Route sequences with no route.
    normalize: Drop empty routes and pack every row.
    seq_cost / seq_revenue / seq_profit: Objective terms of route sequences.
    delete_positions: Remove the bins at the given positions.
    insert_nodes: Insert one bin per row at a given gap.
    random_removal / worst_removal / shaw_removal / string_removal / cluster_removal /
    neighbor_removal: Destroy operators (cost or profit flavour).
    greedy_insertion / regret_insertion / savings_insertion: Repair operators (cost or
        profit flavour).
    LS_OPERATORS: Operator keys of the simulator's random local search.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .vrpp import VRPPBatch

BIG = 1e12


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def width(n_bins: int) -> int:
    """Route-sequence width for ``n_bins`` bins."""
    return 2 * n_bins + 2


def empty_routes(batch: int, n_bins: int, device: torch.device) -> torch.Tensor:
    """Route sequences with no route (all zeros)."""
    return torch.zeros(batch, width(n_bins), dtype=torch.long, device=device)


def _stable_keep(S: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    L = S.size(1)
    key = (~keep).long() * (L + 1) + torch.arange(L, device=S.device).unsqueeze(0)
    order = torch.argsort(key, dim=1)
    out = torch.gather(S, 1, order)
    n_keep = keep.sum(dim=1, keepdim=True)
    pos = torch.arange(L, device=S.device).unsqueeze(0)
    return torch.where(pos < n_keep, out, torch.zeros_like(out))


def normalize(S: torch.Tensor) -> torch.Tensor:
    """Drop empty routes (repeated zeros) and pack every row to the left.

    Args:
        S: Route sequences ``[B, L]``.

    Returns:
        torch.Tensor: Normalised route sequences (row starts with the depot).
    """
    prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)
    keep = (S != 0) | (prev != 0)
    out = _stable_keep(S, keep)
    # Ensure a leading depot.
    lead_missing = out[:, 0] != 0
    if bool(lead_missing.any()):
        shifted = torch.cat([torch.zeros_like(out[:, :1]), out[:, :-1]], dim=1)
        out = torch.where(lead_missing.unsqueeze(1), shifted, out)
    return out


def from_route_lists(routes: List[List[List[int]]], n_bins: int, device: torch.device) -> torch.Tensor:
    """Route sequences from Python route lists.

    Args:
        routes: Route lists, one per instance.
        n_bins: Number of bins.
        device: Target device.

    Returns:
        torch.Tensor: Route sequences ``[B, L]``.
    """
    S = empty_routes(len(routes), n_bins, device)
    for b, rl in enumerate(routes):
        seq = [0]
        for r in rl:
            if r:
                seq.extend(int(n) for n in r)
                seq.append(0)
        S[b, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return S


def to_route_lists(S: torch.Tensor) -> List[List[List[int]]]:
    """Python route lists from route sequences.

    Args:
        S: Route sequences ``[B, L]``.

    Returns:
        List of route lists.
    """
    out: List[List[List[int]]] = []
    for row in S.cpu().tolist():
        routes: List[List[int]] = []
        cur: List[int] = []
        for n in row:
            if n == 0:
                if cur:
                    routes.append(cur)
                    cur = []
            else:
                cur.append(n)
        if cur:
            routes.append(cur)
        out.append(routes)
    return out


def from_giant_tours(tours: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Route sequences from packed giant tours, cutting a new route when the load would exceed capacity."""
    B, N = tours.shape
    rows = tours.cpu().tolist()
    w = inst.waste.cpu().tolist()
    cap = inst.capacity.cpu().tolist()
    routes = []
    for b in range(B):
        rl: List[List[int]] = []
        cur: List[int] = []
        load = 0.0
        for n in rows[b]:
            if n == 0:
                continue
            if cur and load + w[b][n] > cap[b] + 1e-9:
                rl.append(cur)
                cur, load = [], 0.0
            cur.append(n)
            load += w[b][n]
        if cur:
            rl.append(cur)
        routes.append(rl)
    return from_route_lists(routes, N, tours.device)


def to_giant_tours(S: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Packed giant tours (visited bins in route order) from route sequences."""
    keep = S != 0
    out = _stable_keep(S, keep)
    return out[:, :n_bins]


def _bidx(S: torch.Tensor) -> torch.Tensor:
    return torch.arange(S.size(0), device=S.device).unsqueeze(1)


def seq_cost(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Total distance of route sequences ``[B]``."""
    return inst.dist[_bidx(S), S[:, :-1], S[:, 1:]].sum(dim=1)


def seq_revenue(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Collected revenue ``R * waste`` of route sequences ``[B]``."""
    return torch.gather(inst.waste, 1, S).sum(dim=1) * inst.R


def seq_profit(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Profit ``R * waste - C * distance`` of route sequences ``[B]``."""
    return seq_revenue(S, inst) - inst.C * seq_cost(S, inst)


def route_ids(S: torch.Tensor) -> torch.Tensor:
    """Route index of every position (a depot zero opens the route that follows it)."""
    return torch.cumsum((S == 0).long(), dim=1) - 1


def route_loads(S: torch.Tensor, inst: VRPPBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route ids per position and load per route id ``([B, L], [B, L])``."""
    rid = route_ids(S)
    w = torch.gather(inst.waste, 1, S)
    loads = torch.zeros(S.shape, device=S.device, dtype=w.dtype).scatter_add_(1, rid.clamp(min=0), w)
    return rid, loads


def prune_unprofitable_routes(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Vectorised ``prune_unprofitable_routes``: drop every route with profit < -1e-4.

    Args:
        S: Route sequences.
        inst: Instances.

    Returns:
        torch.Tensor: Route sequences without the loss-making routes.
    """
    rid = route_ids(S).clamp(min=0)
    b = _bidx(S)
    nxt = torch.roll(S, -1, dims=1)
    nxt[:, -1] = 0
    leg = inst.dist[b, S, nxt] * ((S != 0) | (nxt != 0)).to(inst.dist.dtype)
    gain = torch.gather(inst.waste, 1, S) * inst.R - inst.C * leg
    route_profit = torch.zeros(S.shape, device=S.device, dtype=gain.dtype).scatter_add_(1, rid, gain)
    drop = (torch.gather(route_profit, 1, rid) < -1e-4) & (S != 0)
    return delete_positions(S, drop) if bool(drop.any()) else S


def visited(S: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Boolean mask ``[B, M]`` of the bins present in ``S`` (depot excluded)."""
    m = torch.zeros(S.size(0), n_nodes, dtype=torch.bool, device=S.device)
    m.scatter_(1, S, True)
    m[:, 0] = False
    return m


def n_visited(S: torch.Tensor) -> torch.Tensor:
    """Number of bins per row ``[B]``."""
    return (S != 0).sum(dim=1)


def neighbours(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Previous and next node of every position (depot outside the row)."""
    prev = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], dim=1)
    nxt = torch.cat([S[:, 1:], torch.zeros_like(S[:, :1])], dim=1)
    return prev, nxt


def removal_detour(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Distance saved by removing the bin at each position (0 at depot positions)."""
    prev, nxt = neighbours(S)
    b = _bidx(S)
    d = inst.dist[b, prev, S] + inst.dist[b, S, nxt] - inst.dist[b, prev, nxt]
    return torch.where(S != 0, d, torch.zeros_like(d))


def delete_positions(S: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """Remove the bins at the masked positions and drop routes left empty."""
    keep = ~(pos_mask & (S != 0))
    return normalize(_stable_keep(S, keep))


def delete_nodes(S: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Remove the masked nodes ``[B, M]`` wherever they are."""
    return delete_positions(S, torch.gather(node_mask, 1, S))


def insert_nodes(S: torch.Tensor, node: torch.Tensor, gap: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Insert ``node[b]`` right after position ``gap[b]`` for the active rows.

    Args:
        S: Route sequences ``[B, L]``.
        node: Bin to insert per row ``[B]``.
        gap: Position after which to insert ``[B]``.
        active: Rows to modify ``[B]``.

    Returns:
        torch.Tensor: Updated route sequences.
    """
    L = S.size(1)
    j = torch.arange(L, device=S.device).unsqueeze(0)
    g = gap.unsqueeze(1)
    src = torch.where(j <= g, j, j - 1).clamp(min=0)
    out = torch.gather(S, 1, src)
    out = torch.where(j == g + 1, node.unsqueeze(1), out)
    out = torch.where(active.unsqueeze(1), out, S)
    # Rows keep at least one slack zero, so a bin inserted between trailing zeros is
    # followed by a depot zero and forms a closed route.
    return normalize(out)


# ---------------------------------------------------------------------------
# Insertion evaluation
# ---------------------------------------------------------------------------


def _gap_tables(S: torch.Tensor, inst: VRPPBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per gap ``k`` (between positions k and k+1): validity, new-route flag, route id and load."""
    rid, loads = route_loads(S, inst)
    a, b = S[:, :-1], S[:, 1:]
    is_zero_pair = (a == 0) & (b == 0)
    # Only the first (0, 0) gap is kept: it opens a new route.
    first_pair = is_zero_pair & (torch.cumsum(is_zero_pair.long(), dim=1) == 1)
    valid = (~is_zero_pair) | first_pair
    g_rid = rid[:, :-1]
    g_load = torch.gather(loads, 1, g_rid.clamp(min=0))
    g_load = torch.where(first_pair, torch.zeros_like(g_load), g_load)
    return valid, first_pair, g_rid, g_load


def insertion_costs(
    S: torch.Tensor, cand: torch.Tensor, inst: VRPPBatch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insertion detours of candidate bins at every gap.

    Args:
        S: Route sequences ``[B, L]``.
        cand: Candidate bins ``[B, K]`` (0 = no candidate).
        inst: Instances.

    Returns:
        Tuple of (detour ``[B, K, L-1]``, capacity-feasible mask, new-route gap mask ``[B, L-1]``,
        gap route ids ``[B, L-1]``).
    """
    valid, newr, g_rid, g_load = _gap_tables(S, inst)
    b = torch.arange(S.size(0), device=S.device).view(-1, 1, 1)
    a = S[:, :-1].unsqueeze(1)
    nx = S[:, 1:].unsqueeze(1)
    c = cand.unsqueeze(2)
    detour = inst.dist[b, a, c] + inst.dist[b, c, nx] - inst.dist[b, a, nx]
    w = torch.gather(inst.waste, 1, cand).unsqueeze(2)
    feas = valid.unsqueeze(1) & (g_load.unsqueeze(1) + w <= inst.capacity.view(-1, 1, 1) + 1e-9) & (c != 0)
    return detour, feas, newr, g_rid


def _seed_hurdle(cand: torch.Tensor, unassigned: torch.Tensor, inst: VRPPBatch, new_cost: torch.Tensor) -> torch.Tensor:
    """Vectorised ``_dynamic_seed_hurdle`` of the simulator (no mandatory bins)."""
    b = torch.arange(cand.size(0), device=cand.device).unsqueeze(1)
    M = inst.dist.size(1)
    n_total = max(M - 1, 1)
    d0 = inst.dist[:, 0, :]  # [B, M]
    # synergy[b, k, v] = (d(0,c) + d(0,v) - d(c,v)) * C over unassigned v != c
    dcv = inst.dist[b.unsqueeze(2), cand.unsqueeze(2), torch.arange(M, device=cand.device).view(1, 1, -1)]
    syn = (d0[b, cand].unsqueeze(2) + d0.unsqueeze(1) - dcv) * inst.C
    others = unassigned.unsqueeze(1) & (torch.arange(M, device=cand.device).view(1, 1, -1) != cand.unsqueeze(2))
    cnt = others.sum(dim=2).clamp(min=0)
    mean_syn = torch.where(others, syn, torch.zeros_like(syn)).sum(dim=2) / cnt.clamp(min=1)
    mean_syn = torch.where(cnt > 0, mean_syn, torch.zeros_like(mean_syn))
    rho = cnt.float() / n_total
    hurdle = -rho * mean_syn.clamp(min=0)
    return torch.maximum(hurdle, -new_cost * inst.C).clamp(max=0.0)


def _candidates(mask: torch.Tensor) -> torch.Tensor:
    """Sorted candidate bin ids ``[B, K]`` (0-padded) from a node mask ``[B, M]``."""
    M = mask.size(1)
    ids = torch.arange(M, device=mask.device).unsqueeze(0).expand_as(mask)
    key = torch.where(mask, ids, ids + M)
    srt = torch.sort(key, dim=1).values
    out = torch.where(srt < M, srt, torch.zeros_like(srt))
    K = int(mask.sum(dim=1).max().item()) if mask.numel() else 0
    return out[:, : max(K, 1)]


def greedy_insertion(
    S: torch.Tensor,
    pending: torch.Tensor,
    inst: VRPPBatch,
    profit_aware: bool,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Greedy (cheapest / most profitable) insertion of the pending bins.

    Cost flavour (``greedy_insertion``): insert the bin with the cheapest feasible
    insertion; when no pending bin fits an existing route, open a new route for the
    lowest-id pending bin. Profit flavour (``greedy_profit_insertion``): insert the bin
    with the largest insertion profit ``R w - C detour`` (>= -1e-4); a new route is
    opened when its profit beats every existing slot and clears the dynamic seed hurdle;
    bins with no admissible slot stay unvisited.

    Args:
        S: Route sequences.
        pending: Bins to insert ``[B, M]``.
        inst: Instances.
        profit_aware: Use the profit flavour.
        noise: Additive detour noise per row ``[B]`` (``max(0, detour + noise)``).

    Returns:
        torch.Tensor: Repaired route sequences.
    """
    pending = pending.clone()
    pending[:, 0] = False
    B = S.size(0)
    noise_v = noise.view(-1, 1, 1) if noise is not None else None
    for _ in range(int(pending.sum(dim=1).max().item()) if pending.numel() else 0):
        if not bool(pending.any()):
            break
        cand = _candidates(pending)
        detour, feas, newr, _ = insertion_costs(S, cand, inst)
        if noise_v is not None:
            detour = (detour + noise_v).clamp(min=0.0)
        b = torch.arange(B, device=S.device)
        if not profit_aware:
            existing = feas & ~newr.unsqueeze(1)
            score = torch.where(existing, detour, torch.full_like(detour, BIG))
            flat = score.flatten(1)
            best, arg = flat.min(dim=1)
            has = best < BIG / 2
            k_idx = arg // score.size(2)
            g_idx = arg % score.size(2)
            node = cand[b, k_idx]
            # No existing route fits any pending bin: new route for the lowest id bin.
            new_gap = torch.argmax(newr.long(), dim=1)
            fallback = cand[:, 0]
            active = pending.any(dim=1)
            node = torch.where(has, node, fallback)
            gap = torch.where(has, g_idx, new_gap)
            ok_new = torch.gather(inst.waste, 1, fallback.unsqueeze(1)).squeeze(1) <= inst.capacity + 1e-9
            active = active & (has | ok_new)
            S = insert_nodes(S, node, gap, active)
            done = torch.zeros_like(pending)
            done[b, node] = True
            # Bins that can never be inserted are dropped.
            drop = (~has) & ~ok_new & pending.any(dim=1)
            done[b, fallback] = done[b, fallback] | drop
            pending = pending & ~(done & (active | drop).unsqueeze(1))
        else:
            w = torch.gather(inst.waste, 1, cand)
            rev = (w * inst.R).unsqueeze(2)
            prof = rev - inst.C * detour
            existing = feas & ~newr.unsqueeze(1) & (prof >= -1e-4)
            ex_score = torch.where(existing, prof, torch.full_like(prof, -BIG))
            # New route option.
            new_cost = inst.dist[b.unsqueeze(1), 0, cand] + inst.dist[b.unsqueeze(1), cand, 0]
            if noise is not None:
                new_cost = (new_cost + noise.view(-1, 1)).clamp(min=0.0)
            new_prof = w * inst.R - inst.C * new_cost
            hurdle = _seed_hurdle(cand, pending, inst, new_cost)
            new_ok = (new_prof >= hurdle) & (w <= inst.capacity.view(-1, 1) + 1e-9) & (cand != 0)
            new_score = torch.where(new_ok, new_prof, torch.full_like(new_prof, -BIG))
            best_ex, arg_ex = ex_score.max(dim=2)
            use_new = new_score > best_ex
            node_score = torch.where(use_new, new_score, best_ex)
            best, k_idx = node_score.max(dim=1)
            has = best > -BIG / 2
            node = cand[b, k_idx]
            new_gap = torch.argmax(newr.long(), dim=1)
            gap = torch.where(use_new[b, k_idx], new_gap, arg_ex[b, k_idx])
            active = has & pending.any(dim=1)
            S = insert_nodes(S, node, gap, active)
            done = torch.zeros_like(pending)
            done[b, node] = True
            pending = pending & ~(done & active.unsqueeze(1))
            # Rows with nothing admissible stop: their remaining bins stay unvisited.
            pending = pending & has.unsqueeze(1)
    return prune_unprofitable_routes(S, inst) if profit_aware else S


def regret_insertion(
    S: torch.Tensor,
    pending: torch.Tensor,
    inst: VRPPBatch,
    k: int,
    profit_aware: bool,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Regret-k insertion of the pending bins (cost or profit flavour).

    Per bin, the best slot of every route is ranked; the bin with the most missing
    routes, then the largest regret ``sum_{h=2..k} |c_h - c_1|``, then the best value,
    is inserted at its best slot. Bins with no admissible slot are left unvisited. The
    profit flavour also offers a new route (seed hurdle) as an option.

    Args:
        S: Route sequences.
        pending: Bins to insert ``[B, M]``.
        inst: Instances.
        k: Regret depth.
        profit_aware: Use the profit flavour.
        noise: Additive detour noise per row.

    Returns:
        torch.Tensor: Repaired route sequences.
    """
    pending = pending.clone()
    pending[:, 0] = False
    B = S.size(0)
    L = S.size(1)
    for _ in range(int(pending.sum(dim=1).max().item()) if pending.numel() else 0):
        if not bool(pending.any()):
            break
        cand = _candidates(pending)
        K = cand.size(1)
        detour, feas, newr, g_rid = insertion_costs(S, cand, inst)
        if noise is not None:
            detour = (detour + noise.view(-1, 1, 1)).clamp(min=0.0)
        b = torch.arange(B, device=S.device)
        w = torch.gather(inst.waste, 1, cand)
        if profit_aware:
            val = (w * inst.R).unsqueeze(2) - inst.C * detour  # maximise
            ok = feas & ~newr.unsqueeze(1) & (val >= -1e-4)
            sval = torch.where(ok, -val, torch.full_like(val, BIG))  # minimise -profit
        else:
            ok = feas & ~newr.unsqueeze(1)
            sval = torch.where(ok, detour, torch.full_like(detour, BIG))
        # Best slot per route: reduce over gaps with the same route id.
        rid = g_rid.clamp(min=0).unsqueeze(1).expand(-1, K, -1)
        per_route = torch.full((B, K, L), BIG, device=S.device)
        per_route = per_route.scatter_reduce(2, rid, sval, reduce="amin", include_self=True)
        new_opt: Optional[torch.Tensor] = None
        if profit_aware:
            new_cost = inst.dist[b.unsqueeze(1), 0, cand] + inst.dist[b.unsqueeze(1), cand, 0]
            if noise is not None:
                new_cost = (new_cost + noise.view(-1, 1)).clamp(min=0.0)
            new_prof = w * inst.R - inst.C * new_cost
            hurdle = _seed_hurdle(cand, pending, inst, new_cost)
            new_ok = (new_prof >= hurdle) & (w <= inst.capacity.view(-1, 1) + 1e-9) & (cand != 0)
            new_opt = torch.where(new_ok, -new_prof, torch.full_like(new_prof, BIG))
            per_route = torch.cat([per_route, new_opt.unsqueeze(2)], dim=2)
        srt = torch.sort(per_route, dim=2).values
        n_opt = (srt < BIG / 2).sum(dim=2)
        missing = (k - n_opt).clamp(min=0)
        kk = min(k, srt.size(2))
        diffs = srt[:, :, 1:kk] - srt[:, :, :1]
        diffs = torch.where(srt[:, :, 1:kk] < BIG / 2, diffs, torch.zeros_like(diffs))
        regret = diffs.sum(dim=2)
        best_val = srt[:, :, 0]
        has_opt = (n_opt > 0) & (cand != 0)
        # Lexicographic key: missing routes, regret, then the best value (smaller is better).
        key = missing.double() * 1e15 + regret.double().clamp(max=1e14) * 1e0
        key = key - best_val.double() * 1e-6
        key = torch.where(has_opt, key, torch.full_like(key, -float("inf")))
        _, k_idx = key.max(dim=1)
        has = has_opt.any(dim=1)
        node = cand[b, k_idx]
        # Best gap for that bin (existing route) or the new-route gap.
        node_slots = sval[b, k_idx]  # [B, L-1]
        best_slot_val, best_gap = node_slots.min(dim=1)
        gap = best_gap
        if profit_aware:
            assert new_opt is not None
            nv = new_opt[b, k_idx]
            use_new = nv < best_slot_val
            gap = torch.where(use_new, torch.argmax(newr.long(), dim=1), best_gap)
        active = has & pending.any(dim=1)
        S = insert_nodes(S, node, gap, active)
        done = torch.zeros_like(pending)
        done[b, node] = True
        # Bins without any admissible slot leave the pool.
        no_opt_nodes = torch.zeros_like(pending)
        no_opt_nodes.scatter_(1, cand, (~has_opt) & (cand != 0))
        pending = pending & ~(done & active.unsqueeze(1)) & ~no_opt_nodes
        pending = pending & has.unsqueeze(1)
    return prune_unprofitable_routes(S, inst) if profit_aware else S


# ---------------------------------------------------------------------------
# Destroy operators
# ---------------------------------------------------------------------------


def _uniform(shape, gen: Optional[torch.Generator], device: torch.device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, device=device)


def random_removal(
    S: torch.Tensor, n: torch.Tensor, gen: Optional[torch.Generator]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove ``n[b]`` bins chosen uniformly at random.

    Returns:
        Tuple of (partial route sequences, removed node mask ``[B, M]`` built by the caller).
    """
    r = _uniform(S.shape, gen, S.device)
    r = torch.where(S != 0, r, torch.full_like(r, 2.0))
    rank = torch.argsort(torch.argsort(r, dim=1), dim=1)
    pos = rank < n.unsqueeze(1)
    return delete_positions(S, pos), pos & (S != 0)


def _removed_nodes(S: torch.Tensor, pos: torch.Tensor, n_nodes: int) -> torch.Tensor:
    m = torch.zeros(S.size(0), n_nodes, dtype=torch.bool, device=S.device)
    m.scatter_(1, torch.where(pos, S, torch.zeros_like(S)), True)
    m[:, 0] = False
    return m


def worst_removal(
    S: torch.Tensor,
    n: torch.Tensor,
    inst: VRPPBatch,
    p: float,
    gen: Optional[torch.Generator],
    profit_aware: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomised worst removal (Ropke & Pisinger 2006): pick rank ``floor(y^p |L|)``.

    Cost flavour ranks by largest detour saving; profit flavour by lowest marginal
    profit ``R w - C detour``. Savings are recomputed after every removal.

    Returns:
        Tuple of (partial route sequences, removed node mask ``[B, M]``).
    """
    M = inst.dist.size(1)
    removed = torch.zeros(S.size(0), M, dtype=torch.bool, device=S.device)
    for t in range(int(n.max().item()) if n.numel() else 0):
        act = (t < n) & (n_visited(S) > 0)
        if not bool(act.any()):
            break
        det = removal_detour(S, inst)
        # Ascending order: least profitable (profit-aware) or largest saving first.
        score = torch.gather(inst.waste, 1, S) * inst.R - inst.C * det if profit_aware else -det
        score = torch.where(S != 0, score, torch.full_like(score, BIG))
        order = torch.argsort(score, dim=1, stable=True)
        cnt = n_visited(S)
        y = _uniform((S.size(0),), gen, S.device)
        idx = torch.minimum((y**p * cnt).long(), (cnt - 1).clamp(min=0))
        pos_idx = order[torch.arange(S.size(0), device=S.device), idx]
        pos = torch.zeros_like(S, dtype=torch.bool)
        pos[torch.arange(S.size(0), device=S.device), pos_idx] = True
        pos = pos & act.unsqueeze(1)
        removed |= _removed_nodes(S, pos, M)
        S = delete_positions(S, pos)
    return S, removed


def shaw_removal(
    S: torch.Tensor,
    n: torch.Tensor,
    inst: VRPPBatch,
    randomization: float,
    gen: Optional[torch.Generator],
    profit_aware: bool,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shaw (relatedness) removal: random seed, then the rank ``floor(y^r |L|)`` most
    related bin to a random already-removed pivot.

    Relatedness is ``phi d/d_max + chi |w_i - w_j|/w_range`` (cost flavour) or
    ``phi d/d_max + psi |pi_i - pi_j|/pi_range`` with marginal profits ``pi``
    (profit flavour).

    Returns:
        Tuple of (partial route sequences, removed node mask ``[B, M]``).
    """
    B, M = S.size(0), inst.dist.size(1)
    dev = S.device
    b = torch.arange(B, device=dev)
    inroute = visited(S, M)
    dmax = inst.dist.flatten(1).max(dim=1).values.clamp(min=1e-9)
    if profit_aware:
        det = removal_detour(S, inst)
        prof_pos = torch.gather(inst.waste, 1, S) * inst.R - inst.C * det
        attr = torch.zeros(B, M, device=dev).scatter_(1, S, prof_pos)
        wmax = inst.waste.max(dim=1).values * inst.R
        wmin = inst.waste[:, 1:].min(dim=1).values * inst.R if M > 1 else torch.zeros(B, device=dev)
        rng_ = (wmax - (wmin - 2 * dmax * inst.C)).clamp(min=1e-9)
        coef = psi
    else:
        attr = inst.waste
        rng_ = (
            (inst.waste[:, 1:].max(dim=1).values - inst.waste[:, 1:].min(dim=1).values).clamp(min=1e-9)
            if M > 1
            else torch.ones(B, device=dev)
        )
        coef = chi
    removed = torch.zeros(B, M, dtype=torch.bool, device=dev)
    # Seed.
    r = _uniform((B, M), gen, dev)
    r = torch.where(inroute, r, torch.full_like(r, -1.0))
    seed = r.argmax(dim=1)
    act0 = (n > 0) & inroute.any(dim=1)
    removed[b, seed] = act0
    for t in range(1, int(n.max().item()) if n.numel() else 0):
        act = (t < n) & ((inroute & ~removed).any(dim=1))
        if not bool(act.any()):
            break
        pr = _uniform((B, M), gen, dev)
        pr = torch.where(removed, pr, torch.full_like(pr, -1.0))
        pivot = pr.argmax(dim=1)
        rel = coef * (attr - attr[b, pivot].unsqueeze(1)).abs() / rng_.unsqueeze(1)
        rel = rel + phi * inst.dist[b, :, pivot] / dmax.unsqueeze(1)
        cand = inroute & ~removed
        rel = torch.where(cand, rel, torch.full_like(rel, BIG))
        order = torch.argsort(rel, dim=1, stable=True)
        cnt = cand.sum(dim=1)
        y = _uniform((B,), gen, dev)
        idx = torch.minimum((y**randomization * cnt).long(), (cnt - 1).clamp(min=0))
        pick = order[b, idx]
        removed[b, pick] = removed[b, pick] | act
    return delete_nodes(S, removed), removed


def _row_seeds(B: int, gen: Optional[torch.Generator], device: torch.device) -> List[int]:
    return torch.randint(0, 2**62, (B,), generator=gen, device=device).tolist()


def _per_row_ruin(
    S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch, gen: Optional[torch.Generator], fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a simulator ruin operator ``fn(routes, n, dist, waste, rng)`` row by row."""
    import random as _random

    B, M = S.size(0), inst.dist.size(1)
    routes = to_route_lists(S)
    dist = inst.dist.detach().cpu().double().numpy()
    waste = inst.waste.detach().cpu().double().numpy()
    seeds = _row_seeds(B, gen, S.device)
    nn = n.tolist()
    out: List[List[List[int]]] = []
    removed = torch.zeros(B, M, dtype=torch.bool, device=S.device)
    for b in range(B):
        if nn[b] <= 0 or not any(routes[b]):
            out.append(routes[b])
            continue
        w = {i: float(waste[b, i]) for i in range(1, M)}
        new, rem = fn([r[:] for r in routes[b]], int(nn[b]), dist[b], w, _random.Random(seeds[b]))
        out.append(new)
        if rem:
            removed[b, torch.tensor(rem, dtype=torch.long, device=S.device)] = True
    return from_route_lists(out, M - 1, S.device), removed


def string_removal(
    S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch, gen: Optional[torch.Generator], profit_aware: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SISR string removal (``string_removal`` / ``string_profit_removal``).

    The SISR ruin (Christiaens & Vanden Berghe 2020) is a sequential, data-dependent
    walk (string lengths, the split-string geometric draw, tours already ruined), so it
    runs row by row with the simulator's own operator and a per-row ``random.Random``
    seeded from ``gen``.
    """
    from logic.src.policies.helpers.operators.destroy_ruin.string import string_profit_removal
    from logic.src.policies.helpers.operators.destroy_ruin.string import string_removal as _sr

    R, C = float(inst.R), float(inst.C)
    if profit_aware:
        return _per_row_ruin(
            S, n, inst, gen, lambda r, k, d, w, rng: string_profit_removal(r, k, d, wastes=w, R=R, C=C, rng=rng)
        )
    return _per_row_ruin(S, n, inst, gen, lambda r, k, d, w, rng: _sr(r, k, d, rng=rng))


def _node_profits(S: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
    """Marginal profit ``R w - C detour`` of every visited bin ``[B, M]`` (+inf elsewhere)."""
    M = inst.dist.size(1)
    prof_pos = torch.gather(inst.waste, 1, S) * inst.R - inst.C * removal_detour(S, inst)
    out = torch.full((S.size(0), M), float("inf"), device=S.device, dtype=prof_pos.dtype)
    out.scatter_(1, torch.where(S != 0, S, torch.zeros_like(S)), torch.where(S != 0, prof_pos, out[:, :1].expand_as(S)))
    out[:, 0] = float("inf")
    return out


def _profit_similarity_removal(
    S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch, gen: Optional[torch.Generator]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``neighbor_profit_removal`` / ``cluster_profit_removal`` (same selection rule).

    Bins sorted by (marginal profit, id); the seed is drawn uniformly from the bottom
    quartile (at least one); then the ``n - 1`` bins with the closest marginal profit to
    the seed's (ties by id) are removed with it.
    """
    B, M = S.size(0), inst.dist.size(1)
    dev = S.device
    b = torch.arange(B, device=dev)
    inroute = visited(S, M)
    prof = _node_profits(S, inst)
    order = torch.argsort(prof, dim=1, stable=True)  # ids ascending within ties
    cnt = inroute.sum(dim=1)
    bq = (cnt // 4).clamp(min=1)
    pick = (torch.rand(B, generator=gen, device=dev) * bq).long().clamp(max=(bq - 1).clamp(min=0))
    seed = order[b, pick]
    diff = (prof - prof[b, seed].unsqueeze(1)).abs()
    diff = torch.where(
        inroute & (torch.arange(M, device=dev).unsqueeze(0) != seed.unsqueeze(1)),
        diff,
        torch.full_like(diff, float("inf")),
    )
    rank = torch.argsort(torch.argsort(diff, dim=1, stable=True), dim=1)
    removed = inroute & (rank < (n - 1).clamp(min=0).unsqueeze(1))
    removed[b, seed] = True
    act = (cnt > 0) & (n > 0)
    removed = removed & act.unsqueeze(1)
    return delete_nodes(S, removed), removed


def cluster_removal(
    S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch, gen: Optional[torch.Generator], profit_aware: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cluster removal (``cluster_removal`` / ``cluster_profit_removal``).

    Cost flavour (Shaw-style, Kruskal two-cluster splits of a random route, then of the
    route of the nearest unremoved bin to a random removed pivot): a sequential,
    data-dependent procedure run row by row with the simulator's operator and a per-row
    ``random.Random`` seeded from ``gen``. Profit flavour: batched (see
    :func:`_profit_similarity_removal`).
    """
    if profit_aware:
        return _profit_similarity_removal(S, n, inst, gen)
    from logic.src.policies.helpers.operators.destroy_ruin.cluster import cluster_removal as _cr

    nodes = list(range(1, inst.dist.size(1)))
    return _per_row_ruin(S, n, inst, gen, lambda r, k, d, w, rng: _cr(r, k, d, nodes=nodes, rng=rng))


def neighbor_removal(
    S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch, gen: Optional[torch.Generator], profit_aware: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Neighbour removal (``neighbor_removal`` / ``neighbor_profit_removal``).

    Cost flavour: a uniformly random visited seed and the ``min(n, |visited|)`` visited
    bins closest to it (the seed included). Profit flavour: see
    :func:`_profit_similarity_removal`.
    """
    if profit_aware:
        return _profit_similarity_removal(S, n, inst, gen)
    B, M = S.size(0), inst.dist.size(1)
    dev = S.device
    b = torch.arange(B, device=dev)
    inroute = visited(S, M)
    r = torch.rand(B, M, generator=gen, device=dev)
    seed = torch.where(inroute, r, torch.full_like(r, -1.0)).argmax(dim=1)
    d = torch.where(inroute, inst.dist[b, seed], torch.full((B, M), float("inf"), device=dev))
    d[b, seed] = torch.where(inroute[b, seed], torch.full_like(d[b, seed], -1.0), d[b, seed])
    rank = torch.argsort(torch.argsort(d, dim=1, stable=True), dim=1)
    removed = inroute & (rank < n.unsqueeze(1))
    return delete_nodes(S, removed), removed


def savings_insertion(S: torch.Tensor, pending: torch.Tensor, inst: VRPPBatch, profit_aware: bool) -> torch.Tensor:
    """Savings repair (``savings_insertion`` / ``savings_profit_insertion``).

    Every step inserts the (bin, slot) of largest saving ``d(0,i) + d(i,0) - detour``
    over the capacity-feasible slots of the existing routes (first in bin-id, route,
    position order on ties).

    Cost flavour: when no slot fits, the lowest-id pending bin opens a new route (or is
    dropped when it exceeds the capacity). Profit flavour: only slots with insertion
    profit ``R w - C detour >= -1e-4`` count, and opening a new route scores 0 for a bin
    whose round-trip profit clears the dynamic seed hurdle, compared in the same
    iteration order; the loop stops when nothing qualifies and the result goes through
    ``prune_unprofitable_routes``.
    """
    pending = pending.clone()
    pending[:, 0] = False
    B = S.size(0)
    b = torch.arange(B, device=S.device)
    for _ in range(int(pending.sum(dim=1).max().item()) if pending.numel() else 0):
        if not bool(pending.any()):
            break
        cand = _candidates(pending)
        detour, feas, newr, _ = insertion_costs(S, cand, inst)
        ded = inst.dist[b.unsqueeze(1), 0, cand] + inst.dist[b.unsqueeze(1), cand, 0]
        saving = ded.unsqueeze(2) - detour
        existing = feas & ~newr.unsqueeze(1)
        new_gap = torch.argmax(newr.long(), dim=1)
        w = torch.gather(inst.waste, 1, cand)
        if not profit_aware:
            score = torch.where(existing, saving, torch.full_like(saving, -BIG))
            flat = score.flatten(1)
            best = flat.max(dim=1).values
            arg = flat.argmax(dim=1)
            has = best > -BIG / 2
            node = torch.where(has, cand[b, arg // score.size(2)], cand[:, 0])
            gap = torch.where(has, arg % score.size(2), new_gap)
            ok_new = torch.gather(inst.waste, 1, cand[:, :1]).squeeze(1) <= inst.capacity
            act = pending.any(dim=1) & (has | ok_new)
            S = insert_nodes(S, node, gap, act)
            done = torch.zeros_like(pending)
            done[b, node] = True
            pending = pending & ~(done & pending.any(dim=1).unsqueeze(1))
        else:
            prof = (w * inst.R).unsqueeze(2) - inst.C * detour
            ok = existing & (prof >= -1e-4)
            score = torch.where(ok, saving, torch.full_like(saving, -BIG))
            new_profit = w * inst.R - ded * inst.C
            hurdle = _seed_hurdle(cand, pending, inst, ded)
            new_ok = (new_profit >= hurdle) & (cand != 0)
            new_score = torch.where(new_ok, torch.zeros_like(new_profit), torch.full_like(new_profit, -BIG))
            full = torch.cat([score, new_score.unsqueeze(2)], dim=2)  # per bin: slots, then new route
            flat = full.flatten(1)
            best = flat.max(dim=1).values
            arg = flat.argmax(dim=1)
            has = (best > -BIG / 2) & pending.any(dim=1)
            k = arg // full.size(2)
            g = arg % full.size(2)
            use_new = g == score.size(2)
            node = cand[b, k]
            gap = torch.where(use_new, new_gap, g)
            S = insert_nodes(S, node, gap, has)
            done = torch.zeros_like(pending)
            done[b, node] = True
            pending = pending & ~(done & has.unsqueeze(1))
            pending = pending & has.unsqueeze(1)
    return prune_unprofitable_routes(S, inst) if profit_aware else S


# ---------------------------------------------------------------------------
# Local search operator names (see ``sweeps`` and ``pair_ls``)
# ---------------------------------------------------------------------------

LS_OPERATORS = (
    "two_opt",
    "swap",
    "relocate",
    "two_opt_star",
    "swap_star",
    "three_opt",
    "or_opt",
    "relocate_chain",
    "cross_exchange",
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def greedy_profit_construction(inst: VRPPBatch, gen: Optional[torch.Generator]) -> torch.Tensor:
    """Vectorised ``build_greedy_routes`` (no mandatory bins).

    Bins are taken in a random order; each goes to its most profitable feasible slot
    (insertion profit >= -1e-4) or to a new route when that is more profitable and clears
    the dynamic seed hurdle; otherwise it stays unvisited.

    Args:
        inst: Instances.
        gen: Torch generator for the bin order.

    Returns:
        torch.Tensor: Route sequences.
    """
    B, M = inst.dist.size(0), inst.dist.size(1)
    N = M - 1
    dev = inst.device
    S = empty_routes(B, N, dev)
    if N == 0:
        return S
    order = torch.argsort(torch.rand(B, N, generator=gen, device=dev), dim=1) + 1
    b = torch.arange(B, device=dev)
    remaining = torch.ones(B, M, dtype=torch.bool, device=dev)
    remaining[:, 0] = False
    for t in range(N):
        node = order[:, t]
        remaining[b, node] = False
        cand = node.unsqueeze(1)
        detour, feas, newr, _ = insertion_costs(S, cand, inst)
        w = torch.gather(inst.waste, 1, cand)
        prof = (w * inst.R).unsqueeze(2) - inst.C * detour
        ok = feas & ~newr.unsqueeze(1) & (prof >= -1e-4)
        sc = torch.where(ok, prof, torch.full_like(prof, -BIG)).squeeze(1)
        best, gi = sc.max(dim=1)
        new_cost = inst.dist[b, 0, node] + inst.dist[b, node, 0]
        new_prof = w.squeeze(1) * inst.R - inst.C * new_cost
        hurdle = _seed_hurdle(cand, remaining, inst, new_cost.unsqueeze(1)).squeeze(1)
        use_new = (new_prof > best) & (new_prof >= hurdle) & (w.squeeze(1) <= inst.capacity + 1e-9)
        gap = torch.where(use_new, torch.argmax(newr.long(), dim=1), gi)
        active = use_new | (best > -BIG / 2)
        S = insert_nodes(S, node, gap, active)
    return prune_unprofitable_routes(S, inst)


def random_giant_tours(B: int, N: int, gen: Optional[torch.Generator], device: torch.device) -> torch.Tensor:
    """Random permutations of the bins ``[B, N]``."""
    return torch.argsort(torch.rand(B, N, generator=gen, device=device), dim=1) + 1
