"""Batched granular local search mirroring the simulator's ``LocalSearch._optimize_internal``.

The simulator's HGS and ACO local searches
(:class:`~logic.src.policies.helpers.local_search.local_search_hgs.HGSLocalSearch`,
:class:`~logic.src.policies.helpers.local_search.local_search_aco.ACOLocalSearch`) share
one loop: every iteration shuffles the candidate pairs ``(u, v in neighbors[u])``,
applies the first pair whose ``_process_pair`` finds an improving move and restarts;
the loop stops after an iteration without improvement or ``local_search_iterations``
iterations. ``_process_pair`` tries, in this order:

1. ``drop(u)``, ``drop(v)`` (profit gain, soft capacity penalty);
2. when exactly one of the two is routed, the best insertion of the unrouted one into
   the other's route (``unrouted_insert``, soft capacity penalty);
3. two routes: relocate ``u`` after ``v``, swap, 2-opt* (tail exchange; the
   ``cross`` operator is the same move), shift(2,0), swap(2,1), swap(2,2);
4. one route: relocate, swap, 2-opt, Or-opt with a chain length drawn from {1, 2, 3}.

SWAP* and 3-opt are only reached with ``target_neighborhood="inter_swap_star"`` and
``params.use_3opt`` respectively, neither of which the HGS/ACO solvers set.

Here every pair is evaluated at once with the exact deltas of the simulator operators.
The pair applied in a row is the improving pair with the smallest i.i.d. random key,
which has the same distribution as "shuffle, then first improving pair". The
simulator's per-node Top-3 insertion cache is replaced by a fresh best-insertion
computation (the value the cache is meant to hold).

Attributes:
    granular_neighbors: ``nb_granular`` nearest bins of every bin (depot excluded).
    candidate_neighbors: Neighbour table from K-sparse candidate lists.
    pair_local_search: The batched local-search loop.
"""

from __future__ import annotations

from typing import Optional

import torch

from .routes import normalize, route_ids
from .vrpp import VRPPBatch

EPS = 1e-4
INF = float("inf")


def granular_neighbors(inst: VRPPBatch, nb_granular: int) -> torch.Tensor:
    """``[B, M, K]`` neighbour table: the ``nb_granular`` closest bins (no self, no depot), -1 padded.

    Row 0 (the depot) is empty, as in ``LocalSearch.__init__``.
    """
    B, M = inst.batch_size, inst.n_bins + 1
    dev = inst.device
    d = inst.dist.clone()
    eye = torch.eye(M, dtype=torch.bool, device=dev).unsqueeze(0)
    d = d.masked_fill(eye, INF)
    d[:, :, 0] = INF
    k = max(0, min(int(nb_granular), M - 2))
    nb = torch.full((B, M, max(k, 1)), -1, dtype=torch.long, device=dev)
    if k > 0:
        order = torch.argsort(d, dim=2, stable=True)[:, :, :k]
        nb[:, :, :k] = order
    nb[:, 0, :] = -1
    return nb


def candidate_neighbors(cand: torch.Tensor) -> torch.Tensor:
    """Neighbour table from candidate lists ``[B, M, K]`` (the ACO's ``candidate_lists``, depot included)."""
    return cand.clone()


def _g(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Row-wise gather ``t[b, idx[b, ...]]`` for ``t`` of shape ``[B, X]``."""
    flat = idx.reshape(idx.size(0), -1)
    return torch.gather(t, 1, flat).reshape(idx.shape)


def _pen(load: torch.Tensor, Q: torch.Tensor, w: float) -> torch.Tensor:
    return w * (load - Q).clamp(min=0.0)


class _State:
    """Per-position and per-node tables of a batch of route sequences."""

    def __init__(self, S: torch.Tensor, inst: VRPPBatch) -> None:
        B, L = S.shape
        M = inst.n_bins + 1
        dev = S.device
        self.S = S
        j = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        isz = S == 0
        self.st = torch.cummax(torch.where(isz, j, torch.zeros_like(j)), dim=1).values
        rev = torch.where(isz, j, torch.full_like(j, L - 1)).flip(1)
        self.en = torch.cummin(rev, dim=1).values.flip(1)
        self.rid = route_ids(S).clamp(min=0)
        w = torch.gather(inst.waste, 1, S)
        self.wpos = w
        self.cumW = torch.cumsum(w, dim=1)
        loads = torch.zeros(B, L, dtype=w.dtype, device=dev).scatter_add_(1, self.rid, w)
        self.loads = loads
        self.ld = torch.gather(loads, 1, self.rid)
        pos = torch.full((B, M), -1, dtype=torch.long, device=dev)
        pos.scatter_(1, torch.where(isz, torch.zeros_like(S), S), torch.where(isz, torch.full_like(j, -1), j))
        pos[:, 0] = -1
        self.pos = pos
        bi = torch.arange(B, device=dev).unsqueeze(1)
        nxt = torch.cat([S[:, 1:], torch.zeros_like(S[:, :1])], dim=1)
        self.fwd = inst.dist[bi, S, nxt]  # edge cost S[i] -> S[i+1]
        self.cumF = torch.cat([torch.zeros_like(self.fwd[:, :1]), torch.cumsum(self.fwd, dim=1)], dim=1)
        bwd = inst.dist[bi, nxt, S]
        rd = torch.where((S != 0) & (nxt != 0), bwd - self.fwd, torch.zeros_like(bwd))
        self.cumRD = torch.cat([torch.zeros_like(rd[:, :1]), torch.cumsum(rd, dim=1)], dim=1)
        self.L = L

    def at(self, p: torch.Tensor) -> torch.Tensor:
        return _g(self.S, p.clamp(0, self.L - 1))


def _dist(inst: VRPPBatch, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B = a.size(0)
    bi = torch.arange(B, device=a.device).view(B, *([1] * (a.dim() - 1)))
    return inst.dist[bi, a, b]


def _best_insertion(st: _State, inst: VRPPBatch, node: torch.Tensor, route_pos: torch.Tensor):
    """Best insertion (min detour, first gap on ties) of ``node [B,P]`` into the route of position ``route_pos``.

    Returns (delta, gap position) with the gap being the S position after which to insert.
    """
    S, L = st.S, st.L
    B, P = node.shape
    a = S[:, :-1].unsqueeze(1).expand(B, P, L - 1)
    b = S[:, 1:].unsqueeze(1).expand(B, P, L - 1)
    x = node.unsqueeze(2).expand(B, P, L - 1)
    delta = _dist(inst, a, x) + _dist(inst, x, b) - _dist(inst, a, b)
    r_st = _g(st.st, route_pos).unsqueeze(2)
    r_en = _g(st.en, route_pos).unsqueeze(2)
    gi = torch.arange(L - 1, device=S.device).view(1, 1, -1)
    ok = (gi >= r_st) & (gi < r_en)
    delta = torch.where(ok, delta, torch.full_like(delta, INF))
    best, arg = delta.min(dim=2)  # first minimum
    return best, arg


def pair_local_search(
    S: torch.Tensor,
    inst: VRPPBatch,
    neighbors: torch.Tensor,
    max_iterations: int,
    gen: Optional[torch.Generator] = None,
    penalty_capacity: float = 1.0,
    active: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the simulator's granular local search on every row.

    Args:
        S: Route sequences ``[B, L]``.
        inst: Instances.
        neighbors: Neighbour table ``[B, M, K]`` (-1 padded).
        max_iterations: ``local_search_iterations``.
        gen: Torch generator.
        penalty_capacity: Capacity penalty weight of drop / insertion / swap moves.
        active: Rows to optimise (default all).

    Returns:
        torch.Tensor: Improved route sequences.
    """
    B, L = S.shape
    dev = S.device
    M = inst.n_bins + 1
    K = neighbors.size(2)
    if M < 2 or K == 0:
        return S
    U = torch.arange(M, device=dev).view(1, M, 1).expand(B, M, K).reshape(B, -1)
    V = neighbors.reshape(B, -1)
    pvalid = V >= 0
    V = V.clamp(min=0)
    P = U.size(1)
    run = torch.ones(B, dtype=torch.bool, device=dev) if active is None else active.clone()
    for _ in range(int(max_iterations)):
        if not bool(run.any()):
            break
        imp, aux = _pair_moves(S, inst, U, V, gen, penalty_capacity)
        gu0, gv0, enu, env, gap_u, gap_v, cl, oo_tgt = aux
        n_ops = imp.size(2)
        imp = imp & pvalid.unsqueeze(2)
        pair_imp = imp.any(dim=2) & run.unsqueeze(1)
        anyrow = pair_imp.any(dim=1)
        run = run & anyrow
        if not bool(anyrow.any()):
            break
        key = torch.rand(B, P, generator=gen, device=dev)
        key = torch.where(pair_imp, key, torch.full_like(key, 2.0))
        pi = key.argmin(dim=1, keepdim=True)
        op = imp.gather(1, pi.unsqueeze(2).expand(B, 1, n_ops)).squeeze(1).float().argmax(dim=1)
        sel = lambda t, pi=pi: t.gather(1, pi).squeeze(1)  # noqa: E731
        S = _apply(
            S,
            op,
            anyrow,
            sel(U),
            sel(V),
            sel(gu0),
            sel(gv0),
            sel(enu),
            sel(env),
            sel(gap_u),
            sel(gap_v),
            sel(cl),
            sel(oo_tgt),
        )
    return S


def _pair_moves(S, inst, U, V, gen, penalty_capacity):
    """Improvement flags ``[B, P, 14]`` of every operator of ``_process_pair`` for each pair."""
    B, L = S.shape
    P = U.size(1)
    dev = S.device
    R, C, Q = float(inst.R), float(inst.C), inst.capacity.view(B, 1)
    wt = inst.waste
    jL = torch.arange(L, device=dev).view(1, 1, L)
    st = _State(S, inst)
    D = lambda a, b: _dist(inst, a, b)  # noqa: E731
    gu, gv = _g(st.pos, U), _g(st.pos, V)
    ru_ok, rv_ok = (gu >= 0) & (U != 0), (gv >= 0) & (V != 0)
    gu0, gv0 = gu.clamp(min=0), gv.clamp(min=0)
    wu, wv = _g(wt, U), _g(wt, V)
    pu, nu = st.at(gu0 - 1), st.at(gu0 + 1)
    pv, nv = st.at(gv0 - 1), st.at(gv0 + 1)
    ldu, ldv = _g(st.ld, gu0), _g(st.ld, gv0)
    ru, rv = _g(st.rid, gu0), _g(st.rid, gv0)
    enu, env = _g(st.en, gu0), _g(st.en, gv0)
    stu, stv = _g(st.st, gu0), _g(st.st, gv0)
    n_ops = 14
    imp = torch.zeros(B, P, n_ops, dtype=torch.bool, device=dev)
    # 0/1 drop u / drop v
    for k, (x, ok, px, nx, ldx, wx) in enumerate([(U, ru_ok, pu, nu, ldu, wu), (V, rv_ok, pv, nv, ldv, wv)]):
        dd = D(px, nx) - D(px, x) - D(x, nx)
        tot = -wx * R - dd * C - (_pen(ldx - wx, Q, penalty_capacity) - _pen(ldx, Q, penalty_capacity))
        imp[:, :, k] = ok & (tot > EPS)
    # 2/3 unrouted insertion
    best_u, gap_u = _best_insertion(st, inst, U, gv0)
    best_v, gap_v = _best_insertion(st, inst, V, gu0)
    tot_u = wu * R - best_u * C - (_pen(ldv + wu, Q, penalty_capacity) - _pen(ldv, Q, penalty_capacity))
    tot_v = wv * R - best_v * C - (_pen(ldu + wv, Q, penalty_capacity) - _pen(ldu, Q, penalty_capacity))
    imp[:, :, 2] = ~ru_ok & rv_ok & (U != 0) & (tot_u > EPS)
    imp[:, :, 3] = ru_ok & ~rv_ok & (V != 0) & (tot_v > EPS)
    both = ru_ok & rv_ok & (U != V)
    inter = both & (ru != rv)
    intra = both & (ru == rv)
    # relocate u after v (inter 4 / intra 10)
    rel = -D(pu, U) - D(U, nu) + D(pu, nu) - D(V, nv) + D(V, U) + D(U, nv)
    rel_ok = rel * C < -EPS
    imp[:, :, 4] = inter & (ldv + wu <= Q) & rel_ok
    imp[:, :, 10] = intra & (gu != gv + 1) & rel_ok
    # swap (inter 5 / intra 11)
    sw = -D(pu, U) - D(U, nu) - D(pv, V) - D(V, nv) + D(pu, V) + D(V, nu) + D(pv, U) + D(U, nv)
    a_first = gu < gv
    fa, fb = torch.where(a_first, U, V), torch.where(a_first, V, U)
    pfa = torch.where(a_first, pu, pv)
    nfb = torch.where(a_first, nv, nu)
    sw_adj = D(pfa, fb) + D(fb, fa) + D(fa, nfb) - (D(pfa, fa) + D(fa, fb) + D(fb, nfb))
    adj = (gu - gv).abs() == 1
    sw_i = torch.where(adj, sw_adj, sw)
    imp[:, :, 5] = inter & (ldu - wu + wv <= Q) & (ldv - wv + wu <= Q) & (sw * C < -EPS)
    imp[:, :, 11] = intra & (sw_i * C < -EPS)
    # 2-opt* (6)
    hlu = _g(st.cumW, gu0) - _g(st.cumW, stu)
    hlv = _g(st.cumW, gv0) - _g(st.cumW, stv)
    tlu, tlv = ldu - hlu, ldv - hlv
    gain = D(U, nu) + D(V, nv) - D(U, nv) - D(V, nu)
    imp[:, :, 6] = inter & (hlu + tlv <= Q) & (hlv + tlu <= Q) & (gain > 0) & (gain * C > EPS)
    # shift(2,0) (7), swap(2,1) (8), swap(2,2) (9)
    u2, v2 = nu, nv
    has_u2, has_v2 = u2 != 0, v2 != 0
    nnu, nnv = st.at(gu0 + 2), st.at(gv0 + 2)
    wu2, wv2 = _g(wt, u2), _g(wt, v2)
    ch_u = D(U, u2)
    rem_u = D(pu, U) + ch_u + D(u2, nnu)
    s20 = D(pu, nnu) - rem_u + (D(V, U) + ch_u + D(u2, nv)) - D(V, nv)
    imp[:, :, 7] = inter & has_u2 & (ldv + wu + wu2 <= Q) & (s20 * C < -EPS)
    s21 = (D(pu, V) + D(V, nnu)) - rem_u + (D(pv, U) + ch_u + D(u2, nv)) - (D(pv, V) + D(V, nv))
    ok21 = (ldu - wu - wu2 + wv <= Q) & (ldv - wv + wu + wu2 <= Q)
    imp[:, :, 8] = inter & has_u2 & ok21 & (s21 * C < -EPS)
    ch_v = D(V, v2)
    s22 = (D(pu, V) + ch_v + D(v2, nnu)) - rem_u + (D(pv, U) + ch_u + D(u2, nnv)) - (D(pv, V) + ch_v + D(v2, nnv))
    ok22 = (ldu - wu - wu2 + wv + wv2 <= Q) & (ldv - wv - wv2 + wu + wu2 <= Q)
    imp[:, :, 9] = inter & has_u2 & has_v2 & ok22 & (s22 * C < -EPS)
    # 2-opt intra (12): p_u < p_v, not adjacent
    t2 = -D(U, nu) - D(V, nv) + D(U, V) + D(nu, nv)
    t2 = t2 + (_g(st.cumRD, gv0) - _g(st.cumRD, gu0 + 1))
    imp[:, :, 12] = intra & (gu < gv) & (gu + 1 != gv) & (t2 * C < -EPS)
    # Or-opt (13): chain length drawn per pair
    cl = torch.randint(1, 4, (B, P), generator=gen, device=dev)
    clast = gu0 + cl - 1
    fits = clast < enu
    last = st.at(clast)
    nxc = st.at(clast + 1)
    internal = _g(st.cumF, clast) - _g(st.cumF, gu0)
    removal = D(pu, U) + internal + D(last, nxc)
    repair = D(pu, nxc)
    Sj = S.unsqueeze(1).expand(B, P, L)
    Sn = torch.cat([S[:, 1:], torch.zeros_like(S[:, :1])], dim=1).unsqueeze(1).expand(B, P, L)
    Ue = U.unsqueeze(2).expand(B, P, L)
    ins = _dist(inst, Sj, Ue) + internal.unsqueeze(2) + _dist(inst, last.unsqueeze(2).expand(B, P, L), Sn)
    od = _dist(inst, Sj, Sn)
    dl = repair.unsqueeze(2) - removal.unsqueeze(2) + ins - od
    tgt = (jL > stu.unsqueeze(2)) & (jL < enu.unsqueeze(2)) & ((jL < gu0.unsqueeze(2) - 1) | (jL > clast.unsqueeze(2)))
    good = tgt & (dl * C < -EPS)
    oo_any = good.any(dim=2)
    oo_tgt = torch.where(good, jL.expand(B, P, L), torch.full_like(Sj, L)).min(dim=2).values
    imp[:, :, 13] = intra & fits & oo_any
    return imp, (gu0, gv0, enu, env, gap_u, gap_v, cl, oo_tgt)


def _apply(S, op, rows, u, v, gu, gv, enu, env, gap_u, gap_v, cl, oo_tgt) -> torch.Tensor:
    B, L = S.shape
    dev = S.device
    j = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
    key = j.to(torch.float64).clone()
    eps = 1.0 / (4 * (L + 1))
    o = op.unsqueeze(1)
    r = rows.unsqueeze(1)
    gu_, gv_, enu_, env_ = gu.unsqueeze(1), gv.unsqueeze(1), enu.unsqueeze(1), env.unsqueeze(1)
    jf = j.to(torch.float64)
    gu_f, gv_f = gu_.to(torch.float64), gv_.to(torch.float64)
    dele = torch.zeros_like(S, dtype=torch.bool)
    # drops
    dele |= r & (o == 0) & (j == gu_)
    dele |= r & (o == 1) & (j == gv_)
    # relocate u after v
    m = r & ((o == 4) | (o == 10)) & (j == gu_)
    key = torch.where(m, gv_f + 0.5, key)
    # swap
    m = r & ((o == 5) | (o == 11))
    key = torch.where(m & (j == gu_), gv_f, key)
    key = torch.where(m & (j == gv_), gu_f, key)
    # 2-opt*: exchange tails
    m = r & (o == 6)
    tu = m & (j > gu_) & (j < enu_)
    tv = m & (j > gv_) & (j < env_)
    key = torch.where(tu, gv_f + 0.5 + (jf - gu_f) * eps, key)
    key = torch.where(tv, gu_f + 0.5 + (jf - gv_f) * eps, key)
    # shift(2,0)
    m = r & (o == 7) & (j >= gu_) & (j <= gu_ + 1)
    key = torch.where(m, gv_f + 0.5 + (jf - gu_f + 1) * eps, key)
    # swap(2,1) / swap(2,2)
    h = torch.where(o == 9, 2, 1)
    m = r & ((o == 8) | (o == 9))
    ck = m & (j >= gu_) & (j <= gu_ + 1)
    chh = m & (j >= gv_) & (j < gv_ + h)
    key = torch.where(ck, gv_f - 0.5 + (jf - gu_f + 1) * eps, key)
    key = torch.where(chh, gu_f - 0.5 + (jf - gv_f + 1) * eps, key)
    # 2-opt intra: reverse positions gu+1 .. gv
    m = r & (o == 12) & (j > gu_) & (j <= gv_)
    key = torch.where(m, (gu_f + 1 + gv_f) - jf, key)
    # Or-opt: chain gu .. gu+cl-1 after target
    cl_ = cl.unsqueeze(1)
    tg = oo_tgt.unsqueeze(1).to(torch.float64)
    m = r & (o == 13) & (j >= gu_) & (j < gu_ + cl_)
    key = torch.where(m, tg + 0.5 + (jf - gu_f + 1) * eps, key)
    key = torch.where(dele, torch.full_like(key, 1e9), key)
    order = torch.argsort(key, dim=1, stable=True)
    out = torch.gather(S, 1, order)
    n_del = dele.sum(dim=1, keepdim=True)
    out = torch.where(j >= L - n_del, torch.zeros_like(out), out)
    # insertions (one node into a gap)
    ins_u = rows & (op == 2)
    ins_v = rows & (op == 3)
    if bool((ins_u | ins_v).any()):
        node = torch.where(ins_u, u, v)
        gap = torch.where(ins_u, gap_u, gap_v)
        jj = j
        g = gap.unsqueeze(1)
        src = torch.where(jj <= g, jj, jj - 1).clamp(min=0)
        moved = torch.gather(S, 1, src)
        moved = torch.where(jj == g + 1, node.unsqueeze(1), moved)
        out = torch.where((ins_u | ins_v).unsqueeze(1), moved, out)
    return normalize(out)
