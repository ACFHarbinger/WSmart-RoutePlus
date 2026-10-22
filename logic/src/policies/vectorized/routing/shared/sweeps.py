"""Batched first-improvement sweeps mirroring the simulator's ``SweepLocalSearch``.

The simulator's random local search route improver
(:class:`~logic.src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover`)
calls one sweep of :class:`~logic.src.policies.helpers.local_search.local_search_sweeps.SweepLocalSearch`
per iteration. A sweep scans its moves in a fixed nested-loop order and applies the
first improving one. Here every candidate move of a sweep is evaluated at once with the
simulator's delta and feasibility rules, and the first improving move in the
simulator's loop order is applied in each row.

Routes left empty by a move stay in the solution, as in the simulator (they only
matter to ``cross_exchange``, which may move a segment into an empty route). A batch of
solutions is a route sequence whose ``n_routes + 1`` first zeros delimit the routes
(adjacent zeros are an empty route); see :class:`SweepState`.

The 3-opt sweep draws, as ``_apply_3opt``, five random third cut points per candidate
pair; the draws are made for all pairs at once instead of lazily, so the random stream
differs from the simulator's ``random.Random`` while the law of the search is the same.

Attributes:
    SweepState: Route sequences with empty routes kept.
    sweep: Apply one sweep of a named operator.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .routes import from_route_lists, to_route_lists
from .vrpp import VRPPBatch

EPS = 1e-4
INF = float("inf")


def _bi(t: torch.Tensor) -> torch.Tensor:
    return torch.arange(t.size(0), device=t.device).view(-1, *([1] * (t.dim() - 1)))


class SweepState:
    """Route sequences ``[B, L]`` that keep empty routes, with their route counts ``[B]``."""

    def __init__(self, S: torch.Tensor, n_routes: torch.Tensor) -> None:
        self.S = S
        self.n_routes = n_routes

    @staticmethod
    def from_seqs(S: torch.Tensor) -> "SweepState":
        """From normalised route sequences."""
        n_routes = ((S[:, :-1] == 0) & (S[:, 1:] != 0)).sum(dim=1)
        return SweepState(S.clone(), n_routes)

    def to_seqs(self, n_bins: int) -> torch.Tensor:
        """Normalised route sequences (empty routes dropped)."""
        return from_route_lists([[r for r in rl if r] for rl in to_route_lists(self.S)], n_bins, self.S.device)


class _Tables:
    def __init__(self, st: SweepState, inst: VRPPBatch) -> None:
        S = st.S
        B, L = S.shape
        dev = S.device
        j = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        isz = S == 0
        rid = torch.cumsum(isz.long(), dim=1) - 1
        self.rid = rid.clamp(min=0)
        self.inr = (~isz) & (rid < st.n_routes.unsqueeze(1))  # bin positions
        self.st = torch.cummax(torch.where(isz, j, torch.zeros_like(j)), dim=1).values
        rev = torch.where(isz, j, torch.full_like(j, L - 1)).flip(1)
        self.en = torch.cummin(rev, dim=1).values.flip(1)
        # For a zero at position p (route start), its route's end is the next zero.
        nxt_zero = torch.cummin(torch.where(isz, j, torch.full_like(j, L - 1)).flip(1), dim=1).values.flip(1)
        shifted = torch.cat([nxt_zero[:, 1:], torch.full_like(nxt_zero[:, :1], L - 1)], dim=1)
        self.en = torch.where(isz, shifted, self.en)
        w = torch.gather(inst.waste, 1, S)
        self.w = w
        self.cumW = torch.cumsum(w, dim=1)
        loads = torch.zeros(B, L + 1, dtype=w.dtype, device=dev).scatter_add_(1, self.rid, w)
        self.ld = torch.gather(loads, 1, self.rid)
        self.L = L
        self.S = S
        self.n_routes = st.n_routes
        self.j = j
        nxt = torch.cat([S[:, 1:], torch.zeros_like(S[:, :1])], dim=1)
        b = _bi(S)
        fwd = inst.dist[b, S, nxt]
        bwd = inst.dist[b, nxt, S]
        rd = torch.where((S != 0) & (nxt != 0), bwd - fwd, torch.zeros_like(bwd))
        self.cumRD = torch.cat([torch.zeros_like(rd[:, :1]), torch.cumsum(rd, dim=1)], dim=1)
        self.cumF = torch.cat([torch.zeros_like(fwd[:, :1]), torch.cumsum(fwd, dim=1)], dim=1)

    def at(self, p: torch.Tensor) -> torch.Tensor:
        flat = p.clamp(0, self.L - 1).reshape(p.size(0), -1)
        return torch.gather(self.S, 1, flat).reshape(p.shape)

    def g(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        flat = p.clamp(0, t.size(1) - 1).reshape(p.size(0), -1)
        return torch.gather(t, 1, flat).reshape(p.shape)


def _first(mask: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per row: any improving move, and the flat index of the one with the smallest key."""
    B = mask.size(0)
    k = torch.where(mask, key, torch.full_like(key, torch.iinfo(torch.long).max)).reshape(B, -1)
    return mask.reshape(B, -1).any(dim=1), k.argmin(dim=1)


def _permute(S: torch.Tensor, key: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(key, dim=1, stable=True)
    return torch.where(rows.unsqueeze(1), torch.gather(S, 1, order), S)


def sweep(
    state: SweepState, inst: VRPPBatch, name: str, gen: Optional[torch.Generator] = None
) -> Tuple[SweepState, torch.Tensor]:
    """Apply one ``SweepLocalSearch`` sweep of operator ``name`` (RLS op-map key) per row.

    Args:
        state: Current solutions.
        inst: Instances.
        name: ``two_opt``, ``swap``, ``relocate``, ``two_opt_star``, ``swap_star``,
            ``three_opt``, ``or_opt`` (chain 2), ``relocate_chain`` (chains 2-3) or
            ``cross_exchange`` (segments up to 2).
        gen: Torch generator (3-opt cut draws).

    Returns:
        Tuple of (new state, improved flags ``[B]``).
    """
    fn = {
        "relocate": _relocate,
        "swap": _swap,
        "two_opt": _two_opt,
        "two_opt_star": _two_opt_star,
        "swap_star": _swap_star,
        "three_opt": _three_opt,
        "or_opt": _or_opt,
        "relocate_chain": _relocate_chain,
        "cross_exchange": _cross_exchange,
    }[name]
    T = _Tables(state, inst)
    if name == "three_opt":
        S, imp = fn(T, inst, gen)
    else:
        S, imp = fn(T, inst)
    return SweepState(S, state.n_routes), imp


def _pairs(T: _Tables):
    B, L = T.S.shape
    gu = torch.arange(L, device=T.S.device).view(1, L, 1).expand(B, L, L)
    gv = torch.arange(L, device=T.S.device).view(1, 1, L).expand(B, L, L)
    return gu, gv


def _D(inst: VRPPBatch, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return inst.dist[_bi(a), a, b]


def _relocate(T: _Tables, inst: VRPPBatch):
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    gu, gv = _pairs(T)
    u, v = T.at(gu), T.at(gv)
    pu, nu, nv = T.at(gu - 1), T.at(gu + 1), T.at(gv + 1)
    ok = T.g(T.inr.long(), gu).bool() & T.g(T.inr.long(), gv).bool()
    same = T.g(T.rid, gu) == T.g(T.rid, gv)
    ok = ok & ~(same & ((gu == gv + 1) | (gu == gv)))
    ok = ok & (same | (T.g(T.ld, gv) + T.g(T.w, gu) <= Q))
    d = -_D(inst, pu, u) - _D(inst, u, nu) + _D(inst, pu, nu) - _D(inst, v, nv) + _D(inst, v, u) + _D(inst, u, nv)
    m = ok & (d * C < -EPS)
    has, idx = _first(m, gu * L + gv)
    a, b_ = idx // L, idx % L
    key = T.j.to(torch.float64).clone()
    r = torch.arange(B, device=T.S.device)
    key[r, a] = b_.to(torch.float64) + 0.5
    return _permute(T.S, key, has), has


def _swap_delta(T, inst, gu, gv):
    u, v = T.at(gu), T.at(gv)
    pu, nu, pv, nv = T.at(gu - 1), T.at(gu + 1), T.at(gv - 1), T.at(gv + 1)
    gen_d = (
        -_D(inst, pu, u)
        - _D(inst, u, nu)
        - _D(inst, pv, v)
        - _D(inst, v, nv)
        + _D(inst, pu, v)
        + _D(inst, v, nu)
        + _D(inst, pv, u)
        + _D(inst, u, nv)
    )
    fa = torch.where(gu < gv, u, v)
    fb = torch.where(gu < gv, v, u)
    pfa = torch.where(gu < gv, pu, pv)
    nfb = torch.where(gu < gv, nv, nu)
    adj_d = (
        _D(inst, pfa, fb)
        + _D(inst, fb, fa)
        + _D(inst, fa, nfb)
        - (_D(inst, pfa, fa) + _D(inst, fa, fb) + _D(inst, fb, nfb))
    )
    same = T.g(T.rid, gu) == T.g(T.rid, gv)
    return torch.where(same & ((gu - gv).abs() == 1), adj_d, gen_d), same


def _apply_swap(T, has, a, b_):
    B = T.S.size(0)
    key = T.j.to(torch.float64).clone()
    r = torch.arange(B, device=T.S.device)
    key[r, a] = b_.to(torch.float64)
    key[r, b_] = a.to(torch.float64)
    return _permute(T.S, key, has)


def _swap(T: _Tables, inst: VRPPBatch):
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    gu, gv = _pairs(T)
    ok = T.g(T.inr.long(), gu).bool() & T.g(T.inr.long(), gv).bool() & (gv > gu)
    d, same = _swap_delta(T, inst, gu, gv)
    wu, wv = T.g(T.w, gu), T.g(T.w, gv)
    ok = ok & (same | ((T.g(T.ld, gu) - wu + wv <= Q) & (T.g(T.ld, gv) - wv + wu <= Q)))
    has, idx = _first(ok & (d * C < -EPS), gu * L + gv)
    return _apply_swap(T, has, idx // L, idx % L), has


def _swap_star(T: _Tables, inst: VRPPBatch):
    """``move_swap_star`` in a sweep: the Top-3 insertion cache is empty in
    ``SweepLocalSearch``, so both bins go to each other's place (soft capacity penalty)."""
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    gu, gv = _pairs(T)
    ru, rv = T.g(T.rid, gu), T.g(T.rid, gv)
    ok = T.g(T.inr.long(), gu).bool() & T.g(T.inr.long(), gv).bool() & (ru < rv)
    d, _ = _swap_delta(T, inst, gu, gv)
    wu, wv = T.g(T.w, gu), T.g(T.w, gv)
    lu, lv = T.g(T.ld, gu), T.g(T.ld, gv)
    pen = (
        (lu - wu + wv - Q).clamp(min=0)
        - (lu - Q).clamp(min=0)
        + (lv - wv + wu - Q).clamp(min=0)
        - (lv - Q).clamp(min=0)
    )
    m = ok & (d * C + pen < -EPS)
    key = (ru * L + rv) * L * L + gu * L + gv
    has, idx = _first(m, key)
    return _apply_swap(T, has, idx // L, idx % L), has


def _two_opt(T: _Tables, inst: VRPPBatch):
    """``two_opt_intra``: u is the route's start depot (``p_u = -1``) or a bin, v a bin at
    least two positions later; the segment after u up to v is reversed."""
    B, L = T.S.shape
    C = float(inst.C)
    gu, gv = _pairs(T)
    inr_u, inr_v = T.g(T.inr.long(), gu).bool(), T.g(T.inr.long(), gv).bool()
    u_ok = inr_u | ((T.at(gu) == 0) & T.g(T.inr.long(), gu + 1).bool())
    ok = u_ok & inr_v & (T.g(T.rid, gv) == T.g(T.rid, gu)) & (gv >= gu + 2)
    u, v = T.at(gu), T.at(gv)
    un, vn = T.at(gu + 1), T.at(gv + 1)
    d = -_D(inst, u, un) - _D(inst, v, vn) + _D(inst, u, v) + _D(inst, un, vn)
    d = d + (T.g(T.cumRD, gv) - T.g(T.cumRD, gu + 1))
    has, idx = _first(ok & (d * C < -EPS), gu * L + gv)
    a, b_ = (idx // L).unsqueeze(1), (idx % L).unsqueeze(1)
    key = torch.where((T.j > a) & (T.j <= b_), a + 1 + b_ - T.j, T.j).to(torch.float64)
    return _permute(T.S, key, has), has


def _two_opt_star(T: _Tables, inst: VRPPBatch):
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    gu, gv = _pairs(T)
    ru, rv = T.g(T.rid, gu), T.g(T.rid, gv)
    ok = T.g(T.inr.long(), gu).bool() & T.g(T.inr.long(), gv).bool() & (ru < rv)
    u, v, nu, nv = T.at(gu), T.at(gv), T.at(gu + 1), T.at(gv + 1)
    hu = T.g(T.cumW, gu) - T.g(T.cumW, T.g(T.st, gu))
    hv = T.g(T.cumW, gv) - T.g(T.cumW, T.g(T.st, gv))
    tu, tv = T.g(T.ld, gu) - hu, T.g(T.ld, gv) - hv
    gain = _D(inst, u, nu) + _D(inst, v, nv) - _D(inst, u, nv) - _D(inst, v, nu)
    ok = ok & (hu + tv <= Q) & (hv + tu <= Q) & (gain > 0) & (gain * C > EPS)
    has, idx = _first(ok, (ru * L + rv) * L * L + gu * L + gv)
    a, b_ = (idx // L).unsqueeze(1), (idx % L).unsqueeze(1)
    ea = T.g(T.en, a)
    eb = T.g(T.en, b_)
    jf = T.j.to(torch.float64)
    eps = 1.0 / (4 * (L + 1))
    key = jf.clone()
    tail_u = (T.j > a) & (T.j < ea)
    tail_v = (T.j > b_) & (T.j < eb)
    key = torch.where(tail_u, b_.double() + 0.5 + (jf - a.double()) * eps, key)
    key = torch.where(tail_v, a.double() + 0.5 + (jf - b_.double()) * eps, key)
    return _permute(T.S, key, has), has


def _three_opt(T: _Tables, inst: VRPPBatch, gen: Optional[torch.Generator]):
    B, L = T.S.shape
    C = float(inst.C)
    dev = T.S.device
    gu, gv = _pairs(T)
    stu = T.g(T.st, gu)
    ln = T.g(T.en, gu) - stu - 1
    pu = gu - stu - 1
    pv = gv - stu - 1
    same = T.g(T.rid, gu) == T.g(T.rid, gv)
    ok = T.g(T.inr.long(), gu).bool() & T.g(T.inr.long(), gv).bool() & same & (ln >= 4)
    ok = ok & (pu <= ln - 3) & (pv >= pu + 2)
    A = 5
    pw = _draw_cuts(ln, A, gen)
    pu4, pv4 = pu.unsqueeze(3), pv.unsqueeze(3)
    skip = (pw == pu4) | (pw == pv4) | (pw == pu4 + 1) | (pw == pv4 + 1) | (pw == pu4 - 1) | (pw == pv4 - 1)
    i = torch.minimum(torch.minimum(pu4, pv4), pw)
    k = torch.maximum(torch.maximum(pu4, pv4), pw)
    jj = pu4 + pv4 + pw - i - k
    base = stu.unsqueeze(3) + 1
    lnn = ln.unsqueeze(3)

    def node(p):
        return torch.where(p < lnn, T.at((base + p).reshape(B, -1)).reshape(p.shape), torch.zeros_like(p))

    a_, b_ = node(i), node(i + 1)
    c_, d_ = node(jj), node(jj + 1)
    e_, f_ = node(k), node(k + 1)

    def D4(x, y):
        return inst.dist[torch.arange(B, device=dev).view(B, 1, 1, 1), x, y]

    base_d = D4(a_, b_) + D4(c_, d_) + D4(e_, f_)

    def rdv(s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        hi = T.g(T.cumRD, (base + e).reshape(B, -1)).reshape(s.shape)
        return hi - T.g(T.cumRD, (base + s).reshape(B, -1)).reshape(s.shape)

    r2 = rdv(i + 1, jj)
    r3 = rdv(jj + 1, k)
    g4 = base_d - (D4(a_, c_) + D4(b_, e_) + D4(d_, f_)) - r2 - r3
    g5 = base_d - (D4(a_, d_) + D4(e_, b_) + D4(c_, f_))
    g6 = base_d - (D4(a_, d_) + D4(e_, c_) + D4(b_, f_)) - r2
    g7 = base_d - (D4(a_, e_) + D4(d_, b_) + D4(c_, f_)) - r3
    gains = torch.stack([g4, g5, g6, g7], dim=4)
    best_g, case = gains.max(dim=4)
    good = ~skip & (best_g * C > EPS) & ok.unsqueeze(3)
    att = torch.where(good, torch.arange(A, device=dev).view(1, 1, 1, A), torch.full_like(pw, A)).min(dim=3).values
    pair_ok = att < A
    has, idx = _first(pair_ok, gu * L + gv)
    r = torch.arange(B, device=dev)
    a_i, b_i = idx // L, idx % L
    at_ = att[r, a_i, b_i].clamp(max=A - 1)
    I = i[r, a_i, b_i, at_].unsqueeze(1)
    J = jj[r, a_i, b_i, at_].unsqueeze(1)
    K = k[r, a_i, b_i, at_].unsqueeze(1)
    cs = case[r, a_i, b_i, at_].unsqueeze(1)
    st0 = stu[r, a_i, b_i].unsqueeze(1) + 1
    t = T.j - st0
    a1 = I + 1
    l2, l3 = J - I, K - J
    in2 = (t > I) & (t <= J)
    in3 = (t > J) & (t <= K)
    newpos = t.clone()
    newpos = torch.where(in2 & (cs == 0), a1 + (J - t), newpos)
    newpos = torch.where(in3 & (cs == 0), a1 + l2 + (K - t), newpos)
    newpos = torch.where(in3 & ((cs == 1) | (cs == 2)), a1 + (t - (J + 1)), newpos)
    newpos = torch.where(in2 & (cs == 1), a1 + l3 + (t - (I + 1)), newpos)
    newpos = torch.where(in2 & (cs == 2), a1 + l3 + (J - t), newpos)
    newpos = torch.where(in3 & (cs == 3), a1 + (K - t), newpos)
    newpos = torch.where(in2 & (cs == 3), a1 + l3 + (t - (I + 1)), newpos)
    key = (st0 + newpos).to(torch.float64)
    return _permute(T.S, key, has), has


def _draw_cuts(ln: torch.Tensor, n: int, gen: Optional[torch.Generator]) -> torch.Tensor:
    """``n`` uniform third cut positions in ``[0, len - 1]`` per pair (``rng.randint``)."""
    u = torch.rand(*ln.shape, n, generator=gen, device=ln.device)
    return (u * ln.unsqueeze(-1).clamp(min=1)).long()


def _chain_moves(T: _Tables, inst: VRPPBatch, cl: int, intra_only: bool):
    """Improving ``relocate_chain`` moves ``[B, L(src), L(dst)]`` for chain length ``cl``."""
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    gu, gv = _pairs(T)
    last = gu + cl - 1
    enu = T.g(T.en, gu)
    ok = T.g(T.inr.long(), gu).bool() & (last < enu) & T.g(T.inr.long(), gv).bool()
    same = T.g(T.rid, gu) == T.g(T.rid, gv)
    dem = T.g(T.cumW, last) - T.g(T.cumW, gu - 1)
    ok = ok & torch.where(same, ~((gv >= gu) & (gv <= last)), T.g(T.ld, gv) + dem <= Q)
    if intra_only:
        ok = ok & same
    prev_c, next_c = T.at(gu - 1), T.at(last + 1)
    u, z = T.at(gu), T.at(last)
    internal = T.g(T.cumF, last) - T.g(T.cumF, gu)
    removal = _D(inst, prev_c, u) + internal + _D(inst, z, next_c)
    repair = _D(inst, prev_c, next_c)
    v = T.at(gv)
    vn = torch.where(same & (gv == gu - 1), next_c, T.at(gv + 1))
    ins = _D(inst, v, u) + internal + _D(inst, z, vn)
    d = repair - removal + ins - _D(inst, v, vn)
    return ok & (d * C < -EPS)


def _apply_chain(T: _Tables, has, src, dst, cl):
    L = T.S.size(1)
    eps = 1.0 / (4 * (L + 1))
    s, t = src.unsqueeze(1), dst.unsqueeze(1)
    clv = cl.unsqueeze(1) if torch.is_tensor(cl) else cl
    jf = T.j.to(torch.float64)
    m = (T.j >= s) & (T.j < s + clv)
    key = torch.where(m, t.double() + 0.5 + (jf - s.double() + 1) * eps, jf)
    return _permute(T.S, key, has)


def _or_opt(T: _Tables, inst: VRPPBatch):
    """``or_opt(chain_len=2)``: per start position (route order), the first improving target."""
    B, L = T.S.shape
    gu, gv = _pairs(T)
    m = _chain_moves(T, inst, 2, intra_only=True)
    has, idx = _first(m, gu * L + gv)
    return _apply_chain(T, has, idx // L, idx % L, 2), has


def _relocate_chain(T: _Tables, inst: VRPPBatch):
    """``relocate_chain_op(max_chain_len=3)``: order (route, chain length, start, target route, target)."""
    B, L = T.S.shape
    gu, gv = _pairs(T)
    ru, rv = T.g(T.rid, gu), T.g(T.rid, gv)
    lnu = T.g(T.en, gu) - T.g(T.st, gu) - 1
    ms, keys = [], []
    for ci, cl in enumerate((2, 3)):
        m = _chain_moves(T, inst, cl, intra_only=False) & (lnu >= cl)
        ms.append(m)
        keys.append(((ru * 2 + ci) * L + gu) * L * L + rv * L + gv)
    m = torch.stack(ms, dim=1)
    key = torch.stack(keys, dim=1)
    has, idx = _first(m, key)
    ci = idx // (L * L)
    rem = idx % (L * L)
    return _apply_chain(T, has, rem // L, rem % L, ci + 2), has


def _cross_exchange(T: _Tables, inst: VRPPBatch, max_seg: int = 2):
    """``cross_exchange_op(max_seg_len=2)``.

    Loop order (route a, route b > a, segment lengths (la, lb) != (0, 0), start a, start
    b); a segment starts at a bin or, for the empty segment at the end of a route (or in
    an empty route), at the route's closing depot zero.
    """
    B, L = T.S.shape
    C, Q = float(inst.C), inst.capacity.view(B, 1, 1)
    dev = T.S.device
    ga, gb = _pairs(T)
    isz = T.S == 0
    rid_s = torch.where(isz, T.rid - 1, T.rid)
    closing = isz & (rid_s >= 0) & (rid_s < T.n_routes.unsqueeze(1))
    start_ok = T.inr | closing
    en_s = torch.where(isz, T.j, T.en)
    ra, rb = T.g(rid_s, ga), T.g(rid_s, gb)
    ea, eb = T.g(en_s, ga), T.g(en_s, gb)
    base_ok = T.g(start_ok.long(), ga).bool() & T.g(start_ok.long(), gb).bool() & (ra < rb)
    lda, ldb = _route_load(T, ra), _route_load(T, rb)
    ms, keys, combos = [], [], []
    for la in range(max_seg + 1):
        for lb in range(max_seg + 1):
            if la == 0 and lb == 0:
                continue
            ok = base_ok & (ga + la <= ea) & (gb + lb <= eb)
            wa = T.g(T.cumW, ga + la - 1) - T.g(T.cumW, ga - 1) if la else torch.zeros_like(lda)
            wb = T.g(T.cumW, gb + lb - 1) - T.g(T.cumW, gb - 1) if lb else torch.zeros_like(lda)
            ok = ok & (lda - wa + wb <= Q) & (ldb - wb + wa <= Q)
            ap, an = T.at(ga - 1), T.at(ga + la)
            bp, bn = T.at(gb - 1), T.at(gb + lb)
            a0, al = T.at(ga), T.at(ga + la - 1)
            b0, bl = T.at(gb), T.at(gb + lb - 1)
            rem_a = _D(inst, ap, a0) + _D(inst, al, an) if la else _D(inst, ap, an)
            ins_a = _D(inst, ap, b0) + _D(inst, bl, an) if lb else _D(inst, ap, an)
            rem_b = _D(inst, bp, b0) + _D(inst, bl, bn) if lb else _D(inst, bp, bn)
            ins_b = _D(inst, bp, a0) + _D(inst, al, bn) if la else _D(inst, bp, bn)
            d = ins_a - rem_a + (ins_b - rem_b)
            ms.append(ok & (d * C < -EPS))
            keys.append(((ra * L + rb) * 16 + la * (max_seg + 1) + lb) * L * L + ga * L + gb)
            combos.append((la, lb))
    has, idx = _first(torch.stack(ms, dim=1), torch.stack(keys, dim=1))
    cidx = (idx // (L * L)).clamp(max=len(combos) - 1)
    rem = idx % (L * L)
    a1, b1 = (rem // L).unsqueeze(1), (rem % L).unsqueeze(1)
    la1 = torch.tensor([c[0] for c in combos], device=dev)[cidx].unsqueeze(1)
    lb1 = torch.tensor([c[1] for c in combos], device=dev)[cidx].unsqueeze(1)
    eps = 1.0 / (4 * (L + 1))
    jf = T.j.to(torch.float64)
    seg_a = (T.j >= a1) & (T.j < a1 + la1)
    seg_b = (T.j >= b1) & (T.j < b1 + lb1)
    key = torch.where(seg_a, b1.double() - 0.5 + (jf - a1.double() + 1) * eps, jf)
    key = torch.where(seg_b, a1.double() - 0.5 + (jf - b1.double() + 1) * eps, key)
    return _permute(T.S, key, has), has


def _route_load(T: _Tables, r: torch.Tensor) -> torch.Tensor:
    B, L = T.S.shape
    loads = torch.zeros(B, L + 1, dtype=T.w.dtype, device=T.S.device).scatter_add_(1, T.rid, T.w)
    return torch.gather(loads, 1, r.clamp(min=0).reshape(B, -1)).reshape(r.shape)
