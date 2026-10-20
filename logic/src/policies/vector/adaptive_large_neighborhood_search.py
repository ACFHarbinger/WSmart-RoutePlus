"""Batched ALNS engine mirroring the simulator's ``ALNSSolver``.

Same loop as :class:`~logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`
(Ropke & Pisinger 2005/2006), run on every instance of a batch at once:

- initial solution: ``build_greedy_routes`` (random-order profitable insertion with the
  dynamic seed hurdle), unless a starting solution is given;
- destroy operators: random, worst, Shaw (+ string, cluster, neighbour with
  ``extended_operators``), in the cost or profit flavour (``profit_aware_operators``);
- repair slots: greedy and regret-k for ``k`` in ``regret_pool``, each as a clean and a
  noisy slot (noise ``U[-noise_factor, noise_factor] * max_dist``), plus the savings
  insertion (single slot) with ``regret_pool="regretAll"``;
- removal count ``U{min(n, 4), max(lower, min(max_removal_cap, int(xi * n_nodes)))}``;
- roulette-wheel operator choice per instance, sigma_1/2/3 scoring with
  visited-solution tracking, segment weight updates every ``segment_size`` iterations
  with ``reaction_factor``;
- acceptance by the configured simulator acceptance criterion (one per instance);
- stop after ``max_iterations`` or ``time_limit`` seconds.

Attributes:
    VectorizedALNSEngine: The batched ALNS search.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Tuple

import torch

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine import MIN_REMOVAL_LOWER_BOUND
from logic.src.policies.vector.shared import routes as rs
from logic.src.policies.vector.shared.acceptance import BatchedAcceptance
from logic.src.policies.vector.shared.vrpp import VRPPBatch, hash_rows


def _regret_ks(pool: str) -> List[int]:
    if pool == "regret2":
        return [2]
    return [2, 3, 4]


class VectorizedALNSEngine:
    """Batched Adaptive Large Neighbourhood Search for the VRPP.

    Attributes:
        cfg: ``ALNSConfig`` of the simulator ALNS.
        expand_pool: Repairs may insert any unvisited bin (VRPP mode).
        destroy_names: Destroy operator names.
        repair_names: Repair slot names.
        repair_noisy: Noisy flag per repair slot.
    """

    def __init__(
        self,
        cfg: ALNSConfig,
        generator: torch.Generator,
        expand_pool: bool = True,
        recorder: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Set up the operator registries.

        Args:
            cfg: ALNS configuration.
            generator: Torch generator for every random draw.
            expand_pool: Repairs may insert any unvisited bin.
            recorder: Optional telemetry callback.
        """
        self.cfg = cfg
        self.gen = generator
        self.expand_pool = expand_pool
        self._record = recorder or (lambda **k: None)
        pa = bool(cfg.profit_aware_operators)
        p = "Profit" if pa else ""
        self.destroy_names = ["RandomRemoval", f"Worst{p}Removal", f"Shaw{p}Removal"]
        if cfg.extended_operators:
            self.destroy_names += [f"String{p}Removal", f"Cluster{p}Removal", f"Neighbor{p}Removal"]
        base = [("Greedy" + p, 0)] + [(f"Regret{k}{p}", k) for k in _regret_ks(cfg.regret_pool)]
        self.repair_names: List[str] = []
        self.repair_k: List[int] = []
        self.repair_noisy: List[bool] = []
        for name, k in base:
            for noisy in (False, True):
                self.repair_names.append(f"{name}{'Noisy' if noisy else 'Clean'}")
                self.repair_k.append(k)
                self.repair_noisy.append(noisy)
        if cfg.regret_pool == "regretAll":
            self.repair_names.append("Savings" + p)
            self.repair_k.append(-1)
            self.repair_noisy.append(False)

    # ------------------------------------------------------------------
    def _destroy(self, d: int, S: torch.Tensor, n: torch.Tensor, inst: VRPPBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        pa = bool(self.cfg.profit_aware_operators)
        name = self.destroy_names[d]
        if name == "RandomRemoval":
            P, pos = rs.random_removal(S, n, self.gen)
            removed = torch.zeros(S.size(0), inst.dist.size(1), dtype=torch.bool, device=S.device)
            removed.scatter_(1, torch.where(pos, S, torch.zeros_like(S)), True)
            removed[:, 0] = False
            return P, removed
        if name.startswith("Worst"):
            return rs.worst_removal(S, n, inst, self.cfg.worst_removal_randomness, self.gen, pa)
        if name.startswith("Shaw"):
            return rs.shaw_removal(S, n, inst, self.cfg.shaw_randomization, self.gen, pa)
        if name.startswith("String"):
            return rs.string_removal(S, n, inst, self.gen, pa)
        if name.startswith("Cluster"):
            return rs.cluster_removal(S, n, inst, self.gen, pa)
        return rs.neighbor_removal(S, n, inst, self.gen, pa)

    def _repair(self, r: int, S: torch.Tensor, removed: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
        pa = bool(self.cfg.profit_aware_operators)
        pending = removed.clone()
        if self.expand_pool:
            pending = pending | ~rs.visited(S, inst.dist.size(1))
            pending[:, 0] = False
        noise = None
        if self.repair_noisy[r]:
            max_d = inst.dist.flatten(1).max(dim=1).values
            u = torch.rand(S.size(0), generator=self.gen, device=S.device) * 2 - 1
            noise = u * self.cfg.noise_factor * max_d
        k = self.repair_k[r]
        if k < 0:
            return rs.savings_insertion(S, pending, inst, pa)
        if k == 0:
            return rs.greedy_insertion(S, pending, inst, pa, noise)
        return rs.regret_insertion(S, pending, inst, k, pa, noise)

    # ------------------------------------------------------------------
    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ALNS on the batch.

        Args:
            inst: Instances.
            initial: Optional starting route sequences.

        Returns:
            Tuple of (best route sequences, best profit ``[B]``).
        """
        cfg = self.cfg
        B, M = inst.dist.size(0), inst.dist.size(1)
        n_nodes = M - 1
        dev = inst.device
        start = time.perf_counter()
        S = initial.clone() if initial is not None else rs.greedy_profit_construction(inst, self.gen)
        cur_profit = rs.seq_profit(S, inst)
        best_S, best_profit = S.clone(), cur_profit.clone()
        if n_nodes == 0 or cfg.max_iterations <= 0:
            return best_S, best_profit

        nD, nR = len(self.destroy_names), len(self.repair_names)
        d_w = torch.ones(B, nD, device=dev)
        r_w = torch.ones(B, nR, device=dev)
        d_sc = torch.zeros(B, nD, device=dev)
        r_sc = torch.zeros(B, nR, device=dev)
        d_ct = torch.zeros(B, nD, device=dev)
        r_ct = torch.zeros(B, nR, device=dev)
        rf = float(cfg.reaction_factor)
        visited_sets = [set() for _ in range(B)]
        for b, h in enumerate(hash_rows(S)):
            visited_sets[b].add(h)
        acc = BatchedAcceptance(cfg.acceptance_criterion, B)
        acc.setup(cur_profit)
        upper_cap = min(int(cfg.max_removal_cap), int(cfg.xi * n_nodes))
        b_idx = torch.arange(B, device=dev)

        for it in range(int(cfg.max_iterations)):
            if cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit:
                break
            d_sel = torch.multinomial(d_w, 1, generator=self.gen).squeeze(1)
            r_sel = torch.multinomial(r_w, 1, generator=self.gen).squeeze(1)
            cur_n = rs.n_visited(S)
            lower = torch.clamp(cur_n, max=MIN_REMOVAL_LOWER_BOUND)
            upper = torch.maximum(lower, torch.full_like(lower, upper_cap))
            u = torch.rand(B, generator=self.gen, device=dev)
            n_rm = lower + (u * (upper - lower + 1).float()).long().clamp(max=(upper - lower).max().item() if B else 0)
            n_rm = torch.minimum(n_rm, upper)
            n_rm = torch.where(cur_n == 0, torch.zeros_like(n_rm), n_rm)

            cand = S.clone()
            for d in range(nD):
                for r in range(nR):
                    rows = ((d_sel == d) & (r_sel == r)).nonzero(as_tuple=True)[0]
                    if rows.numel() == 0:
                        continue
                    sub = inst.subset(rows)
                    P, removed = self._destroy(d, S[rows], n_rm[rows], sub)
                    cand[rows] = self._repair(r, P, removed, sub)
            new_profit = rs.seq_profit(cand, inst)

            hashes = hash_rows(cand)
            is_new = torch.tensor([h not in visited_sets[b] for b, h in enumerate(hashes)], device=dev)
            accepted = acc.accept(cur_profit, new_profit, best_profit, it, int(cfg.max_iterations))
            score = torch.zeros(B, device=dev)
            new_best = new_profit > best_profit + 1e-6
            better = new_profit > cur_profit + 1e-6
            score = torch.where(accepted & is_new & ~better, torch.full_like(score, cfg.sigma_3), score)
            score = torch.where(accepted & is_new & better, torch.full_like(score, cfg.sigma_2), score)
            score = torch.where(new_best, torch.full_like(score, cfg.sigma_1), score)

            best_S = torch.where(new_best.unsqueeze(1), cand, best_S)
            best_profit = torch.where(new_best, new_profit, best_profit)
            for b in accepted.nonzero(as_tuple=True)[0].tolist():
                visited_sets[b].add(hashes[b])
            S = torch.where(accepted.unsqueeze(1), cand, S)
            cur_profit = torch.where(accepted, new_profit, cur_profit)

            d_sc[b_idx, d_sel] += score
            r_sc[b_idx, r_sel] += score
            d_ct[b_idx, d_sel] += 1
            r_ct[b_idx, r_sel] += 1
            if (it + 1) % int(cfg.segment_size) == 0:
                d_w = torch.where(d_ct > 0, (1 - rf) * d_w + rf * d_sc / d_ct.clamp(min=1), d_w)
                r_w = torch.where(r_ct > 0, (1 - rf) * r_w + rf * r_sc / r_ct.clamp(min=1), r_w)
                # Keep the roulette wheel valid when every weight decayed to 0.
                d_w = torch.where(d_w.sum(1, keepdim=True) > 0, d_w, torch.ones_like(d_w))
                r_w = torch.where(r_w.sum(1, keepdim=True) > 0, r_w, torch.ones_like(r_w))
                d_sc.zero_()
                r_sc.zero_()
                d_ct.zero_()
                r_ct.zero_()
            acc.step(cur_profit, new_profit, accepted, best_profit, it, int(cfg.max_iterations))
            self._record(
                iteration=it,
                best_profit=float(best_profit.mean()),
                current_profit=float(cur_profit.mean()),
                accepted=int(accepted.sum()),
            )
        return best_S, best_profit
