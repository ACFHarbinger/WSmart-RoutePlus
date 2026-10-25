"""Destroy and repair operator registries of the ALNS solver.

:class:`ALNSOperatorRegistryMixin` holds the
:class:`~logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`
methods that build the named destroy (removal) and repair (insertion) operator lists from
the configuration, in their cost-only and profit-aware variants. Moved unchanged from
``alns.py``.
"""

from typing import Callable, Dict, List, Tuple

from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    cluster_profit_removal,
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    neighbor_profit_removal,
    neighbor_removal,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    regret_3_insertion,
    regret_3_profit_insertion,
    regret_4_insertion,
    regret_4_profit_insertion,
    savings_insertion,
    savings_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_profit_removal,
    string_removal,
    worst_profit_removal,
    worst_removal,
)


class ALNSOperatorRegistryMixin:
    """Destroy and repair operator registry builders of :class:`ALNSSolver` (moved unchanged from ``alns.py``)."""

    # ------------------------------------------------------------------
    # Operator registry builders
    # ------------------------------------------------------------------

    def _build_destroy_ops(self) -> Tuple[List[Callable], List[str]]:
        """
        Build the destroy operator registry.

        Core set (always included):
            random, worst[_profit], shaw[_profit]

        Extended set (when extended_operators=True):
            + string[_profit], cluster[_profit], neighbor[_profit]

        Returns:
            Tuple of (operator callables, operator names).
        """
        params = self.params
        pa = self.profit_aware_operators

        if pa:
            core: List[Tuple[str, Callable]] = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstProfitRemoval",
                    lambda r, n: worst_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        self.wastes,
                        self.R,
                        self.C,
                        p=params.worst_removal_randomness,
                        rng=self.random,  # type: ignore[arg-type]
                    ),
                ),
                (
                    "ShawProfitRemoval",
                    lambda r, n: shaw_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended: List[Tuple[str, Callable]] = [
                (
                    "StringProfitRemoval",
                    lambda r, n: string_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "ClusterProfitRemoval",
                    lambda r, n: cluster_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
                (
                    "NeighborProfitRemoval",
                    lambda r, n: neighbor_profit_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        R=self.R,
                        C=self.C,
                        rng=self.random,
                    ),
                ),
            ]
        else:
            core = [
                (
                    "RandomRemoval",
                    lambda r, n: random_removal(r, n, rng=self.random),
                ),
                (
                    "WorstRemoval",
                    lambda r, n: worst_removal(
                        r, n, self.dist_matrix, p=params.worst_removal_randomness, rng=self.np_random
                    ),
                ),
                (
                    "ShawRemoval",
                    lambda r, n: shaw_removal(
                        r,
                        n,
                        self.dist_matrix,
                        wastes=self.wastes,
                        randomization_factor=params.shaw_randomization,
                        rng=self.random,
                    ),
                ),
            ]
            extended = [
                (
                    "StringRemoval",
                    lambda r, n: string_removal(r, n, self.dist_matrix, rng=self.random),
                ),
                (
                    "ClusterRemoval",
                    lambda r, n: cluster_removal(r, n, self.dist_matrix, nodes=self.nodes, rng=self.random),
                ),
                (
                    "NeighborRemoval",
                    lambda r, n: neighbor_removal(r, n, self.dist_matrix, rng=self.random),
                ),
            ]

        registry = core + extended if self.extended_operators else core
        ops = [fn for _, fn in registry]
        names = [nm for nm, _ in registry]
        return ops, names

    def _build_repair_ops(self) -> Tuple[List[Callable], List[str], List[bool]]:
        """Build the repair operator registry with clean/noisy operator pairs.

        Noise-capable operators (greedy, regret-k) produce two consecutive slots:
            [<Name>Clean, <Name>Noisy]
        The adaptive weight mechanism tracks them independently, learning
        whether noisy or clean variants perform better (paper Conf. 15).

        Operators that do not accept a noise parameter (e.g. savings) produce
        a single slot only — including them in both slots would be redundant.

        Operator set is controlled by params.regret_pool:
            "regret2"   → greedy, regret-2                        (4 slots)
            "regret234" → greedy, regret-2, regret-3, regret-4    (8 slots, default)
            "regretAll" → greedy, regret-2, regret-3, regret-4,
                          savings                                  (9 slots)

        Returns:
            Tuple of (operator callables, operator names, is_noisy flags).
        """
        params = self.params
        pa = self.profit_aware_operators
        regret_pool: str = getattr(params, "regret_pool", "regret234")

        # ------------------------------------------------------------------
        # Define the logical repair base functions.
        # Each entry: (label, fn, supports_noise)
        #   supports_noise=True  → expanded into (Clean, Noisy) slot pair
        #   supports_noise=False → single slot only
        # ------------------------------------------------------------------
        common_kw: Dict = dict(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=current_expand_pool(),
        )
        profit_kw: Dict = dict(**common_kw, R=self.R, C=self.C)

        if pa:
            base_repair: List[Tuple[str, Callable, bool]] = [
                (
                    "GreedyProfit",
                    lambda r, n, _noise=0.0: greedy_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
                (
                    "Regret2Profit",
                    lambda r, n, _noise=0.0: regret_2_profit_insertion(r, n, noise=_noise, **profit_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3Profit",
                        lambda r, n, _noise=0.0: regret_3_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                    (
                        "Regret4Profit",
                        lambda r, n, _noise=0.0: regret_4_profit_insertion(r, n, noise=_noise, **profit_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "SavingsProfit",
                        lambda r, n, _noise=0.0: savings_profit_insertion(r, n, **profit_kw),
                        False,  # savings_profit_insertion has no noise parameter; _noise ignored
                    ),
                ]
        else:
            base_repair = [
                (
                    "Greedy",
                    lambda r, n, _noise=0.0: greedy_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
                (
                    "Regret2",
                    lambda r, n, _noise=0.0: regret_2_insertion(r, n, noise=_noise, **common_kw),
                    True,
                ),
            ]
            if regret_pool in ("regret234", "regretAll"):
                base_repair += [
                    (
                        "Regret3",
                        lambda r, n, _noise=0.0: regret_3_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                    (
                        "Regret4",
                        lambda r, n, _noise=0.0: regret_4_insertion(r, n, noise=_noise, **common_kw),
                        True,
                    ),
                ]
            if regret_pool == "regretAll":
                base_repair += [
                    (
                        "Savings",
                        lambda r, n, _noise=0.0: savings_insertion(r, n, **common_kw),
                        False,  # savings_insertion has no noise parameter; _noise ignored
                    ),
                ]

        # ------------------------------------------------------------------
        # Expand each base heuristic:
        #   supports_noise=True  → [<Name>Clean (even idx), <Name>Noisy (odd idx)]
        #   supports_noise=False → [<Name>] (single slot, always clean)
        #
        # All slots are called as repair_op(routes, removed, noise_val) from
        # _select_and_apply_operators, so every lambda accepts a 3rd positional
        # _noise argument (even if it is discarded for clean / no-noise ops).
        # ------------------------------------------------------------------
        ops: List[Callable] = []
        names: List[str] = []
        noisy_flags: List[bool] = []

        for label, fn, supports_noise in base_repair:
            if supports_noise:
                # Clean slot: always passes 0.0; the _noise arg (from the call site) is discarded.
                # Using positional args avoids any keyword-name mismatch across base lambdas.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, 0.0))
                names.append(f"{label}Clean")
                noisy_flags.append(False)
                # Noisy slot: forwards _noise (the noise value computed by _get_noise) positionally.
                ops.append(lambda r, n, _noise=0.0, _fn=fn: _fn(r, n, _noise))
                names.append(f"{label}Noisy")
                noisy_flags.append(True)
            else:
                # Single slot: no noise expansion; fn already accepts and discards _noise=0.0.
                ops.append(fn)
                names.append(label)
                noisy_flags.append(False)

        return ops, names, noisy_flags
