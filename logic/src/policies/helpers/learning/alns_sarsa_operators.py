"""Repair, unstringing and perturbation operators of the SARSA-driven ALNS.

Attributes:
    ALNSSARSARepairMixin: Repair and perturbation operators of ``ALNSSARSASolver``.

Example:
    >>> class ALNSSARSASolver(ALNSSARSARepairMixin): ...
"""

from random import Random
from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion import (
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
)
from logic.src.policies.helpers.operators.perturbation_shaking import kick as kick_op
from logic.src.policies.helpers.operators.perturbation_shaking import kick_profit as kick_profit_op
from logic.src.policies.helpers.operators.perturbation_shaking import perturb as perturb_op
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion as greedy_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion_with_blinks as greedy_insertion_with_blinks_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_profit_insertion as greedy_profit_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_profit_insertion_with_blinks as greedy_profit_insertion_with_blinks_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_2_insertion as regret_2_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_2_profit_insertion as regret_2_profit_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_k_insertion as regret_k_insertion_op,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    regret_k_profit_insertion as regret_k_profit_insertion_op,
)

from .alns_perturbation_context import ALNSPerturbationContext


class ALNSSARSARepairMixin:
    """Repair, unstringing and perturbation operators of :class:`ALNSSARSASolver`."""

    # Attribute types from ALNSSARSASolver.__init__. Annotation only.
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    mandatory_nodes: List[int]
    expand_pool: bool
    profit_aware_operators: bool
    random: Random
    params: Any

    def _repair_greedy(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Greedy insertion.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return greedy_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return greedy_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.expand_pool,
        )

    def _repair_regret2(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Regret-2 insertion.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return regret_2_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.expand_pool,
            )
        return regret_2_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=self.expand_pool,
        )

    def _repair_regretk(self, routes: List[List[int]], removed: List[int], k: int) -> List[List[int]]:
        """Regret-k insertion (extended from regret-2).

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.
            k: Number of alternatives to consider for regret.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return regret_k_profit_insertion_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                k=k,
                expand_pool=self.expand_pool,
            )
        return regret_k_insertion_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            k=k,
            expand_pool=self.expand_pool,
        )

    def _repair_greedy_blink(
        self, routes: List[List[int]], removed: List[int], blink_rate: float = 0.1
    ) -> List[List[int]]:
        """Greedy insertion with randomized blinks.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.
            blink_rate: Probability of skipping a potential insertion point.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return greedy_profit_insertion_with_blinks_op(
                routes,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                blink_rate=blink_rate,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
                expand_pool=self.expand_pool,
            )
        return greedy_insertion_with_blinks_op(
            routes,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            blink_rate=blink_rate,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_i(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-I stringing repair.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                1,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            1,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_ii(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-II stringing repair.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                2,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_iii(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-III stringing repair.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                3,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _repair_string_type_iv(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Type-IV stringing repair.

        Args:
            routes: Current set of routes.
            removed: List of nodes to reinsert.

        Returns:
            Repaired routes.
        """

        if self.profit_aware_operators:
            return stringing_profit_insertion(
                routes,
                removed,
                4,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.random,
                self.expand_pool,
            )
        return stringing_insertion(
            routes,
            removed,
            4,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            expand_pool=self.expand_pool,
        )

    def _unstring_wrapper(self, routes: List[List[int]], n: int, op_type: int) -> Tuple[List[List[int]], List[int]]:
        """Wrapper to apply unstringing moves globally as a destroy operator.

        Args:
            routes: Current set of routes.
            n: Max number of nodes to remove.
            op_type: Type of unstringing operation (1-4).

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        if self.profit_aware_operators:
            return unstringing_profit_removal(
                routes,
                n,
                op_type,
                self.dist_matrix,
                self.wastes,
                self.R,
                self.C,
                rng=self.random,
            )
        return unstringing_removal(routes, n, op_type, self.dist_matrix, rng=self.random)

    def _unstring_type_i(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-I unstringing.

        Args:
            routes: Current set of routes.
            n: Max number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return self._unstring_wrapper(routes, n, 1)

    def _unstring_type_ii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-II unstringing.

        Args:
            routes: Current set of routes.
            n: Max number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return self._unstring_wrapper(routes, n, 2)

    def _unstring_type_iii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-III unstringing.

        Args:
            routes: Current set of routes.
            n: Max number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return self._unstring_wrapper(routes, n, 3)

    def _unstring_type_iv(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Type-IV unstringing.

        Args:
            routes: Current set of routes.
            n: Max number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return self._unstring_wrapper(routes, n, 4)

    def _perturb_kick(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: removes random nodes and reinserts them greedily.

        Args:
            routes: Current set of routes.

        Returns:
            Perturbed routes.
        """

        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        if self.profit_aware_operators:
            kick_profit_op(ctx, destroy_ratio=0.2, bias=2.0, rng=self.random)
        else:
            kick_op(ctx, destroy_ratio=0.2, rng=self.random)
        return ctx.routes

    def _perturb_random(self, routes: List[List[int]]) -> List[List[int]]:
        """Perturbation operator: performs random swaps to escape local optima.

        Args:
            routes: Current set of routes.

        Returns:
            Perturbed routes.
        """

        ctx = ALNSPerturbationContext(routes, self.dist_matrix, self.wastes, self.capacity)
        perturb_op(
            ctx,
            k=self.params.perturb_k if hasattr(self.params, "perturb_k") else 3,
            prob_unvisited=self.params.prob_unvisited if hasattr(self.params, "prob_unvisited") else 0.0,
            rng=self.random,
        )
        return ctx.routes
