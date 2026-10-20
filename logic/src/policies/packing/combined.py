"""
Combined Packing Algorithm Module.

This module implements a strategy that combines multiple other selection
strategies using logical operators (OR, AND). This allows creating complex rules
like "Select if (Revenue > X OR Fill > Y)".

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_combined import CombinedSelection
    >>> strategy = CombinedSelection(strategies=[...], logic="or")
    >>> bins = strategy.pack(context)
"""

from dataclasses import fields, replace
from typing import Any, Dict, List, Optional, Tuple, cast

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.utils.registry import ALL

from .base.registry import PackingRegistry


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.ORCHESTRATOR,
)
@PackingRegistry.register("combined")
class CombinedSelection(IPackingAlgorithm):
    """Combines multiple packing algorithms with logical OR or AND.

    Attributes:
        strategy_configs (List[Dict[str, Any]]): List of strategy configurations.
        logic (str): Logical operator ('or', 'and').
    """

    def __init__(
        self,
        strategies: Optional[List[Dict[str, Any]]] = None,
        combined_strategies: Optional[List[Dict[str, Any]]] = None,
        logic: str = "or",
    ):
        """
        Initialize CombinedSelection.

        Args:
            strategies: List of strategy configurations.
            combined_strategies: Alias for strategies (used in PackingConfig).
            logic: Logical operator ('or', 'and').
        """
        self.strategy_configs = strategies or combined_strategies or []
        self.logic = logic.lower()
        if self.logic not in ["or", "and"]:
            raise ValueError(f"Unknown logic: {self.logic}. Must be 'or' or 'and'.")

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins based on combined strategies.

        Aggregates results from multiple underlying strategies using the specified
        logical operator.

        Args:
            context: The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        if not self.strategy_configs:
            return [], SearchContext.initialize(selection_metrics={"strategy": "CombinedSelection"})

        # Instantiate strategies
        from .base.factory import PackingFactory

        strategies = []
        for config in self.strategy_configs:
            name = config.get("name")
            params = config.get("params", {})
            if name:
                strategy = PackingFactory.create_strategy(name, **params)
                strategies.append((strategy, params))

        if not strategies:
            return [], SearchContext.initialize(selection_metrics={"strategy": "CombinedSelection"})

        # Execute first strategy
        first_strat, first_params = strategies[0]
        ctx_0 = self._update_context(context, first_params)

        mandatory_0, master_ctx = first_strat.pack(ctx_0)
        result_set = set(mandatory_0)

        # Merge metrics into master_ctx
        master_ctx = master_ctx.merge(
            SearchContext.initialize(selection_metrics={"strategy": "CombinedSelection", "logic": self.logic})
        )

        for strategy, params in strategies[1:]:
            ctx_i = self._update_context(context, params)
            current_bins_list, sub_ctx = strategy.pack(ctx_i)
            current_bins = set(current_bins_list)

            result_set = result_set.union(current_bins) if self.logic == "or" else result_set.intersection(current_bins)
            master_ctx = master_ctx.merge(sub_ctx)

        return sorted(list(result_set)), master_ctx

    def _update_context(self, context: PackingContext, params: Dict[str, Any]) -> PackingContext:
        """Create a shallow copy of context with updated parameters.

        Args:
            context: The original selection context.
            params: Parameters to update in the context.

        Returns:
            PackingContext: The updated selection context.
        """
        # Copy every child parameter that PackingContext actually stores.
        # Constructor kwargs cover strategies that read their own fields;
        # the rest (last-minute's threshold, and any strategy that only
        # reads the context) need the same values on the context.
        context_fields = {item.name for item in fields(PackingContext)} - {"bin_ids", "current_fill"}
        updates = {key: value for key, value in params.items() if key in context_fields}
        if "threshold" in updates:
            updates["threshold"] = float(updates["threshold"])
        if updates:
            return replace(context, **cast(Dict[str, Any], updates))
        return context
