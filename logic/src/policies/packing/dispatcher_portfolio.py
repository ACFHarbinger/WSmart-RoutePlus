"""Portfolio Dispatcher Strategy Module.

Implements an ensemble strategy that aggregates the decisions of multiple
candidate "mandatory" strategies. It supports robust consensus by returning either
the intersection (requires all strategies to agree) or the union (requires
only one strategy to select a bin) of the candidate results.

Attributes:
    None

Example:
    >>> from logic.src.policies.packing.dispatcher_portfolio import PortfolioDispatcher
    >>> strategy = PortfolioDispatcher()
    >>> bins = strategy.pack(context)
"""

from typing import List, Optional, Set, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, PackingContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.ADAPTIVE_ALGORITHM,
)
@PackingRegistry.register("dispatcher_portfolio")
class PortfolioDispatcher(IPackingAlgorithm):
    """Portfolio dispatcher for ensemble selection.

    Attributes:
        None
    """

    def __init__(
        self,
        dispatcher_mode: Optional[str] = None,
        dispatcher_candidate_strategies: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            dispatcher_mode: 'union' or 'intersection'; ``None`` uses ``context.dispatcher_mode``.
            dispatcher_candidate_strategies: Candidate names; ``None`` uses the context.
        """
        self.dispatcher_mode = dispatcher_mode
        self.dispatcher_candidate_strategies = dispatcher_candidate_strategies

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """
        Run multiple strategies and aggregate via union or intersection.

        Args:
            context: PackingContext with portfolio configuration.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs and search context.
        """
        from logic.src.policies.packing.base.factory import PackingFactory

        candidates = (
            self.dispatcher_candidate_strategies
            or context.dispatcher_candidate_strategies
            or ["last_minute", "deadline", "mip_knapsack"]
        )
        mode = self.dispatcher_mode or context.dispatcher_mode or "union"

        if mode not in ("union", "intersection"):
            raise ValueError(f"PortfolioDispatcher: Unknown mode '{mode}'. Use 'union' or 'intersection'.")

        results: List[Set[int]] = []
        master_ctx = SearchContext.initialize(selection_metrics={"strategy": "PortfolioDispatcher", "mode": mode})

        for name in candidates:
            strategy = PackingFactory.create_strategy(name)
            mandatory, sub_ctx = strategy.pack(context)
            results.append(set(mandatory))
            master_ctx = master_ctx.merge(sub_ctx)

        if not results:
            return [], master_ctx

        # union or intersection
        final_set = set().union(*results) if mode == "union" else results[0].intersection(*results[1:])

        return sorted(list(final_set)), master_ctx
