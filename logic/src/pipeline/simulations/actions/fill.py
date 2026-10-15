"""
Action for daily bin filling.

Attributes:
    FillAction: Command to execute daily bin filling.

Example:
    >>> # action = FillAction()
    >>> # action.execute(context)
"""

from typing import Any, Dict

from .base import SimulationAction, bind_context


class FillAction(SimulationAction):
    """
    Executes daily bin filling simulation.

    Queries the bin management object to update fill levels based on
    either stochastic generation or empirical data from files.

    Attributes:
        None
    """

    timed = False
    reads = ("bins", "day")
    writes = ("new_overflows", "fill", "total_fill", "sum_lost", "overflows")
    finalizes = ()
    requires_final = ()

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute daily bin filling.

        Args:
            context: Shared dictionary containing simulation state.
        """

        context = bind_context(context)
        bins = context.bins
        day = context.day

        new_overflows, fill, total_fill, sum_lost = bins.load_filling(day)
        context.new_overflows = new_overflows
        context.fill = fill
        context.total_fill = total_fill
        context.sum_lost = sum_lost

        # Accumulate overflows in context (if needed by subsequent steps or final return)
        if "overflows" in context:
            context.overflows += new_overflows
