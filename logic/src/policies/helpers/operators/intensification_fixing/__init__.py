"""Intensification operators: Held-Karp DP, Fix-and-Optimize Sub-MIP, and Set-Partitioning Polish.

This package provides three exact intensification operators that apply
Gurobi-backed optimisation or dynamic programming to improve the within-route
sequencing or cross-route assignment of a VRP solution.

Operators:
    - ``dp_route_reopt`` / ``dp_route_reopt_profit``:
        Held-Karp DP re-sequences each route to its exact TSP optimum.
    - ``fix_and_optimize`` / ``fix_and_optimize_profit``:
        Selects the worst routes and re-solves them as a Gurobi Sub-MIP.
    - ``set_partitioning_polish`` / ``set_partitioning_polish_profit``:
        Picks the optimal combination of routes from a pool via Set-Partitioning MIP.

Attributes:
    INTENSIFICATION_OPERATORS (Dict[str, Callable]): Registry mapping operator
        names to their corresponding intensification functions.
    INTENSIFICATION_NAMES (List[str]): Sorted list of operator names.

Example:
    >>> from logic.src.policies.helpers.operators.intensification_fixing import (
    ...     dp_route_reopt, fix_and_optimize, set_partitioning_polish
    ... )
    >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity)
"""

from logic.src.policies.helpers.operators.intensification_fixing.fix_and_optimize_profit import fix_and_optimize_profit

from .dp_route_reopt import dp_route_reopt, dp_route_reopt_profit
from .fix_and_optimize import fix_and_optimize
from .set_partitioning_polish import set_partitioning_polish, set_partitioning_polish_profit

INTENSIFICATION_OPERATORS = {
    "DP_REOPT": dp_route_reopt,
    "DP_REOPT_PROFIT": dp_route_reopt_profit,
    "FIX_OPT": fix_and_optimize,
    "FIX_OPT_PROFIT": fix_and_optimize_profit,
    "SP_POLISH": set_partitioning_polish,
    "SP_POLISH_PROFIT": set_partitioning_polish_profit,
}

INTENSIFICATION_NAMES = list(INTENSIFICATION_OPERATORS.keys())

__all__ = [
    "INTENSIFICATION_NAMES",
    "INTENSIFICATION_OPERATORS",
    "dp_route_reopt",
    "dp_route_reopt_profit",
    "fix_and_optimize",
    "fix_and_optimize_profit",
    "set_partitioning_polish",
    "set_partitioning_polish_profit",
]
