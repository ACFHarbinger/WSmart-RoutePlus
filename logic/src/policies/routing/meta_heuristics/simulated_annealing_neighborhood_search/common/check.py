"""
Consistency and feasibility validation for look-ahead routing solutions.

This module provides routines to verify that routing solutions adhere to
physical constraints (tanker capacity) and temporal predictions (overflow
risk). It is used during the local search process to prune inadmissible
mutations and ensure valid final outputs for the simulator.

Attributes:
    None

Example:
    None
"""

# Functions for condition checks


# Check route feasibility
# function to check routes feasibility regarding number of overflowing bins allowed
# the visited number of bins to which stock + accumulation_rate >= capacity has to be greater
# or equal than the number of bins that verify this condition minus the total number of bins
# multiplied by the percentage of bins that can overflow
def check_bins_overflowing_feasibility(data, routes_list, number_of_bins, perc_bins_can_overflow, E, B):
    """
    Check if the solution is feasible regarding the number of overflowing bins.

    A solution is considered feasible if the number of bins currently in the
    routes that were predicted to overflow (or are near capacity) satisfies
     the tolerance threshold.

    Args:
        data: Bin data containing 'Stock' and 'Accum_Rate'.
        routes_list: Current routing solution.
        number_of_bins: Total number of bins in the area.
        perc_bins_can_overflow: Threshold (delta) for allowed overflow.
        E: Bin volume.
        B: Bin density.

    Returns:
        str: 'pass' if feasible, 'fail' otherwise.
    """
    threshold = E * B
    mask = (data["Stock"] + data["Accum_Rate"]) >= threshold
    if "#bin" in data.columns:
        overflowing_bins = set(data.loc[mask, "#bin"])
    else:
        overflowing_bins = set(data.index[mask])

    number_bins_overflowing = len(overflowing_bins)
    check = number_bins_overflowing - number_of_bins * perc_bins_can_overflow

    visited = {node for route in routes_list for node in route}
    total_ovf_bins_in_routes = len(overflowing_bins & visited)

    return "pass" if total_ovf_bins_in_routes >= check else "fail"


# Check route admissibility
# checks if every function is working fine and admissible solutions are being generated
def check_solution_admissibility(routes_list, removed_bins, number_of_bins):
    """
    Verify if all bins are accounted for in the routes or removed set.

    Args:
        routes_list: Current routing solution.
        removed_bins: Set of bins not collected.
        number_of_bins: Total number of collectable bins.

    Returns:
        bool: True if the solution is consistent, False otherwise.
    """
    total_length = 0
    for i in routes_list:
        total_length += len(i)
    admissible = total_length + len(removed_bins) == number_of_bins + 2 * len(routes_list)
    return admissible
