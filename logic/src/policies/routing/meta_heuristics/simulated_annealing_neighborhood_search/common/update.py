"""
Temporal state transition and bin accumulation logic for simulation days.

Manages the evolution of bin fill levels and determines optimal collection
schedules based on predicted overflow. Provides utilities for updating bin
states after collection and identifying 'mandatory' candidates for upcoming
planning cycles.

Attributes:
    MAX_CAPACITY_PERCENT (int): Maximum fill capacity percentage.
    should_bin_be_collected: Check if a bin's fill level will exceed MAX_CAPACITY_PERCENT% by the next day.
    add_bins_to_collect: Predictively identify bins that will overflow before the next planned collection day and add them to the must-collect list.
    update_fill_levels_after_first_collection: Simulate bin emptying after collection by resetting fill levels to zero.
    initialize_lists_of_bins: Initialize a zeroed list for tracking next collection days.
    calculate_next_collection_days: Calculate how many days remain until each mandatory bin overflows again.
    get_next_collection_day: Determine the minimum number of days until the first bin in the set overflows.

Example:
    >>> import numpy as np
    >>> from logic.src.data.bins import Bins
    >>> from logic.src.configs.policies.routing.meta_heuristics.sans import SANSConfig
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.policy_sans import SANSPolicy
    >>>
    >>> # Setup data (same as LAC)
    >>> bin_data = Bins.from_yaml("tests/data/bins.yaml")
    >>> bin_coords = {b.id: (b.location.latitude, b.location.longitude) for b in bin_data.bins}
    >>> customer_location = bin_coords[0]
    >>> bin_state = pd.DataFrame({"fill_level": [0] * len(bin_data.bins), "last_collection": pd.to_datetime(["2024-05-01"] * len(bin_data.bins))})
    >>> # Simulation parameters
    >>> sim_params = SANSConfig(
    ...     max_iterations=10,
    ...     num_sim_days=15,
    ...     initial_temp=100.0,
    ...     cooling_rate=0.95,
    ...     max_route_time_minutes=480,
    ...     vehicle_capacity=12.0,
    ...     route_collection_size_ratio=1.0,
    ...     max_bins_per_route=5,
    ... )
    >>> # Create policy and dispatcher
    >>> policy = SANSPolicy(sim_params, bins_df=bin_data.bins, bin_coordinates=bin_coords)
    >>> dispatcher = Dispatcher(policy=policy, bin_coordinates=bin_coords)
    >>> # Run simulation
    >>> routes_results, bin_state_final = dispatcher.execute_og(
    ...     start_time=pd.to_datetime("2024-05-05"),
    ...     bin_data=bin_data,
    ...     bin_state=bin_state,
    ... )
    >>> print(f"Simulation completed. Routes generated: {len(routes_results)}")
    Simulation completed. Routes generated: 1
"""

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT


def should_bin_be_collected(current_fill_level, accumulation_rate):
    """
    Check if a bin's fill level will exceed MAX_CAPACITY_PERCENT% by the next day.

    Args:
        current_fill_level: Current percentage full.
        accumulation_rate: Daily fill rate.

    Returns:
        bool: True if overflow is predicted, None otherwise.
    """
    if current_fill_level + accumulation_rate >= MAX_CAPACITY_PERCENT:
        return True


def add_bins_to_collect(
    binsids,
    next_collection_day,
    mandatory_bins,
    current_fill_levels,
    accumulation_rates,
    current_collection_day,
):
    """
    Predictively identify bins that will overflow before the next planned
    collection day and add them to the must-collect list.

    Args:
        binsids: All relevant bin IDs.
        next_collection_day: Target future day.
        mandatory_bins: Current set of mandatory collections.
        current_fill_levels: State.
        accumulation_rates: Growth rate.
        current_collection_day: Today.

    Returns:
        List[int]: Updated must-collect list.
    """
    for i in binsids:
        if i in mandatory_bins:
            continue
        else:
            for j in range(current_collection_day + 1, next_collection_day):
                if current_fill_levels[i] + j * accumulation_rates[i] >= MAX_CAPACITY_PERCENT:
                    mandatory_bins.append(i)
                    break
    return mandatory_bins


def update_fill_levels_after_first_collection(binsids, mandatory_bins, current_fill_levels):
    """
    Simulate bin emptying after collection by resetting fill levels to zero.

    Args:
        binsids: All bins.
        mandatory_bins: Bins that were collected.
        current_fill_levels: State to update.

    Returns:
        np.ndarray: Updated fill levels.
    """
    current_fill_levels = current_fill_levels.copy()
    for i in binsids:
        if i in mandatory_bins:
            current_fill_levels[i] = 0
    return current_fill_levels


def initialize_lists_of_bins(binsids):
    """
    Initialize a zeroed list for tracking next collection days.

    Args:
        binsids: Bins to track.

    Returns:
        List[int]: Zero-initialized list.
    """
    next_collection_days = []
    for _i in range(0, len(binsids)):
        next_collection_days.append(0)
    return next_collection_days


def calculate_next_collection_days(mandatory_bins, current_fill_levels, accumulation_rates, binsids):
    """
    Calculate how many days remain until each mandatory bin overflows again.

    Args:
        mandatory_bins: Bins to analyze.
        current_fill_levels: Current state.
        accumulation_rates: Growth rates.
        binsids: Total bins.

    Returns:
        List[int]: Days until next overflow per bin.
    """
    next_collection_days = initialize_lists_of_bins(binsids)
    temporary_fill_levels = current_fill_levels.copy()
    for i in mandatory_bins:
        current_day = 0
        while temporary_fill_levels[i] < MAX_CAPACITY_PERCENT:
            temporary_fill_levels[i] = temporary_fill_levels[i] + accumulation_rates[i]
            current_day = current_day + 1
        next_collection_days[i] = current_day  # assuming collection happens at the beginning of the day
    return next_collection_days


def get_next_collection_day(mandatory_bins, current_fill_levels, accumulation_rates, binsids):
    """
    Determine the minimum number of days until the first bin in the set overflows.

    Args:
        mandatory_bins: Target set.
        current_fill_levels: State.
        accumulation_rates: Daily rates.
        binsids: Bin IDs.

    Returns:
        int: Minimum days to next overflow.
    """
    next_collection_days = calculate_next_collection_days(
        mandatory_bins, current_fill_levels, accumulation_rates, binsids
    )
    next_collection_days_array = np.array(next_collection_days)
    next_collection_day = np.min(next_collection_days_array[np.nonzero(next_collection_days_array)])
    return next_collection_day
