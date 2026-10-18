"""
Random removal of bins from depot-terminated routes into a pool of removed bins.

These are the removal moves of the LAC and SANS annealers: random perturbations
without an improvement check, which leave mandatory bins in place. The LAC
operators mutate ``routes_list`` and append to a ``removed_bins`` list; the
SANS operators return a copy of the routes and add to a ``removed_bins`` set.

Attributes:
    remove_n_bins_random: (LAC) Remove n random bins from one route (n=1 removes one bin).
    remove_n_bins_consecutive: (LAC) Remove n consecutive non-mandatory bins from one route.
    remove_bins_from_route: (SANS) Copy with random non-mandatory bins removed from one route.
    remove_n_bins_across_routes: (SANS) Copy with n random non-mandatory bins removed from any routes.
    remove_segment_without_mandatory: (SANS) Copy with the first mandatory-free segment of n bins, in
        random order of start positions, removed from one route.

Example:
    >>> from logic.src.policies.helpers.operators.destroy_ruin.bin_pool import remove_n_bins_random
    >>> routes, removed = [[0, 1, 2, 3, 0]], []
    >>> remove_n_bins_random(routes, removed, [], rng, n=1)
"""

import copy
import random
from random import Random


def remove_n_bins_random(
    routes_list,
    removed_bins,
    bins_cannot_removed,
    rng,
    n=None,
    require_spare_bin=True,
    max_resamples=100,
    prune_chosen_empty_route=False,
):
    """
    Remove n random bins from one random route and append them to the removed list.

    Args:
        routes_list: Current routes (depot-terminated), mutated in place.
        removed_bins: Removed bins, extended in place.
        bins_cannot_removed: Mandatory bins.
        rng: Random number generator.
        n: Number of bins to remove; drawn from 2-5 when None.
        require_spare_bin: If True, the route must hold more than n + len(bins_cannot_removed) bins;
            if False, at least that many.
        max_resamples: Redraws allowed while the drawn bin is mandatory; after the last one the bin is
            removed even if mandatory. None redraws until a removable bin is drawn.
        prune_chosen_empty_route: Also drop the chosen route when it held no bins to begin with.

    Returns:
        Tuple[List[int], int]: The removed bins and n.
    """
    bins_to_remove_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = n if n is not None else rng.sample(possible_n, 1)[0]
    n_routes = len(routes_list)
    bin_to_remove = None
    if n_routes > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
        spare = len(chosen_route) - 2 - len(bins_cannot_removed) - chosen_n
        if spare > 0 or (spare == 0 and not require_spare_bin):
            for _ in range(0, chosen_n):
                if len(bins_cannot_removed) > 0:
                    bin_to_remove = rng.sample(bins_cannot_removed, 1)[0]
                    i = 0
                    while bin_to_remove in bins_cannot_removed and (max_resamples is None or i <= max_resamples):
                        bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                        i += 1
                else:
                    bin_to_remove = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                removed_bins.append(bin_to_remove)
                chosen_route.remove(bin_to_remove)
                bins_to_remove_random.append(bin_to_remove)
            if len(chosen_route) == 2:
                routes_list.remove(chosen_route)
        elif prune_chosen_empty_route and len(chosen_route) == 2:
            routes_list.remove(chosen_route)
    return bins_to_remove_random, chosen_n


def _extract_valid_segment(
    chosen_route: list[int], chosen_n: int, bins_cannot_removed: list[int], rng: random.Random
) -> list[int]:
    """Attempt to find a valid segment of length chosen_n to remove.

    Args:
        chosen_route: Route from which to remove bins.
        chosen_n: Number of bins to remove.
        bins_cannot_removed: Bins that cannot be removed.
        rng: Random number generator.

    Returns:
        list[int]: Valid segment of length chosen_n to remove.
    """
    # Try finding a valid segment up to 100 times
    # Note: original loop just tried 100 times randomly.
    # A more deterministic approach would be sliding window, but let's stick to the stochastic nature.

    if len(chosen_route) - 2 < chosen_n:
        return []

    for _ in range(100):
        # start index range: 1 to len - 1 - chosen_n
        max_start = len(chosen_route) - 1 - chosen_n
        if max_start < 1:
            break

        start_idx = rng.sample(range(1, max_start + 1), 1)[0]  # rsample returns list

        # Check if segment contains mandatory bins
        segment = chosen_route[start_idx : start_idx + chosen_n]
        if not any(b in bins_cannot_removed for b in segment):
            return segment

    return []


def remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed, rng):
    """
    Remove n consecutive bins from a route.

    Args:
        routes_list: Current routes.
        removed_bins: Target set.
        bins_cannot_removed: Mandatory nodes.
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    bins_to_remove_consecutive = []  # type: ignore[var-annotated]
    possible_n = [2, 3, 4, 5]
    chosen_n = rng.sample(possible_n, 1)[0]

    if not routes_list:
        return bins_to_remove_consecutive, chosen_n

    chosen_route = rng.sample(routes_list, 1)[0]

    # Calculate if possible to remove n

    # Check if route is long enough and feasible
    # Optimization: count mandatory bins in route
    mandatory_in_route = sum(1 for b in chosen_route if b in bins_cannot_removed)
    available_bins = len(chosen_route) - 2  # excluding depots

    if available_bins - mandatory_in_route < chosen_n:
        # Not enough removable bins total, certainly no consecutive segment of n
        return bins_to_remove_consecutive, chosen_n

    if not bins_cannot_removed:
        # Fast path: just pick any segment
        if available_bins >= chosen_n:
            max_start = len(chosen_route) - 1 - chosen_n
            start_idx = rng.sample(range(1, max_start + 1), 1)[0]
            segment = chosen_route[start_idx : start_idx + chosen_n]
            bins_to_remove_consecutive.extend(segment)
            removed_bins.extend(segment)
            del chosen_route[start_idx : start_idx + chosen_n]
    else:
        # Complex path with mandatory constraints
        segment = _extract_valid_segment(chosen_route, chosen_n, bins_cannot_removed, rng)
        if segment:
            bins_to_remove_consecutive.extend(segment)
            removed_bins.extend(segment)

            # To delete efficiently, we need index.
            if len(segment) > 0:
                first_bin_idx = chosen_route.index(segment[0])
                del chosen_route[first_bin_idx : first_bin_idx + len(segment)]

    if len(chosen_route) <= 2 and chosen_route in routes_list:
        routes_list.remove(chosen_route)

    # Clean up empty routes
    routes_list[:] = [r for r in routes_list if len(r) > 2]

    return bins_to_remove_consecutive, chosen_n


def remove_bins_from_route(routes_list: list, bins_cannot_removed: set, rng: Random, num_bins: int = 1) -> list:
    """
    Remove random bins from a single route.

    Args:
        routes_list: Solution routes list to mutate.
        bins_cannot_removed: Bins that cannot be removed.
        rng: Random number generator.
        num_bins: Number of bins to remove.

    Returns:
        New routes list with bins removed.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [i for i, r in enumerate(new_routes) if len(r) > 2]

    if not valid_routes:
        return new_routes

    r_idx = rng.choice(valid_routes)
    route = new_routes[r_idx]

    valid_indices = [i for i in range(1, len(route) - 1) if route[i] not in bins_cannot_removed]
    num_to_remove = min(num_bins, len(valid_indices))

    if num_to_remove <= 0:
        return new_routes

    indices_to_remove = set(rng.sample(valid_indices, num_to_remove))
    new_routes[r_idx] = [n for i, n in enumerate(route) if i not in indices_to_remove]
    return new_routes


def remove_n_bins_across_routes(
    routes_list: list, removed_bins: set, bins_cannot_removed: set, rng: Random, n: int = 2
) -> list:
    """
    Remove n random bins from routes and add them to the removed pool.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        bins_cannot_removed: Bins that must stay in routes.
        rng: Random number generator.
        n: Number of bins to remove.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    all_removable = []
    for r in new_routes:
        for b in r[1:-1]:
            if b not in bins_cannot_removed:
                all_removable.append(b)

    if not all_removable:
        return new_routes

    to_remove = rng.sample(all_removable, min(n, len(all_removable)))
    for b in to_remove:
        removed_bins.add(b)
        for _i, r in enumerate(new_routes):
            if b in r:
                r.remove(b)
                break
    return new_routes


def remove_segment_without_mandatory(
    routes_list: list, removed_bins: set, bins_cannot_removed: set, rng: Random, n: int = 2
) -> list:
    """
    Remove a consecutive sequence of bins from a route.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        bins_cannot_removed: Bins that must stay in routes.
        rng: Random number generator.
        n: Number of bins to remove.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [i for i, r in enumerate(new_routes) if len(r) >= n + 2]
    if not valid_routes:
        return new_routes

    r_idx = rng.choice(valid_routes)
    route = new_routes[r_idx]

    # Try to find a segment without 'mandatory' bins
    start_indices = list(range(1, len(route) - n))
    rng.shuffle(start_indices)

    for start in start_indices:
        segment = route[start : start + n]
        if not any(b in bins_cannot_removed for b in segment):
            for b in segment:
                removed_bins.add(b)
            new_routes[r_idx] = route[:start] + route[start + n :]
            break

    return new_routes


def remove_bins_end(routes_list, removed_bins, bins_cannot_removed, profit_fn, real_profit_fn):
    """
    Greedily drop non-mandatory bins wherever the objective improves.

    Args:
        routes_list: Initial routes (depot-terminated).
        removed_bins: Removed bins, extended in place.
        bins_cannot_removed: Mandatory bins.
        profit_fn: Objective to maximise, such as the penalised profit.
        real_profit_fn: Objective reported for the result, such as the unpenalised profit.

    Returns:
        Tuple[List[List[int]], float, float, List[int]]: Solution, its ``profit_fn``, its
        ``real_profit_fn`` and the removed bins.
    """
    previous_solution = copy.deepcopy(routes_list)
    previous_profit = profit_fn(previous_solution)
    for it, i in enumerate(previous_solution):
        i = routes_list[it]
        for j in i[1 : len(i) - 1]:
            i = routes_list[it]
            if j not in bins_cannot_removed:
                i.remove(j)
            new_profit = profit_fn(routes_list)
            if new_profit > previous_profit:
                previous_solution = copy.deepcopy(routes_list)
                previous_profit = new_profit
                removed_bins.append(j)
            else:
                routes_list = copy.deepcopy(previous_solution)
    previous_real_profit = real_profit_fn(previous_solution)
    return previous_solution, previous_profit, previous_real_profit, removed_bins
