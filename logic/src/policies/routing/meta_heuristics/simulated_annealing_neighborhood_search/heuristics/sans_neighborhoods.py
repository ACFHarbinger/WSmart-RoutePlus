"""
Cheapest insertion of one bin into a depot-terminated SANS route.

The other SANS neighbourhood moves live in
``logic.src.policies.helpers.operators.perturbation_shaking.sans`` and
``logic.src.policies.helpers.operators.evolutionary_mutation.random_2opt``.

Attributes:
    insert_bin_in_route: Insert a bin into a route.

Example:
    >>> new_route = insert_bin_in_route([0, 1, 2, 0], 3, id_to_index, distance_matrix)
"""

__all__ = ["insert_bin_in_route"]


def insert_bin_in_route(route: list, bin_id: int, id_to_index: dict, distance_matrix) -> list:
    """
    Insert a bin into a route at the position minimizing cost increase.

    Args:
        route: Current route.
        bin_id: Bin to insert.
        id_to_index: Mapping from bin ID to matrix index.
        distance_matrix: Distance matrix.

    Returns:
        New route with the inserted bin.
    """
    best_route = route[:]
    best_increase = float("inf")

    # Start and end positions for insertion (between depots)
    for i in range(1, len(route)):
        prev_node = route[i - 1]
        next_node = route[i]

        idx_prev = id_to_index.get(prev_node, prev_node)
        idx_next = id_to_index.get(next_node, next_node)
        idx_bin = id_to_index.get(bin_id, bin_id)

        try:
            increase = (
                distance_matrix[idx_prev][idx_bin]
                + distance_matrix[idx_bin][idx_next]
                - distance_matrix[idx_prev][idx_next]
            )
        except (IndexError, TypeError):
            continue

        if increase < best_increase:
            best_increase = increase
            best_route = route[:i] + [bin_id] + route[i:]

    return best_route
