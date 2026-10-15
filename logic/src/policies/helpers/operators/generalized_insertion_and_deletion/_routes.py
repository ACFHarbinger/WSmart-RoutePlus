"""
Route extraction utilities for Generalized Insertion and Deletion.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import _extract_working_route
    >>> n, is_closed, work_route, n_work = _extract_working_route(route)
"""

from typing import List, Tuple


def _extract_working_route(route: List[int]) -> Tuple[int, bool, List[int], int]:
    """
    Extracts working route characteristics to handle open/closed route representations.

    Args:
        route: The tour as a list of node IDs.

    Returns:
        Tuple[int, bool, List[int], int]: (n, is_closed, work_route, n_work)
            - n: Original route length.
            - is_closed: True if the route is a closed tour.
            - work_route: Route without the closing depot if closed.
            - n_work: Length of the working route.
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)
    return n, is_closed, work_route, n_work
