"""
Dummy Problem wrapper for legacy component initialization.

Attributes:
    DummyProblem: Minimal problem wrapper for legacy component initialization.

Example:
    >>> import dummy_problem
    >>> dummy_problem.DummyProblem("test")
"""


class DummyProblem:
    """
    Minimal problem wrapper for legacy component initialization.

    Attributes:
        NAME (str): Name of the problem.
    """

    def __init__(self, name: str):
        """
        Initialize DummyProblem.

        Args:
            name: Name of the problem.
        """
        self.NAME = name
