"""GP tree node types (constant, terminal, function), the protected division helper, and the terminal and function sets used by the genetic operators."""

from typing import (
    Dict,
    Tuple,
    Union,
)


def protected_div(a: float, b: float) -> float:
    """Protected division: returns 1.0 if denominator is near-zero.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        float: Result of the division.
    """
    return a / b if abs(b) > 1e-9 else 1.0


class ConstantNode:
    """
    Ephemeral Random Constant (ERC) node: stores a fixed random scalar.

    Attributes:
        __slots__: Tuple[str, ...] = ("val",)
        val: Randomly generated float value.
    """

    __slots__: Tuple[str, ...] = ("val",)

    def __init__(self, val: float):
        """Initialize the constant node with a random value.

        Args:
            val: Fixed random scalar value.
        """
        self.val = val

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """Return the stored constant value.

        Args:
            ctx: Insertion-context dictionary.

        Returns:
            float: Constant value.
        """
        return self.val

    def copy(self) -> "ConstantNode":
        """Create a deep copy of this constant node.

        Returns:
            ConstantNode: Deep copy of the constant node.
        """
        return ConstantNode(self.val)

    def size(self) -> int:
        """Return the size of this constant node (always 1).

        Returns:
            int: Size of the constant node.
        """
        return 1

    def depth(self) -> int:
        """Return the maximum depth of this constant (always 1).

        Returns:
            int: Maximum depth of the constant.
        """
        return 1

    def compile(self) -> str:
        """Return the string representation of the constant.

        Returns:
            str: String representation of the constant.
        """
        return str(self.val)


class TerminalNode:
    """
    GP terminal node: extracts a continuous feature from the insertion context.

    Attributes:
        __slots__: tuple[str, ...] = ("feature",)
        feature: Name of the feature to extract from the context dictionary.
    """

    __slots__: Tuple[str, ...] = ("feature",)

    def __init__(self, feature: str):
        """Initialize the terminal node for a specific context feature.

        Args:
            feature: Name of the feature to extract.
        """
        self.feature = feature

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Evaluate the terminal by extracting its feature value.

        Args:
            ctx: Insertion-context dictionary.

        Returns:
            Float value of the feature (default 0.0 if not found).
        """
        return ctx.get(self.feature, 0.0)

    def copy(self) -> "TerminalNode":
        """Create a deep copy of this terminal node.

        Returns:
            TerminalNode: Deep copy of the terminal node.
        """
        return TerminalNode(self.feature)

    def size(self) -> int:
        """Return the size of this terminal node (always 1).

        Returns:
            int: Size of the terminal node.
        """
        return 1

    def depth(self) -> int:
        """Return the maximum depth of this terminal (always 1).

        Returns:
            int: Maximum depth of the terminal.
        """
        return 1

    def compile(self) -> str:
        """Return the feature name as a raw Python variable string.

        Returns:
            str: Feature name.
        """
        return self.feature


class FunctionNode:
    """
    GP function node: applies an arithmetic operation to two child subtrees.

    All operations return continuous float values, maintaining the Closure
    Property required for type-consistent GP.

    Attributes:
        fn: Name of the arithmetic function (ADD, SUB, MUL, DIV, MAX, MIN).
        left: Left child subtree.
        right: Right child subtree.
    """

    __slots__ = ("fn", "left", "right")

    def __init__(self, fn: str, left: "GPNode", right: "GPNode"):
        """Initialize the function node with an operation and two subtrees.

        Args:
            fn: Name of the arithmetic operation (ADD, SUB, etc.).
            left: Left child subtree.
            right: Right child subtree.
        """
        self.fn = fn
        self.left = left
        self.right = right

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Recursively evaluate the arithmetic function.

        Uses protected division (returns 1.0 when denominator ≈ 0) to
        prevent runtime errors and maintain closure.

        Args:
            ctx: Insertion-context dictionary passed to children.

        Returns:
            Float result of the arithmetic operation.
        """
        left_val = self.left.evaluate(ctx)
        right_val = self.right.evaluate(ctx)

        if self.fn == "ADD":
            return left_val + right_val
        elif self.fn == "SUB":
            return left_val - right_val
        elif self.fn == "MUL":
            return left_val * right_val
        elif self.fn == "DIV":
            # Protected division: return 1.0 if denominator is near-zero
            return protected_div(left_val, right_val)
        elif self.fn == "MAX":
            return max(left_val, right_val)
        elif self.fn == "MIN":
            return min(left_val, right_val)
        else:
            return 0.0  # Fallback for unknown functions

    def copy(self) -> "FunctionNode":
        """Create a deep copy of this function node and its subtrees.

        Returns:
            FunctionNode: Deep copy of the function node.
        """
        return FunctionNode(
            self.fn,
            self.left.copy(),
            self.right.copy(),
        )

    def size(self) -> int:
        """Recursively calculate the total number of nodes in this subtree.

        Returns:
            int: Total number of nodes in the subtree.
        """
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        """Recursively calculate the maximum depth of this subtree.

        Returns:
            int: Maximum depth of the subtree.
        """
        return 1 + max(self.left.depth(), self.right.depth())

    def compile(self) -> str:
        """Recursively build a Python-parsable mathematical expression string.

        Returns:
            str: Python expression string.
        """
        l = self.left.compile()
        r = self.right.compile()

        if self.fn == "ADD":
            return f"({l} + {r})"
        elif self.fn == "SUB":
            return f"({l} - {r})"
        elif self.fn == "MUL":
            return f"({l} * {r})"
        elif self.fn == "DIV":
            # Map DIV to the protected_div helper in the compiled string
            return f"protected_div({l}, {r})"
        elif self.fn == "MAX":
            return f"max({l}, {r})"
        elif self.fn == "MIN":
            return f"min({l}, {r})"
        return "0.0"


GPNode = Union[TerminalNode, FunctionNode, ConstantNode]

# ---------------------------------------------------------------------------
# Terminals & Functions
# ---------------------------------------------------------------------------
# Local insertion features — synchronised with solver._build_insertion_context().
_TERMINALS = [
    "node_profit",  # Revenue of candidate node
    "distance_to_route",  # Min distance from candidate to nearest route node
    "insertion_cost",  # Delta route distance at cheapest insertion position
    "remaining_capacity",  # Vehicle capacity remaining after insertion
]

# Available arithmetic functions (all return float → Closure Property)
_FUNCTIONS = ["ADD", "SUB", "MUL", "DIV", "MAX", "MIN"]
