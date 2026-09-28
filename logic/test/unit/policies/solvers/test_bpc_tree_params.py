"""Regression tests for B-kimi-34: BranchAndBoundTree must read the real BPCParams fields.

Before the fix, the tree read non-existent aliases (``max_branch_nodes``,
``tree_search_strategy``) via ``getattr`` fallbacks, so a params-only caller
silently got ``best_first`` search and a 1000-node cap regardless of the
configured ``search_strategy`` / ``max_bb_nodes``, and the engine's explicit
arguments triggered a DeprecationWarning on every run.
"""

import warnings

import pytest

from logic.src.policies.helpers.solvers_and_matheuristics.branching.tree import BranchAndBoundTree
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_price_and_cut.params import (
    BPCParams,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_tree_reads_real_params_fields() -> None:
    """A params-only construction must honour search_strategy and max_bb_nodes."""
    params = BPCParams.from_config(
        {
            "search_strategy": "depth_first",
            "branching_strategy": "divergence",
            "max_bb_nodes": 7,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any DeprecationWarning fails the test
        tree = BranchAndBoundTree(v_model=None, params=params)
    assert tree.search_strategy == "depth_first"
    assert tree.strategy == "divergence"
    assert tree.max_nodes == 7


def test_tree_defaults_without_params() -> None:
    """The legacy no-params path keeps its documented defaults."""
    tree = BranchAndBoundTree(v_model=None)
    assert tree.search_strategy == "best_first"
    assert tree.strategy == "edge"
    assert tree.max_nodes == 1000
