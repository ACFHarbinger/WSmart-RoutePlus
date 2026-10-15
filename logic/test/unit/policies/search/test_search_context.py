"""
Unit tests for functional SearchContext tracking.
"""

import pytest
from logic.src.interfaces.context.search_context import (
    AcceptanceMetrics,
    ConstructionMetrics,
    ImprovementMetrics,
    SearchContext,
    SearchPhase,
    SelectionMetrics,
    merge_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_search_context_initialization():
    """Verify context initialization."""
    sel_metrics: SelectionMetrics = {"strategy": "TestSelection"}
    ctx = SearchContext.initialize(selection_metrics=sel_metrics)

    assert ctx.selection_metrics["strategy"] == "TestSelection"
    assert ctx.construction_metrics == {}
    assert ctx.improvement_metrics == {}


def test_merge_construction():
    """Verify merging construction metrics (Phase 2)."""
    ctx = SearchContext.initialize(selection_metrics={"strategy": "Sel"})
    cons_metrics: ConstructionMetrics = {"algorithm": "Exact", "n_mandatory": 5, "profit": 100.0}

    new_ctx = merge_context(ctx, phase=SearchPhase.ROUTING, construction_metrics=cons_metrics)

    # Original should be untouched (immutable)
    assert ctx.construction_metrics == {}
    # New should have metrics
    assert new_ctx.construction_metrics["algorithm"] == "Exact"
    assert new_ctx.construction_metrics["profit"] == 100.0


def test_merge_improvement():
    """Verify merging improvement metrics (Phase 3)."""
    ctx = SearchContext.initialize(selection_metrics={"strategy": "Sel"})
    imp_metrics: ImprovementMetrics = {"algorithm": "SA", "n_iterations": 10}

    new_ctx = merge_context(ctx, phase=SearchPhase.IMPROVEMENT, improvement_metrics=imp_metrics)

    assert ctx.improvement_metrics == {}
    assert new_ctx.improvement_metrics["algorithm"] == "SA"

    # Merge second improver (it will overwrite or merge)
    imp_metrics2: ImprovementMetrics = {"best_delta": -5.0}
    final_ctx = merge_context(new_ctx, phase=SearchPhase.IMPROVEMENT, improvement_metrics=imp_metrics2)

    assert final_ctx.improvement_metrics["algorithm"] == "SA"
    assert final_ctx.improvement_metrics["best_delta"] == -5.0


def test_full_flow():
    """Verify end-to-end telemetry flow."""
    # 1. Selection
    ctx = SearchContext.initialize({"strategy": "Learned"})

    # 2. Construction
    ctx = merge_context(ctx, phase=SearchPhase.ROUTING, construction_metrics={"algorithm": "AM", "profit": 50.0})

    # 3. Improvement
    ctx = merge_context(ctx, phase=SearchPhase.IMPROVEMENT, improvement_metrics={"algorithm": "LS", "best_delta": -5.0})

    assert ctx.selection_metrics["strategy"] == "Learned"
    assert ctx.construction_metrics["algorithm"] == "AM"
    assert ctx.improvement_metrics["algorithm"] == "LS"


def test_acceptance_trace():
    """Verify appending to acceptance trace."""
    ctx = SearchContext.initialize()
    trace1: AcceptanceMetrics = {"criterion": "bmc", "accepted": True}
    ctx = merge_context(ctx, acceptance_metrics=trace1)

    assert len(ctx.acceptance_trace) == 1
    assert ctx.acceptance_trace[0]["criterion"] == "bmc"

    trace2: AcceptanceMetrics = {"criterion": "bmc", "accepted": False}
    ctx = merge_context(ctx, acceptance_metrics=trace2)

    assert len(ctx.acceptance_trace) == 2
    assert ctx.acceptance_trace[1]["accepted"] is False
