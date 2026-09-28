"""Runtime contract tests for the EGH/LASM params classes (M-mistral-02 revision).

The #82 refactor made ``ExactGuidedHeuristicParams`` and
``LASMPipelineParams`` thin subclasses of their config dataclasses. The
first delivery dropped the derived runtime members that live between the
field block and ``from_config`` (``stage_budgets``, ``alns_iterations``,
``bpc_ng_size``, ``bpc_max_bb_nodes``, ``as_alns_values_dict``) and LASM's
``__post_init__`` — the EGH dispatcher failed before reaching a solver and
LASM's iterable consumers received ``None``. These tests pin every one of
those runtime contracts to the *actual* dispatch paths, so the regression
cannot recur silently.
"""

import pytest

from logic.src.configs.policies import ExactGuidedHeuristicConfig, LASMPipelineConfig
from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic import dispatcher as egh_dispatcher
from logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.params import (
    ExactGuidedHeuristicParams,
)
from logic.src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.params import (
    LASMPipelineParams,
)


# ---------------------------------------------------------------- EGH


def test_egh_stage_budgets_sums_to_time_limit():
    p = ExactGuidedHeuristicParams(alpha=0.5, time_limit=120.0)
    budgets = p.stage_budgets()
    assert len(budgets) == 4
    assert all(b > 0 for b in budgets)
    assert pytest.approx(sum(budgets)) == p.time_limit


def test_egh_derived_scalars():
    p = ExactGuidedHeuristicParams(alpha=1.0, time_limit=60.0)
    assert isinstance(p.alns_iterations(), int) and p.alns_iterations() > 0
    assert p.bpc_ng_size() >= p.bpc_ng_size_min
    assert p.bpc_max_bb_nodes() >= p.bpc_max_bb_nodes_min


def test_egh_as_alns_values_dict_shape():
    p = ExactGuidedHeuristicParams()
    values = p.as_alns_values_dict()
    assert isinstance(values, dict)
    assert values["engine"] == p.alns_engine
    assert values["max_iterations"] == p.alns_iterations()
    assert values["cooling_rate"] == p.alns_cooling_rate


def test_egh_dispatcher_builds_alns_params():
    """The dispatcher's first step for the ALNS stage: it consumes
    ``as_alns_values_dict()`` through ``_build_alns_params`` — the exact
    call that failed before this revision restored the members."""
    p = ExactGuidedHeuristicParams(alpha=0.5, time_limit=30.0)
    alns_params = egh_dispatcher._build_alns_params(p, time_limit=7.5)
    assert alns_params.time_limit == 7.5
    assert alns_params.max_iterations == p.alns_iterations()


def test_egh_from_config_object_roundtrip():
    cfg = ExactGuidedHeuristicConfig(alpha=0.25, time_limit=45.0)
    p = ExactGuidedHeuristicParams.from_config(cfg)
    assert isinstance(p, ExactGuidedHeuristicParams)
    assert p.alpha == 0.25 and p.time_limit == 45.0
    assert p.stage_budgets() is not None


# ---------------------------------------------------------------- LASM


def test_lasm_post_init_materializes_lists():
    p = LASMPipelineParams()
    assert p.lbbd_cut_families == ["nogood", "optimality", "pareto"]
    assert p.rl_state_features is not None
    assert "n_nodes" in p.rl_state_features and "fill_mean" in p.rl_state_features


def test_lasm_lists_are_iterable():
    """Codex's runtime regression: with __post_init__ dropped, consumers
    iterating these fields received None."""
    p = LASMPipelineParams()
    families = [f for f in p.lbbd_cut_families]
    features = [f for f in p.rl_state_features]
    assert len(families) == 3
    assert len(features) >= 5


def test_lasm_explicit_lists_are_kept():
    p = LASMPipelineParams(
        lbbd_cut_families=["nogood"],
        rl_state_features=["alpha"],
    )
    assert p.lbbd_cut_families == ["nogood"]
    assert p.rl_state_features == ["alpha"]


def test_lasm_stage_budgets_and_alns_dict():
    p = LASMPipelineParams(alpha=0.5, time_limit=100.0)
    budgets = p.stage_budgets()
    assert len(budgets) == 5
    assert all(b >= 0 for b in budgets)
    assert pytest.approx(sum(budgets)) == p.time_limit
    values = p.as_alns_values_dict()
    assert isinstance(values, dict)
    assert values["max_iterations"] == p.alns_iterations()


def test_lasm_from_config_object_roundtrip():
    cfg = LASMPipelineConfig(alpha=0.5, time_limit=50.0)
    p = LASMPipelineParams.from_config(cfg)
    assert isinstance(p, LASMPipelineParams)
    assert p.alpha == 0.5
    # __post_init__ ran during construction
    assert p.lbbd_cut_families is not None
