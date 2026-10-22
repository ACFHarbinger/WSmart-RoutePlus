"""WP-09: NumPy packing definitions stay torch-free; EOQ is not last-minute."""

import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest
from logic.src.interfaces.context.packing_context import PackingContext
from logic.src.policies.packing.eoq_last_minute import LastMinuteEOQSelection
from logic.src.policies.packing.last_minute import LastMinuteSelection
from logic.src.policies.packing.spec import last_minute, lookahead, regular, service_level

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _ctx(**kwargs):
    defaults = dict(bin_ids=np.arange(3), current_fill=np.array([50.0, 80.0, 95.0]), threshold=90.0)
    defaults.update(kwargs)
    return PackingContext(**defaults)


def test_last_minute_is_fill_threshold_even_when_eoq_flag_is_set():
    ctx = _ctx(
        use_eoq_threshold=True,
        holding_cost_per_kg_day=0.02,
        ordering_cost_per_visit=50.0,
        bin_volume=2.5,
        bin_density=400.0,
        accumulation_rates=np.array([10.0, 10.0, 10.0]),
        threshold=90.0,
    )
    selected, _ = LastMinuteSelection().pack(ctx)
    assert selected == [3]


def test_eoq_last_minute_uses_per_bin_harris_trigger():
    ctx = _ctx(
        holding_cost_per_kg_day=0.02,
        ordering_cost_per_visit=50.0,
        bin_volume=2.5,
        bin_density=400.0,
        accumulation_rates=np.array([10.0, 10.0, 10.0]),
        threshold=90.0,
    )
    selected, _ = LastMinuteEOQSelection().pack(ctx)
    # Q* / mass_cap ≈ 0.707 → trigger ≈ 70.7%, so 80 and 95 are collected.
    assert selected == [2, 3]


def test_spec_rules_match_documented_predicates():
    fills = np.array([50.0, 70.0, 95.0])
    assert last_minute(fills, 70.0).tolist() == [False, True, True]
    assert regular(fills, current_day=4, frequency=3).tolist() == [True, True, True]
    assert regular(fills, current_day=5, frequency=3).tolist() == [False, False, False]
    sl = service_level(
        np.array([95.0, 50.0, 80.0]),
        np.array([5.0, 5.0, 30.0]),
        np.array([1.0, 1.0, 1.0]),
        z=0.84,
        horizon_days=1,
    )
    assert sl.tolist() == [True, False, True]
    la = lookahead(np.array([80.0, 40.0, 95.0]), np.array([25.0, 20.0, 10.0]), current_collection_day=0)
    assert la.tolist() == [True, True, True]


def test_released_packing_pack_does_not_import_torch():
    script = dedent(
        """
        import sys
        from logic.src.policies.packing.base.factory import PackingFactory
        from logic.src.interfaces.context.packing_context import PackingContext
        import numpy as np

        fills = np.array([50.0, 80.0, 95.0])
        ctx = PackingContext(
            bin_ids=np.arange(3),
            current_fill=fills,
            accumulation_rates=np.array([5.0, 5.0, 30.0]),
            std_deviations=np.array([1.0, 1.0, 1.0]),
            threshold=70.0,
            current_day=4,
            horizon_days=1,
        )
        for name in ("last_minute", "regular", "service_level", "lookahead"):
            PackingFactory.create_strategy(name).pack(ctx)
        banned = [
            name
            for name in sys.modules
            if name == "torch"
            or name.startswith("torch.")
            or name in ("torchrl", "tensordict", "torch_geometric")
            or name.startswith(("torchrl.", "tensordict.", "torch_geometric."))
        ]
        assert not banned, banned
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_released_vectorized_path_does_not_import_hrl_manager():
    """P-05: unreleased HRL stays out of the released packing import graph."""
    script = dedent(
        """
        import ast
        import sys
        from pathlib import Path

        from logic.src.policies.vectorized import packing as vec_pkg
        from logic.src.policies.vectorized.packing.factory import (
            create_selector_from_config,
            get_vectorized_selector,
        )
        from logic.src.policies.vectorized.packing.last_minute import LastMinuteSelector
        from logic.src.policies.vectorized.packing.lookahead import LookaheadSelector
        from logic.src.policies.vectorized.packing.regular import RegularSelector
        from logic.src.policies.vectorized.packing.service_level import ServiceLevelSelector
        import logic.src.policies.vectorized.packing.manager as manager_mod

        for name in ("last_minute", "regular", "service_level", "lookahead"):
            get_vectorized_selector(name)
        create_selector_from_config({"strategy": "last_minute", "threshold": 70.0})
        assert LastMinuteSelector and RegularSelector and LookaheadSelector and ServiceLevelSelector
        leaked = [name for name in sys.modules if "hrl_manager" in name]
        assert not leaked, leaked
        assert "ManagerSelector" not in vec_pkg.__all__
        src = Path(manager_mod.__file__).read_text()
        tree = ast.parse(src)
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and "hrl_manager" in node.module
            for alias in node.names
        ]
        assert not imported, imported
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
