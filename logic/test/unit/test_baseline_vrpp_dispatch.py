"""Baseline adapter amendment: runtime seed must not hide optional routing."""

from unittest.mock import patch

import pytest
from logic.src.pipeline.simulations.actions.routing import RoutingAction
from omegaconf import OmegaConf


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("vrpp", [False, True])
def test_runtime_seed_preserves_optional_day_dispatch(structured, vrpp):
    config = {"alns": {"vrpp": vrpp, "max_iterations": 17}}
    context = {
        "config": OmegaConf.create(config) if structured else config,
        "full_policy": "alns",
        "policy_seed": 123,
        "mandatory": [],
    }
    with (
        patch("logic.src.pipeline.simulations.actions.routing.RoutingAlgorithmFactory.ensure_registered"),
        patch(
            "logic.src.pipeline.simulations.actions.routing.RoutingAlgorithmRegistry.list_routing_algorithms",
            return_value=["alns"],
        ),
        patch(
            "logic.src.pipeline.simulations.actions.routing.RoutingAlgorithmFactory.get_adapter",
            side_effect=RuntimeError("dispatch reached"),
        ) as build,
    ):
        for day in range(2):
            context["day"] = day
            if vrpp:
                with pytest.raises(RuntimeError, match="dispatch reached"):
                    RoutingAction().execute(context)
                assert build.call_args.kwargs["config"]["seed"] == 123
            else:
                RoutingAction().execute(context)
                assert context["solver_status"] == "skipped"
                build.assert_not_called()
            assert "seed" not in context["config"]
