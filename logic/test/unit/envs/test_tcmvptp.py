"""Regression tests for Time-Constrained Multi-Vehicle Profitable Tour Problem feasibility."""

import pytest
import torch
from logic.src.envs.generators.tcmvptp import TCMVPTPGenerator
from logic.src.envs.routing.tcmvptp import TCMVPTPEnv
from logic.src.envs.tasks.tcmvptp import TCMVPTP
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _dataset(shift_hours: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "depot": torch.tensor([[0.0, 0.0]]),
        "locs": torch.tensor([[[0.5, 0.0]]]),
        "waste": torch.tensor([[1.0]]),
        "capacity": torch.tensor([10.0]),
        "shift_hours": torch.tensor([shift_hours]),
        "avg_speed_kmh": torch.tensor([1.0]),
        "service_time_h": torch.tensor([0.0]),
    }


class TestCTOPCosts:
    """The static evaluator must enforce every depot-to-depot trip."""

    @pytest.mark.parametrize("tour", [[[1, 0]], [[0, 1]], [[1]]])
    def test_time_matrix_overrides_distance_and_includes_return(self, tour):
        data = _dataset(shift_hours=1.0)
        data["time_matrix"] = torch.tensor([[[0.0, 0.2], [0.7, 0.0]]])
        _, costs, _ = TCMVPTP.get_costs(data, torch.tensor(tour), None)
        assert costs["time"].item() == pytest.approx(0.9)
        data["time_matrix"][0, 1, 0] = 0.9
        with pytest.raises(AssertionError, match="trip time"):
            TCMVPTP.get_costs(data, torch.tensor(tour), None)

    def test_final_depot_return_is_time_feasible(self):
        """A route that cannot return before shift end is rejected."""
        data = _dataset()
        data["locs"] = torch.tensor([[[0.6, 0.0]]])
        with pytest.raises(AssertionError, match="trip time"):
            TCMVPTP.get_costs(data, torch.tensor([[1, 0]]), None)

    def test_road_distance_matrix_governs_time_feasibility(self):
        """Time validation follows supplied road legs rather than coordinates."""
        dist_matrix = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
        with pytest.raises(AssertionError, match="trip time"):
            TCMVPTP.get_costs(_dataset(shift_hours=3.0), torch.tensor([[1, 0]]), None, dist_matrix)

    def test_one_microhour_tolerance_is_accepted(self):
        """The documented 1e-6 feasibility tolerance remains inclusive."""
        data = _dataset()
        data["locs"] = torch.tensor([[[0.5000005, 0.0]]])
        _, costs, _ = TCMVPTP.get_costs(data, torch.tensor([[1, 0]]), None)
        assert costs["time"].item() == pytest.approx(1.000001, abs=1e-6)


class TestCTOPEnvironment:
    """Live TCMVPTP state uses the same physical constraints as its evaluator."""

    def test_time_matrix_controls_mask_and_resource_consumption(self):
        data = _dataset(shift_hours=1.0)
        data["time_matrix"] = torch.tensor([[[0.0, 0.2], [0.7, 0.0]]])
        env = self._env()
        state = env.reset(TensorDict(data, batch_size=[1]))
        assert state["action_mask"][0, 1]
        state["action"] = torch.tensor([1])
        state = env.step(state)["next"]
        assert state["remaining_time"].item() == pytest.approx(0.8)
        state["action"] = torch.tensor([0])
        state = env.step(state)["next"]
        assert state["time_spent"].item() == pytest.approx(0.9)
        data["time_matrix"][0, 1, 0] = 0.9
        state = env.reset(TensorDict(data, batch_size=[1]))
        assert not state["action_mask"][0, 1]

    @staticmethod
    def _env() -> TCMVPTPEnv:
        return TCMVPTPEnv(
            generator=TCMVPTPGenerator(
                num_loc=1,
                capacity=10.0,
                shift_hours=2.5,
                avg_speed_kmh=20.0,
                service_time_h=0.1,
            )
        )

    def test_generator_and_env_import_without_simulation_repository_cycle(self):
        """TCMVPTP remains constructible without initializing simulation repositories."""
        td = self._env().reset(batch_size=[1])
        assert {"shift_hours", "avg_speed_kmh", "service_time_h"} <= set(td.keys())
        assert td["shift_hours"].item() == 2.5
        assert td["avg_speed_kmh"].item() == 20.0
        assert td["service_time_h"].item() == pytest.approx(0.1)

    def test_get_env_factory_builds_a_tcmvptp_generator_not_a_plain_ptp_one(self):
        """get_env("tcmvptp", **kwargs) must route through TCMVPTPGenerator.

        Regression test: TCMVPTPEnv previously had no __init__ override, so it
        inherited PTPEnv's, which always built a plain PTPGenerator. Any
        shift_hours/avg_speed_kmh/service_time_h kwarg passed through
        get_env (including a Hydra-composed override) was silently
        swallowed by PTPGenerator's **kwargs and never reached the
        environment -- _reset_instance fell back to
        get_default_temporal_params() regardless of what was requested.
        """
        from logic.src.envs.routing import get_env

        env = get_env("tcmvptp", num_loc=5, shift_hours=3.5, avg_speed_kmh=40.0)
        assert isinstance(env.generator, TCMVPTPGenerator)
        assert env.generator.shift_hours == 3.5
        assert env.generator.avg_speed_kmh == 40.0

        td = env.reset(batch_size=[1])
        assert td["shift_hours"].item() == 3.5
        assert td["avg_speed_kmh"].item() == 40.0

    def test_action_mask_uses_road_distance_and_allows_tolerance_boundary(self):
        """A customer needs enough road time for both the visit and return."""
        data = _dataset(shift_hours=1.0)
        data["dm"] = torch.tensor([[[0.0, 0.5000005], [0.5000005, 0.0]]])
        td = TensorDict(data, batch_size=[1])
        state = self._env().reset(td)
        assert state["action_mask"].tolist() == [[True, True]]

        data["dm"] = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])
        state = self._env().reset(TensorDict(data, batch_size=[1]))
        assert state["action_mask"].tolist() == [[True, False]]

    def test_resume_preserves_resources_and_depot_coordinate(self):
        """Resetting an initialized state does not erase its trip resources."""
        state = self._env().reset(TensorDict(_dataset(shift_hours=3.0), batch_size=[1]))
        state["current_node"] = torch.tensor([[1]])
        state["visited"] = torch.tensor([[True, True]])
        state["remaining_capacity"] = torch.tensor([4.0])
        state["remaining_time"] = torch.tensor([1.5])
        state["time_spent"] = torch.tensor([1.5])

        resumed = self._env().reset(state)
        assert torch.equal(resumed["locs"][:, 0], resumed["depot"])
        assert resumed["remaining_capacity"].item() == 4.0
        assert resumed["remaining_time"].item() == 1.5
        assert resumed["time_spent"].item() == 1.5
