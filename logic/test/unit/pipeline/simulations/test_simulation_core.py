

import pytest
import torch  # noqa: F401
from logic.src.pipeline.simulations.checkpoints import CheckpointError
from logic.src.pipeline.simulations.simulator import sequential_simulations, single_simulation

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestSimulation:
    """Integration tests for high-level simulation orchestration."""

    @pytest.mark.integration
    def test_single_simulation_happy_path(self, wsr_opts, mock_torch_device, mock_lock_counter, mocker):
        """Test a single simulation run successfully."""
        lock, counter = mock_lock_counter
        # Mock globals for workers manually
        import logic.src.pipeline.simulations.simulator as sim
        sim._lock = lock
        sim._counter = counter

        mocker.patch("logic.src.pipeline.simulations.states.base.context.SimulationContext.run",
                     return_value={"test_policy_gamma1": [1.0], "success": True})

        indices = [None]
        res = single_simulation(wsr_opts, mock_torch_device, indices, 0, 0, "weights", 1)
        assert res.get("success") is True

    @pytest.mark.integration
    def test_single_simulation_checkpoint_error(self, wsr_opts, mock_torch_device, mock_lock_counter, mocker):
        """Test error handling for broken checkpoints."""
        lock, counter = mock_lock_counter
        import logic.src.pipeline.simulations.simulator as sim


        sim._lock = lock
        sim._counter = counter

        mocker.patch("logic.src.pipeline.simulations.states.base.context.SimulationContext.run",
                     side_effect=CheckpointError({"error": "broken"}))

        res = single_simulation(wsr_opts, mock_torch_device, [None], 0, 0, "weights", 1)
        assert res.get("success") is False
        assert "broken" in res.get("error", "")

    @pytest.mark.integration
    def test_sequential_simulations_multi_sample(self, wsr_opts, mock_torch_device, mock_lock_counter, mocker, tmp_path):
        """Test running multiple samples sequentially."""
        lock, _ = mock_lock_counter
        cfg = wsr_opts
        cfg.sim.graph.n_samples = 2
        cfg.sim.full_policies = ["pol1"]
        cfg.sim.policies = ["pol1"]
        mocker.patch("logic.src.pipeline.simulations.simulator.ROOT_DIR", str(tmp_path))
        mocker.patch("logic.src.tracking.logging.log_utils.log_to_json")

        mocker.patch("logic.src.pipeline.simulations.states.base.context.SimulationContext.run",
                     return_value={"none_pol1_none": [1.0, 2.0], "success": True})

        # Create output dir
        results_dir = tmp_path / cfg.sim.output_dir / f"{cfg.sim.graph.n_days}days" / f"{cfg.sim.graph.area}{cfg.sim.graph.num_loc}_{cfg.sim.graph.waste_type}" / cfg.sim.data_distribution
        results_dir.mkdir(parents=True, exist_ok=True)

        log, log_std, failed = sequential_simulations(
            cfg, mock_torch_device, [None, None], [[0, 1]], "weights", lock
        )

        assert "none_pol1_none" in log
        assert len(log["none_pol1_none"]) == 2
        assert log_std is not None
        assert "none_pol1_none" in log_std
