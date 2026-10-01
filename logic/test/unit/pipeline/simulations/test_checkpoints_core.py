import os

import pytest
from logic.src.pipeline.simulations.checkpoints.manager import CheckpointError, checkpoint_manager
from logic.src.pipeline.simulations.checkpoints.persistence import SimulationCheckpoint

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestCheckpoints:
    """Class for SimulationCheckpoint tests."""

    @pytest.mark.unit
    def test_checkpoint_io(self, basic_checkpoint):
        """Test saving and loading a checkpoint."""
        state = {"data": "test_state"}
        day = 5
        basic_checkpoint.save_state(state, day)

        # Verify file exists with real pattern
        # checkpoint_{policy}_{sample_id}_day{day}.pkl
        pattern = f"checkpoint_{basic_checkpoint.policy}_{basic_checkpoint.sample_id}_day{day}.pkl"
        expected_file = os.path.join(basic_checkpoint.checkpoint_dir, pattern)
        assert os.path.exists(expected_file)

        # Load state
        loaded_state, loaded_day = basic_checkpoint.load_state(day=day)
        assert loaded_day == day
        assert loaded_state == state

    @pytest.mark.unit
    def test_checkpoint_resume_logic(self, basic_checkpoint):
        """Test finding the last checkpoint day."""
        # Create multiple checkpoints
        basic_checkpoint.save_state({"d": 1}, 10)
        basic_checkpoint.save_state({"d": 2}, 20)

        # find_last_checkpoint_day should return 20
        assert basic_checkpoint.find_last_checkpoint_day() == 20

        # Load without specifying day should get the latest (20)
        state, day = basic_checkpoint.load_state()
        assert day == 20
        assert state == {"d": 2}

    @pytest.mark.unit
    def test_checkpoint_error_handling(self, basic_checkpoint, mocker):
        """Test error handling during checkpoint operations using manager."""

        def failing_task():
            raise ValueError("Test error")

        with pytest.raises(CheckpointError) as excinfo, checkpoint_manager(basic_checkpoint, 1, lambda: {"state": 1}):
            failing_task()

        assert "Test error" in str(excinfo.value)
        # CheckpointError stores the error_result dict
        assert excinfo.value.error_result["error"] == "Test error"

    @pytest.mark.unit
    def test_get_checkpoint_file_explicit(self, basic_checkpoint):
        """Test get_checkpoint_file with explicit day."""
        path = basic_checkpoint.get_checkpoint_file(day=10)
        pattern = f"checkpoint_{basic_checkpoint.policy}_{basic_checkpoint.sample_id}_day10.pkl"
        assert pattern in path

    @pytest.mark.unit
    def test_load_state_file_not_found(self, basic_checkpoint):
        """Test load_state when file is missing."""
        # Ensure dir is empty for this test
        basic_checkpoint.clear()
        state, day = basic_checkpoint.load_state(day=99)
        assert state is None
        assert day == 0

    @pytest.mark.unit
    def test_distinct_run_directories_do_not_share_checkpoints(self, tmp_path):
        """Two results directories with the same policy and sample stay isolated.

        The results directory already carries ``sim.run_name``. A second run must
        not load, list, or clear the first run's live checkpoint. The same
        directory still resumes at day N, which is the state after day N.
        Completing a run clears only the live files; the end-of-run copy remains,
        and resume does not treat that copy as a live checkpoint.
        """
        policy = "swc_tcf_gurobi_cls_gamma3"
        sample_id = 0
        lookahead = SimulationCheckpoint(
            output_dir=str(tmp_path / "results" / "la"),
            checkpoint_dir="checkpoints",
            policy=policy,
            sample_id=sample_id,
        )
        service_level = SimulationCheckpoint(
            output_dir=str(tmp_path / "results" / "sl2"),
            checkpoint_dir="checkpoints",
            policy=policy,
            sample_id=sample_id,
        )

        lookahead.save_state({"run": "la", "bins": [1.0]}, day=3)

        live_file = lookahead.get_checkpoint_file(day=3)
        assert os.path.isfile(live_file)
        assert str(tmp_path / "results" / "la") in live_file
        assert os.path.basename(os.path.dirname(live_file)) == "checkpoints_live"

        assert service_level.find_last_checkpoint_day() == 0
        other_state, other_day = service_level.load_state()
        assert other_state is None
        assert other_day == 0

        resumed = SimulationCheckpoint(
            output_dir=str(tmp_path / "results" / "la"),
            checkpoint_dir="checkpoints",
            policy=policy,
            sample_id=sample_id,
        )
        state, day = resumed.load_state()
        assert day == 3
        assert state == {"run": "la", "bins": [1.0]}

        lookahead.save_state({"run": "la", "bins": [2.0]}, day=4)
        state, day = resumed.load_state()
        assert day == 4
        assert state["bins"] == [2.0]

        service_level.clear()
        state, day = resumed.load_state()
        assert day == 4

        final_file = lookahead.get_checkpoint_file(day=4, end_simulation=True)
        lookahead.save_state({"run": "la", "done": True}, day=4, end_simulation=True)
        assert lookahead.clear() >= 1
        assert os.path.isfile(final_file)
        assert final_file != live_file
        state, day = resumed.load_state()
        assert state is None
        assert day == 0
