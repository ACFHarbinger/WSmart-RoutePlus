import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.utils.model.loader import load_model

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestLoadModel:
    """Class for load_model tests."""

    @patch("logic.src.utils.model.problem_factory.load_problem")
    @patch("logic.src.utils.model.loader.load_args")
    @patch("torch.load")
    def test_load_model_basic(self, mock_torch_load, mock_load_args, mock_load_problem, tmp_path):
        """Test load_model by mocking internal dependencies and file structure."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        hparams = {
            "problem": "ptp",
            "model": "am",
            "encoder": "gat",
            "embed_dim": 128,
            "hidden_dim": 512,
            "n_encode_layers": 3,
            "n_encode_sublayers": 1,
            "n_decode_layers": 1,
            "n_heads": 8,
            "normalization": "batch",
            "tanh_clipping": 10.0,
            "learn_affine": True,
            "track_stats": True,
            "epsilon_alpha": 1e-5,
            "momentum_beta": 0.1,
            "lrnorm_k": 1,
            "gnorm_groups": 1,
            "activation": "relu",
            "af_param": 0.0,
            "af_threshold": 0.0,
            "af_replacement": 0.0,
            "af_nparams": 1,
            "af_urange": 0.0,
            "dropout": 0.0,
            "aggregation": "mean",
            "aggregation_graph": "mean",
        }
        mock_load_args.return_value = hparams

        # Create dummy epoch file and args.json with content
        (model_dir / "epoch-10.pt").touch()

        with open(model_dir / "args.json", "w") as f:
            json.dump(hparams, f)

        mock_problem = MagicMock()
        mock_load_problem.return_value = mock_problem
        mock_torch_load.return_value = {"model": {"encoder.dummy": torch.zeros(1)}}
        match = SimpleNamespace(missing_keys=[], unexpected_keys=[])

        with patch("os.listdir", return_value=["epoch-10.pt"]), patch(
            "logic.src.models.AttentionModel.load_state_dict", return_value=match
        ):
            model, args = load_model(str(model_dir))

        assert model is not None
        assert args == mock_load_args.return_value
        assert mock_torch_load.called

        # An empty checkpoint used to load a randomly initialised model silently.
        mock_torch_load.return_value = {"model": {}}
        with patch("os.listdir", return_value=["epoch-10.pt"]), pytest.raises(ValueError, match="no model parameters"):
            load_model(str(model_dir))
