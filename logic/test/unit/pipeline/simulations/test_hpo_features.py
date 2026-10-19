"""Unit tests for HPO base classes and search spaces."""

import dataclasses
import json
from typing import Any, List, Union
from unittest.mock import MagicMock, patch

import optuna
import pytest
from logic.src.configs import Config
from logic.src.pipeline.simulations.hpo.base import (
    PolicyHPOBase,
    _build_grid_from_search_space,
)
from logic.src.pipeline.simulations.hpo.search_spaces import (
    _extract_params_from_config,
    compose_search_space,
    generate_acceptance_criteria_templates,
    generate_packing_templates,
    generate_route_improvement_templates,
    generate_routing_templates,
    get_component_search_space,
    get_search_space,
    load_all_search_spaces,
    validate_search_space,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



@dataclasses.dataclass
class DummySubConfig:
    param_int: int = 10
    param_float: float = 0.5
    param_bool: bool = True
    param_str: str = "default"


@dataclasses.dataclass
class DummyConfig:
    sub: DummySubConfig = dataclasses.field(default_factory=DummySubConfig)
    some_list: List[Any] = dataclasses.field(default_factory=lambda: [1, 2, 3])


class DummyHPO(PolicyHPOBase):
    def run(self, cfg: Config) -> Union[float, List[float]]:
        return 0.5


class TestPolicyHPOBase:
    @pytest.mark.unit
    def test_apply_params(self) -> None:
        cfg = DummyConfig()
        hpo = DummyHPO(cfg=MagicMock())

        # Test attribute setting
        hpo._apply_params(cfg, {"sub.param_int": 42, "sub.param_float": 1.5, "sub.param_str": "new"})
        assert cfg.sub.param_int == 42
        assert cfg.sub.param_float == 1.5
        assert cfg.sub.param_str == "new"

        # Test setting list index
        hpo._apply_params(cfg, {"some_list.1": 99})
        assert cfg.some_list[1] == 99

        # Test invalid path segment raises AttributeError
        with pytest.raises(AttributeError):
            hpo._apply_params(cfg, {"sub.non_existent": 10})

    @pytest.mark.unit
    def test_validate_search_space(self) -> None:
        # Valid spaces
        hpo_class = DummyHPO
        hpo_class.validate_search_space({"p_float": {"type": "float", "low": 0.0, "high": 1.0}}, "dummy")
        hpo_class.validate_search_space({"p_int": {"type": "int", "low": 1, "high": 5}}, "dummy")
        hpo_class.validate_search_space({"p_cat": {"type": "categorical", "choices": [1, 2, 3]}}, "dummy")

        # Unknown type
        with pytest.raises(ValueError, match="unknown type"):
            hpo_class.validate_search_space({"p": {"type": "unknown"}}, "dummy")

        # Missing low/high
        with pytest.raises(ValueError, match="missing required key"):
            hpo_class.validate_search_space({"p": {"type": "float", "low": 0.0}}, "dummy")

        # low >= high
        with pytest.raises(ValueError, match="must be strictly less than"):
            hpo_class.validate_search_space({"p": {"type": "int", "low": 5, "high": 2}}, "dummy")

        # categorical empty choices
        with pytest.raises(ValueError, match="empty or missing 'choices'"):
            hpo_class.validate_search_space({"p": {"type": "categorical", "choices": []}}, "dummy")

    @pytest.mark.unit
    def test_suggest_param(self) -> None:
        trial = MagicMock()
        hpo_class = DummyHPO

        # Float
        trial.suggest_float.return_value = 0.75
        val = hpo_class.suggest_param(trial, "p_float", {"type": "float", "low": 0.0, "high": 1.0, "log": True})
        assert val == 0.75
        trial.suggest_float.assert_called_with("p_float", 0.0, 1.0, step=None, log=True)

        # Int
        trial.suggest_int.return_value = 3
        val = hpo_class.suggest_param(trial, "p_int", {"type": "int", "low": 1, "high": 5, "step": 2})
        assert val == 3
        trial.suggest_int.assert_called_with("p_int", 1, 5, step=2, log=False)

        # Categorical lists conversion
        trial.suggest_categorical.return_value = (1, 2)
        val = hpo_class.suggest_param(trial, "p_cat", {"type": "categorical", "choices": [[1, 2], [3, 4]]})
        assert val == [1, 2]
        trial.suggest_categorical.assert_called_with("p_cat", [(1, 2), (3, 4)])

        # Unknown type
        with pytest.raises(ValueError, match="Unknown parameter type"):
            hpo_class.suggest_param(trial, "p_bad", {"type": "unsupported"})

    @pytest.mark.unit
    def test_suggest_params(self) -> None:
        trial = MagicMock()
        search_space = {
            "p_float": {"type": "float", "low": 0.0, "high": 1.0},
            "p_cat": {"type": "categorical", "choices": ["a", "b"]},
        }
        hpo = DummyHPO(cfg=MagicMock(), search_space=search_space)
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = "b"

        params = hpo.suggest_params(trial)
        assert params == {"p_float": 0.5, "p_cat": "b"}

    @pytest.mark.unit
    def test_build_sampler(self) -> None:
        hpo_class = DummyHPO
        assert isinstance(hpo_class.build_sampler("tpe", 42), optuna.samplers.TPESampler)
        assert isinstance(hpo_class.build_sampler("random", 42), optuna.samplers.RandomSampler)
        assert isinstance(hpo_class.build_sampler("cmaes", 42), optuna.samplers.CmaEsSampler)
        assert isinstance(hpo_class.build_sampler("nsgaii", 42), optuna.samplers.NSGAIISampler)

        # Grid sampler
        space = {"p": {"type": "categorical", "choices": [1, 2]}}
        assert isinstance(hpo_class.build_sampler("grid", 42, space), optuna.samplers.GridSampler)

        with pytest.raises(ValueError, match="requires the search_space"):
            hpo_class.build_sampler("grid", 42)

        # Unknown method
        with pytest.raises(ValueError, match="Unknown sampler method"):
            hpo_class.build_sampler("unknown", 42)

    @pytest.mark.unit
    def test_build_grid_from_search_space(self) -> None:
        # Valid grid spaces
        space = {
            "p_cat": {"type": "categorical", "choices": ["a", "b"]},
            "p_int_choices": {"type": "int", "choices": [1, 3, 5]},
            "p_int_step": {"type": "int", "low": 1, "high": 5, "step": 2},
        }
        grid = _build_grid_from_search_space(space)
        assert grid == {
            "p_cat": ["a", "b"],
            "p_int_choices": [1, 3, 5],
            "p_int_step": [1, 3, 5],
        }

        # Invalid grid spaces
        with pytest.raises(ValueError, match="Grid search cannot enumerate parameter"):
            _build_grid_from_search_space({"p_float": {"type": "float", "low": 0.0, "high": 1.0}})


class TestSearchSpaces:
    @pytest.mark.unit
    def test_validate_search_space_func(self) -> None:
        errors = validate_search_space(
            {"_comment": "metadata", "valid_p": {"type": "float", "low": 0.0, "high": 1.0}}, "dummy"
        )
        assert len(errors) == 0

        errors = validate_search_space({"bad_spec": "not_a_dict"}, "dummy")
        assert len(errors) == 1
        assert "spec must be a dictionary" in errors[0]

        errors = validate_search_space({"bad_type": {"type": "unsupported"}}, "dummy")
        assert len(errors) == 1
        assert "unknown type" in errors[0]

        errors = validate_search_space({"bad_range": {"type": "int", "low": 10, "high": 5}}, "dummy")
        assert len(errors) == 1
        assert "must be strictly less than" in errors[0]

    @pytest.mark.unit
    def test_load_all_search_spaces(self, tmp_path: Any) -> None:
        # Directory does not exist
        spaces = load_all_search_spaces(str(tmp_path / "non_existent"))
        assert spaces == {}

        # Valid JSON file
        json_dir = tmp_path / "spaces"
        json_dir.mkdir()
        with open(json_dir / "alns.json", "w") as fh:
            json.dump({"param_a": {"type": "float", "low": 0.0, "high": 1.0}}, fh)

        # Invalid JSON file
        with open(json_dir / "bad.json", "w") as fh:
            fh.write("invalid json")

        # Invalid spec inside JSON file (soft-validated)
        with open(json_dir / "invalid_spec.json", "w") as fh:
            json.dump({"param_b": {"type": "categorical", "choices": []}}, fh)

        spaces = load_all_search_spaces(str(json_dir))
        assert "alns" in spaces
        assert "param_a" in spaces["alns"]
        assert "bad" not in spaces
        assert "invalid_spec" in spaces  # soft-validated and loaded

    @pytest.mark.unit
    def test_get_component_search_space(self) -> None:
        # Unknown component type
        with pytest.raises(ValueError, match="Unknown component type"):
            get_component_search_space("invalid_type", "name")

        # Component not found
        assert get_component_search_space("packing", "non_existent") == {}

    @pytest.mark.unit
    @patch(
        "logic.src.pipeline.simulations.hpo.search_spaces.PACKING_SPACES",
        {"test_filter": {"p": {"type": "float", "low": 0.0, "high": 1.0}}},
    )
    @patch(
        "logic.src.pipeline.simulations.hpo.search_spaces.ROUTE_IMPROVEMENT_SPACES",
        {"test_interceptor": {"p": {"type": "int", "low": 1, "high": 10}}},
    )
    @patch(
        "logic.src.pipeline.simulations.hpo.search_spaces.ROUTING_SPACES",
        {"test_job": {"p": {"type": "categorical", "choices": ["a"]}}},
    )
    @patch(
        "logic.src.pipeline.simulations.hpo.search_spaces.ACCEPTANCE_CRITERIA_SPACES",
        {"test_rule": {"p": {"type": "float", "low": 0.0, "high": 1.0}}},
    )
    def test_compose_search_space(self) -> None:
        composed = compose_search_space(
            routing="test_job",
            packing="test_filter",
            route_improvement=["test_interceptor"],
            acceptance_criteria="test_rule",
        )
        assert "p" in composed
        assert "packing.0.p" in composed
        assert "route_improvement.0.p" in composed
        assert "acceptance_criterion.params.p" in composed

        # Keywords check
        composed_kw = compose_search_space(
            routing="test_job",
            routing_keywords="non_existent",
        )
        assert composed_kw == {}

    @pytest.mark.unit
    @patch(
        "logic.src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES",
        {"test_policy": {"p": {"type": "float", "low": 0.0, "high": 1.0}}},
    )
    def test_get_search_space(self) -> None:
        space = get_search_space("test_policy")
        assert "p" in space

        # Mock invalid spec
        with (
            patch(
                "logic.src.pipeline.simulations.hpo.search_spaces.POLICY_SEARCH_SPACES",
                {"test_policy": {"p": {"type": "float", "low": 5.0, "high": 1.0}}},
            ),
            pytest.raises(ValueError, match="has 1 error"),
        ):
            get_search_space("test_policy")

    @pytest.mark.unit
    def test_extract_params_from_config(self) -> None:
        params = _extract_params_from_config(DummySubConfig)
        assert params["param_int"] == {"type": "int", "_NEEDS_BOUNDS": True, "default": 10}
        assert params["param_float"] == {"type": "float", "_NEEDS_BOUNDS": True, "default": 0.5}
        assert params["param_bool"] == {"type": "categorical", "choices": [True, False], "default": True}
        assert params["param_str"] == {"type": "categorical", "choices": ["default"], "default": "default"}

    @pytest.mark.unit
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("builtins.open")
    def test_scaffold_generators(self, mock_open: Any, mock_makedirs: Any, mock_exists: Any) -> None:
        # We test that the generator functions complete without errors when paths do not exist.
        # Since we patched os.path.exists to return False, they should try to write files.
        with patch("logic.src.configs.policies", MagicMock(__path__=["mock_path"])):
            generate_routing_templates()
        generate_route_improvement_templates()
        generate_packing_templates()
        generate_acceptance_criteria_templates()
        assert mock_makedirs.called
