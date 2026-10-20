"""Tests for project definitions and constants."""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]




# Test module mapping
TEST_MODULES = {
    "parser": "unit/cli/test_arg_parser.py",
    "train": "unit/pipeline/learning/test_train.py",
    "mrl": "unit/pipeline/learning/meta/test_mrl_train.py",
    "hp_optim": "unit/pipeline/learning/test_hp_optim.py",
    "gen_data": "unit/pipeline/learning/envs/test_generators.py",
    "problems": "unit/pipeline/learning/envs/test_problems.py",
    "eval": "unit/pipeline/features/test_eval.py",
    "models": "unit/models/test_models.py",
    "subnets": "unit/models/subnets/test_subnets.py",
    "modules": "unit/models/modules/test_modules.py",
    "simulator": "unit/pipeline/simulations/test_simulator_integration.py",
    "integration": "integration/test_integration.py",
}
