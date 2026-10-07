"""
Fixtures providing base command-line arguments (sys.argv mocks) for various CLI commands.
"""

import pytest


@pytest.fixture
def base_train_args():
    """Fixture providing base training arguments"""
    return [
        "script.py",
        "train",
        "--problem",
        "ptp",
        "--graph_size",
        "20",
        "--batch_size",
        "256",
        "--epoch_size",
        "128000",
        "--train.env.graph.n_days",
        "25",
    ]


@pytest.fixture
def base_mrl_args():
    """Fixture providing base MRL training arguments"""
    return [
        "script.py",
        "mrl_train",
        "--problem",
        "ptp",
        "--batch_size",
        "256",
        "--epoch_size",
        "128000",
        "--mrl_method",
        "cb",
    ]


@pytest.fixture
def base_hp_optim_args():
    """Fixture providing base hyperparameter optimization arguments"""
    return [
        "script.py",
        "hp_optim",
        "--batch_size",
        "256",
        "--epoch_size",
        "128000",
        "--hpo_method",
        "bo",
    ]


@pytest.fixture
def base_gen_data_args():
    """Fixture providing base data generation arguments"""
    return [
        "script.py",
        "gen_data",
        "--problem",
        "ptp",
        "--dataset_size",
        "10000",
        "--graph_sizes",
        "20",
    ]


@pytest.fixture
def base_eval_args():
    """Fixture providing base evaluation arguments"""
    return ["script.py", "eval", "--datasets", "dataset1.pkl", "--model", "model.pt"]


@pytest.fixture
def base_test_args():
    """Fixture providing base simulator test arguments"""
    return [
        "script.py",
        "test_sim",
        "--policies",
        "policy1",
        "--days",
        "31",
        "--size",
        "50",
    ]


@pytest.fixture
def base_file_system_update_args():
    """Fixture providing base file system update arguments"""
    return ["script.py", "file_system", "update", "--target_entry", "path/to/file.pkl"]


@pytest.fixture
def base_file_system_delete_args():
    """Fixture providing base file system delete arguments"""
    return ["script.py", "file_system", "delete"]


@pytest.fixture
def base_file_system_crypto_args():
    """Fixture providing base file system cryptography arguments"""
    return ["script.py", "file_system", "cryptography"]
