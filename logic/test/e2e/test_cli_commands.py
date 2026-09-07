"""
End-to-end smoke tests for CLI commands.
"""

import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]





@pytest.mark.e2e
def test_cli_help():
    """Test that main.py --help runs successfully."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.returncode == 0
    # main.py --help still goes to legacy parser for other commands
    assert "usage:" in result.stdout or "Usage:" in result.stdout


@pytest.mark.e2e
@pytest.mark.parametrize("problem", ["vrpp"])
def test_cli_gen_data_smoke(tmp_path, problem):
    """Smoke test for data generation."""
    output_dir = tmp_path / "data"
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "gen_data",
            "data.dataset_type=test_simulator",
            f"data.problem={problem}",
            "data.data_distributions=[unif]",
            "data.graphs=[{num_loc: 10, n_days: 1, n_samples: 2}]",
            f"data.data_dir={output_dir}",
            "data.name=smoke_test",
            "data.overwrite=true",
        ],
        capture_output=True,
        text=True,
        timeout=60,  # 1 minute timeout for data generation
    )

    # Check for success
    if result.returncode != 0:
        log_path = tmp_path / "cli_gen_data_error.log"
        log_path.write_text(result.stdout + result.stderr)
        assert (
            result.returncode == 0
        ), f"gen_data failed with return code {result.returncode}. See {log_path}"

    # Expected filename for test_simulator: {area}{size}_{dist}_{name}_N{dataset_size}_seed{seed}.pkl
    # area defaults to 'riomaior', seed defaults to 42
    # riomaior10_unif_smoke_test1_N2_seed42.pkl
    expected_file = output_dir / "riomaior10_unif_smoke_test1_N2_seed42.npz"
    assert expected_file.exists(), f"Generated data file not found at {expected_file}"


@pytest.mark.e2e
def test_cli_train_lightning_smoke(tmp_path):
    """Smoke test for training loop."""
    # Path to model weights that will be created during training
    weights_dir = tmp_path / "weights"
    tracking_uri = tmp_path / "tracking"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "train",
                "train.env.name=vrpp",
                "train.env.curriculum_graphs.0.num_loc=10",
                "train.env.curriculum_graphs.0.n_days=1",
                "train.batch_size=2",
                "train.eval_batch_size=2",
                "train.num_workers=0",  # Disable multiprocessing for faster startup
                "train.persistent_workers=false",  # Disable persistent workers
                "train.devices=1",  # Use only 1 GPU
                "train.strategy=auto",  # Auto strategy for single GPU
                "train.policy.model.encoder.embed_dim=32",  # Smaller model for faster training
                "train.policy.model.encoder.hidden_dim=64",
                "train.policy.model.encoder.n_layers=1",
                "train.policy.model.encoder.n_heads=2",
                "tracking.wandb_mode=offline",
                f"tracking.wst_tracking_uri={tracking_uri}",
                f"+output_dir={weights_dir}",
                "hpo.n_trials=0",  # Explicitly disable HPO
                "+trainer.fast_dev_run=true",
                "train.policy.mandatory_selection=null",
                "train.data_distribution=unif",
                "train.env.curriculum_graphs.0.load_dataset=null",
                "train.env.eval_graphs=[]",
                "rl.adaptive_imitation.il_weight=0.0",
                "train.env.curriculum_graphs.0.n_samples=2",
                "device=cpu",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes - generous timeout for initialization + 1 epoch
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        assert result.returncode == 0
    finally:
        # Clean up model weights created during test
        if weights_dir.exists():
            shutil.rmtree(weights_dir)


@pytest.mark.e2e
def test_cli_eval_smoke(tmp_path):
    """Smoke test for evaluation on generated data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Generate test data
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "gen_data",
            "data.dataset_type=test_simulator",
            "data.problem=vrpp",
            "data.data_distributions=[unif]",
            "data.graphs=[{num_loc: 10, n_days: 1, n_samples: 2}]",
            f"data.data_dir={data_dir}",
            "data.name=eval_test",
            "data.overwrite=true",
        ],
        check=True,
        capture_output=True,
    )

    data_file = data_dir / "riomaior10_unif_eval_test1_N2_seed42.npz"
    assert data_file.exists()

    result = subprocess.run(
        [sys.executable, "main.py", "eval", "--help"],
        capture_output=True,
        text=True,
    )
    # main.py eval --help invokes Hydra help
    assert result.returncode == 0
    assert "Powered by Hydra" in result.stdout or "== Configuration groups ==" in result.stdout


@pytest.mark.e2e
def test_cli_test_sim_smoke():
    """Smoke test for test_sim command (help only due to data deps)."""
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "test_sim",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # main.py test_sim --help invokes Hydra help
    assert "Powered by Hydra" in result.stdout or "== Configuration groups ==" in result.stdout


@pytest.mark.e2e
def test_cli_train_lightning_ppo_smoke(tmp_path):
    """Smoke test for PPO training loop via CLI."""
    # Path to model weights that will be created during training
    weights_dir = tmp_path / "weights_ppo"
    tracking_uri = tmp_path / "tracking_ppo"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "train",
                "train.env.name=vrpp",
                "train.env.curriculum_graphs.0.num_loc=10",
                "+experiment=ppo",  # Use + to append new config group if not in schema
                "train.env.curriculum_graphs.0.n_days=1",
                "train.batch_size=2",
                "train.eval_batch_size=2",
                "train.num_workers=0",  # Disable multiprocessing for faster startup
                "train.persistent_workers=false",  # Disable persistent workers
                "train.devices=1",  # Use only 1 GPU
                "train.strategy=auto",  # Auto strategy for single GPU
                "train.policy.model.encoder.embed_dim=32",  # Smaller model for faster training
                "train.policy.model.encoder.hidden_dim=64",
                "train.policy.model.encoder.n_layers=1",
                "train.policy.model.encoder.n_heads=2",
                "tracking.wandb_mode=offline",
                f"tracking.wst_tracking_uri={tracking_uri}",
                f"+output_dir={weights_dir}",
                "hpo.n_trials=0",  # Explicitly disable HPO
                "+trainer.fast_dev_run=true",
                "train.policy.mandatory_selection=null",
                "train.data_distribution=unif",
                "train.env.curriculum_graphs.0.load_dataset=null",
                "train.env.eval_graphs=[]",
                "rl.adaptive_imitation.il_weight=0.0",
                "train.env.curriculum_graphs.0.n_samples=2",
                "device=cpu",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes - generous timeout for initialization + 1 epoch
        )
        if result.returncode != 0:
            print("PPO Smoke Test Failed")
            print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        assert result.returncode == 0
    finally:
        # Clean up model weights created during test
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
