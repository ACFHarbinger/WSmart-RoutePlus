"""Surviving config-loading and CLI defects from the 2026-10-20 Cursor review."""

import argparse
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import logic.controllers.hydra_dispatch  # noqa: F401  registers the Config schema as "config"

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS_DIR = _REPO_ROOT / "logic" / "configs"


def _compose(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(_CONFIGS_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


@pytest.mark.parametrize("task", ["meta_train", "hpo_sim", "slurm"])
def test_task_composes_against_the_schema(task: str) -> None:
    cfg = _compose(f"tasks={task}")
    assert cfg is not None


def test_sim_days_and_samples_reach_the_day_loop() -> None:
    cfg = _compose("tasks=test_sim", "sim.days=2", "sim.n_samples=3")
    assert cfg.sim.graph.n_days == 2
    assert cfg.sim.graph.n_samples == 3


def test_sim_defaults_keep_the_paper_horizon_and_dataset_name() -> None:
    cfg = _compose("tasks=test_sim")
    assert cfg.sim.graph.n_days == 30
    assert cfg.sim.graph.load_dataset.endswith("_wsr30_N1_seed42.npz")


@pytest.mark.parametrize(
    "task, path",
    [("test_sim", "sim.seed"), ("eval", "eval.seed"), ("gen_data", "data.seed")],
)
def test_root_seed_reaches_the_task_seed(task: str, path: str) -> None:
    cfg = _compose(f"tasks={task}", "seed=7")
    assert OmegaConf.select(cfg, path) == 7


def test_file_system_delete_selects_nothing_by_default() -> None:
    from logic.controllers.cli.fs_parser_args import add_files_delete_args

    parser = add_files_delete_args(argparse.ArgumentParser())
    opts = vars(parser.parse_args([]))
    assert not opts["wandb"]
    assert not opts["log"]
    assert not opts["output"]


def test_file_system_delete_collects_simulator_outputs(tmp_path, monkeypatch) -> None:
    from logic.controllers.cli import fs_parser

    (tmp_path / "sim_out").mkdir()
    (tmp_path / "sim_ckpt").mkdir()
    monkeypatch.setattr(fs_parser, "ROOT_DIR", str(tmp_path))
    found = fs_parser._collect_directories_to_delete(
        {"test": True, "test_dir": "sim_out", "test_checkpoint": True, "test_checkpoint_dir": "sim_ckpt"}
    )
    paths = {Path(p).name for _, p in found}
    assert {"sim_out", "sim_ckpt"} <= paths


def test_update_packing_rewrites_a_block_section(tmp_path) -> None:
    from logic.src.utils.target.policy_link_updater import update_packing

    other = tmp_path / "other"
    other.mkdir()
    (other / "pk_last_minute.yaml").write_text("last_minute_cf70:\n  strategy: last_minute\n")
    policy = tmp_path / "policy_demo.yaml"
    policy.write_text(
        "demo:\n"
        "  custom:\n"
        "    packing:\n"
        "      other/packing/pk_lookahead.yaml:\n"
        "      - default\n"
        "    time_limit: 60.0\n"
    )
    modified = update_packing(["demo"], "pk_last_minute", ["last_minute_cf70"], configs_dir=str(tmp_path), verbose=False)
    assert modified
    import yaml

    data = yaml.safe_load(policy.read_text())
    assert data["demo"]["custom"]["packing"] == {"other/packing/pk_last_minute.yaml": ["last_minute_cf70"]}
    assert data["demo"]["custom"]["time_limit"] == 60.0


def test_test_suite_points_at_the_real_suite() -> None:
    from logic.controllers.cli.ts_parser import add_test_suite_args
    from logic.test.test_definitions import TEST_MODULES

    opts = add_test_suite_args(argparse.ArgumentParser()).parse_args([])
    assert opts.test_dir == "logic/test"
    for rel in TEST_MODULES.values():
        assert (_REPO_ROOT / "logic" / "test" / rel).is_file(), rel
