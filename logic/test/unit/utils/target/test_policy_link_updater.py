"""Round-trip tests for the merged policy-link updater (M-mistral-01).

Writes a synthetic ``logic/configs/policies`` tree in a tmp_path, runs both
public sides of :mod:`logic.src.utils.target.policy_link_updater` against
it, and asserts the rewritten override text. These tests are the
precondition the refactor row set before the two updater modules were
merged: no test covered them before.
"""

import os
from pathlib import Path

from logic.src.utils.target.policy_link_updater import (
    list_available_ms_strategies,
    list_available_ri_improvers,
    list_improver_keys,
    list_strategy_keys,
    update_mandatory_selection,
    update_route_improvement,
)

POLICY_YAML = """name: test_constructor
mandatory_selection: { "other/ms_last_minute.yaml": ["old_key"] }
route_improvement: { "other/ri_cls.yaml": ["old_key"] }
other_key: 1
"""

MS_YAML = "last_minute_cf70:\n  threshold: 70\nlast_minute_cf90:\n  threshold: 90\n"
RI_YAML = "ftsp:\n  time_limit: 30\ncls:\n  iterations: 1000\n"


def _make_tree(tmp_path: Path) -> str:
    configs = tmp_path / "configs" / "policies"
    other = configs / "other"
    other.mkdir(parents=True)
    (other / "ms_last_minute.yaml").write_text(MS_YAML, encoding="utf-8")
    (other / "ri_ftsp.yaml").write_text(RI_YAML, encoding="utf-8")
    (configs / "policy_testc.yaml").write_text(POLICY_YAML, encoding="utf-8")
    return str(configs)


def test_update_mandatory_selection_roundtrip(tmp_path):
    configs = _make_tree(tmp_path)
    modified = update_mandatory_selection(
        constructors=["testc"],
        ms_yaml="ms_last_minute",
        keys=["last_minute_cf70", "last_minute_cf90"],
        configs_dir=configs,
        verbose=False,
    )
    assert len(modified) == 1
    filepath, count = modified[0]
    assert count == 1
    assert os.path.basename(filepath) == "policy_testc.yaml"
    content = Path(filepath).read_text(encoding="utf-8")
    assert (
        'mandatory_selection: { "other/ms_last_minute.yaml": ["last_minute_cf70", "last_minute_cf90"] }'
        in content
    )
    # the unrelated override is untouched
    assert 'route_improvement: { "other/ri_cls.yaml": ["old_key"] }' in content


def test_update_route_improvement_roundtrip(tmp_path):
    configs = _make_tree(tmp_path)
    modified = update_route_improvement(
        constructors=["testc"],
        ri_yaml="ri_ftsp",
        keys=["ftsp"],
        configs_dir=configs,
        verbose=False,
    )
    assert len(modified) == 1
    content = Path(modified[0][0]).read_text(encoding="utf-8")
    assert 'route_improvement: { "other/ri_ftsp.yaml": ["ftsp"] }' in content
    assert 'mandatory_selection: { "other/ms_last_minute.yaml": ["old_key"] }' in content


def test_listing_helpers(tmp_path):
    configs = _make_tree(tmp_path)
    assert list_available_ms_strategies(configs) == ["ms_last_minute"]
    assert list_available_ri_improvers(configs) == ["ri_ftsp"]
    assert set(list_strategy_keys("ms_last_minute", configs)) == {
        "last_minute_cf70",
        "last_minute_cf90",
    }
    assert list_improver_keys("ri_ftsp.yaml", configs) == ["ftsp", "cls"]


def test_unknown_key_raises(tmp_path):
    configs = _make_tree(tmp_path)
    try:
        update_mandatory_selection(
            constructors=["testc"],
            ms_yaml="ms_last_minute",
            keys=["missing_key"],
            configs_dir=configs,
            verbose=False,
        )
    except ValueError as exc:
        assert "missing_key" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")


def test_dry_run_does_not_write(tmp_path):
    configs = _make_tree(tmp_path)
    modified = update_mandatory_selection(
        constructors=["testc"],
        ms_yaml="ms_last_minute",
        keys=["last_minute_cf70"],
        configs_dir=configs,
        dry_run=True,
        verbose=False,
    )
    assert len(modified) == 1
    content = (Path(configs) / "policy_testc.yaml").read_text(encoding="utf-8")
    assert '["old_key"]' in content
