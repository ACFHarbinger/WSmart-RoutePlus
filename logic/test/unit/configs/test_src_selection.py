"""SRC's yaml must name a packing, or the simulator hands it an empty set every day."""

from pathlib import Path

from omegaconf import OmegaConf


def test_src_yaml_sets_a_packing():
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "policies" / "policy_src.yaml"
    section = OmegaConf.to_container(OmegaConf.load(yaml_path))["src"]
    merged = {k: v for item in section for k, v in item.items()}
    assert merged.get("packing") == {"other/ms_lookahead.yaml": ["default"]}
