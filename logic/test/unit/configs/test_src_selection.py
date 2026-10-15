"""SRC's yaml must name a packing, or the simulator hands it an empty set every day."""

from pathlib import Path

from omegaconf import OmegaConf


def test_src_yaml_sets_a_packing():
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "policies" / "policy_src.yaml"
    section = OmegaConf.to_container(OmegaConf.load(yaml_path))["src"]
    assert section.get("packing") == {"other/pk_lookahead.yaml": ["default"]}
