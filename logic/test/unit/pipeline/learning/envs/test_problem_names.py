"""The problem names are ptp, cptp and tcptp; the former names are rejected (no aliases)."""

import pytest
from logic.src.models.subnets.embeddings import INIT_EMBEDDING_REGISTRY, get_init_embedding
from logic.src.models.subnets.embeddings.context import CONTEXT_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.edges import EDGE_EMBEDDING_REGISTRY
from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY
from logic.src.pipeline.learning.envs import ENV_TARGETS, GENERATOR_REGISTRY, PROBLEMS, get_env
from logic.src.utils.model.problem_factory import load_problem

pytestmark = [pytest.mark.unit, pytest.mark.fast]

NEW = ("ptp", "cptp", "tcptp")
OLD = ("vrpp", "cvrpp", "ctop", "wcvrp", "cwcvrp", "swcvrp", "scwcvrp", "sdwcvrp")


@pytest.mark.parametrize("name", NEW)
def test_new_names_are_registered(name):
    assert name in ENV_TARGETS
    assert name in GENERATOR_REGISTRY
    assert name in INIT_EMBEDDING_REGISTRY
    assert name in CONTEXT_EMBEDDING_REGISTRY
    assert name in EDGE_EMBEDDING_REGISTRY
    assert name in STATE_EMBEDDING_REGISTRY
    assert name == load_problem(name).NAME


@pytest.mark.parametrize("name", OLD)
def test_old_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown environment"):
        get_env(name)
    with pytest.raises(ValueError):
        get_init_embedding(name, embed_dim=8)
    with pytest.raises(AssertionError, match="unsupported problem"):
        load_problem(name)


@pytest.mark.parametrize("name", NEW)
def test_env_and_spec_registries_list_the_ptp_family(name):
    """The env registry and the ProblemSpec registry must both list the PTP family."""
    assert name in PROBLEMS
    assert name in ENV_TARGETS
