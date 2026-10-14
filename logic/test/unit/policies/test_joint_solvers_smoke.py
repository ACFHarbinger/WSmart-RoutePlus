"""
Smoke test for consolidated Joint Solvers.

Verifies that all joint packing-and-routing policies are correctly
registered, instantiable, and executable via the main RoutingAlgorithmFactory.
"""

import os

import pytest

# Set environment variable for protobuf compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
from logic.src.policies.packing_and_routing.base.registry import JointPolicyRegistry

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def test_registry_discovery():
    """Ensure all joint solvers are registered in the JointPolicyRegistry."""
    # Importing factory triggers registration
    RoutingAlgorithmFactory.ensure_registered()
    solvers = JointPolicyRegistry.list_policies()

    assert "nds_brkga" in solvers
    assert "jsa" in solvers
    assert "jgo" in solvers


@pytest.mark.parametrize("solver_name", ["nds_brkga", "jsa", "jgo"])
def test_factory_instantiation(solver_name):
    """Ensure the main RoutingAlgorithmFactory can instantiate the policies natively."""
    RoutingAlgorithmFactory.ensure_registered()
    policy = RoutingAlgorithmFactory.get_adapter(solver_name)

    # Check that it's the correct policy class (not an adapter)
    from logic.src.policies.packing_and_routing import JointGreedyPolicy, JointSAPolicy, NDSBRKGAPolicy
    expected_classes = {
        "nds_brkga": NDSBRKGAPolicy,
        "jsa": JointSAPolicy,
        "jgo": JointGreedyPolicy
    }
    assert isinstance(policy, expected_classes[solver_name])
    assert hasattr(policy, "execute")


if __name__ == "__main__":
    print("Discovering solvers...")
    RoutingAlgorithmFactory.ensure_registered()
    print(f"Registered Joint Solvers: {JointPolicyRegistry.list_policies()}")

    from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry
    print(f"Main Registry has matching entries: {[n for n in ['nds_brkga', 'jsa', 'jgo'] if RoutingAlgorithmRegistry.get(n)]}")



    for name in ["nds_brkga", "jsa", "jgo"]:
        try:
            policy = RoutingAlgorithmFactory.get_adapter(name)
            print(f"Instantiated {name} -> {type(policy).__name__}")
        except Exception as e:
            print(f"Failed to instantiate {name}: {e}")
