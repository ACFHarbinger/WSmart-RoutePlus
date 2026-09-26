"""ZenML simulation pipeline for WSmart-Route.

Wraps :func:`~logic.src.pipeline.features.test.engine.run_simulator_test`
in a ZenML pipeline with **per-policy fan-out** steps:

1. **prepare_sim_config** — serialise the Hydra config to a plain dict.
2. **run_policy_step** (×N) — one step per policy, each with an
   ``experiment_tracker`` decorator so ZenML manages an MLflow run.
   Steps are chained via a ``batch_id: str`` pass-through to create a
   sequential DAG dependency.
3. **aggregate_sim_results** — collect ``batch_id`` from the last policy
   step and log a summary.

The pipeline is invoked from
:func:`~logic.src.pipeline.features.test.engine._run_sim_via_zenml`
when ``cfg.tracking.zenml_enabled`` is ``True``.

Attributes:
    _ZENML_AVAILABLE: Whether ZenML is available.
    zenml_pipeline: ZenML pipeline function.
    step: ZenML step function.
    prepare_sim_config: Prepares the simulation configuration.
    run_policy_step: Runs the simulation for a single policy.
    aggregate_sim_results: Aggregates the simulation results.

Example:
    >>> from logic.src.pipeline.features.test.zenml_sim_pipeline import simulation_pipeline
    >>> simulation_pipeline(config)
    Traceback (most recent call last):
        ...  # doctest: +ELLIPSIS
    ImportError: zenml is not installed — cannot run ZenML simulation pipeline
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from omegaconf import OmegaConf

from logic.src.configs import Config
from logic.src.tracking.integrations.zenml_bridge import ZenMLBridge
from logic.src.tracking.logging.pylogger import get_pylogger

try:
    from zenml import pipeline as zenml_pipeline  # type: ignore[import-not-found]
    from zenml import step  # type: ignore[import-not-found]

    _ZENML_AVAILABLE = True
except ImportError:
    _ZENML_AVAILABLE = False
    zenml_pipeline = None  # type: ignore[assignment]
    step = None  # type: ignore[assignment]

logger = get_pylogger(__name__)

# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

if _ZENML_AVAILABLE:

    @step  # type: ignore[misc]
    def prepare_sim_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Pass-through: makes the serialised config a ZenML artifact.

        Args:
            config_dict: Serialised root config.

        Returns:
            Serialised root config.
        """
        return config_dict

    @step(experiment_tracker="mlflow_tracker")  # type: ignore[misc]
    def run_policy_step(
        config_dict: Dict[str, Any],
        policy_name: str,
        batch_id: str,
    ) -> str:
        """Run the full simulation for a single policy inside a ZenML step.

        The ``experiment_tracker`` decorator causes ZenML to manage an MLflow
        run.  A :class:`ZenMLBridge` is attached so WSTracker metrics are
        dual-written.

        Args:
            config_dict: Serialised root config.
            policy_name: Name of the policy to run (used for step naming).
            batch_id: Pass-through identifier for DAG dependency chaining.

        Returns:
            The same *batch_id* (creates a sequential DAG edge to the next
            policy step).
        """
        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(config_dict))
        cfg = OmegaConf.to_object(cfg)
        assert isinstance(cfg, Config)

        # Filter to only the requested policy
        sim = cfg.sim
        original_policies = list(sim.full_policies)

        filtered_policies = [
            p
            for p in original_policies
            if ((list(p.keys())[0] if p else "") if isinstance(p, dict) else str(p)) == policy_name
        ]

        sim.full_policies = filtered_policies

        bridge = ZenMLBridge()
        from logic.src.pipeline.features.test.engine import run_simulator_test

        run_simulator_test(cfg, sinks=[bridge])
        return batch_id

    @step  # type: ignore[misc]
    def aggregate_sim_results(batch_id: str) -> str:
        """Terminal step — logs completion and returns the batch ID.

        Args:
            batch_id: Pass-through identifier for DAG dependency chaining.

        Returns:
            The same *batch_id* (creates a sequential DAG edge to the next
            policy step).
        """
        logger.info(f"All simulation policies completed (batch {batch_id})")
        return batch_id

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------

    @zenml_pipeline(name="wsmart_route_simulation")  # type: ignore[misc]
    def _simulation_pipeline(
        config_dict: Dict[str, Any],
        policy_names: List[str],
    ) -> str:
        """Per-policy fan-out simulation pipeline.

        Args:
            config_dict: Serialised root config.
            policy_names: List of policy names to run.

        Returns:
            The same *batch_id* (creates a sequential DAG edge to the next
            policy step).
        """
        cfg_art = prepare_sim_config(config_dict)
        batch_id = str(uuid.uuid4())[:8]

        # Sequential chain: each policy step depends on the previous one
        for policy_name in policy_names:
            batch_id = run_policy_step.with_options(  # type: ignore[union-attr, call-arg, misc]
                name=f"sim_{policy_name}",  # type: ignore[call-arg]
            )(config_dict=cfg_art, policy_name=policy_name, batch_id=batch_id)

        return aggregate_sim_results(batch_id)


# ---------------------------------------------------------------------------
# Public entry point (called from engine.py)
# ---------------------------------------------------------------------------


def simulation_pipeline(cfg: Any) -> None:
    """Serialise *cfg* and launch the ZenML simulation pipeline.

    Args:
        cfg: Root Hydra :class:`~logic.src.configs.Config` object.

    Raises:
        ImportError: If ZenML is not installed.
    """
    if not _ZENML_AVAILABLE:
        raise ImportError("zenml is not installed — cannot run ZenML simulation pipeline")

    config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    # Extract policy names from the config
    sim = cfg.sim
    policy_names: List[str] = []
    for p in sim.full_policies:
        p_obj: object = p
        if isinstance(p_obj, dict):
            if p_obj:
                policy_names.append(str(list(p_obj.keys())[0]))
        else:
            policy_names.append(str(p))

    _simulation_pipeline(config_dict=config_dict, policy_names=policy_names)
