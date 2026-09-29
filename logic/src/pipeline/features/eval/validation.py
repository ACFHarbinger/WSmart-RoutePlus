"""
Validation logic for evaluation pipeline.

Attributes:
    validate_eval_config: Validate and sanitize evaluation configuration.
    validate_eval_args: Alias for validate_eval_config.

Example:
    >>> config = Config()
    >>> validate_eval_config(config)
"""

import re

from logic.src.configs import Config
from logic.src.constants import MAP_DEPOTS, WASTE_TYPES


def validate_eval_config(cfg: Config) -> None:
    """
    Validates and sanitizes evaluation configuration values in-place.

    Performs the same checks previously done by ``validate_eval_args``
    on the flattened args dict, now applied directly to the typed Config.

    Args:
        cfg: Root Hydra configuration with ``cfg.eval`` containing evaluation
            parameters.

    Raises:
        AssertionError: If any validation constraint is violated.
    """
    ev = cfg.eval
    # EvalConfig carries the graph under ``env`` (first eval graph) or not at all when
    # evaluating a pre-generated dataset file; sanitize only when a graph is present.
    graph = getattr(ev, "graph", None)
    if graph is None:
        env_cfg = getattr(ev, "env", None)
        eval_graphs = getattr(env_cfg, "eval_graphs", None) or []
        graph = getattr(env_cfg, "graph", None) or (eval_graphs[0] if eval_graphs else None)

    # --- Output filename constraint ---
    if ev.output_filename is not None:
        datasets = ev.datasets or []
        beam_widths = ev.decoding.beam_width if ev.decoding else None
        bw_count = len(beam_widths) if isinstance(beam_widths, (list, tuple)) else (1 if beam_widths else 0)
        assert len(datasets) == 1 and bw_count <= 1, (
            "Cannot specify result filename with more than one dataset or more than one beam_width"
        )

    if graph is None:
        return

    # --- Sanitize area ---
    area = re.sub(r"[^a-zA-Z]", "", (graph.area or "").lower())
    assert area in MAP_DEPOTS, f"Unknown area {area}, available areas: {list(MAP_DEPOTS.keys())}"
    graph.area = area

    # --- Sanitize waste_type ---
    waste = re.sub(r"[^a-zA-Z]", "", (graph.waste_type or "").lower())
    assert waste in WASTE_TYPES or waste == "", (
        f"Unknown waste type {waste}, available waste types: {list(WASTE_TYPES.keys())}"
    )
    if waste:
        graph.waste_type = waste


# Keep backward-compatible alias
validate_eval_args = validate_eval_config
