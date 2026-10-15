"""Q-14(a) gate cells and per-solver timing-off overrides (WP-12, #117).

Cell definition (owner ruling Q-14): each released routing algorithm x one
packing algorithm x CLS, on Rio Maior 100 over 5 days, seed 42.

Packing choice: ``last_minute_cf70`` (the paper's ``lm`` baseline; first
variant in ``other/ms_last_minute.yaml``). Improver: CLS (``other/ri_cls.yaml``
``default``). Acceptance stays each solver's paper wiring (ALNS/BMC, HGS/OI);
the harness does not re-wire it.

Override mechanism: one ``sim.policies`` entry per cell,
``{policy: {field: value, ...}}``. ``expand_policy_configs``
(``features/test/config.py``) flattens the YAML variant and merges the
caller's dict over it, and a caller-supplied packing replaces the YAML
fan-out — so this works for list-form and mapping-form YAMLs alike, pre-
and post-P-13. Unknown fields raise at structured-load time (WP-03), so a
typo fails loudly instead of silently running paper defaults.

Gate-(a) "timing off" convention: ``time_limit: 0`` disables every
``time_limit > 0`` wall-clock guard (verified across the routing tree,
including HGS ``hgs.py:627`` and the BPC outer/CG/node budgets). Fixed
iteration caps from the paper YAMLs then bind first. RQ-codex-2 calibration
repeats prove bit-identity empirically; any cell that fails it drops to
certificate/tolerance mode (BPC does so by design, RQ-kimi-1).

Per-solver notes:

* BPC: ``time_limit: 0`` disables the outer loop, CG and per-node budgets.
  ``rcspp_timeout`` is *not* YAML-addressable (params default 30 s), so exact
  pricing can still truncate on wall clock: bit-identity is attempted, and
  the RQ-kimi-1 certificate gate (``test_bpc_exactness_gate.py``, WP-11) is
  the accepted fallback. Calibration evidence selects the mode.
* HGS: ``restart_timer`` must also be 0 or it restarts on wall clock.
* CVRP: OR-Tools GLS under a wall clock is not deterministic; the
  deterministic budget is ``solution_limit`` (WP-12 knob, default None).
* AM runs as the ``na`` (NeuralAgent) simulator policy serving the chain's
  AM checkpoint with greedy decoding. Skipped with reason when the
  checkpoint is absent.
* SWC-TCF (Gurobi): ``time_limit: 0`` semantics for the TCF backend are
  confirmed at calibration; if Gurobi threading proves nondeterministic the
  cell pins single-threaded solve there (documented, no code change here).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Unchanged-code baseline the tolerances are calibrated on (issue #117).
BASELINE_COMMIT = "a5eb27eec"

# Gate-(a) shared simulation axes (Hydra dotlist values).
GATE_A_SIM: Dict[str, Any] = {
    "sim.graph.area": "riomaior",
    "sim.graph.num_loc": 100,
    "sim.graph.n_days": 5,
    "sim.graph.n_samples": 1,
    "sim.data_distribution": "emp",
    "seed": 42,
    "sim.graph.load_dataset": "data/simulator/datasets/riomaior100_emp_wsr5_N1_seed42.npz",
}

# Gate-(a) packing x improver pins (paper lm_cls baseline).
GATE_A_PACKING = {"other/ms_last_minute.yaml": ["last_minute_cf70"]}
GATE_A_IMPROVER = {"other/ri_cls.yaml": ["default"]}

# Per-cell caller overrides merged over the flattened YAML variant.
# ``policy`` is the test_sim variant key; ``constructor`` is the summary label.
GATE_A_CELLS: List[Dict[str, Any]] = [
    {"constructor": "SWC-TCF", "policy": "swc_tcf", "overrides": {"time_limit": 0}},
    {
        "constructor": "BPC",
        "policy": "bpc",
        "overrides": {"time_limit": 0},
        "compare_mode": "certificate_fallback",
    },
    {"constructor": "ALNS", "policy": "alns", "overrides": {"time_limit": 0}},
    {"constructor": "SANS", "policy": "sans", "overrides": {"time_limit": 0}},
    {
        "constructor": "HGS",
        "policy": "hgs",
        "overrides": {"time_limit": 0, "restart_timer": 0},
    },
    {"constructor": "PSOMA", "policy": "psoma", "overrides": {"time_limit": 0}},
    {"constructor": "ACO-HH", "policy": "aco_hh", "overrides": {"time_limit": 0}},
    {"constructor": "PG-CLNS", "policy": "pg_clns", "overrides": {"time_limit": 0}},
    {
        "constructor": "AM",
        "policy": "na",
        # model_path points at the chain's released checkpoint at run time;
        # greedy decoding (YAML default) is deterministic.
        "overrides": {"model_path": "CHAIN_CHECKPOINT_PLACEHOLDER"},
    },
    {
        "constructor": "CVRP",
        "policy": "cvrp",
        "overrides": {"solution_limit": 1000},
    },
]

# Calibration protocol (RQ-codex-2).
CALIBRATION_REPEATS = 3


def cell_names() -> List[str]:
    """Return the gate-(a) constructor names in gate order."""
    return [c["constructor"] for c in GATE_A_CELLS]


def get_cell(constructor: str) -> Dict[str, Any]:
    """Return the cell record for a constructor (case-insensitive)."""
    for cell in GATE_A_CELLS:
        if cell["constructor"].lower() == constructor.lower():
            return cell
    raise KeyError(f"Unknown gate-(a) constructor: {constructor!r}")


def cell_policy_entry(constructor: str, model_path: str = "") -> Dict[str, Any]:
    """Build one ``sim.policies`` entry for a gate-(a) cell.

    Args:
        constructor: Gate constructor label (e.g. ``"BPC"``).
        model_path: Released AM checkpoint dir; fills the AM cell's
            ``model_path`` placeholder.

    Returns:
        ``{policy: merged_overrides}`` ready for the ``sim.policies`` list.
    """
    cell = get_cell(constructor)
    overrides: Dict[str, Any] = {
        "packing": GATE_A_PACKING,
        "route_improvement": GATE_A_IMPROVER,
    }
    overrides.update(cell.get("overrides", {}))
    if overrides.get("model_path") == "CHAIN_CHECKPOINT_PLACEHOLDER":
        if not model_path:
            raise ValueError("AM cell needs --model-path (chain checkpoint).")
        overrides["model_path"] = model_path
    return {cell["policy"]: overrides}
