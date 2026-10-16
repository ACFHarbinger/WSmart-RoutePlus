"""Observe baseline BPC proofs without changing the frozen optimization code.

The legacy CG logger emits its bound only after convergence and exhaustion.
This observer retains the full-precision local value at that log event and
pairs it with the returned incumbent. It never recomputes or fabricates a bound.
Newer trees publish proof metadata natively and need no observer.
"""

import importlib
import importlib.util
import inspect
import logging
import math
import os
import runpy
import sys
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_PROOF = ContextVar("gate_baseline_bpc_proof", default=None)


class _RootProofFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if "z_UB=" not in record.getMessage():
            return True
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "column_generation_loop":
                local = frame.f_locals
                if (
                    local.get("node_depth") == 0
                    and local.get("converged")
                    and local.get("pricing_exhausted")
                    and not local.get("timed_out")
                ):
                    bound = local.get("lagrangian_ub")
                    if isinstance(bound, (int, float)) and math.isfinite(bound):
                        prior = _PROOF.get()
                        bound = min(bound, prior["bound"]) if prior else bound
                        _PROOF.set({"bound": float(bound), "source": "root_cg_exhausted"})
                break
            frame = frame.f_back
        return True


def install_legacy_observer() -> bool:
    """Install logging-only proof capture when the target tree has no native publisher."""
    status = importlib.import_module("logic.src.pipeline.simulations.solver_status")
    if hasattr(status, "current_solver_certificate"):
        return False
    module = importlib.import_module(
        "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine"
    )
    policy = importlib.import_module(
        "logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.policy_bpc"
    )
    original = module.run_bpc
    fingerprint_path = Path(__file__).resolve().parents[2] / "logic/src/utils/routing/problem_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gate_problem_fingerprint", fingerprint_path)
    fingerprint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fingerprint_module)
    fingerprint = fingerprint_module.problem_fingerprint

    @wraps(original)
    def observed_solve(*args, **kwargs):
        _PROOF.set(None)
        bound_args = inspect.signature(original).bind(*args, **kwargs)
        bound_args.apply_defaults()
        values = bound_args.arguments
        instance = fingerprint(
            values["dist_matrix"],
            values["wastes"],
            values["capacity"],
            values["R"],
            values["C"],
            values.get("mandatory_indices"),
            values.get("vehicle_limit"),
        )
        result = original(*args, **kwargs)
        proof = _PROOF.get()
        incumbent = float(result[1])
        if proof is not None and math.isfinite(incumbent) and proof["bound"] >= incumbent - 1e-6:
            _PROOF.set(dict(proof, incumbent=incumbent, instance=instance))
        else:
            _PROOF.set(None)
        return result

    module.run_bpc = observed_solve
    policy.run_bpc = observed_solve
    cg_logger = logging.getLogger("logic.src.policies.helpers.solvers_and_matheuristics.search.column_generation")
    cg_logger.setLevel(logging.INFO)
    cg_logger.addFilter(_RootProofFilter())

    from logic.src.pipeline.simulations.actions.logging import LogAction

    execute = LogAction.execute

    @wraps(execute)
    def observed_log(self, context):
        result = execute(self, context)
        daily = context.get("daily_log") if isinstance(context, dict) else context.daily_log
        proof = _PROOF.get()
        if isinstance(daily, dict):
            daily["solver_certificate"] = dict(proof) if proof and "incumbent" in proof else None
        _PROOF.set(None)
        return result

    LogAction.execute = observed_log
    return True


def main() -> None:
    """Launch the target checkout's CLI under the proof observer."""
    sys.path.insert(0, os.getcwd())
    install_legacy_observer()
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
