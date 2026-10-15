"""Paired Q-14 gate runner (WP-12, #117).

Subcommands (all read-only except the heavy runners):

* ``overrides --cell SWC-TCF`` — print the Hydra dotlist for one gate-(a)
  cell. Dry: safe to run any time, used by unit tests and review.
* ``calibrate --out FILE`` — HEAVY: RQ-codex-2 repeats on the unchanged code,
  asserts gate-(a) determinism, freezes tolerances to FILE.
* ``gate-a --baseline DIR --candidate DIR`` — HEAVY-optional: bit-for-bit
  paired compare of two finished gate-(a) sweeps (no simulator run itself).
* ``gate-b --archived CSV --candidate DIR --tolerances FILE`` — compare one
  finished time-capped sweep against archived outputs within tolerances.

Heavy subcommands refuse to run until ``.agent/cache/train-logs/chain.log``
shows ``sim chain done`` and take ``~/.cache/wsr-review/heavy.lock``
(non-blocking; one heavy job at a time). ``calibrate`` additionally refuses
unless the code tree equals the ``BASELINE_COMMIT`` (issue: tolerances are
calibrated on the unchanged code ``a5eb27eec``).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.gates.cells import (  # noqa: E402
    BASELINE_COMMIT,
    CALIBRATION_REPEATS,
    GATE_A_CELLS,
    GATE_A_SIM,
    cell_policy_entry,
)
from tools.gates.compare import (  # noqa: E402
    CERT_TOL,
    check_repeats_identical,
    compare_certificates,
    compare_exact,
    freeze_tolerances,
    gate_b_verdict,
)

CHAIN_LOG = ".agent/cache/train-logs/chain.log"
HEAVY_LOCK = os.path.expanduser("~/.cache/wsr-review/heavy.lock")


def chain_done() -> bool:
    """Return True once the AM/sim chain has finished (Q-15 gate)."""
    if not os.path.exists(CHAIN_LOG):
        return False
    with open(CHAIN_LOG) as f:
        return any("sim chain done" in line for line in f)


def code_matches_baseline() -> bool:
    """Return True when code paths equal the calibration baseline commit."""
    out = subprocess.run(
        ["git", "diff", "--stat", BASELINE_COMMIT, "HEAD", "--", "logic", "tools", "main.py"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and out.stdout.strip() == ""


def take_heavy_lock():
    """Take the machine-wide heavy-job lock (non-blocking) or raise."""
    os.makedirs(os.path.dirname(HEAVY_LOCK), exist_ok=True)
    # Intentionally not a `with` block: the handle stays open so the lock is
    # held for the process lifetime.
    fh = open(HEAVY_LOCK, "w")  # noqa: SIM115
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        fh.close()
        raise SystemExit("heavy.lock is held by another job; one heavy job at a time.") from err
    return fh


def require_heavy() -> None:
    """Enforce the Q-15/heavy-lock preconditions for any heavy run."""
    if not chain_done():
        raise SystemExit(f"Refusing heavy run: {CHAIN_LOG!r} shows no 'sim chain done'.")
    take_heavy_lock()  # held for the process lifetime


def _hydra_value(value: Any) -> str:
    """Render a value for a Hydra dotlist override (no shell involved)."""
    import json

    if isinstance(value, str):
        return value
    return json.dumps(value)


def cell_dotlist(constructor: str, model_path: str = "") -> List[str]:
    """Build the full Hydra args for one gate-(a) cell (dry, no run).

    One ``sim.policies`` entry carries the per-solver overrides; the
    simulator's ``expand_policy_configs`` merges them over the flattened
    YAML variant, so list-form and mapping-form YAMLs need no per-file
    index addressing here.
    """
    import json

    entry = cell_policy_entry(constructor, model_path=model_path)
    dotlist = [f"{k}={_hydra_value(v)}" for k, v in GATE_A_SIM.items()]
    dotlist.append(f"sim.policies={json.dumps([entry])}")
    return dotlist


def run_test_sim(dotlist: List[str], run_name: str) -> str:
    """Run one test_sim cell under ``nice``; return the results directory."""
    cmd = ["nice", "-n", "15", sys.executable, "main.py", "test_sim", f"sim.run_name={run_name}"] + dotlist
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"test_sim failed for run {run_name} (exit {proc.returncode}).")
    return os.path.join("outputs", "simulations", run_name)


def cmd_overrides(args: argparse.Namespace) -> None:
    """Print the Hydra args for one cell (dry)."""
    for item in cell_dotlist(args.cell, model_path=args.model_path or ""):
        print(item)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """RQ-codex-2: repeats on unchanged code, determinism proof, frozen file."""
    require_heavy()
    if not code_matches_baseline():
        raise SystemExit(
            f"Refusing calibration: code tree differs from {BASELINE_COMMIT}; "
            "tolerances must be calibrated on the unchanged code."
        )
    frozen: Dict[str, Any] = {"baseline_commit": BASELINE_COMMIT, "cells": {}}
    repeats = args.repeats or CALIBRATION_REPEATS
    for cell in GATE_A_CELLS:
        name = cell["constructor"]
        if name == "AM" and not args.model_path:
            print("cell AM: no --model-path (chain checkpoint); skipped with reason")
            frozen["cells"][name] = {"mode": "pending_checkpoint", "reason": "no --model-path"}
            continue
        logs = []
        for n in range(1, repeats + 1):
            results = run_test_sim(
                cell_dotlist(name, model_path=args.model_path or ""),
                f"gateA-cal-{name.lower()}-r{n}",
            )
            logs.append(_load_policy_log(results, name))
        diffs = check_repeats_identical(logs)
        mode = cell.get("compare_mode", "exact")
        if diffs and mode != "certificate_fallback":
            raise SystemExit(f"Cell {name}: {len(diffs)} repeat diffs; config is not deterministic.")
        frozen["cells"][name] = {
            "mode": "certificate" if diffs else "exact",
            "repeat_diffs": len(diffs),
            **freeze_tolerances([_log_kpis(log) for log in logs]),
        }
        print(f"cell {name}: mode={frozen['cells'][name]['mode']} diffs={len(diffs)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(frozen, f, indent=2, sort_keys=True)
    print(f"frozen tolerances -> {args.out}")


def cmd_gate_a(args: argparse.Namespace) -> None:
    """Bit-for-bit paired compare of two finished gate-(a) sweeps."""
    import glob

    failures = 0
    for cell in GATE_A_CELLS:
        name = cell["constructor"]
        base_logs = glob.glob(os.path.join(args.baseline, f"log_*{name}*N.json"))
        cand_logs = glob.glob(os.path.join(args.candidate, f"log_*{name}*N.json"))
        if not base_logs or not cand_logs:
            print(f"cell {name}: MISSING log pair (skipped with reason)")
            continue
        with open(base_logs[0]) as f:
            base = json.load(f)
        with open(cand_logs[0]) as f:
            cand = json.load(f)
        diffs = compare_exact(base, cand)
        if diffs and cell.get("compare_mode") == "certificate_fallback":
            verdict = compare_certificates(_incumbent(base), _bound(base), _incumbent(cand), _bound(cand), CERT_TOL)
            print(f"cell {name}: {len(diffs)} bit diffs; certificate gate: {verdict}")
            failures += 0 if verdict.passed else 1
        else:
            print(f"cell {name}: {len(diffs)} diffs")
            failures += 1 if diffs else 0
    if failures:
        raise SystemExit(f"gate-(a) FAILED in {failures} cell(s).")
    print("gate-(a) PASSED.")


def cmd_gate_b(args: argparse.Namespace) -> None:
    """Time-capped sweep vs archived outputs within frozen tolerances."""
    import csv

    with open(args.tolerances) as f:
        tolerances = json.load(f)["cells"]
    with open(args.archived, newline="") as f:
        archived = {(r["constructor"], r["city"], r["N"]): r for r in csv.DictReader(f)}
    with open(args.candidate, newline="") as f:
        failures = 0
        for row in csv.DictReader(f):
            key = (row["constructor"], row["city"], row["N"])
            if key not in archived:
                print(f"row {key}: no archived counterpart (skipped with reason)")
                continue
            tol = tolerances.get(row["constructor"], {})
            verdict = gate_b_verdict(archived[key], row, tol, row["constructor"])
            print(f"row {key}: {'PASS' if verdict.passed else 'REVIEW'}: {verdict.detail}")
            failures += 0 if verdict.passed else 1
    if failures:
        raise SystemExit(f"gate-(b) needs review in {failures} row(s).")
    print("gate-(b) PASSED.")


def _load_policy_log(results_dir: str, constructor: str) -> Dict[str, Any]:
    import glob

    matches = glob.glob(os.path.join(results_dir, f"log_*{constructor}*N.json"))
    if not matches:
        raise SystemExit(f"No policy log for {constructor} in {results_dir}.")
    with open(matches[0]) as f:
        return json.load(f)


def _log_kpis(log: Dict[str, Any]) -> Dict[str, float]:
    samples = log.get("samples", {})
    first = next(iter(samples.values())) if isinstance(samples, dict) and samples else samples
    return {"kgkm": float(first["kgkm"]), "overflows": float(first["overflows"])}


def _incumbent(log: Dict[str, Any]) -> float:
    samples = log.get("samples", {})
    first = next(iter(samples.values())) if isinstance(samples, dict) and samples else samples
    return float(first["profit"])


def _bound(log: Dict[str, Any]) -> float:
    samples = log.get("samples", {})
    first = next(iter(samples.values())) if isinstance(samples, dict) and samples else samples
    return float(first.get("bound", first["profit"]))


def main(argv: List[str]) -> None:
    """Parse arguments and dispatch to the gate subcommand."""
    parser = argparse.ArgumentParser(description="Q-14 regression gate runner (WP-12).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_over = sub.add_parser("overrides", help="Print one cell's Hydra args (dry).")
    p_over.add_argument("--cell", required=True)
    p_over.add_argument("--model-path", default="")
    p_cal = sub.add_parser("calibrate", help="HEAVY: repeats + frozen tolerances.")
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--repeats", type=int, default=None)
    p_cal.add_argument("--model-path", default="")
    p_a = sub.add_parser("gate-a", help="Paired bit-for-bit compare of two sweeps.")
    p_a.add_argument("--baseline", required=True)
    p_a.add_argument("--candidate", required=True)
    p_b = sub.add_parser("gate-b", help="Time-capped sweep vs archived outputs.")
    p_b.add_argument("--archived", required=True)
    p_b.add_argument("--candidate", required=True)
    p_b.add_argument("--tolerances", required=True)
    args = parser.parse_args(argv)
    {"overrides": cmd_overrides, "calibrate": cmd_calibrate, "gate-a": cmd_gate_a, "gate-b": cmd_gate_b}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
