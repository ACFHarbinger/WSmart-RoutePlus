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
unless the code tree equals ``BASELINE_COMMIT`` plus the exact source-hashed
optional-routing adapter amendment (see ``tools/gates/BASELINE.md``).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.gates.cells import (  # noqa: E402
    BASELINE_AMENDMENTS,
    BASELINE_COMMIT,
    CALIBRATION_REPEATS,
    GATE_A_CELLS,
    GATE_A_SIM,
    cell_policy_entry,
)
from tools.gates.compare import (  # noqa: E402
    canonical_gate_log,
    check_repeats_identical,
    compare_daily_certificates,
    compare_exact,
    freeze_tolerances,
    gate_b_verdict,
)

CHAIN_LOG = ".agent/cache/train-logs/chain.log"
HEAVY_LOCK = os.path.expanduser("~/.cache/wsr-review/heavy.lock")
_HEAVY_LOCK_HANDLE = None


def chain_done() -> bool:
    """Return True once the AM/sim chain has finished (Q-15 gate)."""
    log = Path(CHAIN_LOG)
    if not log.is_absolute():
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
        )
        if common.returncode == 0 and common.stdout.strip():
            log = Path(common.stdout.strip()).resolve().parent / CHAIN_LOG
    if not log.is_file():
        return False
    with log.open() as f:
        return any("sim chain done" in line for line in f)


def code_matches_baseline() -> bool:
    """Match the named reference plus only the exact pinned adapter amendment."""
    out = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            BASELINE_COMMIT,
            "--",
            "logic/src",
            "logic/controllers",
            "logic/configs",
            "assets/configs",
            "main.py",
        ],
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "logic/src",
            "logic/controllers",
            "logic/configs",
            "assets/configs",
            "main.py",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode or untracked.returncode or untracked.stdout.strip():
        return False
    if set(out.stdout.splitlines()) != set(BASELINE_AMENDMENTS):
        return False
    return all(
        Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in BASELINE_AMENDMENTS.items()
    )


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
    global _HEAVY_LOCK_HANDLE
    if not chain_done():
        raise SystemExit(f"Refusing heavy run: {CHAIN_LOG!r} shows no 'sim chain done'.")
    if _HEAVY_LOCK_HANDLE is None or _HEAVY_LOCK_HANDLE.closed:
        _HEAVY_LOCK_HANDLE = take_heavy_lock()


def _hydra_value(value: Any) -> str:
    """Render a value for a Hydra dotlist override (no shell involved)."""
    import json

    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return json.dumps(value)


def cell_dotlist(constructor: str, model_path: str = "", time_capped: bool = False) -> List[str]:
    """Build the full Hydra args for one gate-(a) cell (dry, no run).

    One ``sim.policies`` entry carries the per-solver overrides; the
    simulator's ``expand_policy_configs`` merges them over the flattened
    YAML variant, so list-form and mapping-form YAMLs need no per-file
    index addressing here.
    """
    entry = cell_policy_entry(constructor, model_path=model_path, time_capped=time_capped)
    dotlist = [f"{k}={_hydra_value(v)}" for k, v in GATE_A_SIM.items()]
    dotlist.append(f"sim.policies={_hydra_value([entry])}")
    return dotlist


def run_test_sim(dotlist: List[str], run_name: str) -> str:
    """Run one test_sim cell under ``nice``; return the results directory."""
    results_root = os.path.abspath(os.path.join(".agent", "cache", "gates", run_name))
    if os.path.exists(results_root):
        raise SystemExit(f"Refusing to reuse gate results directory: {results_root}")
    cmd = [
        "nice",
        "-n",
        "15",
        sys.executable,
        str(Path(__file__).with_name("capture.py").resolve()),
        "main.py",
        "test_sim",
        f"sim.output_dir={results_root}",
        f"sim.run_name={run_name}",
    ] + dotlist
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"test_sim failed for run {run_name} (exit {proc.returncode}).")
    return results_root


def cmd_overrides(args: argparse.Namespace) -> None:
    """Print the Hydra args for one cell (dry)."""
    for item in cell_dotlist(args.cell, model_path=args.model_path or ""):
        print(item)


def cmd_calibrate(args: argparse.Namespace) -> None:
    """RQ-codex-2: repeats on the pinned reference, determinism proof, frozen file."""
    require_heavy()
    if not code_matches_baseline():
        raise SystemExit(
            f"Refusing calibration: code tree differs from {BASELINE_COMMIT} plus its pinned adapter amendment; "
            "no other baseline source changes are permitted."
        )
    frozen: Dict[str, Any] = {
        "baseline_commit": BASELINE_COMMIT,
        "baseline_amendments": BASELINE_AMENDMENTS,
        "simulation_axes": dict(GATE_A_SIM),
        "cells": {},
    }
    repeats = CALIBRATION_REPEATS if args.repeats is None else args.repeats
    if repeats < 2:
        raise SystemExit("Calibration needs at least two repeats")
    if os.path.exists(args.out):
        raise SystemExit("Frozen tolerances already exist; choose a fresh output path")
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
        if diffs and mode == "certificate_fallback":
            for other in logs[1:]:
                verdict = compare_daily_certificates(logs[0], other)
                if not verdict.passed:
                    raise SystemExit(f"Cell {name}: uncertified calibration repeat: {verdict.detail}")
        timed_logs = []
        for n in range(1, repeats + 1):
            results = run_test_sim(
                cell_dotlist(name, model_path=args.model_path or "", time_capped=True),
                f"gateB-cal-{name.lower()}-r{n}",
            )
            timed_logs.append(_load_policy_log(results, name))
        frozen["cells"][name] = {
            "mode": "certificate" if diffs else "exact",
            "repeat_diffs": len(diffs),
            "time_limit_seconds": 60.0,
            **freeze_tolerances([_log_kpis(log) for log in timed_logs]),
        }
        print(f"cell {name}: mode={frozen['cells'][name]['mode']} diffs={len(diffs)}", flush=True)
    with open(args.out, "w") as f:
        json.dump(frozen, f, indent=2, sort_keys=True)
    print(f"frozen tolerances -> {args.out}")


def cmd_gate_a(args: argparse.Namespace) -> None:
    """Bit-for-bit paired compare of two finished gate-(a) sweeps."""
    failures = 0
    for cell in GATE_A_CELLS:
        name = cell["constructor"]
        try:
            base = _load_policy_log(args.baseline, name)
            cand = _load_policy_log(args.candidate, name)
        except (SystemExit, ValueError) as exc:
            print(f"cell {name}: {exc}")
            failures += 1
            continue
        diffs = compare_exact(base, cand)
        if diffs and cell.get("compare_mode") == "certificate_fallback":
            try:
                verdict = compare_daily_certificates(base, cand)
            except (KeyError, ValueError, TypeError) as exc:
                print(f"cell {name}: certificate evidence missing or invalid: {exc}")
                failures += 1
                continue
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
        frozen = json.load(f)
    if frozen.get("baseline_commit") != BASELINE_COMMIT:
        raise SystemExit("Frozen tolerances do not identify the required baseline")
    if frozen.get("baseline_amendments") != BASELINE_AMENDMENTS:
        raise SystemExit("Frozen tolerances do not identify the required baseline adapter amendment")
    tolerances = frozen["cells"]
    with open(args.archived, newline="") as f:
        archived_reader = csv.DictReader(f)
        archive_fields = set(archived_reader.fieldnames or ())
        archive_rows = list(archived_reader)
    with open(args.candidate, newline="") as f:
        candidate_reader = csv.DictReader(f)
        candidate_fields = set(candidate_reader.fieldnames or ())
        candidate_rows = list(candidate_reader)
    axes = (
        "constructor",
        "city",
        "N",
        "dist",
        "improver",
        "strategy",
        "cf",
        "sl_var",
        "acceptance",
        "days_simulated",
        "seed",
        "sample_id",
    )
    selected_axes = [key for key in axes if key in archive_fields or key in candidate_fields]
    required = set(selected_axes) | {"constructor", "city", "N"}
    if not required <= archive_fields or not required <= candidate_fields:
        raise SystemExit("Archive and candidate must expose the same complete pairing axes")

    def index_rows(rows):
        indexed = {}
        for row in rows:
            key = tuple(row[name] for name in selected_axes)
            if key in indexed:
                raise SystemExit(f"Duplicate comparison cell: {key}")
            indexed[key] = row
        return indexed

    archived = index_rows(archive_rows)
    candidate = index_rows(candidate_rows)
    if not archived or archived.keys() != candidate.keys():
        raise SystemExit("Missing or unmatched required cells; no gate can pass an incomplete sweep")
    failures = 0
    for key, row in candidate.items():
        normalized = row["constructor"].replace("-", "_").upper()
        tolerance_keys = [name for name in tolerances if name.replace("-", "_").upper() == normalized]
        if len(tolerance_keys) != 1:
            raise SystemExit(f"No unique frozen tolerance for {row['constructor']}")
        tol = tolerances[tolerance_keys[0]]
        if tol.get("time_limit_seconds") != 60.0:
            raise SystemExit("Gate-(b) requires tolerances calibrated under time-capped runs")
        if row.get("status", "completed") != "completed" or float(row.get("n_failed", 0)) != 0:
            raise SystemExit(f"Failed candidate samples in {key}")
        verdict = gate_b_verdict(archived[key], row, tol, row["constructor"])
        print(f"row {key}: {'PASS' if verdict.passed else 'REVIEW'}: {verdict.detail}")
        failures += 0 if verdict.passed else 1
    if failures:
        raise SystemExit(f"gate-(b) needs review in {failures} row(s).")
    print("gate-(b) PASSED.")


def _load_policy_log(results_dir: str, constructor: str) -> Dict[str, Any]:
    from tools.gates.cells import get_cell

    token = get_cell(constructor)["policy"].lower()
    matches = [
        path for path in Path(results_dir).rglob("log_*N.json") if f"_{token}_" in path.name.lower().replace("-", "_")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one log for {constructor} in {results_dir}, found {len(matches)}")
    with matches[0].open() as f:
        log = json.load(f)
    if not isinstance(log, dict) or not log.get("samples"):
        raise ValueError(f"Empty or malformed samples for {constructor}")
    _log_kpis(log)  # failed/non-finite samples cannot pass even if both logs match
    return canonical_gate_log(log)


def _log_kpis(log: Dict[str, Any]) -> Dict[str, float]:
    from tools.gates.compare import extract_cell_kpis

    samples = log.get("samples", {})
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Gate requires nonempty per-sample results")
    if any(row.get("status", "completed") != "completed" for row in samples.values()):
        raise ValueError("Failed samples cannot pass a regression gate")
    values = [extract_cell_kpis(row) for row in samples.values()]
    if not all(math.isfinite(value) for pair in values for value in pair):
        raise ValueError("Gate KPIs must be finite")
    return {"kgkm": sum(v[0] for v in values) / len(values), "overflows": sum(v[1] for v in values) / len(values)}


def _single_sample(log: Dict[str, Any]) -> Dict[str, Any]:
    samples = log.get("samples", {})
    if not isinstance(samples, dict) or len(samples) != 1:
        raise ValueError("Certificate fallback needs one explicitly paired sample")
    sample = next(iter(samples.values()))
    if sample.get("status", "completed") != "completed":
        raise ValueError("Failed sample has no acceptable certificate")
    return sample


def _incumbent(log: Dict[str, Any]) -> float:
    return float(_single_sample(log)["profit"])


def _bound(log: Dict[str, Any]) -> float:
    # A missing certificate is NOT a proof that bound equals incumbent.
    return float(_single_sample(log)["bound"])


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
    p_cal.add_argument("--dataset", help="Explicit frozen dataset path for both paired checkouts")
    p_cal.add_argument("--data-dir", help="Read-only simulator data directory shared by both checkouts")
    p_cal.add_argument("--distance-matrix", help="Explicit frozen road distance matrix CSV")
    p_a = sub.add_parser("gate-a", help="Paired bit-for-bit compare of two sweeps.")
    p_a.add_argument("--baseline", required=True)
    p_a.add_argument("--candidate", required=True)
    p_b = sub.add_parser("gate-b", help="Time-capped sweep vs archived outputs.")
    p_b.add_argument("--archived", required=True)
    p_b.add_argument("--candidate", required=True)
    p_b.add_argument("--tolerances", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "dataset", None):
        dataset = Path(args.dataset).resolve()
        if not dataset.is_file():
            raise SystemExit(f"Gate dataset does not exist: {dataset}")
        GATE_A_SIM["sim.graph.load_dataset"] = str(dataset)
    if getattr(args, "data_dir", None):
        GATE_A_SIM["sim.data_dir"] = str(Path(args.data_dir).resolve())
    if getattr(args, "distance_matrix", None):
        matrix = Path(args.distance_matrix).resolve()
        if not matrix.is_file():
            raise SystemExit(f"Gate distance matrix does not exist: {matrix}")
        GATE_A_SIM["sim.graph.dm_filepath"] = str(matrix)
    {"overrides": cmd_overrides, "calibrate": cmd_calibrate, "gate-a": cmd_gate_a, "gate-b": cmd_gate_b}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
