"""CLI parser for the cloud runner (``cloud`` sub-command).

Example::

    python main.py cloud submit --manifest cells.json --max-parallel 4 --wait
    python main.py cloud submit --command "test_sim sim.graph.area=riomaior sim.graph.num_loc=100 ..."
    python main.py cloud status job-1f3a9c2e5b7d --json
    python main.py cloud metrics task-1f3a9c2e5b7d-0000 --json
    python main.py cloud track job-1f3a9c2e5b7d
    python main.py cloud worker

Attributes:
    add_cloud_args: Register the ``cloud`` sub-commands on an existing sub-parser.
"""

from __future__ import annotations

import argparse


def add_cloud_args(parser: argparse.ArgumentParser) -> None:
    """Register the ``cloud`` sub-commands on *parser*.

    Args:
        parser: An :class:`argparse.ArgumentParser` (or sub-parser) to populate.
    """
    parser.add_argument(
        "--root",
        default="outputs/cloud",
        help="Local state directory of the local profile: control plane, objects and task outputs.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Repository root whose main.py local tasks run (default: this checkout).",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    commands = parser.add_subparsers(dest="cloud_command", required=True, help="cloud command")

    submit = commands.add_parser("submit", help="Submit a job: a manifest of cells or one test_sim command")
    source = submit.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Manifest file (native or rerun format, JSON or YAML).")
    source.add_argument("--command", dest="sim_command", help="One test_sim command line, quoted.")
    submit.add_argument("--policy", default=None, help="Keep only this routing algorithm's cells (rerun manifests).")
    submit.add_argument("--name", default=None, help="Job name.")
    submit.add_argument("--provider", default=None, choices=["local", "gcp", "aws", "azure"])
    submit.add_argument("--compute", default=None, choices=["process", "batch", "serverless", "vm"])
    submit.add_argument("--max-parallel", type=int, default=None, help="Tasks of this job running at once.")
    submit.add_argument("--cpus", type=float, default=None)
    submit.add_argument("--memory-gb", type=float, default=None)
    submit.add_argument("--spot", action="store_true", help="Use spot or preemptible capacity.")
    submit.add_argument("--machine-type", default=None, help="Provider machine type (prices and predictions use it).")
    submit.add_argument("--region", default=None, help="Provider region.")
    submit.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Job cost cap in USD: refused above it, cancelled when it is reached.",
    )
    submit.add_argument("--retries", type=int, default=None, help="Automatic retries per task.")
    submit.add_argument("--max-wall", type=float, default=None, help="Wall-time limit per attempt, in seconds.")
    submit.add_argument(
        "--idle", type=float, default=None, help="Kill an attempt after this many seconds without progress."
    )
    submit.add_argument("--gurobi", default=None, choices=["none", "wls_secret", "byo"])
    submit.add_argument(
        "--gurobi-secret", default=None, help="Secret-store name of the Gurobi license (never the value)."
    )
    submit.add_argument("--wait", action="store_true", help="Run the scheduler here until the job finishes.")
    submit.add_argument(
        "--no-worker",
        action="store_true",
        help="Do not start a background worker (by default one starts unless a worker is already running).",
    )
    submit.add_argument("--poll", type=float, default=5.0, help="Seconds between scheduling passes with --wait.")

    status = commands.add_parser("status", help="A job's state, or the recent jobs")
    status.add_argument("job_id", nargs="?", default=None)

    logs = commands.add_parser("logs", help="A task's current log")
    logs.add_argument("task_id")
    logs.add_argument("--tail", type=int, default=None)

    for name, text in (("cancel", "Cancel a job"), ("retry", "Requeue a job's failed and cancelled tasks")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("job_id")

    metrics = commands.add_parser("metrics", help="A task's resource samples, progress and per-day KPIs")
    metrics.add_argument("task_id")
    metrics.add_argument("--since", type=float, default=None, help="Only records after this Unix time.")

    track = commands.add_parser("track", help="Import a job's tasks, metrics and events into the tracking database")
    track.add_argument("job_id")
    track.add_argument("--tracking-uri", default=None, help="Tracking database directory (default: the tracker's).")
    track.add_argument("--experiment", default="cloud", help="Tracking experiment name.")

    predict = commands.add_parser("predict", help="Runtime and memory predictions (CR-11)")
    predict_commands = predict.add_subparsers(dest="predict_command", required=True)
    estimate = predict_commands.add_parser(
        "estimate", help="Predict a manifest's or command's cells without submitting"
    )
    estimate_source = estimate.add_mutually_exclusive_group(required=True)
    estimate_source.add_argument("--manifest", help="Manifest file (native or rerun format, JSON or YAML).")
    estimate_source.add_argument("--command", dest="sim_command", help="One test_sim command line, quoted.")
    estimate.add_argument("--policy", default=None, help="Keep only this routing algorithm's cells (rerun manifests).")
    estimate.add_argument("--machine-type", default=None, help="Machine type the cells would run on.")
    predict_commands.add_parser("evaluate", help="Held-out error of the bundled model")
    accuracy = predict_commands.add_parser("accuracy", help="Predicted against actual wall time of a finished job")
    accuracy.add_argument("job_id")

    cost = commands.add_parser("cost", help="Estimates, accrued cost, prices and billing imports (CR-10)")
    cost_commands = cost.add_subparsers(dest="cost_command", required=True)
    cost_estimate = cost_commands.add_parser(
        "estimate", help="Estimate a manifest's or command's cost without submitting"
    )
    cost_source = cost_estimate.add_mutually_exclusive_group(required=True)
    cost_source.add_argument("--manifest", help="Manifest file (native or rerun format, JSON or YAML).")
    cost_source.add_argument("--command", dest="sim_command", help="One test_sim command line, quoted.")
    cost_estimate.add_argument("--policy", default=None, help="Keep only this routing algorithm's cells.")
    cost_estimate.add_argument("--provider", default=None, choices=["local", "gcp", "aws", "azure"])
    cost_estimate.add_argument("--compute", default=None, choices=["process", "batch", "serverless", "vm"])
    cost_estimate.add_argument("--machine-type", default=None)
    cost_estimate.add_argument("--region", default=None)
    cost_estimate.add_argument("--spot", action="store_true")
    cost_estimate.add_argument("--max-parallel", type=int, default=None, help="Tasks running at once (for the ETA).")
    cost_show = cost_commands.add_parser("show", help="A job's accrued cost per task, the month's total and billing")
    cost_show.add_argument("job_id")
    cost_prices = cost_commands.add_parser("prices", help="Hourly price of a provider machine type")
    cost_prices.add_argument("provider", choices=["gcp", "aws", "azure"])
    cost_prices.add_argument("machine_type")
    cost_prices.add_argument("--region", default=None)
    cost_billing = cost_commands.add_parser("import-billing", help="Record a provider billing export's per-job totals")
    cost_billing.add_argument("path", help="Billing export CSV with a wsr_job_id label column.")
    cost_billing.add_argument("--provider", required=True, choices=["gcp", "aws", "azure"])

    download = commands.add_parser("download", help="Download a job's results")
    download.add_argument("job_id")
    download.add_argument("--dest", default=None, help="Destination root (default: <root>/results/<job_id>).")

    commands.add_parser("backends", help="List the configured backends")

    commands.add_parser("credentials", help="Which providers have a usable CLI login or keychain credential")
    commands.add_parser("profiles", help="List the profiles in the cloud profiles file")
    secrets = commands.add_parser("secrets", help="Store or remove a secret in the OS keychain (never printed)")
    secrets.add_argument("action", choices=["set", "delete", "exists"])
    secrets.add_argument("name", help="Secret name (for example gurobi-wls), or provider/profile for a credential.")
    secrets.add_argument("--from-file", default=None, help="Read the value from this file (set).")
    secrets.add_argument("--stdin", action="store_true", help="Read the value from standard input (set).")

    worker = commands.add_parser("worker", help="Run the scheduler until every job finishes")
    worker.add_argument("--job", default=None, help="Only wait for this job.")
    worker.add_argument("--poll", type=float, default=5.0)
    worker.add_argument("--once", action="store_true", help="Run one scheduling pass and exit.")
