# Cloud Runner Guide

**Module**: `logic/controllers/cloud`
**Purpose**: Run simulation jobs locally or on GCP, AWS and Azure, and follow them from the CLI and the Studio
**Version**: 1.1.0
**Last Updated**: October 2026

> **Status.** Every provider backend, store and control plane is tested against in-process
> fakes only. Nothing in v1.1.0 has been run against a real cloud API yet; the first real
> run per provider is a one-day Rio Maior 100 cell behind a small budget cap (see the
> [cloud roadmap](moon/roadmaps/cloud.md)).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Profiles](#3-profiles)
4. [Credentials and Secrets](#4-credentials-and-secrets)
5. [Launching Jobs](#5-launching-jobs)
6. [Following Jobs](#6-following-jobs)
7. [Placement, Priorities and the Warm Pool](#7-placement-priorities-and-the-warm-pool)
8. [Costs and Limits](#8-costs-and-limits)
9. [Runtime Prediction](#9-runtime-prediction)
10. [The Studio](#10-the-studio)
11. [Testing Without a Cloud](#11-testing-without-a-cloud)

---

## 1. Overview

A **job** is one `test_sim` command or a manifest of cells; each cell becomes a **task**.
A scheduler (`cloud worker`) places tasks on backends, polls them, retries failures and
preemptions (resuming from the last finished day), and enforces wall-time, idle and budget
limits. Tasks run the release image's task runtime, which stages inputs from object
storage, writes the Gurobi license file, mirrors outputs back, and streams telemetry.

| Layer | Local | Cloud |
|---|---|---|
| Control plane (queue, state, events, metrics) | SQLite (`sqlite://`) | Firestore (`firestore://<project>`), Supabase (`supabase://<ref>`) |
| Object store (inputs, outputs, logs) | Filesystem (`file://`) | GCS (`gs://`), S3 (`s3://`), Azure Blob (`az://`) |
| Compute | local processes | GCP Batch / Cloud Run Jobs / Compute Engine; AWS Batch / ECS Fargate / EC2; Azure Batch / Container Apps Jobs / VMs |

Every command is available as `python main.py cloud ...` and, faster (no torch or Hydra
start-up), as `python -m logic.controllers.cloud ...`.

## 2. Installation

The core install runs the local profile. The provider SDKs are the optional `[cloud]` extra:

```bash
uv sync --extra cloud            # or: pip install "wsmart-route[cloud]"
```

Without the extra every cloud module still imports (the SDKs are loaded on first use), so
the local profile, estimates and the CLI work everywhere.

## 3. Profiles

A profile says where jobs run. Profiles live in `$WSR_CLOUD_CONFIG` or
`~/.config/wsmart-route/cloud.yaml`, and hold no secrets:

```yaml
profiles:
  gcp-batch:
    provider: gcp                 # gcp | aws | azure
    compute: batch                # default compute kind: batch | serverless | vm
    region: europe-west1
    credentials: cli              # cli (gcloud/aws/az login) | keychain
    keychain_profile: default     # keychain account <provider>/<keychain_profile>
    control_plane: firestore://my-project
    store: gs://my-bucket/wsmart-route
    image: europe-west1-docker.pkg.dev/my-project/wsr/wsmart-route:1.1.0
    gurobi_secret: gurobi-wls     # optional: secret-store name of the Gurobi license
    backend_options:              # optional, non-secret provider settings
      project: my-project
```

`cloud profiles` lists them. `--profile NAME` on any command uses that profile's control
plane, store and provider backends (all three compute kinds, plus local processes);
without it, the **local profile** under `--root` (default `outputs/cloud`) is used.
`backend_options` keys are passed to the backends that take them, for example `project`
(GCP), `job_queue` or `cluster` (AWS), `resource_group` or `subscription_id` (Azure); keys
that look like secrets are rejected.

## 4. Credentials and Secrets

- **CLI logins** (`credentials: cli`): the provider's own login chain (`gcloud auth
  application-default login`, `aws sso login`, `az login`). `cloud credentials` shows which
  providers have a usable login.
- **Keychain** (`credentials: keychain`): a service-account key in the OS keychain, stored
  with `cloud secrets set gcp/default --from-file key.json` (values are never taken as
  command arguments and never printed).
- **Gurobi**: `--gurobi wls_secret --gurobi-secret gurobi-wls` references a license stored in
  the provider's secret manager (Secret Manager, Secrets Manager, Key Vault); `byo` uses one
  you supply. Batch and serverless backends inject it natively; GCP VMs get a reference
  (`WSR_SECRET_REFS`) that the runtime resolves with the machine's identity. The license
  value never passes through the control plane, the logs or the instance metadata.

## 5. Launching Jobs

```bash
# One command, locally, waiting for it
python -m logic.controllers.cloud submit --command "test_sim sim.graph.area=riomaior sim.graph.num_loc=100 sim.graph.n_days=30 'sim.policies=[alns]'" --wait

# A rerun manifest's ALNS cells on GCP Batch spot capacity, eight at a time
python -m logic.controllers.cloud --profile gcp-batch submit --manifest cells.json --policy alns --spot --max-parallel 8
```

Common options:

| Option | Meaning |
|---|---|
| `--provider`, `--compute` | Where to run (defaults: the profile's, else local processes) |
| `--machine-type`, `--region`, `--cpus`, `--memory-gb`, `--spot` | Resources per task |
| `--retries`, `--max-wall`, `--idle` | Automatic retries, wall-time limit and idle kill per attempt |
| `--budget USD` | Job cost cap (see [Costs and Limits](#8-costs-and-limits)) |
| `--priority N` | Higher runs first when jobs compete |
| `--target ...` (repeatable) | Placement targets with spill-over (see [Placement](#7-placement-priorities-and-the-warm-pool)) |
| `--pool` | Run on the warm worker pool |
| `--gurobi`, `--gurobi-secret` | License mode and secret name |
| `--wait` / `--no-worker` | Schedule in the foreground / do not start a background worker |

Submit starts a background `cloud worker` (one per state directory) unless `--wait` or
`--no-worker` is given; `cloud worker` can also be run by hand.

## 6. Following Jobs

| Command | Shows |
|---|---|
| `cloud status [JOB]` | Job state and tasks (attempt, duration, cost, CPU, memory, day) |
| `cloud metrics TASK` | Resource samples, peaks, current day and per-day KPIs |
| `cloud logs TASK [--tail N]` | The attempt's log |
| `cloud cost show JOB` | Accrued cost per task, the job and the month; the estimate; billing |
| `cloud predict accuracy JOB` | Predicted against actual wall time |
| `cloud track JOB` | Imports the job into the tracking database (`assets/tracking`) |
| `cloud cancel JOB`, `cloud retry JOB` | Stop a job / requeue its failed and cancelled tasks |
| `cloud download JOB [--dest DIR]` | Fetch the results, with hashes |

Add `--json` for machine-readable output.

## 7. Placement, Priorities and the Warm Pool

**Placement.** A job may list several targets, `provider/compute[/region[/machine_type[/spot]]][:max_tasks]`:

```bash
submit --manifest cells.json --target gcp/batch/europe-west1/e2-standard-4/spot:20 --target aws/batch/eu-west-1/c6i.xlarge:10
```

Each start ranks the eligible targets by expected cost (predicted time x price; spot is
penalised by its recent preemption rate), then by queue depth, then by list order. A full
or failing target spills the task over to the next one; three consecutive failures take a
target out for five minutes. `max_tasks` is a quota across all jobs.

**Priorities.** `--priority` orders jobs competing for parallelism and quota.

**Warm pool.** `--pool` jobs are run by long-lived pool workers instead of one launch per
task. The coordinator keeps enough workers alive for the queue, packing tasks onto worker
machines by their predicted memory; workers exit after being idle (scale to zero). The pool
is configured in `<root>/pool.json`:

```json
{"target": "gcp/vm/europe-west1/e2-standard-8", "cpus": 8, "memory_gb": 32, "max_workers": 4, "idle_seconds": 300, "image": "europe-west1-docker.pkg.dev/my-project/wsr/wsmart-route:1.1.0"}
```

## 8. Costs and Limits

All amounts are **list-price estimates** in USD, not invoices.

- **Before submitting:** `cloud cost estimate --manifest cells.json --provider gcp --compute batch --machine-type e2-standard-4 --max-parallel 8`
  prints the cost range (from the p10/p50/p90 runtime predictions) and the ETA.
- **Job cap:** `--budget USD`. A job whose p50 estimate exceeds it is refused; a running job
  is cancelled when its accrued cost reaches it, or when its remaining budget can no longer
  pay for its next task.
- **Monthly cap and warnings:** `<root>/budget.json`, for example
  `{"monthly_usd": 200, "warn_fractions": [0.5, 0.8, 0.95]}`. Queued work counts against the
  month at its estimate; warnings are recorded as `budget_warning` events.
- **Prices:** `cloud cost prices gcp e2-standard-4`. The providers' static tables can be
  overridden in `<root>/prices.json`:
  `{"gcp": {"*": {"e2-standard-4": [0.134, 0.040]}}}` (on-demand, spot per hour).
- **Billing exports:** `cloud cost import-billing export.csv --provider gcp` records the
  invoiced total per job (resources carry a `wsr_job_id` label).

## 9. Runtime Prediction

Each cell's wall time (p10, p50, p90) and memory are predicted from the paper's 810 runs
(quantile gradient boosting; held-out median error 20 %), corrected per machine type by the
jobs that ran on it. Submit fills in `expected_wall_seconds`, which the scheduler and the
Analysis page use.

```bash
cloud predict estimate --manifest cells.json --machine-type e2-standard-4
cloud predict evaluate
```

## 10. The Studio

The Studio's cloud pages call the same CLI:

- **Launcher** builds a submit request, shows its CLI equivalent, and estimates cost and ETA
  as the form changes (machine type, region and budget fields included).
- **Monitor** lists jobs and tasks, tails logs, cancels and retries, and plots a task's CPU
  and memory and its current day.
- **Analysis** shows durations, accrued and estimated cost, spot savings and prediction
  accuracy per job.

## 11. Testing Without a Cloud

`logic.controllers.cloud.testing` holds the contract suites (`backend_contract`,
`object_store_contract`, `secret_store_contract`, `control_plane_contract`), a scripted
backend, and `contract_launch(outcome)`; every provider passes them against in-process
fakes:

```bash
uv run pytest logic/test/unit/controllers/cloud
```

CI runs these suites with and without the `[cloud]` extra, and builds and smoke-tests the
release tree.
