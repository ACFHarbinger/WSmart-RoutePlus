> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## E — New Features

### §E.1 — Multi-Problem Benchmarking Suite

**Pain**: Comparing neural models (AM, TAM, DDAM, MoE) against classical policies (ALNS, HGS, Gurobi) across all three problem types (VRPP, WCVRP, SCWCVRP) and multiple graph sizes requires manually running multiple `main.py eval` commands and aggregating CSV results by hand.

**Options**

- **A** — Add a `benchmark` subcommand to `main.py` that: runs a configurable matrix of (policy × problem × graph_size), collects metrics, and writes a unified `benchmark_report.csv` and Markdown table. `[Quick Win]`
- **B** — Integrate with `ray[tune]` sweep (already a dependency) to parallelize the benchmark matrix across CPU cores.
- **C** — Add a "Benchmark" tab to the Studio (synergises with §A.5) that configures the matrix via checkboxes and shows a live results table.

**Recommendation**: **Option A** for the CLI benchmark runner, **Option C** for Studio-accessible results.

**Effort × Impact**: Medium effort / High impact

---

### §E.2 — TSPLIB / Solomon Benchmark Instance Support

**Pain**: The framework generates synthetic instances internally but cannot load standard benchmark instances (TSPLIB95, Solomon C/R/RC, Christofides). This makes it impossible to compare against published results.

**Options**

- **A** — Add a `data/loaders/tsplib_loader.py` that parses `.vrp` / `.tsp` files (standard TSPLIB format) into the framework's `TensorDict` input format.
- **B** — Use the `tsplib95` Python library (pure-Python, no C dependency) as a parser backend; wrap its output in the framework's schema.
- **C** — Add a `gen_data` subcommand option `--source tsplib --instance pr76` that downloads and converts instances from the TSPLIB repository.

**Recommendation**: **Option B** is the fastest path (the library handles all edge cases in the format spec); **Option C** makes it accessible from the CLI in one command.

**Effort × Impact**: Medium effort / Very High impact `[Research]`

---

### §E.3 — REST API for Remote Inference

**Pain**: The framework has no HTTP interface. Integrating the routing engine into a larger fleet management system requires either subprocess calls or direct Python imports, both of which are fragile.

**Options**

- **A** — Add a `main.py serve` subcommand using `FastAPI` that exposes: `POST /solve` (accepts a problem instance JSON, returns a solution), `GET /health`, and `GET /models` (lists available weights). `[Research]`
- **B** — Use `Flask` for a simpler synchronous server with lower dependency overhead.
- **C** — Implement a `gRPC` interface for higher-throughput production use cases.
- **D** — Wrap in a Docker container with a `docker-compose.yml` for deployment.

**Recommendation**: **Option A** — FastAPI is the modern standard, its async design fits the non-blocking inference pattern, and it auto-generates OpenAPI docs. **Option D** is the natural packaging step after.

**Effort × Impact**: High effort / High impact

---

### §E.4 — Online Learning / Warm-Starting

**Pain**: In multi-day waste collection simulation, each new day presents a slightly different distribution of bin fill levels. The current pipeline re-runs inference from a static checkpoint with no adaptation mechanism.

**Options**

- **A** — Add a `warm_start` mode to the training pipeline: initialize from an existing checkpoint and fine-tune for N epochs on the current day's distribution before evaluating. `[Research]`
- **B** — Implement the `MetaRNN` (already exists in `logic/src/models/meta/`) as the online adapter: on each day, perform one or more gradient steps using the day's context as the meta-input.
- **C** — Use the contextual bandit module (already in `logic/src/pipeline/rl/meta/`) to select among a portfolio of pre-trained policies based on day context, without gradient updates.
- **D** — Implement reservoir sampling of "hard" instances encountered during simulation and periodically fine-tune on them.

**Recommendation**: **Option C** is the lowest-risk path (no gradient updates in production, just policy selection) and leverages existing code. **Option B** is the research-grade approach that the MetaRNN architecture was designed for.

**Effort × Impact**: High effort / Very High impact `[Research]`

---

### §E.5 — Real-World Data Integration (Smart Bin Sensors)

**Pain**: The WCVRP and SCWCVRP environments model bin fill rates stochastically, but there is no pipeline to ingest real sensor data (IoT fill-level readings) and use it to calibrate the stochastic parameters.

**Options**

- **A** — Add a `data/loaders/sensor_loader.py` that reads the CSV format defined in `CLAUDE.md §12.3` and converts it to the framework's bin fill tensor format; expose it via `gen_data --source sensor --file bins.csv`.
- **B** — Add a `calibration` subcommand that fits the stochastic fill-rate distribution parameters (mean, variance per bin) to historical sensor data using MLE.
- **C** — Integrate with MQTT/HTTP sensor APIs for live streaming fill-level updates during simulation.

**Recommendation**: **Option A + B** as a research pipeline; **Option C** only for production deployments where live sensor APIs are available.

**Effort × Impact**: Medium effort (Options A/B) / Very High impact

---

### §E.6 — LLM-Assisted Problem Instance Generation

**Pain**: Research teams need diverse problem instances to test policy robustness. Hand-crafting instance parameters (node clustering, demand distributions, time windows) is labour-intensive.

**Options**

- **A** — Add a `gen_data --mode llm_assisted` command that uses an LLM API to generate natural-language scenario descriptions, translates them to parameter overrides, and creates instances. `[Research]`
- **B** — Use a simple constraint-satisfaction generator with richer parameter coverage (clustered vs. random vs. mixed depot placement, heterogeneous demand distributions) without LLM involvement.
- **C** — Train a conditional generator (VAE or diffusion model) on existing instance distributions to sample novel but realistic instances. `[Research]`

**Recommendation**: **Option B** is the pragmatic choice — richer parameterization of the existing generator provides immediate value without LLM API costs. **Option C** is a research-grade addition for distribution-shift robustness studies.

**Effort × Impact**: Low effort (Option B) / High impact

---

### §E.7 — Cross-Environment Generalization (Zero-Shot Transfer)

**Pain**: Models trained on VRPP do not generalize to WCVRP without retraining. There is no evaluation protocol measuring zero-shot or few-shot transfer across problem types.

**Options**

- **A** — Add a `transfer_eval` subcommand that loads a checkpoint trained on problem A and evaluates it on problem B; logs a transfer performance gap metric. `[Research]`
- **B** — Adapt the `MetaRNN` / hypernet architecture to condition on problem-type embeddings, enabling a single model to handle multiple VRP variants. `[Research]`
- **C** — Use curriculum learning: train sequentially on VRPP → WCVRP → SCWCVRP with increasing difficulty; measure generalization at each stage.

**Recommendation**: **Option A** first (pure evaluation, no training changes), to establish the baseline gap. **Option B/C** follow once the gap magnitude is known.

**Effort × Impact**: Low effort (Option A) / High impact `[Research]`

---

### Effort × Impact Matrix — New Features

| Item                                               | Effort    | Impact    | Priority        |
| -------------------------------------------------- | --------- | --------- | --------------- |
| §E.6 Option B (richer instance generator)          | Low       | High      | P0              |
| §E.1 Option A (CLI benchmark runner)               | Medium    | High      | P1              |
| §E.5 Option A (sensor data loader)                 | Medium    | Very High | P1              |
| §E.7 Option A (transfer eval command)              | Low       | High      | P1 `[Research]` |
| §E.2 Option B+C (TSPLIB loader)                    | Medium    | Very High | P2              |
| §E.5 Option B (fill-rate calibration)              | Medium    | Very High | P2              |
| §E.4 Option C (contextual bandit policy selection) | Medium    | Very High | P2 `[Research]` |
| §E.3 Option A (FastAPI server)                     | High      | High      | P3              |
| §E.4 Option B (MetaRNN online adaptation)          | Very High | Very High | P3 `[Research]` |
| §E.6 Option C (conditional generator)              | Very High | High      | P3 `[Research]` |

---

