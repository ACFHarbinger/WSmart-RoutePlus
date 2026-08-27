> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## A — Analytics & Interpretability

### §A.1 — Interactive Route Solution Visualizer

**Pain**: Solutions (tours, collected bins, costs) are currently logged as JSON arrays. There is no visual overlay of routes on a spatial canvas, making debugging decoder outputs and comparing policies against each other extremely tedious.

**Options**

- **A** — Add an ECharts panel inside the Studio analysis view: render depot, nodes, edges, and colour-code routes per vehicle using an ECharts `custom` series or a lightweight 2D canvas renderer. Low friction, consistent with the rest of the Studio tech stack.
- **B** — Export solutions to GeoJSON and open them in a browser via Folium/Leaflet, decoupled from the application.
- **C** — Use Plotly Dash as a standalone web dashboard, running in a background process launched from `main.py`.
- **D** — Integrate `rerun.io` for a time-scrubbing 2-D trajectory viewer (works well with simulation day-by-day replay).
- **E** — Use the deck.gl `PathLayer` + `ScatterplotLayer` inside the Studio's geospatial view (§G Phase 3) — the same WebGL renderer used for geospatial routing, repurposed for abstract Cartesian coordinates with the OrbitView camera.

**Recommendation**: **Option A first** (lowest cost, consistent with Studio stack), then **Option E** as the production-quality widget — deck.gl scales better and integrates with the Studio's geospatial phase. Option D is interesting for multi-day simulation replay but requires an external runtime.

**Effort × Impact**: Medium effort / High impact

**Delivered (§A.1 Option A — hundred-fourteenth pass)**

- [x] ``RouteViz`` React component — shared ECharts spatial panel with star depot, demand-sized tour nodes, per-vehicle coloured edges, and optional failure overlay (overflow / skipped high-fill highlights)
- [x] ``routeViz.ts`` — ``buildRouteVizOption`` utility; reuses ``resolveBinPositions`` + ``splitVehicleTourIndices``
- [x] Simulation Monitor — refactored inline ``RouteMapChart`` to ``RouteViz`` (ECharts mode)
- [x] Simulation Summary — day scrubber + multi-policy route comparison grid in analysis view
- [x] PNG/SVG export via ``ChartExportButtons`` (§G.7)

**Status**: §A.1 Option A complete — Option E (deck.gl PathLayer) already delivered via ``DeckRouteMap`` (§G.3 / §G.16); Options B/C/D deferred.

---

### §A.2 — Attention Map Visualization for Neural Decoders

**Pain**: The AM, TAM, DDAM, and MoE decoders compute multi-head attention over node embeddings, but these attention weights are never exported or displayed. Without visibility into what the model attends to, diagnosing routing errors or comparing trained heads is guesswork.

**Options**

- **A** — Hook `nn.MultiheadAttention` outputs with forward hooks; buffer the last batch's attention tensors in a ring-buffer on the model object. Visualize as a heatmap in the Studio ML introspection phase (§G Phase 5).
- **B** — Integrate `BertViz`-style row-column attention visualizer adapted for graph problems (node × node matrix).
- **C** — Log attention weights to WandB / TensorBoard as image summaries during evaluation; no GUI integration needed.
- **D** — Export attention weights to `.npz` per inference call and build a separate offline viewer script.

**Recommendation**: **Option C** for fast iteration (zero GUI work), then **Option A** for the Studio integration once Option C has validated that the data is interpretable. Option B is academic-grade but requires a browser runtime.

**Effort × Impact**: Low effort (Option C) → Medium effort (Option A) / High impact

**Delivered (§A.2 Option C — hundred-thirteenth pass)**

- [x] ``logic/src/tracking/logging/visualization/heatmaps.py`` — runtime attention capture via ``add_attention_hooks``, PNG rendering, WandB ``wandb.Image`` + TensorBoard ``add_image`` logging
- [x] ``AttentionHeatmapCallback`` — validation-epoch hook; respects ``tracking.log_attention``, ``tracking.log_attention_heatmaps``, and ``viz_every_n_epochs``
- [x] ``WSTrainer`` — auto-registers callback when tracking flags enabled
- [x] Eval engine — ``maybe_log_eval_attention_heatmaps()`` after ``evaluate_policy`` when ``tracking.log_attention*`` set
- [x] Unit tests in ``logic/test/unit/tracking/test_attention_heatmaps.py``

**Delivered (§A.2 Option A — hundred-seventeenth pass)**

- [x] ``AttentionRingBuffer`` — fixed-capacity ring-buffer for encoder attention snapshots (layer, head, decode step, normalised matrix)
- [x] ``install_attention_ring_buffer`` / ``ensure_attention_buffer`` — persistent forward hooks on encoder MHA layers
- [x] ``attention_emit.py`` — ``ATTENTION_VIZ_START:`` stdout + ``attention_viz.jsonl`` append when ``tracking.log_attention`` enabled
- [x] ``maybe_log_eval_attention_heatmaps`` — integrates ring-buffer capture + Studio emission after eval/validation
- [x] Rust ``parse_attention_viz_line`` + ``load_attention_viz_log`` command
- [x] Studio ``RuntimeAttentionPanel`` — ECharts heatmap with snapshot/layer/head selectors on Training Monitor + ML Introspection Attention tab
- [x] Unit tests in ``logic/test/unit/tracking/test_attention_buffer.py``

**Delivered (§A.2 Option A — hundred-thirtieth pass)**

- [x] ``collectAttentionVizFromLogLines`` — shared ``ATTENTION_VIZ_START:`` parser for process stdout
- [x] Training Hub — ``RuntimeAttentionPanel`` during live train/hpo runs; stdout ingest alongside metrics (§G.10 / §A.2)
- [x] Process Monitor — ``RuntimeAttentionPanel`` for selected ``train_`` / ``hpo_`` processes (§G.15 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-first pass)**

- [x] ``findActiveLiveTrainProcessId`` / ``findActiveHpoProcessId`` — shared train/HPO process detection for live analytics
- [x] Training Monitor — live stdout ingest for ``hpo_*`` processes; ``Live HPO`` label when HPO active (§G.17 / §A.2)
- [x] HPO Tracker — ``RuntimeAttentionPanel`` during live ``hpo_*`` runs; ``Process Monitor →`` navigation shortcut (§G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-second pass)**

- [x] Experiment Tracker — ``RuntimeAttentionPanel`` during live ``hpo_*`` runs; ``HPO Tracker →`` + ``Process Monitor →`` shortcuts (§G.18 / §A.2)
- [x] Training Monitor / Process Monitor / HPO Tracker — cross-page navigation shortcuts for live train/HPO workflows (§G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-third pass)**

- [x] Experiment Tracker — ``Training Monitor →`` shortcut during live ``hpo_*`` runs (§G.18 / §A.2)
- [x] Training Monitor / HPO Tracker / Process Monitor / Training Hub — ``Experiment Tracker →`` shortcut when live HPO active (§G.10 / §G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-fourth pass)**

- [x] Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker — ``Training Hub →`` shortcut during live train/HPO workflows (§G.10 / §G.15 / §G.17 / §G.18 / §A.2)

**Delivered (§A.2 Option A — hundred-thirty-fifth pass)**

- [x] ``TrainHpoNavMesh`` — shared cross-page train/HPO navigation component; replaces duplicated shortcut buttons on Training Hub, Training Monitor, Process Monitor, HPO Tracker, and Experiment Tracker (§G.7 / §A.2 / §A.4)

**Status**: §A.2 Options A+C complete — Option B (BertViz) deferred.

---

### §A.3 — Policy Telemetry Dashboard (Extension of `PolicyVizMixin`)

**Pain**: `logic/src/tracking/viz_mixin.py` already records per-iteration metrics (cost, feasibility, elapsed time) into a fixed-capacity ring-buffer via `_viz_record()`. However, this data is only accessible programmatically through `get_viz_data()` and is never surfaced to the user during or after a run.

**Options**

- **A** — Wire `get_viz_data()` output into the Studio's analytics view: after a simulation run, populate an ECharts bar chart with per-policy metrics (cost trajectories, improvement curves). Synergises with §G Phase 1. `[Quick Win]`
- **B** — Emit ring-buffer snapshots over a WebSocket / Tauri event channel to a React panel refreshed at 2 Hz while the simulation runs. Synergises with §G Phase 15 (Real-Time Process Monitor).
- **C** — Persist ring-buffer dumps to a SQLite database (`assets/telemetry.db`) and query them across runs for cross-policy trending.
- **D** — Push telemetry to Prometheus and visualize in Grafana (overkill for single-machine runs).

**Recommendation**: **Option A** immediately (hours of work), **Option C** for multi-run analytics once the database schema is stable.

**Effort × Impact**: Very Low effort (Option A) / High impact

**Delivered (§A.3 Option A — hundred-ninth pass)**

- [x] ``POLICY_VIZ_START:`` stdout + JSONL log marker from ``policy_viz_emit.py`` after route construction / improvement when ``PolicyVizMixin.get_viz_data()`` is non-empty
- [x] Rust ``parse_policy_viz_line`` + ``load_policy_viz_log`` + ``sim:policy_viz_update`` watcher events
- [x] Studio ``PolicyTelemetryPanel`` — ECharts cost trajectories, operator histograms, and algorithm-specific charts (ALNS/HGS/ACO/ILS/selector/generic) on Simulation Monitor
- [x] Live ingest via ``process:stdout`` parser + historical load on log open; PNG/SVG export via ``ChartExportButtons`` (§G.7)

**Delivered (§A.3 Option B — hundred-nineteenth pass)**

- [x] ``PolicyVizStreamSession`` — daemon thread emits growing ring-buffer snapshots every 0.5 s (2 Hz) during route construction / improvement
- [x] Route actions wrap ``adapter.execute`` and ``processor.process`` in stream sessions; final snapshot on context exit
- [x] Studio sim store upserts policy-viz entries by policy/sample/day/type (replaces stale snapshots during live runs)
- [x] ``PolicyTelemetryPanel`` — 2 Hz throttled ECharts refresh + **Live · 2 Hz** badge when file-watcher or ``test_sim`` process is active
- [x] Live ingest via ``process:stdout`` (§G.15) + ``sim:policy_viz_update`` file-watcher events
- [x] Unit tests in ``logic/test/unit/tracking/test_policy_viz_emit.py``

**Delivered (§A.3 Option C — hundred-twentieth pass)**

- [x] ``policy_telemetry_db.py`` — SQLite store at ``assets/telemetry.db`` with ``simulation_runs`` + ``policy_viz_snapshots`` tables
- [x] ``persist_policy_viz_snapshot`` — upserts terminal ring-buffer per run × policy × sample × day on each ``POLICY_VIZ_START:`` emit
- [x] ``query_policy_telemetry_trends`` — cross-run rows with ``final_metric``, ``step_count``, and algorithm family filter
- [x] Rust ``load_policy_telemetry_trends`` command (Python subprocess bridge)
- [x] Studio ``PolicyTelemetryTrendsPanel`` — cross-run comparison bar chart, steps chart, and history table on Simulation Monitor
- [x] Unit tests in ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Delivered (§A.3 Option C — hundred-twenty-first pass)**

- [x] ``query_policy_trajectory_series`` — extracts improvement curves (``best_cost`` / ``global_best_cost`` / etc.) from persisted ``data_json`` ring-buffers
- [x] Rust ``load_policy_trajectory_trends`` command — Python subprocess bridge for trajectory payloads
- [x] Studio ``PolicyTelemetryTrendsPanel`` — cross-run improvement trajectory line chart with policy filter + optional EMA smoothing; PNG export via ``ChartExportButtons`` (§G.7)
- [x] Unit tests for trajectory query roundtrip and policy-type filtering

**Delivered (§A.3 Option C — hundred-twenty-second pass)**

- [x] ``buildTrendTrajectoryOption`` — trajectory x-axis uses unioned solver step indices (iteration / generation) from persisted ring-buffers instead of array index
- [x] ``PolicyTelemetryTrendsPanel`` — history table CSV export via ``exportPolicyTelemetryTrendsCsv``; row click brushes global policy / ``run_label`` filter (§G.6 / §G.7)
- [x] Simulation Monitor — passes ``initialPolicy`` to pre-filter trajectory dropdown from active policy selection
- [x] Benchmark Analysis — ``PolicyTelemetryTrendsPanel`` for portfolio cross-run solver telemetry (§G.1 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-third pass)**

- [x] ``filterTrendRows`` / ``filterTrajectorySeries`` — global policy / ``run_label`` brush filters comparison, steps, and trajectory chart data (§G.6 / §G.7)
- [x] ``PolicyTelemetryTrendsPanel`` — chart click brushes global policy / run; active-brush badge + clear control; trajectory CSV via ``exportPolicyTrajectoryCsv``
- [x] ``query_policy_trajectory_series`` — includes ``run_label`` on each trajectory payload for run-key brush parity
- [x] Simulation Summary — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from active chart brush (§G.1 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-fourth pass)**

- [x] ``buildTrendComparisonOption`` / ``buildTrendStepsOption`` / ``buildTrendTrajectoryOption`` — brush dimming via ``TrendBrushFilter`` + ``chartHighlight`` opacity (non-selected series stay visible at 25%)
- [x] ``PolicyTelemetryTrendsPanel`` — history table uses ``filteredRows``; empty-state when brush excludes all rows; charts dim from full dataset (not hard-filtered)
- [x] Algorithm Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from global brush (§G.1 / §A.3)
- [x] City Comparison — ``PolicyTelemetryTrendsPanel`` with ``initialPolicy`` from global brush (§G.1.6 / §A.3)
- [x] Benchmark Analysis — ``initialPolicy`` brush sync on ``PolicyTelemetryTrendsPanel`` (parity with Simulation Summary)

**Delivered (§A.3 Option C — hundred-twenty-fifth pass)**

- [x] ``query_policy_telemetry_trends`` / ``query_policy_trajectory_series`` — optional ``run_label`` SQL filter for server-side portfolio scoping
- [x] Rust ``load_policy_telemetry_trends`` / ``load_policy_trajectory_trends`` — ``run_label`` bridge arg; panel passes active global brush to Python queries
- [x] ``PolicyTelemetryTrendsPanel`` — ``initialRunLabel`` prop syncs global run brush; steps chart click indexes ``displayStepRows`` (fixes brush click parity)
- [x] Simulation Summary / Benchmark Analysis / City Comparison / Algorithm Comparison — ``initialRunLabel`` from portfolio single-run brush (§G.1 / §G.6)
- [x] OLAP Explorer — ``PolicyTelemetryTrendsPanel`` with policy + run_label brush sync (§G.6 / §A.3)
- [x] Unit tests for ``run_label`` filter roundtrip in ``logic/test/unit/tracking/test_policy_telemetry_db.py``

**Delivered (§A.3 Option C — hundred-twenty-sixth pass)**

- [x] ``runLabelFromPath`` — shared ``Path.stem`` helper for SQLite ``run_label`` keys from log paths
- [x] ``PolicyTelemetryTrendsPanel`` — trajectory chart click indexes ``allSeries`` (fixes brush click when chart shows dimmed full dataset)
- [x] Simulation Monitor — ``initialRunLabel`` from active log path stem; cross-run trends scoped to open simulation (§G.15 / §A.3)
- [x] Data Explorer — ``PolicyTelemetryTrendsPanel`` with policy + run_label brush sync (§G.16 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-seventh pass)**

- [x] Output Browser — ``PolicyTelemetryTrendsPanel`` scoped to selected run via ``runJsonlPath`` stem + global brush sync (§G.14 / §A.3)
- [x] Output Browser — KPI summary policy rows click-to-brush global policy filter (parity with trends panel table)

**Delivered (§A.3 Option C — hundred-twenty-eighth pass)**

- [x] Output Browser — auto ``run_label`` brush on run select via ``setRunLabel`` + run list ring highlight when brush active (§G.14 / §A.3)
- [x] ``extractJsonlPathFromLogLines`` — scan process stdout for ``.jsonl`` paths to derive SQLite ``run_label`` keys
- [x] ``collectPolicyVizFromLogLines`` / ``uniquePolicyVizPolicies`` — parse per-process ``POLICY_VIZ_START:`` markers from stdout
- [x] Process Monitor — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` for selected ``test_sim`` processes; policy chip brush + live 2 Hz refresh (§G.15 / §A.3)

**Delivered (§A.3 Option C — hundred-twenty-ninth pass)**

- [x] ``runLabelFromLogLines`` — shared ``run_label`` derivation from process stdout + fallback id
- [x] Process Monitor — always sync ``run_label`` brush on ``test_sim`` select; process row ring highlight when global brush matches (§G.15 / §A.3)
- [x] Simulation Launcher — ``PolicyTelemetryPanel`` + ``PolicyTelemetryTrendsPanel`` during live runs; policy chip + KPI card brush + ``run_label`` auto-sync (§G.9 / §A.3)

**Status**: §A.3 Options A+B+C complete.

---

### §A.4 — RL Loss Landscape & Training Health Monitoring

**Pain**: The Lightning-based RL pipeline (`logic/src/pipeline/rl/`) logs loss values but provides no automated detection of training instability (exploding/vanishing gradients, policy collapse, reward stagnation). Researchers must manually inspect WandB logs.

**Options**

- **A** — Add a `TrainingHealthCallback` (Lightning callback) that raises structured warnings when: gradient norm > 100, reward moving average stagnates for > 50 epochs, entropy < threshold. Log to the structured logging system.
- **B** — Use `PyHessian` to compute the top-K Hessian eigenvalues of the policy network periodically; log sharpness as a training health proxy. `[Research]`
- **C** — Visualize the loss landscape slice (perturbation method) after training completes; save as a PNG artefact to `assets/analysis/`. See §G Phase 5 for the Studio's 3D loss landscape viewer.
- **D** — Add gradient norm and entropy to the existing WandB sweep metrics so Optuna / DEHB can prune unhealthy runs early.

**Recommendation**: **Option A** is a mandatory baseline — training health guardrails belong in every production RL pipeline. **Option D** pairs naturally with HPO (already integrated) and costs one additional metric log line. **Option B/C** are research-grade extras.

**Effort × Impact**: Low–Medium effort / High impact

**Delivered (§A.4 Option A — hundred-eleventh pass)**

- [x] ``TrainingHealthCallback`` — Lightning callback detecting gradient norm explosion (>100), reward stagnation (>50 epochs), and entropy collapse (<0.01); loguru warnings + alert cooldown
- [x] ``training_health_emit.py`` — ``TRAINING_HEALTH_START:`` stdout + ``training_health.jsonl`` under Lightning ``log_dir``
- [x] ``WSTrainer`` — auto-registers ``TrainingHealthCallback`` alongside checkpoint and tracking callbacks
- [x] Rust ``parse_training_health_line`` + ``load_training_health_log`` command
- [x] Studio ``TrainingHealthPanel`` — severity-coded alert list on Training Monitor; live stdout ingest + historical ``training_health.jsonl`` load
- [x] Unit tests in ``logic/test/unit/pipeline/callbacks/test_training_health.py``

**Delivered (§A.4 Option D — hundred-eighteenth pass)**

- [x] ``HpoHealthMetricsCallback`` — per-epoch ``train/grad_norm`` + ``train/entropy`` reporting to Optuna trial user attrs and WSTracker ``hpo/*`` metrics
- [x] Optuna objective — health callback wired alongside ``PyTorchLightningPruningCallback``; unhealthy trials pruned via ``TrialPruned``
- [x] DEHB objective — ``apply_dehb_health_penalty`` penalises fitness on grad explosion / entropy collapse
- [x] Ray Tune objective — per-epoch ``ray.train.report`` with ``grad_norm`` + ``entropy`` for ASHA schedulers
- [x] Studio HPO Tracker — trial health table with grad norm, entropy, and ``health_pruned`` badge (§G.18 bridge)
- [x] Unit tests in ``logic/test/unit/pipeline/callbacks/test_hpo_health.py``

**Delivered (§A.4 Option A — hundred-thirtieth pass)**

- [x] ``collectTrainingHealthFromLogLines`` — shared ``TRAINING_HEALTH_START:`` parser for process stdout
- [x] Training Hub — ``TrainingHealthPanel`` during live train/hpo runs; stdout ingest alongside metrics (§G.10 / §A.4)
- [x] Process Monitor — ``TrainingHealthPanel`` for selected ``train_`` / ``hpo_`` processes (§G.15 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-first pass)**

- [x] ``isTrainOrHpoProcess`` — shared train/HPO command matcher (Process Monitor parity)
- [x] Training Monitor — live health alerts for ``hpo_*`` processes alongside ``train_*`` (§G.17 / §A.4)
- [x] HPO Tracker — ``TrainingHealthPanel`` during live ``hpo_*`` runs; bridges §A.4 Option D trial health table (§G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-second pass)**

- [x] Experiment Tracker — ``TrainingHealthPanel`` during live ``hpo_*`` runs (§G.18 / §A.4)
- [x] Training Hub — ``liveTrainProcessLabel`` for Live HPO header; ``HPO Tracker →`` shortcut during live HPO (§G.10 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-third pass)**

- [x] Training Hub — ``Experiment Tracker →`` shortcut during live HPO; ``Process Monitor →`` label parity (§G.10 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-fourth pass)**

- [x] Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker — ``Training Hub →`` shortcut during live train/HPO workflows (§G.10 / §G.15 / §G.17 / §G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-fifth pass)**

- [x] ``LiveTrainProgressBar`` — epoch/trial progress bar + elapsed + ETA on Training Hub, Training Monitor, HPO Tracker, and Experiment Tracker during live runs; shared ``processProgress.ts`` helpers (§D.2 / §G.10 / §G.17 / §G.18 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-sixth pass)**

- [x] Process Monitor — ``LiveTrainProgressBar`` replaces inline ``PROGRESS:`` row bar; elapsed + ETA parity on all running processes (train/hpo/sim/data gen) (§D.2 / §G.15 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-seventh pass)**

- [x] Simulation Launcher — ``LiveTrainProgressBar`` in live status panel during running ``test_sim`` processes (§D.2 / §G.9 / §A.4)
- [x] Data Generation Wizard — ``LiveTrainProgressBar`` in live progress panel during ``gen_data`` runs (§D.2 / §G.11 / §A.4)

**Delivered (§A.4 Option A — hundred-thirty-eighth pass)**

- [x] Evaluation Runner — live progress panel with per-checkpoint ``LiveTrainProgressBar`` during ``eval`` runs; multi-checkpoint aggregate status header + stdout tail (§D.2 / §G.12 / §A.4)

**Status**: §A.4 Options A+D complete — Options B/C (PyHessian, loss landscape PNG) deferred.

---

### §A.5 — HPO Analytics: Cross-Trial Visualizer

**Pain**: The HPO module supports Optuna, Ray Tune, and DEHB, but the results are stored as trial databases without a unified post-hoc analysis view. Users cannot easily compare hyperparameter importance or visualize Pareto frontiers across objectives.

**Options**

- **A** — Use `optuna.visualization` (already a transitive dependency) to render parallel-coordinates and importances plots; export to `assets/hpo_reports/`. `[Quick Win]`
- **B** — Add a dedicated HPO Analysis panel in the Studio (§G Phase 10) wrapping the Optuna visualization calls in a native WebView or exporting Plotly HTML to be rendered inline.
- **C** — Export all trial results to a Pandas DataFrame; add a `hpo_summary.ipynb` notebook template that loads and plots them.
- **D** — Integrate SHAP to compute hyperparameter contribution scores across trials. `[Research]`

**Recommendation**: **Option A** for immediate wins (one function call with Optuna's built-in plotting), **Option C** as the notebook companion for sharing results.

**Effort × Impact**: Very Low effort (Option A) / Medium impact

**Delivered (§A.5 Option A — hundred-tenth pass)**

- [x] ``hpo_reports.py`` — ``optuna.visualization`` parallel-coordinates, param-importances, and optimisation-history Plotly HTML (+ optional PNG when kaleido present) under ``assets/hpo_reports/<study>_<timestamp>/``
- [x] ``manifest.json`` per export with study metadata, artefact list, and plot errors
- [x] ``run_hpo_sim`` post-run hook auto-exports reports after fANOVA analysis
- [x] Rust ``export_optuna_reports`` command; HPO Tracker **Export Plotly** + **Reports** folder open (§G.18 bridge)
- [x] Unit tests in ``logic/test/unit/pipeline/simulations/test_hpo_reports.py``

**Status**: §A.5 Option A complete — Option B (Studio WebView inline Plotly) largely superseded by ECharts HPO Tracker; Option C (notebook template) deferred.

---

### §A.6 — Causal Simulation Failure Analysis

**Pain**: When a simulation day ends with overflows or negative profit, the root cause (fill-rate spike, capacity miscalculation, policy sub-optimality) is not automatically identified. Post-hoc debugging requires re-reading JSON logs line by line.

**Options**

- **A** — Add a `FailureAnalyzer` class to `logic/src/pipeline/simulations/` that, after each day, compares predicted vs. actual bin fill levels, flags bins that caused overflow, and writes a structured summary to the day's JSON log entry.
- **B** — Build a counterfactual engine: re-run the day with the optimal policy (Gurobi) whenever a heuristic fails, and log the gap. `[Research]`
- **C** — Visualize the failure mode as a route-diff overlay in the Studio geospatial view (§G Phase 3): bins that were skipped vs. bins that overflowed highlighted in red. Depends on §A.1.
- **D** — Use causal inference (DoWhy) to identify which features (fill_rate, capacity, graph_size) most predict failure across simulation episodes. `[Research]`

**Recommendation**: **Option A** is purely additive and requires no new dependencies — pure logic in the existing simulator. **Option C** is the natural follow-on once §G Phase 3 is implemented.

**Effort × Impact**: Medium effort / High impact

**Delivered (§A.6 Option A — hundred-twelfth pass)**

- [x] ``FailureAnalyzer`` — post-day root-cause analysis comparing predicted vs. actual fill, flagging overflow bins, fill-rate spikes, and skipped high-fill bins
- [x] ``failure_emit.py`` — ``SIM_FAILURE_START:`` stdout marker + JSONL append; embedded ``failure_analysis`` in ``GUI_DAY_LOG_START`` payloads
- [x] ``LogAction`` — runs analyzer after each day; attaches summary to daily log dict
- [x] Rust ``parse_sim_failure_line`` + ``load_sim_failure_log`` command; ``sim:failure_update`` watcher events
- [x] Studio ``FailureAnalysisPanel`` — severity-coded causes, overflow bin table, skipped high-fill chips on Simulation Monitor
- [x] Unit tests in ``logic/test/unit/pipeline/simulations/test_failure_analyzer.py``

**Delivered (§A.6 Option C — hundred-fifteenth pass)**

- [x] ``routeFailureOverlay.ts`` — shared overflow/skipped bin id extraction + tour-diff sets for multi-policy compare
- [x] ``FailureOverlayLegend`` — reusable legend for overflow (red), skipped high-fill (orange), and tour-diff rings
- [x] ``DeckRouteMap`` — failure highlight ``ScatterplotLayer`` on Mercator + OrbitView; tour-diff ring overlay when two policies compared in overlay layout
- [x] Simulation Monitor — **Show/Hide failure overlay** + **Show/Hide route diff** toggles; wired to deck.gl and ECharts ``RouteViz`` (failure colours via embedded ``failure_analysis``)
- [x] ``RouteViz`` — legend when failure bins present; ``routeViz.ts`` uses shared overlay helper

**Delivered (§A.6 Option C — hundred-sixteenth pass)**

- [x] ``routeViz.ts`` — ``showFailureOverlay`` toggle; dual-policy overlay paths; tour-diff ring borders via ``TOUR_DIFF_RGB`` on ECharts scatter nodes
- [x] ``RouteViz`` — ``compareData`` / ``showTourDiff`` props; combined ``FailureOverlayLegend`` for failure + diff modes
- [x] Simulation Monitor — ECharts overlay compare when two map policies visible; failure + route-diff toggles propagate to ``RouteViz`` (parity with deck.gl)
- [x] Simulation Summary — **Show/Hide failure overlay** + **Show/Hide route diff** toggles; overlay-compare ``RouteViz`` when exactly two brushed policies share a day

**Status**: §A.6 Options A+C complete — Options B/D (counterfactual engine, DoWhy) deferred.

---

### Effort × Impact Matrix — Analytics & Interpretability

| Item                                     | Effort    | Impact | Priority        |
| ---------------------------------------- | --------- | ------ | --------------- |
| §A.3 Option A (PolicyVizMixin → Studio)  | Very Low  | High   | P0 ✅            |
| §A.3 Option B (2 Hz live telemetry stream) | Low    | High   | P1 ✅            |
| §A.3 Option C (SQLite cross-run trending) | Low    | Medium | P2 ✅            |
| §A.5 Option A (Optuna plots)             | Very Low  | Medium | P0 ✅            |
| §A.4 Option A (TrainingHealthCallback)   | Low       | High   | P1 ✅            |
| §A.4 Option D (HPO health prune metrics) | Low       | High   | P1 ✅            |
| §A.2 Option C (WandB attention heatmaps) | Low       | High   | P1 ✅            |
| §A.2 Option A (Studio attention ring-buffer) | Medium | High | P1 ✅            |
| §A.6 Option A (FailureAnalyzer)          | Medium    | High   | P1 ✅            |
| §A.6 Option C (route-diff overlay)       | Medium    | High   | P2 ✅            |
| §A.1 Option A (ECharts route viz)        | Medium    | High   | P2 ✅            |
| §A.1 Option E (deck.gl PathLayer)        | High      | High   | P2 ✅ (§G.3/§G.16) |
| §A.4 Option B (PyHessian)                | High      | Medium | P3 `[Research]` |
| §A.6 Option B (counterfactual engine)    | Very High | High   | P3 `[Research]` |

### §A — Analytics & Interpretability Complete ✅

All P0–P2 analytics bridges are delivered (§A.1–§A.6). Remaining items are research-grade extras (PyHessian, counterfactual engine, DoWhy, BertViz) or release-adjacent notebook templates (§A.5 Option C).

---

