> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## D — GUI / UX

> **Context**: The existing PySide6/Qt GUI (`gui/src/`) is being migrated to WSmart-Route Studio, a Tauri 2.0 application (§G). The requirements in this section remain valid; the implementation guidance is updated to reflect the new Tauri/React/TypeScript stack. All references to Qt-specific APIs (QApplication, QThread, QSettings, QWidget subclasses, etc.) have been replaced with their Tauri/React equivalents.

---

### §D.1 — Route Visualization Panel

**Pain**: The Studio's analysis views show dataset statistics and fill-rate charts, but have no panel for visualizing computed routes. After running a simulation or evaluation, users must read JSON output to understand what routes were computed.

**Options**

- **A** — Add a `RouteViz` React component in the Studio using ECharts `custom` series or a 2D `<canvas>` renderer: plot depot (star), customer nodes (circles sized by demand), and route edges (colour per vehicle). Load routes from simulation JSON output. Synergises with §A.1.
- **B** — Use the deck.gl `PathLayer` + `ScatterplotLayer` already integrated for the geospatial phase (§G Phase 3) in Cartesian OrbitView mode — repurpose the same renderer for abstract coordinate systems.
- **C** — Open routes in an external browser tab via a locally-served Plotly map each time the user clicks "Visualize". Breaks the desktop-app UX.

**Recommendation**: **Option A** immediately (pure React, no additional dependencies), **Option B** as the production upgrade once §G Phase 3 (deck.gl) is in place.

**Effort × Impact**: Medium effort / High impact

---

### §D.2 — Training Progress Enhancements

**Pain**: The Studio's training launcher shows progress via a streamed log view (reading subprocess stdout), but the UX is a plain text area. There is no live loss curve, no epoch progress bar, and no ETA display.

**Options**

- **A** — Parse the structured JSON log emitted by the training pipeline inside the Rust backend; forward parsed metric events to React via Tauri's event system (`emit` / `listen`). Update: a live ECharts line chart (loss / reward curves), a `<progress>` element for epoch progress, and a computed ETA label. The file-watch approach in §G Phase 15 (Real-Time Process Monitor) provides the streaming infrastructure. `[Quick Win]` for the progress bar; more work for the live chart.
- **B** — Add a Rust `TrainingMetricsWatcher` that watches the WandB run directory for new log entries and forwards them to the React frontend as Tauri events.
- **C** — Embed the live WandB dashboard URL in a Tauri WebView panel. Requires an active WandB connection.

**Recommendation**: **Option A** — parse structured logs (already JSON-formatted) via the Tauri file-watch event system. Zero external dependency; consistent with §G Phase 15 infrastructure.

**Effort × Impact**: Medium effort / High impact

---

### §D.3 — Dark / Light Theme Toggle

**Pain**: The Studio uses a fixed dark theme. There is no runtime toggle exposed to the user, and the system theme preference is not respected.

**Options**

- **A** — Implement a theme toggle in the Studio's settings panel using Tailwind CSS `dark:` variant classes combined with a root `data-theme` attribute toggled via React state. Persist selection to `localStorage`. `[Quick Win]`
- **B** — Use the Tauri Store plugin (`@tauri-apps/plugin-store`) to persist the theme preference so it is restored across app restarts. Synergises with §D.4.
- **C** — Add a system-theme-following mode using the CSS `prefers-color-scheme` media query; detect the system preference on startup and switch automatically.

**Recommendation**: **Option A + B** together — both are trivial once Tailwind dark mode is configured, and Store plugin persistence is a one-liner.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §D.4 — Session Persistence

**Pain**: When the Studio is closed and re-opened, all configured parameters (problem type, model path, dataset path, number of days, policy selections) are reset to defaults. Users must reconfigure every session.

**Options**

- **A** — Persist the current form state (all input values, selected options) using Zustand's `persist` middleware with `localStorage` as the storage backend. Restore on app mount. `[Quick Win]`
- **B** — Use the Tauri Store plugin (`@tauri-apps/plugin-store`) for cross-platform native key-value persistence — writes to an OS-appropriate config directory rather than browser storage. More robust than localStorage for a desktop app.
- **C** — Allow users to name and save multiple "session profiles" (e.g., "VRPP-50-nodes", "WCVRP-simulation") and switch between them from a dropdown.

**Recommendation**: **Option B** first (idiomatic Tauri, writes to a proper config path), **Option C** for power users.

**Effort × Impact**: Low effort / High impact

---

### §D.5 — Progress & Cancellation for Long Operations

**Pain**: Data generation, training, and simulation runs can take hours. Users have no mechanism to cancel a running operation without force-quitting the app, and there is no progress indicator for operations that don't emit epoch-level logs.

**Options**

- **A** — Add a cancel mechanism: the Rust backend spawns each long-running Python process via `tokio::process::Command`; a `cancel` Tauri command sends SIGTERM (or Windows equivalent) to the child process. A React "Cancel" button invokes this command. Implements the Rust `AsyncTask` trait from §B.8 Option B.
- **B** — For multiprocessing-based operations (simulation uses `multiprocessing`), pass a cancellation flag via a shared file sentinel (`assets/.cancel_flag`) that the Python side polls; the Rust backend creates/removes the file on cancel request.
- **C** — Show a React modal progress dialog for operations with known total steps; show an indeterminate spinner for open-ended operations. Subscribe to Tauri progress events (emitted by §G Phase 15 infrastructure) to update the progress bar.

**Recommendation**: **Option A + C** — the Tauri command (A) provides the cancel mechanism; the React progress modal (C) provides the UX. Option B as a fallback for multiprocessing operations where SIGTERM does not propagate to worker processes.

**Effort × Impact**: Medium effort / High impact

---

### §D.6 — Configuration Panel for Hydra Overrides

**Pain**: The Studio exposes only a subset of available Hydra configuration options. Advanced users who want to override `train.batch_size`, `model.embedding_dim`, or `env.num_loc` must edit config files or use the CLI — bypassing the Studio entirely.

**Options**

- **A** — Add an "Advanced Overrides" collapsible section in each launcher panel (simulation, training, data gen) rendering a React table of key-value rows. Users can add/edit/delete rows; the Rust backend translates them to Hydra override strings (`key=value`) appended to the subprocess call. `[Quick Win]`
- **B** — Parse the Hydra config schema (via `OmegaConf.to_yaml`) at startup and generate a typed React form using `react-hook-form` + `zod` for validation: dropdowns for string enums, sliders for bounded numerics, checkboxes for bools.
- **C** — Embed a Monaco Editor YAML panel that is passed directly as a Hydra config override file — maximum power, minimal guardrails.

**Recommendation**: **Option A** for immediate usefulness (generic override table, one afternoon of work), **Option B** as the polished version once the config schema introspection is stable, **Option C** for expert users who prefer raw YAML access.

**Effort × Impact**: Medium effort / High impact

---

### §D.7 — Keyboard Shortcuts & Command Palette

**Pain**: All Studio operations require mouse clicks. Power users running repeated experiments have no keyboard-driven workflow.

**Options**

- **A** — Register global keyboard shortcuts in React using `react-hotkeys-hook` (in-window shortcuts) or `@tauri-apps/plugin-global-shortcut` (OS-level shortcuts): `Ctrl+R` (run), `Ctrl+.` (cancel), `Ctrl+1`–`Ctrl+9` (switch tabs), `Ctrl+S` (save config). Display shortcuts in a Help overlay. `[Quick Win]`
- **B** — Implement a command palette (`Ctrl+Shift+P`) as a floating React component (`cmdk` library or equivalent) backed by a registry of all Studio actions; filter by typing. Particularly useful as the Studio grows beyond 10 top-level views.

**Recommendation**: **Option A** first (one `useHotkeys` call per action), **Option B** once the Studio has more than one launcher and multiple analytics views.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

**Delivered (§D.7 — hundred-thirty-ninth pass)**

- [x] ``LauncherNavMesh`` — shared cross-page sim / data-gen / eval launcher navigation component; replaces duplicated shortcut buttons on Simulation Launcher, Data Generation Wizard, Evaluation Runner, and Process Monitor (§G.9 / §G.11 / §G.12 / §G.15)
- [x] ``launcherProcess.ts`` — shared ``isSimProcess`` / ``isGenDataProcess`` / ``isEvalProcess`` helpers for Process Monitor launcher workflow panels
- [x] Keyboard shortcuts ``L`` → Simulation Launcher, ``D`` → Data Generation, ``V`` → Evaluation Runner; help overlay updated (§D.7)

**Delivered (§D.7 — hundred-forty-first pass)**

- [x] ``LauncherNavMesh`` ``Output Browser →`` + ``Load in Eval Runner →`` on completed eval processes (§G.12 / §G.14 / §G.15)
- [x] Keyboard shortcuts ``B`` → Benchmark Analysis, ``O`` → Output Browser; help overlay updated (§D.7)

**Delivered (§D.7 — hundred-forty-third pass)**

- [x] ``outputRunPath.ts`` — derive assets/output run directory from process stdout ``.jsonl`` paths (§G.14 / §G.9 / §G.15)
- [x] ``LauncherNavMesh`` / ``TrainHpoNavMesh`` — ``outputRunPath`` prop sets ``pendingRunPath`` before navigating to Output Browser (§G.14 / §D.7)
- [x] Simulation Launcher + Data Generation — post-run Output Browser deep-links to the completed run when stdout contains a log path (§G.9 / §G.11 / §G.14)
- [x] Process Monitor — ``Output Browser →`` on completed ``test_sim`` / ``gen_data`` processes with run deep-link (§G.15 / §G.14)

**Delivered (§D.7 — hundred-forty-fourth pass)**

- [x] ``outputRunPath.ts`` — Hydra snapshot / pruned-config / ``assets/output`` path parsing as fallback when no ``.jsonl`` in stdout (§G.14 / §G.9 / §G.12 / §G.15)
- [x] ``trainingRunPath.ts`` + ``pendingTrainingRunPath`` — Training Monitor deep-link from completed train/HPO processes (§G.10 / §G.17 / §D.7)
- [x] Evaluation Runner + Process Monitor eval — ``outputRunPath`` deep-link parity on completed eval workflows (§G.12 / §G.14 / §G.15)

**Delivered (§D.7 — hundred-forty-fifth pass)**

- [x] ``findRecentHpoProcessId`` / ``findRecentTrainOrHpoProcessId`` — retain newest train/HPO process after completion for post-run panels (§G.17 / §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker — post-run ``outputRunPath`` + ``trainingRunPath`` on ``TrainHpoNavMesh`` when HPO sweep completes (§G.18 / §G.14 / §G.17 / §D.7)
- [x] Training Monitor — post-run deep-link parity on live/recent train panel; auto-refresh + select completed run from ``trainingRunPath`` (§G.17 / §G.10 / §D.7)

**Delivered (§D.7 — hundred-forty-sixth pass)**

- [x] ``findRecentLauncherProcessId`` / ``findRecentEvalProcessIds`` — retain newest sim / data-gen / eval launcher processes after completion (§G.9 / §G.11 / §G.12 / §D.7)
- [x] ``findRecentTrainProcessId`` — train-only recent process helper for Training Hub train mode (§G.10 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — live/post-run panels rehydrate from ``useProcessStore`` when navigation clears local ``liveProcessId`` state (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)

**Delivered (§D.7 — hundred-forty-seventh pass)**

- [x] ``trainingMetrics.ts`` — ``normalizeTrainingMetricRow`` exported for CSV + stdout parity (§G.17 / §G.10)
- [x] Training Monitor — post-run metrics/health/attention rehydrate from ``useProcessStore`` log lines; ``LIVE_KEY`` overlay chart persists after completion (§G.17 / §D.7)
- [x] HPO Tracker + Experiment Tracker — live metric snapshot row from persisted process stdout (§G.18 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-forty-eighth pass)**

- [x] ``TrainingMetricSparklines`` — shared ``GradNormSparkline`` + ``LrSparkline`` + ``TrainingMetricSnapshot`` for train/HPO analytics panels (§G.15 / §G.17 / §G.18)
- [x] Process Monitor — train/HPO metrics rehydrate from ``useProcessStore``; grad-norm + LR sparklines persist after completion (§G.15 / §D.7)
- [x] HPO Tracker + Experiment Tracker — post-run grad-norm + LR sparklines from persisted HPO stdout (§G.18 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-forty-ninth pass)**

- [x] Training Hub — post-run grad-norm + LR sparklines from persisted train/HPO stdout via ``TrainingMetricSparklines``; ``TrainingMetricSnapshot`` + rehydration banner (§G.10 / §D.7)
- [x] Training Monitor — deduplicated local sparklines; imports shared ``GradNormSparkline`` + ``LrSparkline`` + ``TrainingMetricSnapshot`` (§G.17 / §D.7)
- [x] §G.10 / §G.17 launcher + monitor post-run sparkline parity across all train/HPO workflow pages (§D.7)

**Delivered (§D.7 — hundred-fiftieth pass)**

- [x] ``postRunTrainingRehydrationMessage`` — shared post-run banner helper; mentions metrics, health alerts, and attention snapshots when rehydrated from ``useProcessStore`` (§G.10 / §G.15 / §G.17 / §G.18)
- [x] HPO Tracker + Experiment Tracker — deduplicated inline metric snapshot rows; import shared ``TrainingMetricSnapshot`` (§G.18 / §G.17 / §D.7)
- [x] Training Hub + Training Monitor + Process Monitor — post-run banner uses shared helper for health/attention rehydration parity (§G.10 / §G.15 / §G.17 / §D.7)
- [x] §G.18 / §G.17 analytics post-run snapshot + health/attention banner parity across all train/HPO workflow pages (§D.7)

**Delivered (§D.7 — hundred-fifty-first pass)**

- [x] ``TrainHpoAnalyticsStrip`` — shared snapshot + sparklines + health/attention + post-run banner component for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18)
- [x] Training Hub + Process Monitor + HPO Tracker + Experiment Tracker — deduplicated inline analytics blocks; import shared ``TrainHpoAnalyticsStrip`` (§G.10 / §G.15 / §G.18 / §D.7)
- [x] Training Monitor — live/recent card uses ``TrainHpoAnalyticsStrip`` for post-run sparkline rehydration without ``LIVE_KEY`` selection (§G.17 / §D.7)
- [x] Training Hub + Training Monitor — ``metric updates`` label parity with Process Monitor / HPO / Experiment Tracker (§G.10 / §G.17 / §D.7)

**Delivered (§D.7 — hundred-fifty-second pass)**

- [x] Training Monitor — ``TrainHpoAnalyticsStrip`` receives rehydrated ``healthEntries`` + ``attentionEntries`` for post-run banner counts while page-level panels remain separate (``showHealthAttention={false}``) (§G.17 / §A.2 / §A.4 / §D.7)
- [x] Training Monitor — ``metric updates`` label on non-checkbox live/recent header when metrics are rehydrated from process store (§G.17 / §D.7)
- [x] Training Hub — ``metric updates`` label uses ``text-accent-success`` styling parity with Process Monitor / HPO / Experiment Tracker (§G.10 / §D.7)

**Delivered (§D.7 — hundred-fifty-third pass)**

- [x] ``TrainHpoRehydrationBadges`` — shared metric / health / attention count badges for train/HPO live panel headers (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline ``metric updates`` labels; header badges surface health alerts + attention snapshots when rehydrated from ``useProcessStore`` (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Monitor — checkbox live/recent header no longer shows ``0 metric updates`` when only health/attention are rehydrated (§G.17 / §A.2 / §A.4 / §D.7)

**Delivered (§D.7 — hundred-fifty-fourth pass)**

- [x] ``TrainHpoLivePanelHeader`` — shared status icon + title + process id + rehydration badges + ``TrainHpoNavMesh`` header row for train/HPO live panels (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub — ``split`` layout + ``activity`` running icon parity via shared header (§G.10 / §D.7)
- [x] HPO Tracker + Experiment Tracker — deduplicated inline live HPO header blocks (§G.18 / §G.17 / §D.7)
- [x] Process Monitor — ``muted`` analytics subtitle header + badges-before-nav ordering parity (§G.15 / §D.7)

**Delivered (§D.7 — hundred-fifty-fifth pass)**

- [x] ``TrainHpoLivePanelHeader`` — ``overlaySelect`` prop for ``LIVE_KEY`` overlay-chart checkbox on Training Monitor (§G.17 / §A.2 / §A.4 / §D.7)
- [x] Training Monitor — deduplicated inline live/recent header blocks; shared status icon + title + process id + rehydration badges + ``TrainHpoNavMesh`` row (§G.17 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow header row parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-fifty-sixth pass)**

- [x] ``TrainHpoLivePanel`` — shared header + ``LiveTrainProgressBar`` + ``TrainHpoAnalyticsStrip`` shell for train/HPO live/post-run panels (§G.10 / §G.15 / §G.17 / §G.18 / §A.2 / §A.4 / §D.7)
- [x] Training Hub + Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline live panel card markup; ``card`` vs ``embedded`` variant parity (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Hub — ``footer`` process-id row + ``showAnalytics`` / ``analyticsWrapperClassName`` slots preserved via shared panel (§G.10 / §D.7)
- [x] Training Monitor — ``overlaySelect`` + ``showHealthAttention={false}`` analytics options preserved via shared panel (§G.17 / §A.2 / §A.4 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow live panel shell parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-fifty-seventh pass)**

- [x] ``LauncherLivePanelHeader`` — shared status icon + title + ``LauncherNavMesh`` header row for sim / data-gen / eval launcher workflows (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``LauncherLivePanel`` — shared header + ``LiveTrainProgressBar`` + children shell with ``card`` vs ``embedded`` variant parity (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner — deduplicated inline live progress card markup; ``navTrailing`` slot preserves sim auto-summary countdown (§G.9 / §G.11 / §G.12 / §D.7)
- [x] Process Monitor — ``embedded`` variant for selected ``test_sim`` / ``gen_data`` / ``eval`` analytics sections; run-label + live suffix parity on sim panel (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] §G.9 / §G.11 / §G.12 / §G.15 launcher workflow live panel shell parity across all four pages (§D.7)

**Delivered (§D.7 — hundred-fifty-eighth pass)**

- [x] ``ProcessIdFooter`` — shared process-id footer row for launcher and train/HPO live panels (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)
- [x] Simulation Launcher + Training Hub — deduplicated inline process-id footer markup; import shared ``ProcessIdFooter`` (§G.9 / §G.10 / §D.7)
- [x] Data Generation Wizard + Evaluation Runner — ``LauncherLivePanel`` ``footer`` process-id row parity with Simulation Launcher (§G.11 / §G.12 / §D.7)
- [x] Evaluation Runner — multi-checkpoint footer lists all ``displayProcessIds`` when batch eval is active (§G.12 / §D.7)
- [x] ``EvalResultKpiRow`` — shared cost / gap / time / policy KPI row for eval live panels (§G.12 / §G.15 / §D.7)
- [x] ``EvalResultCard`` — shared eval result card with ``Open in Analytics →`` for Process Monitor embedded eval section (§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner — per-checkpoint live panel uses ``EvalResultKpiRow`` ``compact`` variant (§G.12 / §D.7)
- [x] §G.12 / §G.15 eval result KPI + footer parity across Evaluation Runner and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-fifty-ninth pass)**

- [x] ``ProcessIdFooter`` — monitor-page footer parity: Training Monitor, HPO Tracker, Experiment Tracker, and Process Monitor embedded sections; process id removed from inline headers (§G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Monitor + HPO Tracker + Experiment Tracker — ``TrainHpoLivePanel`` ``footer`` process-id row parity with Training Hub (§G.10 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor — ``LauncherLivePanel`` + ``TrainHpoLivePanel`` embedded sections use ``ProcessIdFooter``; simplified analytics subtitles without inline process id (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` — shared per-checkpoint live eval row with KPI, progress bar, and stdout tail (§G.12 / §D.7)
- [x] Evaluation Runner — deduplicated inline per-checkpoint live panel markup; import shared ``EvalCheckpointLiveCard`` (§G.12 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow footer parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-sixtieth pass)**

- [x] ``processLogTail`` — shared stdout/stderr tail helper for live eval panels (§G.12 / §G.15 / §D.7)
- [x] Process Monitor — selected ``eval`` processes use ``EvalCheckpointLiveCard`` during live runs and while waiting for structured JSON; ``EvalResultCard`` retained on completion with metrics (§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner — deduplicated inline log tail formatting; import shared ``processLogTail`` (§G.12 / §D.7)
- [x] §G.12 / §G.15 eval live checkpoint card parity across Evaluation Runner and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-sixty-first pass)**

- [x] ``ProcessLogTail`` — shared stdout/stderr tail display component for launcher live panels (§G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` — deduplicated inline log tail markup; import shared ``ProcessLogTail`` (§G.12 / §D.7)
- [x] Data Generation Wizard — deduplicated inline log tail markup; ``processLogTail`` + ``ProcessLogTail`` parity (§G.11 / §D.7)
- [x] Process Monitor — selected ``gen_data`` processes show ``ProcessLogTail`` in embedded workflow section (§G.11 / §G.15 / §D.7)
- [x] §G.11 / §G.12 / §G.15 launcher log tail display parity across Data Generation, Evaluation Runner, and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-sixty-second pass)**

- [x] ``EvalCheckpointLiveCard`` — accepts ``logLines`` directly; deduplicated ``processLogTail`` calls at call sites (§G.12 / §D.7)
- [x] Evaluation Runner — passes raw ``logLines`` to ``EvalCheckpointLiveCard`` instead of pre-formatted tail (§G.12 / §D.7)
- [x] Process Monitor — eval embedded section passes raw ``logLines`` to ``EvalCheckpointLiveCard`` (§G.12 / §G.15 / §D.7)
- [x] Simulation Launcher — ``ProcessLogTail`` in live status panel during ``test_sim`` runs (§G.9 / §D.7)
- [x] Process Monitor — selected ``test_sim`` processes show ``ProcessLogTail`` in embedded workflow section (§G.9 / §G.15 / §D.7)
- [x] §G.9 / §G.11 / §G.12 / §G.15 launcher log tail display parity across all four launcher pages + Process Monitor embedded sections (§D.7)

**Delivered (§D.7 — hundred-sixty-third pass)**

- [x] ``TrainHpoLivePanel`` — optional ``logLines`` + ``logTailWaiting`` props render shared ``ProcessLogTail`` below analytics strip (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Hub — ``ProcessLogTail`` in live progress panel during train/hpo/eval runs (§G.10 / §D.7)
- [x] Process Monitor — selected ``train_`` / ``hpo_`` processes show ``ProcessLogTail`` in embedded analytics section (§G.15 / §D.7)
- [x] Training Monitor + HPO Tracker + Experiment Tracker — ``ProcessLogTail`` on live/recent train/HPO panels (§G.17 / §G.18 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow log tail display parity across all five pages + Process Monitor embedded section (§D.7)

**Delivered (§D.7 — hundred-sixty-fourth pass)**

- [x] ``LauncherLivePanel`` — optional ``logLines`` + ``logTailWaiting`` props render shared ``ProcessLogTail`` below children (§G.9 / §G.11 / §G.15 / §D.7)
- [x] Simulation Launcher — deduplicated inline ``ProcessLogTail`` child; pass ``logLines`` to shared panel shell (§G.9 / §D.7)
- [x] Data Generation Wizard — deduplicated inline ``ProcessLogTail`` child; pass ``logLines`` to shared panel shell (§G.11 / §D.7)
- [x] Process Monitor — selected ``test_sim`` / ``gen_data`` embedded sections pass ``logLines`` to ``LauncherLivePanel`` instead of inline ``ProcessLogTail`` (§G.9 / §G.11 / §G.15 / §D.7)
- [x] §G.9 / §G.11 / §G.15 launcher workflow log tail display parity via shared panel props across all launcher pages + Process Monitor embedded sections (§D.7)

**Delivered (§D.7 — hundred-sixty-fifth pass)**

- [x] ``EvalCheckpointLiveCard`` — optional ``showLogTail`` prop; parent ``LauncherLivePanel`` renders shared log tail for single-checkpoint eval (§G.12 / §D.7)
- [x] Evaluation Runner — single-checkpoint live panel passes ``logLines`` to ``LauncherLivePanel``; multi-checkpoint batch retains per-card compact tails (§G.12 / §D.7)
- [x] Process Monitor — selected ``eval`` embedded section passes ``logLines`` to ``LauncherLivePanel`` instead of inline ``ProcessLogTail`` on ``EvalCheckpointLiveCard`` (§G.12 / §G.15 / §D.7)
- [x] §G.12 / §G.15 eval launcher log tail shell parity across Evaluation Runner and Process Monitor embedded section (§D.7)

**Delivered (§D.7 — hundred-sixty-sixth pass)**

- [x] Training Hub — eval mode uses ``LauncherLivePanel`` + ``EvalCheckpointLiveCard`` / ``EvalResultCard`` instead of ``TrainHpoLivePanel``; shared log tail via panel ``logLines`` prop (§G.10 / §G.12 / §D.7)
- [x] Training Hub — eval live panel ``LauncherNavMesh`` post-run shortcuts (Output Browser, Evaluation Runner reload, Benchmark Analysis) parity with Evaluation Runner (§G.10 / §G.12 / §D.7)
- [x] §G.10 / §G.12 eval launcher live panel shell parity across Training Hub and Evaluation Runner (§D.7)

**Delivered (§D.7 — hundred-sixty-seventh pass)**

- [x] Training Hub — eval live panel omits duplicate ``LauncherLivePanel`` progress bar; ``EvalCheckpointLiveCard`` owns ``LiveTrainProgressBar`` during runs (§G.10 / §G.12 / §D.7)
- [x] ``LauncherNavMesh`` — ``Training Hub →`` shortcut on eval workflows; ``hideHub`` prop suppresses self-link on Training Hub eval panel (§G.10 / §G.12 / §G.15 / §D.7)
- [x] §G.10 / §G.12 / §G.15 eval launcher progress + navigation parity across Training Hub, Evaluation Runner, and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-sixty-eighth pass)**

- [x] ``evalLivePanelTitle`` — shared live/post-run eval panel title helper in ``evalResults.ts`` (§G.10 / §G.12 / §G.15 / §D.7)
- [x] Training Hub + Evaluation Runner — deduplicated inline eval live title strings; import shared ``evalLivePanelTitle`` (§G.10 / §G.12 / §D.7)
- [x] Process Monitor — selected ``eval`` embedded section uses dynamic ``evalLivePanelTitle`` instead of static ``Eval results`` subtitle (§G.12 / §G.15 / §D.7)
- [x] §G.10 / §G.12 / §G.15 eval launcher live panel title parity across Training Hub, Evaluation Runner, and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-sixty-ninth pass)**

- [x] ``simLivePanelTitle`` — shared live/post-run sim panel title helper in ``launcherProcess.ts`` (§G.9 / §G.15 / §D.7)
- [x] ``dataGenLivePanelTitle`` — shared live/post-run data-gen panel title helper in ``launcherProcess.ts`` (§G.11 / §G.15 / §D.7)
- [x] Simulation Launcher + Data Generation — deduplicated inline sim/data-gen live title strings; import shared title helpers (§G.9 / §G.11 / §D.7)
- [x] Process Monitor — selected ``test_sim`` / ``gen_data`` embedded sections use dynamic title helpers instead of static subtitles (§G.9 / §G.11 / §G.15 / §D.7)
- [x] §G.9 / §G.11 / §G.15 sim + data-gen launcher live panel title parity across Simulation Launcher, Data Generation, and Process Monitor (§D.7)

**Delivered (§D.7 — hundred-seventieth pass)**

- [x] ``trainHpoLivePanelTitle`` — shared live/post-run train/HPO panel title helper in ``trainingProcess.ts`` (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Training Hub + Training Monitor + HPO Tracker + Experiment Tracker — deduplicated inline train/HPO live title strings; import shared ``trainHpoLivePanelTitle`` (§G.10 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor — selected ``train_`` / ``hpo_`` embedded sections use dynamic ``trainHpoLivePanelTitle`` instead of static ``Training analytics`` subtitle (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] §G.10 / §G.15 / §G.17 / §G.18 train/HPO workflow live panel title parity across all five pages (§D.7)

**Delivered (§D.7 — hundred-seventy-first pass)**

- [x] ``TrainHpoLivePanelHeader`` — ``runLabel`` prop for Process Monitor embedded run-label suffix parity with ``LauncherLivePanelHeader`` (§G.15 / §D.7)
- [x] ``TrainHpoLivePanel`` — ``embedded`` variant defaults ``titleTone: muted`` + ``showLiveSuffix: true`` for train/HPO analytics subtitles (§G.15 / §D.7)
- [x] Process Monitor — eval + data-gen + train/HPO embedded sections pass ``runLabel`` + live suffix; process row ring highlight + global ``run_label`` brush sync for all workflow kinds (§G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] §G.15 Process Monitor embedded run-label + live suffix parity across sim, data-gen, eval, and train/HPO workflow sections (§D.7)

**Delivered (§D.7 — hundred-seventy-second pass)**

- [x] ``useProcessRunLabelBrush`` — shared hook deriving ``run_label`` from process stdout and syncing global brush (§G.9–§G.18 / §D.7)
- [x] ``LauncherLivePanelHeader`` — ``runLabel`` + ``showLiveSuffix`` on card variant headers (§G.9 / §G.11 / §G.12 / §G.10 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` — ``runLabel`` + ``showLiveSuffix`` on split and inline card layouts (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — card live panel headers pass ``runLabel``; ``GlobalFilterBar`` ``runLabels`` when process active (§G.9 / §G.10 / §G.11 / §G.12 / §D.7)
- [x] Training Monitor + HPO Tracker + Experiment Tracker — ``TrainHpoLivePanel`` card headers pass ``runLabel`` + ``showLiveSuffix``; ``GlobalFilterBar`` ``runLabels`` sync (§G.15 / §G.17 / §G.18 / §D.7)
- [x] §G.9 / §G.10 / §G.11 / §G.12 / §G.15 / §G.17 / §G.18 launcher + monitor workflow card header run-label + live suffix parity across all eight pages (§D.7)

**Delivered (§D.7 — hundred-seventy-third pass)**

- [x] ``runLabelMapFromProcesses`` — shared helper deriving per-process ``run_label`` from stdout for row ring highlights (§G.15 / §D.7)
- [x] Process Monitor — ``useProcessRunLabelBrush`` replaces inline ``runLabelFromLogLines`` + manual ``setRunLabel`` effect; brush sync parity with launcher/monitor card pages (§G.15 / §D.7)
- [x] §G.15 Process Monitor shared run-label brush hook parity across all workflow kinds (§D.7)

**Delivered (§D.7 — hundred-seventy-fourth pass)**

- [x] ``useLogPathRunLabelBrush`` — shared hook deriving ``run_label`` from log/run paths and syncing global brush (§G.14 / §G.16 / §D.7)
- [x] Simulation Monitor — ``GlobalFilterBar`` ``runLabels`` when a log is open; global brush sync on log open via shared hook (§G.16 / §D.7)
- [x] Output Browser — ``useLogPathRunLabelBrush`` replaces inline ``setRunLabel`` in ``selectRun``; trends panel uses hook-derived label (§G.14 / §D.7)
- [x] §G.14 / §G.16 file-based workflow run-label brush hook parity with process-based launcher/monitor pages (§D.7)

**Delivered (§D.7 — hundred-seventy-fifth pass)**

- [x] ``runLabelMapFromPaths`` — shared helper deriving per-run ``run_label`` from paths for file-based row ring highlights (§G.1 / §G.14 / §D.7)
- [x] Simulation Summary — ``useLogPathRunLabelBrush`` on primary log open; ``GlobalFilterBar`` ``runLabels`` in single-log mode; trends panel uses hook-derived label (§G.1 / §G.16 / §D.7)
- [x] Output Browser — ``runLabelMapFromPaths`` replaces inline ``runLabelFromPath`` in run list ring highlights (§G.14 / §D.7)
- [x] Simulation Summary — comparison-run list ring highlight via ``runLabelMapFromPaths`` (§G.1 / §G.6 / §D.7)
- [x] §G.1 / §G.14 / §G.16 file-based run-label brush + ring-highlight parity across Simulation Summary, Simulation Monitor, and Output Browser (§D.7)

**Delivered (§D.7 — hundred-seventy-sixth pass)**

- [x] Benchmark Analysis + City Comparison — ``runLabelMapFromPaths`` + ``handleRunLabelClick`` on loaded-run lists; ring highlight when global brush matches (§G.1 / §G.6 / §D.7)
- [x] Benchmark Analysis + City Comparison — ``GlobalFilterBar`` ``runLabels`` in single-run portfolio mode (§G.1 / §G.6 / §D.7)
- [x] Algorithm Comparison — ``useLogPathRunLabelBrush`` on Simulation Monitor watch path; ``GlobalFilterBar`` ``runLabels`` when log active (§G.1 / §G.16 / §D.7)
- [x] Data Explorer — ``useLogPathRunLabelBrush`` on open CSV path; path-derived ``runLabels`` + trends ``initialRunLabel`` fallback when CSV lacks ``run_label`` column (§G.6 / §G.16 / §D.7)
- [x] §G.1 / §G.6 portfolio + analytics page run-label brush + ring-highlight parity across Summary / Benchmark / City / Algorithm / Data Explorer (§D.7)

**Delivered (§D.7 — hundred-seventy-seventh pass)**

- [x] ``runLabelMapFromTablePaths`` — shared helper deriving per-table ``run_label`` from ingest source paths (§G.6 / §D.7)
- [x] OLAP Explorer — ``useLogPathRunLabelBrush`` on selected custom-table ingest path; global brush sync on table select (§G.6 / §G.16 / §D.7)
- [x] OLAP Explorer — ingested-table picker ring highlight + click-to-brush parity with Output Browser run list (§G.6 / §G.14 / §D.7)
- [x] OLAP Explorer — path-derived ``GlobalFilterBar`` ``runLabels`` + trends ``initialRunLabel`` fallback when table lacks ``run_label`` column (§G.6 / §G.16 / §D.7)
- [x] §G.6 OLAP Explorer file-based run-label brush + ring-highlight parity across all analysis views (§D.7)

**Delivered (§D.7 — hundred-seventy-eighth pass)**

- [x] ``runLabelMapFromSingleTableLabels`` / ``tableRunLabelBrushActive`` — DuckDB table ``run_label`` helpers for portfolio table-picker brush parity (§G.6 / §D.7)
- [x] ``useTableRunLabelBrush`` — shared hook syncing global brush when a built-in DuckDB table has exactly one ``run_label`` (§G.6 / §D.7)
- [x] OLAP Explorer — ``refreshTables`` indexes distinct ``run_label`` values per table; built-in portfolio tables (``summary_sim`` / ``benchmark_sim`` / ``city_sim`` / ``algorithm_sim``) share table-picker ring-highlight + click-to-brush parity (§G.6 / §G.14 / §D.7)
- [x] OLAP Explorer — single-run built-in table brush sync + ``GlobalFilterBar`` / trends fallback when no custom ingest path is tracked (§G.6 / §G.16 / §D.7)
- [x] §G.6 OLAP Explorer built-in DuckDB portfolio table run-label brush + ring-highlight parity across all analysis views (§D.7)

**Delivered (§D.7 — hundred-seventy-ninth pass)**

- [x] ``annotateTableWithRunLabelIfMissing`` — single-log ``runSimulationArrowPipeline`` / ``runCsvArrowPipeline`` DuckDB tables gain ``run_label`` + ``city_scale`` when absent (portfolio ingest parity; §G.6 / §D.7)
- [x] ``pathRunLabelBrushActive`` / ``useRunLabelBrushToggle`` / ``PathRunLabelChip`` — shared path-chip ring-highlight + click-to-brush helpers (§G.14–§G.16 / §D.7)
- [x] Simulation Monitor — watch-path ``PathRunLabelChip`` + ``monitor_sim`` ``SqlQueryPanel`` ``brushSqlSync`` run-label parity (§G.16 / §D.7)
- [x] Algorithm Comparison — watch-path ``PathRunLabelChip`` + ``algorithm_sim`` ``SqlQueryPanel`` run-label brush sync (§G.1 / §G.16 / §D.7)
- [x] Data Explorer — open-file ``PathRunLabelChip`` + ``useTableRunLabelBrush`` on ``explorer_csv`` when CSV lacks ``run_label`` column (§G.6 / §G.16 / §D.7)
- [x] §G.14–§G.16 file-path run-label brush + ring-highlight parity across Monitor / Algorithm Comparison / Data Explorer (§D.7)

**Delivered (§D.7 — hundred-eightieth pass)**

- [x] Simulation Summary — open-log ``PathRunLabelChip`` ring-highlight + click-to-brush parity with Simulation Monitor (§G.1 / §G.14 / §D.7)
- [x] Output Browser — selected-run + open-jsonl ``PathRunLabelChip`` ring-highlight + click-to-brush parity with run list (§G.14 / §D.7)
- [x] OLAP Explorer — custom-ingest ``PathRunLabelChip`` ring-highlight + click-to-brush parity with ingested-table picker (§G.6 / §G.14 / §D.7)
- [x] §G.14–§G.16 file-path run-label brush + ring-highlight parity across all file-based analysis views (§D.7)

**Delivered (§D.7 — hundred-eighty-first pass)**

- [x] ``LoadedRunRow`` — shared portfolio loaded-run row wrapping ``PathRunLabelChip`` with optional remove, leading slots, and trailing metadata (§G.1 / §G.14 / §D.7)
- [x] Benchmark Analysis — loaded-run list ``LoadedRunRow`` replaces inline font-mono brush buttons (§G.1 / §G.6 / §D.7)
- [x] City Comparison — loaded-run list ``LoadedRunRow`` parity with Benchmark Analysis (§G.1.6 / §G.6 / §D.7)
- [x] Simulation Summary — comparison-run list ``LoadedRunRow`` parity with portfolio analytics pages (§G.1 / §G.6 / §D.7)
- [x] Output Browser — run-directory list ``LoadedRunRow`` with compare checkbox + folder select leading slots; chip click-to-brush parity with header chips (§G.14 / §D.7)
- [x] §G.1 / §G.6 / §G.14 portfolio loaded-run list path-chip run-label brush + ring-highlight parity across all analysis views (§D.7)

**Delivered (§D.7 — hundred-eighty-second pass)**

- [x] ``brushLogPathFromProcessLines`` — resolve ``.jsonl`` / Lightning ``logs/`` / ``assets/output`` paths from process stdout for header chip brush (§G.9–§G.18 / §D.7)
- [x] ``RunLabelHeaderSuffix`` — shared inline header suffix rendering ``PathRunLabelChip`` when ``logPath`` known, else plain ``· runLabel`` text (§G.9–§G.18 / §D.7)
- [x] ``LauncherLivePanelHeader`` + ``TrainHpoLivePanelHeader`` — optional ``logPath`` prop replaces plain run-label suffix with ``PathRunLabelChip`` ring-highlight + click-to-brush (§G.9–§G.18 / §G.15 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner + Training Hub — live panel headers pass ``logPath`` from process stdout (§G.9–§G.12 / §D.7)
- [x] Process Monitor + Training Monitor + HPO Tracker + Experiment Tracker — live/post-run panel headers pass ``logPath`` from selected/recent process stdout (§G.15 / §G.17 / §G.18 / §D.7)
- [x] Simulation Summary — ``ConfigMetaBanner`` run path uses ``PathRunLabelChip`` instead of plain font-mono text (§G.1 / §G.14 / §D.7)
- [x] §G.9–§G.18 launcher/monitor card header path-chip run-label brush + ring-highlight parity across all live workflow pages (§D.7)

**Delivered (§D.7 — hundred-eighty-third pass)**

- [x] ``processLogPathKind`` / ``brushLogPathMapFromProcesses`` — derive per-process log/run paths from stdout for row + footer path-chip brush (§G.15 / §D.7)
- [x] Process Monitor — process list rows render ``PathRunLabelChip`` when stdout resolves a log path; ring-highlight parity preserved (§G.15 / §D.7)
- [x] ``ProcessIdFooter`` — optional ``logPath`` prop renders ``PathRunLabelChip`` with muted process-id suffix on launcher/monitor live panels (§G.9–§G.18 / §D.7)
- [x] Simulation Launcher + Data Generation + Evaluation Runner + Training Hub + Training Monitor + HPO Tracker + Experiment Tracker — live panel footers pass ``logPath`` from process stdout (§G.9–§G.18 / §D.7)
- [x] Command Palette — recent log/run/csv entries use ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.7 / §D.7)
- [x] §G.15 Process Monitor process-row path-chip run-label brush + ring-highlight parity across all workflow kinds (§D.7)

**Delivered (§D.7 — hundred-eighty-fourth pass)**

- [x] Training Monitor — run discovery list uses ``LoadedRunRow`` + ``PathRunLabelChip`` instead of inline font-mono run names; ring-highlight parity via ``activeRunLabel`` (§G.17 / §D.7)
- [x] Training Monitor — ``RunPanel`` per-run header uses ``PathRunLabelChip`` instead of plain font-mono text (§G.17 / §D.7)
- [x] Training Monitor — ``GlobalFilterBar`` ``runLabels`` from selected Lightning log paths when no live process brush is active (§G.17 / §D.7)
- [x] Process Monitor — process list rows show muted process-id suffix alongside ``PathRunLabelChip`` when stdout resolves a log path; footer parity (§G.15 / §D.7)
- [x] §G.17 Training Monitor run-discovery list path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-eighty-fifth pass)**

- [x] ``localPathFromUri`` / ``mlflowRunDirFromArtifactUri`` — resolve MLflow ``artifact_uri`` to local run directory for path-chip brush (§G.18 / §D.7)
- [x] ``PathRunLabelChip`` — optional ``label`` prop overrides brush + display text; ``LoadedRunRow`` passes ``label`` through to chip (§G.1 / §G.14 / §D.7)
- [x] Experiment Tracker — MLflow run table rows render ``PathRunLabelChip`` when ``artifact_uri`` resolves a local path; muted run-id suffix parity with Process Monitor (§G.18 / §D.7)
- [x] Experiment Tracker — output directory list ``LoadedRunRow`` + ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.18 / §G.14 / §D.7)
- [x] Experiment Tracker — ``GlobalFilterBar`` ``runLabels`` from selected MLflow runs when no live process brush is active (§G.18 / §D.7)
- [x] §G.18 Experiment Tracker MLflow + output-dir path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-eighty-sixth pass)**

- [x] ``trialLogDirFromUserAttrs`` / ``sqlitePathFromStorageUrl`` — resolve Optuna trial ``log_dir`` user attribute and local SQLite storage path for path-chip brush (§G.18 / §D.7)
- [x] ``HpoHealthMetricsCallback`` — persist ``log_dir`` on Optuna trial user attributes from Lightning ``trainer.log_dir`` (§G.18 / §A.4 / §D.7)
- [x] HPO Tracker — trial health table rows render ``PathRunLabelChip`` when trial ``log_dir`` is known; muted trial-number suffix parity with Process Monitor (§G.18 / §D.7)
- [x] HPO Tracker — Optuna storage DB + exported Plotly report directory ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.18 / §D.7)
- [x] HPO Tracker — ``GlobalFilterBar`` ``runLabels`` from selected trials, post-run ``trainingRunPath`` / ``outputRunPath``, or live process brush (§G.18 / §D.7)
- [x] §G.18 HPO Tracker trial-table + storage/report path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-eighty-seventh pass)**

- [x] ``PathRunLabelChip`` — optional ``brushLabel`` prop decouples display text from brush run_label when path stem differs (§G.17 / §G.13 / §G.5 / §D.7)
- [x] Training Monitor — Lightning ``logs/`` root ``PathRunLabelChip`` in controls + empty-state banner (§G.17 / §D.7)
- [x] Training Monitor — checkpoint browser rows render ``PathRunLabelChip`` with parent-run brush label + checkpoint filename display (§G.17 / §G.12 / §D.7)
- [x] Configuration Editor — open-file ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on primary YAML path (§G.13 / §D.7)
- [x] ML Introspection — tensor archive ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on open ``.npz`` / ``.npy`` / ``.td`` path (§G.5 / §D.7)
- [x] §G.17 / §G.13 / §G.5 file-based workflow path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-eighty-eighth pass)**

- [x] Output Browser — checkpoint sidebar rows render ``PathRunLabelChip`` with parent-run ``brushLabel`` parity with Training Monitor (§G.14 / §G.12 / §D.7)
- [x] Output Browser — file viewer header + checkpoint preview use ``PathRunLabelChip`` for all artefact paths (not only ``.jsonl``); checkpoint files brush parent run (§G.14 / §D.7)
- [x] Output Browser — ``useLogPathRunLabelBrush`` derives label from ``runJsonlPath ?? selectedRun.path`` (§G.14 / §D.7)
- [x] Evaluation Runner — checkpoint list rows show ``PathRunLabelChip`` when path is set; ring-highlight + click-to-brush parity (§G.12 / §D.7)
- [x] Training Hub — eval-mode checkpoint path ``PathRunLabelChip`` below input when path is set (§G.10 / §G.12 / §D.7)
- [x] Configuration Editor — diff comparison file ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on ``diffPath``; diff summary uses chips for both files (§G.13 / §D.7)
- [x] §G.14 / §G.12 / §G.10 eval checkpoint path-chip run-label brush + ring-highlight parity across Output Browser, Eval Runner, and Training Hub checked (§D.7)

**Delivered (§D.7 — hundred-eighty-ninth pass)**

- [x] ``parentRunBrushLabelFromCheckpointPath`` — shared helper deriving parent-run brush label from ``checkpoints/`` path segments (§G.12 / §G.14 / §G.17 / §D.7)
- [x] ``EvalResult`` / ``EvalAnalyticsRow`` — optional ``checkpointPath`` field propagated from Hydra eval command via ``checkpointPathFromEvalCommand`` (§G.12 / §G.1 / §D.7)
- [x] Evaluation Runner — post-eval results table renders ``PathRunLabelChip`` when checkpoint path is known; parent-run ``brushLabel`` parity with input rows (§G.12 / §D.7)
- [x] Benchmark Analysis — eval results panel checkpoint column ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.1 / §G.12 / §D.7)
- [x] ``EvalResultCard`` — checkpoint header chip on Process Monitor + Training Hub eval panels when path known (§G.10 / §G.12 / §G.15 / §D.7)
- [x] Output Browser — ``.wsroute`` manifest file table rows use ``PathRunLabelChip`` with selected-run brush label (§G.8 / §G.14 / §D.7)
- [x] Output Browser — ``.wsroute`` manifest member paths gain auto-classified ``PathRunLabelChip`` handoffs (two-hundred-and-thirty-second pass; §G.8 / §G.14 / §D.7)
- [x] Output Browser + Training Monitor — checkpoint ``brushLabel`` uses shared ``parentRunBrushLabelFromCheckpointPath`` helper (§G.14 / §G.17 / §D.7)
- [x] §G.12 / §G.1 / §G.8 eval-results + bundle-manifest path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninetieth pass)**

- [x] ``EvalCheckpointLiveCard`` — optional ``checkpointPath`` prop renders ``PathRunLabelChip`` with parent-run ``brushLabel`` on live eval rows (§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner + Training Hub + Process Monitor — live eval cards pass Hydra checkpoint path to ``EvalCheckpointLiveCard`` (§G.12 / §G.10 / §G.15 / §D.7)
- [x] Evaluation Runner + Training Hub — eval dataset path ``PathRunLabelChip`` below filled dataset inputs (§G.12 / §G.10 / §D.7)
- [x] Evaluation Runner + Training Hub — checkpoint input chips use ``parentRunBrushLabelFromCheckpointPath`` ``brushLabel`` parity with results table (§G.12 / §G.10 / §D.7)
- [x] Data Generation Wizard — TSPLIB instance + sensor CSV source path ``PathRunLabelChip`` ring-highlight + click-to-brush parity (§G.11 / §D.7)
- [x] ``PolicyTelemetryTrendsPanel`` — SQLite ``db_path`` header uses ``PathRunLabelChip`` instead of plain font-mono text (§G.7 / §A.3 / §D.7)
- [x] ``PolicyTelemetryTrendsPanel`` — SQLite ``db_path`` header migrates to ``OpenPathToolbar`` (two-hundred-and-thirtieth pass; §A.3 / §D.7)
- [x] §G.12 / §G.10 / §G.11 / §G.15 live-eval + dataset + data-gen source path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-first pass)**

- [x] ``resolveLocalProjectPath`` — resolve MLflow tracking URI / relative paths against ``projectRoot`` for path-chip brush (§G.18 / §G.19 / §D.7)
- [x] Settings — project root + Python executable ``PathRunLabelChip`` below filled path inputs (§G.19 / §D.7)
- [x] Experiment Tracker — MLflow tracking URI ``PathRunLabelChip`` below filled tracking URI when local path resolves (§G.18 / §D.7)
- [x] HPO Tracker — Optuna storage URL ``PathRunLabelChip`` below filled input; inline chip parity with eval dataset inputs (§G.18 / §D.7)
- [x] §G.18 / §G.19 Settings + tracker storage/tracking URI path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-second pass)**

- [x] ``sqliteStoragePathFromUrl`` — resolve Optuna ``sqlite:///`` storage URL against ``projectRoot`` for path-chip brush (§G.18 / §G.19 / §D.7)
- [x] HPO Tracker — storage DB + exported report directory ``PathRunLabelChip`` use ``projectRoot``-resolved absolute paths (§G.18 / §D.7)
- [x] Data Generation Wizard — instance preview ``.pkl`` / ``.pt`` path ``PathRunLabelChip`` below preview panel (§G.11 / §D.7)
- [x] Settings — Arrow pipeline benchmark + import-settings JSON ``PathRunLabelChip`` below filled paths (§G.19 / §D.7)
- [x] ``PolicyTelemetryTrendsPanel`` — SQLite ``db_path`` resolved against ``projectRoot`` before path-chip brush (§G.7 / §A.3 / §D.7)
- [x] §G.18 / §G.11 / §G.19 relative-path storage/preview/import path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-third pass)**

- [x] ``PathRunLabelChip`` — optional ``projectRoot`` prop resolves relative paths via ``resolveLocalProjectPath`` before brush + tooltip (§G.10–§G.13 / §D.7)
- [x] ``parentRunBrushLabelFromCheckpointPath`` — optional ``projectRoot`` resolves checkpoint paths before parent-run brush label derivation (§G.12 / §G.14 / §G.17 / §D.7)
- [x] Evaluation Runner — checkpoint list, dataset input, results table, and live eval cards use ``projectRoot``-resolved path chips (§G.12 / §D.7)
- [x] Training Hub — eval checkpoint + dataset path chips use ``projectRoot`` resolution (§G.10 / §G.12 / §D.7)
- [x] Data Generation Wizard — TSPLIB/sensor source + instance preview path chips use ``projectRoot`` resolution (§G.11 / §D.7)
- [x] Configuration Editor — open YAML + diff comparison path chips use ``projectRoot`` resolution (§G.13 / §D.7)
- [x] ML Introspection — tensor archive path chip uses ``projectRoot`` resolution (§G.5 / §D.7)
- [x] Benchmark Analysis + Process Monitor — eval results / live eval cards use ``projectRoot``-resolved checkpoint path chips (§G.1 / §G.12 / §G.15 / §D.7)
- [x] Settings — Python executable + import JSON + Arrow benchmark path chips resolve against draft project root (§G.19 / §D.7)
- [x] §G.10 / §G.11 / §G.12 / §G.13 launcher + workflow relative-path path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-fourth pass)**

- [x] ``PathRunLabelChip`` — falls back to ``useAppStore`` ``projectRoot`` when prop omitted; analysis/monitor/file chips auto-resolve relative paths (§G.1 / §G.14–§G.18 / §D.7)
- [x] ``RunLabelHeaderSuffix`` — optional ``projectRoot`` prop; inherits store fallback via ``PathRunLabelChip`` (§G.9–§G.18 / §D.7)
- [x] HPO Tracker — trial ``log_dir`` user-attribute paths resolved against ``projectRoot`` before path-chip brush (§G.18 / §D.7)
- [x] Experiment Tracker — MLflow ``artifact_uri`` run directories resolved against ``projectRoot`` before path-chip brush (§G.18 / §D.7)
- [x] Training Monitor — logs root, run-discovery list, per-run headers, and checkpoint browser use ``projectRoot``-resolved path chips (§G.17 / §G.12 / §D.7)
- [x] Output Browser — selected-run, checkpoint sidebar, file viewer, checkpoint preview, and ``.wsroute`` manifest rows use ``projectRoot``-resolved path chips (§G.14 / §G.8 / §G.12 / §D.7)
- [x] §G.14 / §G.17 / §G.18 analysis + monitor + file browser relative-path path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-fifth pass)**

- [x] ``useLogPathRunLabelBrush`` — resolves log/run paths against ``useAppStore`` ``projectRoot`` before global ``run_label`` brush sync (§G.1 / §G.14–§G.16 / §D.7)
- [x] ``LoadedRunRow`` — optional ``projectRoot`` prop + store fallback; portfolio ring-highlight compares resolved run labels (§G.1 / §G.14 / §G.17 / §D.7)
- [x] Simulation Summary + Benchmark Analysis + City Comparison + Experiment Tracker + Training Monitor + Output Browser — ``LoadedRunRow`` passes ``projectRoot`` for portfolio/run-list path-chip brush parity (§G.1 / §G.14 / §G.17 / §G.18 / §D.7)
- [x] Simulation Summary + Data Explorer + OLAP Explorer + Algorithm Comparison + Simulation Monitor — open-file ``PathRunLabelChip`` headers pass ``projectRoot`` (§G.1 / §G.6 / §G.16 / §G.15 / §D.7)
- [x] Process Monitor — process-row ``PathRunLabelChip`` passes ``projectRoot`` for stdout-resolved log paths (§G.15 / §D.7)
- [x] Training Monitor — ``GlobalFilterBar`` ``runLabels`` derived from ``projectRoot``-resolved Lightning log paths (§G.17 / §D.7)
- [x] Output Browser — ``parentRunBrushLabel`` resolves selected-run path against ``projectRoot`` before manifest brush (§G.14 / §G.8 / §D.7)
- [x] §G.1 / §G.6 / §G.14–§G.17 portfolio + open-file relative-path path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-sixth pass)**

- [x] ``runLabelFromLogLines`` / ``runLabelMapFromProcesses`` / ``runLabelMapFromPaths`` / ``runLabelMapFromTablePaths`` / ``pathRunLabelBrushActive`` — resolve paths against ``projectRoot`` before ``run_label`` derivation (§G.7 / §G.15 / §G.16 / §D.7)
- [x] ``useProcessRunLabelBrush`` — resolves stdout log paths against ``useAppStore`` ``projectRoot`` before global ``run_label`` brush sync (§G.9–§G.18 / §D.7)
- [x] Process Monitor — ``runLabelMapFromProcesses`` passes ``projectRoot`` for process-row ring-highlight parity (§G.15 / §D.7)
- [x] OLAP Explorer — ``runLabelMapFromTablePaths`` passes ``projectRoot`` for ingest-table picker ring-highlight parity (§G.6 / §D.7)
- [x] HPO Tracker — ``GlobalFilterBar`` ``runLabels`` from ``projectRoot``-resolved post-run ``trainingRunPath`` / ``outputRunPath`` (§G.18 / §D.7)
- [x] Command Palette — recent log/run/csv entries pass ``projectRoot`` for path-chip brush parity (§G.7 / §D.7)
- [x] ``ProcessIdFooter`` + ``LauncherLivePanelHeader`` + ``TrainHpoLivePanelHeader`` — optional ``projectRoot`` prop + store fallback on live-panel log-path suffix chips (§G.9–§G.18 / §D.7)
- [x] §G.7 / §G.9–§G.18 derived run-label + live-panel relative-path path-chip run-label brush + ring-highlight parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-seventh pass)**

- [x] ``runLabelFromSourcePath`` — resolve paths against ``projectRoot`` before ``run_label`` derivation (§G.0 / §G.6 / §G.7 / §D.7)
- [x] ``annotateTableWithRunLabelIfMissing`` + ``runCsvArrowPipeline`` / ``runSimulationArrowPipeline`` — optional ``projectRoot`` resolves ingest source paths before DuckDB ``run_label`` annotation (§G.0 / §G.6 / §D.7)
- [x] OLAP Explorer + Data Explorer + Algorithm Comparison + Simulation Monitor + Settings — Arrow pipeline callers pass ``projectRoot`` for single-log DuckDB ingest (§G.0 / §G.6 / §G.16 / §G.19 / §D.7)
- [x] Simulation Summary — portfolio DuckDB log labels derived via ``runLabelFromSourcePath`` for brush/SQL parity (§G.1 / §G.6 / §D.7)
- [x] ``PolicyTelemetryTrendsPanel`` — telemetry ``db_path`` ``PathRunLabelChip`` passes ``projectRoot`` (§A.3 / §G.7 / §D.7)
- [x] Launcher + train/HPO live panels — explicit ``projectRoot`` on ``LauncherLivePanelHeader`` / ``TrainHpoLivePanelHeader`` + ``ProcessIdFooter`` across Simulation Launcher, Data Generation, Evaluation Runner, Training Hub, Process Monitor, Training Monitor, HPO Tracker, and Experiment Tracker (§G.9–§G.18 / §D.7)
- [x] §G.0 / §G.6 DuckDB ingest + §G.9–§G.18 live-panel explicit ``projectRoot`` path-chip run-label brush parity checked (§D.7)

**Delivered (§D.7 — hundred-ninety-eighth pass)**

- [x] ``portfolioRunLabel`` + ``runPortfolioSimulationArrowPipeline`` — optional ``projectRoot`` resolves portfolio DuckDB ``run_label`` columns via ``runLabelFromSourcePath`` (§G.0 / §G.1 / §G.6 / §D.7)
- [x] Benchmark Analysis + City Comparison — portfolio load/add-run labels derived via ``portfolioRunLabel``; DuckDB ingest passes ``projectRoot`` (§G.1 / §G.1.6 / §D.7)
- [x] Simulation Summary + OLAP Explorer — portfolio DuckDB pipeline callers pass ``projectRoot`` for multi-log union ingest (§G.1 / §G.6 / §D.7)
- [x] §G.1 / §G.1.6 portfolio DuckDB ``run_label`` relative-path brush/SQL parity across Benchmark Analysis, City Comparison, Simulation Summary, and OLAP Explorer checked (§D.7)

**Delivered (§D.7 — hundred-ninety-ninth pass)**

- [x] Simulation Summary — ``portfolioRunLabel`` on add-comparison-run, output-portfolio load, ``allRuns`` portfolio brush, and ``allDuckDbLogs`` union ingest for UI/DuckDB ``run_label`` parity (§G.1 / §G.6 / §D.7)
- [x] §G.1 Simulation Summary portfolio loaded-run list + filter-bar relative-path ``run_label`` brush/SQL parity checked (§D.7)

**Delivered (§D.7 — two-hundredth pass)**

- [x] Benchmark Analysis — ``normalizedRuns`` + ``portfolioDuckDbLogs`` derive labels via ``portfolioRunLabel`` for loaded-run list, ``filteredRuns`` portfolio brush, and DuckDB union ingest (§G.1 / §G.6 / §D.7)
- [x] City Comparison — ``normalizedRuns`` + ``portfolioDuckDbLogs`` derive labels via ``portfolioRunLabel`` for loaded-run list, ``filteredRuns`` portfolio brush, and DuckDB union ingest (§G.1.6 / §G.6 / §D.7)
- [x] OLAP Explorer — custom JSONL ingest uses ``portfolioRunLabel`` for ingest path label parity with DuckDB ``run_label`` (§G.6 / §D.7)
- [x] Simulation Summary — comparison-run ``LoadedRunRow`` labels re-derived via ``portfolioRunLabel`` when ``projectRoot`` changes (§G.1 / §D.7)
- [x] §G.1 / §G.1.6 Benchmark Analysis + City Comparison portfolio loaded-run list + filter-bar relative-path ``run_label`` brush/SQL parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-first pass)**

- [x] Data Explorer — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push when CSV lacks ``run_label`` column (§G.6 / §D.7)
- [x] Algorithm Comparison — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, and Policy Telemetry Trends (§G.16 / §D.7)
- [x] Simulation Monitor — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push (§G.16 / §D.7)
- [x] Simulation Summary — recent-file push label via ``portfolioRunLabel`` for Command Palette brush parity (§G.1 / §D.7)
- [x] §G.6 / §G.16 single-log open-file relative-path ``run_label`` brush/SQL parity across Data Explorer, Algorithm Comparison, and Simulation Monitor checked (§D.7)

**Delivered (§D.7 — two-hundred-and-second pass)**

- [x] Output Browser — ``sourceRunLabel`` via ``portfolioRunLabel`` on run select, Policy Telemetry Trends scoping, and ``.wsroute`` manifest path-chip brush; ``pushRecent`` uses ``portfolioRunLabel`` for Command Palette parity (§G.14 / §D.7)
- [x] OLAP Explorer — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, and Policy Telemetry Trends when custom ingest lacks portfolio ``run_label`` column; ``pushRecent`` on ingest (§G.6 / §D.7)
- [x] Benchmark Analysis + City Comparison — ``pushRecent`` on add-run via ``portfolioRunLabel`` for Command Palette parity (§G.1 / §G.1.6 / §D.7)
- [x] §G.14 Output Browser + §G.6 OLAP Explorer single-log / run-directory relative-path ``run_label`` brush/SQL + recent-file parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-third pass)**

- [x] Output Browser — ``compareSelectedRuns`` Benchmark handoff refs use ``portfolioRunLabel``; ``pushRecent`` on multi-run compare for Command Palette parity (§G.14 / §G.1 / §D.7)
- [x] Simulation Summary — ``pushRecent`` on add-comparison-run via ``portfolioRunLabel`` (§G.1 / §D.7)
- [x] Benchmark Analysis + City Comparison — ``pushRecent`` on ``pendingBenchmarkLogs`` consume for Output Browser compare handoff parity (§G.1 / §G.1.6 / §D.7)
- [x] §G.14 Output Browser compare + §G.1 / §G.1.6 portfolio handoff recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-fourth pass)**

- [x] Output Browser — ``openInSimSummary`` + ``extractBundleAndOpen`` ``pushRecent`` via ``portfolioRunLabel`` on Simulation Summary handoff (§G.14 / §G.1 / §D.7)
- [x] ``useGlobalFileDrop`` + ``useWsrouteImport`` — ``pushRecent`` via ``portfolioRunLabel`` on dropped/imported ``.jsonl`` / ``.wsroute`` log handoff (§G.8 / §G.14 / §D.7)
- [x] Benchmark Analysis + City Comparison + Simulation Summary — ``pushRecent`` on ``loadOutputPortfolio`` for each scanned log via ``portfolioRunLabel`` (§G.1 / §G.1.6 / §D.7)
- [x] Command Palette — refresh recent log/run/csv labels via ``portfolioRunLabel`` on open for ``projectRoot`` brush parity (§G.7 / §D.7)
- [x] §G.1 / §G.1.6 / §G.8 / §G.14 portfolio load + bundle/drop handoff recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-fifth pass)**

- [x] ``refreshRecentLabels`` — re-derive persisted recent log/run/csv labels via ``portfolioRunLabel`` when Command Palette opens or ``projectRoot`` changes (§G.7 / §D.7)
- [x] Command Palette — ``refreshRecentLabels`` on palette open; ``pendingCsvPath`` handoff for recent CSV entries (§G.7 / §G.6 / §D.7)
- [x] Data Explorer — consume ``pendingCsvPath`` on mount for Command Palette CSV recent-file parity (§G.6 / §D.7)
- [x] Output Browser — ``pushRecent`` via ``portfolioRunLabel`` on inline ``.jsonl`` / ``.csv`` file open in run tree viewer (§G.14 / §G.8 / §D.7)
- [x] §G.7 / §G.14 recent-file label refresh + inline file-open ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-sixth pass)**

- [x] ``RecentFileKind`` — ``training`` kind for Lightning log directories alongside log/run/csv (§G.17 / §G.7 / §D.7)
- [x] Training Monitor — ``pushRecent`` via ``portfolioRunLabel`` on run select, ``pendingTrainingRunPath`` consume, and post-run auto-select; filter-bar labels use ``portfolioRunLabel`` (§G.17 / §D.7)
- [x] Command Palette — open recent training runs via ``pendingTrainingRunPath`` + Training Monitor mode (§G.7 / §G.17 / §D.7)
- [x] ``LauncherNavMesh`` + ``TrainHpoNavMesh`` — Output Browser / Training Monitor handoff ``pushRecent`` via ``portfolioRunLabel`` (§G.9–§G.12 / §G.15 / §G.17 / §D.7)
- [x] §G.17 / §G.7 Training Monitor + nav-mesh recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-seventh pass)**

- [x] ``RecentFileKind`` — ``checkpoint`` kind for ``.pt`` / ``.ckpt`` / ``.pth`` model files alongside log/run/csv/training (§G.12 / §G.7 / §D.7)
- [x] Training Monitor — checkpoint browser ``handleLoadCheckpoint`` ``pushRecent`` via ``portfolioRunLabel`` before ``pendingCheckpoint`` handoff (§G.17 / §G.12 / §D.7)
- [x] Output Browser — ``loadInEvalRunner`` checkpoint handoff ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.12 / §D.7)
- [x] ``LauncherNavMesh`` — post-run ``Load in Eval Runner`` ``pushRecent`` via ``portfolioRunLabel`` (§G.9–§G.12 / §G.15 / §D.7)
- [x] Evaluation Runner — ``pendingCheckpoint`` consume + file-picker ``pushRecent`` via ``portfolioRunLabel`` (§G.12 / §D.7)
- [x] Training Hub — eval-mode checkpoint file picker ``pushRecent`` via ``portfolioRunLabel`` (§G.10 / §G.12 / §D.7)
- [x] Command Palette — open recent checkpoints via ``pendingCheckpoint`` + Evaluation Runner mode (§G.7 / §G.12 / §D.7)
- [x] §G.12 / §G.7 eval checkpoint recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-eighth pass)**

- [x] ``RecentFileKind`` — ``config`` kind for YAML / TOML / cfg / ini config files alongside log/run/csv/training/checkpoint (§G.13 / §G.7 / §D.7)
- [x] ``pendingConfigPath`` — Config Editor deep-link from Output Browser and Command Palette (§G.13 / §G.14 / §G.7 / §D.7)
- [x] Configuration Editor — file picker + ``pendingConfigPath`` consume ``pushRecent`` via ``portfolioRunLabel`` (§G.13 / §D.7)
- [x] Output Browser — inline config open ``pushRecent`` + **Open in Config Editor →** handoff via ``pendingConfigPath`` (§G.14 / §G.13 / §D.7)
- [x] Command Palette — open recent configs via ``pendingConfigPath`` + Config Editor mode (§G.7 / §G.13 / §D.7)
- [x] §G.13 / §G.7 / §G.14 config recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-ninth pass)**

- [x] ``recentKindFromPath`` — shared path → recent-file kind classifier for drop/open handoff (§G.7 / §G.8 / §D.7)
- [x] ``useGlobalFileDrop`` — multi-kind drop routes ``.csv`` / checkpoint / config through ``portfolioRunLabel`` + ``pendingCsvPath`` / ``pendingCheckpoint`` / ``pendingConfigPath`` (§G.8 / §G.6 / §G.12 / §G.13 / §D.7)
- [x] Output Browser — inline checkpoint open ``pushRecent`` via ``portfolioRunLabel`` (§G.14 / §G.12 / §D.7)
- [x] Output Browser — **Open in Data Explorer →** CSV handoff via ``pendingCsvPath`` + ``pushRecent`` (§G.14 / §G.6 / §D.7)
- [x] §G.8 / §G.14 global drop + Output Browser multi-kind recent-file ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-tenth pass)**

- [x] ``recentHandoff.ts`` — shared kind → mode / pending-path / toast handoff + ``makeRecentEntry`` (§G.7 / §G.8 / §D.7)
- [x] ``recentKindFromPath`` — training ``logs/`` + ``assets/output`` run directory heuristics (§G.8 / §G.14 / §G.17 / §D.7)
- [x] Command Palette — unified recent open via shared handoff; keyboard nav over recents + commands (§G.7 / §D.7)
- [x] ``useGlobalFileDrop`` — multi-path classify/push + directory drop + highest-priority handoff (§G.8 / §G.6 / §G.12 / §G.14 / §G.17 / §D.7)
- [x] Output Browser — directory picker + inline open via ``recentKindFromPath`` / ``makeRecentEntry`` (§G.14 / §D.7)
- [x] Evaluation Runner + Training Hub — CSV dataset pick ``pushRecent`` for Data Explorer reopen (§G.12 / §G.10 / §G.6 / §D.7)
- [x] §G.7 / §G.8 / §G.14 shared recent-file handoff + directory drop ``portfolioRunLabel`` parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-eleventh pass)**

- [x] ``applyRecentHandoff`` — shared push + pending-path + mode navigation for kind handoffs (§G.7 / §G.8 / §D.7)
- [x] ``LauncherNavMesh`` + ``TrainHpoNavMesh`` — post-run Output Browser / Training Monitor / eval checkpoint handoffs via ``applyRecentHandoff`` (§G.9–§G.12 / §G.15 / §G.17 / §D.7)
- [x] Output Browser — ``openInSimSummary`` / ``loadInEvalRunner`` / ``openInConfigEditor`` / ``openInDataExplorer`` / bundle extract via ``applyRecentHandoff`` (§G.14 / §D.7)
- [x] Command Palette + global drop + wsroute import + Training Monitor checkpoint load via ``applyRecentHandoff`` / ``makeRecentEntry`` (§G.7 / §G.8 / §G.17 / §D.7)
- [x] All page-level ``pushRecent`` call sites migrated to ``makeRecentEntry`` (analytics, launchers, monitors, Config Editor) (§G.1 / §G.6 / §G.10 / §G.12 / §G.13 / §G.16 / §D.7)
- [x] §G.7 / §G.8 / §G.14 studio-wide ``makeRecentEntry`` / ``applyRecentHandoff`` recent-file parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twelfth pass)**

- [x] ``useRecentHandoff`` / ``useRecentPendingSetters`` — shared React hook for pending-path setters + ``handoff(path, kind)`` (§G.7 / §G.8 / §D.7)
- [x] ``LauncherNavMesh`` ``simLogPath`` — post-run **Simulation Summary →** hands off ``.jsonl`` via ``pendingLogPath`` + ``makeRecentEntry`` (§G.9 / §G.1 / §D.7)
- [x] Simulation Launcher — auto-Summary countdown and nav mesh use ``simJsonlPath`` handoff (§G.9 / §G.1 / §D.7)
- [x] Process Monitor — sim live panel ``simLogPath`` from stdout ``.jsonl`` (§G.15 / §G.9 / §D.7)
- [x] Nav meshes, Command Palette, global drop, wsroute import, Output Browser, Training Monitor checkpoint load migrate to ``useRecentHandoff`` (§G.7–§G.15 / §G.17 / §D.7)
- [x] §G.9 / §G.1 post-run Simulation Summary log handoff + shared hook parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirteenth pass)**

- [x] ``applyRecentHandoff`` / ``handoff`` optional ``mode`` override — open logs in Simulation Monitor (Digital Twin) without changing ``pendingLogPath`` (§G.7 / §G.16 / §D.7)
- [x] ``LauncherNavMesh`` **Simulation Monitor →** hands off ``simLogPath`` with ``mode: "simulation"`` (§G.9 / §G.16 / §D.7)
- [x] Command Palette log open — shared handoff; Summary on successful load, Monitor fallback via mode override (§G.7 / §G.1 / §G.16 / §D.7)
- [x] Local open / multi-select call sites migrate to ``handoff(…, { navigate: false })`` (launchers, monitors, analytics, Output Browser, global drop) (§G.1 / §G.6 / §G.10–§G.17 / §D.7)
- [x] §G.16 / §G.9 post-run Simulation Monitor log handoff + studio-wide ``navigate: false`` recent-file parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-fourteenth pass)**

- [x] Output Browser — **Open in Simulation Monitor →** for ``.jsonl`` via ``handoff(…, { mode: "simulation" })`` (§G.14 / §G.16 / §D.7)
- [x] Simulation Summary — **Simulation Monitor →** hands off the open log into Digital Twin (§G.1 / §G.16 / §D.7)
- [x] Algorithm Comparison — **Compare on Map** hands off ``watchPath`` with mode override so Digital Twin reloads + recents stay in sync (§G.16 / §D.7)
- [x] §G.14 / §G.1 / §G.16 analytics + Output Browser Simulation Monitor log-handoff mode-override parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-fifteenth pass)**

- [x] ``LoadedRunRow`` — optional ``logHandoffs`` Summary + Simulation Monitor buttons via shared mode override (§G.1 / §G.16 / §D.7)
- [x] Benchmark Analysis — portfolio loaded-run rows open Summary / Digital Twin (§G.1 / §D.7)
- [x] City Comparison — portfolio loaded-run rows open Summary / Digital Twin (§G.1.6 / §D.7)
- [x] Simulation Summary — comparison-run rows open Summary / Digital Twin (§G.1 / §D.7)
- [x] §G.1 / §G.1.6 portfolio loaded-run log-handoff mode-override parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-sixteenth pass)**

- [x] Simulation Monitor — **Simulation Summary →** hands off the open log for reverse open-log parity with Summary's Monitor button (§G.16 / §G.1 / §D.7)
- [x] Algorithm Comparison — **Simulation Summary →** hands off ``watchPath`` alongside **Compare on Map** (§G.1 / §D.7)
- [x] §G.16 / §G.1 open-log Simulation Summary handoff parity across Digital Twin + Algorithm Comparison checked (§D.7)

**Delivered (§D.7 — two-hundred-and-seventeenth pass)**

- [x] ``LogHandoffButtons`` / ``isSimulationLogPath`` — shared Summary + Simulation Monitor log handoff controls (icon + labeled modes) (§G.1 / §G.16 / §D.7)
- [x] ``LoadedRunRow`` — ``logHandoffs`` uses shared ``LogHandoffButtons`` (§G.1 / §D.7)
- [x] OLAP Explorer — labeled Summary / Monitor handoffs when the selected ingest path is ``.jsonl`` (§G.6 / §G.1 / §G.16 / §D.7)
- [x] §G.6 / §G.1 / §G.16 shared log-handoff control + OLAP JSONL ingest path parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-eighteenth pass)**

- [x] ``LogHandoffButtons`` ``targets`` prop — single-direction Summary-only or Monitor-only toolbars (§G.1 / §G.16 / §D.7)
- [x] Simulation Monitor — **Simulation Summary →** via shared control (§G.16 / §D.7)
- [x] Algorithm Comparison — **Simulation Summary →** via shared control (§G.1 / §D.7)
- [x] Simulation Summary — **Simulation Monitor →** via shared control (§G.1 / §G.16 / §D.7)
- [x] Output Browser — JSONL viewer dual handoffs via shared labeled control (§G.14 / §D.7)
- [x] §G.1 / §G.14 / §G.16 studio-wide ``LogHandoffButtons`` toolbar migration checked (§D.7)

**Delivered (§D.7 — two-hundred-and-nineteenth pass)**

- [x] ``LogHandoffButtons`` optional ``path`` — empty path navigates to Summary / Monitor mode without pending-path handoff (§G.1 / §G.16 / §D.7)
- [x] ``LauncherNavMesh`` — sim **Simulation Summary →** / **Simulation Monitor →** via shared labeled control; Monitor-only while running, both when ``showPostRun`` (§G.9 / §G.15 / §D.7)
- [x] Simulation Launcher + Process Monitor sim panels inherit shared log-handoff control through ``simLogPath`` (§G.9 / §G.15 / §G.16 / §D.7)
- [x] §G.9 / §G.15 launcher nav-mesh ``LogHandoffButtons`` migration checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twentieth pass)**

- [x] Output Browser — selected-run tree header exposes icon Summary / Monitor handoffs when ``runJsonlPath`` is discovered (no need to open the ``.jsonl`` file first) (§G.14 / §G.1 / §G.16 / §D.7)
- [x] §G.14 run-select log-handoff surface parity with JSONL file viewer checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-first pass)**

- [x] ``LogHandoffButtons`` optional ``onAfterOpen`` — e.g. close Command Palette after handoff (§G.1 / §G.7 / §D.7)
- [x] Process Monitor process rows — icon Summary / Monitor handoffs when stdout-derived ``logPath`` is a ``.jsonl`` (no need to select the process first) (§G.15 / §G.1 / §G.16 / §D.7)
- [x] Command Palette recent log entries — dual icon handoffs for explicit Summary vs Digital Twin open (§G.7 / §G.1 / §G.16 / §D.7)
- [x] §G.15 / §G.7 process-row + palette recent-log handoff surface parity with Output Browser run header checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-second pass)**

- [x] ``applyStoreRecentHandoff`` / ``recentPendingSettersFromStore`` — non-React handoff helpers for toast / event handlers (§G.7 / §D.7 / §D.8)
- [x] ``ProcessIdFooter`` — icon Summary / Monitor handoffs when footer ``logPath`` is a ``.jsonl`` (launcher + Process Monitor live panels) (§G.9 / §G.15 / §G.1 / §G.16 / §D.7)
- [x] Process completion / failure / cancel toasts — **Summary** + **Monitor** actions when stdout yields a ``.jsonl``; **Training** for train/HPO run paths; **Output** for assets/output run roots (§D.8 / §G.1 / §G.14 / §G.16 / §G.17 / §D.7)
- [x] §G.9 / §G.15 live-panel footer + §D.8 toast log-handoff surface parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-third pass)**

- [x] ``PathHandoffButtons`` — path-kind handoff control: dual Summary / Monitor for ``.jsonl`` via ``LogHandoffButtons``; single-icon Training / Output / Data Explorer / Eval / Config for other ``RecentFileKind`` values (§G.7 / §G.14 / §G.15 / §G.17 / §D.7)
- [x] Process Monitor process rows + ``ProcessIdFooter`` — non-log path chips (training dirs, run roots) expose matching handoff icons, not only ``.jsonl`` (§G.15 / §G.9 / §G.10 / §G.17 / §D.7)
- [x] Command Palette recent entries — all known kinds show icon handoffs (logs keep dual Summary / Monitor) (§G.7 / §D.7)
- [x] Process train/HPO toasts — dual **Training** + **Output** actions when both paths are present in stdout (§D.8 / §G.17 / §G.14 / §D.7)
- [x] §G.15 / §G.7 non-log path-handoff surface parity with log dual-handoff surfaces checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-fourth pass)**

- [x] ``PathHandoffButtons`` optional empty ``path`` + explicit ``kind`` — mode-only navigation without pending-path handoff (nav-mesh parity with ``LogHandoffButtons``) (§G.7 / §G.9 / §G.10 / §D.7)
- [x] ``LauncherNavMesh`` — Output Browser / checkpoint / Training Monitor / Data Explorer path shortcuts via shared labeled ``PathHandoffButtons`` (§G.9 / §G.11 / §G.12 / §D.7)
- [x] ``TrainHpoNavMesh`` — Output Browser + Training Monitor path shortcuts via shared labeled ``PathHandoffButtons`` (§G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Output Browser — run-header path-kind handoffs (log dual or run single); file-viewer CSV / config / checkpoint / log via labeled ``PathHandoffButtons``; checkpoint sidebar icon handoffs (§G.14 / §G.12 / §D.7)
- [x] Training Monitor — run-panel training path + checkpoint browser icon handoffs via ``PathHandoffButtons`` (§G.17 / §G.12 / §D.7)
- [x] §G.14 / §G.9 / §G.10 / §G.17 Output Browser + nav-mesh + Training Monitor path-handoff surface parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-fifth pass)**

- [x] ``LoadedRunRow`` — ``pathHandoffs`` + optional ``handoffKind`` via shared ``PathHandoffButtons`` (supersedes direct ``LogHandoffButtons``; ``logHandoffs`` alias retained) (§G.1 / §G.7 / §G.14 / §D.7)
- [x] Portfolio loaded-run lists (Simulation Summary / Benchmark Analysis / City Comparison) migrate to ``pathHandoffs`` (§G.1 / §G.1.6 / §D.7)
- [x] Output Browser run list + Experiment Tracker output dirs + Training Monitor run discovery — ``pathHandoffs`` with explicit ``handoffKind`` (§G.14 / §G.17 / §G.18 / §D.7)
- [x] Evaluation Runner — checkpoint input rows + results table checkpoint ``PathHandoffButtons`` (§G.12 / §D.7)
- [x] Training Hub eval checkpoint + ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint icon handoffs (§G.10 / §G.12 / §G.15 / §D.7)
- [x] Benchmark Analysis eval-results checkpoint column ``PathHandoffButtons`` (§G.1 / §G.12 / §D.7)
- [x] Data Generation sensor CSV + HPO Tracker trial log dirs / report dir + Experiment Tracker MLflow run dirs — path-kind icon handoffs (§G.11 / §G.18 / §D.7)
- [x] §G.12 / §G.10 / §G.1 / §G.14 / §G.17 / §G.18 eval + portfolio + tracker path-handoff surface parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-sixth pass)**

- [x] ``PathHandoffButtons`` ``targets`` prop — forwarded to ``LogHandoffButtons`` for Summary-only / Monitor-only toolbars (§G.1 / §G.16 / §D.7)
- [x] ``LauncherNavMesh`` sim log shortcuts migrate to ``PathHandoffButtons`` ``kind="log"`` + ``targets`` (§G.9 / §D.7)
- [x] Simulation Summary / Simulation Monitor / Algorithm Comparison toolbars migrate off direct ``LogHandoffButtons`` onto ``PathHandoffButtons``; open-path chips gain dual icon handoffs (§G.1 / §G.16 / §D.7)
- [x] OLAP Explorer — ingest path labeled + chip ``PathHandoffButtons`` for ``.jsonl`` dual and CSV Data Explorer handoffs (§G.6 / §D.7)
- [x] Data Explorer open-CSV chip + Config Editor primary / diff path chips — path-kind icon handoffs (§G.6 / §G.13 / §D.7)
- [x] §G.1 / §G.6 / §G.9 / §G.13 / §G.16 analytics + config path-handoff surface parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-seventh pass)**

- [x] ``PathRunLabelChip`` ``handoff`` prop — composes kind-aware ``PathHandoffButtons`` (optional ``handoffStoredLabel`` / ``handoffTargets`` / ``handoffOnAfterOpen``) with custom ``trailing`` (§G.7 / §G.14 / §D.7)
- [x] ``RunLabelHeaderSuffix`` defaults ``handoff`` on live-panel header chips — Simulation Launcher / Process Monitor / train-HPO headers gain path handoffs without per-page wiring (§G.9–§G.18 / §D.7)
- [x] ``ProcessIdFooter`` + Process Monitor process-row chips migrate to ``handoff``; ``LoadedRunRow`` uses chip ``handoff`` for portfolio / run lists (§G.15 / §G.1 / §D.7)
- [x] Simulation Summary ConfigMetaBanner + open-path chips; Output Browser run header / checkpoint sidebar / file viewer / checkpoint panel chips (§G.1 / §G.14 / §G.16 / §D.7)
- [x] Shared eval / launcher / tracker chips (Eval cards, Evaluation Runner, Training Hub, Training Monitor, Benchmark, HPO, Experiment Tracker, Command Palette, Data Explorer, OLAP, Config Editor) use ``handoff`` instead of ad-hoc ``trailing={<PathHandoffButtons/>}`` (§G.7 / §G.10–§G.18 / §D.7)
- [x] §G.7 / §G.9–§G.18 path-chip handoff unification via ``PathRunLabelChip`` checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-eighth pass)**

- [x] ``OpenPathToolbar`` — shared open-path cluster: optional labeled reverse-handoff + ``PathRunLabelChip`` icon handoffs (``labeledTargets`` / host-aware chip targets / ``order`` / children) (§G.7 / §G.1 / §G.16 / §D.7)
- [x] Simulation Summary / Simulation Monitor / Algorithm Comparison open-log toolbars migrate to ``OpenPathToolbar`` with reverse-destination ``labeledTargets`` (§G.1 / §G.16 / §D.7)
- [x] OLAP Explorer ingest path toolbar migrates to ``OpenPathToolbar`` (§G.6 / §D.7)
- [x] Output Browser file viewer + checkpoint panel migrate labeled + chip dual control to ``OpenPathToolbar`` (§G.14 / §D.7)
- [x] §G.1 / §G.6 / §G.14 / §G.16 open-path labeled+chip toolbar unification checked (§D.7)

**Delivered (§D.7 — two-hundred-and-twenty-ninth pass)**

- [x] Data Explorer open-CSV toolbar migrates to ``OpenPathToolbar`` with export ``children`` (CSV / Parquet) (§G.6 / §G.7 / §D.7)
- [x] Config Editor primary YAML + diff comparison path chips migrate to ``OpenPathToolbar`` (§G.13 / §D.7)
- [x] Output Browser run header migrates to ``OpenPathToolbar``; labeled Summary / Monitor dual when run log is known (§G.14 / §G.1 / §D.7)
- [x] Training Monitor logs-root discovery chip migrates to ``OpenPathToolbar`` (§G.17 / §D.7)
- [x] Simulation Summary ``ConfigMetaBanner`` migrates to ``OpenPathToolbar`` with reverse Monitor ``labeledTargets`` (§G.1 / §D.7)
- [x] §G.1 / §G.6 / §G.13 / §G.14 / §G.17 remaining open-path toolbar surface expansion checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirtieth pass)**

- [x] HPO Tracker storage DB path + report directory migrate to ``OpenPathToolbar``; reports gain labeled Output Browser handoff + file-manager ``children`` (§G.18 / §D.7)
- [x] Experiment Tracker MLflow tracking URI path migrates to ``OpenPathToolbar`` (§G.18 / §D.7)
- [x] Training Monitor empty-state logs-root chip migrates to ``OpenPathToolbar`` (discovery toolbar parity) (§G.17 / §D.7)
- [x] ``PolicyTelemetryTrendsPanel`` SQLite ``db_path`` chip migrates to ``OpenPathToolbar`` (§A.3 / §D.7)
- [x] ML Introspection tensor archive path migrates to ``OpenPathToolbar`` (§G.5 / §D.7)
- [x] Settings project root / Python / import JSON / Arrow benchmark paths migrate to ``OpenPathToolbar``; benchmark auto-classifies CSV/log handoffs (§G.19 / §D.7)
- [x] §G.5 / §G.17 / §G.18 / §G.19 residual open-path toolbar surface expansion checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-first pass)**

- [x] Evaluation Runner checkpoint-row + dataset path previews migrate to ``OpenPathToolbar`` (§G.12 / §D.7)
- [x] Training Hub eval checkpoint + dataset path previews migrate to ``OpenPathToolbar``; checkpoint labeled Eval Runner handoff (§G.10 / §G.12 / §D.7)
- [x] Data Generation sensor CSV / TSPLIB / preview path previews migrate to ``OpenPathToolbar``; sensor labeled Data Explorer handoff (§G.11 / §G.6 / §D.7)
- [x] ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint headers migrate to ``OpenPathToolbar`` (§G.12 / §G.15 / §D.7)
- [x] §G.10 / §G.11 / §G.12 launcher selected-path open-path toolbar parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-second pass)**

- [x] Training Monitor ``RunPanel`` path header migrates to ``OpenPathToolbar``; epochs meta as ``children`` (§G.17 / §D.7)
- [x] Training Monitor checkpoint browser rows migrate to ``OpenPathToolbar``; labeled Eval Runner handoff + size ``children`` (§G.17 / §G.12 / §D.7)
- [x] Output Browser checkpoint sidebar rows migrate to ``OpenPathToolbar``; labeled Eval Runner handoff + size ``children`` (§G.14 / §G.12 / §D.7)
- [x] Output Browser ``.wsroute`` manifest member paths gain ``PathRunLabelChip`` auto-classified handoffs (§G.8 / §G.14 / §D.7)
- [x] §G.14 / §G.17 residual panel + checkpoint open-path toolbar parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-third pass)**

- [x] HPO Tracker trial log-dir table cells migrate to ``OpenPathToolbar``; trial number as ``children`` (§G.18 / §D.7)
- [x] Experiment Tracker MLflow run-dir table cells migrate to ``OpenPathToolbar``; run-id meta as ``children`` (§G.18 / §D.7)
- [x] Evaluation Runner + Benchmark Analysis results-table checkpoint cells migrate to ``OpenPathToolbar`` (§G.12 / §G.1 / §D.7)
- [x] Process Monitor process-row path chips migrate to ``OpenPathToolbar``; process id as ``children`` (§G.15 / §D.7)
- [x] §G.1 / §G.12 / §G.15 / §G.18 residual table-row open-path toolbar parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-fourth pass)**

- [x] ``ProcessIdFooter`` log-path chip migrates to ``OpenPathToolbar``; process id as ``children`` — live-panel footer parity across sim / train / HPO / eval / data-gen (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] ``LoadedRunRow`` migrates to ``OpenPathToolbar``; day-count / meta ``trailing`` as ``children`` — portfolio list handoff shell parity (§G.1 / §G.14 / §D.7)
- [x] Output Browser ``.wsroute`` manifest members migrate to ``OpenPathToolbar`` (auto-classify handoff) (§G.8 / §G.14 / §D.7)
- [x] §G.1 / §G.8 / §G.9–§G.15 / §G.17 / §G.18 shared footer + portfolio open-path toolbar parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-fifth pass)**

- [x] ``OpenPathToolbar`` gains ``handoffOnAfterOpen`` (labeled + chip) for palette close-after-handoff (§G.7 / §D.7)
- [x] ``RunLabelHeaderSuffix`` migrates to ``OpenPathToolbar``; co-located with toolbar module (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] ``LauncherLivePanelHeader`` / ``TrainHpoLivePanelHeader`` import ``RunLabelHeaderSuffix`` from ``OpenPathToolbar`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Command Palette recent-file rows migrate to ``OpenPathToolbar``; kind meta as ``children`` + ``handoffOnAfterOpen`` (§G.7 / §D.7)
- [x] §G.7 / §G.9–§G.12 / §G.15 / §G.17 / §G.18 live-header + palette open-path toolbar parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-sixth pass)**

- [x] ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set so live headers sync with ``GlobalFilterBar`` (§G.9–§G.18 / §D.7)
- [x] ``ProcessIdFooter`` multi-process batch meta as ``children`` when ``logPath`` is set (eval multi-checkpoint) (§G.12 / §D.7)
- [x] Command Palette recent rows: kind meta as ``children``, stored ``label`` display, non-nested row shell (§G.7 / §D.7)
- [x] Graph Topology distance-matrix path migrates to ``OpenPathToolbar``; CSV Data Explorer handoff (§G.4 / §D.7)
- [x] §G.4 / §G.7 / §G.9–§G.18 residual open-path toolbar shell polish checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-seventh pass)**

- [x] ``ProcessIdFooter`` accepts process-derived ``runLabel`` → ``OpenPathToolbar`` ``brushLabel`` so footer chips match live headers / ``GlobalFilterBar`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor process-row ``OpenPathToolbar`` brushes ``processRunBrushById`` labels (path-stem vs process-id fallback parity) (§G.15 / §D.7)
- [x] Live-panel footers on Simulation Launcher / Data Generation / Evaluation Runner / Training Hub / Training Monitor / HPO Tracker / Experiment Tracker / Process Monitor pass ``runLabel`` into ``ProcessIdFooter`` (§G.9–§G.12 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] §G.9–§G.12 / §G.15 / §G.17 / §G.18 process-row + footer brush-label residual parity checked (§D.7)

**Delivered (§D.7 — two-hundred-and-thirty-eighth pass)**

- [x] HPO Tracker trial log-dir cells pass ``brushLabel`` / ``storedLabel`` from ``trialBrushLabel`` so click-to-brush matches row rings (§G.18 / §D.7)
- [x] Experiment Tracker MLflow run-dir cells pass explicit ``brushLabel`` (run name) for path-stem divergence parity (§G.18 / §D.7)
- [x] Training Monitor ``RunPanel`` path header brushes ``run.name`` (list / panel shell parity) (§G.17 / §D.7)
- [x] ``LoadedRunRow`` passes ``brushLabel`` from resolved portfolio run label (§G.1 / §D.7)
- [x] ``LauncherNavMesh`` accepts optional ``csvPath`` for data-gen post-run Data Explorer handoff (§G.11 / §G.6 / §D.7)
- [x] Data Generation + Process Monitor data-gen panels pass sensor / generated CSV into ``LauncherNavMesh`` (§G.11 / §G.15 / §D.7)
- [x] §G.1 / §G.11 / §G.15 / §G.17 / §G.18 tracker + portfolio + run-panel brush-label residual parity checked (§D.7)

---

### §D.8 — Toast Notifications for Background Completions

**Pain**: When a training job or data generation task finishes in the background, there is no notification. Users must check the process monitor tab to see if the job completed.

**Options**

- **A** — Use a React toast library (`sonner` or `react-hot-toast`) for in-app notifications: auto-dismissing toasts in the bottom-right corner for job completion, failure, and warnings. Triggered by Tauri events from the process monitor. `[Quick Win]`
- **B** — Use the Tauri notification plugin (`@tauri-apps/plugin-notification`) to display a native OS notification when a job finishes and the Studio window is not in focus.
- **C** — Play an OS sound via the Tauri shell plugin on job completion.

**Recommendation**: **Option A + B** — the React toast for when the window is focused, Tauri native notification for when the user has switched away. Option C is optional polish.

**Effort × Impact**: Low effort / High impact

**Delivered (§D.8 Option A+B — base)**

- [x] Sonner in-app toasts on process completed / failed / cancelled via ``useProcessMonitor``
- [x] Tauri native OS notification when the Studio window is not focused

**Delivered (§D.8 — two-hundred-and-twenty-second pass)**

- [x] Completion / failure / cancel toast action buttons hand off into Summary / Monitor (``.jsonl``), Training Monitor (train/HPO path), or Output Browser (run root) via ``applyStoreRecentHandoff`` (§D.8 / §G.1 / §G.14 / §G.16 / §G.17 / §D.7)

**Delivered (§D.8 — two-hundred-and-twenty-third pass)**

- [x] Train/HPO completion toasts expose dual **Training** + **Output** actions when stdout yields both a training run path and an assets/output root (§D.8 / §G.17 / §G.14 / §D.7)

**Delivered (§D.8 — two-hundred-and-thirty-seventh pass)**

- [x] Eval completion toasts expose **Eval** (checkpoint → Evaluation Runner) and optional **Output** when stdout / command yields a load path + run root (§D.8 / §G.12 / §G.14 / §D.7)

**Delivered (§D.8 — two-hundred-and-thirty-eighth pass)**

- [x] Data-gen completion toasts expose **Data** (sensor CSV / generated ``.csv`` → Data Explorer) and optional **Output** when stdout / command yields a path + run root (§D.8 / §G.11 / §G.6 / §G.14 / §D.7)
- [x] ``genDataPath.ts`` shared sensor / TSPLIB / ``Generated`` path extractors for toasts + nav mesh (§G.11 / §D.8 / §D.7)

**Status**: §D.8 Options A+B complete — Option C (sound) deferred.

---

### Effort × Impact Matrix — GUI / UX

| Item                                        | Effort   | Impact | Priority                          |
| ------------------------------------------- | -------- | ------ | --------------------------------- |
| §D.3 Option A+B (theme toggle + persist)    | Very Low | Medium | P0 `[Quick Win]` ✅              |
| §D.3 Option C (system theme following)      | Very Low | Medium | P0 `[Quick Win]` ✅              |
| §D.7 Option A (keyboard shortcuts)          | Very Low | Medium | P0 `[Quick Win]` ✅ (incl. T/H/E/B/O train + L/D/V launcher workflow) |
| §D.4 Option B (Tauri Store persistence)     | Low      | High   | P0 ✅ (Zustand persist)           |
| §D.8 Option A+B (toast + OS notification)   | Low      | High   | P1 ✅ (toast + OS notification done) |
| §D.5 Option A+C (cancel + progress modal)   | Medium   | High   | P1 ✅ (cancel + progress bars)    |
| §D.2 Option A (live training charts)        | Medium   | High   | P1 ✅ (all launchers + monitors + eval progress/ETA) |
| §D.1 Option A (ECharts route panel)         | Medium   | High   | P2 ✅ (RouteViz + Summary)        |
| §D.6 Option A (override table)              | Medium   | High   | P2 ✅ (all launchers)             |
| §D.1 Option B (deck.gl PathLayer)           | High     | High   | P2 ✅ (§G.3 / §G.16)              |
| §D.6 Option B (typed config form)           | High     | High   | P3                                |

---

