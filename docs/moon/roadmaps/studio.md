> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## G — WSmart-Route Studio

> WSmart-Route Studio is the Tauri 2.0 desktop application that replaces the existing PySide6 GUI and extends it with deep analytics visualization, geospatial routing replay, ML introspection, and an OLAP query interface. The §D section above defines the UX requirements these phases must satisfy. The Studio is the primary interface for all user-facing operations: launching simulations and training runs, generating data, executing scripts, browsing results, and performing post-hoc analysis.

**Technology Stack**

| Concern | Technology |
| --- | --- |
| Desktop shell | Tauri 2.0 (Rust backend + native WebView) |
| Frontend framework | React 19 + TypeScript |
| Styling | Tailwind CSS |
| Data serialization | Apache Arrow IPC (zero-copy Rust ↔ JS) |
| In-browser OLAP | DuckDB-Wasm (Web Worker) |
| 2D charts | Apache ECharts |
| Geospatial rendering | deck.gl (WebGL, TripsLayer, OrbitView) |
| Graph visualization | Sigma.js v4 + Graphology / Cosmograph |
| 3D ML visualization | React Three Fiber (Three.js) |
| Tensor I/O (Rust) | ndarray-npy crate |
| State management | Zustand |
| Process management | tokio::process (Rust), Tauri event system |
| Config persistence | Tauri Store plugin |
| OS notifications | Tauri notification plugin |

---

### §G.0 — Phase 0: Foundation & Tooling ✅

**Goal**: Establish the project scaffold, dev environment, and data pipeline so all subsequent phases have a stable base.

- [x] Bootstrap Tauri 2.0 project (`app/src-tauri/` + React/TypeScript frontend); window 1600×1000, min 1200×700
- [x] Configure Tailwind CSS with dark theme defaults (`canvas-*` / `accent-*` palette) and `dark:` class toggle (§D.3)
- [x] Set up Rust backend with `tauri 2.0`, `tauri-plugin-{notification,store,dialog,shell}`, `serde`, `tokio`, `csv`, `anyhow`
- [x] Implement Tauri Store plugin setup for session and theme persistence (§D.3, §D.4)
- [x] `tools/app/justfile` — dev/build/check/clean commands; wired to root justfile as `just studio`, `just studio-build`, `just studio-install`
- [x] Arrow IPC schema for simulation log rows and Rust CSV → Arrow IPC stream: `commands/arrow.rs` — `csv_to_arrow_ipc`, `simulation_log_to_arrow_ipc` (typed KPI schema: policy/sample_id/day + profit/km/overflows/kg_per_km/…)
- [x] Spawn DuckDB-Wasm in a Web Worker; ingest Arrow IPC on CSV/log open: `duckdbClient.ts` + `useDuckDbInit` on app mount; Data Explorer + Simulation Monitor auto-ingest
- [x] Verify end-to-end latency: Settings "Run Arrow Pipeline Benchmark" + Data Explorer timing badge; 500 ms budget constant in `arrowPipeline.ts` (§G.0 partial — hardware baseline varies)
- [x] Arrow sidecar fast-path: `runCsvArrowPipeline()` + `runSimulationArrowPipeline()` prefer sibling ``.arrow`` IPC from extracted `.wsroute` bundles via `path_exists` + `runArrowSidecarPipeline()` (skips Rust CSV/JSONL re-parse; §G.0 / §G.8)
- [x] Portfolio DuckDB union: `runPortfolioSimulationArrowPipeline()` unions multiple JSONL logs into one table with `run_label`; `formatPipelineTimingBadge()` shared toolbar timing text (§G.0 / §G.1.4)

---

### §G.1 — Phase 1: Statistical Overview Dashboard (ECharts 2D)

**Goal**: Reproduce and extend the existing static `simulation_analysis.md` charts as interactive ECharts panels.

#### 1.1 KPI Summary Bar / Box Charts
- [x] Mean ± std overflows per constructor, grouped by mandatory-selection strategy: `GroupedMetricBarChart` overflows by selection strategy on Simulation Summary; portfolio mode swaps to overflows by city/scale across loaded runs (§G.1.1)
- [x] Mean ± std kg/km per constructor, grouped by city/scale: `GroupedMetricBarChart` kg/km by constructor on Simulation Summary; portfolio mode swaps to kg/km by city/scale across loaded runs (§G.1.1)
- [x] Grouped metric bar charts follow global ``logScale``: overflows groups use symlog y-axis; kg/km groups use log y-axis; error-bar whiskers via ``errorBarBounds`` when log on (§G.1.1 / §G.7)
- [x] Interactive brushing: selecting a bar cross-filters all panels on the dashboard: `PolicyBrushBar` + `toggleBrush` dims non-selected policies across all charts; `SimulationSummary` ingests log → DuckDB + `SqlQueryPanel` `brushSqlSync` / `brushedPoliciesSql` (§G.1)

#### 1.2 Overflow vs Efficiency Scatter (Pareto Front)
- [x] 4-panel layout: Gamma-3/FTSP · Empirical/FTSP · Gamma-3/CLS · Empirical/CLS: `BenchmarkAnalysis` + `SimulationSummary` `BenchmarkParetoPanel` grid + `paretoPortfolio.ts` / `paretoPanels.ts` run classifier (§G.1.2)
- [x] Color encoding: LA · LM-CF70 · LM-CF90 · SL-SL1 · SL-SL2: `strategyColor()` on Pareto scatter + efficiency ranking bars (§G.1.2)
- [x] Marker shape: RM-100 circle · RM-170 square · FFZ-350 diamond: `citySymbol()` from `parseLogPath()` on `BenchmarkParetoPanel` multi-run scatter (§G.1.2)
- [x] Computed Pareto front drawn as white dashed step line: `PolicyParetoChart` + `BenchmarkParetoPanel` on Simulation Summary / Benchmark Analysis (§G.1.2)
- [x] Log-scale toggle on Simulation Summary policy bar charts (§G.1)
- [x] Pareto scatter follows global ``logScale``: symlog overflows y-axis + log profit x-axis on ``PolicyParetoChart`` + ``BenchmarkParetoPanel`` (§G.1.2 / §G.7)
- [x] BenchmarkParetoPanel per-facet PNG export: ``exportChartPng()`` on each 4-panel Pareto facet with toast feedback (§G.1.2 / §G.7)
- [x] Symlog bar charts: `symlog.ts` + `useSymlog` on profit · km · overflows `MetricBarChart` when log scale on; secondary log-scale row adds profit/km symlog duplicates (§G.1)
- [x] BenchmarkAnalysis multi-run comparison bar charts follow global ``logScale`` (§G.1 / §G.7)
- [x] AlgorithmComparison per-metric bar charts follow global ``logScale`` (§G.1 / §G.7)
- [x] AlgorithmComparison error-bar whiskers on metric bars: mean ± std toggle via ``showErrorBars``; log/symlog whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.1 / §G.7)
- [x] AlgorithmComparison symlog overflows on log-scale metric bars (§G.1.1 / §G.7)
- [x] Policy radar chart on Simulation Summary: normalised multi-metric overlay per policy with PNG export; log-normalised axes when global ``logScale`` on (Simulation Summary + Algorithm Comparison; §G.1 / §G.7)
- [x] Error-bar whiskers on Simulation Summary bar charts: custom ECharts series showing mean ± std; log/symlog whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.1 / §G.7)
- [x] Hover tooltip: all config values + KPI values: `simMetadata.ts` + `policyTooltipFooter()` on bar/Pareto/heatmap/radar/parallel charts; `BenchmarkParetoPanel` adds `formatLogMeta` + `formatPolicyMeta` per run×policy point (§G.1.2)

#### 1.3 Policy Configuration Heatmaps
- [x] Heatmap split by distribution (Gamma-3 vs Empirical): `DistributionFacetHeatmaps` on Simulation Summary when policies span distributions; portfolio mode adds `BenchmarkDistributionHeatmap` facets via `groupRunsByDistribution()` (§G.1.3)
- [x] Heatmap split by graph (RM-100 vs RM-170 vs FFZ-350): shared `BenchmarkGraphHeatmap` facets by `cityScaleLabel()` on Benchmark Analysis + Simulation Summary portfolio mode (§G.1.3)
- [x] Cell value = mean overflows or mean kg/km (toggle): unified `heatmapMode` buttons (all / overflows / kg/km) on Simulation Summary + Benchmark Analysis; portfolio distribution/graph facets share the same mode (§G.1.3)
- [x] Color gradient from dark (worst) to bright (best): `PolicyHeatmapChart` + `BenchmarkPortfolioHeatmap` + shared `heatmapMetrics.ts` normalised indigo→green gradient; portfolio policy×metric heatmap when ≥2 runs loaded (§G.1.3)
- [x] Policy configuration heatmaps follow global ``logScale``: ``buildNormalizedHeatmapCells`` symlog/log-transforms KPI values before min–max normalisation on ``PolicyHeatmapChart``, ``BenchmarkPortfolioHeatmap``, ``BenchmarkDistributionHeatmap``, and ``BenchmarkGraphHeatmap``; tooltips show raw KPI values (§G.1.3 / §G.7)
- [x] BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap facet PNG export: ``exportChartPng()`` per distribution/graph facet with toast feedback (§G.1.3 / §G.7)

#### 1.4 Parallel Coordinates (Hyper-Dimensional Policy Explorer)
- [x] Axes: city · N · dist · improver · strategy · constructor · overflows · kgkm · km · profit: `PolicyParallelChart` + `parallelPolicyAxes.ts` ten-axis schema on Simulation Summary; shared `BenchmarkPortfolioParallel` on Simulation Summary + Benchmark Analysis (§G.1.4)
- [x] Each of the 480 simulation logs rendered as a polyline: `BenchmarkPortfolioParallel` + `scanOutputPortfolio()` / `loadPortfolioLogs()` batch loader (up to 480 runs) on Benchmark Analysis (§G.1.4)
- [x] Portfolio DuckDB ingest: Simulation Summary unions primary + comparison runs into `summary_sim`; Benchmark Analysis → `benchmark_sim`; City Comparison → `city_sim` with sidecar-aware timing badges (§G.1.4 / §G.6)
- [x] Brushing on any axis instantly filters all other panels: ECharts parallel-axis brush toolbox on `PolicyParallelChart` → `handleBrushPolicies` cross-filter; click polyline → `toggleBrush`; DuckDB SQL sync via `brushSqlSync` (§G.1)
- [x] Highlight corridor: drag brush on overflows ≤ threshold to identify zero-overflow configs: overflow corridor slider + parallel-axis overflows brush syncs `overflowMax` + `effectiveBrushed` cross-filter on Simulation Summary (§G.1)
- [x] Parallel coordinates follow global ``logScale``: ``PolicyParallelChart`` + ``BenchmarkPortfolioParallel`` log-normalise profit · kg/km · km axes; symlog overflows; corridor brush inverts symlog via ``invertParallelAxisValue`` (§G.1.4 / §G.7)
- [x] Color polylines by mandatory-selection strategy: `strategyColor()` on `PolicyParallelChart` polylines; `BenchmarkPortfolioParallel` colours run polylines via `resolveRunSelectionStrategy()` + `selectionStrategyColor()` from log path / dominant policy with strategy legend (§G.1.4)
- [x] BenchmarkPortfolioParallel PNG export: ``exportChartPng()`` on portfolio parallel-coordinates panel with toast feedback (§G.1.4 / §G.7)

#### 1.5 Constructor Ranking Chart
- [x] Horizontal bar chart: `EfficiencyRankingChart` ranks policies by mean kg/km, bottom-up ordering; portfolio mode adds `PortfolioEfficiencyRanking` for run×policy configs (§G.1.5)
- [x] Rank by mean kg/km across all configurations: Simulation Summary efficiency ranking + `PortfolioEfficiencyRanking` + BenchmarkAnalysis `kg/km` metric column (§G.1.5)
- [x] Error bars showing std deviation: Simulation Summary bar-chart whiskers toggle (§G.1)
- [x] Error bars on efficiency ranking chart: horizontal kg/km whiskers toggle via `showErrorBars` (§G.1)
- [x] Efficiency ranking charts follow global ``logScale``: ``EfficiencyRankingChart`` + ``PortfolioEfficiencyRanking`` log x-axis; horizontal whiskers via ``errorBarBounds`` when log on (§G.1.5 / §G.7)
- [x] BenchmarkAnalysis efficiency ranking follows global ``logScale``: multi-run ``PortfolioEfficiencyRanking`` + single-run inline chart log x-axis; shared ``showErrorBars`` toggle; horizontal kg/km whiskers via ``errorBarBounds`` when log on (§G.1.5 / §G.7)
- [x] BenchmarkAnalysis multi-run metric bars use symlog for overflows when global ``logScale`` on (§G.1.1 / §G.7)
- [x] BenchmarkAnalysis multi-run metric bar error-bar whiskers: grouped run×policy bars show mean ± std via shared ``showErrorBars`` toggle; log/symlog whiskers via ``errorBarBounds`` + ``groupedBarWhiskerX`` when global ``logScale`` on (§G.1 / §G.7)

#### 1.6 Secondary Log-Scale Views
- [x] Auto-generate log-scale version below each chart that benefits from it (overflow counts, profit ranges): duplicate profit · km · kg · symlog-overflows row when global log toggle off (§G.1)
- [x] City Comparison section follows global ``logScale``: log-scale profit + symlog-overflows bars when on; linear raw values when off; `BenchmarkAnalysis` + `SimulationSummary` + dedicated `CityComparison` page with portfolio load + summary table (§G.1.6 / §G.7)
- [x] City Comparison error-bar whiskers: profit · symlog-overflows · kg/km grouped bars show mean ± std via ``showErrorBars`` toggle on ``cityComparisonChartOption``; log/symlog whiskers via ``errorBarBounds`` + ``groupedBarWhiskerX`` on Benchmark Analysis, Simulation Summary portfolio mode, and City Comparison page (§G.1.6 / §G.7)
- [x] Simulation Summary / Algorithm Comparison open-log toolbars + path chips use shared ``PathHandoffButtons`` (two-hundred-and-twenty-sixth pass; §G.1 / §G.16 / §D.7)
- [x] Simulation Summary ConfigMetaBanner + open-path chips use ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.1 / §D.7)
- [x] Simulation Summary / Algorithm Comparison open-log toolbars use shared ``OpenPathToolbar`` labeled+chip cluster (two-hundred-and-twenty-eighth pass; §G.1 / §G.16 / §D.7)
- [x] Simulation Summary ``ConfigMetaBanner`` uses shared ``OpenPathToolbar`` with reverse Monitor ``labeledTargets`` (two-hundred-and-twenty-ninth pass; §G.1 / §D.7)
- [x] Benchmark Analysis results-table checkpoint cells use shared ``OpenPathToolbar`` (two-hundred-and-thirty-third pass; §G.1 / §G.12 / §D.7)
- [x] ``LoadedRunRow`` portfolio lists pass explicit ``brushLabel`` from resolved run label (two-hundred-and-thirty-eighth pass; §G.1 / §D.7)

**Status**: §G.1 complete — all checklist items delivered.

---

### §G.2 — Phase 2: Hierarchical Drill-Down (Sunburst / Treemap)

**Goal**: Enable macro → micro navigation from algorithm family level down to individual config variants.

- [x] Top-level Sunburst chart: inner ring = city/scale · middle ring = selection strategy · outer ring = constructor: `PolicyHierarchyPanel` + `policyHierarchy.ts` on Simulation Summary; `buildPortfolioHierarchy()` multi-root sunburst when ≥2 logs loaded (§G.2)
- [x] Angular span mapped to accumulated profit; color gradient = kg/km efficiency: sunburst/treemap segment `value` = profit sum; `itemStyle.color` from kg/km gradient; middle strategy ring adds `selectionStrategyColor()` border stroke (§G.2)
- [x] Click on any segment fires DuckDB-Wasm filter query: segment click → `policiesAtPath` → `toggleBrush` cross-filter; `SqlQueryPanel` `brushSqlSync` + `autoRunOnBrushSync` executes `brushedPoliciesSql` (§G.2)
- [x] Drill-down transition: Sunburst morphs into horizontal bar chart (mean ± variance per variant): `PolicyHierarchyPanel` `universalTransition` morphs sunburst/treemap → drill-down profit bars on segment click; log-scale profit x-axis when global ``logScale`` on (§G.2 / §G.7)
- [x] Error bars on drill-down bars representing variance across Empirical vs Gamma-3 distributions: `enrichDrillChildren` profit std + Empirical↔Gamma spread whiskers on `PolicyHierarchyPanel` drill-down; log-scale profit x-axis whiskers via ``errorBarBounds`` when global ``logScale`` on (§G.2 / §G.7)
- [x] Breadcrumb trail showing current filter path; click to navigate back up: `HierarchyBreadcrumb` in `PolicyHierarchyPanel` with root **All** reset (§G.2)
- [x] Treemap alternative view: area = profit, color = overflows (toggle with Sunburst): sunburst/treemap view toggle on Simulation Summary; kg/km vs overflows colour mode selector on `PolicyHierarchyPanel` (§G.2)
- [x] Shared strategy colour legend: `SELECTION_STRATEGY_LEGEND` + `StrategyLegend` chips on `PolicyParallelChart`, `BenchmarkPortfolioParallel`, and `PolicyHierarchyPanel` (§G.1.4 / §G.2)
- [x] Drill-down profit bars coloured by mandatory-selection strategy at strategy depth; constructor depth uses kg/km or overflow gradient via `resolveDrillBarColor()` (§G.2)

**Status**: §G.2 complete — all checklist items delivered.

---

### §G.3 — Phase 3: Geospatial Routing Visualization (deck.gl)

**Goal**: Animate the physical routes constructed by each algorithm over the real-world city graphs.

#### 3.1 Base Map Layer
- [x] Integrate deck.gl with MapLibre GL (OpenStreetMap tiles): `DeckRouteMap` uses `react-map-gl/maplibre` + Carto dark basemap (§G.16)
- [x] Load node coordinates for Rio Maior (N=100, N=170) and Figueira da Foz (N=350) from graph JSON files: `graphCoords.ts` presets + SimulationMonitor "Load graph coords" (§G.3.1)
- [x] Auto-detect graph preset from log path segments or day-1 bin count: `guessGraphPreset()` + SimulationMonitor auto-select (§G.3.1)
- [x] Render nodes as ScatterplotLayer: fill-level colour-coded tour stops + dimmed idle bins in `DeckRouteMap`; radius scales with fill % and collected kg (`bin_state_collected`) (§G.3.1)
- [x] Render depot as distinct marker: gold `ScatterplotLayer` with white stroke in `DeckRouteMap`
- [x] Pan/zoom/tilt with 3D perspective: `DeckRouteMap` controlled view state + 3D pitch toggle (0°/45°); OrbitView Cartesian mode in §G.3.4 (§G.3.1)

#### 3.2 Route Animation (TripsLayer)
- [x] Parse per-day route from `tour_indices` + `all_bin_coords` into timestamped coordinate arrays (`DeckRouteMap`)
- [x] Feed routes into deck.gl `TripsLayer` with animated trail during day playback (§G.3.2 / §G.16)
- [x] Timeline slider: day scrubber with range input + ◀/▶ step buttons (SimulationMonitor)
- [x] Playback controls: play / pause / 1×·2×·4× speed multiplier on day scrubber; `TripsLayer` animated trail in Mercator and OrbitView Cartesian modes (§G.3.2)
- [x] Multi-vehicle rendering with distinct color coding per vehicle: `vehicleTours.ts` splits depot-delimited `tour` sequences; `DeckRouteMap` + `RouteMapChart` render per-vehicle paths and per-vehicle tour-stop scatter layers (§G.3.2)

#### 3.3 Algorithm Comparison Mode
- [x] Side-by-side view: overlay/split toggle in SimulationMonitor when 2 policies visible; split renders dual `DeckRouteMap` or dual ECharts `RouteMapChart` panels (§G.16)
- [x] Algorithm Comparison → map deep link: `pendingMapCompare` sets visible policies + split layout when 2 policies present
- [x] Toggle visibility per policy: map policy chip row in SimulationMonitor; `DeckRouteMap` multi-route overlay with per-policy colour paths
- [x] Overlay skipped vs visited nodes: idle bins dimmed grey, tour stops fill-coded (bright) in `DeckRouteMap`

#### 3.4 Non-Geographic Cartesian Mode (OrbitView)
- [x] Switch between geographic (Mercator) and abstract Cartesian coordinate system: Simulation Monitor ECharts vs deck.gl toggle; `DeckRouteMap` auto-selects Mercator (geo) or OrbitView (abstract) (§G.3.4)
- [x] OrbitView camera: orbit, pan, zoom a 3D point cloud: `DeckRouteMap` OrbitView with fill-scaled Z elevation on tour stops (§G.3.4)
- [x] Used for normalized/synthetic datasets where coordinates are not GPS: circular `resolveBinPositions()` layout when log lacks lat/lng (§G.3.4)

**Status**: §G.3 complete — all checklist items delivered.

---

### §G.4 — Phase 4: Topological Graph Analytics (Sigma.js / Cosmograph)

**Goal**: Visualize the raw optimization graph structure, pheromone trails, and node-edge weights.

- [x] Load distance matrix from `assets/` as a weighted edge list: `graphTopology.ts` resolves sibling `gmaps_distmat.csv` or project `data/wsr_simulator/distance_matrix/`; k-NN edge list builder (§G.4)
- [x] Render graph using Sigma.js (WebGL): node radius ∝ profit, edge thickness ∝ inverse distance: `GraphTopologyPanel` ECharts `graph` series — node size ∝ bin fill %, edge width ∝ inverse distance; View toggle adds `TopologySigmaView` Sigma.js WebGL with fill/pheromone styling + `TopologyCosmographView` dense point-mode WebGL (§G.4)
- [x] Force-directed layout (ForceAtlas2) via Graphology: `TopologySigmaView` runs `graphology-layout-forceatlas2` on force layout; ECharts path keeps Fruchterman-Reingold in `forceDirectedLayout()` (§G.4)
- [x] ACO pheromone trail visualization: edge opacity/color intensity ∝ accumulated pheromone weight after each iteration: `accumulateTourPheromone()` deposits τ on consecutive tour edges; amber edge styling in `GraphTopologyPanel` ECharts + Sigma.js + Cosmograph views (live ACO solver τ matrix deferred to logic layer)
- [x] Cross-filter from DuckDB-Wasm: brushing a profit range highlights matching nodes: fill-% dual slider + SQL "Brush profit range" / day row click → topology panel; click node in ECharts/Sigma/Cosmograph view → fill-% brush (§G.4)
- [x] Dynamic re-layout when filter applied: clusters emerge based on algorithm prioritization: "Re-layout on filter" toggle re-runs spring layout on filtered subgraph (§G.4)
- [x] Cosmograph alternative for large dense graphs (N=350): `radialDenseLayout()` + auto radial when N≥200; layout mode selector (auto/force/radial) on `GraphTopologyPanel`; `TopologyCosmographView` Sigma.js point renderer with ForceAtlas2 dense settings (§G.4)
- [x] Timeline slider synced with route animation to show pheromone evolution over iterations: pheromone day slider syncs with Simulation Monitor day scrubber + playback; "By tour step" mode steps τ per consecutive tour edge via `accumulateTourPheromoneByStep` (§G.4)
- [x] Topology pheromone trails follow global ``logScale``: ``pheromoneWeightDisplay()`` + ``normalizePheromone()`` / ``pheromoneIntensity()`` log-transform τ before edge opacity/width on ECharts, Sigma.js, and Cosmograph views; ``GraphTopologyPanel`` receives ``logScale`` from Simulation Monitor (§G.4 / §G.7)
- [x] ECharts topology PNG export: ``exportChartPng()`` on ``GraphTopologyPanel`` when View = ECharts (§G.4 / §G.7)
- [x] ECharts topology SVG export: ``exportChartSvg()`` on ``GraphTopologyPanel`` when View = ECharts; toast feedback (§G.4 / §G.7)
- [x] Sigma.js / Cosmograph WebGL PNG export: ``exportContainerCanvasPng()`` on ``GraphTopologyPanel`` when View = Sigma.js or Cosmograph; toast feedback (§G.4 / §G.7)
- [x] Graph Topology distance-matrix path uses shared ``OpenPathToolbar`` with CSV Data Explorer handoff (two-hundred-and-thirty-sixth pass; §G.4 / §D.7)

**Status**: §G.4 complete — all checklist items delivered.

---

### §G.5 — Phase 5: Machine Learning Introspection Dashboard

**Goal**: Expose the internals of trained neural CO models (Attention Models, Routing Transformers).

#### 5.1 TensorDict Data Pipeline
- [x] Rust backend: load `.npy`/`.npz` TensorDict files via `ndarray-npy` crate: `tensor.rs` `inspect_npz_archive` + `load_tensor_slice` (§G.5.1 — full native `.td` parse deferred to logic layer)
- [x] TensorDict (`.td`) inspect + slice via Python subprocess (`torch.load` + key/shape listing; slice export matches NPZ path): `inspect_npz_archive` / `load_tensor_slice` accept `project_root` + `python_executable`; Archive tab opens `.td` files (§G.5.1)
- [x] Memory-map large tensor files (avoid full RAM load): `load_npy_plane_mmap` + `load_npz_plane_mmap` via `memmap2` reads only the trailing 2-D plane for standalone `.npy` or stored `.npz` entries > 8 MB; `load_npz_plane_decompress` slices deflated `.npz` entries after single-entry inflate; `TensorSlicePreview.used_memmap` / `used_decompress_slice` surfaced in Archive/Attention tabs; `probe_npy_mmap` covers large stored or compressed `.npz` arrays (§G.5.1)
- [x] Stream specific tensor slices to frontend over Arrow IPC on demand: `tensor_slice_to_arrow_ipc` long-format `(row, col, value)`; `runTensorArrowPipeline` ingests into DuckDB-Wasm as `studio_tensor` from Archive tab; `.td` slices supported via Python handoff (§G.5.1)

#### 5.2 3D Loss Landscape Visualization (React Three Fiber)
- [x] Python utility script: compute loss surface grid using Li et al. filter-normalized random directions: `logic/gen/export_loss_landscape.py` with `--probe-mode auto|training|proxy` and `--batch-size` (default 4); training probe averages greedy forward-loss across N synthetic instances per grid point; bundles `probe_mode` + `batch_size` in NPZ (§G.5.2)
- [x] Export 2D grid of loss values as `.npz`: `loss_grid`, `theta1`, `theta2` keys (§G.5.2)
- [x] React Three Fiber: render grid as vertex-displaced `PlaneGeometry` 3D topography: `LossLandscape3D` lazy chunk (§G.5.2)
- [x] `InstancedMesh` voxel alternative: per-cell `boxGeometry` cubes with height ∝ loss; Loss tab "Surface mesh / InstancedMesh voxels" toggle (§G.5.2)
- [x] Color gradient: low loss = deep blue, high loss = bright red (`lossToColor` vertex colours)
- [x] Camera: orbit, zoom, perspective controls (`OrbitControls` + `Canvas`)
- [x] Overlay 2D ECharts contour map adjacent to the 3D canvas (CSS positioned): `MLIntrospectionPanel` Loss tab side-by-side grid; log-scale colour map when global ``logScale`` on with raw-loss tooltips (§G.5.2 / §G.7)
- [x] Loss landscape 3D terrain follows global ``logScale``: ``LossLandscape3D`` log-transforms height/colour via ``transformMatrixLogScale`` when on; minima sharpness analysis stays on raw loss grid (§G.5.2 / §G.7)
- [x] Project exact-solver solutions (BPC optimum) as a marker on the landscape: `export_loss_landscape.py` bundles `bpc_theta1`/`bpc_theta2`/`bpc_loss`; `load_npz_vectors` + `resolveBpcMarker` + amber octahedron in `LossLandscape3D` + ECharts `markPoint` on contour (§G.5.2)
- [x] Identify sharp vs flat minima; annotate with generalization notes (Gamma-3 vs Empirical): `analyzeLossMinima` Laplacian sharpness + ``generalizationNote`` per basin label on 3D terrain + Loss tab (§G.5.2)

#### 5.3 Attention Weight Visualization (Sigma.js overlay)
- [x] Load attention weight matrices from TensorDict for a selected simulation step: `load_tensor_slice` with leading-dim indices + decode-step slider (§G.5.3)
- [x] Render as bipartite graph on top of node coordinates: edge opacity ∝ attention weight magnitude: ECharts `buildAttentionGraphOption` + Sigma.js WebGL `AttentionSigmaView` (ForceAtlas2, lazy `sigma` chunk) with graph preset loader; View toggle: Heatmap / ECharts graph / Sigma.js (§G.5.3)
- [x] Attention head selector: `detectHeadAxis` + per-head index dropdown; Q/K/V role filter + per-role colour palettes via `classifyAttentionRole` / `groupAttentionKeys` (§G.5.3)
- [x] Timeline slider: step through sequential decoding steps: decode-step range on Attention tab (§G.5.3)
- [x] Sparse Routing Transformer mode: `applySparseTopK` keeps top-k connections per query row (§G.5.3)
- [x] Spherical k-means query-row clustering: `sphericalKMeans` + row reorder + ECharts `markArea` cluster bands; K-means selector (2–8) on Attention tab (§G.5.3)
- [x] Compare attention patterns of model trained on Empirical vs Gamma-3 distributions: Attention tab "Empirical vs Gamma-3" compare mode; dual archive picker; `inferDistributionLabel` path heuristics; side-by-side heatmaps + overlay Δ diff (§G.5.3)
- [x] Side-by-side vs overlay toggle: decode-step compare (side-by-side dual heatmap / overlay Δ diff) (§G.5.3)
- [x] Attention weight heatmaps follow global ``logScale``: ``MLIntrospectionPanel`` log-transforms raw Q/K/V weight cells when on; overlay/distribution Δ diff panels stay linear; tooltips show raw weights (§G.5.3 / §G.7)
- [x] Attention bipartite graph overlays follow global ``logScale``: ``buildAttentionGraphOption`` + ``AttentionSigmaView`` log-transform edge opacity/width via ``attentionWeightDisplay``; tooltips and edge weight attributes retain raw attention values (§G.5.3 / §G.7)
- [x] ML introspection ECharts PNG/SVG export: ``exportChartPng()`` / ``exportChartSvg()`` on ``MLIntrospectionPanel`` attention heatmap (primary + compare panels), attention bipartite graph, and loss contour map (§G.5 / §G.7)
- [x] Loss landscape 3D terrain PNG export: ``exportContainerCanvasPng()`` on ``LossLandscape3D`` R3F canvas (surface mesh + InstancedMesh voxels) via ``MLIntrospectionPanel`` Loss tab (§G.5.2 / §G.7)
- [x] Attention Sigma.js WebGL PNG export: ``exportContainerCanvasPng()`` on ``AttentionSigmaView`` canvas via ``MLIntrospectionPanel`` Attention tab (§G.5.3 / §G.7)
- [x] ML Introspection tensor archive path uses shared ``OpenPathToolbar`` (two-hundred-and-thirtieth pass; §G.5 / §D.7)

**Status**: §G.5 complete — all checklist items delivered.

---

### §G.6 — Phase 6: OLAP Data Cube Explorer

**Goal**: Give the researcher a free-form SQL/pivot interface backed by DuckDB-Wasm for custom analysis queries.

- [x] DuckDB-Wasm query editor with syntax highlighting (Monaco or CodeMirror): `SqlQueryPanel` lazy Monaco SQL editor on Data Explorer + standalone `OlapExplorer` page with table picker + CSV/JSONL ingest (prefers ``.arrow`` sidecars; §G.6)
- [x] Portfolio SQL panels: `SqlQueryPanel` on Benchmark Analysis (`benchmark_sim`) and City Comparison (`city_sim`) when multi-run portfolios are loaded (§G.6)
- [x] Portfolio query templates: `portfolioSqlTemplates()` cross-run robustness, run leaderboard, run×policy variance, Pareto-by-run; `SqlQueryPanel` `portfolioMode` on multi-log views (§G.6)
- [x] Algorithm Comparison DuckDB ingest: `runSimulationArrowPipeline()` → `algorithm_sim` + `SqlQueryPanel` + timing badge when Simulation Monitor watch path is active (§G.6)
- [x] Algorithm Comparison SQL templates: `algorithmSqlTemplates()` policy ranking, worst overflow days, zero-overflow rate, day-over-day profit Δ; `SqlQueryPanel` `algorithmMode` (§G.6)
- [x] Algorithm Comparison brush SQL sync: chart click → global policy filter → `brushSqlSync` + `autoRunOnBrushSync` on `algorithm_sim` (§G.6)
- [x] Benchmark Analysis brush SQL sync: efficiency ranking + metric bar click → global policy filter → `brushSqlSync` + `autoRunOnBrushSync` on `benchmark_sim` (§G.6)
- [x] City Comparison brush SQL sync: city chart / summary table click → `run_label` filter → `brushSqlSync` + `autoRunOnBrushSync` on `city_sim`; `brushedPortfolioSql()` unifies policy + run_label brushes (§G.6)
- [x] Simulation Summary portfolio run_label brush SQL sync: comparison-run click, city chart click, portfolio efficiency ranking click → `highlightRunLabels` + `brushSqlSync` on `summary_sim` (§G.6)
- [x] Simulation Summary ``useLogPathRunLabelBrush`` on primary log open; ``GlobalFilterBar`` ``runLabels`` in single-log mode; comparison-run ring highlight via ``runLabelMapFromPaths`` (hundred-seventy-fifth pass; §G.1 / §G.16 / §D.7)
- [x] Benchmark Analysis + City Comparison loaded-run list ring highlight + click-to-brush via ``runLabelMapFromPaths`` + ``handleRunLabelClick``; single-run ``GlobalFilterBar`` ``runLabels`` (hundred-seventy-sixth pass; §G.1 / §G.6 / §D.7)
- [x] Portfolio loaded-run lists — ``LoadedRunRow`` + ``PathRunLabelChip`` on Benchmark Analysis, City Comparison, Simulation Summary comparison runs, and Output Browser run-directory list (hundred-eighty-first pass; §G.1 / §G.6 / §G.14 / §D.7)
- [x] Launcher/monitor live panel headers — ``RunLabelHeaderSuffix`` + ``PathRunLabelChip`` on sim/data-gen/eval/train/HPO live cards when process stdout resolves a log path (hundred-eighty-second pass; §G.9–§G.18 / §D.7)
- [x] Process Monitor process rows + live panel footers — ``PathRunLabelChip`` when stdout resolves a log path; Command Palette recent files path-chip brush parity (hundred-eighty-third pass; §G.7 / §G.15 / §D.7)
- [x] Training Monitor run discovery list + per-run panel headers — ``LoadedRunRow`` + ``PathRunLabelChip`` on Lightning log directories; Process Monitor row muted process-id suffix (hundred-eighty-fourth pass; §G.17 / §G.15 / §D.7)
- [x] Experiment Tracker MLflow run table + output directory list — ``PathRunLabelChip`` / ``LoadedRunRow`` on ``artifact_uri`` and ``assets/output`` paths; ``GlobalFilterBar`` ``runLabels`` from selected MLflow runs (hundred-eighty-fifth pass; §G.18 / §G.14 / §D.7)
- [x] HPO Tracker trial health table + storage/report paths — ``PathRunLabelChip`` on trial ``log_dir``, SQLite storage URL, and exported Plotly report directory; ``GlobalFilterBar`` ``runLabels`` from selected trials + post-run paths (hundred-eighty-sixth pass; §G.18 / §D.7)
- [x] Training Monitor logs root + checkpoint browser — ``PathRunLabelChip`` on ``logs/`` directory and per-checkpoint rows with parent-run brush label (hundred-eighty-seventh pass; §G.17 / §G.12 / §D.7)
- [x] Configuration Editor + ML Introspection open-file headers — ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on YAML config and tensor archive paths (hundred-eighty-seventh pass; §G.13 / §G.5 / §D.7)
- [x] Output Browser checkpoint browser + file viewer — ``PathRunLabelChip`` on checkpoint sidebar rows, artefact viewer header, and checkpoint preview with parent-run brush label (hundred-eighty-eighth pass; §G.14 / §G.12 / §D.7)
- [x] Evaluation Runner + Training Hub eval checkpoint inputs — ``PathRunLabelChip`` below filled checkpoint paths for click-to-brush parity (hundred-eighty-eighth pass; §G.12 / §G.10 / §D.7)
- [x] Configuration Editor diff comparison file — ``PathRunLabelChip`` + ``useLogPathRunLabelBrush`` on ``diffPath``; diff summary chip parity (hundred-eighty-eighth pass; §G.13 / §D.7)
- [x] Evaluation Runner + Benchmark Analysis eval results tables — ``PathRunLabelChip`` on checkpoint rows with parent-run ``brushLabel`` when Hydra path known (hundred-eighty-ninth pass; §G.12 / §G.1 / §D.7)
- [x] EvalResultCard + Process Monitor / Training Hub eval panels — checkpoint header ``PathRunLabelChip`` parity (hundred-eighty-ninth pass; §G.10 / §G.12 / §G.15 / §D.7)
- [x] Output Browser ``.wsroute`` manifest file table — ``PathRunLabelChip`` on bundle member paths with selected-run brush label (hundred-eighty-ninth pass; §G.8 / §G.14 / §D.7)
- [x] Output Browser ``.wsroute`` manifest member paths — auto-classified ``PathRunLabelChip`` handoffs for JSONL / CSV / config / checkpoint members (two-hundred-and-thirty-second pass; §G.8 / §G.14 / §D.7)
- [x] EvalCheckpointLiveCard + launcher eval live panels — ``PathRunLabelChip`` on per-checkpoint live rows when Hydra path known (hundred-ninetieth pass; §G.12 / §G.10 / §G.15 / §D.7)
- [x] Evaluation Runner + Training Hub eval dataset inputs — ``PathRunLabelChip`` below filled dataset paths (hundred-ninetieth pass; §G.12 / §G.10 / §D.7)
- [x] Data Generation Wizard TSPLIB + sensor source paths — ``PathRunLabelChip`` on external data source file inputs (hundred-ninetieth pass; §G.11 / §D.7)
- [x] PolicyTelemetryTrendsPanel SQLite store path — ``PathRunLabelChip`` on ``db_path`` header (hundred-ninetieth pass; §G.7 / §A.3 / §D.7)
- [x] Settings project root + Python path — ``PathRunLabelChip`` below filled path inputs (hundred-ninety-first pass; §G.19 / §D.7)
- [x] Experiment Tracker MLflow tracking URI — ``PathRunLabelChip`` below filled tracking URI when local path resolves (hundred-ninety-first pass; §G.18 / §D.7)
- [x] HPO Tracker Optuna storage URL — ``PathRunLabelChip`` below filled storage input; inline chip parity with eval dataset inputs (hundred-ninety-first pass; §G.18 / §D.7)
- [x] HPO Tracker storage/report relative-path resolution — ``sqliteStoragePathFromUrl`` + ``projectRoot``-resolved report dir chips (hundred-ninety-second pass; §G.18 / §D.7)
- [x] Data Generation Wizard instance preview path — ``PathRunLabelChip`` on previewed ``.pkl`` / ``.pt`` dataset (hundred-ninety-second pass; §G.11 / §D.7)
- [x] Settings Arrow benchmark + import JSON paths — ``PathRunLabelChip`` on benchmark CSV/JSONL + imported settings file (hundred-ninety-second pass; §G.19 / §D.7)
- [x] PolicyTelemetryTrendsPanel ``db_path`` — ``resolveLocalProjectPath`` before path-chip brush (hundred-ninety-second pass; §G.7 / §A.3 / §D.7)
- [x] Launcher workflow path chips — ``PathRunLabelChip`` ``projectRoot`` prop on eval checkpoint/dataset, data-gen source/preview, config editor, ML introspection, and Settings secondary paths (hundred-ninety-third pass; §G.5 / §G.10–§G.13 / §G.19 / §D.7)
- [x] Eval live/result cards + Benchmark Analysis eval table — ``projectRoot``-resolved checkpoint ``brushLabel`` parity (hundred-ninety-third pass; §G.1 / §G.12 / §G.15 / §D.7)
- [x] ``PathRunLabelChip`` store fallback — auto ``projectRoot`` resolution for analysis/monitor/file path chips when prop omitted (hundred-ninety-fourth pass; §G.1 / §G.14–§G.18 / §D.7)
- [x] HPO Tracker trial ``log_dir`` + Experiment Tracker MLflow ``artifact_uri`` — ``projectRoot``-resolved path-chip brush parity (hundred-ninety-fourth pass; §G.18 / §D.7)
- [x] Training Monitor + Output Browser — ``projectRoot``-resolved checkpoint + run-directory path-chip brush parity (hundred-ninety-fourth pass; §G.14 / §G.17 / §G.12 / §D.7)
- [x] ``useLogPathRunLabelBrush`` + ``LoadedRunRow`` — ``projectRoot``-resolved portfolio/open-file brush sync + ring-highlight parity (hundred-ninety-fifth pass; §G.1 / §G.6 / §G.14–§G.17 / §D.7)
- [x] Simulation Summary + Data Explorer + OLAP Explorer + Algorithm Comparison + Simulation Monitor + Process Monitor — open-file/process-row path-chip ``projectRoot`` parity (hundred-ninety-fifth pass; §G.1 / §G.6 / §G.15 / §G.16 / §D.7)
- [x] Derived run-label utilities + live-panel headers — ``runLabelFromLogLines`` / ``useProcessRunLabelBrush`` / ``runLabelMapFrom*`` ``projectRoot`` resolution; Command Palette + Process Monitor row ring-highlight + live-panel footer/header path-chip parity (hundred-ninety-sixth pass; §G.7 / §G.9–§G.18 / §D.7)
- [x] DuckDB Arrow ingest + explicit live-panel ``projectRoot`` — ``runLabelFromSourcePath`` + pipeline ``projectRoot`` annotation; Simulation Summary portfolio labels; Policy Telemetry db_path chip; launcher/train/HPO page header/footer parity (hundred-ninety-seventh pass; §G.0 / §G.6 / §G.9–§G.18 / §A.3 / §D.7)
- [x] Portfolio DuckDB union ingest ``projectRoot`` — ``portfolioRunLabel`` + ``runPortfolioSimulationArrowPipeline`` ``projectRoot``; Benchmark Analysis + City Comparison portfolio label parity; Simulation Summary + OLAP Explorer multi-log callers (hundred-ninety-eighth pass; §G.0 / §G.1 / §G.1.6 / §G.6 / §D.7)
- [x] Simulation Summary portfolio UI ``run_label`` — ``portfolioRunLabel`` on add-comparison-run, output-portfolio load, ``allRuns`` brush, and ``allDuckDbLogs`` ingest (hundred-ninety-ninth pass; §G.1 / §G.6 / §D.7)
- [x] Benchmark Analysis + City Comparison portfolio UI ``run_label`` — ``normalizedRuns`` + ``portfolioDuckDbLogs`` ``portfolioRunLabel`` on loaded-run list, portfolio brush, and DuckDB ingest (two-hundredth pass; §G.1 / §G.1.6 / §G.6 / §D.7)
- [x] OLAP Explorer custom JSONL ingest ``run_label`` — ``portfolioRunLabel`` on ingest path for DuckDB brush/SQL parity (two-hundredth pass; §G.6 / §D.7)
- [x] Data Explorer single-log open-file ``run_label`` — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push (two-hundred-and-first pass; §G.6 / §D.7)
- [x] Algorithm Comparison + Simulation Monitor single-log ``run_label`` — ``sourceRunLabel`` via ``portfolioRunLabel`` on filter bar, DuckDB ``SqlQueryPanel``, Policy Telemetry Trends, and recent-file push (two-hundred-and-first pass; §G.16 / §D.7)
- [x] Simulation Summary recent-file push ``run_label`` — ``portfolioRunLabel`` on log open for Command Palette brush parity (two-hundred-and-first pass; §G.1 / §D.7)
- [x] Output Browser compare handoff ``run_label`` — ``compareSelectedRuns`` refs use ``portfolioRunLabel`` + ``pushRecent`` for Benchmark Analysis handoff (two-hundred-and-third pass; §G.14 / §G.1 / §D.7)
- [x] Simulation Summary add-comparison-run recent-file push — ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-third pass; §G.1 / §D.7)
- [x] Benchmark Analysis + City Comparison ``pendingBenchmarkLogs`` recent-file push — ``pushRecent`` on Output Browser compare consume (two-hundred-and-third pass; §G.1 / §G.1.6 / §D.7)
- [x] Output Browser Simulation Summary handoff recent-file push — ``openInSimSummary`` + ``extractBundleAndOpen`` ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-fourth pass; §G.14 / §G.1 / §D.7)
- [x] Global file drop + wsroute import recent-file push — ``useGlobalFileDrop`` + ``useWsrouteImport`` ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-fourth pass; §G.8 / §G.14 / §D.7)
- [x] Portfolio load recent-file push — ``loadOutputPortfolio`` on Benchmark Analysis, City Comparison, and Simulation Summary (two-hundred-and-fourth pass; §G.1 / §G.1.6 / §D.7)
- [x] Command Palette recent-file label refresh — ``portfolioRunLabel`` on log/run/csv open (two-hundred-and-fourth pass; §G.7 / §D.7)
- [x] ``refreshRecentLabels`` store action — persisted recent-file labels re-derived on palette open + ``projectRoot`` change (two-hundred-and-fifth pass; §G.7 / §D.7)
- [x] Command Palette CSV handoff — ``pendingCsvPath`` + Data Explorer consume for recent CSV open parity (two-hundred-and-fifth pass; §G.6 / §G.7 / §D.7)
- [x] Output Browser inline file open recent-file push — ``.jsonl`` / ``.csv`` tree viewer ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-fifth pass; §G.14 / §G.8 / §D.7)
- [x] ``RecentFileKind`` ``training`` — Lightning log directory recent-file kind (two-hundred-and-sixth pass; §G.17 / §G.7 / §D.7)
- [x] Training Monitor run select + ``pendingTrainingRunPath`` + post-run auto-select recent-file push — ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-sixth pass; §G.17 / §D.7)
- [x] Command Palette training recent-file handoff — ``pendingTrainingRunPath`` + Training Monitor mode (two-hundred-and-sixth pass; §G.7 / §G.17 / §D.7)
- [x] Launcher / train-HPO nav-mesh Output Browser + Training Monitor handoff recent-file push — ``portfolioRunLabel`` (two-hundred-and-sixth pass; §G.9–§G.12 / §G.15 / §G.17 / §D.7)
- [x] ``RecentFileKind`` ``checkpoint`` — model checkpoint recent-file kind (two-hundred-and-seventh pass; §G.12 / §G.7 / §D.7)
- [x] Training Monitor + Output Browser + ``LauncherNavMesh`` eval checkpoint handoff recent-file push — ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-seventh pass; §G.12 / §G.14 / §G.17 / §D.7)
- [x] Evaluation Runner + Training Hub checkpoint pick / ``pendingCheckpoint`` consume recent-file push — ``portfolioRunLabel`` (two-hundred-and-seventh pass; §G.10 / §G.12 / §D.7)
- [x] Command Palette checkpoint recent-file handoff — ``pendingCheckpoint`` + Evaluation Runner mode (two-hundred-and-seventh pass; §G.7 / §G.12 / §D.7)
- [x] ``RecentFileKind`` ``config`` — YAML / TOML / cfg / ini config recent-file kind (two-hundred-and-eighth pass; §G.13 / §G.7 / §D.7)
- [x] Configuration Editor open + ``pendingConfigPath`` consume recent-file push — ``portfolioRunLabel`` (two-hundred-and-eighth pass; §G.13 / §D.7)
- [x] Output Browser config open + **Open in Config Editor →** handoff recent-file push — ``pendingConfigPath`` (two-hundred-and-eighth pass; §G.14 / §G.13 / §D.7)
- [x] Command Palette config recent-file handoff — ``pendingConfigPath`` + Config Editor mode (two-hundred-and-eighth pass; §G.7 / §G.13 / §D.7)
- [x] ``recentKindFromPath`` + ``useGlobalFileDrop`` multi-kind drop — ``.csv`` / checkpoint / config ``portfolioRunLabel`` + pending handoffs (two-hundred-and-ninth pass; §G.8 / §G.6 / §G.12 / §G.13 / §D.7)
- [x] Output Browser checkpoint open + **Open in Data Explorer →** CSV handoff recent-file push — ``portfolioRunLabel`` (two-hundred-and-ninth pass; §G.14 / §G.12 / §G.6 / §D.7)
- [x] ``recentHandoff.ts`` shared handoff + Command Palette keyboard nav + multi-path / directory drop (two-hundred-and-tenth pass; §G.7 / §G.8 / §G.14 / §G.17 / §D.7)
- [x] Output Browser ``pickOutputDir`` + eval/train CSV dataset pick recent-file push — ``portfolioRunLabel`` (two-hundred-and-tenth pass; §G.14 / §G.10 / §G.12 / §G.6 / §D.7)
- [x] ``applyRecentHandoff`` + studio-wide ``makeRecentEntry`` migration for nav-mesh / Output Browser / page open paths (two-hundred-and-eleventh pass; §G.7 / §G.8 / §G.14 / §D.7)
- [x] Algorithm Comparison ``useLogPathRunLabelBrush`` + ``GlobalFilterBar`` ``runLabels`` on watch path (hundred-seventy-sixth pass; §G.1 / §G.16 / §D.7)
- [x] Data Explorer ``useLogPathRunLabelBrush`` path-derived ``runLabels`` + trends fallback when CSV lacks ``run_label`` column (hundred-seventy-sixth pass; §G.6 / §G.16 / §D.7)
- [x] OLAP Explorer ``useLogPathRunLabelBrush`` on selected ingest path; table picker ring highlight + click-to-brush via ``runLabelMapFromTablePaths``; path-derived ``GlobalFilterBar`` ``runLabels`` when table lacks ``run_label`` column (hundred-seventy-seventh pass; §G.6 / §G.16 / §D.7)
- [x] OLAP Explorer built-in DuckDB portfolio tables — ``useTableRunLabelBrush`` + ``runLabelMapFromSingleTableLabels`` table-picker ring highlight + click-to-brush when ``summary_sim`` / ``benchmark_sim`` / ``city_sim`` / ``algorithm_sim`` are loaded without custom ingest paths (hundred-seventy-eighth pass; §G.6 / §G.14 / §D.7)
- [x] Single-log DuckDB ingest — ``annotateTableWithRunLabelIfMissing`` on ``runSimulationArrowPipeline`` / ``runCsvArrowPipeline``; Simulation Monitor + Algorithm Comparison ``SqlQueryPanel`` run-label ``brushSqlSync`` (hundred-seventy-ninth pass; §G.6 / §G.16 / §D.7)
- [x] File-path chip brush parity — ``PathRunLabelChip`` + ``useRunLabelBrushToggle`` on Simulation Monitor, Algorithm Comparison, and Data Explorer open-file headers (hundred-seventy-ninth pass; §G.14–§G.16 / §D.7)
- [x] Benchmark Analysis city chart run_label brush: city comparison chart click → `highlightRunLabels` + `brushSqlSync` on `benchmark_sim` (§G.6)
- [x] OLAP Explorer global policy brush SQL sync: `GlobalFilterBar` policy → `brushSqlSync` + `autoRunOnBrushSync`; portfolio/algorithm template modes per ingested table (§G.6)
- [x] OLAP Explorer global run_label brush SQL sync: `GlobalFilterBar` run selector + `highlightRunLabels` on portfolio tables; distinct ``run_label`` values from DuckDB (§G.6)
- [x] SQL result row + pivot run_label cross-filter: click policy or ``run_label`` cell → `useGlobalFiltersStore` → `brushSqlSync` + row dimming (§G.6)
- [x] Portfolio global run_label filter bar: `usePortfolioRunBrush` + `GlobalFilterBar` run selector on Simulation Summary, Benchmark Analysis, and City Comparison when ≥2 runs loaded (§G.6)
- [x] Portfolio global city/scale filter bar: `brushedCity` in `useGlobalFiltersStore` + `GlobalFilterBar` city selector on Summary/Benchmark/City when ≥2 city groups loaded (§G.6)
- [x] OLAP Explorer global city/scale brush SQL sync: `groupRunLabelsByCity()` + `GlobalFilterBar` city selector on portfolio tables; `resolveBrushedRunLabels()` expands city brush to ``run_label`` IN clause via `SqlQueryPanel` ``portfolioRunLabels`` (§G.6)
- [x] Global filter bar → SQL brush sync: `SqlQueryPanel` ``brushFilter`` merges ``useGlobalFiltersStore`` policy / ``run_label`` / city brush when chart props are absent; ``autoRunOnBrushSync`` fires on filter-bar changes (§G.6)
- [x] Portfolio DuckDB ``city_scale`` column: `runPortfolioSimulationArrowPipeline()` adds parsed city/scale label alongside ``run_label``; city leaderboard SQL template (§G.6)
- [x] Portfolio single-log ``run_label`` + ``city_scale`` columns: `runPortfolioSimulationArrowPipeline()` always annotates logs (including one-run Summary/Benchmark/City/OLAP ingests) (§G.6)
- [x] SQL result row ``city_scale`` cross-filter: click ``city_scale`` cell → global ``brushedCity``; row dimming + active highlight (§G.6)
- [x] Pivot table ``city_scale`` cross-filter: `PivotTablePanel` row highlight + click sets global ``brushedCity`` (§G.6)
- [x] City×policy matrix SQL template: `portfolioSqlTemplates()` grouped ``city_scale`` × ``policy`` kg/km matrix (§G.6)
- [x] Auto-chart portfolio GROUP BY detection: `queryAutoChart.ts` prefers ``city_scale`` / ``run_label`` / ``policy`` dimensions + KPI metrics (§G.6)
- [x] Data Explorer global filter bar + SQL brush sync when CSV has ``policy`` column (§G.6)
- [x] Data Explorer CSV-derived policy / ``run_label`` / city filter bar + row cross-filter dimming (§G.6)
- [x] OLAP dynamic portfolio mode: `duckDbHasColumn()` detects ``run_label`` on any ingested table (§G.6)
- [x] Auto-chart grouped bar for multi-dimension GROUP BY (``city_scale`` × ``policy``; §G.6)
- [x] Auto-chart heatmap for city×policy / run×policy matrix query results: `queryAutoChart.ts` ``heatmap`` type (§G.6)
- [x] OLAP Explorer DuckDB-derived policy / ``city_scale`` filter bar: ``listDuckDbDistinctValues()`` on active table (§G.6)
- [x] Data Explorer cell-level cross-filter: click brush column cell only (policy / ``run_label`` / ``city_scale``) (§G.6)
- [x] Data Explorer brush-aware CSV export: export respects global filter + text search + sort (§G.6)
- [x] SQL result grid cell-level cross-filter: click brush column cell only in ``SqlQueryPanel`` (§G.6)
- [x] Auto-chart click cross-filter: bar / grouped-bar / heatmap clicks apply global policy / ``run_label`` / ``city_scale`` brush (§G.6)
- [x] Auto-chart PNG export: ``exportChartPng()`` on ``SqlQueryPanel`` auto-chart (§G.6)
- [x] Auto-chart type override: ``suggestChartAlternatives()`` chips switch bar / grouped-bar / heatmap (§G.6)
- [x] Run×policy matrix SQL template: ``portfolio-run-policy-matrix`` in ``portfolioSqlTemplates()`` (§G.6)
- [x] Pareto efficiency frontier SQL template: ``pareto-frontier`` + ``portfolio-pareto-frontier`` in ``duckdbTemplates.ts`` (§G.6)
- [x] Auto-chart scatter cross-filter: labeled profit vs overflows scatter click → global policy / ``run_label`` / ``city_scale`` brush (§G.6)
- [x] Auto-chart SVG export: ``exportChartSvg()`` on ``SqlQueryPanel`` auto-chart (§G.6)
- [x] Pre-built query templates: robustness profile, variance analysis, Pareto efficiency frontier: `duckdbTemplates.ts` template chips (§G.6)
- [x] Result grid with sortable columns, row filter search, and filtered CSV export: `SqlQueryPanel` sortable result table + search box + export respects filter (§G.6)
- [x] Auto-chart: map query result columns to ECharts chart type suggestions: `queryAutoChart.ts` + `SqlQueryPanel` bar/line/scatter/heatmap suggestion below results (§G.6)
- [x] Pivot table UI: drag dimensions/measures onto row/column/value wells: `PivotTablePanel` draggable column chips + HTML5 drop wells for row/column/value + agg selector + heatmap on `SqlQueryPanel` (§G.6)
- [x] Cross-filtering from pivot table updates all Phase 1–2 charts bidirectionally: pivot/result row click sets `useGlobalFiltersStore` policy; `GlobalFilterBar` policy highlights matching SQL rows + dims pivot heatmap rows via `highlightRowLabels` (§G.6)
- [x] Auto-chart Pareto frontier step-line overlay: labeled profit vs overflows scatter highlights frontier points + dashed ``paretoStepLine()`` (§G.6)
- [x] Auto-chart log-scale on profit vs overflows scatter: symlog overflows y-axis + log profit x-axis when global ``logScale`` on (§G.6 / §G.1 / §G.7)
- [x] Auto-chart log-scale on bar / grouped-bar / line when y-axis metric is overflow, loss, or KPI (§G.6 / §G.7)
- [x] Auto-chart heatmap log-scale visualMap: matrix cell values transformed via ``displayBarValue`` when global ``logScale`` on (§G.6 / §G.7)
- [x] Pivot table heatmap log-scale: ``PivotTablePanel`` passes global ``logScale`` + value column to ``pivotHeatmapOption`` (§G.6 / §G.7)
- [x] Auto-chart line cross-filter: time-series point click → ``onDaySelect`` when ``xKey`` is ``day`` (§G.6)
- [x] Auto-chart line type in override alternatives for day/epoch/step queries (§G.6)
- [x] Pivot table heatmap PNG export: ``exportChartPng()`` on ``PivotTablePanel`` pivot heatmap with toast feedback (§G.6 / §G.7)
- [x] OLAP Explorer ingest path + Data Explorer open-CSV path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-sixth pass; §G.6 / §D.7)
- [x] OLAP / Data Explorer open-path chips migrate to ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.6 / §D.7)
- [x] OLAP Explorer ingest path toolbar uses shared ``OpenPathToolbar`` labeled+chip cluster (two-hundred-and-twenty-eighth pass; §G.6 / §D.7)
- [x] Data Explorer open-CSV toolbar uses shared ``OpenPathToolbar`` with export ``children`` (two-hundred-and-twenty-ninth pass; §G.6 / §G.7 / §D.7)

**Status**: §G.6 complete — all checklist items delivered.

---

### §G.7 — Phase 7: Integrated Workflow & UX Polish

**Goal**: Connect all analytics phases into a single cohesive analytical narrative flow, and satisfy all §D UX requirements.

- [x] App-level navigation: `WorkflowNav` strip — Overview → Drill-Down → Geospatial → Registry → ML → HPO → Launch (§G.7)
- [x] Global filter state management (Zustand): `useGlobalFiltersStore` + `GlobalFilterBar` propagates policy/sample filters across SimulationMonitor, AlgorithmComparison, SimulationSummary, and BenchmarkAnalysis
- [x] Bookmarkable analysis states (serialize filter + view to URL hash for deep-linking via `useHashSync`)
- [x] Bookmarkable ``run_label`` filter: `useHashSync` serializes global ``runLabel`` as ``r`` query param; restored on load and browser back/forward (§G.7)
- [x] Bookmarkable city/scale brush: `useHashSync` serializes global ``brushedCity`` as ``c`` query param; restored on load and browser back/forward (§G.7)
- [x] Global log-scale filter: ``logScale`` in ``useGlobalFiltersStore`` + ``GlobalFilterBar`` toggle propagates to Simulation Summary (incl. per-day trajectory + policy radar + policy/portfolio parallel coordinates + hierarchy drill-down profit bars + drill-down error-bar whiskers + grouped metric bar whiskers + city-comparison error-bar whiskers + Pareto symlog scatter + policy configuration heatmaps), Benchmark Analysis (incl. portfolio parallel + Pareto panels + graph heatmaps + multi-run metric-bar error-bar whiskers + city-comparison error-bar whiskers + efficiency-ranking error-bar whiskers), Algorithm Comparison (radar + metric bars + error-bar whiskers), City Comparison (city-comparison error-bar whiskers), Evaluation Runner, Training Monitor, Training Hub, HPO Tracker (incl. parallel coordinates objective axis), Experiment Tracker (ZenML step durations + ML loss contour + 3D loss terrain + attention weight heatmaps + attention bipartite graph overlays), Simulation Monitor daily KPI charts + graph topology ACO pheromone edge styling, Data Generation demand histogram, OLAP/Data Explorer auto-charts (incl. symlog profit vs overflows scatter + heatmap visualMap) and pivot heatmaps (§G.1 / §G.7)
- [x] Bookmarkable log-scale toggle: `useHashSync` serializes global ``logScale`` as ``l=1`` query param; restored on load and browser back/forward (§G.7)
- [x] Dark/light theme toggle with Tauri Store persistence (§D.3, §D.4): `TopBar` toggle + Settings appearance radio; `useAppStore` Zustand `persist`
- [x] Keyboard shortcuts: `G` → simulation monitor, `Q` → HPO tracker, `P` → process monitor, `M` → map/simulation twin, `T`/`H`/`E` → train/HPO workflow, `L`/`D`/`V` → sim/data-gen/eval launchers, `Ctrl+.` → cancel first running process, `Ctrl+Shift+P` → process monitor, `Ctrl+R` → launch on active launcher page, digits `1`–`8` → quick nav, `?` → shortcuts help overlay (§D.7)
- [x] Keyboard shortcuts help overlay: `KeyboardShortcutsHelp` modal + TopBar button; `Escape` dismisses
- [x] Lazy-loaded page components: all 17 views behind `React.lazy` + `Suspense` in `App.tsx` (§G.7)
- [x] Command palette: `CommandPalette` fuzzy-search overlay for all views + actions; `Ctrl+K` / TopBar search button; arrow keys + Enter navigation over **recents and commands** via shared ``applyRecentHandoff`` (two-hundred-and-eleventh pass; §G.7 / §D.7)
- [x] Vite `manualChunks`: echarts, maplibre, deck.gl, monaco, duckdb, r3f, sigma split into separate vendor bundles (§G.7)
- [x] Sidebar page prefetch: `prefetchPage()` warms lazy route chunks on nav item hover
- [x] Command palette bundle import: "Import .wsroute Bundle" action via `useWsrouteImport` hook
- [x] Recent files quick open: `useRecentFilesStore` persisted list (log/run/csv/training/checkpoint/config); command palette Recent section; shared ``recentKindFromPath`` + ``applyRecentHandoff`` / ``makeRecentEntry`` for drop/open parity (two-hundred-and-eleventh pass; §G.7 / §G.8 / §D.7)
- [x] Startup route prefetch: `App.tsx` warms all 18 lazy route chunks (monitor, analytics, launch, files, settings) on mount (§G.7)
- [x] Startup vendor prefetch: echarts, maplibre-gl, @deck.gl/react, @monaco-editor/react, @duckdb/duckdb-wasm, sigma, @react-three/fiber + DeckRouteMap warmed on mount (§G.7)
- [x] Startup timing probe: `useStartupTiming` reports module-load → first React mount + route prefetch complete in Settings About (§G.7)
- [x] React toast notifications + Tauri OS notifications for background job completion when window is not focused (§D.8)
- [x] Responsive layout: `Layout` max-width `1920px` container, `sm:` padding breakpoints, `lg:` grid columns; collapsible sidebar with mobile overlay backdrop (`useLayoutStore`); sidebar auto-collapses below `lg` breakpoint via `matchMedia`; analytics chart grids use `grid-cols-1 sm:grid-cols-2` / `md:grid-cols-2` breakpoints (§G.7)
- [x] BenchmarkAnalysis responsive chart grids: Pareto panels `md:grid-cols-2`, metric bars `sm:grid-cols-2`, eval checkpoint charts `sm:grid-cols-2 lg:grid-cols-3` (§G.7)
- [x] AlgorithmComparison responsive chart grids: metric bars `sm:grid-cols-2 lg:grid-cols-4` (§G.7)
- [x] EvaluationRunner responsive inline chart grid: `sm:grid-cols-2 lg:grid-cols-3` (§G.12 / §G.7)
- [x] Performance budget probe: Settings About shows prefetch timing vs 2s target with pass/fail badge; "Run Chart Render Benchmark" measures representative ECharts first-paint vs 500 ms budget (§G.7)
- [x] Settings Arrow benchmark uses shared `formatPipelineTimingBadge()` for last-ingest summary (§G.0 / §G.7)
- [x] Export helpers with toast feedback: ``exportChartPngWithToast()`` / ``exportChartSvgWithToast()`` / ``exportContainerCanvasPngWithToast()`` / ``exportCanvasPngWithToast()`` centralise Sonner success/failure toasts in ``chartExport.ts``; ``ChartExportButtons`` pairs PNG + SVG on ECharts panels; ``CanvasExportButton`` wraps WebGL/canvas PNG export (§G.7)
- [x] ``ChartExportButtons`` propagated to portfolio facets, OLAP pivot/auto-chart, route-map preview, graph topology ECharts view, and ML introspection ECharts panels (§G.7)
- [x] ``CanvasExportButton`` propagated to deck.gl route map, graph topology Sigma.js/Cosmograph WebGL views, and ML introspection Attention Sigma.js + LossLandscape3D R3F canvas exports (§G.7)
- [x] Export: ECharts PNG export via ``exportChartPngWithToast()`` on SimulationMonitor, SimulationSummary (trajectory + radar + heatmap + Pareto + efficiency ranking + bar charts), AlgorithmComparison (radar + bar charts), BenchmarkAnalysis (sim + eval charts incl. kg/km), BenchmarkParetoPanel (per-facet Pareto scatter), BenchmarkPortfolioParallel, BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap (facet heatmaps), BenchmarkPortfolioHeatmap, PortfolioEfficiencyRanking, TrainingMonitor (overlay + sparklines), TrainingHub (live chart + sparklines), DataGeneration (demand histogram), ExperimentTracker, HPOTracker charts, GraphTopologyPanel (ECharts view), MLIntrospectionPanel (attention heatmap primary + compare, attention graph, loss contour), PivotTablePanel (pivot heatmap), SqlQueryPanel auto-chart; WebGL/canvas PNG via ``CanvasExportButton`` (``exportContainerCanvasPngWithToast()`` / ``exportCanvasPngWithToast()``) on ``DeckRouteMap`` (Mercator tile / OrbitView Cartesian), GraphTopologyPanel (Sigma.js + Cosmograph views), and MLIntrospectionPanel (LossLandscape3D terrain + AttentionSigmaView); ECharts SVG via ``exportChartSvgWithToast()`` / ``ChartExportButtons`` on SimulationMonitor (route map + daily KPI timeseries), SimulationSummary (trajectory + radar + heatmap + parallel + hierarchy + Pareto + efficiency ranking + bar charts + city comparison), AlgorithmComparison (radar + metric bars), BenchmarkAnalysis (sim + eval + efficiency ranking), CityComparison, PortfolioEfficiencyRanking, TrainingMonitor (overlay + sparklines), TrainingHub (live chart + sparklines), DataGeneration (demand histogram), EvaluationRunner (inline checkpoint charts), ExperimentTracker (MLflow metric comparison), HPOTracker (history + importance + cross-study + parallel), ZenMLPipelineView (step durations), GraphTopologyPanel (ECharts view), MLIntrospectionPanel attention/loss charts, BenchmarkParetoPanel, BenchmarkPortfolioParallel, BenchmarkDistributionHeatmap / BenchmarkGraphHeatmap, BenchmarkPortfolioHeatmap, PivotTablePanel, and SqlQueryPanel auto-chart; table CSV via `downloadCsv()` on MLflow runs, ZenML runs, Simulation Summary ranking, Data Explorer; Parquet via `export_csv_to_parquet` / `export_table_parquet` on Data Explorer, Output Browser CSV viewer, Simulation Summary ranking
- [x] Data Explorer: sortable column headers (click header to toggle asc/desc numeric/text sort; §G.6)
- [x] Data Explorer: row filter search box matching any column with filtered/total row count (§G.6)
- [x] Data Explorer: CSV export respects active filter and sort order (exports visible subset; §G.6)
- [x] Command Palette recent-file rows use shared ``OpenPathToolbar``; kind meta as ``children`` + ``handoffOnAfterOpen`` palette close (two-hundred-and-thirty-fifth pass; §G.7 / §D.7)
- [x] ``OpenPathToolbar`` ``handoffOnAfterOpen`` forwarded to labeled + chip handoffs (two-hundred-and-thirty-fifth pass; §G.7 / §D.7)
- [x] Command Palette recent rows: kind meta as ``OpenPathToolbar`` ``children``, stored ``label`` display, non-nested row shell (two-hundred-and-thirty-sixth pass; §G.7 / §D.7)

**Status**: §G.7 complete — all checklist items delivered.

---

### §G.8 — Phase 8: Data Export & Packaging

**Goal**: Make the Studio distributable and extend the Python pipeline to output Studio-compatible data bundles.

- [x] Python export script: `logic/gen/export_for_studio.py` — packages simulation CSV + graph JSONs + TensorDict NPZs + `.td` datasets into a `.wsroute` zip bundle with `manifest.json`; `--arrow` emits Arrow IPC (`.arrow`) sidecars for each CSV and simulation JSONL log (§G.8)
- [x] Rust bundle Arrow export: `create_wsroute_bundle(..., include_arrow)` emits `.arrow` sidecars via `write_csv_arrow_sidecar()` + `write_simulation_log_arrow_sidecar()`; Output Browser checkbox + manifest `arrow_sidecars` count (§G.8)
- [x] Studio sidecar ingest: DuckDB-Wasm pipeline auto-loads sibling `.arrow` when opening CSV or JSONL in Data Explorer / Simulation Summary / OLAP / Settings benchmark (§G.8)
- [x] Rust backend: `inspect_wsroute_bundle` lists bundle contents in Output Browser
- [x] Rust backend: `create_wsroute_bundle` packages a run directory into a `.wsroute` zip with `manifest.json`
- [x] Rust backend: `extract_wsroute_bundle` decompresses a bundle; returns first `.jsonl` path for Simulation Summary
- [x] Output Browser: "Export as .wsroute" on selected run (save dialog); "Extract & Open" on `.wsroute` files
- [x] Output Browser: drag-drop `.wsroute` bundle onto file viewer via Tauri `onDragDropEvent` (`useFileDrop` hook); inspects manifest without directory picker
- [x] Global file drop: `useGlobalFileDrop` in `Layout` extracts `.wsroute` to `assets/output/.imports/` or opens `.jsonl` logs in Simulation Summary; routes `.csv` / checkpoints / configs / training ``logs/`` dirs / ``assets/output`` run dirs via shared ``applyRecentHandoff`` + multi-path push (two-hundred-and-eleventh pass; §G.8 / §G.6 / §G.12 / §G.13 / §G.14 / §G.17 / §D.7)
- [x] Integration test: `wsroute_bundle_round_trip_preserves_jsonl` + `simulation_arrow_sidecar_row_parity` Rust unit tests — create bundle → extract → verify `.jsonl` log content and Arrow sidecar row counts match parsed entries (§G.8)
- [x] Tauri bundler config: `tauri.conf.json` targets `deb`/`appimage`/`msi`/`dmg`; Linux deb section + Windows NSIS; `npm run tauri:build` / `tauri:build:linux` scripts; `createUpdaterArtifacts: true` emits `.sig` sidecars (partial — code-signing keys deferred)
- [x] App version command: `system::get_app_version` surfaced in Settings About (§G.8 / §G.19)
- [x] Update check command: `system::check_for_updates` uses Tauri updater plugin when `WSMART_UPDATER_PUBKEY` + `WSMART_UPDATE_URL` are set; falls back to JSON manifest version compare; Settings "Check for Updates" + conditional "Download & Install" button (§G.8)
- [x] Signed update install: `system::install_app_update` downloads/installs pending signed update via `tauri-plugin-updater`; `updater:default` capability; example manifest at `app/updater.example.json` (partial — release signing keys + CDN hosting deferred)
- [x] Output Browser ``.wsroute`` manifest member paths — auto-classified ``PathRunLabelChip`` handoffs for JSONL / CSV / config / checkpoint members (two-hundred-and-thirty-second pass; §G.8 / §G.14 / §D.7)
- [x] Output Browser ``.wsroute`` manifest members use shared ``OpenPathToolbar`` (two-hundred-and-thirty-fourth pass; §G.8 / §G.14 / §D.7)

**Status**: §G.8 complete — updater plugin wired; code-signing keys and hosted signed releases deferred to release engineering.

---

### §G.9 — Phase 9: Simulation Launcher & Run Manager ✅

**Goal**: Port the PySide6 simulation tab to Tauri/React and add the improvements identified in §D.

- [x] React form: Hydra override textarea → `spawn_python_process main.py test_sim <overrides>`
- [x] Rust backend: spawn `main.py test_sim <overrides>` via `tokio::process::Command`; `process:spawn` event emitted on start; stdout streamed as `process:stdout` events
- [x] Cancel button: sends cancel signal via `tokio::sync::watch` channel (§D.5)
- [x] Toast notification on launch success / failure (§D.8) via `useSpawnProcess` hook
- [x] React form: full parameter set — 8-policy multi-select checkboxes, graph area text input, `num_loc` / `n_samples` / `cpu_cores` / `seed` number fields, data distribution radio (Normal / Gamma / Empirical); exactly mirrors `just controller::test-sim` Hydra args
- [x] "Advanced Overrides" collapsible panel: free-form textarea for arbitrary Hydra overrides (§D.6 Option A); live command preview below the form
- [x] Policy selection panel: load registered policy names from `test_sim.yaml` via `list_sim_policies` Rust command at runtime (89 policies; falls back to 8 defaults when file missing)
- [x] Live status display: after launch, subscribes to `process:stdout` events for the spawned process ID; parses `GUI_DAY_LOG_START:` markers; displays a per-policy card grid with day / profit / km / overflows in real time; "View Summary →" and "Process Monitor" navigation buttons shown on completion
- [x] On completion: auto-navigate to `simulation_summary` after 5-second countdown with cancel button; countdown driven by `useEffect` on `simStatus === "completed"`; "View Summary →" manual button always shown alongside countdown
- [x] Session persistence for form values: `useSimLauncherStore` (Zustand `persist`, key `wsroute-sim-launcher`) stores `selectedPolicies`, `area`, `numLoc`, `samples`, `nCores`, `seed`, `distribution`, `extraOverrides`; ephemeral runtime state stays in component state
- [x] Live progress + ETA (hundred-thirty-seventh pass): ``LiveTrainProgressBar`` in live status panel during running simulations (§D.2 / §G.9)
- [x] ``LauncherNavMesh`` shared navigation + ``Simulation Monitor →`` / ``Simulation Summary →`` post-run shortcuts (hundred-thirty-ninth pass; §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut on completed simulations (hundred-forty-second pass; §G.14 / §D.7)
- [x] Post-run Output Browser deep-link via ``outputRunPath`` + ``pendingRunPath`` when stdout contains ``.jsonl`` (hundred-forty-third pass; §G.14 / §D.7)
- [x] Post-run panel persistence via ``findRecentLauncherProcessId`` when navigation clears local state (hundred-forty-sixth pass; §G.9 / §D.7)
- [x] ``LauncherLivePanel`` shared live/post-run panel shell with ``navTrailing`` auto-summary countdown slot (hundred-fifty-seventh pass; §G.9 / §D.7)
- [x] ``ProcessIdFooter`` shared process-id footer row on Simulation Launcher live panel (hundred-fifty-eighth pass; §G.9 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on Simulation Launcher live panel via ``LauncherLivePanel`` (hundred-sixty-fourth pass; §G.9 / §D.7)
- [x] ``simLivePanelTitle`` shared sim live panel title helper; Simulation Launcher imports shared title (hundred-sixty-ninth pass; §G.9 / §D.7)
- [x] Simulation Launcher card live panel header passes ``runLabel`` + · live suffix via ``useProcessRunLabelBrush`` (hundred-seventy-second pass; §G.9 / §D.7)
- [x] ``LauncherNavMesh`` path-kind shortcuts via ``PathHandoffButtons`` (two-hundred-and-twenty-fourth pass): Output Browser / checkpoint / Training Monitor / Data Explorer labeled handoffs with empty-path mode-only fallback (§G.9 / §G.11 / §G.12 / §D.7)
- [x] ``LauncherNavMesh`` sim log dual/single-target shortcuts migrate to ``PathHandoffButtons`` ``kind="log"`` + ``targets`` (two-hundred-and-twenty-sixth pass; §G.9 / §G.1 / §G.16 / §D.7)
- [x] Live-panel header path chips gain handoffs via ``RunLabelHeaderSuffix`` default ``handoff`` (two-hundred-and-twenty-seventh pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] Live-panel ``RunLabelHeaderSuffix`` uses shared ``OpenPathToolbar`` (two-hundred-and-thirty-fifth pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] Live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] Live-panel ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)

---

### §G.10 — Phase 10: Training & HPO Launch Hub ✅

**Goal**: Port the PySide6 reinforcement learning/training tab to Tauri/React.

- [x] Mode selector: train / hpo / eval
- [x] Hydra override textarea → `spawn_python_process main.py <mode> <overrides>`
- [x] Cancel and toast notifications via `useSpawnProcess` hook (§D.5, §D.8)
- [x] React form (train mode): problem selector (vrpp/wcvrp/scwcvrp), model selector (am/tam/ddam/moe), encoder selector (gat/gcn/mha), batch size, max epochs; mirrors controller justfile `train` recipe
- [x] React form (hpo mode): problem/model/encoder selectors + HPO method (nsgaii/tpe/dehb/random), trial count, num_workers; mirrors controller justfile `hpo` recipe
- [x] React form (eval mode): checkpoint path picker (Tauri dialog; .pt/.ckpt/.pth), dataset path picker (.pkl/.json/.csv), problem selector, decoding strategy (greedy/sampling/beam), val_size; mirrors controller justfile `eval` recipe
- [x] WandB toggle: adds `tracker.enabled=false` when disabled
- [x] Live command preview (via `useMemo`): exact `python main.py <mode> <args>` shown before launch
- [x] Live training progress panel (§D.2): `parseMetricLine` parses JSON and `key=value` stdout lines; `LiveChart` ECharts canvas shows train_loss (solid), val_loss (dashed), reward (dotted, right y-axis); latest snapshot row shows epoch/train_loss/val_loss/reward/grad_norm inline
- [x] Live training charts follow global ``logScale``: ``LiveChart`` + ``MiniSparkline`` log y-axis on loss/grad_norm/entropy when on; ``GlobalFilterBar`` in live progress panel (§G.10 / §G.7)
- [x] Gradient norm and entropy sparklines: `MiniSparkline` component (70 px ECharts, area fill at 13% opacity); grad_norm in red `#f87171`, entropy in purple `#a78bfa`; rendered as 2-column grid below `LiveChart`; PNG export on live chart and sparklines; component returns `null` when no data for the given metric key
- [x] On completion: "Output Browser →" button appears in live progress header when training completes successfully; navigates to `output_browser` mode
- [x] Session persistence: `useTrainHubStore` (Zustand `persist`, key `wsroute-train-hub`) stores all form fields across train/hpo/eval modes; ephemeral runtime state stays in component state
- [x] Live training health + runtime attention (§A.4 / §A.2): ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` in live progress panel during train/hpo; ``Training Monitor →`` navigation shortcut (hundred-thirtieth pass)
- [x] Live HPO label + ``HPO Tracker →`` navigation during live HPO runs (hundred-thirty-second pass)
- [x] ``TrainHpoNavMesh`` shared navigation + ``LiveTrainProgressBar`` epoch progress/ETA during live train/HPO (hundred-thirty-fifth pass; §D.2)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on Training Hub live panel (hundred-forty-fourth pass; §G.14 / §G.17 / §D.7)
- [x] Post-run panel persistence via ``findRecentHubProcessId`` (train/HPO/eval) when navigation clears local state (hundred-forty-sixth pass; §G.10 / §D.7)
- [x] Post-run grad-norm + LR sparklines via ``TrainingMetricSparklines``; ``TrainingMetricSnapshot`` + rehydration banner when train/HPO completes (hundred-forty-ninth pass; §G.17 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared live/post-run analytics strip (hundred-fifty-first pass; §G.10 / §D.7)
- [x] ``metric updates`` label ``text-accent-success`` styling parity with Process Monitor / HPO / Experiment Tracker (hundred-fifty-second pass; §G.10 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.10 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` shared live panel header row with ``split`` layout + ``activity`` running icon (hundred-fifty-fourth pass; §G.10 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell with ``footer`` process-id row + ``showAnalytics`` slots (hundred-fifty-sixth pass; §G.10 / §D.7)
- [x] ``ProcessIdFooter`` shared process-id footer row on Training Hub live panel (hundred-fifty-eighth pass; §G.10 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on Training Hub train/HPO live panel via ``TrainHpoLivePanel`` (hundred-sixty-third pass; §G.10 / §D.7)
- [x] Training Hub eval mode — ``LauncherLivePanel`` + ``EvalCheckpointLiveCard`` / ``EvalResultCard`` live panel shell parity with Evaluation Runner (hundred-sixty-sixth pass; §G.10 / §G.12 / §D.7)
- [x] Training Hub eval mode — single-checkpoint progress bar via ``EvalCheckpointLiveCard`` only; ``LauncherNavMesh`` ``Training Hub →`` + ``hideHub`` parity (hundred-sixty-seventh pass; §G.10 / §G.12 / §D.7)
- [x] ``evalLivePanelTitle`` shared eval live panel title helper; Training Hub eval mode imports shared title (hundred-sixty-eighth pass; §G.10 / §D.7)
- [x] ``trainHpoLivePanelTitle`` shared train/HPO live panel title helper; Training Hub train/HPO modes import shared title (hundred-seventieth pass; §G.10 / §D.7)
- [x] Training Hub card live panel headers pass ``runLabel`` + · live suffix for eval and train/HPO modes (hundred-seventy-second pass; §G.10 / §D.7)
- [x] ``TrainHpoNavMesh`` path-kind shortcuts via ``PathHandoffButtons`` (two-hundred-and-twenty-fourth pass): Output Browser + Training Monitor labeled handoffs with empty-path mode-only fallback (§G.10 / §G.15 / §G.17 / §D.7)
- [x] Training Hub eval checkpoint path-kind handoff via ``PathHandoffButtons`` (two-hundred-and-twenty-fifth pass; §G.10 / §G.12 / §D.7)
- [x] Training Hub eval checkpoint + dataset path previews use shared ``OpenPathToolbar``; checkpoint labeled Eval Runner handoff (two-hundred-and-thirty-first pass; §G.10 / §G.12 / §D.7)
- [x] Training Hub live-panel ``RunLabelHeaderSuffix`` uses shared ``OpenPathToolbar`` (two-hundred-and-thirty-fifth pass; §G.10 / §D.7)
- [x] Training Hub live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.10 / §D.7)
- [x] Training Hub live-panel ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.10 / §D.7)

---

### §G.11 — Phase 11: Data Generation Wizard ✅

**Goal**: Port the PySide6 data generation tab to Tauri/React.

- [x] Script selector (generate_dataset / generate_bins / generate_routes) + extra CLI args textarea
- [x] `spawn_python_process` integration via `useSpawnProcess`; cancel and toasts
- [x] React form: problem selector (vrpp/wcvrp/scwcvrp/all), distribution checkboxes (Gamma-3/Empirical), dataset type selector (test_simulator/train/train_time), overwrite toggle; mirrors `gen_data.yaml`
- [x] Graph form: area selector (figueiradafoz/riomaior), num_loc, n_samples, n_days fields; configures `data.graphs[0]` via Hydra override
- [x] Advanced Overrides collapsible + command preview (`python main.py gen_data ...`)
- [x] TSPLIB source option: `dataSource` radio (synthetic / TSPLIB); `.vrp`/`.tsp` file picker via Tauri dialog; Hydra overrides `data.source=tsplib` + `data.tsplib_instance=<path>`; graph form hidden in TSPLIB mode
- [x] Sensor data source option: third `dataSource` radio; CSV file picker (timestamp,bin_id,fill_level,waste_type); Hydra overrides `data.source=sensor` + `data.sensor_file=<path>`
- [x] Preview panel: `preview_dataset_stats` Rust command + "Preview .pkl/.pt" button; KPI cards (instances, nodes, demand μ±σ, file size) + ECharts demand histogram with PNG export; demand histogram follows global ``logScale`` via ``GlobalFilterBar`` (§G.11 / §G.7)
- [x] Live progress: subscribes to `process:stdout` and `process:status` for the active generation run; shows last 20 stdout lines in a scrollable pre-block; status header with `Activity`/`CheckCircle`/`XCircle` icons; "Process Monitor" navigation button on completion
- [x] Session persistence: `useDataGenStore` (Zustand `persist`, key `wsroute-data-gen`) stores all form fields; ephemeral runtime state stays in component state
- [x] Live progress + ETA (hundred-thirty-seventh pass): ``LiveTrainProgressBar`` in live progress panel during ``gen_data`` runs (§D.2 / §G.11)
- [x] ``LauncherNavMesh`` + ``Data Explorer →`` post-run shortcut (hundred-thirty-ninth pass; §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut on completed data generation runs (hundred-forty-second pass; §G.14 / §D.7)
- [x] Post-run Output Browser deep-link via ``outputRunPath`` + ``pendingRunPath`` when stdout contains a log path (hundred-forty-third pass; §G.14 / §D.7)
- [x] Post-run panel persistence via ``findRecentLauncherProcessId`` when navigation clears local state (hundred-forty-sixth pass; §G.11 / §D.7)
- [x] ``LauncherLivePanel`` shared live progress panel shell (hundred-fifty-seventh pass; §G.11 / §D.7)
- [x] ``ProcessIdFooter`` ``footer`` process-id row on Data Generation live panel (hundred-fifty-eighth pass; §G.11 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on Data Generation live panel via ``LauncherLivePanel`` (hundred-sixty-fourth pass; §G.11 / §D.7)
- [x] ``dataGenLivePanelTitle`` shared data-gen live panel title helper; Data Generation imports shared title (hundred-sixty-ninth pass; §G.11 / §D.7)
- [x] Data Generation card live panel header passes ``runLabel`` + · live suffix via ``useProcessRunLabelBrush`` (hundred-seventy-second pass; §G.11 / §D.7)
- [x] Data Generation sensor CSV path-kind handoff via ``PathHandoffButtons`` (two-hundred-and-twenty-fifth pass; §G.11 / §G.6 / §D.7)
- [x] Data Generation sensor CSV / TSPLIB / instance-preview path previews use shared ``OpenPathToolbar``; sensor labeled Data Explorer handoff (two-hundred-and-thirty-first pass; §G.11 / §G.6 / §D.7)
- [x] Data Generation live-panel ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.11 / §D.7)
- [x] ``LauncherNavMesh`` data-gen post-run Data Explorer accepts optional ``csvPath`` (two-hundred-and-thirty-eighth pass; §G.11 / §G.6 / §D.7)
- [x] Data Generation passes sensor / generated CSV into ``LauncherNavMesh`` post-run handoff (two-hundred-and-thirty-eighth pass; §G.11 / §D.7)
- [x] Data-gen completion toasts hand off sensor CSV → Data Explorer (+ optional Output) (two-hundred-and-thirty-eighth pass; §G.11 / §D.8 / §D.7)

---

### §G.12 — Phase 12: Evaluation Runner ✅

**Goal**: Port the PySide6 evaluation tab and expose multi-checkpoint comparison.

- [x] Dynamic checkpoint list: add/remove entries, each with file picker (Tauri dialog; .pt/.ckpt/.pth)
- [x] Eval parameters: dataset path (optional, Tauri dialog), problem selector, decoding strategy (greedy/sampling/beam), device (cpu/cuda:0/cuda:1), val_size
- [x] Multi-checkpoint launch: one `spawn_python_process main.py eval` call per valid checkpoint, tagged with checkpoint filename; results stream to Process Monitor
- [x] Advanced Overrides collapsible + command preview (shows first-checkpoint invocation)
- [x] Results grid: global `process:stdout` listener parses JSON lines with `cost`/`gap`/`tour_cost`/`time`/`policy` fields; keyed by checkpoint name; dynamic column discovery from first result; updates in real time as results stream in
- [x] "Export CSV" button: builds CSV from result rows, triggers browser download via `Blob` + `URL.createObjectURL`
- [x] "Open in Analytics" button pre-loads eval results into BenchmarkAnalysis via `pendingEvalResults` store field; shows cost/gap/time bar charts + summary table
- [x] Inline results bar charts on Evaluation Runner results grid with per-metric PNG export (§G.12)
- [x] EvaluationRunner inline checkpoint charts follow global ``logScale``: log y-axis on cost/gap/time when on; ``GlobalFilterBar`` toggle above results grid (§G.12 / §G.7)
- [x] Live progress + ETA (hundred-thirty-eighth pass): per-checkpoint ``LiveTrainProgressBar`` in live progress panel during ``eval`` runs; multi-checkpoint aggregate status + stdout tail (§D.2 / §G.12)
- [x] ``LauncherNavMesh`` + ``Benchmark Analysis →`` post-run shortcut in live eval panel (hundred-thirty-ninth pass; §D.7)
- [x] ``evalResults.ts`` shared stdout JSON parsing + ``toEvalAnalyticsRows`` helpers (hundred-fortieth pass; §G.12 / §G.15)
- [x] Live progress per-checkpoint KPI row + ``LauncherNavMesh`` ``Output Browser →`` post-run shortcut (hundred-forty-first pass; §G.12 / §G.14 / §D.7)
- [x] ``checkpointPathFromEvalCommand`` + ``Load in Eval Runner →`` from completed eval processes (hundred-forty-first pass; §G.12 / §G.15)
- [x] Single-checkpoint live panel passes ``checkpointPath`` to ``LauncherNavMesh`` for post-run reload (hundred-forty-second pass; §G.12 / §D.7)
- [x] Post-run ``outputRunPath`` deep-link on Evaluation Runner live panel (hundred-forty-fourth pass; §G.14 / §D.7)
- [x] Multi-checkpoint batch persistence via ``findRecentEvalProcessIds`` + ``collectEvalResultFromLogLines`` when navigation clears local state (hundred-forty-sixth pass; §G.12 / §D.7)
- [x] ``LauncherLivePanel`` shared live progress panel shell for multi-checkpoint eval runs (hundred-fifty-seventh pass; §G.12 / §D.7)
- [x] ``ProcessIdFooter`` multi-process footer + ``EvalResultKpiRow`` per-checkpoint KPI row on Evaluation Runner live panel (hundred-fifty-eighth pass; §G.12 / §D.7)
- [x] ``EvalCheckpointLiveCard`` shared per-checkpoint live eval row on Evaluation Runner (hundred-fifty-ninth pass; §G.12 / §D.7)
- [x] ``processLogTail`` + Process Monitor ``EvalCheckpointLiveCard`` live eval parity (hundred-sixtieth pass; §G.12 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display inside ``EvalCheckpointLiveCard`` (hundred-sixty-first pass; §G.12 / §D.7)
- [x] ``EvalCheckpointLiveCard`` accepts raw ``logLines``; Evaluation Runner passes ``logLines`` instead of pre-formatted tail (hundred-sixty-second pass; §G.12 / §D.7)
- [x] ``EvalCheckpointLiveCard`` ``showLogTail={false}`` + ``LauncherLivePanel`` shell ``logLines`` for single-checkpoint eval (hundred-sixty-fifth pass; §G.12 / §D.7)
- [x] ``LauncherNavMesh`` ``Training Hub →`` shortcut on eval workflows (hundred-sixty-seventh pass; §G.12 / §D.7)
- [x] ``evalLivePanelTitle`` shared eval live panel title; Evaluation Runner imports shared title (hundred-sixty-eighth pass; §G.12 / §D.7)
- [x] Evaluation Runner card live panel header passes ``runLabel`` + · live suffix via ``useProcessRunLabelBrush`` (hundred-seventy-second pass; §G.12 / §D.7)
- [x] Evaluation Runner checkpoint input + results table path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-fifth pass; §G.12 / §D.7)
- [x] ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint icon handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-fifth pass; §G.12 / §G.15 / §D.7)
- [x] Evaluation Runner + eval live/result cards checkpoint chips use ``PathRunLabelChip`` ``handoff="checkpoint"`` (two-hundred-and-twenty-seventh pass; §G.12 / §D.7)
- [x] Evaluation Runner checkpoint-row + dataset path previews use shared ``OpenPathToolbar`` (two-hundred-and-thirty-first pass; §G.12 / §D.7)
- [x] ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint headers use shared ``OpenPathToolbar`` (two-hundred-and-thirty-first pass; §G.12 / §G.15 / §D.7)
- [x] Evaluation Runner results-table checkpoint cells use shared ``OpenPathToolbar`` (two-hundred-and-thirty-third pass; §G.12 / §D.7)
- [x] Evaluation Runner multi-checkpoint ``ProcessIdFooter`` shows batch process count as ``children`` when ``logPath`` is set (two-hundred-and-thirty-sixth pass; §G.12 / §D.7)
- [x] Evaluation Runner live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.12 / §D.7)
- [x] Evaluation Runner ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.12 / §D.7)
- [x] Eval completion toasts hand off checkpoint → Evaluation Runner (+ optional Output) (two-hundred-and-thirty-seventh pass; §G.12 / §D.8 / §D.7)

---

### §G.13 — Phase 13: Configuration Editor (Hydra YAML) ✅

**Goal**: Provide a full-featured Hydra configuration editor so users never need to touch config files manually.

- [x] Three editor modes: Raw (editable textarea), Table (flat key-value, YAML parsed), Diff (compare two YAML files side-by-side)
- [x] File picker via Tauri dialog (YAML / TOML / CFG)
- [x] "Copy Overrides" button: serialises flat key=value lines to clipboard via `navigator.clipboard`
- [x] Config diff view: highlights changed keys between primary and comparison file (e.g. `pruned_config.yaml` from two different runs)
- [x] Rust `read_text_file` command for loading any text file
- [x] Rust `write_text_file` command: creates parent directories if needed; used by the Save button
- [x] "Save" button in toolbar: writes edited Raw content back to the opened file path; active only when unsaved edits exist (dirty state tracked via `savedContentRef`); `Save*` label indicates unsaved changes
- [x] Load the resolved Hydra config tree via `dump_hydra_config` Rust command (`main.py <task> --cfg job`); task selector + "Load via --cfg job" button in ConfigEditor toolbar
- [x] Form mode: fourth view toggle with typed widgets (boolean checkbox, number input, text input) inferred from flat YAML values; edits sync to Raw content via `rowsToYaml()` (OmegaConf schema introspection deferred)
- [x] Monaco Editor integration for the Raw YAML mode (§D.6 Option C): lazy-loaded `YamlEditor` with syntax highlighting and theme sync
- [x] "Apply to Launcher" button: target selector (Simulation Launcher / Training Hub / Data Generation); `applyConfigToLauncher()` maps flat YAML keys to Zustand store patches and navigates to the target page
- [x] ``Ctrl+S`` keyboard shortcut saves dirty config to disk when a file path is open (§D.7 / §G.13)
- [x] Configuration Editor primary YAML + diff comparison path chips — Config Editor path-kind icon handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-sixth pass; §G.13 / §D.7)
- [x] Configuration Editor path chips migrate to ``PathRunLabelChip`` ``handoff="config"`` (two-hundred-and-twenty-seventh pass; §G.13 / §D.7)
- [x] Configuration Editor primary + diff path toolbars use shared ``OpenPathToolbar`` (two-hundred-and-twenty-ninth pass; §G.13 / §D.7)

---

### §G.14 — Phase 14: Output Browser & Session Management ✅

**Goal**: Replace the PySide6 file system tab with a native file browser tailored to WSmart-Route's output directory structure.

- [x] Run list panel: `list_output_dirs` with name, path, created_at, size
- [x] File tree: `list_dir` command; lazy-loads subdirectory contents on expand; `Folder`/`FileText`/`File` icons by extension
- [x] File viewer: CSV files load via `load_csv_file` (table with 200-row preview); text/YAML/JSON via `read_text_file` (syntax-highlighted pre block)
- [x] Directory picker via Tauri dialog for browsing arbitrary directories (not just `assets/output/`); ``pushRecent`` as ``run`` via ``portfolioRunLabel`` (two-hundred-and-tenth pass; §G.14 / §D.7)
- [x] Run metadata panel: auto-loads `pruned_config.yaml` (or `config.yaml`) when a run is selected; flat YAML parsed and filtered by `META_KEYS`; compact two-column card below the file tree
- [x] "Open in Sim Summary" button: shown for `.jsonl` files; sets `pendingLogPath` in app store then navigates to `simulation_summary` mode; `SimulationSummary` consumes `pendingLogPath` on mount via `useEffect`
- [x] "Open in Data Explorer →" button: shown for `.csv` files; sets `pendingCsvPath` + ``pushRecent`` via ``portfolioRunLabel`` then navigates to `data_explorer` (two-hundred-and-ninth pass; §G.14 / §G.6 / §D.7)
- [x] "Open in Config Editor →" button: shown for YAML / TOML / cfg / ini; sets `pendingConfigPath` + ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-eighth pass; §G.14 / §G.13 / §D.7)
- [x] Directory tree view: auto-expand `hydra/` on run selection; `sortEntries()` prioritises config and log artefacts; highlight `pruned_config.yaml` and `.jsonl` in the file tree
- [x] Simulation result summary: on `selectRun`, scans top-level entries for a `.jsonl` file ≤ 20 MB; reads it via `read_text_file`, parses each line as `DayLogEntry`, aggregates overflows / kg/km / profit per policy; displays a compact 3-column KPI table (policy / overflows / kg/km) below the config metadata card; overflows colour-coded (green = 0, amber = low, red > 20)
- [x] "Compare runs": per-run checkbox multi-select (≥2); `findRunJsonl()` locates logs in top-level or `hydra/`; navigates to BenchmarkAnalysis with `pendingBenchmarkLogs`
- [x] Session profiles (§D.4 Option C): `useSessionProfilesStore` persists named snapshots of all three launcher stores; save/load/delete UI in Output Browser sidebar (max 20 profiles)
- [x] Recent files/runs: `useRecentFilesStore` tracks last 12 opened logs, output runs, CSVs, training dirs, checkpoints, and configs; surfaced in command palette
- [x] Checkpoint browser (hundred-forty-second pass): auto-expand ``checkpoints/`` on run select; sidebar card lists ``.pt/.ckpt/.pth`` with **Eval →** shortcut; file tree highlights checkpoint artefacts; **Load in Eval Runner →** on selected checkpoint files via ``pendingCheckpoint``; inline open ``pushRecent`` via ``portfolioRunLabel`` (two-hundred-and-ninth pass; §G.14 / §G.12 / §G.17 / §D.7)
- [x] ``checkpoints.ts`` — shared ``isCheckpointEntry`` / ``filterCheckpointEntries`` helpers used by Output Browser + Training Monitor (§G.14 / §G.12)
- [x] ``outputRunPath.ts`` + ``pendingRunPath`` auto-select when opened from launcher / Process Monitor shortcuts (hundred-forty-third pass; §G.9 / §G.11 / §G.15 / §D.7)
- [x] Output Browser refreshes run list when ``pendingRunPath`` is set but the run is not yet indexed (hundred-forty-third pass; §G.14)
- [x] ``outputRunPathFromHydraArtifact`` + Hydra snapshot / pruned-config stdout parsing (hundred-forty-fourth pass; §G.14 / §G.9 / §G.12)
- [x] Output Browser ``useLogPathRunLabelBrush`` replaces inline ``setRunLabel`` on run select (hundred-seventy-fourth pass; §G.14 / §D.7)
- [x] Output Browser ``runLabelMapFromPaths`` replaces inline ``runLabelFromPath`` in run list ring highlights (hundred-seventy-fifth pass; §G.14 / §D.7)
- [x] Output Browser path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-fourth pass): run-header log dual / run single; file-viewer CSV / config / checkpoint / log labeled controls; checkpoint sidebar icon Eval handoffs (§G.14 / §G.7 / §D.7)
- [x] Output Browser run list ``pathHandoffs`` via ``LoadedRunRow`` (two-hundred-and-twenty-fifth pass; §G.14 / §D.7)
- [x] Output Browser run header / checkpoint sidebar / file-viewer / checkpoint panel chips use ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.14 / §D.7)
- [x] Output Browser file viewer + checkpoint panel labeled+chip dual control uses shared ``OpenPathToolbar`` (two-hundred-and-twenty-eighth pass; §G.14 / §D.7)
- [x] Output Browser run header uses shared ``OpenPathToolbar``; labeled Summary / Monitor dual when run log is known (two-hundred-and-twenty-ninth pass; §G.14 / §G.1 / §D.7)
- [x] Output Browser checkpoint sidebar rows use shared ``OpenPathToolbar``; labeled Eval Runner handoff + size ``children`` (two-hundred-and-thirty-second pass; §G.14 / §G.12 / §D.7)
- [x] Output Browser ``.wsroute`` manifest member paths use ``PathRunLabelChip`` auto-classified handoffs (two-hundred-and-thirty-second pass; §G.8 / §G.14 / §D.7)
- [x] Output Browser ``.wsroute`` manifest members use shared ``OpenPathToolbar`` (two-hundred-and-thirty-fourth pass; §G.8 / §G.14 / §D.7)
- [x] ``LoadedRunRow`` portfolio lists use shared ``OpenPathToolbar``; meta ``trailing`` as ``children`` (two-hundred-and-thirty-fourth pass; §G.1 / §G.14 / §D.7)

---

### §G.15 — Phase 15: Real-Time Process Monitor & Log Viewer ✅

**Goal**: Provide a unified view of all running and recently completed processes, replacing the PySide6 file-tailer pattern.

- [x] `ProcessRegistry` in Rust: global `OnceLock<Arc<Mutex<HashMap<String, (u32, Sender<bool>)>>>>`
- [x] `process:spawn` event emitted immediately after spawn (id, command, pid, start_time); `useProcessMonitor` hook registers process in store
- [x] `process:stdout` events for each stdout/stderr line; stored in per-process `logLines` (capped at 2000)
- [x] `process:status` event on completion/cancel/failure with exit code
- [x] Process list panel: status badge, command, inline log viewer (last 50 lines), cancel button
- [x] `cancel_process` Tauri command: sends `true` via watch channel → `child.kill()`
- [x] `which_python` resolves `<workingDir>/.venv/bin/python` first (uv-managed venv), then system PATH
- [x] Process list panel: full tabular layout — `StatusPill` + process ID + command + PID + live duration (`useLiveDuration` hook, 1s tick, stops when process ends) + exit code badge; sorted newest-first
- [x] Inline log viewer per process: expand/collapse toggle, auto-scroll checkbox, stderr lines coloured `text-accent-warning`; scroll locked at 2000 lines via process store
- [x] Structured log parsing: `LogLine` component tries `JSON.parse` on each line; if successful and has `level`/`msg`/`message` fields, renders timestamp (ISO prefix), colour-coded level badge (danger/warning/muted/gray), and message; falls back to plain text for non-JSON lines
- [x] Remove button per completed process row (`Trash2` icon); "Clear completed (N)" bulk action in the header
- [x] `clearCompleted` action added to process store: removes all non-running entries
- [x] Process history persistence: `useProcessStore` wrapped in Zustand `persist` middleware; `partialize` strips `logLines` and caps at last 50 completed processes; survives app restart
- [x] Progress bar per process: subscribe to structured progress events (epoch, day, instance count) emitted by the Python subprocess via stdout markers — `PROGRESS:{json}` protocol; `getLatestProgress()` scans last 30 log lines; deterministic bar when `total` is known, indeterminate pulse otherwise
- [x] Process row progress + ETA (hundred-thirty-sixth pass): ``LiveTrainProgressBar`` on each running process row; elapsed + ETA via shared ``processProgress.ts`` helpers (§D.2 / §G.15)
- [x] ``LauncherNavMesh`` return shortcuts for selected ``test_sim`` / ``gen_data`` / ``eval`` processes (hundred-thirty-ninth pass; §D.7)
- [x] Cancel any running process (§D.5): button in the process list row; sends SIGTERM (`cancel_process` command already wired in `ProcessRow`)
- [x] Toast notification on process completion / failure (§D.8): `useProcessMonitor` fires `toast.success/error/info` on terminal status transitions; label derived from `id.split("_")[0]`
- [x] Training analytics for ``train_`` / ``hpo_`` processes: ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` parsed from process stdout (§A.4 / §A.2 hundred-thirtieth pass)
- [x] Eval results panel (hundred-fortieth pass): selected ``eval`` processes parse structured JSON from stdout; KPI row (cost / gap / time / policy) + ``Open in Analytics →`` via ``pendingEvalResults`` (§G.12 / §G.15 / §D.7)
- [x] ``LauncherNavMesh`` ``Benchmark Analysis →`` on completed eval processes when metrics are present (hundred-fortieth pass; §D.7 / §G.12)
- [x] ``TrainHpoNavMesh`` ``Output Browser →`` on completed ``train_`` / ``hpo_`` processes (hundred-fortieth pass; §G.10 / §D.7)
- [x] ``LauncherNavMesh`` ``Output Browser →`` + ``Load in Eval Runner →`` on completed eval processes (hundred-forty-first pass; §G.12 / §G.14 / §D.7)
- [x] Process Monitor ``Output Browser →`` on completed ``test_sim`` / ``gen_data`` processes with run deep-link (hundred-forty-third pass; §G.9 / §G.11 / §G.14 / §D.7)
- [x] Process Monitor eval ``outputRunPath`` deep-link parity (hundred-forty-fourth pass; §G.12 / §G.14 / §D.7)
- [x] Process Monitor train/HPO ``outputRunPath`` + ``trainingRunPath`` deep-links on ``TrainHpoNavMesh`` (hundred-forty-fourth pass; §G.10 / §G.17 / §D.7)
- [x] Train/HPO metrics rehydration + grad-norm/LR sparklines on selected processes (hundred-forty-eighth pass; §G.15 / §G.17 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared analytics strip on selected train/HPO processes (hundred-fifty-first pass; §G.15 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.15 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` ``muted`` analytics subtitle header + badges-before-nav ordering parity (hundred-fifty-fourth pass; §G.15 / §D.7)
- [x] ``TrainHpoLivePanel`` ``embedded`` variant for selected train/HPO analytics section (hundred-fifty-sixth pass; §G.15 / §D.7)
- [x] ``LauncherLivePanelHeader`` ``embedded`` muted subtitle header + run-label + live suffix parity on selected ``test_sim`` processes (hundred-fifty-seventh pass; §G.9 / §G.15 / §D.7)
- [x] ``LauncherLivePanel`` ``embedded`` variant for selected ``test_sim`` / ``gen_data`` / ``eval`` analytics sections (hundred-fifty-seventh pass; §G.9 / §G.11 / §G.12 / §G.15 / §D.7)
- [x] ``EvalResultCard`` shared eval result card with ``Open in Analytics →`` on selected ``eval`` processes (hundred-fifty-eighth pass; §G.12 / §G.15 / §D.7)
- [x] ``EvalCheckpointLiveCard`` live progress + stdout tail on selected running ``eval`` processes; ``EvalResultCard`` on completion with metrics (hundred-sixtieth pass; §G.12 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on selected ``gen_data`` embedded workflow section via ``LauncherLivePanel`` (hundred-sixty-fourth pass; §G.11 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on selected ``test_sim`` embedded workflow section via ``LauncherLivePanel`` (hundred-sixty-fourth pass; §G.9 / §G.15 / §D.7)
- [x] Process Monitor eval embedded section passes raw ``logLines`` to ``EvalCheckpointLiveCard`` (hundred-sixty-second pass; §G.12 / §G.15 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on selected ``train_`` / ``hpo_`` embedded analytics section via ``TrainHpoLivePanel`` (hundred-sixty-third pass; §G.15 / §D.7)
- [x] Process Monitor eval embedded section passes ``logLines`` to ``LauncherLivePanel`` instead of inline ``ProcessLogTail`` on ``EvalCheckpointLiveCard`` (hundred-sixty-fifth pass; §G.12 / §G.15 / §D.7)
- [x] ``LauncherNavMesh`` ``Training Hub →`` shortcut on selected ``eval`` embedded section (hundred-sixty-seventh pass; §G.12 / §G.15 / §D.7)
- [x] Process Monitor eval embedded section uses dynamic ``evalLivePanelTitle`` instead of static ``Eval results`` subtitle (hundred-sixty-eighth pass; §G.12 / §G.15 / §D.7)
- [x] Process Monitor sim embedded section uses dynamic ``simLivePanelTitle`` instead of static ``Policy telemetry`` subtitle (hundred-sixty-ninth pass; §G.9 / §G.15 / §D.7)
- [x] Process Monitor data-gen embedded section uses dynamic ``dataGenLivePanelTitle`` instead of static ``Data generation workflow`` subtitle (hundred-sixty-ninth pass; §G.11 / §G.15 / §D.7)
- [x] Process Monitor train/HPO embedded section uses dynamic ``trainHpoLivePanelTitle`` instead of static ``Training analytics`` subtitle (hundred-seventieth pass; §G.10 / §G.15 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor eval + data-gen embedded sections pass ``runLabel`` + live suffix parity with sim panel (hundred-seventy-first pass; §G.11 / §G.12 / §G.15 / §D.7)
- [x] Process Monitor train/HPO embedded section uses muted subtitle header + ``runLabel`` + live suffix via ``TrainHpoLivePanel`` ``embedded`` defaults (hundred-seventy-first pass; §G.15 / §D.7)
- [x] Process Monitor process row ring highlight + global ``run_label`` brush sync for all workflow kinds (hundred-seventy-first pass; §G.15 / §D.7)
- [x] Process Monitor ``useProcessRunLabelBrush`` + ``runLabelMapFromProcesses`` shared run-label brush hook parity (hundred-seventy-third pass; §G.15 / §D.7)
- [x] Process Monitor process-row + ``ProcessIdFooter`` chips + live-header suffix handoffs via ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.15 / §D.7)
- [x] Process Monitor ``EvalResultCard`` / ``EvalCheckpointLiveCard`` checkpoint headers use shared ``OpenPathToolbar`` (two-hundred-and-thirty-first pass; §G.15 / §G.12 / §D.7)
- [x] Process Monitor process-row path chips use shared ``OpenPathToolbar``; process id as ``children`` (two-hundred-and-thirty-third pass; §G.15 / §D.7)
- [x] ``ProcessIdFooter`` log-path chip uses shared ``OpenPathToolbar``; process id as ``children`` (two-hundred-and-thirty-fourth pass; §G.15 / §G.9–§G.12 / §G.17 / §G.18 / §D.7)
- [x] Process Monitor live-panel ``RunLabelHeaderSuffix`` uses shared ``OpenPathToolbar`` (two-hundred-and-thirty-fifth pass; §G.15 / §D.7)
- [x] Process Monitor live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.15 / §D.7)
- [x] ``ProcessIdFooter`` multi-process batch meta as ``children`` when ``logPath`` is set (two-hundred-and-thirty-sixth pass; §G.12 / §G.15 / §D.7)
- [x] Process Monitor process-row + ``ProcessIdFooter`` brush process-derived ``runLabel`` (two-hundred-and-thirty-seventh pass; §G.15 / §D.7)
- [x] Process Monitor data-gen embedded panel passes ``csvPath`` into ``LauncherNavMesh`` (two-hundred-and-thirty-eighth pass; §G.15 / §G.11 / §D.7)

---

### §G.16 — Phase 16: Simulation Digital Twin Page

**Goal**: Full Streamlit `simulation` mode parity — real-time map, KPI dashboard, tour visualization, and bin-fill heatmap — superseding the basic SimulationMonitor scaffold implemented in Phase 0.

Source files ported from: `logic/src/ui/pages/simulation/{kpi,map,charts,bins,tour,summary_sections}.py`, `logic/src/ui/services/simulation_analytics.py`

- [x] **KPI dashboard** (`kpi.py` parity): primary group (profit, distance, waste, overflows) and secondary group (collections, waste lost, efficiency, cost); day-over-day delta badges; secondary group shown/hidden via toggle button
- [x] **Bin-fill strip chart**: top-25 bins sorted by fill descending; 0-100% horizontal bars colour-coded (green <80%, amber 80–99%, red ≥100%); mandatory (!) and collected (✓) badges per row; show/hide toggle
- [x] **Tour table**: stop #, bin ID, fill %, collected ✓/—, mandatory !/— columns; reads `tour_indices` preferentially, falls back to `tour`; limited to 60 rows with count shown; show/hide toggle
- [x] **Daily metrics chart**: ECharts `line` timeseries for all 4 primary KPIs across all loaded days; rendered as a 4-column grid
- [x] **Day scrubber**: ◀/▶ step buttons flanking the range slider; "Following" badge (green pulse) when `selectedDay` is null and watcher is active; "Latest ↓" button to release back to auto-follow
- [x] **Simulation Summary page** (`simulation_summary` mode) — rewritten with: sortable policy ranking table (mean ± std per metric, coloured policy dots); per-day trajectory overlay chart (all policies on one ECharts line chart, metric selector: overflows/profit/km/kg); trajectory chart follows global ``logScale`` (symlog overflows + log profit/km/kg); four metric bar charts with std dev in tooltip hover
- [x] **Route map preview** (ECharts scatter + path): Cartesian tour viz using `all_bin_coords` + `tour_indices`; fill-level colour coding; depot/tour/idle bin layers; PNG export
- [x] **Route map** (deck.gl `PathLayer`): `DeckRouteMap` renders tour path over MapLibre dark basemap; fill-level colour-coded `ScatterplotLayer` stops; idle bins as grey scatter; PNG export via `exportCanvasPng`; ECharts/deck.gl toggle in SimulationMonitor; lazy-loaded chunk (§G.16)
- [x] **Side-by-side route compare**: overlay/split layout toggle when exactly 2 policies visible; split renders labelled dual `DeckRouteMap` or dual ECharts panels (§G.16)
- [x] **Policy / Sample multi-select**: chip-toggle row shown when ≥2 policies present; `chartPolicies` state (default: all); `MetricTimeseries` refactored to accept `policySeries: { policy; entries; color }[]`; 8-colour `POLICY_COLORS` palette; ECharts legend shown when >1 series; detail panels (KpiCard, BinFill, TourTable) still use single `selectedPolicy` dropdown
- [x] **Streamlit parity check**: `PRIMARY_KPIS` and `SECONDARY_KPIS` in `SimulationMonitor.tsx` verified against `_PRIMARY_KPI_MAP` and `_SECONDARY_KPI_MAP` in `kpi.py` — exact match confirmed
- [x] **Daily KPI timeseries follow global ``logScale``**: ``MetricTimeseries`` symlog overflows + log profit/km/kg when on; ``GlobalFilterBar`` on Simulation Monitor (§G.16 / §G.7)
- [x] **deck.gl route map PNG export with toast feedback**: ``DeckRouteMap`` ``exportCanvasPng()`` names export ``route-map-tile.png`` (Mercator) or ``route-map-orbit.png`` (OrbitView) with toast feedback (§G.16 / §G.7)
- [x] Simulation Monitor ``GlobalFilterBar`` ``runLabels`` + ``useLogPathRunLabelBrush`` on log open (hundred-seventy-fourth pass; §G.16 / §D.7)
- [x] Simulation Monitor open-log toolbar + path chip path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-sixth pass; §G.16 / §G.1 / §D.7)
- [x] Simulation Monitor open-path chip uses ``PathRunLabelChip`` ``handoff="log"`` (two-hundred-and-twenty-seventh pass; §G.16 / §D.7)
- [x] Simulation Monitor open-log toolbar uses shared ``OpenPathToolbar`` with reverse Summary ``labeledTargets`` (two-hundred-and-twenty-eighth pass; §G.16 / §D.7)

**Status**: §G.16 complete — all checklist items delivered.

---

### §G.17 — Phase 17: Training Monitor Page ✅

**Goal**: Full Streamlit `training` mode parity — training run discovery, Lightning CSV metrics, hyperparameter inspection, and multi-run comparison.

Source files ported from: `logic/src/ui/pages/training.py`, `logic/src/ui/pages/training_charts.py`, `logic/src/ui/services/data_loader.py`

- [x] **Run discovery** (`discover_training_runs` parity): scan `<projectRoot>/logs/` for Lightning log directories; detect `metrics.csv` and `hparams.yaml`; checkbox multi-select
- [x] **Metrics CSV loading**: `load_training_metrics` Rust command parses Lightning `metrics.csv`; epoch/step x-axis; train_loss, val_loss, reward columns handled
- [x] **Multi-run overlay chart**: single ECharts canvas with one colour-coded series set per run (8-colour palette); train loss (solid), val loss (dashed), reward (dotted, right y-axis); scrollable legend; PNG export; replaces one-chart-per-run layout
- [x] **Global log-scale on training charts**: ``MultiRunChart`` log loss axis + grad-norm/LR sparklines log y-axis when global ``logScale`` on; ``GlobalFilterBar`` on Training Monitor (§G.17 / §G.7)
- [x] **Gradient norm sparkline**: separate compact ECharts chart for `grad_norm` column, shown per selected run
- [x] **Hyperparameter panel**: reads `hparams.yaml` via `read_text_file`; collapsible; flat `key: value` parser; shows first 8 rows with "Show all" expand; skips comment lines
- [x] **Checkpoint browser**: `list_dir` on `<run.path>/checkpoints/`; filters to `.pt/.ckpt/.pth`; shows name + file size; "Load in Eval Runner →" button sets `pendingCheckpoint` in app store and switches to `eval_runner` mode
- [x] **Learning rate schedule chart**: `lr` column rendered as a compact `LrSparkline` (step-level, amber `#fbbf24`) using the shared `MetricSparkline` base component; shown per selected run below the gradient norm sparkline
- [x] **Live training mode**: `LIVE_KEY = "__live__"` virtual entry in `metricsMap`; `activeTrainId` from `useProcessStore` (newest running `train_*` or `hpo_*` process via ``findActiveLiveTrainProcessId``); `process:stdout` listener appends parsed metric rows to `metricsMap[LIVE_KEY]` without touching the CSV; live entry auto-selected in run list with `Radio` icon + pulse animation; ``Live HPO`` label when an ``hpo_*`` process is active; live `RunPanel` shows `GradNormSparkline` + `LrSparkline`; auto-deselected when process exits (hundred-thirty-first pass extends HPO coverage)
- [x] **Column normalization**: `normalizeMetricRow()` maps Lightning CSV aliases (`train/rl_loss` → `train_loss`, `val/cost` → `val_loss`, `lr-Adam` → `lr`) applied at both CSV load time and live stdout parse time; same normalization applied to `TrainingHub.tsx`
- [x] **Streamlit parity check**: Lightning CSV columns `train_loss`, `val_loss`, `reward`, `grad_norm`, `lr`, `epoch`, `step` all rendered; aliased column variants covered by `normalizeMetricRow`
- [x] ``pendingTrainingRunPath`` auto-select when opened from Training Hub / Process Monitor train/HPO shortcuts (hundred-forty-fourth pass; §G.10 / §G.15 / §D.7)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on live/recent train panel; auto-select completed run from stdout ``trainingRunPath`` (hundred-forty-fifth pass; §G.10 / §G.14 / §D.7)
- [x] Post-run metrics/health/attention rehydration from ``useProcessStore`` when live streaming state clears; multi-run overlay chart persists via ``effectiveLiveMetrics`` (hundred-forty-seventh pass; §G.17 / §D.7)
- [x] ``TrainingMetricSparklines`` shared grad-norm + LR sparklines used across Process Monitor and analytics pages (hundred-forty-eighth pass; §G.15 / §G.18 / §D.7)
- [x] Training Monitor deduplicated to shared ``TrainingMetricSparklines`` + ``TrainingMetricSnapshot``; post-run sparkline rehydration banner parity (hundred-forty-ninth pass; §G.10 / §D.7)
- [x] Post-run health/attention rehydration banner via ``postRunTrainingRehydrationMessage`` (hundred-fiftieth pass; §A.2 / §A.4 / §D.7)
- [x] Live/recent card ``TrainHpoAnalyticsStrip`` for post-run sparkline rehydration without ``LIVE_KEY`` selection (hundred-fifty-first pass; §G.17 / §D.7)
- [x] ``metric updates`` label parity across Training Hub + Training Monitor (hundred-fifty-first pass; §G.10 / §D.7)
- [x] Post-run health/attention banner counts via rehydrated entries on ``TrainHpoAnalyticsStrip`` (hundred-fifty-second pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``metric updates`` label on non-checkbox live/recent header + Training Hub accent-success styling (hundred-fifty-second pass; §G.10 / §G.17 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` deduplicated live/recent header with ``overlaySelect`` ``LIVE_KEY`` checkbox parity (hundred-fifty-fifth pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell with ``overlaySelect`` + ``showHealthAttention={false}`` options (hundred-fifty-sixth pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on live/recent train panel via ``TrainHpoLivePanel`` (hundred-sixty-third pass; §G.17 / §D.7)
- [x] ``trainHpoLivePanelTitle`` shared train/HPO live panel title helper; Training Monitor imports shared title (hundred-seventieth pass; §G.17 / §D.7)
- [x] Training Monitor ``TrainHpoLivePanel`` card header passes ``runLabel`` + · live suffix via ``useProcessRunLabelBrush`` (hundred-seventy-second pass; §G.17 / §D.7)
- [x] Training Monitor path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-fourth pass): run-panel training icon + checkpoint browser Eval icon (§G.17 / §G.12 / §D.7)
- [x] Training Monitor run-discovery list ``pathHandoffs`` via ``LoadedRunRow`` (two-hundred-and-twenty-fifth pass; §G.17 / §D.7)
- [x] Training Monitor run-panel + checkpoint browser chips use ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.17 / §G.12 / §D.7)
- [x] Training Monitor logs-root discovery chip uses shared ``OpenPathToolbar`` (two-hundred-and-twenty-ninth pass; §G.17 / §D.7)
- [x] Training Monitor empty-state logs-root chip uses shared ``OpenPathToolbar`` (two-hundred-and-thirtieth pass; §G.17 / §D.7)
- [x] Training Monitor ``RunPanel`` path header uses shared ``OpenPathToolbar``; epochs meta as ``children`` (two-hundred-and-thirty-second pass; §G.17 / §D.7)
- [x] Training Monitor checkpoint browser rows use shared ``OpenPathToolbar``; labeled Eval Runner handoff + size ``children`` (two-hundred-and-thirty-second pass; §G.17 / §G.12 / §D.7)
- [x] Training Monitor live-panel ``RunLabelHeaderSuffix`` uses shared ``OpenPathToolbar`` (two-hundred-and-thirty-fifth pass; §G.17 / §D.7)
- [x] Training Monitor live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.17 / §D.7)
- [x] Training Monitor live-panel ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.17 / §D.7)
- [x] Training Monitor ``RunPanel`` path header brushes ``run.name`` (list / panel shell parity) (two-hundred-and-thirty-eighth pass; §G.17 / §D.7)

---

### §G.18 — Phase 18: Experiment & HPO Tracker ✅


**Goal**: Full Streamlit `experiment_tracker` and `hpo_tracker` mode parity — MLflow/ZenML run browser, Optuna study visualization, and cross-experiment comparison.

Source files ported from: `logic/src/ui/pages/experiment_tracker.py`, `logic/src/ui/pages/experiment_tracker_charts.py`, `logic/src/ui/pages/hpo_tracker.py`

- [x] **MLflow run table** (`experiment_tracker.py` parity): Rust queries MLflow via Python subprocess (`mlflow.search_runs`); display runs with params, metrics, tags, artifact path
- [x] **Metric comparison chart**: select two or more MLflow runs; overlay their logged metrics as ECharts line series; metric name selector and Y-axis normalization toggle
- [x] **ZenML pipeline view** (if ZenML is configured): `list_zenml_pipeline_runs` + `load_zenml_run_steps` Rust commands; pipeline run table; step-duration horizontal bar chart (Gantt-style) with log duration axis when global ``logScale`` on; CSV/PNG export (§G.18 / §G.7)
- [x] **Optuna study browser** (`hpo_tracker.py` parity): `list_optuna_studies` + `load_optuna_study` Rust commands call Optuna via Python subprocess; trials serialised to JSON; HPOTracker displays:
  - Parallel coordinates plot (`echarts` `parallel` series) across hyperparameter dimensions
  - Optimization history scatter plot (trial number vs. objective value) with best-so-far line
  - Parameter importance bar chart (FANOVA via `optuna.importance.get_param_importances`)
- [x] **HPO charts follow global ``logScale``**: optimisation history + cross-study best-so-far lines + parallel-coordinates objective axis use log objective when on; ``GlobalFilterBar`` on HPO Tracker (§G.18 / §G.7)
- [x] **MLflow metric comparison follows global ``logScale``**: multi-run overlay chart log y-axis on loss/objective metrics when on; ``GlobalFilterBar`` on Experiment Tracker (§G.18 / §G.7)
- [x] **Best-trial highlight**: best value KPI card; "Copy best params" button writes trial `params` as Hydra override lines to clipboard
- [x] **Cross-study comparison**: "Compare with" study dropdown in HPOTracker; overlaid best-so-far optimisation history (ECharts); side-by-side best-value KPI cards for both studies
- [x] **MLflow dashboard embed fallback**: Runs/Dashboard tab toggle in ExperimentTracker; iframe embed of local MLflow UI (`http://localhost:5000` default) + open-in-browser via shell plugin (native WebView window deferred)
- [x] **Live HPO analytics** (§A.4 / §A.2 hundred-thirty-first pass): ``TrainingHealthPanel`` + ``RuntimeAttentionPanel`` when an ``hpo_*`` process is running; ``Process Monitor →`` navigation shortcut
- [x] **Experiment Tracker live HPO analytics** (§A.4 / §A.2 hundred-thirty-second pass): health + attention panels during live ``hpo_*``; ``HPO Tracker →`` + ``Process Monitor →`` shortcuts
- [x] **Cross-page train/HPO navigation** (hundred-thirty-second pass): Training Monitor ``Process Monitor →`` + ``HPO Tracker →``; HPO Tracker ``Training Monitor →``; Process Monitor ``Training Monitor →`` + ``HPO Tracker →`` for ``hpo_*`` processes
- [x] **Experiment Tracker navigation mesh** (hundred-thirty-third pass): Experiment Tracker ``Training Monitor →``; HPO Tracker / Training Monitor / Process Monitor / Training Hub ``Experiment Tracker →`` when live HPO active (§G.10 / §G.15 / §G.17 / §G.18)
- [x] **Training Hub navigation mesh** (hundred-thirty-fourth pass): Training Monitor / Process Monitor / HPO Tracker / Experiment Tracker ``Training Hub →`` during live train/HPO workflows — completes bidirectional cross-page shortcuts (§G.10 / §G.15 / §G.17 / §G.18)
- [x] **Live epoch progress + ETA** (hundred-thirty-fifth pass): ``LiveTrainProgressBar`` on Training Hub / Training Monitor / HPO Tracker / Experiment Tracker; ``processProgress.ts`` shared with Process Monitor (§D.2 / §G.17 / §G.18)
- [x] **Train/HPO keyboard shortcuts** (hundred-thirty-fifth pass): ``T`` Training Monitor · ``H`` Training Hub · ``E`` Experiment Tracker (§D.7)
- [x] Post-run ``outputRunPath`` + ``trainingRunPath`` deep-links on HPO Tracker + Experiment Tracker live panels when sweep completes (hundred-forty-fifth pass; §G.14 / §G.17 / §D.7)
- [x] Live metric snapshot row + update count from ``collectTrainingMetricsFromLogLines`` on persisted HPO process stdout (hundred-forty-seventh pass; §G.18 / §G.17 / §D.7)
- [x] Post-run grad-norm + LR sparklines from persisted HPO stdout via ``TrainingMetricSparklines`` (hundred-forty-eighth pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainingMetricSnapshot`` deduplication + ``postRunTrainingRehydrationMessage`` health/attention banner parity (hundred-fiftieth pass; §G.17 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoAnalyticsStrip`` shared live/post-run analytics strip (hundred-fifty-first pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainHpoRehydrationBadges`` shared header badges for metric / health / attention rehydration counts (hundred-fifty-third pass; §G.18 / §A.2 / §A.4 / §D.7)
- [x] ``TrainHpoLivePanelHeader`` deduplicated live HPO header blocks on HPO Tracker + Experiment Tracker (hundred-fifty-fourth pass; §G.18 / §G.17 / §D.7)
- [x] ``TrainHpoLivePanel`` shared live/post-run panel shell on HPO Tracker + Experiment Tracker (hundred-fifty-sixth pass; §G.18 / §G.17 / §D.7)
- [x] ``ProcessLogTail`` shared stdout tail display on HPO Tracker + Experiment Tracker live panels via ``TrainHpoLivePanel`` (hundred-sixty-third pass; §G.18 / §D.7)
- [x] ``trainHpoLivePanelTitle`` shared train/HPO live panel title helper; HPO Tracker + Experiment Tracker import shared title (hundred-seventieth pass; §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker ``TrainHpoLivePanel`` card headers pass ``runLabel`` + · live suffix via ``useProcessRunLabelBrush`` (hundred-seventy-second pass; §G.18 / §D.7)
- [x] HPO Tracker trial log dirs + report directory path-kind handoffs via ``PathHandoffButtons`` (two-hundred-and-twenty-fifth pass; §G.18 / §D.7)
- [x] Experiment Tracker MLflow run dirs + output directory list ``pathHandoffs`` via ``PathHandoffButtons`` / ``LoadedRunRow`` (two-hundred-and-twenty-fifth pass; §G.18 / §G.14 / §D.7)
- [x] HPO Tracker trial log dirs + report dir + Experiment Tracker MLflow run dirs use ``PathRunLabelChip`` ``handoff`` (two-hundred-and-twenty-seventh pass; §G.18 / §D.7)
- [x] HPO Tracker storage DB path + report directory use shared ``OpenPathToolbar``; reports labeled Output Browser + file-manager open (two-hundred-and-thirtieth pass; §G.18 / §D.7)
- [x] Experiment Tracker MLflow tracking URI path uses shared ``OpenPathToolbar`` (two-hundred-and-thirtieth pass; §G.18 / §D.7)
- [x] HPO Tracker trial log-dir table cells use shared ``OpenPathToolbar``; trial number as ``children`` (two-hundred-and-thirty-third pass; §G.18 / §D.7)
- [x] Experiment Tracker MLflow run-dir table cells use shared ``OpenPathToolbar``; run-id meta as ``children`` (two-hundred-and-thirty-third pass; §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker live-panel ``RunLabelHeaderSuffix`` uses shared ``OpenPathToolbar`` (two-hundred-and-thirty-fifth pass; §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker live-panel ``RunLabelHeaderSuffix`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-sixth pass; §G.18 / §D.7)
- [x] HPO Tracker + Experiment Tracker live-panel ``ProcessIdFooter`` brushes process-derived ``runLabel`` when set (two-hundred-and-thirty-seventh pass; §G.18 / §D.7)
- [x] HPO Tracker trial log-dir cells pass ``brushLabel`` / ``storedLabel`` from ``trialBrushLabel`` (two-hundred-and-thirty-eighth pass; §G.18 / §D.7)
- [x] Experiment Tracker MLflow run-dir cells pass explicit ``brushLabel`` (run name) (two-hundred-and-thirty-eighth pass; §G.18 / §D.7)

---

### §G.19 — Phase 19: Settings & First-Run Onboarding ✅

**Goal**: Provide a persistent Settings page so users can configure the project root and Python executable without touching environment variables, and surface a first-run onboarding banner.

- [x] Settings page (`pages/Settings.tsx`): Project Root section (text input + directory picker via Tauri dialog), Python Executable section (text input + path picker, overrides `which_python` resolution), Appearance section (dark/light theme radio), About section (Studio version + links)
- [x] `store/app.ts`: `projectRoot` and `pythonPath` fields persisted via Zustand `persist` + `partialize`; `setPythonPath` action
- [x] `pythonPath` threaded through `useSpawnProcess` → `spawn_python_process` Rust command as `python_executable: Option<String>`; empty string treated as `None` (falls back to `which_python`)
- [x] First-run banner in `TopBar.tsx`: shown when `projectRoot` is empty and mode is not `"settings"`; links directly to Settings with an "Open Settings" button
- [x] Sidebar "App" section with Settings entry and gear icon
- [x] `system::validate_project_root` Rust command: checks path exists, is a directory, contains `main.py`; called on blur + before save; shows inline `CheckCircle` / `XCircle` badge
- [x] `system::probe_python` Rust command: runs `<path> --version` synchronously, handles Python 2 (stderr) and Python 3 (stdout); shows resolved version string inline; called on blur + before save
- [x] Save blocked if either validation fails; toast shown with "Fix validation errors before saving"
- [x] Import/export settings JSON: "Export Settings" serialises `{projectRoot, pythonPath, theme}` to a user-chosen JSON file via `write_text_file`; "Import Settings" reads a JSON file and populates drafts for review before saving
- [x] First-run onboarding wizard: `OnboardingDialog` modal when `projectRoot` is empty; inline directory picker + validation; dismissible via `useLayoutStore.onboardingDismissed` persistence
- [x] Guided tour: `GuidedTour` 5-step overlay with `data-tour` spotlight rings (sidebar, command palette, simulation twin, output browser, launch/monitor); TopBar compass button, command palette action, Settings "Take Guided Tour", `Ctrl+Shift+/` shortcut; auto-offered after first onboarding; dismissal persisted via `guidedTourDismissed`
- [x] System theme following (§D.3 Option C): `theme` preference extended to `dark` / `light` / `system`; `effectiveTheme` resolves `prefers-color-scheme` via `useThemeSync`; TopBar + command palette cycle all three modes; Settings Appearance radio includes System (§G.19 / §D.3)
- [x] Settings project root / Python / import JSON / Arrow benchmark paths use shared ``OpenPathToolbar``; Arrow benchmark auto-classifies CSV/log handoffs (two-hundred-and-thirtieth pass; §G.19 / §D.7)

**Status**: §G.19 complete — all checklist items delivered.

---

### §G — Studio Complete ✅

All twenty phases (§G.0–§G.19) are delivered. WSmart-Route Studio is the primary desktop interface for launching simulations and training runs, browsing results, and performing post-hoc analytics. Post-§G analytics bridges continue under §A (e.g. §A.3 Policy Telemetry in hundred-ninth pass; §A.5 Optuna Plotly export in hundred-tenth pass; §A.4 Training Health in hundred-eleventh pass; §A.6 Failure Analysis in hundred-twelfth pass; §A.2 WandB attention heatmaps in hundred-thirteenth pass; §A.1 Route Solution visualizer in hundred-fourteenth pass; §A.6 route-diff failure overlay in hundred-fifteenth pass; §A.6 ECharts route-diff parity in hundred-sixteenth pass; §A.2 Studio attention ring-buffer in hundred-seventeenth pass; §A.4 HPO health prune metrics in hundred-eighteenth pass; §A.3 live policy telemetry stream in hundred-nineteenth pass; §A.3 SQLite cross-run telemetry trending in hundred-twentieth pass; §A.3 cross-run improvement trajectory chart in hundred-twenty-first pass; §A.3 trajectory brush + Benchmark Analysis panel in hundred-twenty-second pass; §A.3 chart brush filter + Simulation Summary panel in hundred-twenty-third pass; §A.3 chart brush dimming + Algorithm/City Comparison panels in hundred-twenty-fourth pass; §A.3 run_label brush sync + OLAP Explorer panel in hundred-twenty-fifth pass; §A.3 trajectory click fix + Simulation Monitor / Data Explorer panels in hundred-twenty-sixth pass; §A.3 Output Browser trends panel + KPI brush in hundred-twenty-seventh pass; §A.3 Process Monitor telemetry + Output Browser run_label auto-brush in hundred-twenty-eighth pass; §A.3 Simulation Launcher live telemetry + Process Monitor brush parity in hundred-twenty-ninth pass). Remaining release-engineering items (code-signing keys, hosted signed update CDN) are deferred per §G.8.

| Area | Status |
| --- | --- |
| Analytics dashboard (§G.1–§G.2) | ✅ |
| Geospatial + graph topology (§G.3–§G.4) | ✅ |
| ML introspection (§G.5) | ✅ |
| OLAP explorer (§G.6) | ✅ |
| UX polish + export surface (§G.7) | ✅ |
| Data packaging (§G.8) | ✅ (signing keys + CDN deferred) |
| Launchers + monitors (§G.9–§G.15) | ✅ |
| Streamlit parity pages (§G.16–§G.18) | ✅ |
| Settings + onboarding (§G.19) | ✅ |

---

### §G — Dependency Map

```
Phase 0  →  All phases
Phase 1  →  Phase 2, Phase 6
Phase 2  →  Phase 7
Phase 3  →  Phase 4
Phase 4  →  Phase 5 (pheromone + attention share Sigma.js)
Phase 5  →  Phase 7
Phase 6  →  Phase 7
Phase 7  →  Phase 8
Phase 15 →  Phase 9, Phase 10, Phase 11, Phase 12, Phase 16 (all share process streaming)
Phase 9  →  Phase 14 (on-completion navigates to Output Browser)
Phase 10 →  Phase 14 (on-completion opens checkpoint in Output Browser)
Phase 13 →  Phase 9, Phase 10, Phase 11, Phase 12 (config editor feeds all launchers)
Phase 14 →  Phase 1 (analytics dashboard load)
Phase 16 →  Phase 3 (map uses deck.gl from §G.3)
Phase 17 →  Phase 10 (training hub spawns; monitor reads)
Phase 18 →  Phase 1, Phase 17 (builds on analytics dashboard and training runs)
```

---

### Effort × Impact Matrix — WSmart-Route Studio

| Phase | Description | Effort | Impact | Priority |
| --- | --- | --- | --- | --- |
| §G.0 | Foundation & Tooling | High | Very High | P0 |
| §G.19 | Settings & Onboarding | Low | Very High | P0 |
| §G.15 | Real-Time Process Monitor | Medium | Very High | P0 |
| §G.9 | Simulation Launcher | Medium | Very High | P1 ✅ |
| §G.10 | Training & HPO Hub | Medium | Very High | P1 ✅ |
| §G.13 | Configuration Editor | Medium | High | P1 ✅ |
| §G.14 | Output Browser | Medium | High | P1 ✅ |
| §G.1 | Statistical Dashboard | High | High | P1 ✅ |
| §G.11 | Data Generation Wizard | Low | High | P2 ✅ |
| §G.12 | Evaluation Runner | Low | High | P2 ✅ |
| §G.7 | UX Polish | Medium | High | P2 ✅ |
| §G.2 | Drill-Down Sunburst | Medium | High | P2 ✅ |
| §G.3 | Geospatial deck.gl | High | High | P2 ✅ |
| §G.6 | OLAP Explorer | Medium | High | P2 ✅ |
| §G.4 | Graph Topology | Medium | Medium | P3 ✅ |
| §G.8 | Export & Packaging | Medium | High | P3 ✅ |
| §G.5 | ML Introspection | High | High | P3 ✅ |
| §G.16 | Simulation Digital Twin | High | Very High | P1 ✅ |
| §G.17 | Training Monitor | Medium | Very High | P1 ✅ |
| §G.18 | Experiment & HPO Tracker | High | High | P2 ✅ |

---

