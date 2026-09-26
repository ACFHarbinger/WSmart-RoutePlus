> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## H — Analysis & Presentation Studio

> Migration of the `logic/gen/` generation pipeline (archived at `archive/gen/` in 2026-07, **revived to `logic/gen/` in 2026-08** when it was needed again for the MPVRPP paper — see §I) — `gen_dataset_analysis.py`, `gen_simulation_analysis.py`, `gen_presentation.py`, `report_utils.py` and their JSON/Jinja/mplstyle assets — into WSmart-Route Studio as a first-class **document authoring subsystem**. This is explicitly **not a 1:1 port**: the Python scripts are a batch pipeline with hardcoded geometry, colours, fontsizes and content baked into code; the Studio replaces them with a declarative, fully data-driven document model, native TS/Rust rendering, live editable previews, and a far richer feature set for building analysis reports and presentation decks.

**Design decisions** (agreed 2026-07):

1. **Full TS/Rust native rendering** — ECharts renders every chart for both preview and export (PNG/SVG from the same component instance); Rust owns OOXML generation (PPTX/DOCX/XLSX). No Python in the loop: preview ≡ output by construction. `matplotlib`, `plotly`, `python-pptx`, `docxtpl`, `openpyxl` and `pandoc` are all retired for this pipeline.
2. **All output formats, plus new ones** — markdown report + figures, PPTX deck, speaker-script DOCX, XLSX results workbook (parity), **plus** PDF export and a self-contained interactive HTML report/deck (replacing the standalone Plotly HTML files).
3. **Hybrid document model: declarative schema + per-element override patch layer** — a versioned document spec (typed element trees, layout constraints, data bindings) drives default layout and regeneration; user edits (nudged logo, resized figure, changed legend fontsize) are stored as a patch layer *on top of* the schema so they survive data refreshes and regeneration.
4. **Full preview/editing UX suite** — per-element inspector panel, direct canvas manipulation, global theme/style editor, and a diff-&-regenerate workflow.

**Additional technology** (extends the §G stack)

| Concern | Technology |
| --- | --- |
| Chart rendering + export | Apache ECharts (canvas/SVG renderers; `getDataURL`/SVG string export) |
| PPTX/DOCX/XLSX writing (Rust) | OOXML zip assembly via `zip` + `quick-xml` (or `rust_xlsxwriter` for XLSX) |
| Equation preview | KaTeX (in-app live rendering of LaTeX) |
| Equation export | LaTeX → MathML (WASM/Rust converter) → MathML-to-OMML XSLT / native Rust transform |
| PDF export | Tauri print-to-PDF via WebView, or headless render of the HTML deck |
| Document spec | Versioned JSON schema (`document.schema.json`), validated in Rust + TS (serde / zod) |
| Diagram editing | SVG-based node/connector canvas (custom React, reusing Zustand state) |

---

### §H.— — Interim: Report Studio launcher ✅

**Goal**: Make the batch pipeline usable from the Studio *today*, before the native phases ship, and freeze the scripts out of the active tree.

- [x] Archive the pipeline: `gen_dataset_analysis.py`, `gen_simulation_analysis.py`, `gen_presentation.py`, `report_utils.py` + `jinja/ json/ style/ js/ templates/ images/ svg/ links/` moved from `logic/gen/` to `archive/gen/` (frozen, bugfix-only); `gen_dist_matrix.py`, `export_for_studio.py`, `export_loss_landscape.py` remain in `logic/gen/`
- [x] **Report Studio** page (`report_studio` mode, Launch section): three tabs assembling the full CLI for the archived scripts — Dataset Analysis (theme, NPZ/TD CSVs, NPZ dir, out-md, figures-dir, force/figures-only), Simulation Analysis (report mode: fontsize, pareto-points, repeatable horizons, scenario/strategy/constructor/improver/acceptance filters, map-mode, heatmap-labels; parse mode: raw output tree → summary CSV), Presentation Deck (figures-dir, out PPTX, author/coauthors/groups, results-table + split, speaker-script DOCX, image-mode, XLSX export)
- [x] Persisted form state (`useReportGenStore`), command preview, spawn via shared process infra (`reportgen_*` ids), live log tail + status pill, post-run artefact path chips (markdown / CSV / PPTX / DOCX / XLSX)

> Update (same day): the native §H engine shipped for §H.1–§H.6 core scope — the Report Studio page now defaults to the **Native** engine (in-app ECharts figures, MathJax equations, pptxgenjs/docx/exceljs exporters) with the archived scripts behind a **Legacy** toggle. Remaining §H gaps: document spec + override patch layer (§H.0), native OMML equations (§H.4), report PDF export (§H.5), and the inspector/direct-manipulation editing UX (§H.7) — interactive HTML report charts (§H.5), the HTML deck slideshow + deck PDF (§H.6), and the paginated deck preview canvas (§H.7) shipped 2026-07-16. A **headless runner** (`app/scripts/gen-headless.ts`, 2026-07-16) drives the same native engine outside the webview (vite-node + jsdom/node-canvas/resvg + a Node fs shim for the Tauri commands), so analysis + deck regenerate from the CLI without the GUI. Current deliverables (figures + PPTX) are regenerated with the **archived Python pipeline** (`archive/gen/`, user preference, 2026-07-16); the native engine keeps parity fixes and remains the §H target.

---

### §H.0 — Phase 0: Document Spec & Theme Foundation

**Goal**: Define the data model everything else consumes — no element, colour, fontsize, coordinate or label may be hardcoded in rendering code.

- [ ] **Document schema**: versioned JSON schema for `report` and `deck` documents — a tree of typed elements (`section`, `slide`, `chart`, `table`, `equation`, `diagram`, `image`, `text`, `caption`, `legend-box`), each with a stable ID, data binding, style ref and layout constraints (grid/flex-like: rows, columns, weights, anchors) instead of absolute EMU/inch coordinates
- [ ] **Override patch layer**: per-element patch records (`{elementId, path, value}`) stored separately from the base spec; applied at render time; survive regeneration; invalidated-with-warning when the target element disappears
- [ ] **Theme system**: replaces `themes.json` + `*.mplstyle` + the `FS_*` fontsize-scale machinery — named themes (dark/light/custom) defining palettes (city/dist/strategy/improver/constructor/scenario/metric colour maps from `simulation_metadata.json` and `dataset_analysis_config.json`), per-element-class font scales (axis, label, legend, title), grid/marker/hatch cycles; editable and persistable as user themes
- [x] **Content model (interim)**: all deck text lives in `presentation_content.json`, imported as typed data (`gen/config`); full document-spec migration pending
- [ ] **Serde + zod validation**: same schema validated on both sides of the IPC boundary; schema-migration hook for version bumps
- [ ] **Rust commands**: `load_document`, `save_document`, `apply_patch`, `list_themes`, `save_theme`
- [ ] Automatic Figure/Table/Equation numbering as a render-time pass over the element tree (ports `apply_figure_table_numbers`)

---

### §H.1 — Phase 1: Data Ingestion & Statistics Engine

**Goal**: Port every data-side capability of the Python scripts to Rust/DuckDB, reusing the §G.0 Arrow IPC pipeline.

- [x] **Raw output-tree parser**: `app/src/gen/data/simulation.ts` — `parseFilename`/`parseAreaDir`/`parseOutputDir` decode filename-encoded policy metadata from `simulation_metadata.json` (kept as data) over the Rust `list_files_recursive` walker; native parse-output→CSV mode ships in Report Studio (ports `parse_output_dir` / `_parse_filename` / `_parse_area_dir`): walk `assets/output/<horizon>days/`, decode filename-encoded policy metadata (strategy prefixes, CF/SL variants, constructor tokens, acceptance, improver) using the parsing metadata currently in `simulation_metadata.json` — kept as data, editable in the app, so new strategies/constructors need no code change
- [ ] **Summary dataset builder**: emit Arrow IPC / Parquet instead of CSV; ingest into DuckDB-Wasm for filtering, aggregation and pivots (subsumes `aggregate`, `filter_data`, `detect_scenarios`, variant labelling)
- [x] **NPZ/TD statistics**: `app/src/gen/data/dataset.ts` + Rust `load_npz_flat` — extended stats (median, variance, quartiles, IQR, fences, binned mode, skewness from the stats CSVs) plus Gaussian KDE and histogram binning (ports `gen_dataset_analysis` data layer): Rust NPZ reader (`ndarray-npy`, shared with §G.5.1) computing extended stats — median, variance, quartiles, IQR, fences, binned mode, skewness — plus KDE and histogram binning for distribution-shape charts
- [x] **Pareto engine**: `paretoIndices` in `data/simulation.ts` (ports `pareto_indices`) as a reusable Rust function exposed to chart specs
- [x] **Scenario/policy auto-detection with config narrowing**: `detectScenarios`/`filterData`/`buildCtx` + Report Studio filter fields: everything auto-detected from data; the document spec can narrow scenarios, strategies, constructors, improvers, acceptance criteria (ports the CLI filter flags into interactive filter UI)
- [x] **Coordinate loading + repair**: `report/binMaps.ts` (ports `_load_bin_coords`, `_fix_stripped_decimal`): city coordinate sources declared in config, stripped-decimal recovery preserved
- [x] **Multi-horizon support**: horizon specs load smallest→largest; cross-horizon comparison charts + all-horizons results table derive automatically: horizon sets (30d/90d/…) as first-class dimension; cross-horizon comparison datasets derived automatically

---

### §H.2 — Phase 2: Chart Library Parity & Beyond

**Goal**: Reimplement every figure type as a parameterised ECharts component driven by a chart spec (type + data binding + style ref), used identically for in-app preview and export.

- [x] Pareto scatter with step-line fronts (`charts/simulation.ts` `buildParetoScatter`), per-scenario front colours, filled/open improver markers, linear/symlog x-scale toggle, all-points vs front-only mode (ports `gen_pareto_scatter` + the interactive Plotly variant with its button toggles — now just component state)
- [x] KPI bar charts with min/max error bars (`buildKpiBar`/`buildKpiCombined`; hatch decals for improvers), strategy colours, improver hatch/opacity pairing, symlog variants; combined multi-panel KPI figure (ports `gen_kpi_bar`, `gen_kpi_combined`, `_kpi_panel`)
- [x] Violin/box/histogram+KDE distribution charts (`buildKmViolin`, `charts/dataset.ts` violin/box/hist+KDE) (ports `gen_km_violin`, `gen_npz_violin`, `gen_npz_box`, `gen_npz_hist_kde`)
- [x] Policy×scenario and scenario×constructor heatmaps (RdYlGn + symlog normalisation, label mode, empirical facet) with symlog colour normalisation, metric toggle, optional shared side-legend mode (ports both heatmap generators + Plotly heatmap)
- [x] Strategy/improver bubble charts (offset-candidate label dodging ports `_annotate_no_overlap`) with collision-avoiding annotation placement (ports `_annotate_no_overlap` as a label-layout utility)
- [x] Constructor ranking (average-rank bars) and cross-horizon comparison charts (`buildConstructorRanking`, `buildHorizon*`) (ports `gen_constructor_ranking`, `gen_horizon_comparison`)
- [x] Radar charts (curated subset + all-constructors) with normalised axes (`buildRadar`) (ports `_render_radar`)
- [x] Dataset-analysis line/scatter families (`charts/dataset.ts` size-scaling/horizon/city/TD-alignment/extended-stats grids with FFZ-350 reference diamonds): size-scaling lines with reference-city diamond markers, horizon comparison, city comparison, TD/NPZ alignment, extended-stats grid (ports the `_plot_line_plus_ref` family)
- [x] Bin-location maps (scatter mode; street context lives in the deck.gl Digital Twin): `report/binMaps.ts` — replaces OSMnx street basemaps with the existing deck.gl/MapLibre stack (§G.3) — all-bins and selected-bins per scenario, exportable as PNG
- [x] Hierarchical results table as a native element: native PPTX table with merged row/column spans, global best-cell highlighting, split partitioning (`deck/resultsTable.ts`); GFM renderer for reports (merged row/column header spans, global best-cell highlighting, partition/split by any hierarchy level, corner note, target-size stretching — ports `render_hier_table_image` + `compute_global_best` but as a DOM/SVG table, not a raster)
- [ ] **Beyond parity**: every chart gets the §G.1 interactivity for free (tooltips, brushing, cross-filtering, log toggles) in preview; export snapshots any configured state

---

### §H.3 — Phase 3: Diagram Template Engine

**Goal**: Replace the hand-placed native-shape slides and matplotlib-drawn conceptual diagrams with data-driven, editable diagram templates.

- [x] Diagram template types (`deck/deckBuilder.ts` native shapes + `deck/illustrations.ts` SVG): `pipeline` (chevron stages), `column-grid` (policy grid / algo taxonomy), `tree` (DoE tree, B&B tree), `flow` (framework objective: sources → simulator → benchmark), `scene` (bin/truck simulator illustration), `annotated-plot` (LS operators, explore/exploit landscape, trajectory, population, knapsack, QA route illustration)
- [ ] Each diagram is spec data (nodes, labels, colours-by-role, connectors) + a layout algorithm — the current hardcoded EMU coordinates and stage/taxonomy text become editable content
- [ ] SVG diagram renderer with selectable/movable nodes and re-routable connectors; manual positions recorded in the override patch layer
- [x] Seeded procedural illustrations: `qaRouteIllustration(seed, nNodes)` + trajectory/population generators over a mulberry32 PRNG (`deck/svg.ts`)
- [x] Export path: SVG → PNG rasterisation for PPTX embedding (`svgToPngDataUrl`)

---

### §H.4 — Phase 4: Equation Pipeline

**Goal**: Native editable equations without pandoc.

- [x] Equations stored as LaTeX strings in the content model (`presentation_content.json`), with per-equation size/colour/alignment style refs
- [x] Offline LaTeX rendering via MathJax → SVG → PNG (`deck/equations.ts`); slide-canvas WYSIWYG preview still pending
- [ ] Export: LaTeX → MathML → OMML transform in Rust (or WASM) producing the same `mc:AlternateContent`/`a14:m` embedding with plain-text fallback that `_latex_to_omath`/`_apply_equation_to_paragraph` produce today — native editable equations in PowerPoint
- [x] Plain-text fallback generator: `deck/fallback.ts` (ports `_plain_fallback` symbol table, kept as data; verified against the Python reference)
- [ ] Equation numbering integrated with the Phase 0 numbering pass

---

### §H.5 — Phase 5: Report Builder

**Goal**: Replace the Jinja-templated markdown reports with a composable report document.

- [ ] Report composer view: section tree (auto-numbered TOC), narrative text blocks with analysis placeholders, chart elements, tables (Pareto-front membership, KPI min/max/mean, best-per-strategy, NPZ/TD/extended stats, full results matrix)
- [x] Dataset-analysis and simulation-analysis report templates shipped as native TS builders (`report/datasetReport.ts`, `report/simulationReport.ts`) reproducing today's report structures (reproducing today's report structures as starting points, fully editable)
- [x] Multi-horizon report assembly: per-horizon sections small→large + conditional cross-horizon comparison section (ports the orchestration in `gen_simulation_analysis.main`)
- [x] Markdown export (GFM tables, `<figure>` full-width images, Figure/Table numbering — `report/markdown.ts` ports `finalize_markdown`)
- [x] Interactive HTML exports: self-contained pages with the ECharts bundle inlined — pareto (all-points/front views), strategy bubble, policy heatmap (metric toggle), NPZ stats scatter, waste-distribution bars, city & network grid; written to the private dirs with report links (`report/interactiveHtml.ts`; replaces the Plotly-CDN HTML + injected JS snippets)
- [x] Self-contained HTML report export: generated markdown → standalone styled HTML with all figures inlined as data URLs (`report/htmlReport.ts`; "HTML report" toggles on the native engine)
- [ ] PDF export of the report

---

### §H.6 — Phase 6: Presentation Builder & Exporters

**Goal**: Replace `gen_presentation.py` with a deck composer plus native Rust exporters.

- [x] Slide layout templates delivered as builder functions (`cover`, agenda cards, equation+figure, diagrams, figure, results-table, dark-statement); interactive composer still pending: slide list with layout templates (`cover`, `agenda-cards`, `equation+figure`, `diagram`, `figure` with side/bottom legend, `results-table`, `dark-statement` for acknowledgements/Q&A) — all layout parameters (logo slots, band positions, column splits) schema-driven with theme defaults, never hardcoded
- [x] Built-in "WSmart-Route results deck" reproducing the 21-slide structure (`deck/deckBuilder.ts`, content from `presentation_content.json`)
- [x] **PPTX exporter** (pptxgenjs OOXML assembly in TS; equations embed as MathJax images pending §H.4 OMML): shapes, text frames, pictures, connectors, native tables — shapes, text frames, pictures, connectors, native tables, embedded OMML equations (Phase 4), correct 16:9 geometry from the layout engine's resolved boxes
- [x] **Speaker-script DOCX exporter** (TS `docx`, per-slide notes; `deck/speakerScript.ts`) — replaces docxtpl; a Rust OOXML rewrite remains optional (ports `gen_speaker_script`)
- [ ] ~~Rust DOCX exporter~~ superseded by the TS exporter for the speaker script (per-slide notes composed from element captions/notes fields — ports `gen_speaker_script` without docxtpl)
- [x] **XLSX exporter** for the results matrix (exceljs: header fills, alternating rows, best-cell highlight fills — ports `export_results_excel`)
- [x] **PDF export** of the deck: off-screen WebKit rasterisation of the HTML deck slides (html-to-image) assembled into a compressed 16:9 landscape PDF via jsPDF (`deck/pdfExport.ts`) — no print dialog, preview ≡ output
- [x] **HTML deck export**: self-contained offline slideshow — keyboard/click navigation, hash-synced index, speaker-notes panel (`N`), HTML/CSS diagrams, embedded figures/equations, real HTML results table with merged spans + best-cell highlighting (`deck/htmlDeck.ts`; interactive-chart embedding pairs with the §H.5 pages)
- [x] Results-table slide options: horizon selection (single/all), split-by-level partitioning, CLS-only filtering — Report Studio form settings (`deck/resultsTable.ts`)

---

### §H.7 — Phase 7: Live Preview & Editing UX

**Goal**: The core reason for the migration — see and adjust the exact output before producing files, replacing the regenerate-and-inspect loop and the `--fontsize-*` CLI flags.

- [x] **Preview canvas (deck)**: paginated in-app slide preview in Report Studio — shadow-root isolation, true 1600×900 aspect with fit-to-width scaling, ←/→ pagination, speaker-notes strip, rebuild-from-settings; rendered by the same builder as the HTML/PDF exports (`components/gen/DeckPreview.tsx`)
- [x] **Preview canvas (report)**: scrolling in-app preview of the generated report via the shared §H.5 HTML builder in a sandboxed iframe (`components/gen/ReportPreview.tsx`); spec/patch-driven re-rendering arrives with §H.0
- [ ] **Inspector panel**: click any element (chart, axis, legend, logo, diagram node, caption, table) → typed controls for its style/layout properties — fontsize per element class, colours, alignment, scale toggles (linear/symlog), legend position, marker sizes; edits write to the patch layer and re-render instantly
- [ ] **Direct manipulation**: drag/resize elements on the canvas, drag diagram nodes and connector endpoints, snap guides + alignment/distribution tools (e.g. realign the cover-slide logos by eye); positions recorded as patches
- [ ] **Global theme editor**: edit palettes and per-class font scales once, watch every chart/slide update live (supersedes `set_chart_fontsize` and the mplstyle files); switch dark/light per document
- [ ] **Diff & regenerate**: when underlying data changes (new simulation run, refreshed CSV), show per-element old/new preview diff; accept/reject regeneration per element; manual patches preserved where the element survives, flagged where it doesn't
- [ ] **Export panel**: one-click multi-format export (PPTX + DOCX + XLSX + PDF + HTML + markdown) with per-format options; export uses the identical resolved spec as the preview
- [ ] Undo/redo across spec + patch edits (Zustand middleware); document autosave

---

### §H.8 — Phase 8: Beyond Parity

**Goal**: Capabilities the batch scripts could never offer.

- [ ] Template gallery: save any document as a reusable template; parameterise by dataset/run so a new conference deck is "pick template + pick runs"
- [ ] Chart spec designer: build new chart types from the data model (dimension/measure wells over DuckDB, ports naturally onto §G.6 OLAP work) and drop them into documents
- [ ] Live data binding: documents reference runs/output dirs, not frozen CSVs — a document can be re-resolved against a newer run with the Phase 7 diff workflow
- [ ] Deck/report version history with named snapshots; diff between snapshots
- [ ] Batch export CLI (headless Tauri command or thin Rust binary) for CI regeneration of reports when new benchmark data lands (pairs with §B.2)
- [ ] Collaborative review annotations (comments pinned to elements) for co-author feedback before export

---

### §H — Dependency Map

```
Phase 0 (spec + themes)      →  all phases
Phase 1 (data engine)        →  Phase 2, Phase 5, Phase 6
Phase 2 (chart library)      →  Phase 5, Phase 6, Phase 7
Phase 3 (diagrams)           →  Phase 6, Phase 7
Phase 4 (equations)          →  Phase 6
Phase 5 (report builder)     →  Phase 7
Phase 6 (deck builder)       →  Phase 7
Phase 7 (preview/editing)    →  Phase 8
§G.0 (Arrow/DuckDB)          →  Phase 1
§G.1 (ECharts panels)        →  Phase 2 (shared components)
§G.3 (deck.gl maps)          →  Phase 2 (bin-location maps)
§G.6 (OLAP explorer)         →  Phase 8 (chart spec designer)
```

Python scripts are retired per-capability: `gen_dataset_analysis.py` after Phases 1+2+5; `gen_simulation_analysis.py` after Phases 1+2+5; `gen_presentation.py` after Phases 3+4+6. They now live frozen (bugfix-only) under `archive/gen/`, launchable via the interim Report Studio page, until their replacement phase ships end-to-end exports verified against reference outputs.

---

### Effort × Impact Matrix — Analysis & Presentation Studio

| Phase | Description | Effort | Impact | Priority |
| --- | --- | --- | --- | --- |
| §H.0 | Document Spec & Themes | Medium | Very High | P0 |
| §H.1 | Data Ingestion & Stats Engine | Medium | Very High | P0 `[Blocked]` §G.0 |
| §H.2 | Chart Library Parity | High | Very High | P1 `[Blocked]` §H.0, §H.1 |
| §H.4 | Equation Pipeline | Medium | High | P1 `[Blocked]` §H.0 |
| §H.6 | Presentation Builder & Exporters | High | Very High | P1 `[Blocked]` §H.2, §H.3, §H.4 |
| §H.3 | Diagram Template Engine | Medium | High | P2 `[Blocked]` §H.0 |
| §H.5 | Report Builder | Medium | High | P2 `[Blocked]` §H.1, §H.2 |
| §H.7 | Live Preview & Editing UX | High | Very High | P1 `[Blocked]` §H.2 |
| §H.8 | Beyond Parity | High | Medium | P3 `[Blocked]` §H.7 |

---

