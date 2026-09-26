# Native Generation Engine (`src/gen/`) — §H

The `gen/` tree is the native TypeScript port of the archived Python
report/deck pipeline (`logic/gen/`). It powers the **Report Studio** page and
runs entirely in-app: data loading through Rust commands, figures as ECharts
renders, documents via `pptxgenjs` / `docx` / `exceljs` / `jspdf`.

```
gen/
├── assets/        # bundled logos, icons, illustration sources (typed via index.ts)
├── config/        # typed JSON configs: themes, presentation content, analysis
│                  # configs, simulation metadata (filename-decoding tables),
│                  # reference links
├── data/
│   ├── simulation.ts  # output-tree parser, horizon CSVs, scenario detection,
│   │                  # CF/SL aggregation, Pareto, results matrix, md tables
│   └── dataset.ts     # NPZ stats (quartiles/IQR/mode/skew), KDE, histograms
├── charts/        # ECharts option builders for every figure (simulation +
│                  # dataset families) + headless render-to-PNG (common.ts)
├── report/        # markdown report builders, bin maps, interactive HTML
│                  # exports, self-contained HTML report export
├── deck/          # 21-slide PPTX builder: native-shape diagrams, seeded SVG
│                  # illustrations, MathJax equations (+ plain-text fallback),
│                  # native results table, speaker-script DOCX, results XLSX,
│                  # HTML slideshow export, PDF export
└── io.ts          # Rust command wrappers (read/write, list, npz)
```

## Data flow

1. **Ingest** — `data/simulation.ts` walks `assets/output/<horizon>days/` via
   `list_files_recursive`, decoding filename-encoded policy metadata
   (strategy prefixes, CF/SL variants, constructor tokens, acceptance,
   improver) from `config/simulation_metadata.json`. Everything downstream
   (scenarios, strategies, constructors) is auto-detected from data; filters
   can narrow it.
2. **Aggregate** — CF/SL variant averaging, per-scenario metric vectors,
   Pareto fronts (`paretoIndices`), hierarchical results matrix, markdown
   tables.
3. **Figures** — `charts/*` build ECharts options from the aggregated data and
   the merged theme (`config/themes.json` + ported mplstyle values); the same
   options power the in-app preview and headless PNG export, so preview ≡
   export.
4. **Documents** —
   - `report/` renders the dataset and simulation markdown reports (figure and
     table numbering, TOC), self-contained HTML export, and interactive
     single-file HTML chart pages (offline, ECharts inlined).
   - `deck/` renders the 21-slide presentation as PPTX (native shapes, native
     table with merged spans and best-cell highlighting, editable text),
     plus the speaker-script DOCX, results XLSX, an HTML slideshow, and a
     rasterised PDF.

## Engines in Report Studio

- **Native** (default) — everything above, with live progress log and artefact
  chips; previews render in-app (`components/gen/ReportPreview`,
  `components/gen/DeckPreview`).
- **Legacy** — spawns the archived Python scripts under `logic/gen/`
  (`gen_simulation_analysis.py`, `gen_dataset_analysis.py`,
  `gen_presentation.py`) through the process runner. Kept for parity checks;
  frozen, bugfix-only.

## Extending

- New strategies/constructors need **no code change** — extend
  `config/simulation_metadata.json` (prefix/token tables).
- New figures: add a builder in `charts/`, reference it from the report/deck
  builder; use theme lookups instead of hard-coded colours/fontsizes.
- Deck text/content lives in `config/presentation_content.json`, not in code.
