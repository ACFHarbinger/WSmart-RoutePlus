> Split out of [`docs/moon/ROADMAP.md`](../ROADMAP.md) on 2026-08-27 so each theme can be updated independently. See that file for the Anchor Index and Cross-Cutting Themes table.

---

## C — Documentation

### §C.1 — API Reference Docs (mkdocstrings + MkDocs Material)

**Pain**: The module docs in `docs/` are hand-written Markdown files that describe architecture but do not reflect live code. Developers must read source files to find parameter names, return types, and class hierarchies. There is no search-indexed API reference.

**Options**

- **A** — Add `mkdocs` + `mkdocs-material` + `mkdocstrings[python]` as dev dependencies; configure `mkdocs.yml` to auto-generate API pages from existing docstrings. `[Quick Win]`
- **B** — Use `sphinx` + `sphinx-autodoc` + `furo` theme; more established but higher configuration overhead.
- **C** — Use `pdoc` for a zero-configuration auto-generated HTML reference; simpler but less feature-rich.
- **D** — Generate docs only for the public-facing `logic/src/interfaces/` layer; leave internal modules undocumented.

**Recommendation**: **Option A** — MkDocs Material is the modern standard, integrates well with GitHub Pages, and the `.nav` configuration can include the existing hand-written `docs/` pages alongside auto-generated API pages.

**Effort × Impact**: Medium effort / High impact

---

### §C.2 — Enforce Docstring Coverage with `pydoclint`

**Pain**: Public functions in the interfaces and models layers often lack docstrings, or have docstrings that omit parameter types/descriptions. MyPy catches type errors but not documentation gaps.

**Options**

- **A** — Add `pydoclint` to the `pre-commit` hooks and CI `quality-checks` job; fail on missing/mismatched docstrings for public functions. `[Quick Win]`
- **B** — Use `interrogate` (simpler, counts docstring presence percentage) as a softer gate.
- **C** — Configure `ruff` rule `D` (pydocstyle) — already partially available in ruff — for inline enforcement without a separate tool.

**Recommendation**: **Option C** — ruff is already the linter; adding the `D` rule family requires only a config line and keeps the toolchain minimal.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §C.3 — CHANGELOG.md

**Pain**: There is no structured changelog. Contributors cannot tell what changed between training runs or what API breaks occurred across model versions.

**Options**

- **A** — Adopt `Keep a Changelog` format (`CHANGELOG.md` at repo root); commit an initial entry retroactively from `git log`. `[Quick Win]`
- **B** — Use `git-cliff` to auto-generate the changelog from conventional commit messages; integrate into the CI release job.
- **C** — Use GitHub Releases with auto-generated release notes from PR labels.

**Recommendation**: **Option A** first (manual, immediate), then **Option B** to automate future entries once contributors adopt conventional commits.

**Effort × Impact**: Very Low effort / Medium impact `[Quick Win]`

---

### §C.4 — Architecture Diagrams as Code (Mermaid)

**Pain**: `docs/ARCHITECTURE.md` describes the system in prose. There are no visual diagrams showing the data flow from CLI → Pipeline → Environment → Model → Policy, or the Studio mediator pattern, making onboarding slow.

**Options**

- **A** — Embed Mermaid flowcharts directly in `docs/ARCHITECTURE.md`; GitHub renders them natively in Markdown. Add: training data flow, inference pipeline, simulation orchestration, Studio architecture diagrams. `[Quick Win]`
- **B** — Use `diagrams` (Python-as-code diagram library) to generate PNG architecture diagrams; commit the PNGs and Python sources.
- **C** — Use PlantUML for class diagrams of the interfaces layer; integrate into the MkDocs build (depends on §C.1).

**Recommendation**: **Option A** for immediate diagrams (zero tooling overhead, GitHub-native), **Option C** for the interfaces class diagram once §C.1 is set up.

**Effort × Impact**: Low effort / High impact `[Quick Win]`

---

### §C.5 — Jupyter Notebook Tutorials

**Pain**: There are no interactive examples showing how to: generate a VRPP instance, run inference with a trained AM model, compare ALNS vs. Gurobi on a benchmark instance, or load and visualize simulation results. Researchers must read source code to reproduce even basic experiments.

**Options**

- **A** — Add `notebooks/` directory with: `01_getting_started.ipynb`, `02_train_am_vrpp.ipynb`, `03_compare_policies.ipynb`, `04_simulation_analysis.ipynb`. Use the existing `main.py` API internally.
- **B** — Add `nbval` to CI to execute notebooks and validate outputs; prevents notebooks from rotting.
- **C** — Host interactive notebooks on Binder or Google Colab (badge in README).

**Recommendation**: **Option A** as the content investment, **Option B** to keep them passing. **Option C** is optional polish.

**Effort × Impact**: High effort / High impact

---

### §C.6 — Troubleshooting & Compatibility Docs Refresh

**Pain**: `docs/TROUBLESHOOTING.md` and `docs/COMPATIBILITY.md` exist but their content is unclear. CUDA version conflicts, Gurobi license errors, and display backend issues are the most common friction points for new contributors. With the Tauri migration, new Studio-specific setup steps (Rust toolchain, Node.js, Tauri CLI) must also be documented.

**Options**

- **A** — Audit both files; add sections for: Gurobi 11+ license setup, CUDA 12.x / PyTorch 2.2 compatibility matrix, `uv sync` common errors, HGS/PyVRP installation issues, and Tauri/Rust toolchain setup (`cargo tauri dev` prerequisites, Node.js version requirements).
- **B** — Add a `scripts/diagnose.sh` script that checks all critical dependencies and prints a structured health report.

**Recommendation**: **Option A + B** in parallel — one improves static docs, the other gives developers a live diagnostic tool.

**Effort × Impact**: Low effort / Medium impact

---

### §C.7 — CI Documentation Pipeline

**Pain**: Documentation is never validated in CI. A broken import in a module will silently prevent `mkdocstrings` from generating its API page; typos in Mermaid diagrams break rendering without any build error.

**Options**

- **A** — Add a `docs` job to `ci.yml` that runs `mkdocs build --strict`; fail on any warning. Depends on §C.1.
- **B** — Run `mkdocs gh-deploy` automatically on push to `main`, making the live docs always reflect the latest commit.
- **C** — Use `pre-commit` hooks to validate Mermaid syntax locally before commit.

**Recommendation**: **Option A** (blocking build check) + **Option B** (auto-deploy) once §C.1 is implemented.

**Effort × Impact**: Low effort / High impact (after §C.1)

---

### Effort × Impact Matrix — Documentation

| Item                                    | Effort   | Impact | Priority         |
| --------------------------------------- | -------- | ------ | ---------------- |
| §C.3 Option A (CHANGELOG.md)            | Very Low | Medium | P0 `[Quick Win]` |
| §C.2 Option C (ruff D rules)            | Very Low | Medium | P0 `[Quick Win]` |
| §C.4 Option A (Mermaid diagrams)        | Low      | High   | P0 `[Quick Win]` |
| §C.6 Option A (TROUBLESHOOTING refresh) | Low      | Medium | P1               |
| §C.1 Option A (MkDocs Material)         | Medium   | High   | P1               |
| §C.7 Option A (docs CI job)             | Low      | High   | P2 (after §C.1)  |
| §C.5 Option A (Jupyter notebooks)       | High     | High   | P2               |
| §C.5 Option B (nbval CI)                | Low      | High   | P2 (after §C.5)  |

---

