# WSmart+ Route — Moon Roadmap

> **Version**: 1.1  
> **Last Updated**: July 2026  
> **Status**: Living Document — updated each sprint  
> **Scope**: Logic layer (`logic/src/`), GUI layer (migrating from `gui/src/` PySide6 → Tauri), CI/CD, documentation

This document captures medium-to-long-horizon improvements for the WSmart+ Route framework across eight dimensions: Analytics & Interpretability, Architecture, Documentation, GUI/UX, New Features, Performance, the WSmart-Route Studio Tauri application, and the Analysis & Presentation Studio migration. Each item follows a **Pain → Options → Recommendation** structure with effort/impact tags.

Tags: `[Quick Win]` ≤ 1 day · `[Research]` involves novel work · `[Blocked]` depends on another item

---

## Anchor Index

> Each section below now lives in its own file under [`docs/moon/roadmaps/`](roadmaps/), split out on 2026-08-27 so themes can be updated independently. This file keeps the index and the Cross-Cutting Themes table.

| Section                                                              | Topic                                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [§A — Analytics & Interpretability](roadmaps/analytics_interpretability.md) | Telemetry, attention maps, policy dashboards, HPO analytics |
| [§B — Architecture](roadmaps/architecture.md) | Test coverage, plugin system, logging, type safety, interfaces |
| [§C — Documentation](roadmaps/documentation.md) | API docs, architecture diagrams, Jupyter notebooks, CI docs pipeline |
| [§D — GUI / UX](roadmaps/gui_ux.md) | Route visualization, training progress, themes, session persistence |
| [§E — New Features](roadmaps/new_features.md) | Multi-problem benchmarking, REST API, LLM integration, export formats |
| [§F — Performance](roadmaps/performance.md) | Batched inference, GPU memory, test suite speed, simulation throughput |
| [§G — WSmart-Route Studio](roadmaps/studio.md) | Tauri 2.0 app: analytics, geospatial, ML introspection, launcher UIs |
| [§H — Analysis & Presentation Studio](roadmaps/presentation_studio.md) | Migration of `logic/gen/` report + deck generation into the Studio |
| [§I — Publication & Dissemination](roadmaps/publication_dissemination.md) | MPVRPP paper, reproducible LaTeX generation, public website |


---

## Cross-Cutting Themes

Several items across sections are tightly coupled and should be sequenced together:

| Cluster                        | Items                          | Rationale                                                                                |
| ------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------- |
| **Plugin System**              | §B.3, §B.6                     | Policy and env registration should share the same Hydra-based mechanism                  |
| **Async Worker Contract**      | §B.8, §D.5, §G.15              | Rust AsyncTask trait + Python BackgroundTask protocol are prerequisites for cancel UX    |
| **Route Visualization**        | §A.1, §D.1, §G.3               | All three need the same spatial renderer; deck.gl in §G.3 is the shared base             |
| **Docs Infrastructure**        | §C.1, §C.7                     | MkDocs setup is a prerequisite for the CI docs pipeline                                  |
| **Test Quality**               | §B.1, §F.3                     | Coverage uplift and test-suite speed are best addressed together                         |
| **Telemetry**                  | §A.3, §A.4                     | PolicyVizMixin and TrainingHealthCallback both feed the Studio analytics dashboard        |
| **Config System**              | §B.3, §B.6, §D.6, §G.13        | Plugin registry + Hydra `_target_` + Studio config editor all depend on a clean config schema |
| **Process Streaming**          | §G.9, §G.10, §G.11, §G.12, §G.15, §G.16 | All launchers share the same Rust→React stdout streaming infrastructure from §G.15 |
| **Streamlit Parity**           | §G.16, §G.17, §G.18            | These three phases are a 1:1 port of the three most-used Streamlit modes; complete before removing Streamlit dependency |
| **Document Authoring**         | §H.0–§H.8, §G.0, §G.1, §G.3, §G.6 | The Analysis & Presentation Studio reuses the Arrow/DuckDB pipeline, ECharts components, deck.gl maps and OLAP explorer; chart components should be built once and shared between dashboards (§G.1) and documents (§H.2) |

---

_This roadmap is a living document. Update item status inline (✅ Done, 🚧 In Progress, ❌ Blocked) and refresh the Effort × Impact matrices each quarter._
