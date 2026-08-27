# WSmart-Route Studio — Documentation

WSmart-Route Studio is the Tauri 2.0 desktop application that serves as the sole
graphical interface of the WSmart+ Route framework (it superseded the former
Streamlit dashboard and PySide6 GUI in July 2026).

| Document | Contents |
| -------- | -------- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Process model, frontend source layout, state stores, Rust command surface, IPC conventions |
| [PAGES.md](PAGES.md) | Page-by-page guide to every Studio view (Monitor / Analysis / Launch / Files / App) |
| [GEN_PIPELINE.md](GEN_PIPELINE.md) | The native §H Analysis & Presentation engine (`src/gen/`) — reports, charts, deck, exporters |
| [TESTING.md](TESTING.md) | Vitest unit/component/integration suites and Cypress e2e tests |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Prerequisites, dev workflow, npm scripts, building bundles |

Related repository-level docs live in [`../../docs/`](../../docs/), notably
`docs/moon/roadmaps/studio.md` (§G Studio phases) and
`docs/moon/roadmaps/presentation_studio.md` (§H Analysis & Presentation Studio)
and the framework module docs referenced from `AGENTS.md`.
