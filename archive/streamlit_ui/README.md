# Archived Streamlit Dashboard UI

The Streamlit dashboard removed in commit `31b1b19474bccf784618534a15e120493d57fb95`
("refactor!: remove Streamlit dashboard UI module (superseded by Studio)",
2026-07-15), recovered from its parent commit and archived here — frozen,
reference-only. (The former `archive/gen/` report generators were revived and
now live under `logic/gen/`.)

The WSmart-Route Studio (`app/`) is the sole supported interface; this code is
kept for reference (chart/analytics logic, pydeck map styling, service-layer
query patterns) and is not imported by the live codebase, packaged, linted, or
tested.

## Layout

Paths mirror where the files lived in the repository at deletion time:

| Archive path | Original role |
| ------------ | ------------- |
| `.streamlit/config.toml` | Streamlit server configuration |
| `logic/configs/ui/*.yaml` | Hydra configs for dashboard panels |
| `logic/controllers/dashboard_entry.py` | `main.py dashboard` entry point |
| `logic/src/constants/dashboard.py` | dashboard constants (was re-exported from `logic/src/constants`) |
| `logic/src/ui/` | the dashboard package: `app.py`, components (charts, maps, attention/policy viz), pages (simulation, training, benchmark, trackers), services (loaders, log parser, tracking), styles, HTML templates |
| `logic/src/utils/ui/` | map/simulation-page helpers |
| `logic/src/utils/package/remove_ui.py` | packager script that stripped the UI from bundles |
| `logic/test/unit/ui/` | unit tests for pages/services/policy viz |
| `tools/ui/justfile` | dashboard just targets |

## Running (historical)

The dashboard was launched via `streamlit run` through
`logic/controllers/dashboard_entry.py` (`main.py dashboard`) and required the
`streamlit`, `plotly`, `pydeck`, and `folium` dependencies, which were removed
from `pyproject.toml` in the same commit. To resurrect it, restore the files to
their original paths and reinstate those dependencies.
