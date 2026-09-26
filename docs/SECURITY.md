# Security

Security posture and practices for WSmart+ Route.

## Scope

- Simulation, training, and evaluation pipeline (`logic/src/`)
- WSmart-Route Studio desktop application (Tauri/React, `app/`)
- Local tracking database, checkpoints, and cached datasets
- Optional exact-solver backend (Gurobi) and its license/credentials
- Optional REST API / remote-inference backend (see `docs/moon/roadmaps/new_features.md` §E.3), when built
- Submodules (the MPVRPP paper and its conference abstract) and third-party vendors

## Principles

1. **Local-first:** training, simulation, and evaluation run on the user's
   machine or cluster; no private data or model weights are sent to a
   third-party API unless the user explicitly configures one (e.g. an
   LLM-assisted instance generator, see roadmap §E.6).
2. **Explicit process invocation:** prefer `subprocess` with argument lists
   over shell strings.
3. **Path boundaries:** user-selected data/output directories should be
   validated; reject unexpected traversal when reading or writing files
   (dataset loaders, checkpoint browsers, the Studio's file pickers).
4. **Secrets:** Gurobi license files, API keys, and any future backend
   credentials live in env files (see `env/`) or OS keychains; never commit
   live secrets. `.gitignore` excludes `**/.env`.
5. **Dependencies:** follow [DEPENDENCY_POLICY.md](DEPENDENCY_POLICY.md);
   audit high-risk ML, solver, and network packages before upgrading.

## Reporting

Report security issues privately to the project maintainer (ACFHarbinger,
afonso.fernandes100@gmail.com) rather than opening a public issue with
exploit detail.

## Related

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEPENDENCY_POLICY.md](DEPENDENCY_POLICY.md)
