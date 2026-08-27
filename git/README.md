# git/

Human-browsable git tooling and reference docs, kept separate from the
dot-prefixed `.git/` directory (which git itself reads) so contributor-facing
material is easy to find, read, and edit.

| Path | Purpose |
| --- | --- |
| `CONTRIBUTING.md` | Code style, Git workflow, PR process, development guidelines |
| `CODE_OF_CONDUCT.md` | Contributor Covenant v1.4 |
| `CODEOWNERS` | Default reviewer assignment (reference doc; not wired into `.github/` as a live CODEOWNERS file) |
| `codecov.yaml` | Coverage reporting configuration |
| `config/project_labels.json` | Reference snapshot of the GitHub label taxonomy actually in use — not consumed by automation, just documentation |
| `messages/` | `git commit -F git/messages/<agent>_coauthor.msg` trailer snippets for each agent in the multi-agent workflow (see `AGENTS.md`) |
| `hooks/` | Local git hooks (currently `pre-commit`, which shards `assets/tracking/tracking.db` before a large-DB commit) plus `install.sh` to symlink them into `.git/hooks/` |

## Setup

```bash
bash git/hooks/install.sh
```

## Note on Image-Toolkit parity

Image-Toolkit's `git/` directory additionally hosts a live LLM-driven
backlog-sync automation suite (`scripts/agent_tools.py`, `scripts/sync_backlog.py`,
`scripts/check_commit_ref.py`, `config/automation_rules.yaml`, a `post-commit`
hook, and a `git.scripts` Python package with its own `pyproject.toml`) wired
to a GitHub Actions workflow and repository secrets. That automation is
specific to Image-Toolkit's setup and was deliberately **not** ported here —
copying it verbatim would reference workflows, secrets, and a GraphQL client
that don't exist in this repo, i.e. dead/misleading tooling. WSmart-Route's
equivalent coordination mechanism is the `.agent/bus/` + `AGENTS.md`
convention. A genuine local-tooling equivalent (if wanted) should be scoped
and built as its own issue rather than copied.
