"""
One-shot migration scripts.

Converters and rewriters that moved the codebase, its configs or archived outputs from one
layout to the next. Each runs from the repository root, e.g.
``python logic/migration/convert_sim_outputs.py``.

Attributes:
    apply_wp03_renames.py        — P-45 leftover renames for WP-03
    convert_policy_yaml.py       — P-13 / P-54 converter for policy, packing and HPO YAML
    convert_sim_outputs.py       — Convert archived simulator logs to the WP-08 day-record schema
    migrate_unreleased_params.py — Move the unreleased policies to the config layer (WP-06)
    trim_docstrings.py           — Trim generated docstring boilerplate (P-03)
"""
