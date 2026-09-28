"""Regression tests for B-mistral-03 / B-mistral-04 (2026-09-28 cleanup round).

Every field declared on ``TrackingConfig`` and ``SAConfig`` must be
referenced by at least one non-config source file: a config field
that nothing reads is an inert knob (the tracking trio and the SA
``iterations_per_temp`` were exactly that). The test scans file contents
only — it imports nothing heavy.

Before the fix this failed on ``wst_tracking_uri``, ``real_time_log``,
``profiler_buffer_size`` and ``iterations_per_temp``.
"""

import dataclasses
import re
from pathlib import Path
from typing import List

from logic.src.configs.policies.sa import SAConfig
from logic.src.configs.tracking import TrackingConfig

_ROOT = Path(__file__).resolve().parents[4]


def _search_space() -> List[Path]:
    """Reader space: non-config source files. Yaml files only SET keys, so
    they must not count as readers."""
    files = []
    for f in (_ROOT / "logic/src").rglob("*.py"):
        if "__pycache__" not in f.parts and "configs" not in f.parts:
            files.append(f)
    return files


_SEARCH_SPACE = None


def _is_read(field_name: str, defining_module: str) -> bool:
    global _SEARCH_SPACE
    if _SEARCH_SPACE is None:
        _SEARCH_SPACE = _search_space()
    pat = re.compile(r"[.\[\"'\s:]" + re.escape(field_name) + r"[\"'\]\s:=,.()]")
    for f in _SEARCH_SPACE:
        if f.name == defining_module:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pat.search(text):
            return True
    return False


def _unread_fields(config_cls) -> List[str]:
    defining = {TrackingConfig: "tracking.py", SAConfig: "sa.py"}[config_cls]
    return [
        f.name
        for f in dataclasses.fields(config_cls)
        if not _is_read(f.name, defining)
    ]


def test_tracking_config_fields_are_all_read():
    unread = _unread_fields(TrackingConfig)
    assert not unread, (
        "TrackingConfig declares fields that no code or yaml reads: "
        f"{unread}; delete the field (and its yaml key), or wire it."
    )


def test_sa_config_fields_are_all_read():
    unread = _unread_fields(SAConfig)
    assert not unread, (
        "SAConfig declares fields that no code or yaml reads: "
        f"{unread}; delete the field (and its yaml key), or wire it."
    )
