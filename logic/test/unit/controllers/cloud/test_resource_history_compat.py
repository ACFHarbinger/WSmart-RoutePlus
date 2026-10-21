"""Resource status survives arbitrary mixed metric interleaving."""

from unittest.mock import Mock

import pytest
from logic.controllers.cloud.control.documents import DocumentControlPlane
from logic.controllers.cloud.local.control_plane import SQLiteControlPlane
from logic.controllers.cloud.service_status import ServiceStatusMixin


@pytest.mark.parametrize("backend", ["sqlite", "document"])
def test_resource_before_latest_page(backend):
    if backend == "sqlite":
        cp = SQLiteControlPlane(":memory:")
    else:
        adapter = Mock()
        records = []
        adapter.append.side_effect = lambda collection, record: records.append(record)
        adapter.list_appended.side_effect = lambda *args: records[:]
        cp = DocumentControlPlane(adapter)
    cp.put_metrics(
        "task",
        [{"kind": "resource", "cpu_percent": 1, "at": 1}, {"kind": "resource", "cpu_percent": 12, "at": 2}]
        + [{"kind": "progress", "at": i + 3} for i in range(25)],
    )
    view = ServiceStatusMixin()
    view.cp = cp
    assert view._latest_resource("task")["cpu_percent"] == 12


@pytest.mark.parametrize("backend", ["sqlite", "document"])
def test_resource_absent(backend):
    adapter = Mock()
    adapter.list_appended.return_value = []
    cp = SQLiteControlPlane(":memory:") if backend == "sqlite" else DocumentControlPlane(adapter)
    view = ServiceStatusMixin()
    view.cp = cp
    assert view._latest_resource("missing") == {}
