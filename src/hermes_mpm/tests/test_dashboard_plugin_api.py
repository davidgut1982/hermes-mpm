"""Tests for the MPM Runs dashboard plugin API (``dashboard/plugin_api.py``).

Why: The dashboard plugin is one of several processes that load ``runs_db``, and
a maintenance write from a non-gateway process is exactly what corrupted run
state historically (see ``runs_db.sweep_orphaned``). These tests pin two things:
(1) the read routes return ``query_runs`` / aggregate results with filters, and
(2) importing the plugin and hitting its routes triggers NO sweep / init / write
— the read-only contract the panel depends on.
What: Loads ``plugin_api.py`` by file path (the host mounts it the same way),
mounts its router on a bare FastAPI app, and drives it via ``TestClient`` against
an isolated tmp DB. Forbidden maintenance functions are spied to assert they are
never called.
Test: ``pytest src/hermes_mpm/tests/test_dashboard_plugin_api.py`` — all pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hermes_mpm import runs_db

# --- locate + load the plugin module by file path (host-style) -------------

_PLUGIN_API = (
    Path(runs_db.__file__).resolve().parent / "dashboard" / "plugin_api.py"
)


def _load_plugin_api():
    """Import ``dashboard/plugin_api.py`` as a standalone module by file path.

    Why: ``dashboard/`` is not a Python package (the host loads the file via an
    importlib spec, not a normal import); tests must load it the same way so the
    test exercises the real load path.
    What: Builds an importlib spec from the file and executes it.
    Test: Implicit — every test below depends on this returning the module.
    """
    spec = importlib.util.spec_from_file_location(
        "hermes_mpm_dashboard_plugin_api", _PLUGIN_API
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient over a bare FastAPI app mounting the plugin router.

    Why: Exercises the routes exactly as the dashboard would, against an
    isolated tmp DB so there is no cross-test bleed or real-home dependency.
    What: Points ``runs_db._db_path`` at ``tmp_path/mpm_runs.db``, seeds the
    schema via ``init_db`` (the GATEWAY's job in prod — done here only to set up
    the fixture), loads the plugin, mounts its router, yields a TestClient.
    Test: Implicit — used by every route test.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    db_path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: db_path)
    runs_db.init_db()  # fixture setup only — emulates the gateway having run

    module = _load_plugin_api()
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/mpm-runs")
    return TestClient(app)


def _seed(run_id, status, started_at, **kw):
    """Insert one run via the real write path, then set a terminal status.

    Why: Tests assert the read routes reflect real rows; seeding through
    ``record_start`` / ``record_end`` keeps the fixture honest about the schema.
    What: record_start (running) then record_end to the requested status.
    Test: Used by the query/stats tests below.
    """
    runs_db.record_start(
        run_id=run_id,
        parent_session_id=kw.get("session"),
        role=kw.get("role"),
        profile=kw.get("profile"),
        goal=kw.get("goal"),
        started_at=started_at,
        run_type="subagent",
    )
    if status != runs_db.STATUS_RUNNING:
        runs_db.record_end(
            run_id=run_id,
            status=status,
            ended_at=started_at + 5,
            duration_ms=5000,
        )


# --- /runs -----------------------------------------------------------------


def test_get_runs_returns_query_runs(client):
    _seed("r-done", runs_db.STATUS_DONE, 1000, profile="engineer", goal="ship it")
    _seed("r-run", runs_db.STATUS_RUNNING, 2000, profile="qa", goal="test it")

    resp = client.get("/api/plugins/mpm-runs/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert "now" in body and isinstance(body["now"], int)
    ids = [r["run_id"] for r in body["runs"]]
    # newest-first ordering from query_runs
    assert ids == ["r-run", "r-done"]


def test_get_runs_passes_status_filter(client):
    _seed("r-done", runs_db.STATUS_DONE, 1000)
    _seed("r-run", runs_db.STATUS_RUNNING, 2000)

    resp = client.get("/api/plugins/mpm-runs/runs", params={"status": "running"})
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["run_id"] for r in runs] == ["r-run"]


def test_get_runs_passes_session_and_since_filters(client):
    _seed("r-old", runs_db.STATUS_DONE, 1000, session="sess-A")
    _seed("r-new", runs_db.STATUS_DONE, 5000, session="sess-A")
    _seed("r-other", runs_db.STATUS_DONE, 6000, session="sess-B")

    resp = client.get(
        "/api/plugins/mpm-runs/runs",
        params={"session": "sess-A", "since": 4000},
    )
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["run_id"] for r in runs] == ["r-new"]


def test_get_runs_respects_limit(client):
    for i in range(5):
        _seed(f"r-{i}", runs_db.STATUS_DONE, 1000 + i)

    resp = client.get("/api/plugins/mpm-runs/runs", params={"limit": 2})
    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 2


# --- /runs/stats -----------------------------------------------------------


def test_get_runs_stats_aggregates(client):
    _seed("a", runs_db.STATUS_DONE, 1000)
    _seed("b", runs_db.STATUS_DONE, 1001)
    _seed("c", runs_db.STATUS_RUNNING, 1002)
    _seed("d", runs_db.STATUS_FAILED, 1003)

    resp = client.get("/api/plugins/mpm-runs/runs/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["done"] == 2
    assert body["stats"]["running"] == 1
    assert body["stats"]["failed"] == 1
    assert body["stats"]["crashed"] == 0  # zero-filled known status
    assert body["total"] == 4


# --- read-only contract: NO sweep / init / write ---------------------------


def test_import_does_not_trigger_maintenance(monkeypatch):
    """Importing the plugin module must not sweep/init/write the DB.

    Why: The dashboard loads this module at startup; any maintenance call here
    would reap the live gateway's in-flight runs (the historical corruption).
    What: Spies sweep_orphaned/init_db/record_start/record_end/purge_old, then
    loads the module, and asserts none were called.
    Test: This test.
    """
    forbidden = ["sweep_orphaned", "init_db", "record_start", "record_end", "purge_old"]
    calls = {name: 0 for name in forbidden}

    def _make_spy(name):
        def _spy(*a, **k):
            calls[name] += 1
            return None

        return _spy

    for name in forbidden:
        monkeypatch.setattr(runs_db, name, _make_spy(name))

    _load_plugin_api()  # the act under test

    assert calls == {name: 0 for name in forbidden}


def test_routes_do_not_trigger_sweep_or_write(client, monkeypatch):
    """Hitting /runs and /runs/stats must not sweep/init/write the DB.

    Why: The panel polls these routes every 5s from a non-gateway process; a
    write/sweep on a read would corrupt run state under load.
    What: Spies the maintenance functions, calls both routes, asserts zero
    maintenance calls while the routes still return 200.
    Test: This test.
    """
    forbidden = ["sweep_orphaned", "init_db", "record_start", "record_end", "purge_old"]
    calls = {name: 0 for name in forbidden}

    def _make_spy(name):
        def _spy(*a, **k):
            calls[name] += 1
            return None

        return _spy

    for name in forbidden:
        monkeypatch.setattr(runs_db, name, _make_spy(name))

    assert client.get("/api/plugins/mpm-runs/runs").status_code == 200
    assert client.get("/api/plugins/mpm-runs/runs/stats").status_code == 200

    assert calls == {name: 0 for name in forbidden}


def test_routes_degrade_when_db_missing(tmp_path, monkeypatch):
    """With no DB file, the routes return empty AND do not create the file.

    Why: The dashboard may start before the gateway has ever created the DB. The
    panel must show an empty table, not a 500 — and crucially must NOT create
    ``mpm_runs.db`` (or its -wal/-shm sidecars) from a non-gateway process, which
    could leave the file owned by the wrong uid and break the gateway's later
    tracking writes. This locks the documented no-create read-only contract.
    What: Points _db_path at a non-existent file under a non-existent dir; the
    stats route returns zero-filled statuses and /runs an empty list, and the DB
    file (and sidecars) are asserted to still NOT exist afterwards.
    Test: This test.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    missing = tmp_path / "nope" / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: missing)

    module = _load_plugin_api()
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/mpm-runs")
    tc = TestClient(app)

    runs_resp = tc.get("/api/plugins/mpm-runs/runs")
    assert runs_resp.status_code == 200
    assert runs_resp.json()["runs"] == []

    stats_resp = tc.get("/api/plugins/mpm-runs/runs/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total"] == 0

    # The read-only no-create invariant: the dashboard process must never bring
    # the DB (or its WAL sidecars) into existence.
    assert not missing.exists()
    assert not (missing.parent / "mpm_runs.db-wal").exists()
    assert not (missing.parent / "mpm_runs.db-shm").exists()
