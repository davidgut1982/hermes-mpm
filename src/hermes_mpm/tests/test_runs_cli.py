"""Tests for the ``hermes mpm runs`` CLI subcommand.

Why: ``runs`` is the operator-facing read surface for the run DB; it must parse
its filters, format a compact table, and handle the empty case without error.
What: Build the parser via cli.setup, parse ``runs`` args, drive cli.handle
against a tmp-pointed DB, and assert exit code + printed content.
Test: ``pytest src/hermes_mpm/tests/test_runs_cli.py``.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_mpm import cli, runs_db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "mpm_runs.db"
    monkeypatch.setattr(runs_db, "_db_path", lambda: path)
    runs_db.init_db()
    return path


def _run(argv):
    parser = argparse.ArgumentParser()
    cli.setup(parser)
    return cli.handle(parser.parse_args(argv))


def test_runs_action_parses():
    parser = argparse.ArgumentParser()
    cli.setup(parser)
    args = parser.parse_args(["runs", "--status", "running", "--limit", "5"])
    assert args.mpm_action == "runs"
    assert args.status == "running"
    assert args.limit == 5


def test_runs_empty(db, capsys):
    rc = _run(["runs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no runs" in out.lower()


def test_runs_formats_rows(db, capsys):
    runs_db.record_start(
        "child-session-aaaa", "p1", "engineer", "engineer",
        "implement the parser", 1000, "subagent",
    )
    runs_db.record_end("child-session-aaaa", status="done", ended_at=1005, duration_ms=5000)
    runs_db.record_start(
        "child-session-bbbb", "p1", "search", "search", "find news", 2000, "subagent",
    )

    rc = _run(["runs"])
    out = capsys.readouterr().out
    assert rc == 0
    # Short run id (not the full session id) — first 8 chars present.
    assert "child-se" in out
    assert "done" in out
    assert "running" in out
    assert "engineer" in out
    # Goal shown (possibly truncated).
    assert "implement" in out


def test_runs_status_filter(db, capsys):
    runs_db.record_start("a", "p", "r", None, "ga", 1, "subagent")
    runs_db.record_end("a", status="done", ended_at=2)
    runs_db.record_start("b", "p", "r", None, "gb", 3, "subagent")

    rc = _run(["runs", "--status", "running"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gb" in out
    assert "ga" not in out


def test_runs_since_filter_parses_duration(db, capsys):
    # since accepts 1h/24h/7d shorthand; just assert it parses + runs cleanly.
    rc = _run(["runs", "--since", "24h"])
    assert rc == 0
    rc = _run(["runs", "--since", "7d"])
    assert rc == 0
