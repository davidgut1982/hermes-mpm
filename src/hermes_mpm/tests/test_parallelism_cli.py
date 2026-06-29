"""Tests for the ``hermes mpm parallelism`` CLI subcommand.

Why: ``parallelism`` is the operator-facing read surface over the turn-batch
store — it turns per-turn tool-call counts into the one number that answers "is
parallelism working?" (the batch rate), overall and per model. It must parse its
filters, print the rate cleanly, and handle the empty/no-data case without error.
What: Build the parser via cli.setup, parse ``parallelism`` args, drive
cli.handle against a tmp-pointed DB, and assert exit code + printed content.
Test: ``pytest src/hermes_mpm/tests/test_parallelism_cli.py``.
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


def test_parallelism_action_parses():
    parser = argparse.ArgumentParser()
    cli.setup(parser)
    args = parser.parse_args(["parallelism", "--since", "24h", "--model", "glm-4.6"])
    assert args.mpm_action == "parallelism"
    assert args.since == "24h"
    assert args.model == "glm-4.6"


def test_parallelism_empty(db, capsys):
    rc = _run(["parallelism"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no" in out.lower()  # clean no-data notice


def test_parallelism_prints_rate(db, capsys):
    # 3 tool-turns: counts 3, 1, 2 -> rate 66.7%.
    runs_db.record_turn_batch("t1", "a1", "s1", "glm-4.6", 3, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "glm-4.6", 1, 200)
    runs_db.record_turn_batch("t3", "a3", "s1", "claude", 2, 300)

    rc = _run(["parallelism"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "66.7%" in out  # overall batch rate
    assert "3" in out  # tool-turn count surfaced
    # Per-model breakdown present.
    assert "glm-4.6" in out
    assert "claude" in out


def test_parallelism_model_filter(db, capsys):
    runs_db.record_turn_batch("t1", "a1", "s1", "glm-4.6", 3, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "claude", 2, 200)

    rc = _run(["parallelism", "--model", "glm-4.6"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "glm-4.6" in out
    assert "claude" not in out


def test_parallelism_mixed_null_and_named_model_does_not_crash(db, capsys):
    """The model column is nullable (the hook records model=kw.get('model'), which
    can be None). A store with BOTH a NULL-model turn and a named-model turn must
    NOT crash the per-model breakdown sort (TypeError None < str)."""
    runs_db.record_turn_batch("t1", "a1", "s1", None, 2, 100)
    runs_db.record_turn_batch("t2", "a2", "s1", "glm-4.6", 3, 200)

    rc = _run(["parallelism"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "glm-4.6" in out
    assert "?" in out  # the NULL model surfaces as '?'


def test_parallelism_since_unparseable_errors(db, capsys):
    runs_db.record_turn_batch("t1", "a1", "s1", "m", 2, 100)
    rc = _run(["parallelism", "--since", "abc"])
    cap = capsys.readouterr()
    assert rc != 0
    assert "--since" in cap.err
