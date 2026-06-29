"""Tests for the AXI CLI cheatsheet injected by the axi-cheatsheet hook.

Why: The hermes agent flailed using ``gh-axi`` — it guessed flag syntax and
permuted retries (``--repo`` vs ``-R``, flag-before vs after the subcommand) and
used a non-existent ``--comment`` flag on ``pr close``, instead of reading
``gh-axi --help``. The cheatsheet now carries ground-truth gh-axi syntax plus a
general "orient before acting" tool-usage rule. This module pins that content so
it cannot silently regress, and keeps the hook's platform-gating behavior green.
What: Asserts on ``_AXI_CHEATSHEET`` content and drives
``_make_axi_cheatsheet_hook()`` across platforms.
Test: ``pytest src/hermes_mpm/tests/test_axi_cheatsheet.py``.
"""

from __future__ import annotations

import pytest

import hermes_mpm


def test_cheatsheet_has_gh_axi_repo_flag_placement():
    """Why: The agent permuted ``-R``/``--repo`` placement instead of reading help.
    What: The sheet states the flag goes AFTER the command and shows ``-R``.
    Test: this test.
    """
    sheet = hermes_mpm._AXI_CHEATSHEET
    assert "-R" in sheet
    assert "AFTER the command" in sheet
    assert "gh-axi pr view 42 -R owner/name" in sheet


def test_cheatsheet_documents_pr_subcommands_and_comment_then_close():
    """Why: The agent used a non-existent ``--comment`` flag on ``pr close``.
    What: The sheet lists pr subcommands and the comment-then-close pattern, and
    explicitly notes pr close has NO --comment flag.
    Test: this test.
    """
    sheet = hermes_mpm._AXI_CHEATSHEET
    assert "pr comment" in sheet
    assert "NO --comment flag" in sheet
    # Distinctive subcommands from the pr subcommand list.
    for sub in ("close", "merge", "review", "reopen", "ready"):
        assert sub in sheet


def test_cheatsheet_has_403_non_retryable_note():
    """Why: A 403/permission error is non-retryable; the agent must stop, not retry.
    What: The sheet flags 403 / wrong-permissions as non-retryable and scopes
    write access (davidgut1982/* owned, NousResearch/* read-only).
    Test: this test.
    """
    sheet = hermes_mpm._AXI_CHEATSHEET
    assert "403" in sheet
    assert "non-retryable" in sheet
    assert "NousResearch" in sheet


def test_cheatsheet_has_orient_before_acting_rule():
    """Why: General fix for flag-guessing — read help before constructing commands.
    What: The sheet carries the orient-before-acting rule and the no-permuted-retry
    instruction.
    Test: this test.
    """
    sheet = hermes_mpm._AXI_CHEATSHEET
    assert "ORIENT BEFORE ACTING" in sheet
    assert "--help" in sheet
    assert "permuted" in sheet


@pytest.mark.parametrize("platform", ["cli", "interactive", "main", ""])
def test_hook_injects_cheatsheet_on_parent_turns(platform):
    """Why: Parent/PM turns must receive the full sheet (existing behavior).
    What: hook(platform=<parent>) returns a dict whose context is the sheet.
    Test: this test.
    """
    hook = hermes_mpm._make_axi_cheatsheet_hook()
    out = hook(platform=platform)
    assert out is not None
    assert out["context"] == hermes_mpm._AXI_CHEATSHEET
    assert "tavily-axi" in out["context"]


@pytest.mark.parametrize("platform", ["subagent", "leaf", "SubAgent", "LEAF"])
def test_hook_suppressed_on_subagent_turns(platform):
    """Why: Children get a scoped AXI hint via task context, not the full sheet.
    What: hook(platform=<leaf/subagent>) returns None (existing behavior).
    Test: this test.
    """
    hook = hermes_mpm._make_axi_cheatsheet_hook()
    assert hook(platform=platform) is None
