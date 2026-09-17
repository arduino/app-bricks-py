# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the diff mode of the examples alignment check."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_examples_alignment as check  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_github_env(monkeypatch, tmp_path):
    """Keep the script away from the real workflow files when the tests run in CI."""
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "github_output"))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def diagnostic(file: str, line: int, message: str, rule: str = "reportArgumentType") -> dict:
    return {
        "file": file,
        "severity": "error",
        "message": message,
        "rule": rule,
        "range": {"start": {"line": line - 1, "character": 0}, "end": {"line": line - 1, "character": 1}},
    }


def write_report(path: Path, *diagnostics: dict) -> Path:
    path.write_text(json.dumps({"generalDiagnostics": list(diagnostics), "summary": {}}))
    return path


def run_diff(tmp_path: Path, base: Path, head: Path, *extra: str) -> tuple[int, str, str]:
    """Run the diff mode; returns (exit code, summary markdown, stdout).

    Stdout is captured explicitly rather than through capsys: the script prints
    workflow commands (::warning::, ::error::) that the Actions runner would turn
    into annotations of the test job if they reached its log.
    """
    summary = tmp_path / "summary.md"
    argv = ["check_examples_alignment.py", "diff", "--base", str(base), "--head", str(head), "--summary", str(summary), *extra]
    saved, sys.argv = sys.argv, argv
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            code = check.main()
    finally:
        sys.argv = saved
    return code, summary.read_text(), out.getvalue()


@pytest.fixture
def reports(tmp_path: Path) -> tuple[Path, Path, Path]:
    clean = write_report(tmp_path / "clean.json")
    one = write_report(tmp_path / "one.json", diagnostic("bricks/a/python/main.py", 5, "boom"))
    two = write_report(tmp_path / "two.json", diagnostic("bricks/a/python/main.py", 5, "boom"), diagnostic("bricks/b/python/main.py", 9, "bang"))
    return clean, one, two


def test_new_errors_are_informative_by_default(tmp_path, reports):
    clean, _one, two = reports
    code, summary, out = run_diff(tmp_path, clean, two)
    assert code == 0
    assert "## ❌ Examples alignment check" in summary
    assert "(**2 new**, 0 fixed)" in summary
    assert out.count("::warning::examples alignment:") == 2
    assert "::error" not in out


def test_fail_on_new_blocks_and_uses_error_annotations(tmp_path, reports):
    clean, _one, two = reports
    code, _summary, out = run_diff(tmp_path, clean, two, "--fail-on-new")
    assert code == 1
    assert out.count("::error::examples alignment:") == 2


def test_fail_on_new_passes_without_new_errors(tmp_path, reports):
    _clean, one, two = reports
    # Fixed-only and identical reports are both fine for a blocking run.
    assert run_diff(tmp_path, two, one, "--fail-on-new")[0] == 0
    assert run_diff(tmp_path, two, two, "--fail-on-new")[0] == 0


def test_annotate_files_places_annotations_inline(tmp_path, reports):
    clean, one, _two = reports
    _code, _summary, out = run_diff(tmp_path, clean, one, "--fail-on-new", "--annotate-files")
    assert "::error file=bricks/a/python/main.py,line=5::examples alignment: bricks/a/python/main.py:5 [reportArgumentType] boom" in out


def test_pre_existing_errors_are_flagged_when_none_is_new(tmp_path, reports):
    _clean, _one, two = reports
    _code, summary, _out = run_diff(tmp_path, two, two)
    assert "## ✅ Examples alignment check" in summary
    assert "✅ No new errors in this PR." in summary
    assert "⚠️ 2 pre-existing errors" in summary


def test_new_errors_count_is_exposed_to_the_workflow(tmp_path, reports):
    clean, _one, two = reports
    run_diff(tmp_path, clean, two)
    assert "new_errors=2" in (tmp_path / "github_output").read_text()
