# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the pyright check script: rules loading and diff mode."""

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

from scripts import check_pyright as check  # noqa: E402


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
    assert out.count("::warning::Examples alignment check:") == 2
    assert "::error" not in out


def test_fail_on_new_blocks_and_uses_error_annotations(tmp_path, reports):
    clean, _one, two = reports
    code, _summary, out = run_diff(tmp_path, clean, two, "--fail-on-new")
    assert code == 1
    assert out.count("::error::Examples alignment check:") == 2


def test_fail_on_new_passes_without_new_errors(tmp_path, reports):
    _clean, one, two = reports
    # Fixed-only and identical reports are both fine for a blocking run.
    assert run_diff(tmp_path, two, one, "--fail-on-new")[0] == 0
    assert run_diff(tmp_path, two, two, "--fail-on-new")[0] == 0


def test_annotate_files_places_annotations_inline(tmp_path, reports):
    clean, one, _two = reports
    _code, _summary, out = run_diff(tmp_path, clean, one, "--fail-on-new", "--annotate-files")
    assert "::error file=bricks/a/python/main.py,line=5::Examples alignment check: bricks/a/python/main.py:5 [reportArgumentType] boom" in out


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


# --- rules file ---------------------------------------------------------------


def write_rules(path: Path, **overrides) -> Path:
    rules = {
        "schemaVersion": 1,
        "pythonVersion": "3.13",
        "pyrightVersion": "1.1.411",
        "useLibraryCodeForTypes": True,
        "profiles": {
            "app-bricks-py": {"typeCheckingMode": "strict", "rules": {"reportMissingTypeStubs": "none"}},
            "api-user": {"typeCheckingMode": "standard", "rules": {}},
        },
    }
    rules.update(overrides)
    path.write_text(json.dumps(rules))
    return path


def test_profile_config_merges_engine_settings_mode_and_rules(tmp_path):
    rules = check.load_rules(write_rules(tmp_path / "pyright-rules.json"))
    library = check.profile_config(rules, "app-bricks-py")
    assert library == {"typeCheckingMode": "strict", "reportMissingTypeStubs": "none", "pythonVersion": "3.13", "useLibraryCodeForTypes": True}
    api = check.profile_config(rules, "api-user")
    assert api["typeCheckingMode"] == "standard"
    with pytest.raises(ValueError, match="unknown profile"):
        check.profile_config(rules, "nope")


def test_load_rules_rejects_what_app_lab_would_reject(tmp_path):
    for name, overrides in {
        "schema": {"schemaVersion": 2},
        "mode": {"profiles": {"api-user": {"typeCheckingMode": "relaxed", "rules": {}}}},
        "rule name": {"profiles": {"api-user": {"typeCheckingMode": "standard", "rules": {"extraPaths": "/x"}}}},
        "severity": {"profiles": {"api-user": {"typeCheckingMode": "standard", "rules": {"reportUnusedImport": "loud"}}}},
        "no profiles": {"profiles": {}},
    }.items():
        with pytest.raises(ValueError):
            check.load_rules(write_rules(tmp_path / f"{name}.json", **overrides))


def test_find_rules_file_prefers_the_repository_root_then_the_static_assets(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src"
    (src / "arduino" / "app_bricks" / "static").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        check.find_rules_file(src, None)
    static = write_rules(src / "arduino" / "app_bricks" / "static" / "pyright-rules.json")
    assert check.find_rules_file(src, None) == static
    root = write_rules(repo / "pyright-rules.json")
    assert check.find_rules_file(src, None) == root
    explicit = write_rules(tmp_path / "elsewhere.json")
    assert check.find_rules_file(src, str(explicit)) == explicit.resolve()


def test_the_shipped_rules_file_loads_with_both_profiles():
    rules = check.load_rules(REPO_ROOT / "pyright-rules.json")
    assert set(rules["profiles"]) == {"app-bricks-py", "api-user"}
    for profile in rules["profiles"]:
        assert check.profile_config(rules, profile)["typeCheckingMode"] == "standard"


# --- report ---------------------------------------------------------------------


def run_mode(tmp_path: Path, *argv: str) -> tuple[int, str]:
    saved, sys.argv = sys.argv, ["check_pyright.py", *argv]
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            code = check.main()
    finally:
        sys.argv = saved
    return code, out.getvalue()


def make_sections(tmp_path: Path, reports) -> tuple[Path, Path]:
    clean, _one, two = reports
    api = tmp_path / "api.json"
    typing = tmp_path / "typing.json"
    run_mode(
        tmp_path,
        "diff",
        "--base",
        str(clean),
        "--head",
        str(two),
        "--title",
        "API contract",
        "--result",
        str(api),
        "--summary",
        str(tmp_path / "a.md"),
    )
    run_mode(
        tmp_path,
        "diff",
        "--base",
        str(two),
        "--head",
        str(two),
        "--title",
        "Library typing",
        "--subject",
        "Errors in the library",
        "--result",
        str(typing),
        "--summary",
        str(tmp_path / "t.md"),
    )
    return api, typing


def test_report_composes_a_verdict_table_and_one_collapsed_details_block(tmp_path, reports):
    api, typing = make_sections(tmp_path, reports)
    summary = tmp_path / "report.md"
    code, _out = run_mode(tmp_path, "report", str(api), str(typing), "--summary", str(summary))
    text = summary.read_text()
    assert code == 0  # informative sections never fail the report
    assert text.startswith("## ❌ Pyright checks and examples coverage")
    assert "| ❌ API contract | **2 new**, 0 fixed, 0 pre-existing |" in text
    assert "| ✅ Library typing | **0 new**, 0 fixed, 2 pre-existing |" in text
    assert "⚠️ **Library typing**: 2 pre-existing errors" in text
    assert "❌ **API contract**: This PR introduces errors" in text
    assert text.count("<details>") == 1 and text.count("</details>") == 1
    assert "### API contract" in text and "### Library typing" in text
    assert "Errors in the library: base 2 → head 2" in text
    assert "#### Full report: 2 errors against head" in text


def test_report_links_the_shared_outputs_once(tmp_path, reports):
    api, typing = make_sections(tmp_path, reports)
    summary = tmp_path / "report.md"
    run_mode(tmp_path, "report", str(api), str(typing), "--summary", str(summary), "--reports-url", "https://example/artifact")
    text = summary.read_text()
    assert text.count("https://example/artifact") == 1
    # At the bottom, outside the collapsed details, always in sight.
    assert text.index("Download the pyright JSON outputs") > text.rindex("</details>")


def test_report_fails_only_for_blocking_sections(tmp_path, reports):
    clean, _one, two = reports
    blocking = tmp_path / "blocking.json"
    run_mode(
        tmp_path, "diff", "--base", str(clean), "--head", str(two), "--fail-on-new", "--result", str(blocking), "--summary", str(tmp_path / "b.md")
    )
    informative = tmp_path / "informative.json"
    run_mode(tmp_path, "diff", "--base", str(clean), "--head", str(two), "--result", str(informative), "--summary", str(tmp_path / "i.md"))

    assert run_mode(tmp_path, "report", str(informative), "--fail-on-new", "--summary", str(tmp_path / "r1.md"))[0] == 0
    assert run_mode(tmp_path, "report", str(blocking), "--fail-on-new", "--summary", str(tmp_path / "r2.md"))[0] == 1
    assert run_mode(tmp_path, "report", str(blocking), "--summary", str(tmp_path / "r3.md"))[0] == 0
    assert "status=failed" in (tmp_path / "github_output").read_text()


def test_report_heading_carries_the_worst_status(tmp_path, reports):
    clean, _one, two = reports
    passed = tmp_path / "passed.json"
    run_mode(tmp_path, "diff", "--base", str(two), "--head", str(two), "--result", str(passed), "--summary", str(tmp_path / "p.md"))
    failed = tmp_path / "failed.json"
    run_mode(tmp_path, "diff", "--base", str(clean), "--head", str(two), "--result", str(failed), "--summary", str(tmp_path / "f.md"))
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps({
            "title": "Examples coverage",
            "status": "warning",
            "informative": True,
            "result": "⚠️ 1 brick without examples.",
            "cell": "1 brick without examples",
            "notes": ["⚠️ 1 brick without examples: `x` (introduced by this PR)."],
            "details": "- `x`",
        })
    )

    # Coverage alone turns the heading into a warning, never into a failure.
    run_mode(tmp_path, "report", str(passed), str(coverage), "--summary", str(tmp_path / "r1.md"))
    text = (tmp_path / "r1.md").read_text()
    assert text.startswith("## ⚠️ Pyright checks and examples coverage")
    assert "| ⚠️ Examples coverage | 1 brick without examples |" in text
    assert "⚠️ **Examples coverage**: 1 brick without examples: `x` (introduced by this PR)." in text
    # New errors in a pyright section win over the coverage warning.
    run_mode(tmp_path, "report", str(failed), str(coverage), "--summary", str(tmp_path / "r2.md"))
    assert (tmp_path / "r2.md").read_text().startswith("## ❌ Pyright checks and examples coverage")
    # Everything clean: a plain pass, with the verdict table folded away too.
    run_mode(tmp_path, "report", str(passed), "--summary", str(tmp_path / "r3.md"))
    text = (tmp_path / "r3.md").read_text()
    assert text.startswith("## ✅ Pyright checks and examples coverage")
    assert "<summary>All 1 checks passed</summary>" in text
    assert text.index("| Check | Result |") > text.index("<summary>All 1 checks passed</summary>")
    assert text.count("<details>") == 2
    # A warning or a failure keeps the table in plain sight.
    assert "<summary>All" not in (tmp_path / "r1.md").read_text()
    assert (tmp_path / "r1.md").read_text().count("<details>") == 1
