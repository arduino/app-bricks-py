# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Check the alignment between app-bricks-examples code and this library.

Pyright analyzes the examples' Python sources resolving the library directly
from a source checkout (no wheel build needed). Three modes:

  deps      Print the library dependencies (core + recursively expanded extra),
            so the check venv can be built without building the library itself.
  run       Run pyright over the examples trees against a library source path
            and save the diagnostics as JSON.
  diff      Compare two run outputs (base vs head of a PR) and report new/fixed
            errors. Always exits 0: the check is informative, not blocking.
  coverage  Report the library bricks that have no examples, highlighting the
            ones introduced by the PR. Informative by design: a new brick may
            legitimately land before its examples do.

Typical PR usage:
  python3 scripts/check_examples_alignment.py run --examples-dir <examples> --library-src base/src --python <venv> --out base.json
  python3 scripts/check_examples_alignment.py run --examples-dir <examples> --library-src head/src --python <venv> --out head.json
  python3 scripts/check_examples_alignment.py diff --base base.json --head head.json

Quick local usage (defaults: examples in ../app-bricks-examples, library in
src, interpreter from the project .venv, no JSON output, details printed):
  task check:examples-alignment:run
  task check:examples-alignment:coverage
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

PYRIGHT_VERSION = "1.1.406"
EXAMPLES_ROOTS = ["bricks", "core-and-foundational", "inspirational"]
DEFAULT_EXAMPLES_DIR = "../app-bricks-examples"
DEFAULT_VENV_PYTHON = ".venv/bin/python"
SELF_EXTRA_RE = re.compile(r"^arduino[-_]app[-_]bricks\[(.+)\]$")


def cmd_deps(args) -> int:
    project = tomllib.loads(Path(args.pyproject).read_text())["project"]
    optional = project.get("optional-dependencies", {})
    deps: list[str] = []
    seen_extras: set[str] = set()

    def expand(entries: list[str]):
        for entry in entries:
            match = SELF_EXTRA_RE.match(entry.replace(" ", ""))
            if match:
                for extra in match.group(1).split(","):
                    if extra not in seen_extras:
                        seen_extras.add(extra)
                        expand(optional.get(extra, []))
            elif entry not in deps:
                deps.append(entry)

    expand(project.get("dependencies", []))
    expand(optional.get(args.extra, []))
    print("\n".join(deps))
    return 0


def cmd_run(args) -> int:
    examples_dir = Path(args.examples_dir).resolve()
    library_src = Path(args.library_src).resolve()
    if not examples_dir.is_dir():
        print(f"examples checkout not found in {examples_dir}: clone app-bricks-examples there or pass --examples-dir", file=sys.stderr)
        return 2
    include = [root for root in EXAMPLES_ROOTS if (examples_dir / root).is_dir()]
    if not include:
        print(f"no examples roots found in {examples_dir}", file=sys.stderr)
        return 2

    # Pyright resolves relative paths from the config file location, so the
    # config is written inside the examples checkout for the duration of the run.
    config_path = examples_dir / "pyrightconfig.json"
    if config_path.exists():
        print(f"{config_path} already exists, refusing to overwrite it", file=sys.stderr)
        return 2
    config = {
        "include": include,
        "executionEnvironments": [{"root": ".", "extraPaths": [str(library_src)]}],
    }

    # Default to the project venv interpreter for quick local runs.
    python = args.python or (DEFAULT_VENV_PYTHON if Path(DEFAULT_VENV_PYTHON).exists() else None)
    if python and not Path(python).exists():
        # Pyright would silently fall back to another environment, skewing the results.
        print(f"python interpreter not found: {python}", file=sys.stderr)
        return 2
    cmd = ["npx", "-y", f"pyright@{args.pyright_version}", "--project", str(examples_dir), "--outputjson"]
    if python:
        cmd += ["--pythonpath", str(Path(python).resolve())]
    try:
        config_path.write_text(json.dumps(config))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        config_path.unlink(missing_ok=True)

    # Pyright exits 0 (clean) or 1 (diagnostics found); anything else is a real failure.
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout + proc.stderr)
        return proc.returncode

    data = json.loads(proc.stdout)
    for diag in data.get("generalDiagnostics", []):
        diag["file"] = Path(diag["file"]).resolve().relative_to(examples_dir).as_posix()
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=2) + "\n")
    summary = data["summary"]
    print(f"{summary['filesAnalyzed']} files analyzed against {library_src}: {summary['errorCount']} errors, {summary['warningCount']} warnings")
    # Without a JSON output the run is a local one-off: print the details.
    if args.details or not args.out:
        errors = [diag for diag in data["generalDiagnostics"] if diag["severity"] == "error"]
        for diag in sorted(errors, key=lambda d: (d.get("rule", ""), d["file"], d["range"]["start"]["line"])):
            line = diag["range"]["start"]["line"] + 1
            print(f"  [{diag.get('rule', '')}] {diag['file']}:{line}  {diag['message'].splitlines()[0]}")
    return 0


def error_index(data: dict) -> tuple[Counter, dict]:
    """Index error diagnostics by a line-shift-tolerant key: (file, rule, message first line)."""
    counts: Counter = Counter()
    samples: dict = {}
    for diag in data.get("generalDiagnostics", []):
        if diag["severity"] != "error":
            continue
        key = (diag["file"], diag.get("rule", ""), diag["message"].splitlines()[0])
        counts[key] += 1
        samples.setdefault(key, diag)
    return counts, samples


def error_table(entries: dict) -> list[str]:
    """Markdown table rows for an error index, with pipes escaped for the cells."""
    rows = ["| Example file | Rule | Message |", "|---|---|---|"]
    for (file, rule, message), count in sorted(entries.items()):
        suffix = f" (×{count})" if count > 1 else ""
        rows.append(f"| `{file}` | {rule} | {message.replace('|', '\\|')}{suffix} |")
    return rows


def cmd_diff(args) -> int:
    base_counts, _ = error_index(json.loads(Path(args.base).read_text()))
    head_counts, head_samples = error_index(json.loads(Path(args.head).read_text()))

    new = {key: count - base_counts.get(key, 0) for key, count in head_counts.items() if count > base_counts.get(key, 0)}
    fixed = {key: count - head_counts.get(key, 0) for key, count in base_counts.items() if count > head_counts.get(key, 0)}

    lines = [
        "## Examples alignment check",
        "",
        f"Errors against the examples ([app-bricks-examples](https://github.com/arduino/app-bricks-examples)@main): "
        f"base {sum(base_counts.values())} → head {sum(head_counts.values())} "
        f"(**{sum(new.values())} new**, {sum(fixed.values())} fixed)",
    ]
    for title, entries in (("New errors", new), ("Fixed errors", fixed)):
        if entries:
            lines += ["", f"### {title}", ""] + error_table(entries)
    if not new and not fixed:
        lines += ["", "No changes in examples alignment."]
    if head_counts:
        # Pre-existing errors are part of the story too, but collapsed: the diff
        # above stays the signal of the PR.
        lines += ["", "<details>", f"<summary>Full report: {sum(head_counts.values())} errors against head</summary>", ""]
        lines += error_table(head_counts)
        lines += ["", "</details>"]
    if args.reports_url:
        lines += ["", f"Raw pyright outputs (base/head JSON): [workflow artifact]({args.reports_url})"]
    report = "\n".join(lines)

    print(report)
    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report + "\n")
    for (file, rule, message), _count in sorted(new.items()):
        line = head_samples[(file, rule, message)]["range"]["start"]["line"] + 1
        print(f"::warning::examples alignment: {file}:{line} [{rule}] {message}")

    # Informative check by design: new errors are reported, never blocking.
    return 0


DISABLED_RE = re.compile(r"^disabled:\s*true\s*$", re.MULTILINE)
BRICK_ID_RE = re.compile(r"^id:\s*(?:[\w-]+:)?([\w-]+)\s*$", re.MULTILINE)


def library_bricks(library_src: Path) -> set[str]:
    """Names of the non-disabled bricks defined in a library source checkout.

    A brick's name is the one declared in its brick_config.yaml `id` (without
    the vendor prefix) — the identifier App Lab and the examples manifest use —
    which for a few bricks differs from the module directory name.
    """
    bricks = set()
    for config in sorted(library_src.glob("arduino/app_bricks/*/brick_config.yaml")):
        content = config.read_text()
        if not DISABLED_RE.search(content):
            match = BRICK_ID_RE.search(content)
            bricks.add(match.group(1) if match else config.parent.name)
    return bricks


def cmd_coverage(args) -> int:
    examples_dir = Path(args.examples_dir).resolve()
    if not (examples_dir / "bricks").is_dir():
        print(f"examples checkout not found in {examples_dir}: clone app-bricks-examples there or pass --examples-dir", file=sys.stderr)
        return 2
    covered = {path.name for path in examples_dir.glob("bricks/*/*") if path.is_dir()}
    head_bricks = library_bricks(Path(args.head_src).resolve())
    base_bricks = library_bricks(Path(args.base_src).resolve()) if args.base_src else head_bricks

    uncovered = sorted(head_bricks - covered)
    introduced = sorted((head_bricks - base_bricks) - covered)

    lines = ["### Bricks without examples", ""]
    if uncovered:
        lines.append(f"{len(uncovered)} bricks have no examples in app-bricks-examples:")
        lines += [f"- `{name}`" + (" — **introduced by this PR**" if name in introduced else "") for name in uncovered]
        lines += ["", "Informative only: a new brick may legitimately land before its examples do."]
    else:
        lines.append("Every non-disabled brick has at least one example.")
    report = "\n".join(lines)

    print(report)
    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report + "\n")
    for name in introduced:
        print(f"::notice::examples coverage: this PR introduces the brick '{name}', which has no examples in app-bricks-examples yet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    deps = sub.add_parser("deps", help="print library dependencies for the check venv")
    deps.add_argument("--pyproject", default="pyproject.toml")
    deps.add_argument("--extra", default="all")
    deps.set_defaults(func=cmd_deps)

    run = sub.add_parser("run", help="run pyright over the examples against a library source")
    run.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    run.add_argument("--library-src", default="src")
    run.add_argument("--python", help=f"python interpreter of the check venv (defaults to {DEFAULT_VENV_PYTHON} when present)")
    run.add_argument("--pyright-version", default=PYRIGHT_VERSION)
    run.add_argument("--out", help="write the diagnostics as JSON; when omitted, details are printed instead")
    run.add_argument("--details", action="store_true", help="also print the error diagnostics, grouped by rule")
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("diff", help="compare two run outputs and report new/fixed errors")
    diff.add_argument("--base", required=True)
    diff.add_argument("--head", required=True)
    diff.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    diff.add_argument("--reports-url", help="link to the uploaded run outputs, appended to the summary")
    diff.set_defaults(func=cmd_diff)

    coverage = sub.add_parser("coverage", help="report library bricks that have no examples")
    coverage.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    coverage.add_argument("--head-src", default="src")
    coverage.add_argument("--base-src", help="library source of the PR base, to flag bricks introduced by the PR")
    coverage.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    coverage.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
