# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pyright checks shared by app-bricks-py, app-bricks-examples and App Lab.

The rules come from pyright-rules.json (repository root, shipped in the wheel as
arduino/app_bricks/static/pyright-rules.json): the library decides how code is
type-checked through two profiles, app-bricks-py for its own sources and api-user
for code written against its API; this script adds the environment (paths,
interpreter, execution root). Modes:

  deps      Print a project's dependencies (core + recursively expanded extra),
            so the check venv can be built without building the library itself.
  run       Run pyright over the examples trees against a library source path
            (profile api-user) and save the diagnostics as JSON.
  typing    Run pyright over the library sources themselves (profile
            app-bricks-py) and save the diagnostics as JSON.
  diff      Compare two outputs of run or typing (base vs head of a PR) and
            report new/fixed errors. Exits 0 by default: informative, the
            workflow surfaces the report on the PR (summary, comment, label).
            With --fail-on-new it exits 1 on new errors: the examples repository
            runs it that way on its own PRs, where the analyzed files are the
            PR's and blocking is the point.
  coverage  Report the library bricks that have no examples, highlighting the
            ones introduced by the PR. Informative by design: a new brick may
            legitimately land before its examples do.

Typical PR usage:
  python3 scripts/check_pyright.py run --examples-dir <examples> --library-src base/src --python <venv> --out base.json
  python3 scripts/check_pyright.py run --examples-dir <examples> --library-src head/src --python <venv> --out head.json
  python3 scripts/check_pyright.py diff --base base.json --head head.json

Quick local usage (defaults: examples in ../app-bricks-examples, library in
src, interpreter from the project .venv, no JSON output, details printed):
  task check:api      (run + coverage)
  task check:typing
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

# Fallback only: the version normally comes from pyright-rules.json, where it is
# kept equal to the pyright release basedpyright in App Lab is built on, so every
# consumer applies the same rule set.
PYRIGHT_VERSION = "1.1.411"
RULES_FILE = "pyright-rules.json"
RULES_STATIC_PATH = "arduino/app_bricks/static/" + RULES_FILE
# Rows of pre-existing errors shown in a section's full report: a large typing
# debt would otherwise swamp the job summary and hit the PR comment size cap;
# the complete list stays in the uploaded pyright outputs.
FULL_REPORT_MAX_ROWS = 150
PROFILE_LIBRARY = "app-bricks-py"
PROFILE_API_USER = "api-user"
LIBRARY_PACKAGE = "arduino"
EXAMPLES_ROOTS = ["bricks", "core-and-foundational", "inspirational"]
DEFAULT_EXAMPLES_DIR = "../app-bricks-examples"
EXAMPLES_REPO_MD = "[app-bricks-examples](https://github.com/arduino/app-bricks-examples)@main"
DEFAULT_VENV_PYTHON = ".venv/bin/python"
# Library dependencies that share the `arduino` namespace with the library
# itself: when missing from the check interpreter pyright still resolves the
# namespace from the library sources, silently degrading the missing modules
# to Unknown and hiding real errors instead of reporting an unresolved import.
NAMESPACE_DEPENDENCIES = ["arduino.router_bridge"]
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


def find_rules_file(library_src: Path, explicit: str | None) -> Path:
    """Locate pyright-rules.json: explicit path, repository root of a source checkout,
    or the static assets of a built tree (the layout App Lab reads from the wheel)."""
    if explicit:
        return Path(explicit).resolve()
    for candidate in (library_src.parent / RULES_FILE, library_src / RULES_STATIC_PATH):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{RULES_FILE} not found next to {library_src} (repository root or {RULES_STATIC_PATH})")


def load_rules(path: Path) -> dict:
    """Read and validate pyright-rules.json with the same constraints App Lab applies."""
    rules = json.loads(path.read_text())
    if rules.get("schemaVersion") != 1:
        raise ValueError(f"{path}: unsupported schemaVersion {rules.get('schemaVersion')!r}, want 1")
    profiles = rules.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{path}: no profiles defined")
    for name, profile in profiles.items():
        if profile.get("typeCheckingMode") not in ("off", "basic", "standard", "strict"):
            raise ValueError(f"{path}: profile {name} has an unknown typeCheckingMode {profile.get('typeCheckingMode')!r}")
        for rule, value in profile.get("rules", {}).items():
            if not rule.startswith("report") or not (isinstance(value, bool) or value in ("none", "information", "warning", "error")):
                raise ValueError(f"{path}: profile {name}: {rule}={value!r} is not a report* rule with a valid severity")
    return rules


def profile_config(rules: dict, profile: str) -> dict:
    """The pyright settings a profile contributes to a config: the library decides
    how code is checked, the caller adds the environment (paths, interpreter)."""
    if profile not in rules["profiles"]:
        raise ValueError(f"unknown profile {profile!r}, available: {', '.join(rules['profiles'])}")
    entry = rules["profiles"][profile]
    config = {"typeCheckingMode": entry["typeCheckingMode"], **entry.get("rules", {})}
    if "pythonVersion" in rules:
        config["pythonVersion"] = rules["pythonVersion"]
    if "useLibraryCodeForTypes" in rules:
        config["useLibraryCodeForTypes"] = rules["useLibraryCodeForTypes"]
    return config


def resolve_interpreter(explicit: str | None) -> str | None:
    """The interpreter pyright derives its search paths from, checked for the
    library dependencies. Returns None to let pyright use its default environment."""
    # Default to the project venv interpreter for quick local runs.
    python = explicit or (DEFAULT_VENV_PYTHON if Path(DEFAULT_VENV_PYTHON).exists() else None)
    if not python:
        return None
    if not Path(python).exists():
        # Pyright would silently fall back to another environment, skewing the results.
        raise FileNotFoundError(f"python interpreter not found: {python}")
    # Absolute, but NOT resolved: <venv>/bin/python is a symlink to the base
    # interpreter, and pyright derives the search paths from the interpreter it
    # is handed. Resolving the link pointed it at the bare base install, whose
    # site-packages has none of the dependencies, and every third-party import
    # of the examples came back unresolved while the venv sat unused.
    python = os.path.abspath(python)
    preflight = subprocess.run([python, "-c", "import " + ", ".join(NAMESPACE_DEPENDENCIES)], capture_output=True, text=True)
    if preflight.returncode != 0:
        raise RuntimeError(
            f"the check interpreter {python} cannot import {', '.join(NAMESPACE_DEPENDENCIES)}: "
            "the library dependencies are out of date in that environment and the analysis would silently miss errors. "
            'Update it with `pip install -e ".[dev]"` (or `task init`) and retry.'
        )
    return python


def run_pyright(project_dir: Path, config: dict, python: str | None, pyright_version: str, relative_to: Path) -> dict:
    """Run pyright over project_dir with the given config and return its JSON output,
    diagnostics paths made relative to relative_to.

    Pyright resolves relative paths from the config file location, so the config
    is written inside the analyzed tree for the duration of the run.
    """
    config_path = project_dir / "pyrightconfig.json"
    if config_path.exists():
        raise FileExistsError(f"{config_path} already exists, refusing to overwrite it")
    cmd = ["npx", "-y", f"pyright@{pyright_version}", "--project", str(project_dir), "--outputjson"]
    if python:
        cmd += ["--pythonpath", python]
    try:
        config_path.write_text(json.dumps(config))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        config_path.unlink(missing_ok=True)
    # Pyright exits 0 (clean) or 1 (diagnostics found); anything else is a real failure.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"pyright failed with exit code {proc.returncode}:\n{proc.stdout}{proc.stderr}")
    data = json.loads(proc.stdout)
    for diag in data.get("generalDiagnostics", []):
        path = Path(diag["file"]).resolve()
        try:
            diag["file"] = path.relative_to(relative_to).as_posix()
        except ValueError:
            diag["file"] = path.as_posix()
    return data


def report_run(data: dict, subject: str, out: str | None, details: bool) -> None:
    """Print the run outcome and, for a local one-off, the diagnostics themselves."""
    if out:
        Path(out).write_text(json.dumps(data, indent=2) + "\n")
    summary = data["summary"]
    print(f"{summary['filesAnalyzed']} files analyzed {subject}: {summary['errorCount']} errors, {summary['warningCount']} warnings")
    # Without a JSON output the run is a local one-off: print the details,
    # errors first, then the warnings (typically unresolved imports: a
    # dependency missing from the check venv degrades the analysis).
    if details or not out:
        for severity in ("error", "warning"):
            diags = [diag for diag in data["generalDiagnostics"] if diag["severity"] == severity]
            if not diags:
                continue
            print(f"{severity}s:")
            for diag in sorted(diags, key=lambda d: (d.get("rule", ""), d["file"], d["range"]["start"]["line"])):
                line = diag["range"]["start"]["line"] + 1
                print(f"  [{diag.get('rule', '')}] {diag['file']}:{line}  {diag['message'].splitlines()[0]}")


def cmd_run(args) -> int:
    """Analyze the examples against the library sources with the api-user profile."""
    examples_dir = Path(args.examples_dir).resolve()
    library_src = Path(args.library_src).resolve()
    if not examples_dir.is_dir():
        print(f"examples checkout not found in {examples_dir}: clone app-bricks-examples there or pass --examples-dir", file=sys.stderr)
        return 2
    include = [root for root in EXAMPLES_ROOTS if (examples_dir / root).is_dir()]
    if not include:
        print(f"no examples roots found in {examples_dir}", file=sys.stderr)
        return 2
    try:
        rules = load_rules(find_rules_file(library_src, args.rules))
        # extraPaths must sit at the top level, not inside the execution environment
        # of the examples: the library sources live outside that root, so pyright
        # analyzes them with the default environment. Scoped to the environment, the
        # library's own absolute imports (e.g. app_utils/leds.py importing Logger from
        # arduino.app_utils) resolved against site-packages only, where the arduino
        # namespace holds just the router bridge, and every symbol re-exported through
        # such an import came back as "unknown import symbol" in the examples.
        config = {
            **profile_config(rules, args.profile),
            "include": include,
            "extraPaths": [str(library_src)],
            "executionEnvironments": [{"root": "."}],
        }
        python = resolve_interpreter(args.python)
        # Pyright only reports diagnostics for the analyzed files, i.e. the
        # examples: the library reached through extraPaths is never reported,
        # even when the root cause is one of its annotations. Paths are made
        # relative to the examples checkout.
        data = run_pyright(examples_dir, config, python, args.pyright_version or rules.get("pyrightVersion", PYRIGHT_VERSION), examples_dir)
    except (OSError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 2
    report_run(data, f"against {library_src} (profile {args.profile})", args.out, args.details)
    return 0


def cmd_typing(args) -> int:
    """Type-check the library sources themselves with the app-bricks-py profile."""
    library_src = Path(args.library_src).resolve()
    if not (library_src / LIBRARY_PACKAGE).is_dir():
        print(f"library sources not found in {library_src}: expected a {LIBRARY_PACKAGE}/ package", file=sys.stderr)
        return 2
    try:
        rules = load_rules(find_rules_file(library_src, args.rules))
        # The sources root is the execution root: absolute imports of the library
        # resolve from it, as they do from site-packages once installed.
        config = {
            **profile_config(rules, args.profile),
            "include": [LIBRARY_PACKAGE],
            "executionEnvironments": [{"root": "."}],
        }
        python = resolve_interpreter(args.python)
        # Paths relative to the repository root, so they read as src/arduino/...
        data = run_pyright(library_src, config, python, args.pyright_version or rules.get("pyrightVersion", PYRIGHT_VERSION), library_src.parent)
    except (OSError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 2
    report_run(data, f"in {library_src} (profile {args.profile})", args.out, args.details)
    return 0


def error_index(data: dict, severity: str = "error") -> tuple[Counter, dict]:
    """Index the diagnostics of a severity by a line-shift-tolerant key: (file, rule, message first line).

    Also returns the 1-based lines of the occurrences of each key (informational
    only: lines are not part of the key, so moved code does not diff as new).
    """
    counts: Counter = Counter()
    occurrences: dict[tuple, list[int]] = {}
    for diag in data.get("generalDiagnostics", []):
        if diag["severity"] != severity:
            continue
        key = (diag["file"], diag.get("rule", ""), diag["message"].splitlines()[0])
        counts[key] += 1
        occurrences.setdefault(key, []).append(diag["range"]["start"]["line"] + 1)
    return counts, occurrences


def warnings_by_rule(data: dict) -> Counter:
    """Warning diagnostics counted per rule: warnings never weigh on the verdict,
    but the debt they describe (rules downgraded in the profile, mostly Unknown
    propagating from untyped code) deserves to be visible."""
    return Counter(diag.get("rule", "") for diag in data.get("generalDiagnostics", []) if diag["severity"] == "warning")


def warnings_table(rules: Counter) -> list[str]:
    rows = ["| Rule | Warnings |", "|---|---|"]
    rows += [f"| {rule} | {count} |" for rule, count in rules.most_common()]
    return rows


def error_table(entries: dict, occurrences: dict, max_rows: int | None = None) -> list[str]:
    """Markdown table rows for an error index, with pipes escaped for the cells.

    With max_rows the table is cut after that many entries and says how many more
    there are: the complete list is in the pyright outputs the workflow uploads.
    """
    rows = ["| File | Line(s) | Rule | Message |", "|---|---|---|---|"]
    keys = sorted(entries)
    shown = keys if max_rows is None else keys[:max_rows]
    for key in shown:
        file, rule, message = key
        lines = ", ".join(str(line) for line in sorted(occurrences[key]))
        rows.append(f"| `{file}` | {lines} | {rule} | {message.replace('|', '\\|')} |")
    if len(shown) < len(keys):
        rows += ["", f"… and {len(keys) - len(shown)} more, see the pyright outputs artifact for the complete list."]
    return rows


# A section is passed, failed, or warning: the last one is for facts worth a
# look that are not errors of the PR, like a brick without examples yet.
STATUS_ICONS = {"passed": "✅", "warning": "⚠️", "failed": "❌"}


def section_markdown(section: dict) -> str:
    """Standalone markdown of a check section: verdict in the heading, one-line
    result, notes, then the details (this is what a single check prints and what
    a workflow appends to its summary when it runs one check only)."""
    lines = [f"## {STATUS_ICONS[section['status']]} {section['title']}", "", section["result"]]
    for note in section.get("notes", []):
        lines += ["", note]
    if section.get("details"):
        lines += ["", section["details"]]
    if section.get("full_report"):
        # Pre-existing errors are part of the story too, but collapsed: the diff
        # above stays the signal of the PR.
        lines += ["", "<details>", f"<summary>{section['full_report_title']}</summary>", "", section["full_report"], "", "</details>"]
    if section.get("footer"):
        lines += ["", section["footer"]]
    return "\n".join(lines)


def emit_section(section: dict, summary: str | None, result: str | None, no_summary: bool = False) -> None:
    """Print a section, append it to the summary file (unless the report mode will
    compose it later) and save it for the report mode."""
    report = section_markdown(section)
    print(report)
    summary_path = None if no_summary else (summary or os.environ.get("GITHUB_STEP_SUMMARY"))
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report + "\n\n---\n\n")
    if result:
        Path(result).write_text(json.dumps(section, indent=2) + "\n")


def write_github_output(**values) -> None:
    """Expose values to the workflow. Best effort: the file belongs to the runner,
    and a context that only inherits the variable (a test job running as another
    user) must not fail on it."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    try:
        with open(output_path, "a") as f:
            for name, value in values.items():
                f.write(f"{name}={value}\n")
    except OSError as e:
        print(f"could not write to GITHUB_OUTPUT: {e}", file=sys.stderr)


def cmd_diff(args) -> int:
    base_data, head_data = json.loads(Path(args.base).read_text()), json.loads(Path(args.head).read_text())
    base_counts, base_occurrences = error_index(base_data)
    head_counts, head_occurrences = error_index(head_data)
    base_warnings, head_warnings = warnings_by_rule(base_data), warnings_by_rule(head_data)
    # Warnings follow the same rule as errors: the pre-existing ones are debt to
    # keep in sight, the ones this PR adds weigh on the verdict, as a warning.
    base_warning_counts, _ = error_index(base_data, "warning")
    head_warning_counts, head_warning_occurrences = error_index(head_data, "warning")

    new = {key: count - base_counts.get(key, 0) for key, count in head_counts.items() if count > base_counts.get(key, 0)}
    fixed = {key: count - head_counts.get(key, 0) for key, count in base_counts.items() if count > head_counts.get(key, 0)}
    new_warnings = {
        key: count - base_warning_counts.get(key, 0) for key, count in head_warning_counts.items() if count > base_warning_counts.get(key, 0)
    }
    new_total, fixed_total, head_total = sum(new.values()), sum(fixed.values()), sum(head_counts.values())
    new_warnings_total = sum(new_warnings.values())
    head_warnings_total = sum(head_warnings.values())
    pre_existing = head_total - new_total
    pre_existing_warnings = head_warnings_total - new_warnings_total

    notes: list[str] = []
    if not new and not fixed and not new_warnings:
        notes.append("✅ No new errors in this PR.")
    if new_warnings:
        note = f"⚠️ {new_warnings_total} new warning{'s' if new_warnings_total != 1 else ''} in this PR, listed in the details"
        notes.append(note + (f" ({pre_existing_warnings} pre-existing)." if pre_existing_warnings else "."))
    elif head_warnings_total:
        notes.append(f"⚠️ {head_warnings_total} pre-existing warning{'s' if head_warnings_total != 1 else ''}, broken down by rule in the details.")
    if not new and head_counts:
        # Tolerated, but not to be forgotten: without new errors every remaining
        # one is pre-existing, and the full list is in the details.
        notes.append(f"⚠️ {pre_existing} pre-existing error{'s' if pre_existing != 1 else ''}, listed in the full report below.")
    if new and args.guidance:
        notes.append(f"❌ {args.guidance}")

    details: list[str] = []
    for title, entries, occurrences in (("New errors", new, head_occurrences), ("Fixed errors", fixed, base_occurrences)):
        if entries:
            details += [f"#### {title}", ""] + error_table(entries, occurrences) + [""]
    if new_warnings:
        details += ["#### New warnings", ""] + error_table(new_warnings, head_warning_occurrences, FULL_REPORT_MAX_ROWS) + [""]
    if head_warnings:
        details += [f"#### Warnings by rule: {sum(head_warnings.values())} against head", ""] + warnings_table(head_warnings) + [""]
    subject = args.subject or f"Errors in the Python sources of {args.examples_label} analyzed against {args.library_label}"
    section = {
        "title": args.title,
        "status": "failed" if new else "warning" if new_warnings else "passed",
        "informative": not args.fail_on_new,
        "result": (
            f"{subject}: base {sum(base_counts.values())} → head {head_total} (**{new_total} new**, {fixed_total} fixed)"
            f" · warnings {sum(base_warnings.values())} → {sum(head_warnings.values())}"
        ),
        "cell": f"**{new_total} new**, {fixed_total} fixed, {pre_existing} pre-existing"
        + (f" · **{new_warnings_total} new warning{'s' if new_warnings_total != 1 else ''}**" if new_warnings else "")
        + (f" · {head_warnings_total} warning{'s' if head_warnings_total != 1 else ''}" if head_warnings_total else ""),
        "notes": notes,
        "details": "\n".join(details).rstrip(),
        "full_report_title": f"Full report: {head_total} error{'s' if head_total != 1 else ''} against head",
        "full_report": "\n".join(error_table(head_counts, head_occurrences, FULL_REPORT_MAX_ROWS)) if head_counts else "",
        "footer": f"📥 [Download full pyright JSON report]({args.reports_url})" if args.reports_url else "",
        "new_errors": new_total,
    }
    emit_section(section, args.summary, args.result, args.no_summary)

    # Annotations: warnings on an informative run, errors on a blocking one. The
    # file/line properties place them inline in the PR diff, which only makes
    # sense when the analyzed files belong to the repository running the check.
    level = "error" if args.fail_on_new else "warning"
    for key in sorted(new):
        file, rule, message = key
        line = head_occurrences[key][0]
        properties = f" file={file},line={line}" if args.annotate_files else ""
        print(f"::{level}{properties}::{args.title}: {file}:{line} [{rule}] {message}")
    for key in sorted(new_warnings):
        file, rule, message = key
        line = head_warning_occurrences[key][0]
        properties = f" file={file},line={line}" if args.annotate_files else ""
        print(f"::warning{properties}::{args.title}: {file}:{line} [{rule}] {message}")
    # Exposed to the workflow, which turns it into a label on the PR.
    write_github_output(new_errors=new_total)

    return 1 if new and args.fail_on_new else 0


def cmd_report(args) -> int:
    """Compose the sections saved by diff/coverage into one report: a verdict table
    at the top, every accessory information in a single collapsed block."""
    sections = [json.loads(Path(path).read_text()) for path in args.sections]
    # The heading carries the worst status of the sections: ❌ when a pyright
    # section has new errors, ⚠️ when only the coverage has something to say,
    # ✅ otherwise. Whether a failure also fails the job is the workflow's
    # business (--fail-on-new).
    diff_sections = [s for s in sections if "new_errors" in s]
    failed = any(s["status"] == "failed" for s in sections)
    warned = any(s["status"] == "warning" for s in sections)
    blocking_failed = any(s["status"] == "failed" and not s.get("informative") for s in diff_sections)

    heading = "❌" if failed else "⚠️" if warned else "✅"
    lines = [f"## {heading} {args.title}", ""]
    # The verdict table is in plain sight when a check has something to say;
    # when every check passed, the heading says it all and the table is folded.
    table = ["| Check | Result |", "|---|---|"] + [f"| {STATUS_ICONS[s['status']]} {s['title']} | {s['cell']} |" for s in sections]
    if failed or warned:
        lines += table
    else:
        lines += ["<details>", f"<summary>All {len(sections)} checks passed</summary>", ""] + table + ["", "</details>"]
    for s in sections:
        for note in s.get("notes", []):
            if not note.startswith("✅"):
                icon, text = note.split(" ", 1)
                lines += ["", f"{icon} **{s['title']}**: {text}"]
    # Every accessory information in one collapsed block: tables of new and
    # fixed errors, full lists, coverage details, download links.
    lines += ["", "<details>", "<summary>Details</summary>", ""]
    for s in sections:
        lines += [f"### {s['title']}", "", s["result"], ""]
        if s.get("details"):
            lines += [s["details"], ""]
        if s.get("full_report"):
            lines += [f"#### {s['full_report_title']}", "", s["full_report"], ""]
        if s.get("footer"):
            lines += [s["footer"], ""]
    lines += ["</details>"]
    if args.reports_url:
        # One artifact holds the pyright outputs of every section: one link, at
        # the bottom and always in sight.
        lines += ["", f"📥 [Download the pyright JSON outputs of these checks]({args.reports_url})"]
    report = "\n".join(lines) + "\n"

    print(report)
    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report)
    write_github_output(status="failed" if failed else "passed", new_errors=sum(s.get("new_errors", 0) for s in diff_sections))
    return 1 if blocking_failed and args.fail_on_new else 0


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

    # Never a failure: a new brick may legitimately land before its examples do,
    # so missing examples are a warning to keep in sight, not an error of the PR.
    plural = "s" if len(uncovered) != 1 else ""
    if uncovered:
        result = f"⚠️ {len(uncovered)} brick{plural} without examples in {EXAMPLES_REPO_MD}."
        details = "\n".join(f"- `{name}`" + (" — **introduced by this PR**" if name in introduced else "") for name in uncovered)
        details += "\n\nInformative only: a new brick may legitimately land before its examples do."
        named = ", ".join(f"`{name}`" for name in uncovered)
        notes = [f"⚠️ {len(uncovered)} brick{plural} without examples: {named}" + (" (introduced by this PR)" if introduced else "") + "."]
    else:
        result = f"✅ Every non-disabled brick has at least one example in {EXAMPLES_REPO_MD}."
        details = ""
        notes = []
    section = {
        "title": args.title,
        "status": "warning" if uncovered else "passed",
        "informative": True,
        "result": result,
        "cell": f"{len(uncovered)} brick{plural} without examples" if uncovered else "every brick covered",
        "notes": notes,
        "details": details,
    }
    emit_section(section, args.summary, args.result, args.no_summary)
    for name in introduced:
        print(f"::notice::{args.title}: this PR introduces the brick '{name}', which has no examples in app-bricks-examples yet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    deps = sub.add_parser("deps", help="print a pyproject's dependencies for the check venv")
    deps.add_argument("--pyproject", default="pyproject.toml")
    deps.add_argument("--extra", default="all")
    deps.set_defaults(func=cmd_deps)

    def add_analysis_options(parser, default_profile: str) -> None:
        parser.add_argument("--library-src", default="src")
        parser.add_argument(
            "--python",
            help=f"python interpreter of the check venv, which must have the library dependencies installed "
            f"(defaults to {DEFAULT_VENV_PYTHON} when present)",
        )
        parser.add_argument("--rules", help=f"path to {RULES_FILE} (defaults to the one next to --library-src)")
        parser.add_argument("--profile", default=default_profile, help=f"profile of {RULES_FILE} to apply (default: {default_profile})")
        parser.add_argument("--pyright-version", help=f"pyright release to run (defaults to pyrightVersion in {RULES_FILE})")
        parser.add_argument("--out", help="write the diagnostics as JSON; when omitted, details are printed instead")
        parser.add_argument("--details", action="store_true", help="also print the error and warning diagnostics, grouped by rule")

    run = sub.add_parser("run", help="run pyright over the examples against a library source (profile api-user)")
    run.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    add_analysis_options(run, PROFILE_API_USER)
    run.set_defaults(func=cmd_run)

    typing = sub.add_parser("typing", help="run pyright over the library sources themselves (profile app-bricks-py)")
    add_analysis_options(typing, PROFILE_LIBRARY)
    typing.set_defaults(func=cmd_typing)

    diff = sub.add_parser("diff", help="compare two run outputs and report new/fixed errors")
    diff.add_argument("--base", required=True)
    diff.add_argument("--head", required=True)
    diff.add_argument("--title", default="Examples alignment check", help="name of the check in the report")
    diff.add_argument("--subject", help="what the counts describe (default: the examples analyzed against the library)")
    diff.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    diff.add_argument("--result", help="save the check section as JSON, for the report mode")
    diff.add_argument("--no-summary", action="store_true", help="do not append the section to the job summary (the report mode will)")
    diff.add_argument("--reports-url", help="link to the uploaded run outputs, appended to the summary")
    diff.add_argument(
        "--guidance",
        default="This PR introduces errors in the published examples: either adapt the library change to keep the "
        "examples' contract, or open the matching PR on app-bricks-examples and coordinate the merge.",
        help="what to do about new errors, shown when there are some",
    )
    diff.add_argument("--examples-label", default=EXAMPLES_REPO_MD, help="how the summary names the analyzed examples")
    diff.add_argument("--library-label", default="this library", help="how the summary names the library they are analyzed against")
    diff.add_argument("--fail-on-new", action="store_true", help="exit 1 when the head introduces new errors (blocking check)")
    diff.add_argument(
        "--annotate-files",
        action="store_true",
        help="place the annotations inline on file and line; only for a repository that holds the analyzed files",
    )
    diff.set_defaults(func=cmd_diff)

    coverage = sub.add_parser("coverage", help="report library bricks that have no examples")
    coverage.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    coverage.add_argument("--head-src", default="src")
    coverage.add_argument("--base-src", help="library source of the PR base, to flag bricks introduced by the PR")
    coverage.add_argument("--title", default="Bricks coverage", help="name of the check in the report")
    coverage.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    coverage.add_argument("--result", help="save the check section as JSON, for the report mode")
    coverage.add_argument("--no-summary", action="store_true", help="do not append the section to the job summary (the report mode will)")
    coverage.set_defaults(func=cmd_coverage)

    report = sub.add_parser("report", help="compose the sections saved by diff/coverage into one report")
    report.add_argument("sections", nargs="+", help="section JSON files, in display order")
    report.add_argument("--title", default="Pyright checks and examples coverage")
    report.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    report.add_argument("--fail-on-new", action="store_true", help="exit 1 when a non-informative section failed")
    report.add_argument("--reports-url", help="link to the uploaded pyright outputs of all the sections, appended to the details")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
