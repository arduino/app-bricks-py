# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Pyright checks shared by app-bricks-py, app-bricks-examples and App Lab.

The rules come from pyright-rules.json (repository root, shipped in the wheel as
arduino/app_bricks/static/pyright-rules.json): the library decides how code is
type-checked through two profiles, app-bricks-py for its own sources and api-user
for code written against its API; this script adds the environment (paths,
interpreter, execution root). Modes:

  deps      Print the library dependencies (core + recursively expanded extra),
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


def error_index(data: dict) -> tuple[Counter, dict]:
    """Index error diagnostics by a line-shift-tolerant key: (file, rule, message first line).

    Also returns the 1-based lines of the occurrences of each key (informational
    only: lines are not part of the key, so moved code does not diff as new).
    """
    counts: Counter = Counter()
    occurrences: dict[tuple, list[int]] = {}
    for diag in data.get("generalDiagnostics", []):
        if diag["severity"] != "error":
            continue
        key = (diag["file"], diag.get("rule", ""), diag["message"].splitlines()[0])
        counts[key] += 1
        occurrences.setdefault(key, []).append(diag["range"]["start"]["line"] + 1)
    return counts, occurrences


def error_table(entries: dict, occurrences: dict) -> list[str]:
    """Markdown table rows for an error index, with pipes escaped for the cells."""
    rows = ["| File | Line(s) | Rule | Message |", "|---|---|---|---|"]
    for key in sorted(entries):
        file, rule, message = key
        lines = ", ".join(str(line) for line in sorted(occurrences[key]))
        rows.append(f"| `{file}` | {lines} | {rule} | {message.replace('|', '\\|')} |")
    return rows


def cmd_diff(args) -> int:
    base_counts, base_occurrences = error_index(json.loads(Path(args.base).read_text()))
    head_counts, head_occurrences = error_index(json.loads(Path(args.head).read_text()))

    new = {key: count - base_counts.get(key, 0) for key, count in head_counts.items() if count > base_counts.get(key, 0)}
    fixed = {key: count - head_counts.get(key, 0) for key, count in base_counts.items() if count > head_counts.get(key, 0)}

    # The verdict goes in the heading: the summary is also posted as a PR comment,
    # and the outcome must be readable at a glance.
    status = "❌" if new else "✅"
    lines = [
        f"## {status} Examples alignment check",
        "",
        f"Errors in the Python sources of {args.examples_label} analyzed against {args.library_label}: "
        f"base {sum(base_counts.values())} → head {sum(head_counts.values())} "
        f"(**{sum(new.values())} new**, {sum(fixed.values())} fixed)",
    ]
    for title, entries, occurrences in (("New errors", new, head_occurrences), ("Fixed errors", fixed, base_occurrences)):
        if entries:
            lines += ["", f"### {title}", ""] + error_table(entries, occurrences)
    if not new and not fixed:
        lines += ["", "✅ No new errors in this PR."]
    if not new and head_counts:
        # Tolerated, but not to be forgotten: without new errors every remaining
        # one is pre-existing, and the full list is in the collapsed report below.
        pre_existing = sum(head_counts.values())
        lines += ["", f"⚠️ {pre_existing} pre-existing error{'s' if pre_existing != 1 else ''}, listed in the full report below."]
    if head_counts:
        # Pre-existing errors are part of the story too, but collapsed: the diff
        # above stays the signal of the PR.
        lines += ["", "<details>", f"<summary>Full report: {sum(head_counts.values())} errors against head</summary>", ""]
        lines += error_table(head_counts, head_occurrences)
        lines += ["", "</details>"]
    if new:
        lines += [
            "",
            "❌ This PR introduces errors in the published examples: either adapt the library change to keep the "
            "examples' contract, or open the matching PR on app-bricks-examples and coordinate the merge.",
        ]
    if args.reports_url:
        lines += ["", f"📥 [Download full pyright JSON report]({args.reports_url})"]
    # Horizontal rule separating this section from the coverage one, appended
    # to the same job summary by the next step.
    lines += ["", "---"]
    report = "\n".join(lines)

    print(report)
    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report + "\n")
    # Annotations: warnings on an informative run, errors on a blocking one. The
    # file/line properties place them inline in the PR diff, which only makes
    # sense when the analyzed files belong to the repository running the check.
    level = "error" if args.fail_on_new else "warning"
    for key in sorted(new):
        file, rule, message = key
        line = head_occurrences[key][0]
        properties = f" file={file},line={line}" if args.annotate_files else ""
        print(f"::{level}{properties}::examples alignment: {file}:{line} [{rule}] {message}")
    # Exposed to the workflow, which turns it into a label on the PR. Best effort:
    # the file belongs to the runner, and a context that only inherits the
    # variable (a test job running as another user) must not fail on it.
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        try:
            with open(output_path, "a") as f:
                f.write(f"new_errors={sum(new.values())}\n")
        except OSError as e:
            print(f"could not write new_errors to GITHUB_OUTPUT: {e}", file=sys.stderr)

    return 1 if new and args.fail_on_new else 0


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

    status = "❌" if uncovered else "✅"
    lines = [f"### {status} Bricks without examples", ""]
    if uncovered:
        lines.append(f"❌ {len(uncovered)} bricks have no examples in app-bricks-examples:")
        lines += [f"- `{name}`" + (" — **introduced by this PR**" if name in introduced else "") for name in uncovered]
        lines += ["", "Informative only: a new brick may legitimately land before its examples do."]
    else:
        lines.append(f"✅ Every non-disabled brick has at least one example in {EXAMPLES_REPO_MD} repository.")
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
    diff.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    diff.add_argument("--reports-url", help="link to the uploaded run outputs, appended to the summary")
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
    coverage.add_argument("--summary", help="markdown output file (defaults to GITHUB_STEP_SUMMARY)")
    coverage.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
