# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""
hf_downloader — Hugging Face Model Downloader CLI

A command-line tool for downloading GGUF-format models from Hugging Face
repositories. It targets llama.cpp-style repos that may contain multiple
quantization variants and optional multimodal projection (mmproj) files.
After downloading, it auto-generates a ``models.ini`` configuration file
that indexes all downloaded models for use by downstream runners.

Usage — one input, two syntaxes
------------------------------
``--model-url`` is the only way to name a model, and it accepts either form, so the
host has a single variable to set whatever the model is::

    # 1. File URL: downloads that exact file at that exact commit (reproducible).
    hf_downloader --model-url https://huggingface.co/<org>/<repo>/blob/<revision>/<file>.gguf
                  [--model-mmproj-url https://huggingface.co/<org>/<repo>/blob/<revision>/mmproj-<q>.gguf]

    # 2. Compact key, as llama.cpp's "-hf": downloads whatever matches the
    #    quantization, at the tip of the default branch. model_type is optional.
    hf_downloader --model-url [<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>]

A leading ``http://``/``https://`` selects form 1; anything else is parsed as a key.
The quantization field also accepts a full file name or an explicit glob, so a single
file can be pinned by name without a URL.

Key options
-----------
--output-dir DIR        Destination directory (default: current directory).
                        Files are saved under ``<output-dir>/<repo-id>/``.
--hf-token KEY          Hugging Face API token for gated/private repositories.
--verbose               Print resolved parameters before downloading.

After all files are downloaded, ``models.ini`` is written to ``<output-dir>``
mapping each model stem to its GGUF path (and mmproj path where present).
"""

import fnmatch
import os
import re
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
import argparse
import configparser
from pathlib import Path
from tqdm.auto import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.download_marker import write_marker
from common.model_metadata import is_bookkeeping_name, write_metadata


def emit_json_info(description: str, artifacts: list[str] | None = None, downloading: bool | None = None):
    data: dict = {"event": "info", "description": description}
    if artifacts is not None:
        data["artifacts"] = artifacts
    if downloading is not None:
        data["downloading"] = downloading
    print(json.dumps(data), flush=True)


def emit_json_error(description: str, downloading: bool | None = None):
    data: dict = {"event": "error", "description": description}
    if downloading is not None:
        data["downloading"] = downloading
    print(json.dumps(data), flush=True)


def remove_model_dir(output_dir: str, base_dir: str) -> None:
    """Remove the repo directory and prune now-empty parent dirs up to base_dir.

    repo_id may contain a '/', so output_dir is nested (e.g.
    <base>/moondream/moondream2-gguf). Deleting only output_dir would leave an
    empty org directory (<base>/moondream) behind; walk up removing empty parents,
    stopping at base_dir (the mounted /models, which is never removed).
    """
    base = os.path.abspath(base_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    parent = os.path.dirname(os.path.abspath(output_dir))
    while parent != base and parent.startswith(base + os.sep):
        try:
            os.rmdir(parent)  # only succeeds if the directory is empty
        except OSError:
            break
        parent = os.path.dirname(parent)


def has_model_content(output_dir: str) -> bool:
    """True when *output_dir* holds a downloaded file, ignoring bookkeeping entries.

    The ".download" marker, ".arduino_metadata.yaml" and huggingface_hub's ".cache"
    tree are not model content: a directory holding only those is a leftover from an
    interrupted or deleted download, not an installed model.
    """
    base = Path(output_dir)
    if not base.is_dir():
        return False
    return any(p.is_file() and not is_bookkeeping_name(p.name) and ".cache" not in p.parts for p in base.rglob("*"))


def prune_emptied_repo_dir(output_dir: str, base_dir: str) -> bool:
    """Drop *output_dir* once its last GGUF is gone; return whether it was removed.

    ``delete_matched_files`` unlinks the model files and prunes empty directories, but
    the ".arduino_metadata.yaml" record it knows nothing about would keep the repo
    directory alive as a ghost. Removing the whole directory only when no GGUF is left
    means a sibling quantization is never touched.
    """
    if not os.path.isdir(output_dir):
        return False
    if any(p.suffix == ".gguf" for p in Path(output_dir).rglob("*")):
        return False
    remove_model_dir(output_dir, base_dir)
    return True


def install_signal_handlers() -> None:
    """Translate SIGINT/SIGTERM into KeyboardInterrupt so cleanup runs before
    exit. SIGKILL (-9) cannot be caught."""
    import signal

    def _handler(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class JsonProgress(tqdm):
    """tqdm replacement that reports download progress as JSON events on stdout.

    huggingface_hub's Xet downloader tracks two byte counters: bytes written to disk
    ("reconstruction") and bytes pulled from the network ("transfer"), and renders one
    progress bar for each. Only the reconstruction bar honours ``tqdm_class``; the
    transfer bar is created with huggingface_hub's own tqdm — and would print a real
    progress bar next to our JSON — unless the class also exposes ``update_transfer``,
    in which case both counters are routed into this single object instead
    (see ``huggingface_hub.utils._xet_progress_reporting.XetDownloadProgressReporter``).

    This is an internal integration point of huggingface_hub, which is why huggingface_hub
    and hf_xet are pinned in requirements.txt: bump them together with a check of the
    download output (``tests/test_hf_downloader.py`` covers the contract).
    """

    # Minimum seconds between emitted "update" events to avoid flooding stdout.
    EMIT_INTERVAL = 1.0

    # Suffixes huggingface_hub appends to the file name when naming its Xet bars. They
    # describe an implementation detail, so they are stripped from the reported description.
    DESC_SUFFIXES = (": reconstructing file", ": downloading bytes")

    def __init__(self, *args, **kwargs):
        self._complete_emitted = False
        self._last_emit = 0.0
        self._transferred = 0
        super().__init__(*args, **kwargs)
        # Emit an initial "start" event
        self._emit("start")

    def _current(self):
        """Number of bytes to report as downloaded.

        ``self.n`` counts bytes written to disk. For Xet downloads it only moves when
        buffered chunks are flushed, which happens in big bursts — it can sit at 0 for the
        first tens of MB — so on its own it makes progress look frozen. ``_transferred``
        counts bytes received from the network and advances continuously, but can end up
        below the file size when chunks are served from the local Xet cache. Report
        whichever of the two is furthest along, capped at the file size.
        """
        current = max(self.n, self._transferred)
        return min(current, self.total) if self.total else current

    def _description(self):
        # tqdm appends ": " to desc when it is set via set_description().
        desc = (self.desc or "").removesuffix(": ")
        for suffix in self.DESC_SUFFIXES:
            desc = desc.removesuffix(suffix)
        return desc

    def _emit(self, event_type):
        """Helper to print the current state as JSON"""
        self._last_emit = time.monotonic()
        current = self._current()
        pct = round((current / self.total) * 100, 2) if self.total else 0
        data = {
            "event": event_type,
            "description": self._description(),
            "current": current,
            "total": self.total,
            "unit": self.unit,
            "percentage": f"{pct}%",
        }
        print(json.dumps(data), flush=True)

    def update(self, n=1):
        displayed = super().update(n)
        # Throttle: only emit an "update" event once EMIT_INTERVAL has elapsed.
        if time.monotonic() - self._last_emit >= self.EMIT_INTERVAL:
            self._emit("update")
        return displayed

    def update_transfer(self, n=1):
        """Track bytes received from the network, and report them (see _current()).

        This is the counter that makes progress look alive: it is updated roughly ten times
        per second, against disk writes that arrive in multi-MB bursts. It is kept apart
        from ``self.n`` so that completion stays decided by the bytes actually written.
        Implementing this method is also what stops huggingface_hub from creating a second,
        terminal-drawn progress bar for this counter.
        """
        self._transferred = max(0, self._transferred + int(n or 0))
        # Throttle, as update() does, to avoid flooding stdout.
        if time.monotonic() - self._last_emit >= self.EMIT_INTERVAL:
            self._emit("update")

    def set_transfer_postfix_str(self, postfix, refresh=False):
        """Ignore the transfer rate; it is not part of the reported events."""

    def close(self):
        # Only report completion if the transfer actually finished.
        if self.total and self.n >= self.total and not self._complete_emitted:
            self._complete_emitted = True
            self._emit("complete")
        # tqdm writes a bare newline when closing a bar with leave=True. No bar was ever
        # drawn, so there is nothing to leave on screen.
        self.leave = False
        super().close()

    def display(self, msg=None, pos=None):
        # Do not display the progress bar in the terminal, we will emit JSON events instead
        pass


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """Parse a Hugging Face URL and return (repo_id, filename, revision).

    Supports URLs like:
      https://huggingface.co/<org>/<repo>/resolve/<revision>/<filename>
      https://huggingface.co/<org>/<repo>/blob/<revision>/<filename>
    """
    match = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+?)(?:\?.*)?$",
        url,
    )
    if not match:
        raise ValueError(f"Invalid Hugging Face URL: {url}\nExpected format: https://huggingface.co/<org>/<repo>/resolve/<revision>/<filename>")
    repo_id = match.group(1)
    revision = match.group(2)
    filename = match.group(3)
    return repo_id, filename, revision


def parse_model_key(model_key: str) -> tuple[str, str, str, str | None]:
    """Parse a model key into ``(model_type, repo_id, quantization, mmproj_quantization)``.

    Accepted forms, colon-separated::

        <repo_id>:<quantization>                                    # llama.cpp -hf style
        <model_type>:<repo_id>:<quantization>
        <model_type>:<repo_id>:<quantization>:<mmproj_quantization>

    ``model_type`` is optional and purely informative — nothing selects on it — so a
    two-field key is accepted and reads like llama.cpp's ``-hf Qwen/Qwen3-8B-GGUF:Q8_0``.
    The field count alone disambiguates, because the quantization is always required.

    Raises:
        ValueError: when the field count is wrong, or repo_id/quantization are empty.
    """
    parts = model_key.split(":")
    if len(parts) == 2:
        model_type, repo_id, quantization, mmproj_quantization = "", parts[0], parts[1], None
    elif len(parts) == 3:
        model_type, repo_id, quantization, mmproj_quantization = parts[0], parts[1], parts[2], None
    elif len(parts) == 4:
        model_type, repo_id, quantization, mmproj_quantization = parts
    else:
        raise ValueError(
            f"Invalid model key: {model_key}\n"
            "Expected format: [<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>] "
            "(e.g. Qwen/Qwen3-8B-GGUF:Q8_0 or llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0)"
        )
    if repo_id == "":
        raise ValueError("repo_id cannot be empty")
    if quantization == "":
        raise ValueError("quantization cannot be empty")
    return model_type, repo_id, quantization, mmproj_quantization or None


def is_hf_url(spec: str) -> bool:
    """True when *spec* is a URL rather than a compact model key.

    Checked before any ``:`` splitting, since ``https://...`` would otherwise parse
    as a two-field key with repo_id ``https``.
    """
    return spec.startswith(("http://", "https://"))


def gguf_pattern(spec: str, mmproj: bool = False) -> str:
    """Turn a quantization or file name *spec* into an fnmatch pattern for GGUF files.

    A bare quantization is widened (``Q4_0`` -> ``*Q4_0*.gguf``); an explicit glob or
    a full file name is taken as it stands, which is how a single specific file can be
    pinned without a URL (``gemma-4-E2B-it-Q4_0.gguf``).
    """
    if "*" in spec or spec.endswith(".gguf"):
        return spec
    return f"*mmproj*{spec}*.gguf" if mmproj else f"*{spec}*.gguf"


def resolve_model_source(model_url: str, model_mmproj_url: str | None = None) -> dict:
    """Resolve *model_url* into everything needed to fetch, check or delete the model.

    Two syntaxes are accepted, so a single variable covers every case:

    1. A Hugging Face file URL — ``https://huggingface.co/<org>/<repo>/{blob,resolve}/<revision>/<file>``.
       Downloads exactly that file at that revision, which is the reproducible form:
       the commit is pinned in the URL itself. The companion mmproj file is given as a
       second URL in *model_mmproj_url*.
    2. A compact key — ``[<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>]``,
       matching llama.cpp's ``-hf`` form. Downloads whatever files of the repository
       match the quantization, at the tip of the default branch.

    Returns:
        A dict with ``repo_id``, ``allow_pattern`` and ``mmproj_allow_pattern`` (used by
        check/delete/info in both cases), plus ``url_filename``/``url_revision`` and
        their mmproj counterparts, which are set only for syntax 1 and select the
        single-file download path.

    Raises:
        ValueError: when *model_url* is empty or neither syntax parses.
    """
    if not model_url:
        raise ValueError(
            "model_url is required. Give either a Hugging Face file URL "
            "(https://huggingface.co/<org>/<repo>/blob/<revision>/<file>.gguf) or a compact key "
            "([<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>], e.g. Qwen/Qwen3-8B-GGUF:Q8_0)"
        )

    source = {
        "repo_id": "",
        "allow_pattern": None,
        "mmproj_allow_pattern": None,
        "url_filename": None,
        "url_revision": None,
        "mmproj_url_filename": None,
        "mmproj_url_revision": None,
        "model_type": "",
    }

    if is_hf_url(model_url):
        repo_id, url_filename, url_revision = parse_hf_url(model_url)
        source["repo_id"] = repo_id
        source["url_filename"] = url_filename
        source["url_revision"] = url_revision
        # Basename as the pattern, so check/delete/info work the same as for a key.
        source["allow_pattern"] = url_filename.split("/")[-1]
        if model_mmproj_url:
            _, mmproj_filename, mmproj_revision = parse_hf_url(model_mmproj_url)
            source["mmproj_url_filename"] = mmproj_filename
            source["mmproj_url_revision"] = mmproj_revision
            source["mmproj_allow_pattern"] = mmproj_filename.split("/")[-1]
        return source

    model_type, repo_id, quantization, mmproj_quantization = parse_model_key(model_url)
    source["model_type"] = model_type
    source["repo_id"] = repo_id
    source["allow_pattern"] = gguf_pattern(quantization)
    if mmproj_quantization:
        source["mmproj_allow_pattern"] = gguf_pattern(mmproj_quantization, mmproj=True)
    return source


def matches_pattern(path: str, pattern: str) -> bool:
    """fnmatch a repo-relative *path* against *pattern*.

    Patterns are written against file names (e.g. ``*Q4_0*.gguf``), but some repos nest
    their files in per-quantization folders, so the full path is matched too.
    """
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path.split("/")[-1], pattern)


def list_repo_matches(repo_id: str, patterns: list[str], ignore_pattern: str | None = None) -> list[RepoFile]:
    """Return the files of *repo_id* matching any of *patterns*, minus *ignore_pattern*."""
    api = HfApi()
    all_files = [item for item in api.list_repo_tree(repo_id=repo_id, recursive=True) if isinstance(item, RepoFile)]
    matched = [f for f in all_files if any(matches_pattern(f.path, p) for p in patterns)]
    if ignore_pattern:
        matched = [f for f in matched if not matches_pattern(f.path, ignore_pattern)]
    return matched


def download_matched_files(
    repo_id: str,
    allow_pattern: str,
    output_dir: str,
    tqdm_class: type[tqdm],
    ignore_pattern: str | None = None,
    verbose: bool = False,
) -> None:
    """Download every file of *repo_id* matching *allow_pattern* into *output_dir*.

    ``snapshot_download`` is deliberately not used: it hands each individual file an
    internal aggregating progress bar, so per-file byte counts never reach *tqdm_class*
    and the JSON stream would describe huggingface_hub's own summary bars instead of the
    model files. Resolving the file list up front also lets us fail loudly when the
    requested quantization does not exist in the repo, rather than silently downloading
    nothing.
    """
    matched = list_repo_matches(repo_id, [allow_pattern], ignore_pattern=ignore_pattern)
    if not matched:
        raise FileNotFoundError(f"No file matching '{allow_pattern}' found in repository '{repo_id}'")
    for file in matched:
        if verbose:
            emit_json_info(f"Downloading '{file.path}' from {repo_id}")
        hf_hub_download(repo_id=repo_id, filename=file.path, local_dir=output_dir, tqdm_class=tqdm_class)


def delete_matched_files(output_dir: str, models_base: str, allow_pattern: str, verbose: bool = False):
    """Delete files inside output_dir whose names match allow_pattern (fnmatch-style).
    After deletion, removes any empty subdirectories but never output_dir itself.
    """
    base = Path(output_dir)
    models_base_path = Path(models_base)
    if not base.exists():
        emit_json_info(f"Directory does not exist, nothing to delete: {output_dir}")
        return
    matched = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, allow_pattern)]
    if not matched:
        emit_json_info(f"No files matching '{allow_pattern}' found in {output_dir}")
        return
    dirs_to_check: set[Path] = set()
    for f in matched:
        if verbose:
            emit_json_info(f"Deleting: {f}")
        dirs_to_check.add(f.parent)
        f.unlink()
    # Remove empty subdirectories (deepest first), but never output_dir itself
    for d in sorted(dirs_to_check, key=lambda p: len(p.parts), reverse=True):
        if d == models_base:
            continue
        if d.exists() and not any(d.iterdir()):
            if verbose:
                emit_json_info(f"Removing empty directory: {d}")
            d.rmdir()
    # Remove all empty directories up to output_dir. List all directories under models_base and check if they are empty, removing them
    for d in sorted(models_base_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and d != base and not any(d.iterdir()):
            if verbose:
                emit_json_info(f"Removing empty directory: {d}")
            d.rmdir()


def generate_models_ini(models_dir: Path):
    config = configparser.ConfigParser()

    for gguf_file in sorted(models_dir.rglob("*.gguf")):
        if "mmproj" in gguf_file.name:
            continue

        section = gguf_file.stem
        config[section] = {}
        config[section]["model"] = str(gguf_file.as_posix())

        # Look for mmproj file in the same directory
        mmproj_files = sorted(gguf_file.parent.glob("*mmproj*.gguf"))
        if mmproj_files:
            config[section]["mmproj"] = str(mmproj_files[0].as_posix())

    output_path = models_dir / "models.ini"
    with open(output_path, "w") as f:
        config.write(f)

    emit_json_info(f"Generated models.ini with {len(config.sections())} model(s)", artifacts=[str(output_path)])


def main():
    parser = argparse.ArgumentParser(description="Download an Hugging Face model via HF download API")
    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        metavar="URL_OR_KEY",
        help="The model to download, as either a Hugging Face file URL "
        "(e.g. https://huggingface.co/org/repo/blob/<revision>/model.gguf; /resolve/ works too) "
        "or a compact key [<model_type>:]<repo_id>:<quantization>[:<mmproj_quantization>] "
        "(e.g. Qwen/Qwen3-8B-GGUF:Q8_0, llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16).",
    )
    parser.add_argument(
        "--model-mmproj-url",
        type=str,
        metavar="URL",
        help="Direct Hugging Face URL for the mmproj file (e.g. https://huggingface.co/org/repo/resolve/main/mmproj-BF16.gguf). "
        "Only used when --model-url is a URL; with a key, give the mmproj quantization as its fourth field.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory to save the downloaded file (default: current directory).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        metavar="KEY",
        help="Hugging Face API token for authentication.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete already-present files matching the resolved patterns instead of downloading them.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if model files matching the resolved patterns are present on the filesystem.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print the total size (in bytes) of files matching the resolved patterns on Hugging Face.",
    )

    args = parser.parse_args()

    try:
        source = resolve_model_source(args.model_url, args.model_mmproj_url)
    except ValueError as exc:
        emit_json_error(str(exc))
        raise SystemExit(1) from exc

    repo_id = source["repo_id"]
    allow_pattern = source["allow_pattern"]
    mmproj_allow_pattern = source["mmproj_allow_pattern"]
    # Set only for the URL syntax; they select the single-file download path.
    url_filename = source["url_filename"]
    url_revision = source["url_revision"]
    mmproj_url_filename = source["mmproj_url_filename"]
    mmproj_url_revision = source["mmproj_url_revision"]

    if args.verbose:
        emit_json_info(f"Repository ID: {repo_id}")
        if url_filename:
            emit_json_info(f"Filename: {url_filename}")
            emit_json_info(f"Revision: {url_revision}")
            if mmproj_url_filename:
                emit_json_info(f"MMProj Filename: {mmproj_url_filename}")
                emit_json_info(f"MMProj Revision: {mmproj_url_revision}")
        else:
            if source["model_type"]:
                emit_json_info(f"Model type: {source['model_type']}")
            emit_json_info(f"Pattern: {allow_pattern}")
            if mmproj_allow_pattern:
                emit_json_info(f"MMProj pattern: {mmproj_allow_pattern}")

    if args.hf_token and args.hf_token != "":
        # huggingface_hub reads the token from HF_TOKEN; HF_HUB_TOKEN is not a name it knows.
        os.environ["HF_TOKEN"] = args.hf_token

    # Create download folder if it doesn't exist. Patter is: output_dir + / repo_id
    output_dir = f"{args.output_dir}/{repo_id}"

    if args.info:
        patterns = [allow_pattern]
        if mmproj_allow_pattern:
            patterns.append(mmproj_allow_pattern)
        matched_files = [{"file": f.path, "size": f.size} for f in list_repo_matches(repo_id, patterns) if f.size]
        total_bytes = sum(f["size"] for f in matched_files)
        print(
            json.dumps({
                "event": "stat",
                "description": f"Total download size for {repo_id}",
                "size_bytes": total_bytes,
                "size_mb": round(total_bytes / 1024 / 1024, 2),
                "files": matched_files,
            }),
            flush=True,
        )
    elif args.check:
        base = Path(output_dir)
        # A ".download" marker means a download is in progress or was interrupted
        if (base / ".download").is_file():
            emit_json_info(f"Model downloading: {repo_id}", downloading=True)
        else:
            matched = [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, allow_pattern)] if base.exists() else []
            if mmproj_allow_pattern:
                matched += [f for f in base.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, mmproj_allow_pattern)] if base.exists() else []
            if matched:
                emit_json_info(f"Model exists: {allow_pattern}", downloading=False)
            else:
                emit_json_error(f"Model does not exist: {allow_pattern}", downloading=False)
                raise SystemExit(1)
    elif args.delete:
        if args.verbose:
            emit_json_info(f"Deleting files matching '{allow_pattern}' in {output_dir}")
        delete_matched_files(output_dir, args.output_dir, allow_pattern, args.verbose)
        if mmproj_allow_pattern:
            if args.verbose:
                emit_json_info(f"Deleting mmproj files matching '{mmproj_allow_pattern}' in {output_dir}")
            delete_matched_files(output_dir, args.output_dir, mmproj_allow_pattern, args.verbose)

        if prune_emptied_repo_dir(output_dir, args.output_dir) and args.verbose:
            emit_json_info(f"Removed empty model directory: {output_dir}")

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))
    else:
        # Per-repo ".download" marker: present => prior run killed mid-download,
        # wipe and retry; absent but model files present => already complete.
        marker = Path(output_dir) / ".download"
        if marker.is_file():
            emit_json_info(f"Removing incomplete previous download: {repo_id}")
            remove_model_dir(output_dir, args.output_dir)
        elif has_model_content(output_dir):
            emit_json_info(f"Model exists: {repo_id}")
            return
        elif os.path.isdir(output_dir):
            # Bookkeeping-only leftover (e.g. killed between makedirs and the marker
            # write, or a deleted model): wipe it so the download starts clean.
            emit_json_info(f"Removing incomplete previous download: {repo_id}")
            remove_model_dir(output_dir, args.output_dir)

        # The model directory is the repo id: the download always lands in
        # <output_dir>/<repo_id>. models-list.yaml usually spells it out, but it is
        # redundant — repo_id is a substring of the model URL (and of the model key),
        # so derive it when the variable is not set rather than recording nothing.
        model_directory = os.environ.get("model_directory") or repo_id
        # Environment the metadata record is built from, with model_directory filled
        # in: it feeds both the "inputs" block and the models-list.yaml lookup that
        # identifies the model.
        metadata_env = {**os.environ, "model_directory": model_directory}

        os.makedirs(output_dir, exist_ok=True)
        write_marker(
            output_dir,
            handler="hf-handler",
            models_repository=os.environ.get("models_repository", ""),
            model_directory=model_directory,
            model_url=args.model_url or "",
        )

        emit_json_info(f"Downloading to: {os.path.abspath(output_dir)}", artifacts=[os.path.abspath(output_dir)])

        tqdm_class = JsonProgress

        try:
            if url_filename:
                # Single-file download via direct URL
                if args.verbose:
                    emit_json_info(f"Downloading file '{url_filename}' from {repo_id} (revision: {url_revision})")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=url_filename,
                    revision=url_revision,
                    local_dir=output_dir,
                    tqdm_class=tqdm_class,
                )
                if mmproj_url_filename:
                    if args.verbose:
                        emit_json_info(f"Downloading mmproj file '{mmproj_url_filename}' from {repo_id} (revision: {mmproj_url_revision})")
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=mmproj_url_filename,
                        revision=mmproj_url_revision,
                        local_dir=output_dir,
                        tqdm_class=tqdm_class,
                    )
            else:
                # Pattern-based download
                if args.verbose:
                    emit_json_info(f"Downloading model from Hugging Face repository: {repo_id} with allow pattern: {allow_pattern}")
                download_matched_files(
                    repo_id,
                    allow_pattern,
                    output_dir,
                    tqdm_class,
                    ignore_pattern="*mmproj*",
                    verbose=args.verbose,
                )

                if mmproj_allow_pattern:
                    if args.verbose:
                        emit_json_info(
                            f"Downloading mmproj model file from Hugging Face repository: {repo_id} with allow pattern: {mmproj_allow_pattern}"
                        )
                    download_matched_files(repo_id, mmproj_allow_pattern, output_dir, tqdm_class, verbose=args.verbose)
        except BaseException as exc:
            # Network/extraction errors and SIGINT/SIGTERM-driven KeyboardInterrupt
            # leave a partial repo directory; remove it before exiting.
            if os.path.isdir(output_dir):
                remove_model_dir(output_dir, args.output_dir)
            if not isinstance(exc, KeyboardInterrupt):
                # KeyboardInterrupt gets its own event from the top-level handler.
                emit_json_error(f"Download failed: {exc}")
            raise

        # Remove download caches
        cache_path = Path(output_dir) / ".cache"
        if cache_path.is_dir():
            shutil.rmtree(cache_path)

        # Generate models.ini file
        generate_models_ini(Path(args.output_dir))

        # Report the absolute path(s) of the downloaded model file(s).
        downloaded = sorted(str(p.resolve()) for p in Path(output_dir).rglob("*.gguf"))
        emit_json_info(f"Downloaded to: {output_dir}", artifacts=downloaded)

        # Record what was downloaded, then clear the in-progress marker: while the
        # marker is still there the repo directory counts as incomplete, so a crash
        # in between makes the next run retry instead of leaving it unrecorded.
        write_metadata(output_dir, handler="hf-handler", env=metadata_env)

        marker = Path(output_dir) / ".download"
        if marker.exists():
            marker.unlink()


if __name__ == "__main__":
    install_signal_handlers()
    try:
        main()
    except KeyboardInterrupt:
        emit_json_error("Download interrupted by signal; partial files removed")
        raise SystemExit(130)
