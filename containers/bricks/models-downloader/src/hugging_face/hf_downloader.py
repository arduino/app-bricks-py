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
    #    quantization, at the tip of the default branch. model_type is optional, and
    #    so is the quantization — a bare repository defaults to Q4_0.
    hf_downloader --model-url [<model_type>:]<repo_id>[:<quantization>[:<mmproj_quantization>]]

A leading ``<scheme>://`` selects form 1; anything else is parsed as a key.
The quantization field also accepts a full file name or an explicit glob, so a single
file can be pinned by name without a URL. When nothing in the repository matches, the
error lists the GGUF files that are there.

What is accepted
----------------
``model_url`` is host configuration: whoever installs an app can point it anywhere, so it
is validated in two steps instead of being taken at face value.

Form 1 must be a canonical Hugging Face model file URL and nothing else — host
huggingface.co with no userinfo or port, path
``/<owner>/<repo>/{resolve,blob}/<revision>/<file>.gguf``, names the Hub could have issued,
and a repo-relative ``.gguf`` path that cannot climb out of the output directory
(``parse_hf_url``). Form 2 gets the same check on its repository id, which also becomes a
directory under the models mount.

Then the Hub is asked what the target actually is (``validate_hub_source``): the repository
must exist and be readable *anonymously* — a private or gated repository is refused, and a
token found in the environment cannot change that — and a URL, which pins one exact file at
one exact commit, must name a file that is really in the repository. Only ``--hf-token``,
which the operator has to pass explicitly, opens gated and private repositories. ``--check``
and ``--delete`` skip this step: they only read the filesystem and stay usable offline.

Key options
-----------
--output-dir DIR        Destination directory (default: current directory).
                        Files are saved under ``<output-dir>/<repo-id>/``.
--hf-token KEY          Hugging Face API token. Without it only public, non-gated
                        repositories can be downloaded.
--verbose               Print resolved parameters before downloading.

Where the files land
--------------------
Everything is downloaded into ``<output-dir>/<repo-id>/``, so the quantizations of one
repository share a directory and coexist there. Only the files a request actually names
decide whether it is already installed: a ``Q4_0`` on disk does not make ``Q3_K_S``
present, and asking for the second one downloads it next to the first instead of
reporting the repository as complete. The cleanup paths honour the same rule — an
interrupted or failed download only discards the files it was fetching, never a sibling
quantization that finished earlier.

After all files are downloaded, ``models.ini`` is written to ``<output-dir>``
mapping each model name to its GGUF path (and mmproj path where present); names
follow ``common/gguf_naming.py`` — the file stem for models the curated catalog
declares, the output-dir-relative path for ad-hoc downloads.
"""

import fnmatch
import os
import re
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    DisabledRepoError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile
import argparse
import configparser
from collections import ChainMap
from pathlib import Path
from tqdm.auto import tqdm
from urllib.parse import unquote, urlsplit
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.download_marker import MARKER_NAME, read_marker, write_marker
from common.gguf_naming import catalog_gguf_declarations, gguf_model_name
from common.model_metadata import is_bookkeeping_name, write_metadata

# Quantization used when a model key names only a repository. Q4_0 is the quantization
# every llama.cpp GGUF repository publishes and the one all curated entries use.
DEFAULT_QUANTIZATION = "Q4_0"

# The only host a model may come from. The whole netloc is compared against it, so a
# lookalike domain ("huggingface.co.example.com"), embedded credentials that hide the real
# host ("https://huggingface.co@example.com/...") or an explicit port are all rejected.
HF_NETLOC = "huggingface.co"

# Top-level Hub sections that are not repository owners. Recognised only to say so: they
# would otherwise be reported as a malformed path.
HF_NON_MODEL_SECTIONS = frozenset({"datasets", "spaces", "collections", "papers", "blog", "docs", "posts", "models"})

# Owners and repository names, as the Hub itself accepts them: an alphanumeric first
# character, then letters, digits, '-', '_' or '.', up to 96 characters.
HF_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")

# A branch, tag or commit sha. Slashes are excluded because in a file URL the revision is
# a single path segment: "refs/pr/1" could not be told apart from a directory name.
HF_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")

# The path segments that introduce the revision and file part of a file URL.
HF_FILE_MARKERS = ("resolve", "blob")

URL_FORMAT_HINT = "Expected format: https://huggingface.co/<owner>/<repo>/resolve/<revision>/<file>.gguf (/blob/ also works)"


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


def model_files(output_dir: str) -> list[Path]:
    """The downloaded files under *output_dir*, ignoring bookkeeping entries.

    The ".download" marker, ".arduino_metadata.yaml" and huggingface_hub's ".cache"
    tree are not model content: a directory holding only those is a leftover from an
    interrupted or deleted download, not an installed model.
    """
    base = Path(output_dir)
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file() and not is_bookkeeping_name(p.name) and ".cache" not in p.parts)


def has_model_content(output_dir: str) -> bool:
    """True when *output_dir* holds a downloaded file of any kind (see ``model_files``)."""
    return bool(model_files(output_dir))


def matching_files(output_dir: str, patterns: list[str]) -> list[Path]:
    """The files of *output_dir* named by any of *patterns*.

    Matched the way ``matches_pattern`` matches them on the Hub — against the
    repository-relative path as well as the file name — so the quantization of a
    repository that nests its files in per-quantization folders is recognised on disk
    under the same pattern that selected it for download.
    """
    base = Path(output_dir)
    return [p for p in model_files(output_dir) if any(matches_pattern(p.relative_to(base).as_posix(), pattern) for pattern in patterns)]


def is_installed(output_dir: str, patterns: list[str]) -> bool:
    """True when every pattern of the request is satisfied inside *output_dir*.

    A repository directory holds every quantization ever downloaded from that repository,
    so "is this model here" can only be answered against the files the request names —
    ``every`` pattern, not any: a model whose mmproj companion is missing is not installed,
    and neither is a ``Q3_K_S`` in a directory that only holds the ``Q4_0``.
    """
    return bool(patterns) and all(matching_files(output_dir, [pattern]) for pattern in patterns)


def interrupted_patterns(marker_path: Path) -> list[str]:
    """The patterns the download *behind the marker* was fetching.

    Which is not the same thing as the patterns of the request that finds the marker:
    the repository directory is shared by every quantization, so a marker left there by
    an interrupted Q3_K_S says nothing about the Q4_0 asked for next, and cleaning up
    with the caller's patterns would delete a model that is installed and complete.

    Empty when the marker records no ``file_patterns`` — a legacy marker, or one written
    before the field existed — because then there is no way to tell which quantization it
    stood for. Deleting nothing keeps the ".cache" partials and the marker itself going
    (that much is scratch either way) while leaving complete files where they are: a
    truncated file gets caught on load and can be fetched again, whereas an installed
    model deleted on an offline board is gone for good.
    """
    patterns = (read_marker(str(marker_path)) or {}).get("file_patterns")
    if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
        return patterns
    return []


def discard_incomplete_download(output_dir: str, base_dir: str, patterns: list[str]) -> None:
    """Undo an interrupted download of *patterns*, keeping the rest of *output_dir*.

    A killed or failed run leaves the ".download" marker behind and huggingface_hub's
    partial bytes under "<output_dir>/.cache"; the next run starts the transfer over
    rather than resuming it, so both go, along with any file *patterns* names — a file
    that did land is not necessarily one the killed process finished writing.

    *patterns* must be the interrupted download's own (``interrupted_patterns``), never
    the patterns of whichever request happens to be running the cleanup.

    The repository directory is shared by every quantization of the repository, though, so
    it is only removed outright when nothing else lives in it. A sibling quantization
    downloaded earlier is a complete model and must survive both the deletion and the
    parent-pruning ``remove_model_dir`` does.
    """
    requested = set(matching_files(output_dir, patterns))
    if not [p for p in model_files(output_dir) if p not in requested]:
        remove_model_dir(output_dir, base_dir)
        return
    shutil.rmtree(Path(output_dir) / ".cache", ignore_errors=True)
    for path in requested:
        path.unlink()
    # The marker would otherwise keep flagging the surviving models as in progress.
    (Path(output_dir) / MARKER_NAME).unlink(missing_ok=True)


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


def invalid_url_error(url: str, reason: str) -> ValueError:
    """Build the one error every URL rejection raises, so callers can match on it."""
    return ValueError(f"Invalid Hugging Face URL: {url}\n{reason}\n{URL_FORMAT_HINT}")


def validate_repo_id(repo_id: str) -> None:
    """Check *repo_id* is a name the Hub could have issued: ``<owner>/<repo>`` or ``<repo>``.

    Besides rejecting nonsense before it reaches the network, this is what keeps the repo id
    safe to use as a path: every command derives ``<output-dir>/<repo_id>`` from it, and
    ``--delete`` removes that directory tree, so a value such as ``../../etc`` must never
    get that far.

    Raises:
        ValueError: when the id has more than two parts, or a part is empty or carries
            characters no Hub name can contain.
    """
    parts = repo_id.split("/")
    if len(parts) > 2 or not all(HF_NAME_RE.fullmatch(part) for part in parts):
        raise ValueError(
            f"Invalid Hugging Face repository id: '{repo_id}'\n"
            "Expected '<owner>/<repo>' (or '<repo>' for a canonical model), where each part starts with a "
            "letter or a digit and holds only letters, digits, '-', '_' and '.'"
        )


def validate_repo_file_path(url: str, filename: str) -> None:
    """Check *filename* is a repo-relative path to a GGUF file, and only that.

    The path is joined onto the output directory by ``hf_hub_download``, so a traversing
    segment would write outside it — hence no ``.``/``..``/empty segment, no backslash (a
    directory separator once the path reaches a Windows host) and no control character.
    GGUF is also the only format this downloader can install: ``models.ini`` indexes
    ``*.gguf`` and llama.cpp loads nothing else, so a URL naming a README or a safetensors
    file is a configuration mistake worth reporting rather than a download worth starting.
    """
    for segment in filename.split("/"):
        if segment in ("", ".", ".."):
            raise invalid_url_error(url, f"'{filename}' is not a file path inside the repository.")
        if "\\" in segment or any(ord(char) < 32 for char in segment):
            raise invalid_url_error(url, f"'{filename}' contains characters that cannot appear in a repository file name.")
    if not filename.lower().endswith(".gguf"):
        raise invalid_url_error(url, f"'{filename}' is not a GGUF file; only .gguf model files can be downloaded.")


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """Parse a canonical Hugging Face file URL into ``(repo_id, filename, revision)``.

    Accepted, and nothing else::

        https://huggingface.co/<owner>/<repo>/{resolve,blob}/<revision>/<path/to/file>.gguf
        https://huggingface.co/<repo>/{resolve,blob}/<revision>/<path/to/file>.gguf   # canonical repos

    The URL is host configuration — whoever installs the app sets ``model_url`` — so it is
    validated rather than pattern matched: the host must be huggingface.co itself, the path
    must have the shape above, owner, repository and revision must be names the Hub could
    have issued, and the file must be a GGUF (see ``validate_repo_file_path``). Percent
    escapes are decoded before those checks, so a ``..`` written as ``%2e%2e`` is caught
    too. A query string or fragment is dropped: ``?download=true`` is what the Hub's own
    download button produces, and neither part names the file.

    Raises:
        ValueError: when the URL is not a canonical Hugging Face model file URL.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise invalid_url_error(url, f"Unsupported scheme '{parts.scheme}': only http(s) URLs can be downloaded.")
    # netloc, not hostname: userinfo and port must be absent, not merely ignored.
    if parts.netloc.lower() != HF_NETLOC:
        raise invalid_url_error(url, f"'{parts.netloc}' is not {HF_NETLOC}: models can only be downloaded from Hugging Face.")

    segments = [unquote(segment) for segment in parts.path.split("/") if segment]
    if segments and segments[0] in HF_NON_MODEL_SECTIONS:
        raise invalid_url_error(url, f"'/{segments[0]}/' is a Hugging Face section, not a model repository owner.")

    # The marker separates the repo id from the revision, so its position gives the id: one
    # segment before it for a canonical model ("bert-base-uncased"), two for "<owner>/<repo>".
    marker = next((i for i, segment in enumerate(segments) if segment in HF_FILE_MARKERS), -1)
    if marker not in (1, 2) or len(segments) < marker + 3:
        raise invalid_url_error(url, "The path does not name a repository, a revision and a file.")

    repo_id = "/".join(segments[:marker])
    revision = segments[marker + 1]
    filename = "/".join(segments[marker + 2 :])
    validate_repo_id(repo_id)
    if not HF_REVISION_RE.fullmatch(revision):
        raise invalid_url_error(url, f"'{revision}' is not a branch, tag or commit sha.")
    validate_repo_file_path(url, filename)
    return repo_id, filename, revision


def parse_model_key(model_key: str) -> tuple[str, str, str, str | None]:
    """Parse a model key into ``(model_type, repo_id, quantization, mmproj_quantization)``.

    Accepted forms, colon-separated::

        <repo_id>                                                   # quantization defaults to Q4_0
        <repo_id>:<quantization>                                    # llama.cpp -hf style
        <model_type>:<repo_id>:<quantization>
        <model_type>:<repo_id>:<quantization>:<mmproj_quantization>

    ``model_type`` is optional and purely informative — nothing selects on it — so a
    two-field key is accepted and reads like llama.cpp's ``-hf Qwen/Qwen3-8B-GGUF:Q8_0``.
    The field count alone disambiguates: a lone field can only be a repository, and a
    pair can only be repository plus quantization, since ``model_type`` never appears
    without one. Callers detect the defaulted quantization by the absence of a ``:``
    and should report it — a silently substituted quantization would be surprising.

    Raises:
        ValueError: when there are more than four fields, or repo_id/quantization are empty.
    """
    parts = model_key.split(":")
    if len(parts) == 1:
        model_type, repo_id, quantization, mmproj_quantization = "", parts[0], DEFAULT_QUANTIZATION, None
    elif len(parts) == 2:
        model_type, repo_id, quantization, mmproj_quantization = "", parts[0], parts[1], None
    elif len(parts) == 3:
        model_type, repo_id, quantization, mmproj_quantization = parts[0], parts[1], parts[2], None
    elif len(parts) == 4:
        model_type, repo_id, quantization, mmproj_quantization = parts
    else:
        raise ValueError(
            f"Invalid model key: {model_key}\n"
            "Expected format: [<model_type>:]<repo_id>[:<quantization>[:<mmproj_quantization>]] "
            "(e.g. unsloth/Qwen3-0.6B-GGUF, Qwen/Qwen3-8B-GGUF:Q8_0 or llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0)"
        )
    if repo_id == "":
        raise ValueError("repo_id cannot be empty")
    if quantization == "":
        raise ValueError("quantization cannot be empty")
    # The key is host configuration just as a URL is, and its repo id ends up as a path
    # under the models directory, so it gets the same check the URL form gets.
    validate_repo_id(repo_id)
    return model_type, repo_id, quantization, mmproj_quantization or None


def is_hf_url(spec: str) -> bool:
    """True when *spec* is a URL rather than a compact model key.

    Checked before any ``:`` splitting, since ``https://...`` would otherwise parse
    as a two-field key with repo_id ``https``. Any scheme counts, not only http(s): a
    ``ftp://`` or ``file://`` spec is a URL the user got wrong, and reporting that is
    more useful than parsing it as a repository named "ftp".
    """
    return re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", spec) is not None


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
            "([<model_type>:]<repo_id>[:<quantization>[:<mmproj_quantization>]], e.g. unsloth/Qwen3-0.6B-GGUF "
            f"which defaults to {DEFAULT_QUANTIZATION}, or Qwen/Qwen3-8B-GGUF:Q8_0)"
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
        "quantization": None,
        "quantization_defaulted": False,
    }

    if is_hf_url(model_url):
        repo_id, url_filename, url_revision = parse_hf_url(model_url)
        source["repo_id"] = repo_id
        source["url_filename"] = url_filename
        source["url_revision"] = url_revision
        # Basename as the pattern, so check/delete/info work the same as for a key.
        source["allow_pattern"] = url_filename.split("/")[-1]
        if model_mmproj_url:
            mmproj_repo_id, mmproj_filename, mmproj_revision = parse_hf_url(model_mmproj_url)
            # The mmproj file is fetched from the model's own repository, so a URL naming a
            # different one does not do what it says: it would either download a same-named
            # file from the model repository or fail with a puzzling "not found".
            if mmproj_repo_id != repo_id:
                raise ValueError(
                    f"The mmproj URL names repository '{mmproj_repo_id}', but the model comes from '{repo_id}'.\n"
                    "Both files must live in the same Hugging Face repository."
                )
            source["mmproj_url_filename"] = mmproj_filename
            source["mmproj_url_revision"] = mmproj_revision
            source["mmproj_allow_pattern"] = mmproj_filename.split("/")[-1]
        return source

    model_type, repo_id, quantization, mmproj_quantization = parse_model_key(model_url)
    source["model_type"] = model_type
    source["repo_id"] = repo_id
    source["quantization"] = quantization
    # No colon means the key named only a repository, so the quantization above is the
    # default rather than a choice the caller made. main() reports it.
    source["quantization_defaulted"] = ":" not in model_url
    source["allow_pattern"] = gguf_pattern(quantization)
    if mmproj_quantization:
        source["mmproj_allow_pattern"] = gguf_pattern(mmproj_quantization, mmproj=True)
    return source


def source_patterns(source: dict) -> list[str]:
    """The GGUF patterns *source* names: the model's, plus the mmproj's when it has one.

    This is the whole of what a request asks for, and what the filesystem is asked about:
    the same list decides whether the model is installed, what an interrupted download
    has to discard, and how big the download is.
    """
    return [pattern for pattern in (source["allow_pattern"], source["mmproj_allow_pattern"]) if pattern]


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


def public_repo_files(repo_id: str, revision: str | None = None, token: str | None = None) -> set[str]:
    """List the files of *repo_id* at *revision*, or explain why it cannot be downloaded.

    The call is unauthenticated unless *token* is given, so a private repository answers
    404 exactly as it does for an anonymous visitor: a token that merely happens to sit in
    the environment (``HF_TOKEN``, a cached login) cannot turn a repository nobody else can
    read into a valid model source. An explicit ``--hf-token`` is the one way in, because
    passing it is a deliberate act of whoever runs the downloader rather than something a
    model URL can arrange by itself.

    Returns:
        The repo-relative paths of every file in the repository at *revision*.

    Raises:
        ValueError: when the repository does not exist, is private, gated or disabled, when
            *revision* is not in it, or when the Hub cannot be reached to tell.
    """
    try:
        info = HfApi().model_info(repo_id, revision=revision, token=token or False)
    except GatedRepoError as exc:  # a subclass of RepositoryNotFoundError: must come first
        raise ValueError(
            f"Hugging Face repository '{repo_id}' is gated: its files are only served after its conditions "
            "are accepted on huggingface.co. Only freely downloadable models can be configured."
        ) from exc
    except RevisionNotFoundError as exc:
        raise ValueError(f"Revision '{revision}' does not exist in Hugging Face repository '{repo_id}'.") from exc
    except RepositoryNotFoundError as exc:
        raise ValueError(
            f"Hugging Face model repository '{repo_id}' does not exist, or is not public. Only public model repositories can be downloaded."
        ) from exc
    except DisabledRepoError as exc:
        raise ValueError(f"Hugging Face repository '{repo_id}' has been disabled by its authors.") from exc
    except Exception as exc:  # noqa: BLE001 - HTTP, DNS and proxy failures all land here
        raise ValueError(f"Could not verify Hugging Face repository '{repo_id}': {exc}") from exc

    # The 404 above already covers a private repository seen anonymously; these fields are
    # what remains once --hf-token makes it visible, and are checked so the token widens
    # access to gated models the operator has accepted, not to repositories at large.
    if info.private:
        raise ValueError(f"Hugging Face repository '{repo_id}' is private. Only public model repositories can be downloaded.")
    if info.gated:
        raise ValueError(
            f"Hugging Face repository '{repo_id}' is gated: its files are only served after its conditions "
            "are accepted on huggingface.co. Only freely downloadable models can be configured."
        )
    if info.disabled:
        raise ValueError(f"Hugging Face repository '{repo_id}' has been disabled by its authors.")
    return {sibling.rfilename for sibling in info.siblings or []}


def missing_file_message(repo_id: str, filename: str, revision: str | None, available: set[str]) -> str:
    """Explain that *filename* is not in the repository, listing the GGUF files that are.

    The URL syntax pins one exact file, so a typo or a file renamed by a new commit ends as
    "not found" with nothing to act on. The listing is already in hand from the validation
    call, so naming the alternatives costs nothing.
    """
    where = f"Hugging Face repository '{repo_id}' at revision '{revision}'" if revision else f"Hugging Face repository '{repo_id}'"
    ggufs = sorted(path for path in available if path.lower().endswith(".gguf"))
    if not ggufs:
        return f"File '{filename}' does not exist in {where}, which contains no GGUF files at all."
    return f"File '{filename}' does not exist in {where}. Available GGUF files: {', '.join(ggufs)}"


def validate_hub_source(source: dict, token: str | None = None) -> None:
    """Confirm on the Hub that *source* names a model that can actually be downloaded.

    Parsing only proves the model URL or key is well formed; it says nothing about what it
    points at. This is the other half: the repository must exist and be publicly readable
    (see ``public_repo_files``), and a URL — which pins one exact file at one exact commit —
    must name a file that is really there. Applies to both syntaxes, since a compact key
    names a repository just as freely as a URL does.

    Only the file existence check is skipped for a key: the quantization is a pattern, and
    ``download_matched_files`` already fails with the repository's GGUF listing when nothing
    matches it.

    Raises:
        ValueError: with a message meant for the user, on anything that makes the model
            unusable.
    """
    repo_id = source["repo_id"]
    # None for the compact key syntax, which downloads from the default branch.
    revision = source["url_revision"]
    files = public_repo_files(repo_id, revision, token)
    for filename, file_revision in (
        (source["url_filename"], revision),
        (source["mmproj_url_filename"], source["mmproj_url_revision"]),
    ):
        if not filename:
            continue
        # The two URLs may pin different commits of the same repository.
        present = files if file_revision == revision else public_repo_files(repo_id, file_revision, token)
        if filename not in present:
            raise ValueError(missing_file_message(repo_id, filename, file_revision, present))


def fallback_model_id(model_type: str, downloaded: list[str], models_dir: str) -> str | None:
    """Name a model that no models-list.yaml entry declares, from the files fetched.

    Built as ``<namespace>:<model name>`` with the naming ``list_models.py`` and
    models.ini share (``gguf_model_name`` against the catalog baked into the image),
    so the record and the listing agree on what to call an ad-hoc download. mmproj
    files belong to the main GGUF and never name the model.
    """
    main_gguf = next((p for p in sorted(downloaded) if "mmproj" not in os.path.basename(p)), None)
    if not main_gguf:
        return None
    try:
        rel = Path(main_gguf).resolve().relative_to(Path(models_dir).resolve()).as_posix()
        name = gguf_model_name(rel, catalog_gguf_declarations())
    except ValueError:  # not under models_dir: name it by its stem alone
        name = Path(main_gguf).stem
    # The key's model_type is the namespace when given; llamacpp is where GGUF models
    # live, and the prefix list_models.py uses (see common/gguf_naming.py).
    return f"{model_type or 'llamacpp'}:{name}"


def no_match_message(repo_id: str, pattern: str) -> str:
    """Explain that nothing matched *pattern*, listing the GGUF files the repo does have.

    Asking the Hub what is actually there turns "no file matching '*Q4_0*.gguf'" into an
    actionable message — which matters most when the quantization was defaulted rather
    than chosen. Runs only on the failure path, and degrades to the bare statement if
    the extra listing call fails.
    """
    message = f"No file matching '{pattern}' found in repository '{repo_id}'."
    try:
        available = sorted(f.path for f in list_repo_matches(repo_id, ["*.gguf"]))
    except Exception:  # noqa: BLE001 - improving an error message must not raise a new one
        return message
    if not available:
        return f"{message} The repository contains no GGUF files at all."
    return f"{message} Available GGUF files: {', '.join(available)}"


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
        raise FileNotFoundError(no_match_message(repo_id, allow_pattern))
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
        if d == models_base_path:
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


def generate_models_ini(models_dir: Path, declarations=None):
    """Write the models.ini indexing every GGUF under *models_dir*.

    Sections are the names ``gguf_model_name`` assigns — the file stem for a model
    the curated catalog declares, the models_dir-relative path for an ad-hoc one —
    so every file keeps its own section instead of one silently shadowing the
    other, and each section matches the listing id of the same file. *declarations*
    defaults to the catalog baked into the image.
    """
    if declarations is None:
        declarations = catalog_gguf_declarations()
    config = configparser.ConfigParser()

    gguf_files = [p for p in sorted(models_dir.rglob("*.gguf")) if "mmproj" not in p.name]
    for gguf_file in gguf_files:
        section = gguf_model_name(gguf_file.relative_to(models_dir).as_posix(), declarations)
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


def validate_hub_source_or_exit(source: dict, token: str | None = None) -> None:
    """``validate_hub_source``, reported as an error event and a non-zero exit.

    Called from the commands that talk to the Hub anyway (``--info`` and the download), so
    ``--check`` and ``--delete`` keep working on a box with no network: they only read the
    filesystem, and refusing to report an installed model because the Hub is unreachable
    would be worse than not re-verifying it.
    """
    try:
        validate_hub_source(source, token)
    except ValueError as exc:
        emit_json_error(str(exc))
        raise SystemExit(1) from exc


def main():
    parser = argparse.ArgumentParser(description="Download an Hugging Face model via HF download API")
    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        metavar="URL_OR_KEY",
        help="The model to download, as either a Hugging Face file URL "
        "(e.g. https://huggingface.co/org/repo/blob/<revision>/model.gguf; /resolve/ works too) "
        "or a compact key [<model_type>:]<repo_id>[:<quantization>[:<mmproj_quantization>]] "
        "(e.g. unsloth/Qwen3-0.6B-GGUF which defaults to Q4_0, Qwen/Qwen3-8B-GGUF:Q8_0, "
        "llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16).",
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
        help="Hugging Face API token. Without it, only public and non-gated repositories can be downloaded: "
        "the access check is made anonymously, so a token present in the environment does not widen it.",
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
    # Everything this run is about, and the only thing the repository directory is
    # queried for: the other quantizations sharing it belong to other requests.
    patterns = source_patterns(source)
    # Set only for the URL syntax; they select the single-file download path.
    url_filename = source["url_filename"]
    url_revision = source["url_revision"]
    mmproj_url_filename = source["mmproj_url_filename"]
    mmproj_url_revision = source["mmproj_url_revision"]

    # Always reported, not only under --verbose: the caller named a repository without
    # a quantization, so they need to see which one they are getting.
    if source["quantization_defaulted"]:
        emit_json_info(f"No quantization given for '{repo_id}', defaulting to {DEFAULT_QUANTIZATION}. Specify another as '{repo_id}:<quantization>'.")

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
        validate_hub_source_or_exit(source, args.hf_token)
        matched_files = [{"file": f.path, "size": f.size} for f in list_repo_matches(repo_id, patterns) if f.size]
        if not matched_files:
            # Reporting a 0-byte total would read as "this model is free to download".
            emit_json_error(no_match_message(repo_id, allow_pattern))
            raise SystemExit(1)
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
        # Files first, marker second: the marker is per repository, but a repository
        # directory holds several quantizations, so a download in progress there says
        # nothing about the one being asked for — which may well be installed already.
        if is_installed(output_dir, patterns):
            emit_json_info(f"Model exists: {allow_pattern}", downloading=False)
        elif (Path(output_dir) / MARKER_NAME).is_file():
            # A ".download" marker means a download is in progress or was interrupted
            emit_json_info(f"Model downloading: {repo_id}", downloading=True)
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
        # Per-repo ".download" marker: present => prior run killed mid-download, discard
        # what it left and retry; absent but the requested files present => complete.
        # Marker first, files second — the reverse of --check, because the leftovers of
        # the interrupted run have to go before the directory can be judged.
        marker = Path(output_dir) / MARKER_NAME
        if marker.is_file():
            emit_json_info(f"Removing incomplete previous download: {repo_id}")
            discard_incomplete_download(output_dir, args.output_dir, interrupted_patterns(marker))
        if is_installed(output_dir, patterns):
            installed = ", ".join(p.name for p in matching_files(output_dir, patterns))
            emit_json_info(f"Model exists: {repo_id} ({installed})")
            return
        if os.path.isdir(output_dir) and not has_model_content(output_dir):
            # Bookkeeping-only leftover (e.g. killed between makedirs and the marker
            # write, or a deleted model): wipe it so the download starts clean.
            emit_json_info(f"Removing incomplete previous download: {repo_id}")
            remove_model_dir(output_dir, args.output_dir)
        # Anything else the directory holds is another quantization of the same
        # repository: the requested files are downloaded alongside it.

        # Nothing has been written yet, and the model URL or key comes from the host
        # configuration: check what it points at before creating a directory for it.
        validate_hub_source_or_exit(source, args.hf_token)

        # The model directory is the repo id: the download always lands in
        # <output_dir>/<repo_id>. models-list.yaml usually spells it out, but it is
        # redundant — repo_id is a substring of the model URL (and of the model key),
        # so derive it when the variable is not set rather than recording nothing.
        model_directory = os.environ.get("model_directory") or repo_id
        # Environment the metadata record is built from, with model_directory filled
        # in: it feeds both the "inputs" block and the models-list.yaml lookup that
        # identifies the model. ChainMap rather than {**os.environ, ...} because on
        # Windows os.environ upper-cases its keys when copied, which would drop every
        # lowercase download variable; chaining delegates the lookup instead.
        metadata_env = ChainMap({"model_directory": model_directory}, os.environ)

        os.makedirs(output_dir, exist_ok=True)
        write_marker(
            output_dir,
            handler="hf-handler",
            models_repository=os.environ.get("models_repository", ""),
            model_directory=model_directory,
            model_url=args.model_url or "",
            # Which files of a shared repository directory this download is for, so a
            # quantization already installed there is not reported as in progress.
            file_patterns=patterns,
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
            # leave a partial download behind; discard it before exiting, without
            # taking another quantization of the same repository down with it.
            if os.path.isdir(output_dir):
                discard_incomplete_download(output_dir, args.output_dir, patterns)
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

        # Report the absolute path(s) of the downloaded model file(s): the files this
        # request named, not every quantization the shared repository directory holds —
        # a sibling was not downloaded now, and must not name this model either.
        downloaded = sorted(str(p.resolve()) for p in matching_files(output_dir, patterns) if p.suffix == ".gguf")
        emit_json_info(f"Downloaded to: {output_dir}", artifacts=downloaded)

        # Record what was downloaded, then clear the in-progress marker: while the
        # marker is still there the repo directory counts as incomplete, so a crash
        # in between makes the next run retry instead of leaving it unrecorded.
        write_metadata(
            output_dir,
            handler="hf-handler",
            env=metadata_env,
            # Any repository can be downloaded without a models-list.yaml entry, so name
            # it after the file that arrived rather than leaving it unidentified.
            fallback_model_id=fallback_model_id(source["model_type"], downloaded, args.output_dir),
        )

        marker = Path(output_dir) / MARKER_NAME
        if marker.exists():
            marker.unlink()


if __name__ == "__main__":
    install_signal_handlers()
    try:
        main()
    except KeyboardInterrupt:
        emit_json_error("Download interrupted by signal; partial files removed")
        raise SystemExit(130)
