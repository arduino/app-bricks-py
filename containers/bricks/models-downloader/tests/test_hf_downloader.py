# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Hugging Face downloader.

The downloader replaces huggingface_hub's terminal progress bars with a stream of JSON
events, which only works as long as ``JsonProgress`` keeps satisfying the hooks
huggingface_hub looks for. A library upgrade already broke this once, so the contract is
tested explicitly here. The rest covers the local filesystem bookkeeping: what counts as
an installed model, and what the delete path leaves behind.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from huggingface_hub.errors import (
    DisabledRepoError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from common.download_marker import MARKER_NAME, read_marker
from common.model_metadata import METADATA_NAME
from hugging_face import hf_downloader
from hugging_face.hf_downloader import (
    JsonProgress,
    delete_matched_files,
    discard_incomplete_download,
    download_matched_files,
    fallback_model_id,
    generate_models_ini,
    gguf_pattern,
    has_model_content,
    interrupted_patterns,
    is_hf_url,
    is_installed,
    matches_pattern,
    matching_files,
    no_match_message,
    parse_hf_url,
    parse_model_key,
    prune_emptied_repo_dir,
    public_repo_files,
    resolve_model_source,
    source_patterns,
    validate_hub_source,
    validate_repo_id,
)


def read_events(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]


def test_emits_start_and_completion_events(capsys):
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(100)
    bar.close()

    events = read_events(capsys)
    # No "update" in between: it is throttled to one event per EMIT_INTERVAL.
    assert [e["event"] for e in events] == ["start", "complete"]
    assert events[-1] == {
        "event": "complete",
        "description": "model.gguf",
        "current": 100,
        "total": 100,
        "unit": "B",
        "percentage": "100.0%",
    }


def test_emits_throttled_update_events(monkeypatch, capsys):
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(25)
    bar.update(25)

    events = read_events(capsys)
    assert [e["event"] for e in events] == ["start", "update", "update"]
    assert [e["percentage"] for e in events] == ["0.0%", "25.0%", "50.0%"]


def test_incomplete_transfer_does_not_report_completion(capsys):
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update(40)
    bar.close()

    assert [e["event"] for e in read_events(capsys)] == ["start"]


@pytest.mark.parametrize(
    "desc",
    ["model.gguf: reconstructing file", "model.gguf: downloading bytes", "model.gguf: "],
)
def test_description_drops_progress_bar_decorations(desc, capsys):
    JsonProgress(desc=desc, total=100, unit="B")

    assert read_events(capsys)[0]["description"] == "model.gguf"


def test_network_bytes_drive_progress_while_disk_writes_lag(monkeypatch, capsys):
    """Xet flushes to disk in bursts, so progress must follow the network counter.

    Reporting only bytes written to disk leaves the percentage frozen at 0% for tens of MB.
    """
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    read_events(capsys)  # drop the "start" event

    bar.update_transfer(40)
    bar.set_transfer_postfix_str("1.00MB/s")

    assert bar.n == 0  # nothing written to disk yet
    assert read_events(capsys) == [
        {
            "event": "update",
            "description": "model.gguf",
            "current": 40,
            "total": 100,
            "unit": "B",
            "percentage": "40.0%",
        }
    ]


def test_progress_never_exceeds_the_file_size(monkeypatch, capsys):
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    read_events(capsys)

    bar.update_transfer(150)

    assert read_events(capsys)[-1]["percentage"] == "100.0%"


def test_completion_follows_disk_writes_not_the_network(monkeypatch, capsys):
    """Cached Xet chunks make network bytes end below the file size: they cannot signal the end."""
    monkeypatch.setattr(JsonProgress, "EMIT_INTERVAL", 0)
    bar = JsonProgress(desc="model.gguf", total=100, unit="B")
    bar.update_transfer(60)
    bar.update(100)  # all bytes flushed to disk
    bar.close()

    events = read_events(capsys)
    assert events[-1]["event"] == "complete"
    assert events[-1]["current"] == 100


def test_xet_download_routes_both_counters_into_a_single_bar():
    """The Xet downloader must not fall back to its own bar for network transfer.

    huggingface_hub reports Xet downloads with two bars and honours ``tqdm_class`` for the
    reconstruction one only -- unless the class also implements ``update_transfer``, in
    which case both counters share one object. Without that hook a real progress bar is
    printed to the terminal alongside the JSON events.
    """
    reporting = pytest.importorskip("huggingface_hub.utils._xet_progress_reporting")
    if not hasattr(reporting, "XetDownloadProgressReporter"):
        pytest.skip("huggingface_hub predates the dual-bar Xet reporter")

    reporter = reporting.XetDownloadProgressReporter(
        reconstruction_desc="model.gguf: reconstructing file",
        transfer_desc="model.gguf: downloading bytes",
        total=100,
        log_level=20,
        tqdm_class=JsonProgress,
    )
    try:
        assert isinstance(reporter.reconstruction_bar, JsonProgress)
        assert reporter.transfer_bar is reporter.reconstruction_bar
    finally:
        reporter.close()


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("model-Q4_0.gguf", "*Q4_0*.gguf", True),
        ("UD-Q4_0/model-Q4_0.gguf", "*Q4_0*.gguf", True),  # nested per-quantization folder
        ("mmproj-F16.gguf", "*mmproj*", True),
        ("model-Q4_0.gguf", "*Q8_0*.gguf", False),
    ],
)
def test_matches_pattern(path, pattern, expected):
    assert matches_pattern(path, pattern) is expected


# --------------------------------------------------------------------------- #
# parse_model_key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # One field: a bare repository, quantization defaults to Q4_0.
        ("unsloth/Qwen3-0.6B-GGUF", ("", "unsloth/Qwen3-0.6B-GGUF", "Q4_0", None)),
        # Two fields: no model_type, llama.cpp's "-hf <repo>:<quant>" form.
        ("Qwen/Qwen3-8B-GGUF:Q8_0", ("", "Qwen/Qwen3-8B-GGUF", "Q8_0", None)),
        ("unsloth/gemma-4-E2B-it-GGUF:Q4_0", ("", "unsloth/gemma-4-E2B-it-GGUF", "Q4_0", None)),
        # Three fields: the historical form, still accepted.
        ("llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0", ("llamacpp", "Qwen/Qwen3-8B-GGUF", "Q8_0", None)),
        # Four fields: with an mmproj quantization.
        ("llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16", ("llamacpp", "unsloth/gemma-4-E4B-it-GGUF", "Q4_0", "BF16")),
        # A trailing empty mmproj field means "no mmproj".
        ("llamacpp:org/repo:Q4_0:", ("llamacpp", "org/repo", "Q4_0", None)),
        # Repos without an org are fine: the field count decides, not the "/".
        ("bert-base-uncased:Q8_0", ("", "bert-base-uncased", "Q8_0", None)),
    ],
)
def test_parse_model_key(key, expected):
    assert parse_model_key(key) == expected


def test_parse_model_key_rejects_too_many_fields():
    with pytest.raises(ValueError, match="Invalid model key"):
        parse_model_key("llamacpp:org/repo:Q4_0:BF16:extra")


@pytest.mark.parametrize("key", ["", ":Q4_0"])
def test_parse_model_key_rejects_empty_repo_id(key):
    with pytest.raises(ValueError, match="repo_id cannot be empty"):
        parse_model_key(key)


def test_parse_model_key_rejects_empty_quantization():
    with pytest.raises(ValueError, match="quantization cannot be empty"):
        parse_model_key("org/repo:")


# --------------------------------------------------------------------------- #
# is_hf_url / gguf_pattern
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("https://huggingface.co/org/repo/blob/main/m.gguf", True),
        ("http://huggingface.co/org/repo/blob/main/m.gguf", True),
        ("Qwen/Qwen3-8B-GGUF:Q8_0", False),
        ("llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0", False),
    ],
)
def test_is_hf_url(spec, expected):
    assert is_hf_url(spec) is expected


@pytest.mark.parametrize(
    ("spec", "mmproj", "expected"),
    [
        # A bare quantization is widened and anchored to .gguf.
        ("Q4_0", False, "*Q4_0*.gguf"),
        ("Q4_0", True, "*mmproj*Q4_0*.gguf"),
        # A full file name pins one specific file.
        ("gemma-4-E2B-it-Q4_0.gguf", False, "gemma-4-E2B-it-Q4_0.gguf"),
        ("mmproj-F16.gguf", True, "mmproj-F16.gguf"),
        # An explicit glob is left alone.
        ("*UD-Q4_0*", False, "*UD-Q4_0*"),
    ],
)
def test_gguf_pattern(spec, mmproj, expected):
    assert gguf_pattern(spec, mmproj=mmproj) == expected


# --------------------------------------------------------------------------- #
# parse_hf_url — the model URL is host configuration, so it is validated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # The canonical form, as models-list.yaml writes it.
        (
            "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/f0b45be0aac41bd6a100a4b5734cad5f67255bfb/gemma-3-1b-it-Q4_0.gguf",
            ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_0.gguf", "f0b45be0aac41bd6a100a4b5734cad5f67255bfb"),
        ),
        # /resolve/ is the download path of the same file.
        (
            "https://huggingface.co/org/repo/resolve/main/model.gguf",
            ("org/repo", "model.gguf", "main"),
        ),
        # A repository without an owner: canonical models live at the root of the Hub.
        (
            "https://huggingface.co/bert-base-uncased/resolve/main/model.gguf",
            ("bert-base-uncased", "model.gguf", "main"),
        ),
        # Files nested in a per-quantization folder.
        (
            "https://huggingface.co/org/repo/blob/main/UD-Q4_0/model-Q4_0.gguf",
            ("org/repo", "UD-Q4_0/model-Q4_0.gguf", "main"),
        ),
        # What the Hub's own download button produces, and a fragment: neither names the file.
        (
            "https://huggingface.co/org/repo/resolve/main/model.gguf?download=true#anchor",
            ("org/repo", "model.gguf", "main"),
        ),
        # Percent escapes are decoded, so the file name matches what the repository holds.
        (
            "https://huggingface.co/org/repo/resolve/main/model%20final.gguf",
            ("org/repo", "model final.gguf", "main"),
        ),
        # The host is case-insensitive, as hosts are.
        (
            "https://HuggingFace.CO/org/repo/resolve/main/model.gguf",
            ("org/repo", "model.gguf", "main"),
        ),
    ],
)
def test_parse_hf_url_accepts_canonical_urls(url, expected):
    assert parse_hf_url(url) == expected


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        # Another host entirely.
        ("https://example.com/org/repo/resolve/main/model.gguf", "not huggingface.co"),
        # A lookalike domain that a prefix match would have accepted.
        ("https://huggingface.co.example.com/org/repo/resolve/main/model.gguf", "not huggingface.co"),
        # Credentials, which put the real host after the '@'.
        ("https://huggingface.co@example.com/org/repo/resolve/main/model.gguf", "not huggingface.co"),
        # A port on the right host still is not the Hub.
        ("https://huggingface.co:8443/org/repo/resolve/main/model.gguf", "not huggingface.co"),
        # Schemes that are not an HTTP download.
        ("file:///etc/passwd", "Unsupported scheme"),
        ("ftp://huggingface.co/org/repo/resolve/main/model.gguf", "Unsupported scheme"),
        # Hub sections that are not model repositories.
        ("https://huggingface.co/datasets/org/repo/resolve/main/model.gguf", "not a model repository owner"),
        ("https://huggingface.co/spaces/org/repo/resolve/main/model.gguf", "not a model repository owner"),
        # Shapes that do not name a repository, a revision and a file.
        ("https://huggingface.co/org/repo", "does not name a repository"),
        ("https://huggingface.co/org/repo/blob/main", "does not name a repository"),
        ("https://huggingface.co/org/repo/tree/main/model.gguf", "does not name a repository"),
        ("https://huggingface.co/a/b/c/resolve/main/model.gguf", "does not name a repository"),
        # A revision no Hub ref could be named.
        ("https://huggingface.co/org/repo/resolve/-main/model.gguf", "is not a branch, tag or commit sha"),
        # Formats this downloader cannot install.
        ("https://huggingface.co/org/repo/resolve/main/README.md", "is not a GGUF file"),
        ("https://huggingface.co/org/repo/resolve/main/model.safetensors", "is not a GGUF file"),
    ],
)
def test_parse_hf_url_rejects_anything_else(url, reason):
    with pytest.raises(ValueError, match="Invalid Hugging Face URL") as excinfo:
        parse_hf_url(url)
    assert reason in str(excinfo.value)


@pytest.mark.parametrize(
    "url",
    [
        # A traversing file path would be written outside the output directory.
        "https://huggingface.co/org/repo/resolve/main/../../../etc/cron.d/x.gguf",
        # The same, hidden in percent escapes: the check runs on the decoded path.
        "https://huggingface.co/org/repo/resolve/main/%2e%2e/%2e%2e/etc/x.gguf",
        # A backslash is a directory separator once the path reaches a Windows host.
        "https://huggingface.co/org/repo/resolve/main/..\\..\\x.gguf",
    ],
)
def test_parse_hf_url_rejects_paths_that_escape_the_output_directory(url):
    with pytest.raises(ValueError, match="Invalid Hugging Face URL"):
        parse_hf_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/-org/repo/resolve/main/model.gguf",
        "https://huggingface.co/org/.repo/resolve/main/model.gguf",
        "https://huggingface.co/o%2Frg/repo/resolve/main/model.gguf",
    ],
)
def test_parse_hf_url_rejects_repo_ids_the_hub_could_not_have_issued(url):
    with pytest.raises(ValueError, match="Invalid Hugging Face (URL|repository id)"):
        parse_hf_url(url)


# --------------------------------------------------------------------------- #
# validate_repo_id — also guards the directory the repo id becomes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("repo_id", ["unsloth/Qwen3-0.6B-GGUF", "bert-base-uncased", "org/repo.v2", "a/b_c-d.e"])
def test_validate_repo_id_accepts_hub_names(repo_id):
    assert validate_repo_id(repo_id) is None


@pytest.mark.parametrize(
    "repo_id",
    [
        "../../etc",  # would escape the models directory --delete removes
        "..",
        "org/repo/extra",
        "org/",
        "/repo",
        "-org/repo",
        "org/re po",
        "org/repo\nid",
        "a" * 97,
    ],
)
def test_validate_repo_id_rejects_everything_else(repo_id):
    with pytest.raises(ValueError, match="Invalid Hugging Face repository id"):
        validate_repo_id(repo_id)


def test_parse_model_key_rejects_a_traversing_repo_id():
    """The key syntax reaches the same directory the URL syntax does."""
    with pytest.raises(ValueError, match="Invalid Hugging Face repository id"):
        parse_model_key("../../../etc:Q4_0")


# --------------------------------------------------------------------------- #
# resolve_model_source — one input, two syntaxes
# --------------------------------------------------------------------------- #
GEMMA_URL = "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/1894d1fc/gemma-4-E2B_q4_0-it.gguf"


def test_resolve_url_syntax():
    source = resolve_model_source(GEMMA_URL)
    assert source["repo_id"] == "google/gemma-4-E2B-it-qat-q4_0-gguf"
    assert source["url_filename"] == "gemma-4-E2B_q4_0-it.gguf"
    assert source["url_revision"] == "1894d1fc"
    # The basename doubles as the pattern, so check/delete/info behave as for a key.
    assert source["allow_pattern"] == "gemma-4-E2B_q4_0-it.gguf"
    assert source["mmproj_allow_pattern"] is None


def test_resolve_url_syntax_with_mmproj_url():
    source = resolve_model_source(GEMMA_URL, "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/1894d1fc/mmproj-BF16.gguf")
    assert source["mmproj_url_filename"] == "mmproj-BF16.gguf"
    assert source["mmproj_url_revision"] == "1894d1fc"
    assert source["mmproj_allow_pattern"] == "mmproj-BF16.gguf"


def test_resolve_key_syntax_llamacpp_style():
    source = resolve_model_source("Qwen/Qwen3-8B-GGUF:Q8_0")
    assert source["repo_id"] == "Qwen/Qwen3-8B-GGUF"
    assert source["allow_pattern"] == "*Q8_0*.gguf"
    # No URL was given, so the single-file download path stays off.
    assert source["url_filename"] is None
    assert source["url_revision"] is None


def test_resolve_key_syntax_with_model_type_and_mmproj():
    source = resolve_model_source("llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16")
    assert source["model_type"] == "llamacpp"
    assert source["repo_id"] == "unsloth/gemma-4-E4B-it-GGUF"
    assert source["allow_pattern"] == "*Q4_0*.gguf"
    assert source["mmproj_allow_pattern"] == "*mmproj*BF16*.gguf"


def test_resolve_key_syntax_pins_an_exact_file_name():
    """What the removed --model-repo-id/--model-name pair used to be for."""
    source = resolve_model_source("unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf")
    assert source["repo_id"] == "unsloth/gemma-4-E2B-it-GGUF"
    assert source["allow_pattern"] == "gemma-4-E2B-it-Q4_0.gguf"


def test_resolve_rejects_an_empty_model_url():
    with pytest.raises(ValueError, match="model_url is required"):
        resolve_model_source("")


def test_resolve_bare_repo_id_defaults_the_quantization():
    source = resolve_model_source("unsloth/Qwen3-0.6B-GGUF")
    assert source["repo_id"] == "unsloth/Qwen3-0.6B-GGUF"
    assert source["quantization"] == "Q4_0"
    assert source["allow_pattern"] == "*Q4_0*.gguf"
    # Flagged so main() can report the substitution rather than applying it silently.
    assert source["quantization_defaulted"] is True


@pytest.mark.parametrize(
    "key",
    [
        "Qwen/Qwen3-8B-GGUF:Q8_0",
        "llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0",
        "llamacpp:Qwen/Qwen3-8B-GGUF:Q8_0:BF16",
    ],
)
def test_resolve_does_not_flag_an_explicit_quantization(key):
    assert resolve_model_source(key)["quantization_defaulted"] is False


def test_resolve_url_syntax_never_flags_a_default():
    source = resolve_model_source(GEMMA_URL)
    assert source["quantization_defaulted"] is False
    assert source["quantization"] is None


def test_resolve_rejects_a_non_hf_url():
    with pytest.raises(ValueError, match="Invalid Hugging Face URL"):
        resolve_model_source("https://example.com/some/file.gguf")


def test_resolve_rejects_an_mmproj_url_from_another_repository():
    """The mmproj is fetched from the model's repository, so a second one cannot be honoured."""
    with pytest.raises(ValueError, match="Both files must live in the same Hugging Face repository"):
        resolve_model_source(GEMMA_URL, "https://huggingface.co/other/repo/blob/main/mmproj-BF16.gguf")


# --------------------------------------------------------------------------- #
# validate_hub_source — what the URL points at, not just how it is written
# --------------------------------------------------------------------------- #
class _FakeResponse:
    """The minimum ``HfHubHTTPError`` reads off a response to build itself."""

    headers: dict = {}
    request = None


def _hub_error(error_class):
    return error_class("hub says no", response=_FakeResponse())


class _FakeSibling:
    def __init__(self, rfilename):
        self.rfilename = rfilename


class _FakeModelInfo:
    def __init__(self, files=(), private=False, gated=False, disabled=False):
        self.siblings = [_FakeSibling(name) for name in files]
        self.private = private
        self.gated = gated
        self.disabled = disabled


def fake_hub(monkeypatch, info=None, error=None):
    """Replace HfApi with a stub, and return the log of model_info() calls it received."""
    calls: list[dict] = []

    class _FakeApi:
        def model_info(self, repo_id, *, revision=None, token=None):
            calls.append({"repo_id": repo_id, "revision": revision, "token": token})
            if error is not None:
                raise error
            return info

    monkeypatch.setattr("hugging_face.hf_downloader.HfApi", _FakeApi)
    return calls


def test_validate_hub_source_accepts_a_public_file(monkeypatch):
    calls = fake_hub(monkeypatch, _FakeModelInfo(files=["gemma-4-E2B_q4_0-it.gguf", "README.md"]))

    assert validate_hub_source(resolve_model_source(GEMMA_URL)) is None
    assert calls == [{"repo_id": "google/gemma-4-E2B-it-qat-q4_0-gguf", "revision": "1894d1fc", "token": False}]


def test_validate_hub_source_asks_anonymously_unless_a_token_is_given(monkeypatch):
    """A token lying around in the environment must not make a private repo downloadable."""
    calls = fake_hub(monkeypatch, _FakeModelInfo(files=["m-Q4_0.gguf"]))

    validate_hub_source(resolve_model_source("org/repo:Q4_0"))
    assert calls[-1]["token"] is False
    # The compact key downloads from the default branch, so no revision is pinned.
    assert calls[-1]["revision"] is None

    validate_hub_source(resolve_model_source("org/repo:Q4_0"), token="hf_explicit")
    assert calls[-1]["token"] == "hf_explicit"


@pytest.mark.parametrize(
    ("error_class", "reason"),
    [
        (GatedRepoError, "is gated"),
        (RepositoryNotFoundError, "does not exist, or is not public"),
        (RevisionNotFoundError, "Revision '1894d1fc' does not exist"),
        (DisabledRepoError, "has been disabled"),
    ],
)
def test_validate_hub_source_reports_why_a_repo_cannot_be_used(monkeypatch, error_class, reason):
    fake_hub(monkeypatch, error=_hub_error(error_class))

    with pytest.raises(ValueError, match=reason):
        validate_hub_source(resolve_model_source(GEMMA_URL))


@pytest.mark.parametrize(
    ("flag", "reason"),
    [({"private": True}, "is private"), ({"gated": "manual"}, "is gated"), ({"disabled": True}, "has been disabled")],
)
def test_validate_hub_source_rejects_a_repo_that_is_not_freely_downloadable(monkeypatch, flag, reason):
    """With --hf-token the repo answers instead of 404ing, so the fields are checked too."""
    fake_hub(monkeypatch, _FakeModelInfo(files=["gemma-4-E2B_q4_0-it.gguf"], **flag))

    with pytest.raises(ValueError, match=reason):
        validate_hub_source(resolve_model_source(GEMMA_URL), token="hf_explicit")


def test_validate_hub_source_reports_an_unreachable_hub_without_blaming_the_url(monkeypatch):
    fake_hub(monkeypatch, error=OSError("no network"))

    with pytest.raises(ValueError, match="Could not verify Hugging Face repository 'google/"):
        validate_hub_source(resolve_model_source(GEMMA_URL))


def test_validate_hub_source_lists_alternatives_for_a_file_that_is_not_there(monkeypatch):
    fake_hub(monkeypatch, _FakeModelInfo(files=["gemma-4-E2B_q8_0-it.gguf", "README.md"]))

    with pytest.raises(ValueError, match="Available GGUF files: gemma-4-E2B_q8_0-it.gguf") as excinfo:
        validate_hub_source(resolve_model_source(GEMMA_URL))
    assert "at revision '1894d1fc'" in str(excinfo.value)


def test_validate_hub_source_checks_the_mmproj_file_too(monkeypatch):
    fake_hub(monkeypatch, _FakeModelInfo(files=["gemma-4-E2B_q4_0-it.gguf"]))
    source = resolve_model_source(
        GEMMA_URL,
        "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/1894d1fc/mmproj-BF16.gguf",
    )

    with pytest.raises(ValueError, match="File 'mmproj-BF16.gguf' does not exist"):
        validate_hub_source(source)


def test_validate_hub_source_looks_up_a_second_revision_only_when_it_differs(monkeypatch):
    calls = fake_hub(monkeypatch, _FakeModelInfo(files=["gemma-4-E2B_q4_0-it.gguf", "mmproj-BF16.gguf"]))
    source = resolve_model_source(
        GEMMA_URL,
        "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/mmproj-BF16.gguf",
    )

    validate_hub_source(source)
    assert [call["revision"] for call in calls] == ["1894d1fc", "main"]


def test_public_repo_files_returns_the_repo_listing(monkeypatch):
    fake_hub(monkeypatch, _FakeModelInfo(files=["a.gguf", "sub/b.gguf"]))

    assert public_repo_files("org/repo") == {"a.gguf", "sub/b.gguf"}


# --------------------------------------------------------------------------- #
# fallback_model_id
# --------------------------------------------------------------------------- #
def _place_gguf(models_dir, rel_path):
    """Create an empty GGUF at models_dir/rel_path and return its absolute path."""
    path = models_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0")
    return str(path)


def test_fallback_model_id_is_the_path_qualified_name(tmp_path):
    """An ad-hoc download is named by its repository-qualified path, from birth.

    Stable for the whole life of the install, and never shared with a same-named
    file from another repository.
    """
    gguf = _place_gguf(tmp_path, "TheBloke/Mistral-GGUF/mistral.Q4_0.gguf")
    assert fallback_model_id("", [gguf], str(tmp_path)) == "llamacpp:TheBloke/Mistral-GGUF/mistral.Q4_0"


def test_fallback_model_id_uses_the_key_model_type_as_namespace(tmp_path):
    gguf = _place_gguf(tmp_path, "org/repo/m-Q8_0.gguf")
    assert fallback_model_id("llamacpp", [gguf], str(tmp_path)) == "llamacpp:org/repo/m-Q8_0"


def test_fallback_model_id_ignores_mmproj(tmp_path):
    """The mmproj belongs to the main GGUF and must never name the model."""
    files = [
        _place_gguf(tmp_path, "org/repo/mmproj-BF16.gguf"),
        _place_gguf(tmp_path, "org/repo/model-Q4_0.gguf"),
    ]
    assert fallback_model_id("", files, str(tmp_path)) == "llamacpp:org/repo/model-Q4_0"


def test_fallback_model_id_without_any_gguf(tmp_path):
    assert fallback_model_id("", [], str(tmp_path)) is None
    mmproj = _place_gguf(tmp_path, "org/repo/mmproj-BF16.gguf")
    assert fallback_model_id("", [mmproj], str(tmp_path)) is None


def test_fallback_model_id_keeps_the_stem_for_a_catalog_declared_file(tmp_path, monkeypatch):
    """A file downloaded at the location the baked catalog declares is that curated model."""
    monkeypatch.setattr(hf_downloader, "catalog_gguf_declarations", lambda: [("org/repo", "model-Q4_0.gguf", "llamacpp:model-Q4_0")])
    gguf = _place_gguf(tmp_path, "org/repo/model-Q4_0.gguf")
    assert fallback_model_id("", [gguf], str(tmp_path)) == "llamacpp:model-Q4_0"


def test_fallback_model_id_matches_what_the_listing_derives(tmp_path):
    """The record and the listing must agree on what to call an ad-hoc download.

    Including when two repositories publish the same file name: the downloader sees
    the llamacpp directory as its models root, the listing sees its parent.
    """
    import list_models

    models_dir = tmp_path / "llamacpp"
    first = _place_gguf(models_dir, "TheBloke/Mistral-GGUF/mistral.Q4_0.gguf")

    listed = list_models.find_llamacpp_models(str(tmp_path))
    assert len(listed) == 1
    assert fallback_model_id("", [first], str(models_dir)) == listed[0]["id"]

    second = _place_gguf(models_dir, "bartowski/Mistral-GGUF/mistral.Q4_0.gguf")
    listed = {entry["path"]: entry["id"] for entry in list_models.find_llamacpp_models(str(tmp_path))}
    assert fallback_model_id("", [first], str(models_dir)) == listed[first]
    assert fallback_model_id("", [second], str(models_dir)) == listed[second]


# --------------------------------------------------------------------------- #
# generate_models_ini
# --------------------------------------------------------------------------- #
def _read_models_ini(models_dir):
    import configparser

    config = configparser.ConfigParser()
    config.read(models_dir / "models.ini")
    return config


def test_models_ini_names_declared_files_by_stem_and_ad_hoc_files_by_path(tmp_path, capsys):
    """Curated files serve under the stem their fixed id uses; ad-hoc under their path."""
    _place_gguf(tmp_path, "moondream/moondream2-gguf/moondream2-f16.gguf")
    _place_gguf(tmp_path, "moondream/moondream2-gguf/moondream2-mmproj-f16.gguf")
    _place_gguf(tmp_path, "unsloth/Qwen3-GGUF/Qwen3-Q4_0.gguf")
    declarations = [("moondream/moondream2-gguf", "moondream2-f16.gguf", "llamacpp:moondream2-f16")]

    generate_models_ini(tmp_path, declarations)

    config = _read_models_ini(tmp_path)
    assert sorted(config.sections()) == ["moondream2-f16", "unsloth/Qwen3-GGUF/Qwen3-Q4_0"]
    assert config["moondream2-f16"]["model"].endswith("moondream2-f16.gguf")
    assert config["moondream2-f16"]["mmproj"].endswith("moondream2-mmproj-f16.gguf")


def test_models_ini_keeps_a_section_per_repository_for_a_shared_file_name(tmp_path, capsys):
    """Two repositories publishing the same file name: neither may shadow the other."""
    _place_gguf(tmp_path, "unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf")
    _place_gguf(tmp_path, "bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf")

    generate_models_ini(tmp_path)

    config = _read_models_ini(tmp_path)
    assert sorted(config.sections()) == [
        "bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M",
        "unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M",
    ]
    for section in config.sections():
        assert config[section]["model"].endswith(f"{section}.gguf")


def test_models_ini_keeps_the_curated_stem_next_to_a_same_named_impostor(tmp_path, capsys):
    """A file named like a curated model, from another repository, never answers to its name."""
    _place_gguf(tmp_path, "google/gemma-gguf/gemma-Q4_0.gguf")
    _place_gguf(tmp_path, "bartowski/gemma-clone-GGUF/gemma-Q4_0.gguf")
    declarations = [("google/gemma-gguf", "gemma-Q4_0.gguf", "llamacpp:gemma-Q4_0")]

    generate_models_ini(tmp_path, declarations)

    config = _read_models_ini(tmp_path)
    assert sorted(config.sections()) == ["bartowski/gemma-clone-GGUF/gemma-Q4_0", "gemma-Q4_0"]
    # The curated name serves the curated file, whatever else is installed.
    assert config["gemma-Q4_0"]["model"].endswith("google/gemma-gguf/gemma-Q4_0.gguf")


def test_models_ini_ad_hoc_names_are_stable_across_regenerations(tmp_path, capsys):
    """models.ini is regenerated on every download and delete; ad-hoc names never move."""
    _place_gguf(tmp_path, "unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf")
    duplicate = tmp_path / "bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf"
    _place_gguf(tmp_path, "bartowski/SmolLM2-GGUF/SmolLM2-Q4_K_M.gguf")

    generate_models_ini(tmp_path)
    duplicate.unlink()
    generate_models_ini(tmp_path)

    assert _read_models_ini(tmp_path).sections() == ["unsloth/SmolLM2-GGUF/SmolLM2-Q4_K_M"]


# --------------------------------------------------------------------------- #
# no_match_message
# --------------------------------------------------------------------------- #
class _RepoFile:
    """Stand-in for huggingface_hub's RepoFile, which only needs a .path here."""

    def __init__(self, path):
        self.path = path


def test_no_match_message_lists_the_available_gguf_files(monkeypatch):
    files = [_RepoFile("Qwen3-0.6B-Q8_0.gguf"), _RepoFile("Qwen3-0.6B-BF16.gguf")]
    monkeypatch.setattr("hugging_face.hf_downloader.list_repo_matches", lambda *a, **k: files)
    message = no_match_message("unsloth/Qwen3-0.6B-GGUF", "*Q4_0*.gguf")
    assert "No file matching '*Q4_0*.gguf' found in repository 'unsloth/Qwen3-0.6B-GGUF'" in message
    # Sorted, so the caller can see what to ask for instead.
    assert "Available GGUF files: Qwen3-0.6B-BF16.gguf, Qwen3-0.6B-Q8_0.gguf" in message


def test_no_match_message_when_the_repo_has_no_gguf(monkeypatch):
    monkeypatch.setattr("hugging_face.hf_downloader.list_repo_matches", lambda *a, **k: [])
    assert "contains no GGUF files at all" in no_match_message("org/repo", "*Q4_0*.gguf")


def test_no_match_message_degrades_when_the_hub_is_unreachable(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("no network")

    monkeypatch.setattr("hugging_face.hf_downloader.list_repo_matches", _boom)
    message = no_match_message("org/repo", "*Q4_0*.gguf")
    assert message == "No file matching '*Q4_0*.gguf' found in repository 'org/repo'."


def test_download_matched_files_reports_what_is_available(monkeypatch):
    """A defaulted Q4_0 that the repo does not carry must fail actionably."""

    def _list(_repo_id, patterns, **_kwargs):
        # Nothing matches the requested quantization; the repo does hold a Q8_0.
        return [] if "Q4_0" in patterns[0] else [_RepoFile("m-Q8_0.gguf")]

    monkeypatch.setattr("hugging_face.hf_downloader.list_repo_matches", _list)
    with pytest.raises(FileNotFoundError, match="Available GGUF files: m-Q8_0.gguf"):
        download_matched_files("org/repo", "*Q4_0*.gguf", "/tmp/out", JsonProgress)


# --------------------------------------------------------------------------- #
# has_model_content
# --------------------------------------------------------------------------- #
def _repo_dir(tmp_path):
    repo = tmp_path / "llamacpp" / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    repo.mkdir(parents=True)
    return repo


def test_has_model_content_with_gguf(tmp_path):
    repo = _repo_dir(tmp_path)
    (repo / "gemma-4-E2B_q4_0-it.gguf").write_bytes(b"\0")
    assert has_model_content(str(repo)) is True


def test_has_model_content_ignores_metadata_only_dir(tmp_path):
    repo = _repo_dir(tmp_path)
    (repo / METADATA_NAME).write_text("handler: hf-handler\n")
    assert has_model_content(str(repo)) is False


def test_has_model_content_ignores_marker_only_dir(tmp_path):
    repo = _repo_dir(tmp_path)
    (repo / ".download").write_text("{}")
    assert has_model_content(str(repo)) is False


def test_has_model_content_ignores_cache_only_dir(tmp_path):
    repo = _repo_dir(tmp_path)
    (repo / ".cache").mkdir()
    (repo / ".cache" / "blob").write_bytes(b"\0")
    assert has_model_content(str(repo)) is False


def test_has_model_content_empty_or_missing_dir(tmp_path):
    assert has_model_content(str(_repo_dir(tmp_path))) is False
    assert has_model_content(str(tmp_path / "absent")) is False


# --------------------------------------------------------------------------- #
# delete + prune
# --------------------------------------------------------------------------- #
def test_delete_removes_metadata_and_prunes_repo_dir(tmp_path):
    base = tmp_path / "models"
    repo = base / "google" / "gemma-4-E2B-it-qat-q4_0-gguf"
    repo.mkdir(parents=True)
    (repo / "gemma-4-E2B_q4_0-it.gguf").write_bytes(b"\0")
    (repo / METADATA_NAME).write_text("handler: hf-handler\n")

    delete_matched_files(str(repo), str(base), "gemma-4-E2B_q4_0-it.gguf")
    # The metadata record kept the repo directory alive; the prune drops it.
    assert repo.is_dir()
    assert prune_emptied_repo_dir(str(repo), str(base)) is True
    assert not repo.exists()
    # The now-empty org directory goes too, but never the /models mount.
    assert not (base / "google").exists()
    assert base.is_dir()


def test_prune_keeps_dir_when_a_sibling_gguf_remains(tmp_path):
    base = tmp_path / "models"
    repo = base / "unsloth" / "gemma-3-1b-it-GGUF"
    repo.mkdir(parents=True)
    (repo / "gemma-3-1b-it-Q4_0.gguf").write_bytes(b"\0")
    (repo / "gemma-3-1b-it-Q8_0.gguf").write_bytes(b"\0")
    (repo / METADATA_NAME).write_text("handler: hf-handler\n")

    delete_matched_files(str(repo), str(base), "gemma-3-1b-it-Q4_0.gguf")
    assert prune_emptied_repo_dir(str(repo), str(base)) is False
    assert (repo / "gemma-3-1b-it-Q8_0.gguf").is_file()
    assert (repo / METADATA_NAME).is_file()


def test_prune_is_a_noop_for_a_missing_dir(tmp_path):
    assert prune_emptied_repo_dir(str(tmp_path / "absent"), str(tmp_path)) is False
    assert os.path.isdir(tmp_path)


# --------------------------------------------------------------------------- #
# quantizations sharing a repository directory
#
# One Hugging Face repository publishes many quantizations and they all land in the same
# <output-dir>/<repo-id>, so "is this model installed" and "what does an interrupted
# download throw away" can only be answered against the files the request names.
# --------------------------------------------------------------------------- #
def _qwen_repo(tmp_path, *files):
    """Build a repo directory holding *files*; return the models mount and that directory."""
    models_dir = tmp_path / "models" / "llamacpp"
    repo = models_dir / "unsloth" / "Qwen3-0.6B-GGUF"
    repo.mkdir(parents=True)
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0")
    return models_dir, repo


def test_matching_files_selects_only_the_requested_quantization(tmp_path):
    _models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B-Q3_K_S.gguf")
    assert [p.name for p in matching_files(str(repo), ["*Q3_K_S*.gguf"])] == ["Qwen3-0.6B-Q3_K_S.gguf"]


def test_matching_files_ignores_bookkeeping_and_cache_entries(tmp_path):
    _models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf", ".cache/huggingface/download/x.Q4_0.gguf")
    (repo / MARKER_NAME).write_text("{}")
    assert [p.name for p in matching_files(str(repo), ["*"])] == ["Qwen3-0.6B-Q4_0.gguf"]


def test_matching_files_finds_a_quantization_kept_in_its_own_folder(tmp_path):
    """Some repos nest their files per quantization, which the pattern matches as a path."""
    _models_dir, repo = _qwen_repo(tmp_path, "Q3_K_S/model.gguf")
    assert [p.name for p in matching_files(str(repo), ["*Q3_K_S*.gguf"])] == ["model.gguf"]


def test_is_installed_ignores_another_quantization_of_the_same_repo(tmp_path):
    _models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")
    assert is_installed(str(repo), ["*Q4_0*.gguf"]) is True
    assert is_installed(str(repo), ["*Q3_K_S*.gguf"]) is False


def test_is_installed_requires_the_mmproj_file_too(tmp_path):
    _models_dir, repo = _qwen_repo(tmp_path, "model-Q4_0.gguf")
    patterns = ["*Q4_0*.gguf", "*mmproj*BF16*.gguf"]
    assert is_installed(str(repo), patterns) is False
    (repo / "mmproj-BF16.gguf").write_bytes(b"\0")
    assert is_installed(str(repo), patterns) is True


def test_source_patterns_covers_the_model_and_its_mmproj():
    assert source_patterns(resolve_model_source("unsloth/Qwen3-0.6B-GGUF:Q3_K_S")) == ["*Q3_K_S*.gguf"]
    source = resolve_model_source("llamacpp:unsloth/gemma-4-E4B-it-GGUF:Q4_0:BF16")
    assert source_patterns(source) == ["*Q4_0*.gguf", "*mmproj*BF16*.gguf"]


def test_discard_removes_the_repo_dir_when_it_holds_nothing_else(tmp_path):
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q3_K_S.gguf")
    (repo / MARKER_NAME).write_text("{}")

    discard_incomplete_download(str(repo), str(models_dir), ["*Q3_K_S*.gguf"])
    assert not repo.exists()
    # The emptied owner directory goes too, but never the models mount.
    assert not repo.parent.exists()
    assert models_dir.is_dir()


def test_discard_keeps_a_sibling_quantization(tmp_path):
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf", ".cache/huggingface/download/partial.incomplete")
    (repo / MARKER_NAME).write_text("{}")
    (repo / METADATA_NAME).write_text("handler: hf-handler\n")

    discard_incomplete_download(str(repo), str(models_dir), ["*Q3_K_S*.gguf"])
    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()
    assert (repo / METADATA_NAME).is_file()
    # The partial bytes go, and so does the marker that would flag the survivor.
    assert not (repo / ".cache").exists()
    assert not (repo / MARKER_NAME).exists()


def test_discard_drops_a_requested_file_that_landed_before_the_kill(tmp_path):
    """A file being there does not mean the killed process finished writing it."""
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B-Q3_K_S.gguf")
    discard_incomplete_download(str(repo), str(models_dir), ["*Q3_K_S*.gguf"])
    assert not (repo / "Qwen3-0.6B-Q3_K_S.gguf").exists()
    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()


# --------------------------------------------------------------------------- #
# main(): a second quantization into an occupied repository directory
# --------------------------------------------------------------------------- #
def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["hf_downloader.py", *argv])
    hf_downloader.main()


@pytest.fixture
def stub_download(monkeypatch):
    """Stub the Hub out: record the patterns asked for, and create the files they name.

    The returned list is what distinguishes a real download from an early return.
    """
    requested: list[str] = []

    def _download(_repo_id, allow_pattern, output_dir, _tqdm_class, ignore_pattern=None, verbose=False):
        requested.append(allow_pattern)
        Path(output_dir, allow_pattern.replace("*", "")).write_bytes(b"\0")

    monkeypatch.setattr(hf_downloader, "validate_hub_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_downloader, "download_matched_files", _download)
    return requested


def test_download_adds_a_quantization_next_to_the_one_already_there(tmp_path, monkeypatch, stub_download, capsys):
    """The reported bug: a repo directory holding Q4_0 must not answer for Q3_K_S."""
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")

    _run_main(monkeypatch, "--model-url", "llamacpp:unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))

    assert stub_download == ["*Q3_K_S*.gguf"]
    assert (repo / "Q3_K_S.gguf").is_file()
    # The quantization that was already installed is untouched, and no marker is left.
    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()
    assert not (repo / MARKER_NAME).exists()
    # Only the file this run asked for is reported as downloaded.
    ggufs = [a for event in read_events(capsys) for a in event.get("artifacts", []) if a.endswith(".gguf")]
    assert [os.path.basename(a) for a in ggufs] == ["Q3_K_S.gguf"]


def test_download_skips_the_quantization_that_is_already_installed(tmp_path, monkeypatch, stub_download, capsys):
    models_dir, _repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")

    _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q4_0", "--output-dir", str(models_dir))

    assert stub_download == []
    descriptions = [event["description"] for event in read_events(capsys)]
    assert any(d.startswith("Model exists:") and "Qwen3-0.6B-Q4_0.gguf" in d for d in descriptions)


def test_download_marker_names_the_files_it_is_downloading(tmp_path, monkeypatch, stub_download):
    """The marker is per repository, so it has to say which quantization it stands for."""
    models_dir, repo = _qwen_repo(tmp_path)
    seen: list[dict] = []
    monkeypatch.setattr(
        hf_downloader,
        "download_matched_files",
        lambda *args, **kwargs: seen.append(read_marker(str(repo / MARKER_NAME))),
    )

    _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))

    assert seen[0]["file_patterns"] == ["*Q3_K_S*.gguf"]


def test_a_failed_download_does_not_take_the_installed_quantization_with_it(tmp_path, monkeypatch, stub_download):
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")

    def _boom(*_args, **_kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(hf_downloader, "download_matched_files", _boom)
    with pytest.raises(OSError, match="connection reset"):
        _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))

    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()
    assert not (repo / MARKER_NAME).exists()


def test_an_interrupted_download_is_retried_without_losing_a_sibling(tmp_path, monkeypatch, stub_download, capsys):
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")
    (repo / MARKER_NAME).write_text("{}")

    _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))

    assert stub_download == ["*Q3_K_S*.gguf"]
    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()
    descriptions = [event["description"] for event in read_events(capsys)]
    assert "Removing incomplete previous download: unsloth/Qwen3-0.6B-GGUF" in descriptions


def test_a_stale_marker_does_not_destroy_the_installed_quantization(tmp_path, monkeypatch, stub_download, capsys):
    """Requesting Q4_0 while a killed Q3_K_S left its marker behind must not delete Q4_0.

    The cleanup has to go by what the marker says was in flight, not by what the caller
    is asking for — otherwise it wipes a complete model, and on an offline board the
    re-download that was supposed to follow never happens.
    """
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")
    (repo / MARKER_NAME).write_text(json.dumps({"status": "downloading", "file_patterns": ["*Q3_K_S*.gguf"]}))

    _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q4_0", "--output-dir", str(models_dir))

    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()
    # Installed and complete: nothing to download, and the stale marker is cleared.
    assert stub_download == []
    assert not (repo / MARKER_NAME).exists()
    descriptions = [event["description"] for event in read_events(capsys)]
    assert any(d.startswith("Model exists:") and "Qwen3-0.6B-Q4_0.gguf" in d for d in descriptions)


def test_a_stale_marker_still_discards_its_own_partial_files(tmp_path, monkeypatch, stub_download):
    """The other half: what the marker does name is scratch and goes, siblings and all."""
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B-Q3_K_S.gguf")
    (repo / MARKER_NAME).write_text(json.dumps({"status": "downloading", "file_patterns": ["*Q3_K_S*.gguf"]}))

    _run_main(monkeypatch, "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))

    assert stub_download == ["*Q3_K_S*.gguf"]
    assert (repo / "Qwen3-0.6B-Q4_0.gguf").is_file()


def test_interrupted_patterns_reads_what_the_marker_recorded(tmp_path):
    marker = tmp_path / MARKER_NAME
    marker.write_text(json.dumps({"file_patterns": ["*Q3_K_S*.gguf"]}))
    assert interrupted_patterns(marker) == ["*Q3_K_S*.gguf"]

    # No field, unusable field, or no marker at all: nothing is known to be scratch.
    marker.write_text("{}")
    assert interrupted_patterns(marker) == []
    marker.write_text(json.dumps({"file_patterns": "*.gguf"}))
    assert interrupted_patterns(marker) == []
    assert interrupted_patterns(tmp_path / "absent") == []


def test_check_answers_for_the_requested_quantization_only(tmp_path, monkeypatch, capsys):
    models_dir, repo = _qwen_repo(tmp_path, "Qwen3-0.6B-Q4_0.gguf")

    # Installed, even though the repository directory carries a marker for another file.
    (repo / MARKER_NAME).write_text("{}")
    _run_main(monkeypatch, "--check", "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q4_0", "--output-dir", str(models_dir))
    assert read_events(capsys)[-1] == {"event": "info", "description": "Model exists: *Q4_0*.gguf", "downloading": False}

    # The quantization the marker stands for is the one still on its way.
    _run_main(monkeypatch, "--check", "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))
    assert read_events(capsys)[-1]["downloading"] is True

    # Without a marker, a quantization that is not there is simply missing.
    (repo / MARKER_NAME).unlink()
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--check", "--model-url", "unsloth/Qwen3-0.6B-GGUF:Q3_K_S", "--output-dir", str(models_dir))
    assert read_events(capsys)[-1] == {"event": "error", "description": "Model does not exist: *Q3_K_S*.gguf", "downloading": False}
