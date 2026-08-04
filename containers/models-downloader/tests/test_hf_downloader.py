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

import pytest

from common.model_metadata import METADATA_NAME
from hugging_face.hf_downloader import (
    JsonProgress,
    delete_matched_files,
    download_matched_files,
    fallback_model_id,
    gguf_pattern,
    has_model_content,
    is_hf_url,
    matches_pattern,
    no_match_message,
    parse_model_key,
    prune_emptied_repo_dir,
    resolve_model_source,
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


# --------------------------------------------------------------------------- #
# fallback_model_id
# --------------------------------------------------------------------------- #
def test_fallback_model_id_from_the_downloaded_gguf():
    assert fallback_model_id("", ["/models/llamacpp/TheBloke/Mistral-GGUF/mistral.Q4_0.gguf"]) == "llamacpp:mistral.Q4_0"


def test_fallback_model_id_uses_the_key_model_type_as_namespace():
    assert fallback_model_id("llamacpp", ["/models/llamacpp/org/repo/m-Q8_0.gguf"]) == "llamacpp:m-Q8_0"


def test_fallback_model_id_ignores_mmproj():
    """The mmproj belongs to the main GGUF and must never name the model."""
    files = ["/models/llamacpp/org/repo/mmproj-BF16.gguf", "/models/llamacpp/org/repo/model-Q4_0.gguf"]
    assert fallback_model_id("", files) == "llamacpp:model-Q4_0"


def test_fallback_model_id_without_any_gguf():
    assert fallback_model_id("", []) is None
    assert fallback_model_id("", ["/models/llamacpp/org/repo/mmproj-BF16.gguf"]) is None


def test_fallback_model_id_matches_what_the_listing_derives(tmp_path):
    """The record and the listing must agree on what to call an ad-hoc download."""
    import list_models

    gguf = tmp_path / "llamacpp" / "TheBloke" / "Mistral-GGUF" / "mistral.Q4_0.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"\0")

    listed = list_models.find_llamacpp_models(str(tmp_path))
    assert len(listed) == 1
    assert fallback_model_id("", [str(gguf)]) == listed[0]["id"]


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
