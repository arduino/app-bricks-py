# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for the release tag picked as the starting point of the release notes."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

# Make the repo-root ``scripts`` package importable regardless of the cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.previous_release_tag import main, previous_release_tag, tag_names, version_key  # noqa: E402

# The tags of the repository around the switch from release/* to bricks/*, as
# `git ls-remote --tags` prints them, with the binaries published in between.
LS_REMOTE = """\
1111111111111111111111111111111111111111\trefs/tags/release/0.11.1
2222222222222222222222222222222222222222\trefs/tags/release/0.12.0
3333333333333333333333333333333333333333\trefs/tags/release/0.12.0rc7
4444444444444444444444444444444444444444\trefs/tags/release/0.12.0rc7^{}
5555555555555555555555555555555555555555\trefs/tags/bricks/0.13.0rc1
6666666666666666666666666666666666666666\trefs/tags/bricks/0.13.0rc4
7777777777777777777777777777777777777777\trefs/tags/llamacpp/20260902
8888888888888888888888888888888888888888\trefs/tags/ai/0.12.1
"""
TAGS = tag_names(LS_REMOTE.splitlines())


def test_versions_sort_prereleases_before_their_final():
    versions = ["0.13.0", "0.13.0rc10", "0.13.0rc2", "0.13.0b1", "0.13.0a3", "0.12.1", "0.12.0"]
    assert sorted(versions, key=version_key) == ["0.12.0", "0.12.1", "0.13.0a3", "0.13.0b1", "0.13.0rc2", "0.13.0rc10", "0.13.0"]


@pytest.mark.parametrize("version", ["v0.13.0", "0.13", "0.13.0-rc1", "0.13.0rc", "0.13.0.1", "dev-latest", "20260902"])
def test_other_version_shapes_are_rejected(version):
    with pytest.raises(ValueError, match="not X.Y.Z"):
        version_key(version)


def test_tag_names_strips_the_ref_prefix():
    assert TAGS == [
        "release/0.11.1",
        "release/0.12.0",
        "release/0.12.0rc7",
        "release/0.12.0rc7^{}",
        "bricks/0.13.0rc1",
        "bricks/0.13.0rc4",
        "llamacpp/20260902",
        "ai/0.12.1",
    ]


def test_tag_names_accepts_plain_git_tag_output():
    assert tag_names(["release/0.5.0", "", "bricks/0.13.0rc1"]) == ["release/0.5.0", "bricks/0.13.0rc1"]


def test_prerelease_is_preceded_by_the_prerelease_before_it():
    assert previous_release_tag("0.13.0rc5", TAGS) == "bricks/0.13.0rc4"


def test_a_version_already_tagged_is_not_its_own_predecessor():
    assert previous_release_tag("0.13.0rc4", TAGS) == "bricks/0.13.0rc1"


def test_final_is_preceded_by_its_last_prerelease():
    assert previous_release_tag("0.13.0", TAGS) == "bricks/0.13.0rc4"
    assert previous_release_tag("0.12.0", TAGS) == "release/0.12.0rc7"


def test_hotfix_is_preceded_by_its_own_line():
    assert previous_release_tag("0.12.1", TAGS) == "release/0.12.0"


def test_binaries_and_other_prefixes_are_ignored():
    tags = ["llamacpp/20260902", "ai/0.12.1", "opencv/4.13.0.92-20260610", "libcamera/0.7.1-qcom4", "release/0.12.0"]
    assert previous_release_tag("0.13.0", tags) == "release/0.12.0"
    assert previous_release_tag("0.13.0", tags[:-1]) is None


def test_release_tags_with_another_version_shape_are_ignored():
    assert previous_release_tag("0.13.0", ["release/v0.12.0", "release/0.12.0"]) == "release/0.12.0"


def test_peeled_tag_entries_are_ignored():
    assert previous_release_tag("0.12.0", ["release/0.11.1", "release/0.12.0rc7", "release/0.12.0rc7^{}"]) == "release/0.12.0rc7"


def test_first_release_has_no_predecessor():
    assert previous_release_tag("0.5.0", TAGS) is None
    assert previous_release_tag("0.11.0", ["release/0.11.0"]) is None


def test_main_prints_the_previous_tag(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(LS_REMOTE))
    assert main(["0.13.0"]) == 0
    assert capsys.readouterr().out == "bricks/0.13.0rc4\n"


def test_main_prints_nothing_without_a_predecessor(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(LS_REMOTE))
    assert main(["0.5.0"]) == 0
    assert capsys.readouterr().out == ""


def test_main_rejects_other_version_shapes(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(LS_REMOTE))
    assert main(["v0.13.0"]) == 1
    assert "not X.Y.Z" in capsys.readouterr().err
