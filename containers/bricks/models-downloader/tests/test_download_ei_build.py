# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the Edge Impulse build downloader URL selection.

Models declared with a ``history_id`` in models-list.yaml are pinned to one
entry of the project's deployment history and must be fetched from the history
endpoint; everything else keeps using the impulse deployment endpoint with its
``type``/``impulseId`` query parameters.
"""

from edge_impulse.download_ei_build import build_description, build_url


def test_url_without_history_id_uses_impulse_deployment_endpoint():
    url = build_url(project_id=948887, impulse_id=10, target="runner-linux-aarch64-qnn")

    assert url == ("https://studio.edgeimpulse.com/v1/api/948887/deployment/download?type=runner-linux-aarch64-qnn&impulseId=10")


def test_quantization_is_appended_to_the_impulse_deployment_url():
    url = build_url(project_id=948887, impulse_id=10, target="runner-linux-aarch64", quantization="int8")

    assert url.endswith("&modelType=int8")


def test_history_id_switches_to_the_deployment_history_endpoint():
    url = build_url(project_id=995296, impulse_id=6, target="arduino-uno-q", history_id=8)

    assert url == "https://studio.edgeimpulse.com/v1/api/995296/deployment/history/8/download"


def test_history_url_ignores_target_and_quantization():
    """The history entry already fixes the build, so no query parameters are sent."""
    url = build_url(
        project_id=995296,
        impulse_id=6,
        target="runner-linux-aarch64-qnn",
        quantization="int8",
        history_id=7,
    )

    assert "?" not in url
    assert url.endswith("/deployment/history/7/download")


def test_history_id_zero_is_still_treated_as_a_history_build():
    """``0`` is a valid id: only ``None`` means "no history entry"."""
    url = build_url(project_id=995296, impulse_id=6, target="arduino-uno-q", history_id=0)

    assert url.endswith("/deployment/history/0/download")


def test_info_description_names_the_history_entry():
    assert build_description(995296, 6, 8) == "Model info for project 995296 deployment history 8"
    assert build_description(948887, 10) == "Model info for project 948887 impulse 10"
