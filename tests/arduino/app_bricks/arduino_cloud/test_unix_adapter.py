# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from arduino.app_bricks.arduino_cloud.unix_adapter import HTTPEndpoint


def test_unix_url_carries_the_socket_path():
    endpoint = HTTPEndpoint("http+unix://%2Frun%2Farduino-cloud-connector%2Fdaemon.sock")
    assert endpoint.socket_path == "/run/arduino-cloud-connector/daemon.sock"


def test_plain_http_url_has_no_socket_path():
    assert HTTPEndpoint("http://127.0.0.1:5683").socket_path is None


def test_unsupported_scheme_is_rejected():
    with pytest.raises(ValueError):
        HTTPEndpoint("ftp://127.0.0.1:5683")
