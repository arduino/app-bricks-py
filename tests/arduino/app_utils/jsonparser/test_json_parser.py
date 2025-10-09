# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import JSONParser


def test_successful_parser():
    """Test successfull parse."""
    ps = JSONParser()
    out = ps.parse('{"test_key": "test_val"}')
    assert out["test_key"] == "test_val"


def test_drop_data_parser():
    """Test parse of not valid json data."""
    ps = JSONParser(silent=True)
    out = ps.parse("not json text")
    assert out is None
