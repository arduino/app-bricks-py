# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import pytest

from arduino.app_bricks.air_quality_monitoring import AirQualityData, AirQualityLookupError, AirQualityMonitoring

STATION = {
    "city": {"name": "Turin", "geo": [45.07, 7.68], "url": "https://aqicn.org/city/turin"},
    "time": {"s": "2026-10-05 10:00:00"},
    "aqi": 42,
    "dominantpol": "pm25",
    "iaqi": {"pm25": {"v": 42}},
}


@pytest.fixture
def get():
    """Replace requests.get with a mock answering with the given payload."""
    with patch("arduino.app_bricks.air_quality_monitoring.requests.get") as mock_get:

        def respond(payload):
            mock_get.return_value = MagicMock(status_code=200, json=MagicMock(return_value=payload))
            return mock_get

        yield respond


@pytest.mark.parametrize(
    "item, url",
    [
        ({"city": "turin"}, "https://api.waqi.info/feed/turin/"),
        ({"latitude": 45.07, "longitude": 7.68}, "https://api.waqi.info/feed/geo:45.07;7.68/"),
        ({"ip": True}, "https://api.waqi.info/feed/here/"),
    ],
)
def test_process_dispatches_on_the_item_keys(get, item, url):
    mock_get = get({"status": "ok", "data": STATION})

    data = AirQualityMonitoring(token="token").process(item)

    assert mock_get.call_args.args == (url,)
    assert data == AirQualityData(
        city="Turin",
        lat=45.07,
        lon=7.68,
        url="https://aqicn.org/city/turin",
        last_update="2026-10-05 10:00:00",
        aqi=42,
        dominantpol="pm25",
        iaqi={"pm25": {"v": 42}},
    )


@pytest.mark.parametrize("item", [{}, {"latitude": 45.07}, {"ip": False}])
def test_process_rejects_a_dict_without_a_lookup_key(item):
    with pytest.raises(ValueError, match="Input dict must contain 'city', 'latitude' and 'longitude', or 'ip': True"):
        AirQualityMonitoring(token="token").process(item)


@pytest.mark.parametrize("item", [None, "turin", ["city", "turin"]])
def test_process_rejects_a_non_dict(item):
    with pytest.raises(ValueError, match="Input must be a dict"):
        AirQualityMonitoring(token="token").process(item)


@pytest.mark.parametrize(
    "data, message",
    [
        ("Unknown station", "Unknown station"),
        ({"message": "Invalid key"}, "Invalid key"),
        ({"reason": "Over quota"}, "{'reason': 'Over quota'}"),
        (None, "None"),
    ],
)
def test_failed_lookup_raises_the_api_message(get, data, message):
    get({"status": "error", "data": data})

    with pytest.raises(AirQualityLookupError) as error:
        AirQualityMonitoring(token="token").get_air_quality_by_city("nowhere")

    assert error.value.message == message
    assert error.value.status == "error"


@pytest.mark.parametrize("data", [{"data": "Not found"}, {"status": "ok", "data": "Not found"}])
def test_lookup_error_requires_an_error_status(data):
    with pytest.raises(ValueError, match="Status must be 'error'"):
        AirQualityLookupError.from_api_response(data)
