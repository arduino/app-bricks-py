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
    """Replace the HTTP GET with a mock answering with the given payload."""
    with patch("arduino.app_utils.http.get") as mock_get:

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
    assert mock_get.call_args.kwargs == {"params": {"token": "token"}}
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


@pytest.fixture
def waqi(monkeypatch: pytest.MonkeyPatch):
    """A local stand-in for api.waqi.info: answers /feed/<station>/ for Turin, and a JSON error with a 404 otherwise."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            found = self.path.startswith("/feed/Turin/")
            body = json.dumps({"status": "ok", "data": STATION} if found else {"status": "error", "data": "Unknown station"}).encode()
            self.send_response(200 if found else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    monitor = AirQualityMonitoring(token="t k")
    monitor.city_api_url = f"http://127.0.0.1:{server.server_address[1]}/feed/"
    yield monitor, paths
    server.shutdown()
    server.server_close()


def test_city_names_are_percent_encoded_in_the_path(waqi):
    monitor, paths = waqi

    assert monitor.get_air_quality_by_city("Turin").city == "Turin"
    with pytest.raises(AirQualityLookupError, match="Unknown station"):
        monitor.get_air_quality_by_city("São Paulo de Olivença")

    assert paths == ["/feed/Turin/?token=t+k", "/feed/S%C3%A3o%20Paulo%20de%20Oliven%C3%A7a/?token=t+k"]
