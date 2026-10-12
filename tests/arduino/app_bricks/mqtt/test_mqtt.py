# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
import pytest
import paho.mqtt.client as mqtt
from unittest.mock import MagicMock

from arduino.app_bricks.mqtt import MQTT


@pytest.fixture(autouse=True)
def mock_load_client(monkeypatch: pytest.MonkeyPatch):
    """Replace _load_client with a fake client to avoid real network calls."""

    class FakeClient:
        def __init__(self):
            self.connected = False
            self.started = False
            self.published = []
            self.subscribed_to = []

        def username_pw_set(self, u, p):
            pass

        def is_connected(self):
            return self.connected

        def connect(self, addr, port, keepalive):
            self.connected = True
            self.connected_to = (addr, port, keepalive)
            return mqtt.MQTT_ERR_SUCCESS

        def loop_start(self):
            self.started = True

        def loop_stop(self):
            self.started = False

        def disconnect(self):
            self.connected = False
            self.connected_to = None

        def publish(self, topic, payload):
            self.published.append((topic, payload))
            return MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)

        def subscribe(self, topic):
            self.subscribed_to.append(topic)
            return (mqtt.MQTT_ERR_SUCCESS, 1)  # Simulate success with dummy mid

    monkeypatch.setattr("arduino.app_bricks.mqtt._load_client", lambda client_id, username, password, topics=None: FakeClient())


def test_mqtt_publish():
    """Test MQTT publishes strings as is, dicts as JSON and skips empty messages."""
    client = MQTT("127.0.0.1", 1883, "user", "pass")
    client.publish("test/topic", "hello")
    client.publish("test/topic", {"a": 1})
    client.publish("test/topic", "")
    client.publish("test/topic", {})
    topics = [topic for topic, _ in client.client.published]
    payloads = [payload for _, payload in client.client.published]
    assert topics == ["test/topic", "test/topic"]
    assert payloads[0] == "hello"
    assert json.loads(payloads[1]) == {"a": 1}


def test_mqtt_publish_rejects_empty_topic():
    """Test MQTT refuses to publish without a topic."""
    client = MQTT("127.0.0.1", 1883, "user", "pass")
    with pytest.raises(ValueError, match="Topic must be a non-empty string"):
        client.publish("", "hello")


def test_mqtt_subscribe():
    """Test MQTT client subscribes to topic correctly."""
    client = MQTT("127.0.0.1", 1883, "user", "pass")
    fake_client = client.client
    assert fake_client.started is False
    assert fake_client.connected is False
    assert fake_client.subscribed_to == []
    client.start()
    assert fake_client.started is True
    assert fake_client.connected is True
    assert fake_client.connected_to == ("127.0.0.1", 1883, 60)
    assert fake_client.subscribed_to == []
    client.subscribe("test/topic1")
    assert fake_client.subscribed_to == ["test/topic1"]
    client.stop()
    assert fake_client.started is False
    assert fake_client.connected is False
