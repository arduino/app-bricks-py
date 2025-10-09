# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
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

        def publish(self, msg, topic):
            if msg == "" or msg == {} or msg is None:
                return None
            if isinstance(msg, dict):
                msg = json.dumps(msg)
            token = MagicMock()
            token.topic = topic
            token.message = msg
            self.published.append((topic, msg))
            return token

        def subscribe(self, topic):
            self.subscribed_to.append(topic)
            return (mqtt.MQTT_ERR_SUCCESS, 1)  # Simulate success with dummy mid

    monkeypatch.setattr("arduino.app_bricks.mqtt._load_client", lambda username, password, client_id, subscribe_topic=None: FakeClient())


def test_mqtt_publish():
    """Test MQTT client publishes messages correctly."""
    client = MQTT("127.0.0.1", 1883, "user", "pass")
    fake_client = client.client
    # write a plain string
    token1 = fake_client.publish("hello", topic="test/topic")
    assert token1.topic == "test/topic"
    assert token1.message == "hello"
    # write a dict → serialized JSON
    token2 = fake_client.publish({"a": 1}, topic="test/topic")
    assert token2.topic == "test/topic"
    assert json.loads(token2.message) == {"a": 1}
    # empty message or {} or None → returns None
    assert fake_client.publish("", topic="test/topic") is None
    assert fake_client.publish({}, topic="test/topic") is None
    assert fake_client.publish(None, topic="test/topic") is None
    # stop should call loop_stop and disconnect on the fake client
    client.stop()
    assert fake_client.started is False
    assert fake_client.connected is False


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
