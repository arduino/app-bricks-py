# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import paho.mqtt.client as mqtt
import json
import uuid
from typing import Callable, List, Optional
from arduino.app_utils import Logger, brick

logger = Logger("MQTT")

DEFAULT_CLIENT_ID_PREFIX = "arduino"


def _generate_client_id(name: str):
    """Generate a unique client ID for MQTT clients.

    Args:
        name (str): The base name for the client ID.

    Returns:
        str: A unique client ID combining the base name and a UUID.
    """
    return name + "-" + str(uuid.uuid4())


def _load_client(client_id: str, username: Optional[str], password: Optional[str], topics: List[str] = None) -> mqtt.Client:
    """Load and configure an MQTT client with connection and disconnection handlers.

    Args:
        username (str): The username for MQTT authentication.
        password (str): The password for MQTT authentication.
        client_id (str): The unique client ID for the MQTT client.
        topics (List[str], optional): List of topics to subscribe to upon connection. Defaults to None.

    Returns:
        mqtt.Client: Configured MQTT client instance.
    """
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if username and password:
        client.username_pw_set(username, password)

    # Configure automatic reconnection
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    def on_connect(client, userdata, flags, reason_code, properties):
        """Callback function for when the client connects to the MQTT broker."""
        if reason_code != 0:
            logger.error(f"Failed to connect: {mqtt.error_string(reason_code)}")
            return

        logger.info("Client connected")
        if topics:
            for t in topics:
                try:
                    result, _ = client.subscribe(t)
                    if result != mqtt.MQTT_ERR_SUCCESS:
                        raise RuntimeError(mqtt.error_string(result))
                except Exception as e:
                    logger.error("Failed to subscribe to topic %s: %s", t, e)

    def on_disconnect(client, userdata, flags, reason_code, properties):
        """Callback function for when the client disconnects from the MQTT broker."""
        if reason_code == 0:
            logger.debug("Client disconnected gracefully")
        else:
            logger.warning("Reconnecting after connection lost: (%s)...", mqtt.error_string(reason_code))

    def on_subscribe(client, userdata, mid, granted_qos, properties=None):
        """Callback function for when the client successfully subscribes to a topic."""
        logger.info("Subscription successful to topic")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    return client


def _default_on_message(client, userdata, msg):
    """A default callback function for when a message is received from the subscribed topic.

    Args:
        client (mqtt.Client): The MQTT client instance.
        userdata: User-defined data of any type.
        msg (mqtt.MQTTMessage): The received MQTT message.
    """
    try:
        logger.info("Message received on topic '%s': %s", msg.topic, str(msg.payload.decode("utf-8")))
    except Exception as e:
        logger.error("Error reading incoming message: %s", e)


@brick
class MQTT:
    """MQTT class for publishing and subscribing to MQTT topics."""

    def __init__(
        self,
        broker_address: str,
        broker_port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        topics: List[str] = None,
        client_id: str = None,
    ):
        """Initialize the MQTT Publisher.

        Args:
            broker_address (str): The address of the MQTT broker.
            broker_port (int): The port of the MQTT broker.
            username (str): The username for MQTT authentication. Defaults to None.
            password (str): The password for MQTT authentication. Defaults to None.
            topics (List[str], optional): List of topics to subscribe to upon connection. Defaults to None.
            client_id (str, optional): A unique client ID for the MQTT client. If None or empty, a random ID will be generated. Defaults to None.
        """
        self.broker_address = broker_address
        self.broker_port = broker_port
        if not client_id or client_id == "":
            client_id = _generate_client_id(DEFAULT_CLIENT_ID_PREFIX)
        self.client = _load_client(client_id, username, password, topics)
        self.client.on_message = _default_on_message

    def start(self):
        """Start the MQTT client and connect to the broker."""
        try:
            if not self.client.is_connected():
                ec = self.client.connect(self.broker_address, self.broker_port, 60)
                if ec != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(mqtt.error_string(ec))
            self.client.loop_start()
        except Exception as e:
            logger.error("Error during MQTT client startup: %s", e)

    def stop(self):
        """Stop the MQTT client and disconnect from the broker."""
        try:
            self.client.loop_stop()
            if self.client.is_connected():
                ec = self.client.disconnect()
                if ec != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(mqtt.error_string(ec))
        except Exception as e:
            logger.error("Error during MQTT client shutdown: %s", e)

    def publish(self, topic: str, message: str | dict):
        """Publish a message to the MQTT topic.

        Args:
            topic (str): The topic to publish the message to.
            message (str|dict): The message to publish. Can be a string or a dictionary.

        Raises:
            ValueError: If the topic is an empty string.
            RuntimeError: If the publish operation fails.
        """
        if not topic:
            raise ValueError("Topic must be a non-empty string")

        try:
            if isinstance(message, dict) and len(message) > 0:
                message = json.dumps(message)

            if message and message != "":
                res = self.client.publish(topic, message)
                if res.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(mqtt.error_string(res.rc))
                logger.debug("Published message to topic '%s': %s", topic, message)
        except Exception as e:
            raise RuntimeError(f"Failed to publish message to topic {topic}: {e}")

    def subscribe(self, topic: str):
        """Subscribe to a specified MQTT topic.

        Args:
            topic (str): The topic to subscribe to.

        Raises:
            ValueError: If the topic is an empty string.
            RuntimeError: If the subscription fails.
        """
        if not topic:
            raise ValueError("Topic must be a non-empty string")

        try:
            result, _ = self.client.subscribe(topic)
            if result != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(mqtt.error_string(result))
        except Exception as e:
            raise RuntimeError(f"Failed to subscribe to topic {topic}: {e}")

    def on_message(self, topic: str, fn: Callable[[mqtt.Client, object, mqtt.MQTTMessage], None]):
        """Set the callback function for handling incoming messages on a specific topic.

        Args:
            topic (str): The topic to set the callback for.
            fn (Callable[[mqtt.Client, object, mqtt.MQTTMessage], None]): The callback function to handle incoming messages.

        Raises:
            ValueError: If the topic is an empty string or if fn is not callable.
            RuntimeError: If setting the callback fails.
        """
        if not topic:
            raise ValueError("Topic must be a non-empty string")
        if not callable(fn):
            raise ValueError("fn must be a callable function")

        try:
            self.client.message_callback_add(topic, fn)
            logger.info("Callback function set for topic '%s'", topic)
        except Exception as e:
            raise RuntimeError(f"Failed to set callback for topic {topic}: {e}")
