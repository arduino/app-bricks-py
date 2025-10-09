# MQTT Connector brick

An MQTT connector designed to both publish messages to specified topics and subscribe to incoming messages from other clients, enabling seamless two-way communication.

## Code example and usage

Connector can be used as publisher or subscriber to exchange messages over a MQTT server.

### Message publishing

Sample code for message publishing

```python
import time
from arduino.app_bricks.mqtt import MQTT
from arduino.app_utils import App

client = MQTT(
    broker_address="127.0.0.1", broker_port=1883, username="admin", password="password"
)

App.start_brick(client)

def keep_publishing():
    client.publish("test/topic", "Hello Arduino")
    time.sleep(5)

App.run(user_loop=keep_publishing)
```

### Topic subscriber

Sample code for subscribing to a topic and receive messages

```python
from arduino.app_bricks.mqtt import MQTT
from arduino.app_utils import App

client = MQTT(
    broker_address="127.0.0.1", broker_port=1883, username="admin", password="password", topics=["test/topic"]
)

App.run()
```
