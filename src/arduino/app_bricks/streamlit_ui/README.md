# WebUI - Streamlit Brick

This brick enables you to create and host interactive, Python-based web applications powered by the **Streamlit** framework.

## Overview

The WebUI - Streamlit Brick allows you to:

- Build rich, interactive UIs using simple Python syntax
- Display real-time data from sensors, devices, or external APIs
- Trigger actions in other bricks or microcontrollers through buttons, sliders, or inputs

When running, your application will be accessible via a web browser at `http://<device-ip>:7000`

## Features

- Enables Streamlit web server functionality on port 7000
- Supports interactive UI components for data visualization and input
- Easily integrates with other Python modules and Arduino bricks
- Supports themes, layout customization, and Markdown/HTML rendering

## Important: Do NOT use `App.run()` with this brick

The Streamlit brick is executed via `streamlit run`, which manages its own event loop and runs your script in a secondary thread. Calling `App.run()` will raise:

```
ValueError: signal only works in main thread of the main interpreter
```

This happens because `App.run()` internally calls `signal.signal()`, which Python only allows from the main thread.

**If your app only uses Streamlit (no other bricks):** simply omit `App.run()`. Streamlit manages the application lifecycle.

**If your app uses Streamlit together with other bricks:** use `App.start_bricks()` instead of `App.run()`. Wrap the initialization in `@st.cache_resource` to ensure bricks are started only once (Streamlit re-runs the script on every user interaction).

```python
import streamlit as st
from arduino.app_utils import App, Bridge
from arduino.app_bricks.some_brick import SomeBrick

@st.cache_resource
def init_bricks():
    brick = SomeBrick(...)
    App.start_bricks()
    return brick

init_bricks()

st.title("My App")
value = Bridge.call("readSensor")
st.metric("Sensor", value)
```

See the `examples/` folder for complete examples.

## Code example and usage

```python
from arduino.app_bricks.streamlit_ui import st

st.title("Arduino Streamlit UI Example")
st.write("Interact with your Arduino modules using this web interface.")

if st.button("Send Command"):
    st.success("Command sent to Arduino!")
    
```

