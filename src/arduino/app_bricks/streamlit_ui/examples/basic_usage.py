# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.streamlit_ui import st

st.title('Example app')
name = st.text_input("What's your name?")

if name:
    st.success(f"Hello, {name}! 👋")

# Example button
if st.button("Click me!"):
    st.info("Button clicked!")
