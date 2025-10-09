# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
from arduino.app_bricks.streamlit_ui import st


def arduino_header(title: str):
    """Arduino custom header.

    Render a minimal Arduino header: left-aligned title, right-aligned logo SVG, styled. SVG logo loaded by file.

    ---

    Streamlit UI Brick

    This module forwards the full [Streamlit](https://streamlit.io) API.

    For detailed usage of Streamlit components such as buttons, sliders, charts, and layouts, refer to the official Streamlit documentation:
    https://docs.streamlit.io/develop/api-reference

    You can import this brick as:

        from arduino.app_bricks.streamlit_ui import st

    Then use it just like native Streamlit:

        st.title("My App")
        st.button("Click me")

    Additionally, custom components like `st.arduino_header()` are provided to streamline Arduino integration.
    """
    svg_path = os.path.join(os.path.dirname(__file__), "assets", "RGB-Arduino-Logo_Color Inline Loop.svg")
    svg_path = os.path.abspath(svg_path)
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_icon = f.read()
    except Exception as e:
        svg_icon = f"<span style='color:red'>Logo not found: {e}</span>"
    html = f"""
    <div style='display: flex; width:100%; justify-content: space-between; align-items: center; align-self: stretch;'>
        <span style='color: var(--text-accent, #008184); font-family: "Roboto Mono", monospace; font-size: 20px;
        font-style: normal; font-weight: 700; line-height: 170%; letter-spacing: 2.4px;'>
            {title}
        </span>
        <div style='display: flex; width: 48px; height: 48px; padding: 13.5px 1.5px; justify-content: center;
        align-items: center; aspect-ratio: 1/1;'>
            <div style="width:45px; height:21px; flex-shrink:0; display:flex; align-items:center;
            justify-content:center;">{svg_icon}</div>
        </div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)
