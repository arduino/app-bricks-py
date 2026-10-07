# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import streamlit as st
from .addons import arduino_header

# Kept for existing apps, an attribute added to a third-party module cannot be declared to pyright
# TODO: deprecate st.arduino_header in favor of the exported arduino_header, nothing calls it in the examples or the library
st.arduino_header = arduino_header  # pyright: ignore[reportAttributeAccessIssue]

__all__ = ["st", "arduino_header"]
