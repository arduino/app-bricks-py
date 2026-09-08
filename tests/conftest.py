# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""The suite runs without an Arduino Router, while importing arduino.app_utils connects to one
eagerly and fails otherwise: the library's connect() is made to report success without a router.
Bridge traffic then fails as "not connected", exactly as it did before the eager connection.
"""

from unittest.mock import patch

from arduino.router_bridge import Bridge

patch.object(Bridge, "connect", return_value=True).start()
