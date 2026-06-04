# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from aihub import AIHubApp, parse_args
from inference import inference_callback


args = parse_args()

# Create and run the application
app = AIHubApp(inference_cb=inference_callback, **args)
app.run()
