# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import inference
from aihub import AIHubApp, parse_args


args = parse_args()

# Create and run the application. apply_config is optional for a runner to define.
app = AIHubApp(
    inference_cb=inference.inference_callback,
    config_cb=getattr(inference, "apply_config", None),
    **args,
)
app.run()
