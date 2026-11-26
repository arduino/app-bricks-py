# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Category ids a brick can declare in the ``category`` field of its brick_config.yaml.

The id travels unchanged in the bricks list, App Lab maps it to the label and icon it shows.
App Lab can group several ids under one category (video and image both show as Vision), so an id
names what the brick works on rather than how App Lab groups it. App Lab shows the fallback icon
for an id it does not map yet.
"""

BRICK_CATEGORIES: tuple[str, ...] = (
    "api",
    "audio",
    "cloud",
    "image",
    "location",
    "miscellaneous",
    "sensor",
    "storage",
    "text",
    "ui",
    "video",
)

# Category of a brick_config.yaml that does not declare one
DEFAULT_BRICK_CATEGORY = "miscellaneous"
