# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import json
from arduino.app_utils import Logger

logger = Logger("JSONParser")


class JSONParser:
    def __init__(self, silent: bool = False):
        self.silent = silent

    def parse(self, item: str) -> dict:
        try:
            return json.loads(item)
        except Exception as e:
            if not self.silent:
                logger.error(f"Error parsing content: {e}")
            return None

    def process(self, item):
        if isinstance(item, str):
            return self.parse(item)

        return item  # No processing needed
