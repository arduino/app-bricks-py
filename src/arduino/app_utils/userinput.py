# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0


class UserTextInput:
    def __init__(self, prompt: str):
        self.prompt = prompt

    def get(self):
        return input(self.prompt)

    def produce(self):
        return input(self.prompt)
