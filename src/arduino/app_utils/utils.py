# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import os


def get_board_name() -> str:
    """Returns the name of the board currently running the code.

    Returns:
        str: The name of the board, in lowercase (e.g.: unoq, ventunoq, etc.).
            If the board name cannot be determined, returns "unknown".
    """
    return os.environ.get("BOARD_NAME", "unknown").lower()
