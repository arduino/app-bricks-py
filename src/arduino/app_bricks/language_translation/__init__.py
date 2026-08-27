# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from .local_translation import (
    LanguageTranslation,
    TranslationError,
    TranslationNotSupportedError,
    TranslationUnavailableError,
)

__all__ = [
    "LanguageTranslation",
    "TranslationError",
    "TranslationNotSupportedError",
    "TranslationUnavailableError",
]
