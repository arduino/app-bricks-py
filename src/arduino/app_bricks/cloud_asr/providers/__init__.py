# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from enum import Enum

from .openai import OpenAITranscribe
from .google import GoogleSpeech
from .types import ASRProviderEvent, ASRProvider


class CloudProvider(str, Enum):
    OPENAI_TRANSCRIBE = "openai-transcribe"
    GOOGLE_SPEECH = "google-speech"


DEFAULT_PROVIDER = CloudProvider.OPENAI_TRANSCRIBE


def provider_factory(
    api_key: str,
    language: str,
    sample_rate: int,
    name: CloudProvider = DEFAULT_PROVIDER,
) -> ASRProvider:
    """Return the ASR cloud provider implementation."""
    if name == CloudProvider.OPENAI_TRANSCRIBE:
        return OpenAITranscribe(
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
        )
    if name == CloudProvider.GOOGLE_SPEECH:
        return GoogleSpeech(
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
        )
    raise ValueError(f"Unsupported ASR cloud provider: {name}")


__all__ = [
    "ASRProviderEvent",
    "ASRProvider",
    "CloudProvider",
    "DEFAULT_PROVIDER",
    "GoogleSpeech",
    "OpenAITranscribe",
    "provider_factory",
]
