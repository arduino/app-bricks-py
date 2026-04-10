# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Generator

import pytest
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================
ASR_BASE = os.environ.get("ASR_BASE", "http://localhost:8085/audio-analytics/v1/api")
TTS_MODEL = os.environ.get("TTS_MODEL", "melo-tts-en")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "en")
TTS_VOICE = os.environ.get("TTS_VOICE", "default")
TTS_RATE = int(os.environ.get("TTS_RATE", "16000"))


TEST_PHRASES = [
    "Hello, this is a basic text to speech test.",
    "Try adding a light jacket to give your outfit a sharper finish.",
    "A navy shirt looks clean and versatile for a casual smart style.",
    "This sentence is meant to verify that the TTS service returns audio data correctly.",
    "Simple audio generation test with English text and default voice settings.",
]


def iter_tts_audio(
    text: str,
    *,
    asr_base: str = ASR_BASE,
    tts_model: str = TTS_MODEL,
    language: str = TTS_LANGUAGE,
    voice: str = TTS_VOICE,
    sample_rate: int = TTS_RATE,
    timeout=(5, 300),
    chunk_size: int = 64 * 1024,
    headers: Optional[dict] = None,
) -> Generator[bytes, None, None]:
    """
    Yield raw PCM bytes from the TTS service as they arrive.
    """
    url = f"{asr_base.rstrip('/')}/tts/synthesize"
    payload = {
        "text": text,
        "model": tts_model,
        "language": language,
        "voice": voice,
        "sample_rate": sample_rate,
    }
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    with requests.post(
        url,
        headers=request_headers,
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk


def save_tts_stream_to_file(
    chunks: Generator[bytes, None, None],
    out_pcm_path: Path,
) -> Path:
    """
    Write the streamed TTS output to a temporary file and move it atomically
    to the final destination only if the file is not empty.
    """
    out_pcm_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(out_pcm_path.parent),
            prefix="tts_",
            suffix=".pcm",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in chunks:
                tmp.write(chunk)

        if tmp_path is None or not tmp_path.exists():
            raise RuntimeError("TTS failed: temporary file was not created.")

        if tmp_path.stat().st_size == 0:
            try:
                tmp_path.unlink()
            except Exception:
                pass
            raise RuntimeError("TTS stream produced 0 bytes.")

        shutil.move(str(tmp_path), str(out_pcm_path))
        return out_pcm_path

    except Exception:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        raise


def synthesize_tts(
    text: str,
    out_pcm_path: Path,
    *,
    asr_base: str = ASR_BASE,
    tts_model: str = TTS_MODEL,
    language: str = TTS_LANGUAGE,
    voice: str = TTS_VOICE,
    sample_rate: int = TTS_RATE,
    timeout=(5, 300),
) -> Path:
    """
    Stream TTS audio and save it to a PCM file.
    """
    chunks = iter_tts_audio(
        text=text,
        asr_base=asr_base,
        tts_model=tts_model,
        language=language,
        voice=voice,
        sample_rate=sample_rate,
        timeout=timeout,
    )
    return save_tts_stream_to_file(chunks, out_pcm_path)


@pytest.fixture(scope="session")
def test_phrases():
    return TEST_PHRASES


class TestTTSBasic:
    def test_test_phrases_exist(self, test_phrases):
        assert test_phrases, "The test phrase list must not be empty"
        assert all(isinstance(text, str) and text.strip() for text in test_phrases), "All test phrases must be non-empty strings"


class TestTTSPerPhrase:
    @pytest.mark.parametrize(
        ("phrase_index", "text"),
        list(enumerate(TEST_PHRASES)),
        ids=[f"phrase_{i}" for i in range(len(TEST_PHRASES))],
    )
    def test_tts_generates_audio_file_for_each_phrase(self, phrase_index, text, tmp_path):
        out_dir = tmp_path / "tts"
        out_file = out_dir / f"phrase_{phrase_index}.pcm"

        final_pcm = synthesize_tts(
            text=text,
            out_pcm_path=out_file,
            asr_base=ASR_BASE,
            tts_model=TTS_MODEL,
            language=TTS_LANGUAGE,
            voice=TTS_VOICE,
            sample_rate=TTS_RATE,
            timeout=(5, 300),
        )

        assert final_pcm.exists(), f"Generated audio file does not exist for phrase {phrase_index}"
        assert final_pcm.is_file(), f"Generated output is not a file for phrase {phrase_index}"
        assert final_pcm.stat().st_size > 0, f"Generated audio file is empty for phrase {phrase_index}"

        print(f"\nPhrase index: {phrase_index}")
        print(f"Text: {text}")
        print(f"Generated audio: {final_pcm}")
        print(f"Size: {final_pcm.stat().st_size} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
