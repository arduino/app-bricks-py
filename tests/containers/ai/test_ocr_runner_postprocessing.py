# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Allowlist filtering in the ocr-runner's CTC decoder.

The decoder module is loaded standalone by file path (it only depends on numpy)
so the runner's `utils` package name cannot clash with other runners' in the
same test session.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[3] / "containers" / "ai" / "ocr-runner" / "utils" / "post_processing.py"
_spec = importlib.util.spec_from_file_location("ocr_post_processing", _MODULE_PATH)
post_processing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(post_processing)

CHARACTERS = "0123456789AB"


def _probs_for(text: str, converter) -> np.ndarray:
    """[1, T, C] probability matrix whose greedy decode is `text`."""
    preds = np.full((1, len(text), converter.num_classes), 0.01, dtype=np.float32)
    for t, char in enumerate(text):
        preds[0, t, converter.dict[char]] = 5.0
    return post_processing.softmax(preds, axis=2)


def test_decode_without_allowlist_reads_every_character():
    converter = post_processing.CTCLabelConverter(CHARACTERS)
    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "1A2B"


def test_allowlist_restricts_decoding():
    converter = post_processing.CTCLabelConverter(CHARACTERS)
    converter.set_allowlist("0123456789")

    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    text, confidence = converter.decode_greedy(probs)[0]

    assert text == "12"  # letters can no longer be emitted
    assert 0.0 <= confidence <= 1.0


def test_allowlist_can_be_cleared():
    converter = post_processing.CTCLabelConverter(CHARACTERS)
    converter.set_allowlist("0123456789")
    converter.set_allowlist(None)

    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "1A2B"

    converter.set_allowlist("AB")
    converter.set_allowlist("")  # empty string clears too
    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "1A2B"


def test_allowlist_keeps_language_filtering():
    # 'B' is in the model charset but outside the configured language set
    converter = post_processing.CTCLabelConverter(CHARACTERS, lang_char="0123456789A")
    converter.set_allowlist("AB")

    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "A"


def test_allowlist_with_unknown_characters_ignores_them(capsys):
    converter = post_processing.CTCLabelConverter(CHARACTERS)
    converter.set_allowlist("12€")  # '€' is not in the model charset

    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "12"
    assert "ignored" in capsys.readouterr().out


def test_allowlist_disjoint_from_charset_keeps_previous_filtering(capsys):
    converter = post_processing.CTCLabelConverter(CHARACTERS)
    converter.set_allowlist("xyz")  # nothing in common with the model charset

    probs = converter.filter_probabilities(np.log(_probs_for("1A2B", converter)))
    assert converter.decode_greedy(probs)[0][0] == "1A2B"
    assert "no character in common" in capsys.readouterr().out
