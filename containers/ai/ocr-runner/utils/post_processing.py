# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Recognizer logits -> text.

Numpy ports of `easyocr.utils.CTCLabelConverter.decode_greedy` and
`easyocr.recognition.custom_mean`.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def custom_mean(x: np.ndarray) -> float:
    """
    EasyOCR confidence aggregation: geometric-ish mean that penalises long strings less.

    Port of `easyocr.recognition.custom_mean`.
    """
    if len(x) == 0:
        return 0.0
    return float(np.asarray(x, dtype=np.float64).prod() ** (2.0 / np.sqrt(len(x))))


class CTCLabelConverter:
    """
    Greedy CTC decoder over the EasyOCR character set.

    Index 0 is the CTC blank token; index i + 1 maps to `character[i]`.
    """

    def __init__(self, character: str, lang_char: str | None = None) -> None:
        dict_character = list(character)
        self.character = ["[blank]"] + dict_character
        self.dict = {char: i + 1 for i, char in enumerate(dict_character)}
        self.ignore_idx = [0]

        # Characters present in the model but outside the selected language(s) are
        # zeroed out of the probability matrix before decoding.
        lang_char = character if lang_char is None else lang_char
        ignore_char = "".join(set(character) - set(lang_char))
        self._lang_ignore_char_idx = [self.dict[char] for char in ignore_char]
        self.ignore_char_idx = list(self._lang_ignore_char_idx)

    @property
    def num_classes(self) -> int:
        return len(self.character)

    def set_allowlist(self, allowlist: str | None) -> None:
        """
        Restrict decoding to the characters in `allowlist`.

        Every other character is zeroed out of the probability matrix before decoding,
        on top of the language filtering configured at construction time. Passing None
        or an empty string restores the language-only filtering.

        The ignore list is swapped atomically, so it is safe to call while another
        thread is decoding.

        Parameters
        ----------
        allowlist
            The characters to keep, e.g. "0123456789". Characters the model cannot
            emit are ignored with a warning.
        """
        if not allowlist:
            self.ignore_char_idx = list(self._lang_ignore_char_idx)
            return

        allowed = set(allowlist)
        unknown = allowed - set(self.dict)
        if unknown:
            print(f"Warning: allowlist characters not in the model character set are ignored: {''.join(sorted(unknown))!r}")
        if not (allowed & set(self.dict)):
            print("Warning: allowlist has no character in common with the model character set; keeping the previous filtering.")
            return

        lang_ignored = set(self._lang_ignore_char_idx)
        self.ignore_char_idx = [i for char, i in self.dict.items() if char not in allowed or i in lang_ignored]

    def filter_probabilities(self, preds: np.ndarray) -> np.ndarray:
        """
        Softmax the logits, drop out-of-language characters and renormalise.

        Parameters
        ----------
        preds
            [B, T, C] raw recognizer logits.

        Returns
        -------
        probabilities : np.ndarray
            [B, T, C] probabilities summing to 1 along the class axis.
        """
        preds_prob = softmax(preds, axis=2)
        if self.ignore_char_idx:
            preds_prob[:, :, self.ignore_char_idx] = 0.0
            preds_prob = preds_prob / np.sum(preds_prob, axis=2, keepdims=True)
        return preds_prob

    def decode_greedy(self, preds_prob: np.ndarray) -> list[tuple[str, float]]:
        """
        Greedily decode a probability matrix into strings plus confidences.

        Parameters
        ----------
        preds_prob
            [B, T, C] probabilities, as returned by `filter_probabilities`.

        Returns
        -------
        predictions : list[tuple[str, float]]
            One (text, confidence) pair per batch element.
        """
        characters = np.array(self.character)
        ignore = np.array(self.ignore_idx)

        indices = preds_prob.argmax(axis=2)
        values = preds_prob.max(axis=2)

        results: list[tuple[str, float]] = []
        for t, v in zip(indices, values):
            # True where the index is not a repeat of the previous timestep
            not_repeated = np.insert(~(t[1:] == t[:-1]), 0, True)
            # True where the index is not a blank / ignored token
            not_ignored = ~np.isin(t, ignore)
            keep = not_repeated & not_ignored

            text = "".join(characters[t[keep.nonzero()]])

            # Confidence is aggregated over every non-blank timestep, repeats included.
            max_probs = v[t != 0]
            results.append((text, custom_mean(max_probs) if len(max_probs) else 0.0))

        return results
