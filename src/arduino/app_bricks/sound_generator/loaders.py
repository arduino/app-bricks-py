# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils.logger import Logger
from typing import List, Tuple
import re

logger = Logger(__name__)


class ABCNotationLoader:
    """ABC notation parser — ABC 2.1 standard compliant.

    Parses ABC notation strings into ``(note, duration_in_seconds)`` tuples
    suitable for playback through the SoundGenerator brick.

    Supported ABC 2.1 features:
        - Information fields: ``X:``, ``T:``, ``M:``, ``L:``, ``Q:``, ``K:``
        - Key signatures: all major/minor keys, modes (dorian … locrian),
          ``K:none``, ``K:Hp``/``HP``, ``exp`` (explicit), inline accidental
          overrides (e.g. ``K:D =f ^c``)
        - Accidentals: prefix ``^``, ``^^``, ``_``, ``__``, ``=`` with
          bar-local propagation (pitch-class scope)
        - Octave modifiers: ``'`` (up) and ``,`` (down), case-based octave
        - Duration notation: integer multiplier, ``/n``, ``n/m``, repeated
          slashes (``//`` = ``/4``, ``///`` = ``/8``)
        - Rests: ``z`` (visible), ``x`` (invisible), ``Z``/``X``
          (multimeasure, duration computed from ``M:``)
        - Broken rhythm: ``>``, ``>>``, ``<``, ``<<``
        - Tuplets: ``(p``, ``(p:q``, ``(p:q:r``
        - Chord brackets: ``[CEG]`` (flattened to sequential notes)
        - Grace notes ``{abc}``, decorations ``!ff!`` / ``+fermata+``,
          chord annotations ``"Cm"`` — stripped during pre-processing
        - Non-standard extension: ``%%transpose`` (octave shift)
        - Legacy extension: suffix ``#`` / ``b`` accidentals

    Known limitations (not implemented):
        - Multi-voice scores (``V:`` fields)
        - Repeat structures (``|:`` … ``:|``, numbered endings)
        - Ties (``-``) and slurs (``()``)
        - Inline information fields (``[K:Am]``)
        - ``%%propagate-accidentals`` directive (fixed pitch-class scope)
        - ``K:`` clef / transpose parameters
        - ``w:`` lyrics, ``s:`` symbol lines
    """

    NOT_HANDLED_RESERVED_LINES = r"^[A-Za-z]:"

    # ---- Key-signature tables (circle of fifths) ----
    _SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
    _FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]

    _MAJOR_KEY_SHARPS: dict[str, int] = {
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
        "F": -1,
        "BB": -2,
        "EB": -3,
        "AB": -4,
        "DB": -5,
        "GB": -6,
        "CB": -7,
    }

    _MODE_OFFSETS: dict[str, int] = {
        "": 0,
        "maj": 0,
        "major": 0,
        "ion": 0,
        "ionian": 0,
        "m": -3,
        "min": -3,
        "minor": -3,
        "aeo": -3,
        "aeolian": -3,
        "dor": -2,
        "dorian": -2,
        "phr": -4,
        "phrygian": -4,
        "lyd": 1,
        "lydian": 1,
        "mix": -1,
        "mixolydian": -1,
        "loc": -5,
        "locrian": -5,
    }

    # Semitone values for enharmonic resolution (C=0 ... B=11).
    # Names match SoundGenerator.SEMITONE_MAP for direct lookup.
    _BASE_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    _SEMITONE_TO_SHARP = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }
    _SEMITONE_TO_FLAT = {
        0: "C",
        1: "DB",
        2: "D",
        3: "EB",
        4: "E",
        5: "F",
        6: "GB",
        7: "G",
        8: "AB",
        9: "A",
        10: "BB",
        11: "B",
    }

    @staticmethod
    def _get_key_accidentals(key_str: str) -> dict[str, int]:
        """Derive per-pitch-class accidental offsets from an ABC ``K:`` field.

        Supports standard key names (``D``, ``Cm``, ``BbMix``), special keys
        (``none``, ``Hp``, ``HP``), mode keywords (major through locrian), the
        ``exp`` (explicit) keyword, and inline accidental overrides
        (e.g. ``K:D =f`` to cancel the key-signature F#).

        Args:
            key_str (str): Value of the K: header.

        Returns:
            dict[str, int]: Pitch class to semitone offset (+1 sharp, -1 flat).
        """
        if not key_str or not key_str.strip():
            return {}

        s = key_str.strip()

        # Special keys
        low = s.lower()
        if low in ("none", "perc", "drum"):
            return {}
        if s in ("Hp", "HP"):
            return {"F": 1, "C": 1}  # Highland bagpipe

        root = s[0].upper()
        s = s[1:]

        if s and s[0] == "#":
            root += "#"
            s = s[1:]
        elif s and s[0] == "b":
            root += "B"  # internal flat representation
            s = s[1:]

        # Split remaining into mode keyword and optional inline accidentals.
        mode_match = re.match(r"^([A-Za-z]*)(.*)", s.strip())
        mode = mode_match.group(1).lower() if mode_match else ""
        remaining = mode_match.group(2).strip() if mode_match else ""

        # "exp" = explicit key: only the listed inline accidentals apply
        explicit = mode == "exp"
        if explicit:
            mode = ""

        if not explicit:
            n = ABCNotationLoader._MAJOR_KEY_SHARPS.get(root)
            if n is None:
                logger.warning(f"Unknown key root '{root}', defaulting to C major")
                return {}

            mode_offset = ABCNotationLoader._MODE_OFFSETS.get(mode, None)
            if mode_offset is None:
                logger.warning(f"Unknown mode '{mode}', treating as major")
                mode_offset = 0
            n += mode_offset

            acc: dict[str, int] = {}
            if n > 0:
                for i in range(min(n, 7)):
                    acc[ABCNotationLoader._SHARP_ORDER[i]] = 1
            elif n < 0:
                for i in range(min(-n, 7)):
                    acc[ABCNotationLoader._FLAT_ORDER[i]] = -1
        else:
            acc = {}

        # Apply inline accidental overrides (e.g. ^f _b =c after the mode).
        if remaining:
            for acc_str, note_char in re.findall(r"([=^_]{1,2})([A-Ga-g])", remaining):
                pitch = note_char.upper()
                if acc_str == "^^":
                    acc[pitch] = 2
                elif acc_str == "__":
                    acc[pitch] = -2
                elif acc_str == "^":
                    acc[pitch] = 1
                elif acc_str == "_":
                    acc[pitch] = -1
                elif acc_str == "=":
                    acc.pop(pitch, None)  # natural = remove from key sig

        return acc

    @staticmethod
    def _resolve_note_name(pitch_class: str, accidental_offset: int, octave: int) -> tuple[str, int]:
        """Resolve a pitch class with accidental offset to a canonical note name.

        Handles enharmonic equivalents and octave wraparound
        (e.g. B# at octave 4 becomes C at octave 5).  Names are compatible
        with ``SoundGenerator.SEMITONE_MAP``.

        Args:
            pitch_class (str): Uppercase note letter ("C" ... "B").
            accidental_offset (int): Semitone shift (+1 sharp, -1 flat, 0 natural).
            octave (int): Base octave number.

        Returns:
            tuple[str, int]: (note_name, adjusted_octave).
        """
        semitone = ABCNotationLoader._BASE_SEMITONES[pitch_class] + accidental_offset
        octave_adj = 0
        while semitone >= 12:
            semitone -= 12
            octave_adj += 1
        while semitone < 0:
            semitone += 12
            octave_adj -= 1
        table = ABCNotationLoader._SEMITONE_TO_SHARP if accidental_offset > 0 else ABCNotationLoader._SEMITONE_TO_FLAT
        return table[semitone], octave + octave_adj

    @staticmethod
    def _parse_abc_duration(duration_str: str, default_duration_in_seconds: float) -> float:
        """Parse ABC duration notation.

        Supports integer multipliers (``2``), fractions (``/2``, ``3/2``), and
        the ABC shorthand for repeated slashes (``/`` = ``/2``, ``//`` = ``/4``,
        ``///`` = ``/8``).

        Args:
            duration_str (str): Duration string from ABC notation.
            default_duration_in_seconds (float): Duration of one default note
                unit (L: field) in seconds.

        Returns:
            float: Calculated duration in seconds.
        """
        if not duration_str:
            return default_duration_in_seconds

        if "/" in duration_str:
            parts = duration_str.split("/")
            numerator = int(parts[0]) if parts[0] else 1

            # Explicit denominator in the last segment (e.g. /2, 3/2)
            last = parts[-1].strip()
            if last:
                denominator = int(last)
            else:
                # Shorthand: / = /2, // = /4, /// = /8
                slash_count = len(parts) - 1
                denominator = 2**slash_count

            return default_duration_in_seconds * numerator / denominator

        try:
            multiplier = int(duration_str)
            return default_duration_in_seconds * multiplier
        except ValueError:
            return default_duration_in_seconds

    @staticmethod
    def parse_abc_notation(abc_string: str, default_octave: int = 4) -> Tuple[dict, List[Tuple[str, float]]]:
        """Parse an ABC notation string into ``(note, duration_in_seconds)`` tuples.

        See :class:`ABCNotationLoader` for the full list of supported ABC 2.1
        features and known limitations.

        Args:
            abc_string (str): ABC notation string.
            default_octave (int): Default octave for uppercase notes (C4).

        Returns:
            Tuple[dict, List[Tuple[str, float]]]: Metadata dictionary and list
                of (note, duration) tuples.
        """

        metadata = {}

        lines = abc_string.split("\n")
        music_lines = []

        # --- Parse Header Fields ---
        for line in lines:
            line = line.strip()
            if re.match(ABCNotationLoader.NOT_HANDLED_RESERVED_LINES, line):
                if line.startswith("X:"):
                    metadata["reference"] = line[2:].strip()
                elif line.startswith("T:"):
                    metadata["title"] = line[2:].strip()
                elif line.startswith("K:"):
                    metadata["key"] = line[2:].strip()
                elif line.startswith("L:"):
                    metadata["default_length"] = line[2:].strip()
                elif line.startswith("Q:"):
                    metadata["tempo"] = line[2:].strip()
                elif line.startswith("M:"):
                    metadata["meter"] = line[2:].strip()
                elif line.startswith("C:"):
                    metadata["composer"] = line[2:].strip()
                elif line.startswith("R:"):
                    metadata["rhythm"] = line[2:].strip()
            elif line.startswith("%%transpose"):
                # Handle transpose directive if needed
                matched = re.match(r"%%transpose\s+(-?\d+)", line)
                if matched:
                    # only octave transposition is supported
                    octaves = int(matched.group(1)) / 12
                    if octaves + default_octave < 0:
                        octaves = 0
                    metadata["transpose"] = int(octaves)
            elif not line.startswith("%") and line:
                music_lines.append(line)

        # Standard ABC default for L: is 1/8 if not specified.
        default_unit_fraction = 1 / 8

        if "default_length" in metadata and metadata["default_length"]:
            match_L = re.match(r"(\d+)/(\d+)", metadata["default_length"])
            if match_L:
                num, denom = int(match_L.group(1)), int(match_L.group(2))
                default_unit_fraction = num / denom

        bpm = 120  # Default BPM if Q: is not specified
        beat_unit_fraction = 0.25  # Default beat unit (1/4 or quarter note)

        if "tempo" in metadata and metadata["tempo"]:
            # Q: field is typically 'note_fraction=BPM', e.g. '1/4=120'
            match_Q = re.match(r"(\d+/\d+)=(\d+)", metadata["tempo"].replace(" ", ""))

            if match_Q:
                note_str, bpm_str = match_Q.groups()
                bpm = int(bpm_str)

                q_num, q_denom = map(int, note_str.split("/"))
                beat_unit_fraction = q_num / q_denom
            else:
                try:
                    bpm = int(metadata["tempo"].replace(" ", ""))
                except ValueError:
                    pass  # Keep default BPM

        # Duration in seconds of the note specified as the beat unit (Q: note)
        duration_of_beat_unit = 60.0 / bpm

        # Calculate the ratio between the default L: unit and the Q: beat unit.
        # This handles cases where L: and Q: define different note values (e.g., L:1/16, Q:1/4=120)
        ratio_to_beat_unit = default_unit_fraction / beat_unit_fraction

        # The absolute duration in seconds of the note defined by L:
        default_duration_in_seconds = ratio_to_beat_unit * duration_of_beat_unit

        # Informational output
        if "title" in metadata:
            logger.info(f"Playing: {metadata['title']}")
        logger.info(f"BPM: {bpm}, Beat Unit Fraction: {beat_unit_fraction:.3f}, Default L: {default_unit_fraction:.3f}")
        logger.info(f"Duration of 1 beat: {duration_of_beat_unit:.3f}s. Default L: Duration: {default_duration_in_seconds:.3f}s")
        if "transpose" in metadata:
            logger.info(f"Transposing by {metadata['transpose']} octaves. Target default octave: {default_octave + metadata['transpose']}")

        # --- 5. Compute key-signature accidentals ---
        key_accidentals = ABCNotationLoader._get_key_accidentals(metadata.get("key", ""))
        if key_accidentals:
            logger.info(f"Key signature accidentals: {key_accidentals}")

        # --- 6. Compute measure duration (for Z multimeasure rests) ---
        meter_num, meter_denom = 4, 4
        if "meter" in metadata and metadata["meter"]:
            meter_match = re.match(r"(\d+)/(\d+)", metadata["meter"])
            if meter_match:
                meter_num, meter_denom = int(meter_match.group(1)), int(meter_match.group(2))
        measure_duration = (meter_num / meter_denom) / beat_unit_fraction * duration_of_beat_unit

        # --- 7. Parse Music Lines ---
        music_string = " ".join(music_lines)

        # Pre-processing: strip structures that contain note-like characters
        music_string = re.sub(r'"[^"]*"', "", music_string)  # Chord annotations ("Cm", "Ab")
        music_string = re.sub(r"\{[^}]*\}", "", music_string)  # Grace notes {abc}
        music_string = re.sub(r"![^!]*!", "", music_string)  # Decorations (!ff!, !fermata!)
        music_string = re.sub(r"\+[^+]+\+", "", music_string)  # Old-style decorations (+fermata+)

        result: List[Tuple[str, float]] = []

        # Tokenise: notes, rests, multimeasure rests, tuplets,
        # broken-rhythm markers, barlines, and chord brackets.
        tokens = re.findall(
            r"[=^_]{0,2}[A-Ga-g][',]*[#b]?[0-9]*(?:/+[0-9]*)?"
            r"|[zx][0-9]*(?:/+[0-9]*)?"
            r"|[ZX][0-9]*"
            r"|\(\d+(?::\d*(?::\d*)?)?"
            r"|[><]+"
            r"|[:\[\].]*\|+[:\[\].|]*"
            r"|\[|\]",
            music_string,
        )

        # NOTE: bar_accidentals is keyed by pitch class (all octaves).
        # ABC 2.1 §4.6 specifies "same pitch" (octave-specific) as the default,
        # but pitch-class propagation is the more predictable and common
        # behaviour for simple melodies targeted by this brick.
        bar_accidentals: dict[str, int] = {}
        broken_rhythm = None  # (count, direction): +1 for >, -1 for <
        tuplet_state = None  # (factor, remaining): duration multiplier, notes left

        for token in tokens:
            # --- Bar line: reset bar-local accidentals ---
            if "|" in token:
                bar_accidentals.clear()
                continue

            # --- Chord brackets (from [CEG] notation): skip ---
            if token in "[]":
                continue

            # --- Tuplet marker: (p or (p:q:r ---
            if token.startswith("(") and len(token) > 1 and token[1].isdigit():
                parts = token[1:].split(":")
                p = int(parts[0])
                if len(parts) >= 2 and parts[1]:
                    q = int(parts[1])
                else:
                    # Default q depends on p (ABC 2.1 §4.13)
                    q = 3 if p in (2, 4, 8) else 2
                r = int(parts[2]) if len(parts) >= 3 and parts[2] else p
                tuplet_state = (q / p, r)
                continue

            # --- Broken rhythm marker (>, >>, <, <<, ...) ---
            if token[0] in "><":
                direction = 1 if token[0] == ">" else -1
                broken_rhythm = (len(token), direction)
                continue

            # --- Multimeasure rest (Z, X) ---
            if token[0] in "ZX":
                count = int(token[1:]) if len(token) > 1 else 1
                result.append(("REST", measure_duration * count))
                continue

            # --- Rest (z visible, x invisible — identical for playback) ---
            if token[0] in "zx":
                duration = ABCNotationLoader._parse_abc_duration(token[1:], default_duration_in_seconds)
                result.append(("REST", duration))
                continue

            # --- Note ---
            pos = 0

            # 1) Optional prefix accidental (^, ^^, _, __, =)
            prefix_offset = None
            if token[pos] in "=^_":
                if token[pos : pos + 2] == "^^":
                    prefix_offset = 2
                    pos += 2
                elif token[pos : pos + 2] == "__":
                    prefix_offset = -2
                    pos += 2
                elif token[pos] == "^":
                    prefix_offset = 1
                    pos += 1
                elif token[pos] == "_":
                    prefix_offset = -1
                    pos += 1
                elif token[pos] == "=":
                    prefix_offset = 0  # explicit natural
                    pos += 1

            # 2) Note letter
            note_char = token[pos]
            pos += 1

            octave = default_octave
            if "transpose" in metadata:
                octave += metadata["transpose"]
            if note_char.islower():
                octave += 1
                note_char = note_char.upper()

            # 3) Octave markers (' ,)
            while pos < len(token) and token[pos] in "',":
                octave += 1 if token[pos] == "'" else -1
                pos += 1

            # 4) Legacy suffix accidental (# b) -- non-standard, kept for compat
            suffix_offset = None
            if pos < len(token) and token[pos] in "#b":
                suffix_offset = 1 if token[pos] == "#" else -1
                pos += 1

            # 5) Duration
            duration = ABCNotationLoader._parse_abc_duration(token[pos:], default_duration_in_seconds)

            # 6) Determine effective accidental
            if prefix_offset is not None:
                effective = prefix_offset
            elif suffix_offset is not None:
                effective = suffix_offset
            elif note_char in bar_accidentals:
                effective = bar_accidentals[note_char]
            elif note_char in key_accidentals:
                effective = key_accidentals[note_char]
            else:
                effective = 0

            # Propagate explicit accidental for the rest of the bar (ABC standard)
            if prefix_offset is not None or suffix_offset is not None:
                bar_accidentals[note_char] = effective

            # 7) Resolve to canonical note name (enharmonic + octave wrap)
            if effective != 0:
                resolved, octave = ABCNotationLoader._resolve_note_name(note_char, effective, octave)
            else:
                resolved = note_char

            note_name = f"{resolved}{octave}"

            # 8) Apply tuplet timing
            if tuplet_state is not None:
                factor, remaining = tuplet_state
                duration *= factor
                remaining -= 1
                tuplet_state = (factor, remaining) if remaining > 0 else None

            # 9) Apply pending broken rhythm
            if broken_rhythm is not None:
                count, direction = broken_rhythm
                dotted = (2 ** (count + 1) - 1) / (2**count)
                shortened = 1.0 / (2**count)
                if direction > 0:  # > : previous note dotted, current shortened
                    if result:
                        pn, pd = result[-1]
                        result[-1] = (pn, pd * dotted)
                    duration *= shortened
                else:  # < : previous shortened, current dotted
                    if result:
                        pn, pd = result[-1]
                        result[-1] = (pn, pd * shortened)
                    duration *= dotted
                broken_rhythm = None

            result.append((note_name, duration))

        metadata["actual_bpm"] = bpm
        return metadata, result
