# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.sound_generator import ABCNotationLoader


def test_abc_loader():
    full_abc = """
    X:1
    T:Main Theme
    M:4/4
    L:1/8
    Q:1/4=130
    K:Cm
    "Cm"E2 E2 E2 "Ab"C>G | "Cm"E2 "Ab"C>G "Cm"E4 |
    "Cm"B2 B2 B2 "Ab"c>G | "Fm"^D#2 "Ab"C>G "Cm"E4 |
    """

    # K:Cm applies 3 flats: Bb, Eb, Ab.
    # C>G is broken rhythm: first note 3/2x, second 1/2x default duration.
    reference_notes = [
        ("EB4", 60 / 130),
        ("EB4", 60 / 130),
        ("EB4", 60 / 130),
        ("C4", (60 / 130) * 3 / 4),  # C from C>G (default * 3/2)
        ("G4", (60 / 130) / 4),  # G from C>G (default * 1/2)
        ("EB4", 60 / 130),
        ("C4", (60 / 130) * 3 / 4),
        ("G4", (60 / 130) / 4),
        ("EB4", (60 / 130) * 2),
        ("BB4", 60 / 130),
        ("BB4", 60 / 130),
        ("BB4", 60 / 130),
        ("C5", (60 / 130) * 3 / 4),  # c (lowercase = octave+1, C natural in Cm)
        ("G4", (60 / 130) / 4),
        ("D#4", 60 / 130),  # ^D# prefix sharp (D not in Cm key)
        ("C4", (60 / 130) * 3 / 4),
        ("G4", (60 / 130) / 4),
        ("EB4", (60 / 130) * 2),
    ]

    metadata, loaded = ABCNotationLoader.parse_abc_notation(full_abc)
    assert metadata["title"] == "Main Theme"
    assert "transpose" not in metadata
    assert metadata["tempo"] == "1/4=130"

    i_ref = 0
    for note, duration in loaded:
        print(f"Note: {note}, Duration: {duration}")
        assert note == reference_notes[i_ref][0]
        assert abs(duration - reference_notes[i_ref][1]) < 0.01
        i_ref += 1


# ---------------------------------------------------------------------------
# ABC 2.1 compliance tests
# ---------------------------------------------------------------------------


def _parse(body: str, key: str = "C", meter: str = "4/4", length: str = "1/4", tempo: str = "1/4=120"):
    """Helper: wrap a music body in minimal ABC headers and parse."""
    abc = f"X:1\nM:{meter}\nL:{length}\nQ:{tempo}\nK:{key}\n{body}\n"
    return ABCNotationLoader.parse_abc_notation(abc)


def test_key_none():
    """K:none should produce no accidentals — all notes natural."""
    _, notes = _parse("F c B", key="none")
    assert notes[0][0] == "F4"  # F natural (no key)
    assert notes[1][0] == "C5"  # c = C5
    assert notes[2][0] == "B4"  # B natural


def test_key_highland_bagpipe():
    """K:Hp should sharpen F and C only."""
    _, notes = _parse("F C D", key="Hp")
    assert notes[0][0] == "F#4"
    assert notes[1][0] == "C#4"
    assert notes[2][0] == "D4"


def test_key_explicit():
    """K:C exp ^f ^c — explicit mode with only the listed accidentals."""
    _, notes = _parse("F C G", key="C exp ^f ^c")
    assert notes[0][0] == "F#4"
    assert notes[1][0] == "C#4"
    assert notes[2][0] == "G4"


def test_key_inline_accidental_override():
    """K:D =f — D major (F# C#) but F overridden to natural."""
    _, notes = _parse("F C G", key="D =f")
    assert notes[0][0] == "F4"  # natural — override removed F#
    assert notes[1][0] == "C#4"  # still sharp from D major
    assert notes[2][0] == "G4"


def test_repeated_slashes():
    """// = /4, /// = /8 duration shorthand."""
    beat = 0.5  # Q:1/4=120 → 0.5s per quarter
    _, notes = _parse("C C/ C// C///")
    assert abs(notes[0][1] - beat) < 0.001  # C  = 1/4
    assert abs(notes[1][1] - beat / 2) < 0.001  # C/ = 1/8
    assert abs(notes[2][1] - beat / 4) < 0.001  # C// = 1/16
    assert abs(notes[3][1] - beat / 8) < 0.001  # C/// = 1/32


def test_barline_variants_reset_accidentals():
    """All standard barlines should reset bar-local accidentals."""
    # ^F sets bar accidental on F → after barline, F should revert to natural
    for barline in ["|", "||", "|]", "[|", ":|", "|:", ".|"]:
        _, notes = _parse(f"^F {barline} F", key="C")
        assert notes[0][0] == "F#4", f"Before {barline}: expected F#4"
        assert notes[1][0] == "F4", f"After {barline}: expected F4 (accidental reset)"


def test_invisible_rest():
    """x (invisible rest) should behave like z for playback."""
    beat = 0.5
    _, notes = _parse("C x2 D")
    assert notes[0][0] == "C4"
    assert notes[1] == ("REST", beat * 2)
    assert notes[2][0] == "D4"


def test_multimeasure_rest():
    """Z = one measure rest, Z3 = three measures. M:4/4, Q:1/4=120 → 2s/measure."""
    measure = 2.0  # 4 beats × 0.5s
    _, notes = _parse("C Z D", meter="4/4")
    assert notes[0][0] == "C4"
    assert notes[1] == ("REST", measure)
    assert notes[2][0] == "D4"

    _, notes2 = _parse("Z3", meter="3/4")
    measure_34 = 1.5  # 3 beats × 0.5s
    assert abs(notes2[0][1] - measure_34 * 3) < 0.001


def test_chord_brackets_flatten():
    """[CEG] should emit C, E, G as sequential notes (simplified)."""
    _, notes = _parse("[CEG]")
    assert len(notes) == 3
    assert notes[0][0] == "C4"
    assert notes[1][0] == "E4"
    assert notes[2][0] == "G4"


def test_tuplet_basic():
    """(3CEG — triplet: 3 notes in time of 2, each at 2/3 default duration."""
    beat = 0.5
    _, notes = _parse("(3CEG")
    assert len(notes) == 3
    expected = beat * 2 / 3
    for n in notes:
        assert abs(n[1] - expected) < 0.001


def test_tuplet_pqr():
    """(5:4:5 — 5 notes in time of 4."""
    beat = 0.5
    _, notes = _parse("(5:4:5CDEFG A")
    # First 5 notes at 4/5 duration, last note normal
    expected_tuplet = beat * 4 / 5
    for n in notes[:5]:
        assert abs(n[1] - expected_tuplet) < 0.001
    assert abs(notes[5][1] - beat) < 0.001


def test_grace_notes_stripped():
    """{abc}D — grace notes are stripped, only D remains."""
    _, notes = _parse("{gag}D")
    assert len(notes) == 1
    assert notes[0][0] == "D4"


def test_decorations_stripped():
    """!ff!D and +fermata+D — decorations stripped."""
    _, notes = _parse("!ff!D")
    assert len(notes) == 1
    assert notes[0][0] == "D4"

    _, notes2 = _parse("+fermata+D")
    assert len(notes2) == 1
    assert notes2[0][0] == "D4"


def test_abc_loader_with_transpose():
    full_abc = """
    X:1
    T:Main Theme
    M:4/4
    L:1/8
    Q:1/4=130
    K:Cm
    %%transpose -12
    "Cm"E2 E2 E2 "Ab"C>G | "Cm"E2 "Ab"C>G "Cm"E4 |
    "Cm"B2 B2 B2 "Ab"c>G | "Fm"^D#2 "Ab"C>G "Cm"E4 |
    """

    reference_notes = [
        ("EB3", 60 / 130),
        ("EB3", 60 / 130),
        ("EB3", 60 / 130),
        ("C3", (60 / 130) * 3 / 4),
        ("G3", (60 / 130) / 4),
        ("EB3", 60 / 130),
        ("C3", (60 / 130) * 3 / 4),
        ("G3", (60 / 130) / 4),
        ("EB3", (60 / 130) * 2),
        ("BB3", 60 / 130),
        ("BB3", 60 / 130),
        ("BB3", 60 / 130),
        ("C4", (60 / 130) * 3 / 4),
        ("G3", (60 / 130) / 4),
        ("D#3", 60 / 130),
        ("C3", (60 / 130) * 3 / 4),
        ("G3", (60 / 130) / 4),
        ("EB3", (60 / 130) * 2),
    ]

    metadata, loaded = ABCNotationLoader.parse_abc_notation(full_abc)
    assert metadata["title"] == "Main Theme"
    assert "transpose" in metadata
    assert metadata["transpose"] == -1
    assert metadata["tempo"] == "1/4=130"

    i_ref = 0
    for note, duration in loaded:
        print(f"Note: {note}, Duration: {duration}")
        assert note == reference_notes[i_ref][0]
        assert abs(duration - reference_notes[i_ref][1]) < 0.01
        i_ref += 1
