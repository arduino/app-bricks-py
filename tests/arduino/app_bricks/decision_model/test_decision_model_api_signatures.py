# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""API/signature stability tests for the DecisionModel brick.

The README snippets and the code generated from them call these methods by keyword and by
position: a rename, reordering or kind change of a parameter breaks user code silently.
Update the fingerprints deliberately (with a matching changelog entry) when the public contract changes.
"""

import inspect

import pytest

import arduino.app_bricks.decision_model as package
from arduino.app_bricks.decision_model import (
    Answers,
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionModel,
    DecisionModelError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from arduino.app_utils import AppError

_P = inspect.Parameter


def _signature_fingerprint(cls, method_name):
    """Name, kind and has-default of every parameter: default values themselves are not part of the contract."""
    sig = inspect.signature(getattr(cls, method_name))
    return [(name, p.kind, p.default is not inspect.Parameter.empty) for name, p in sig.parameters.items()]


EXPECTED_SIGNATURES = {
    "__init__": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("model", _P.POSITIONAL_OR_KEYWORD, True),
        ("timeout", _P.POSITIONAL_OR_KEYWORD, True),
    ],
    "start": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "decide": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("state", _P.POSITIONAL_OR_KEYWORD, False),
        ("questions", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "choose": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("state", _P.POSITIONAL_OR_KEYWORD, False),
        ("instructions", _P.POSITIONAL_OR_KEYWORD, False),
        ("options", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "score": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("state", _P.POSITIONAL_OR_KEYWORD, False),
        ("instructions", _P.POSITIONAL_OR_KEYWORD, False),
        ("levels", _P.POSITIONAL_OR_KEYWORD, False),
    ],
    "check": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
        ("state", _P.POSITIONAL_OR_KEYWORD, False),
        ("instructions", _P.POSITIONAL_OR_KEYWORD, False),
        ("true_description", _P.POSITIONAL_OR_KEYWORD, True),
        ("false_description", _P.POSITIONAL_OR_KEYWORD, True),
    ],
    "list_models": [
        ("self", _P.POSITIONAL_OR_KEYWORD, False),
    ],
}


@pytest.mark.parametrize("method_name", sorted(EXPECTED_SIGNATURES))
def test_public_signatures_are_stable(method_name):
    assert _signature_fingerprint(DecisionModel, method_name) == EXPECTED_SIGNATURES[method_name]


def test_question_dataclass_fields_are_stable():
    assert [f.name for f in Choice.__dataclass_fields__.values()] == ["instructions", "options"]
    assert [f.name for f in Score.__dataclass_fields__.values()] == ["instructions", "levels"]
    assert [f.name for f in Noul.__dataclass_fields__.values()] == ["instructions", "true_description", "false_description"]


def test_answer_dataclass_fields_and_properties_are_stable():
    assert [f.name for f in ChoiceAnswer.__dataclass_fields__.values()] == ["choice", "probabilities", "confidence"]
    assert [f.name for f in ScoreAnswer.__dataclass_fields__.values()] == ["score", "legend", "probabilities", "confidence"]
    assert [f.name for f in NoulAnswer.__dataclass_fields__.values()] == ["probability"]
    assert isinstance(ScoreAnswer.__dict__["level"], property)
    assert isinstance(NoulAnswer.__dict__["is_true"], property)


def test_package_exports_exactly_the_public_names():
    assert package.__all__ == [
        "DecisionModel",
        "DecisionModelError",
        "Choice",
        "Score",
        "Noul",
        "Question",
        "ChoiceAnswer",
        "ScoreAnswer",
        "NoulAnswer",
        "Answer",
        "Answers",
    ]
    for name in package.__all__:
        assert hasattr(package, name), f"{name} is listed in __all__ but not importable"


def test_type_aliases_cover_the_question_and_answer_classes():
    assert set(Question.__args__) == {Choice, Score, Noul}
    assert set(Answer.__args__) == {ChoiceAnswer, ScoreAnswer, NoulAnswer}


def test_error_is_a_user_facing_app_error():
    assert issubclass(DecisionModelError, AppError)
    assert DecisionModel.LLAMACPP_MODEL == "llamacpp"


def test_answers_is_a_dict_with_one_typed_accessor_per_question_type():
    assert issubclass(Answers, dict)
    assert _signature_fingerprint(Answers, "choice") == [("self", _P.POSITIONAL_OR_KEYWORD, False), ("question_id", _P.POSITIONAL_OR_KEYWORD, False)]
    assert _signature_fingerprint(Answers, "score") == [("self", _P.POSITIONAL_OR_KEYWORD, False), ("question_id", _P.POSITIONAL_OR_KEYWORD, False)]
    assert _signature_fingerprint(Answers, "noul") == [("self", _P.POSITIONAL_OR_KEYWORD, False), ("question_id", _P.POSITIONAL_OR_KEYWORD, False)]
    assert inspect.signature(DecisionModel.decide).return_annotation is Answers
