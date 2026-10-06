# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests of the DecisionModel brick against a fake llama.cpp `/v1/systemone` runner."""

from enum import Enum

import pytest
import requests

import arduino.app_bricks.decision_model.decision_model as dm
from arduino.app_bricks.decision_model import (
    Answers,
    Choice,
    ChoiceAnswer,
    DecisionModel,
    DecisionModelError,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from arduino.app_utils import App

MODULE = "arduino.app_bricks.decision_model.decision_model"
MODEL_ID = "llamacpp:Laya-Q8_0"
MODEL_NAME = "Laya-Q8_0"
BASE_URL = "http://127.0.0.1:9999"


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("not json")
        return self._json_data


class Team(Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"


class IntTeam(Enum):
    ONE = 1
    TWO = 2


class AliasedTeam(Enum):
    A = "same"
    B = "same"


CHOICE_RAW = {"choice": "billing", "probabilities": {"billing": 0.85, "shipping": 0.15}, "confidence": 0.7}
SCORE_RAW = {"score": 1.8, "legend": ["low", "mid", "high"], "probabilities": [0.1, 0.0, 0.9], "confidence": 0.58}
NOUL_RAW = {"noul": 0.92}
# What llama.cpp build 11441 really returns for a score question (recorded on an UNO Q, Laya-Q8_0): the
# `legend` and `probabilities` arrays of the README come back as objects keyed by the index, and every
# answer carries its `type`.
SCORE_RAW_INDEXED = {
    "type": "score",
    "score": 0.9783136922867559,
    "legend": {"0": "low", "1": "medium", "2": "high", "3": "critical"},
    "probabilities": {"0": 0.22455608332934687, "1": 0.6238827455571152, "2": 0.10025256661097276, "3": 0.05130860450256507},
    "confidence": 0.5725741410545502,
}


def answers_response(**answers):
    return FakeResponse(json_data={"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}})


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr(f"{MODULE}.time.sleep", lambda s: recorded.append(s))
    return recorded


@pytest.fixture
def runner(monkeypatch, sleeps):
    """Fakes the runner: records posts, serves scripted responses, lists the configured model."""

    class Runner:
        posts: list[dict] = []
        responses: list = []
        get_outcomes: list = []

        def post(self, url, json=None, **kwargs):
            self.posts.append({"url": url, "json": json, **kwargs})
            outcome = self.responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def get(self, url, **kwargs):
            if self.get_outcomes:
                outcome = self.get_outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
            return FakeResponse(json_data={"data": [{"id": MODEL_NAME}]})

    fake = Runner()
    monkeypatch.setattr(f"{MODULE}.resolve_address", lambda host: "127.0.0.1")
    monkeypatch.setattr(f"{MODULE}.requests.post", fake.post)
    monkeypatch.setattr(f"{MODULE}.requests.get", fake.get)
    fake.sleeps = sleeps
    return fake


@pytest.fixture
def brick(runner):
    instance = DecisionModel(model=MODEL_ID)
    App.unregister(instance)
    return instance


# ---------------------------------------------------------------------------
# Question validation and request building
# ---------------------------------------------------------------------------


def test_choice_with_mapping_keeps_descriptions():
    question = Choice("Pick", {"a": "first", "b": None})
    assert dm._question_to_request(question) == {"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": None}}


def test_choice_with_sequence_has_null_descriptions():
    assert dm._question_to_request(Choice("Pick", ["a", "b"]))["criteria"] == {"a": None, "b": None}
    assert dm._question_to_request(Choice("Pick", ("a",)))["criteria"] == {"a": None}


def test_choice_with_enum_uses_member_values():
    assert dm._question_to_request(Choice("Pick", Team))["criteria"] == {"hardware": None, "software": None}


def test_choice_accepts_structured_instructions():
    instructions = {"task": "route", "notes": ["be strict"]}
    assert dm._question_to_request(Choice(instructions, ["a"]))["instructions"] is instructions


@pytest.mark.parametrize(
    "options, error",
    [
        ("abc", TypeError),
        (b"abc", TypeError),
        ([], ValueError),
        ({}, ValueError),
        ({1: "one"}, TypeError),
        ({"a": 1}, TypeError),
        ([1, 2], TypeError),
        (["a", "a"], ValueError),
        (IntTeam, TypeError),
        (AliasedTeam, ValueError),
        (42, TypeError),
    ],
)
def test_choice_rejects_bad_options(options, error):
    with pytest.raises(error):
        Choice("Pick", options)


@pytest.mark.parametrize("instructions, error", [("", ValueError), (b"x", TypeError), (3, TypeError), ({}, ValueError)])
def test_questions_reject_bad_instructions(instructions, error):
    with pytest.raises(error):
        Choice(instructions, ["a"])
    with pytest.raises(error):
        Score(instructions, ["a", "b"])
    with pytest.raises(error):
        Noul(instructions)


def test_score_request_lists_levels_lowest_first():
    assert dm._question_to_request(Score("Grade", ("low", "high"))) == {"type": "score", "instructions": "Grade", "criteria": ["low", "high"]}


@pytest.mark.parametrize(
    "levels, error",
    [("ab", TypeError), (["only"], ValueError), ([str(i) for i in range(11)], ValueError), (["a", 1], TypeError), (None, TypeError)],
)
def test_score_rejects_bad_levels(levels, error):
    with pytest.raises(error):
        Score("Grade", levels)


def test_score_accepts_boundary_level_counts():
    Score("Grade", ["a", "b"])
    Score("Grade", [str(i) for i in range(10)])


def test_noul_request_omits_criteria_unless_both_descriptions_given():
    assert dm._question_to_request(Noul("Angry?")) == {"type": "noul", "instructions": "Angry?"}
    assert dm._question_to_request(Noul("Angry?", "yes when", "no when"))["criteria"] == {"true": "yes when", "false": "no when"}


@pytest.mark.parametrize("kwargs", [{"true_description": "yes"}, {"false_description": "no"}])
def test_noul_rejects_one_sided_descriptions(kwargs):
    with pytest.raises(ValueError):
        Noul("Angry?", **kwargs)


def test_questions_are_frozen():
    with pytest.raises(AttributeError):
        Noul("Angry?").instructions = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# decide(): request and parsing
# ---------------------------------------------------------------------------


def test_decide_posts_the_systemone_body(brick, runner):
    runner.responses = [answers_response(route=CHOICE_RAW, urgency=SCORE_RAW, angry=NOUL_RAW)]

    answers = brick.decide(
        "I was charged twice",
        {
            "route": Choice("Department?", {"billing": "money", "shipping": None}),
            "urgency": Score("Urgency?", ["low", "mid", "high"]),
            "angry": Noul("Angry?"),
        },
    )

    assert len(runner.posts) == 1
    post = runner.posts[0]
    assert post["url"] == f"{BASE_URL}/v1/systemone"
    assert post["timeout"] == dm.DEFAULT_TIMEOUT_S
    assert post["json"] == {
        "model": MODEL_NAME,
        "state": "I was charged twice",
        "questions": {
            "route": {"type": "choice", "instructions": "Department?", "criteria": {"billing": "money", "shipping": None}},
            "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
            "angry": {"type": "noul", "instructions": "Angry?"},
        },
    }
    assert answers == {
        "route": ChoiceAnswer(choice="billing", probabilities={"billing": 0.85, "shipping": 0.15}, confidence=0.7),
        "urgency": ScoreAnswer(score=1.8, legend=["low", "mid", "high"], probabilities=[0.1, 0.0, 0.9], confidence=0.58),
        "angry": NoulAnswer(probability=0.92),
    }


def test_a_score_answer_keyed_by_index_is_read_in_index_order(brick, runner):
    """The server serves the legend and the probabilities as objects keyed "0", "1", ...; the brick lists them."""
    runner.responses = [answers_response(urgency=SCORE_RAW_INDEXED)]

    answer = brick.decide("s", {"urgency": Score("?", ["low", "medium", "high", "critical"])}).score("urgency")

    assert answer.legend == ["low", "medium", "high", "critical"]
    assert answer.probabilities == pytest.approx([0.22455608332934687, 0.6238827455571152, 0.10025256661097276, 0.05130860450256507])
    assert answer.level == "medium"
    assert answer.score == pytest.approx(0.9783136922867559)


def test_a_score_legend_with_non_index_keys_is_malformed(brick, runner):
    runner.responses = [answers_response(q={"score": 1.0, "legend": {"a": "low", "b": "high"}, "probabilities": [0.5, 0.5]})]

    with pytest.raises(DecisionModelError, match="Malformed answer to question 'q'"):
        brick.decide("s", {"q": Score("?", ["low", "high"])})


def test_a_prompt_too_large_for_the_batch_gets_a_shorten_the_state_hint(brick, runner):
    """What the runner answers to a state longer than the batch of the decision model (observed: 3453 tokens vs 2048)."""
    message = "input (3453 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"
    runner.responses = [FakeResponse(500, {"error": {"code": 500, "message": message, "type": "server_error"}})]

    with pytest.raises(DecisionModelError, match="3453 tokens") as info:
        brick.decide("s" * 10, {"q": Noul("?")})

    assert "Shorten the state" in (info.value.hint or "")


def test_decide_returns_answers_with_typed_accessors(brick, runner):
    """The caller wrote the questions, so it knows their types: the accessors spare it an isinstance test."""
    runner.responses = [answers_response(route=CHOICE_RAW, urgency=SCORE_RAW, angry=NOUL_RAW)]

    answers = brick.decide("s", {"route": Choice("?", ["billing", "shipping"]), "urgency": Score("?", ["low", "mid", "high"]), "angry": Noul("?")})

    assert isinstance(answers, Answers) and isinstance(answers, dict)
    assert answers.choice("route").choice == "billing"
    assert answers.score("urgency").level == "high"
    assert answers.noul("angry").is_true is True


def test_answers_accessors_reject_a_missing_id_and_a_question_of_another_type():
    answers = Answers({"route": ChoiceAnswer(choice="a", probabilities={"a": 1.0}, confidence=1.0), "angry": NoulAnswer(probability=0.2)})

    with pytest.raises(KeyError, match="no question with id 'urgency'"):
        answers.score("urgency")
    with pytest.raises(TypeError, match="question 'angry' was answered with a NoulAnswer, not a ChoiceAnswer"):
        answers.choice("angry")
    with pytest.raises(TypeError, match="not a ScoreAnswer"):
        answers.score("route")
    assert answers.noul("angry") == NoulAnswer(probability=0.2)


@pytest.mark.parametrize("state", [{"temperature": 71.5, "tags": ["hot"]}, [1, 2, 3], "plain text"])
def test_decide_passes_the_state_through_untouched(brick, runner, state):
    runner.responses = [answers_response(q=NOUL_RAW)]

    brick.decide(state, {"q": Noul("Hot?")})

    assert runner.posts[0]["json"]["state"] is state


def test_decide_forwards_the_configured_timeout(runner):
    brick = DecisionModel(model=MODEL_ID, timeout=None)
    App.unregister(brick)
    runner.responses = [answers_response(q=NOUL_RAW)]

    brick.decide("x", {"q": Noul("Hot?")})

    assert runner.posts[0]["timeout"] is None


@pytest.mark.parametrize("state, error", [("", ValueError), ({}, ValueError), (b"x", TypeError), (12, TypeError), (None, TypeError)])
def test_decide_rejects_bad_state(brick, runner, state, error):
    with pytest.raises(error):
        brick.decide(state, {"q": Noul("Hot?")})
    assert runner.posts == []


@pytest.mark.parametrize(
    "questions, error",
    [({}, ValueError), ([Noul("x")], TypeError), ({"q": "not a question"}, TypeError), ({"": Noul("x")}, TypeError), ({1: Noul("x")}, TypeError)],
)
def test_decide_rejects_bad_questions(brick, runner, questions, error):
    with pytest.raises(error):
        brick.decide("state", questions)
    assert runner.posts == []


def test_decide_parses_answers_by_the_question_type_sent(brick, runner):
    # The same raw payload is parsed differently depending on what was asked.
    runner.responses = [answers_response(q={"noul": 0.3, "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0})]
    assert brick.decide("s", {"q": Noul("?")})["q"] == NoulAnswer(probability=0.3)

    runner.responses = [answers_response(q={"noul": 0.3, "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0})]
    assert brick.decide("s", {"q": Choice("?", ["x"])})["q"] == ChoiceAnswer(choice="x", probabilities={"x": 1.0}, confidence=1.0)


def test_decide_raises_when_an_answer_is_missing(brick, runner):
    runner.responses = [answers_response(route=CHOICE_RAW)]

    with pytest.raises(DecisionModelError, match="did not answer question 'angry'") as info:
        brick.decide("s", {"route": Choice("?", ["billing", "shipping"]), "angry": Noul("?")})
    assert info.value.hint


@pytest.mark.parametrize(
    "raw",
    [
        "not an object",
        {"choice": 3, "probabilities": {}},
        {"choice": "a", "probabilities": [0.1]},
        {"choice": "a", "probabilities": {"a": "high"}},
    ],
)
def test_decide_raises_on_malformed_choice_answer(brick, runner, raw):
    runner.responses = [answers_response(q=raw)]
    with pytest.raises(DecisionModelError, match="Malformed answer to question 'q'"):
        brick.decide("s", {"q": Choice("?", ["a"])})


@pytest.mark.parametrize(
    "raw",
    [
        {"score": 1.0, "legend": [], "probabilities": []},
        {"score": "1", "legend": ["a", "b"], "probabilities": [1, 0]},
        {"score": 1.0, "legend": "ab"},
    ],
)
def test_decide_raises_on_malformed_score_answer(brick, runner, raw):
    runner.responses = [answers_response(q=raw)]
    with pytest.raises(DecisionModelError, match="Malformed answer"):
        brick.decide("s", {"q": Score("?", ["a", "b"])})


@pytest.mark.parametrize("raw", [{}, {"noul": "yes"}, {"noul": True}])
def test_decide_raises_on_malformed_noul_answer(brick, runner, raw):
    runner.responses = [answers_response(q=raw)]
    with pytest.raises(DecisionModelError, match="Malformed answer"):
        brick.decide("s", {"q": Noul("?")})


@pytest.mark.parametrize("payload", [{"usage": {}}, [], "text"])
def test_decide_raises_when_answers_are_missing_from_the_response(brick, runner, payload):
    runner.responses = [FakeResponse(json_data=payload)]
    with pytest.raises(DecisionModelError):
        brick.decide("s", {"q": Noul("?")})


def test_decide_raises_on_non_json_success_response(brick, runner):
    runner.responses = [FakeResponse(status_code=200, json_data=None, text="<html>")]
    with pytest.raises(DecisionModelError, match="not valid JSON"):
        brick.decide("s", {"q": Noul("?")})


# ---------------------------------------------------------------------------
# Answer data classes
# ---------------------------------------------------------------------------


def test_score_answer_level_is_the_most_probable_legend_entry():
    answer = ScoreAnswer(score=1.8, legend=["low", "mid", "high"], probabilities=[0.1, 0.0, 0.9], confidence=0.5)
    assert answer.level == "high"
    answer = ScoreAnswer(score=1.8, legend=["low", "mid", "high"], probabilities=[0.2, 0.7, 0.1], confidence=0.5)
    assert answer.level == "mid"


def test_score_answer_level_falls_back_to_rounded_score_when_probabilities_do_not_match():
    assert ScoreAnswer(score=1.4, legend=["low", "mid", "high"], probabilities=[], confidence=0.0).level == "mid"
    assert ScoreAnswer(score=1.6, legend=["low", "mid", "high"], probabilities=[0.5, 0.5], confidence=0.0).level == "high"
    # Clamped into the legend
    assert ScoreAnswer(score=7.0, legend=["low", "mid", "high"], probabilities=[0.5, 0.5], confidence=0.0).level == "high"
    assert ScoreAnswer(score=-3.0, legend=["low", "mid", "high"], probabilities=[], confidence=0.0).level == "low"


def test_score_answer_level_rejects_empty_legend():
    with pytest.raises(ValueError):
        ScoreAnswer(score=0.0, legend=[], probabilities=[], confidence=0.0).level


def test_noul_answer_is_true_at_half_and_above_and_has_no_bool():
    assert NoulAnswer(0.5).is_true is True
    assert NoulAnswer(0.92).is_true is True
    assert NoulAnswer(0.49).is_true is False
    assert "__bool__" not in NoulAnswer.__dict__
    assert bool(NoulAnswer(0.0)) is True  # a dataclass instance, not its probability


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_json_error_message_is_preserved_and_mapped_to_too_many_options_hint(brick, runner):
    error = {"error": {"code": 400, "message": "too many options (30), this model supports at most 20", "type": "invalid_request_error"}}
    runner.responses = [FakeResponse(status_code=400, json_data=error, text="ignored")]

    with pytest.raises(DecisionModelError) as info:
        brick.decide("s", {"q": Choice("?", ["a"])})

    assert "too many options (30), this model supports at most 20" in str(info.value)
    assert "HTTP 400" in str(info.value)
    assert "Reduce the number of options" in info.value.hint


def test_plain_text_error_falls_back_to_the_response_text(brick, runner):
    runner.responses = [FakeResponse(status_code=500, json_data=None, text="something broke\n")]

    with pytest.raises(DecisionModelError, match="something broke") as info:
        brick.decide("s", {"q": Noul("?")})
    assert "logs" in info.value.hint


def test_empty_error_body_reports_the_http_status(brick, runner):
    runner.responses = [FakeResponse(status_code=502, json_data=None, text="")]
    with pytest.raises(DecisionModelError, match="HTTP 502"):
        brick.decide("s", {"q": Noul("?")})


@pytest.mark.parametrize("message", ["model is not loaded", "failed to load model 'Laya-Q8_0'", "model Laya-Q8_0 not found"])
def test_not_loaded_errors_hint_to_download_the_model(brick, runner, message):
    runner.responses = [FakeResponse(status_code=400, json_data={"error": {"code": 400, "message": message, "type": "x"}})]

    with pytest.raises(DecisionModelError) as info:
        brick.decide("s", {"q": Noul("?")})

    assert message in str(info.value)
    assert MODEL_ID in info.value.hint
    assert "Download the model" in info.value.hint


def test_unsupported_decision_model_type_hints_to_pick_a_decision_model(brick, runner):
    runner.responses = [FakeResponse(status_code=400, json_data={"error": {"message": "unsupported decision model type"}})]

    with pytest.raises(DecisionModelError) as info:
        brick.decide("s", {"q": Noul("?")})

    assert "is not a decision model" in info.value.hint
    assert MODEL_ID in info.value.hint


@pytest.mark.parametrize("body", [{"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}}, None])
def test_404_hints_at_the_runner_image_not_at_the_model(brick, runner, body):
    # A runner image older than llamacpp/20261006 has no /v1/systemone endpoint: "File Not Found" is not a missing model.
    runner.responses = [FakeResponse(status_code=404, json_data=body, text="File Not Found")]

    with pytest.raises(DecisionModelError, match="File Not Found") as info:
        brick.decide("s", {"q": Noul("?")})

    assert "/v1/systemone" in info.value.hint
    assert "Download the model" not in info.value.hint


def test_string_error_field_is_used_as_message(brick, runner):
    runner.responses = [FakeResponse(status_code=400, json_data={"error": '"questions" must be a non-empty object'})]
    with pytest.raises(DecisionModelError, match='"questions" must be a non-empty object'):
        brick.decide("s", {"q": Noul("?")})


def test_connection_error_is_reported_with_the_url(brick, runner):
    runner.responses = [requests.ConnectionError("refused")]

    with pytest.raises(DecisionModelError, match=f"Cannot reach the llama.cpp models runner at {BASE_URL}/v1/systemone") as info:
        brick.decide("s", {"q": Noul("?")})

    assert "arduino:llamacpp" in info.value.hint
    assert isinstance(info.value.__cause__, requests.ConnectionError)


def test_timeout_hints_to_raise_the_timeout(brick, runner):
    runner.responses = [requests.ReadTimeout("slow")]

    with pytest.raises(DecisionModelError, match=f"did not answer within {dm.DEFAULT_TIMEOUT_S} s") as info:
        brick.decide("s", {"q": Noul("?")})

    assert "timeout" in info.value.hint


def test_connect_timeout_counts_as_unreachable(brick, runner):
    # requests.ConnectTimeout is both a ConnectionError and a Timeout: the runner is not there.
    runner.responses = [requests.ConnectTimeout("no route")]
    with pytest.raises(DecisionModelError, match="Cannot reach"):
        brick.decide("s", {"q": Noul("?")})


def test_other_request_exceptions_become_decision_model_errors(brick, runner):
    runner.responses = [requests.exceptions.InvalidURL("bad")]
    with pytest.raises(DecisionModelError, match="Request to the llama.cpp models runner failed"):
        brick.decide("s", {"q": Noul("?")})


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def test_explicit_model_strips_the_prefix_and_keeps_the_id(brick):
    assert brick._model_name == MODEL_NAME
    assert brick._model_id == MODEL_ID
    assert brick._base_url == BASE_URL


def test_model_comes_from_the_brick_configuration_when_not_given(runner, monkeypatch):
    seen: dict = {}

    def fake_configured_model(brick_id, brick_config=None):
        seen["brick_id"] = brick_id
        seen["brick_config"] = brick_config
        return "llamacpp:Julia-1-Q8_0"

    monkeypatch.setattr(f"{MODULE}.get_brick_configured_model", fake_configured_model)

    brick = DecisionModel()
    App.unregister(brick)

    assert seen["brick_id"] == "arduino:decision_model"
    assert seen["brick_config"]["id"] == "arduino:decision_model"
    assert brick._model_name == "Julia-1-Q8_0"


@pytest.mark.parametrize(("board", "expected"), [("unoq", "llamacpp:Julia-1-Q8_0"), ("ventunoq", "llamacpp:Laya-Q8_0")])
def test_default_brick_configuration_is_julia_on_unoq_and_laya_on_ventunoq(runner, monkeypatch, board, expected):
    """Measured on the UNO Q: Laya takes ~9 s for three questions where Julia-1 takes ~1 s, so the
    CPU board defaults to the small model and the NPU board to the more accurate one."""
    # No app.yaml in the test environment: the brick default (model_by_boards) applies.
    monkeypatch.setattr("arduino.app_internal.core.module.get_board_name", lambda: board)
    brick = DecisionModel()
    App.unregister(brick)
    assert brick._model_id == expected


def test_genie_model_is_rejected(runner):
    with pytest.raises(DecisionModelError, match="Unsupported model 'genie:qwen'") as info:
        DecisionModel(model="genie:qwen")
    assert "llamacpp:Laya-Q8_0" in info.value.hint
    assert "genie" in info.value.hint


@pytest.mark.parametrize("model", ["Laya-Q8_0", "llamacpp:", "llamacpp"])
def test_models_without_the_llamacpp_prefix_are_rejected(runner, model):
    with pytest.raises(DecisionModelError, match="Unsupported model"):
        DecisionModel(model=model)


@pytest.mark.parametrize("configured", [None, "", "   "])
def test_missing_configuration_is_rejected(runner, monkeypatch, configured):
    monkeypatch.setattr(f"{MODULE}.get_brick_configured_model", lambda brick_id, brick_config=None: configured)
    with pytest.raises(DecisionModelError, match="No model configured") as info:
        DecisionModel()
    assert "app.yaml" in info.value.hint


def test_unresolvable_host_is_rejected(monkeypatch):
    monkeypatch.setattr(f"{MODULE}.resolve_address", lambda host: "")
    with pytest.raises(DecisionModelError, match="Host address resolution failed"):
        DecisionModel(model=MODEL_ID)


def test_construction_logs_an_error_when_the_model_is_not_served(runner, monkeypatch):
    # The brick logger does not propagate to the root logger, so caplog cannot see it.
    errors: list[str] = []
    monkeypatch.setattr(dm.logger, "error", lambda message, *args, **kwargs: errors.append(message))
    runner.get_outcomes = [FakeResponse(json_data={"data": [{"id": "other"}]})]

    brick = DecisionModel(model=MODEL_ID)
    App.unregister(brick)

    assert len(errors) == 1
    assert f"Model '{MODEL_NAME}' not found" in errors[0]
    assert "['other']" in errors[0]


def test_construction_does_not_log_an_error_when_the_model_is_served(runner, monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(dm.logger, "error", lambda message, *args, **kwargs: errors.append(message))

    brick = DecisionModel(model=MODEL_ID)
    App.unregister(brick)

    assert errors == []


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


def test_list_models_queries_v1_models(brick, runner, monkeypatch):
    urls: list[str] = []

    def get(url, **kwargs):
        urls.append(url)
        return FakeResponse(json_data={"data": [{"id": "Laya-Q8_0"}, {"id": "Kev-0.8B-Q8_0"}]})

    monkeypatch.setattr(f"{MODULE}.requests.get", get)

    assert brick.list_models() == ["Laya-Q8_0", "Kev-0.8B-Q8_0"]
    assert urls == [f"{BASE_URL}/v1/models"]


def test_list_models_retries_connection_errors_until_the_runner_is_up(brick, runner):
    runner.sleeps.clear()
    runner.get_outcomes = [requests.ConnectionError(), requests.ConnectionError(), FakeResponse(json_data={"data": [{"id": MODEL_NAME}]})]

    assert brick.list_models() == [MODEL_NAME]
    assert runner.sleeps == [dm.LIST_MODELS_RETRY_DELAY_S] * 2


def test_list_models_gives_up_after_max_attempts(brick, runner):
    runner.sleeps.clear()
    runner.get_outcomes = [requests.ConnectionError() for _ in range(dm.LIST_MODELS_MAX_ATTEMPTS)]

    assert brick.list_models() == []
    assert runner.sleeps == [dm.LIST_MODELS_RETRY_DELAY_S] * (dm.LIST_MODELS_MAX_ATTEMPTS - 1)
    assert runner.get_outcomes == []


@pytest.mark.parametrize(
    "outcome",
    [FakeResponse(status_code=500, json_data=None, text="boom"), FakeResponse(json_data={"models": []}), requests.ReadTimeout(), ValueError("x")],
)
def test_list_models_returns_empty_on_other_failures_without_retrying(brick, runner, outcome):
    runner.sleeps.clear()
    runner.get_outcomes = [outcome, FakeResponse(json_data={"data": [{"id": MODEL_NAME}]})]

    assert brick.list_models() == []
    assert runner.sleeps == []
    assert len(runner.get_outcomes) == 1


# ---------------------------------------------------------------------------
# start(): warm-up
# ---------------------------------------------------------------------------


def test_start_warms_up_with_a_noul_question_and_a_long_timeout(brick, runner):
    runner.responses = [answers_response(ready={"noul": 0.5})]

    brick.start()

    assert runner.posts == [
        {
            "url": f"{BASE_URL}/v1/systemone",
            "json": {
                "model": MODEL_NAME,
                "state": "warm-up",
                "questions": {"ready": {"type": "noul", "instructions": "Is the decision model ready?"}},
            },
            "timeout": dm.WARMUP_TIMEOUT_S,
        }
    ]


def test_start_raises_when_the_model_cannot_be_loaded(brick, runner):
    runner.responses = [FakeResponse(status_code=500, json_data={"error": {"message": "failed to load model"}})]

    with pytest.raises(DecisionModelError, match="failed to load model") as info:
        brick.start()
    assert "Download the model" in info.value.hint


def test_start_raises_when_the_runner_is_unreachable(brick, runner):
    runner.responses = [requests.ConnectionError()]
    with pytest.raises(DecisionModelError, match="Cannot reach"):
        brick.start()


# ---------------------------------------------------------------------------
# Shortcuts
# ---------------------------------------------------------------------------


def test_choose_sends_one_choice_and_returns_a_choice_answer(brick, runner):
    runner.responses = [answers_response(answer={"choice": "software", "probabilities": {"hardware": 0.2, "software": 0.8}, "confidence": 0.6})]

    answer = brick.choose({"ticket": "The app crashes"}, "Which team?", Team)

    assert runner.posts[0]["json"]["questions"] == {
        "answer": {"type": "choice", "instructions": "Which team?", "criteria": {"hardware": None, "software": None}}
    }
    assert answer == ChoiceAnswer(choice="software", probabilities={"hardware": 0.2, "software": 0.8}, confidence=0.6)
    assert Team(answer.choice) is Team.SOFTWARE


def test_score_sends_one_score_and_returns_a_score_answer(brick, runner):
    runner.responses = [answers_response(answer=SCORE_RAW)]

    answer = brick.score("text", "Urgency?", ["low", "mid", "high"])

    assert runner.posts[0]["json"]["questions"] == {"answer": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]}}
    assert isinstance(answer, ScoreAnswer)
    assert answer.level == "high"


def test_check_sends_one_noul_and_returns_a_noul_answer(brick, runner):
    runner.responses = [answers_response(answer=NOUL_RAW)]

    answer = brick.check("text", "Angry?", "The customer is upset", "The customer is calm")

    assert runner.posts[0]["json"]["questions"] == {
        "answer": {"type": "noul", "instructions": "Angry?", "criteria": {"true": "The customer is upset", "false": "The customer is calm"}}
    }
    assert answer == NoulAnswer(probability=0.92)
    assert answer.is_true is True


def test_shortcuts_validate_before_posting(brick, runner):
    with pytest.raises(TypeError):
        brick.choose("text", "Which?", "abc")
    with pytest.raises(ValueError):
        brick.score("text", "How?", ["only"])
    with pytest.raises(ValueError):
        brick.check("text", "Is?", true_description="yes")
    assert runner.posts == []
