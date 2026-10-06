# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

import requests

from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model
from arduino.app_utils import AppError, Logger, brick

logger = Logger("DecisionModel")

# The local models runner lives in a sibling container that may still be starting up when the
# brick is constructed. Connection errors on the model listing are therefore retried before giving up.
LIST_MODELS_MAX_ATTEMPTS = 10
LIST_MODELS_RETRY_DELAY_S = 1.0
LIST_MODELS_TIMEOUT_S = 10.0
DEFAULT_TIMEOUT_S = 60.0
# The first request loads the model into memory, which takes much longer than answering.
WARMUP_TIMEOUT_S = 300.0
RUNNER_HOST = "llamacpp-models-runner"
RUNNER_PORT = 9999

MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

_CONFIGURE_MODEL_HINT = (
    "Configure a llama.cpp decision model such as Laya in app.yaml under arduino:decision_model;"
    " the DecisionModel brick does not support genie models."
)
_RUNNER_RESPONSE_HINT = (
    "The llama.cpp models runner returned an unexpected payload: make sure it supports the /v1/systemone endpoint"
    " and check runner logs."
)


# What `json.dumps` can serialize: the state and the instructions are sent to the model as JSON.
type Json = None | bool | int | float | str | Sequence[Json] | Mapping[str, Json]


class DecisionModelError(AppError):
    """Raised when the decision model cannot be reached, is misconfigured or rejects a request."""


def _as_mapping(value: object) -> Mapping[object, object] | None:
    """The value as a JSON object, or None when it is not one.

    Args:
        value (object): A decoded JSON value, or anything else.

    Returns:
        Mapping[object, object] | None: The mapping, with its keys and values still to be checked.
    """
    return cast(Mapping[object, object], value) if isinstance(value, Mapping) else None


def _as_sequence(value: object) -> Sequence[object] | None:
    """The value as a JSON array, or None for anything else: a string is not an array.

    Args:
        value (object): A decoded JSON value, or anything else.

    Returns:
        Sequence[object] | None: The sequence, with its items still to be checked.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    return cast(Sequence[object], value)


def _json_length(value: object) -> int | None:
    """The length of a string, JSON object or JSON array, None for anything else.

    Args:
        value (object): The value.

    Returns:
        int | None: Its length, or None when it is none of the three.
    """
    if isinstance(value, str):
        return len(value)
    mapping = _as_mapping(value)
    if mapping is not None:
        return len(mapping)
    sequence = _as_sequence(value)
    return None if sequence is None else len(sequence)


def _validate_instructions(instructions: object) -> None:
    """Checks that the instructions of a question are a non-empty string or a JSON object/array.

    Args:
        instructions (object): The value to validate.

    Raises:
        TypeError: If the instructions are not a string, a mapping or a sequence.
        ValueError: If the instructions are empty.
    """
    length = _json_length(instructions)
    if length is None:
        raise TypeError(f"instructions must be a string, a dict or a list, not {type(instructions).__name__}")
    if length == 0:
        raise ValueError("instructions must not be empty")


def _validate_state(state: object) -> None:
    """Checks that a state is a non-empty string or a JSON object/array.

    Args:
        state (object): The value to validate.

    Raises:
        TypeError: If the state is not a string, a mapping or a sequence.
        ValueError: If the state is empty.
    """
    length = _json_length(state)
    if length is None:
        raise TypeError(f"state must be a string, a dict or a list, not {type(state).__name__}")
    if length == 0:
        raise ValueError("state must not be empty")


def _normalize_options(options: object) -> dict[str, str | None]:
    """Turns the options of a choice question into the `{option: description | None}` criteria the server expects.

    Args:
        options (object): A mapping of option to description (or None), a sequence of option names or an Enum class.

    Returns:
        dict[str, str | None]: The options in the server format, in the order they were given.

    Raises:
        TypeError: If the options are a single string, or contain non-string names or descriptions.
        ValueError: If there are no options or the same option appears twice.
    """
    normalized: dict[str, str | None] = {}
    if isinstance(options, type) and issubclass(options, Enum):
        # __members__ includes aliases, so duplicated values are detected
        for member in options.__members__.values():
            value: object = member.value
            if not isinstance(value, str):
                raise TypeError(f"Enum options must have string values: {options.__name__}.{member.name} has {type(value).__name__}")
            if value in normalized:
                raise ValueError(f"Enum options must have distinct values: '{value}' appears more than once in {options.__name__}")
            normalized[value] = None
    elif isinstance(options, (str, bytes)):
        raise TypeError("options must be a dict, a list of strings or an Enum class, not a single string")
    elif (mapping := _as_mapping(options)) is not None:
        for key, description in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f"option names must be strings, not {type(key).__name__}")
            if description is not None and not isinstance(description, str):
                raise TypeError(f"option descriptions must be strings or None, not {type(description).__name__}")
            normalized[key] = description
    elif (sequence := _as_sequence(options)) is not None:
        for option in sequence:
            if not isinstance(option, str):
                raise TypeError(f"option names must be strings, not {type(option).__name__}")
            if option in normalized:
                raise ValueError(f"options must be distinct: '{option}' appears more than once")
            normalized[option] = None
    else:
        raise TypeError(f"options must be a dict, a list of strings or an Enum class, not {type(options).__name__}")
    if not normalized:
        raise ValueError("a choice question needs at least one option")
    return normalized


def _normalize_levels(levels: object) -> list[str]:
    """Checks the levels of a score question and returns them as a list.

    Args:
        levels (object): The level descriptions, lowest first.

    Returns:
        list[str]: The level descriptions.

    Raises:
        TypeError: If the levels are a single string or contain non-string items.
        ValueError: If there are fewer than 2 or more than 10 levels.
    """
    sequence = _as_sequence(levels)
    if sequence is None:
        raise TypeError(f"levels must be a list of strings, not {type(levels).__name__}")
    normalized: list[str] = []
    for level in sequence:
        if not isinstance(level, str):
            raise TypeError(f"level descriptions must be strings, not {type(level).__name__}")
        normalized.append(level)
    if not MIN_SCORE_LEVELS <= len(normalized) <= MAX_SCORE_LEVELS:
        raise ValueError(f"a score question needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {len(normalized)}")
    return normalized


@dataclass(frozen=True)
class Choice:
    """Data class describing a multiple-choice question: the model picks one option.

    Attributes:
        instructions (str | Mapping[str, Json] | Sequence[Json]): What to decide, e.g. "Which team should handle this ticket?".
        options (Mapping[str, str | None] | Sequence[str] | type[Enum]): The options to choose from: a dict mapping each option to a
            description (or None), a list of option names, or an Enum class whose member values are the option names.
            Every decision model has a maximum number of options (255 for Laya and Kev, 20 for Julia-1).
    """

    instructions: str | Mapping[str, Json] | Sequence[Json]
    options: Mapping[str, str | None] | Sequence[str] | type[Enum]

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        _normalize_options(self.options)


@dataclass(frozen=True)
class Score:
    """Data class describing a graded question: the model places the state on a scale of 2 to 10 levels.

    Attributes:
        instructions (str | Mapping[str, Json] | Sequence[Json]): What to grade, e.g. "How urgent is this message?".
        levels (Sequence[str]): The descriptions of the levels, lowest first (between 2 and 10).
    """

    instructions: str | Mapping[str, Json] | Sequence[Json]
    levels: Sequence[str]

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        _normalize_levels(self.levels)


@dataclass(frozen=True)
class Noul:
    """Data class describing a yes/no question: the model returns the probability that the answer is yes.

    Attributes:
        instructions (str | Mapping[str, Json] | Sequence[Json]): The question, e.g. "Is the customer angry?".
        true_description (str | None): Optional description of when the answer is yes. Must be given together with `false_description`.
        false_description (str | None): Optional description of when the answer is no. Must be given together with `true_description`.
    """

    instructions: str | Mapping[str, Json] | Sequence[Json]
    true_description: str | None = None
    false_description: str | None = None

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        if (self.true_description is None) != (self.false_description is None):
            raise ValueError("true_description and false_description must be given together")


Question = Choice | Score | Noul


def _question_to_request(question: Question) -> dict[str, Json]:
    """Renders a question in the `/v1/systemone` format.

    Args:
        question (Question): The question.

    Returns:
        dict[str, Json]: The JSON question with `type`, `instructions` and (except for a plain yes/no) `criteria`.
    """
    if isinstance(question, Choice):
        return {"type": "choice", "instructions": question.instructions, "criteria": _normalize_options(question.options)}
    if isinstance(question, Score):
        return {"type": "score", "instructions": question.instructions, "criteria": _normalize_levels(question.levels)}
    request: dict[str, Json] = {"type": "noul", "instructions": question.instructions}
    if question.true_description is not None and question.false_description is not None:
        request["criteria"] = {"true": question.true_description, "false": question.false_description}
    return request


@dataclass(frozen=True)
class ChoiceAnswer:
    """Data class holding the answer to a `Choice` question.

    Attributes:
        choice (str): The option with the highest probability.
        probabilities (dict[str, float]): The probability of every option, summing to 1.
        confidence (float): How much the model prefers the chosen option over a uniform guess, between 0 and 1.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    """Data class holding the answer to a `Score` question.

    Attributes:
        score (float): The expected level index, between 0 and `len(legend) - 1` (e.g. 1.8 on a 3-level scale).
        legend (list[str]): The level descriptions, as given in the question.
        probabilities (list[float]): The probability of every level, in the order of `legend`.
        confidence (float): How peaked the distribution over the levels is, between 0 and 1.
    """

    score: float
    legend: list[str]
    probabilities: list[float]
    confidence: float

    @property
    def level(self) -> str:
        """The description of the most probable level.

        Returns:
            str: The entry of `legend` with the highest probability, or the one closest to `score` when the probabilities are unavailable.

        Raises:
            ValueError: If the legend is empty.
        """
        if not self.legend:
            raise ValueError("the legend of a ScoreAnswer must not be empty")
        if self.probabilities and len(self.probabilities) == len(self.legend):
            index = max(range(len(self.probabilities)), key=self.probabilities.__getitem__)
        else:
            index = min(max(round(self.score), 0), len(self.legend) - 1)
        return self.legend[index]


@dataclass(frozen=True)
class NoulAnswer:
    """Data class holding the answer to a `Noul` (yes/no) question.

    Attributes:
        probability (float): The probability that the answer is yes, between 0 and 1.
    """

    probability: float

    @property
    def is_true(self) -> bool:
        """Whether the model leans towards yes.

        Returns:
            bool: True if `probability` is at least 0.5.
        """
        return self.probability >= 0.5


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class Answers(dict[str, Answer]):
    """The answers of a `DecisionModel.decide()` call, keyed by question id.

    A plain dict of question id to `ChoiceAnswer`, `ScoreAnswer` or `NoulAnswer`, plus three accessors
    that return the answer with its exact type, so that `answers.choice("route").choice` type-checks
    without an `isinstance` test: the question type is known to the caller, who wrote the question.
    """

    def choice(self, question_id: str) -> ChoiceAnswer:
        """Returns the answer to a `Choice` question.

        Args:
            question_id (str): The id the question was given in `decide()`.

        Returns:
            ChoiceAnswer: The chosen option with the probability of every option.

        Raises:
            KeyError: If no question has this id.
            TypeError: If the question with this id is not a `Choice`.
        """
        return self._typed(question_id, ChoiceAnswer)

    def score(self, question_id: str) -> ScoreAnswer:
        """Returns the answer to a `Score` question.

        Args:
            question_id (str): The id the question was given in `decide()`.

        Returns:
            ScoreAnswer: The expected level and the probability of every level.

        Raises:
            KeyError: If no question has this id.
            TypeError: If the question with this id is not a `Score`.
        """
        return self._typed(question_id, ScoreAnswer)

    def noul(self, question_id: str) -> NoulAnswer:
        """Returns the answer to a `Noul` (yes/no) question.

        Args:
            question_id (str): The id the question was given in `decide()`.

        Returns:
            NoulAnswer: The probability that the answer is yes.

        Raises:
            KeyError: If no question has this id.
            TypeError: If the question with this id is not a `Noul`.
        """
        return self._typed(question_id, NoulAnswer)

    def _typed[T: Answer](self, question_id: str, answer_type: type[T]) -> T:
        if question_id not in self:
            raise KeyError(f"no question with id '{question_id}', the answered ids are {sorted(self)}")
        answer = self[question_id]
        if not isinstance(answer, answer_type):
            raise TypeError(f"question '{question_id}' was answered with a {type(answer).__name__}, not a {answer_type.__name__}")
        return answer


def _validate_questions(questions: object) -> dict[str, Question]:
    """Checks the questions passed to `DecisionModel.decide` and returns them as a dict.

    Args:
        questions (object): The questions keyed by id.

    Returns:
        dict[str, Question]: The validated questions.

    Raises:
        TypeError: If `questions` is not a mapping of string ids to `Choice`, `Score` or `Noul` instances.
        ValueError: If there are no questions.
    """
    mapping = _as_mapping(questions)
    if mapping is None:
        raise TypeError(f"questions must be a dict of question id to Choice, Score or Noul, not {type(questions).__name__}")
    validated: dict[str, Question] = {}
    for question_id, question in mapping.items():
        if not isinstance(question_id, str) or not question_id:
            raise TypeError(f"question ids must be non-empty strings, not {question_id!r}")
        if not isinstance(question, (Choice, Score, Noul)):
            raise TypeError(f"question '{question_id}' must be a Choice, Score or Noul, not {type(question).__name__}")
        validated[question_id] = question
    if not validated:
        raise ValueError("at least one question is required")
    return validated


def _as_float(value: object) -> float:
    """Converts a JSON number to float, rejecting anything else.

    Args:
        value (object): The JSON value.

    Returns:
        float: The number.

    Raises:
        TypeError: If the value is not a number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def _parse_answer(question: Question, raw: object) -> Answer:
    """Parses the server answer to a question into the matching answer data class.

    Args:
        question (Question): The question that was sent, which decides the expected answer shape.
        raw (object): The JSON answer returned by the server.

    Returns:
        Answer: The typed answer.

    Raises:
        TypeError: If the answer does not have the shape expected for the question type.
    """
    answer = _as_mapping(raw)
    if answer is None:
        raise TypeError(f"expected an object, got {type(raw).__name__}")
    if isinstance(question, Choice):
        choice = answer.get("choice")
        probabilities = _as_mapping(answer.get("probabilities"))
        if not isinstance(choice, str) or probabilities is None:
            raise TypeError("a choice answer needs a 'choice' string and a 'probabilities' object")
        parsed_probabilities: dict[str, float] = {}
        for option, probability in probabilities.items():
            if not isinstance(option, str):
                raise TypeError("the keys of 'probabilities' must be strings")
            parsed_probabilities[option] = _as_float(probability)
        return ChoiceAnswer(choice=choice, probabilities=parsed_probabilities, confidence=_as_float(answer.get("confidence", 0.0)))
    if isinstance(question, Score):
        legend = _as_sequence(answer.get("legend"))
        probabilities = _as_sequence(answer.get("probabilities"))
        if legend is None:
            raise TypeError("a score answer needs a 'legend' array")
        if probabilities is None:
            raise TypeError("a score answer needs a 'probabilities' array")
        parsed_legend: list[str] = []
        for level in legend:
            if not isinstance(level, str):
                raise TypeError("the entries of 'legend' must be strings")
            parsed_legend.append(level)
        if not parsed_legend:
            raise TypeError("the 'legend' of a score answer must not be empty")
        return ScoreAnswer(
            score=_as_float(answer.get("score")),
            legend=parsed_legend,
            probabilities=[_as_float(probability) for probability in probabilities],
            confidence=_as_float(answer.get("confidence", 0.0)),
        )
    return NoulAnswer(probability=_as_float(answer.get("noul")))


def _model_ids(payload: object) -> list[str]:
    """Extracts the model ids from a `/v1/models` listing.

    Args:
        payload (object): The JSON body of the listing.

    Returns:
        list[str]: The served model ids.

    Raises:
        TypeError: If the payload is not an OpenAI-style model listing.
    """
    listing = _as_mapping(payload)
    if listing is None:
        raise TypeError(f"expected an object, got {type(payload).__name__}")
    entries = _as_sequence(listing.get("data"))
    if entries is None:
        raise TypeError("expected a 'data' array")
    ids: list[str] = []
    for item in entries:
        entry = _as_mapping(item)
        if entry is None:
            raise TypeError("every model entry must be an object")
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            raise TypeError("every model entry needs an 'id' string")
        ids.append(model_id)
    return ids


@brick
class DecisionModel:
    """A Brick that asks a local decision model typed questions about a text or JSON state.

    A decision model (TypeSafe "System One" API, served by llama.cpp) does not generate text: in a single
    forward pass it answers multiple-choice (`Choice`), graded (`Score`) and yes/no (`Noul`) questions about a
    state with calibrated probabilities. It is faster and more predictable than a chat LLM for routing,
    classification, grading and checks.
    """

    LLAMACPP_MODEL = "llamacpp"

    def __init__(self, model: str | None = None, timeout: float | None = DEFAULT_TIMEOUT_S) -> None:
        """Initializes the DecisionModel brick.

        Args:
            model (str | None): The model identifier with the `llamacpp:` prefix (e.g. "llamacpp:Laya-Q8_0").
                If not provided, the model is read from the app configuration (app.yaml), falling back
                to the brick default for the board.
            timeout (float | None): The maximum duration in seconds to wait for an answer. None waits forever.
                Defaults to 60 seconds.

        Raises:
            DecisionModelError: If the runner address cannot be resolved, no model is configured, or the configured
                model is not a llama.cpp model.
        """
        host = resolve_address(RUNNER_HOST)
        if not host:
            raise DecisionModelError(
                "Host address resolution failed for the llama.cpp models runner.",
                hint="Check the LOCAL_DEV/REMOTE_DEV environment variables and the app configuration.",
            )

        if model is None:
            logger.info("No model specified in constructor. Attempting to retrieve from app configuration or default brick configuration...")
            brick_config = get_brick_config(self.__class__)
            # app.yaml override first, then the brick default (model_by_boards or model)
            model = get_brick_configured_model(brick_config.get("id") if brick_config else None, brick_config=brick_config)
            logger.info(f"Using model: '{model}'.")
        else:
            logger.debug(f"Forcing use of model: '{model}'.")

        model = (model or "").strip()
        if not model:
            raise DecisionModelError("No model configured for the DecisionModel brick.", hint=_CONFIGURE_MODEL_HINT)

        prefix = f"{self.LLAMACPP_MODEL}:"
        if not model.startswith(prefix) or len(model) == len(prefix):
            raise DecisionModelError(f"Unsupported model '{model}' for the DecisionModel brick.", hint=_CONFIGURE_MODEL_HINT)

        self._model_id = model
        self._model_name = model[len(prefix) :]
        self._timeout = timeout
        self._base_url = f"http://{host}:{RUNNER_PORT}"

        logger.info(f"Initializing brick with model '{self._model_name}' at {self._base_url}")

        available_models = self.list_models()
        if self._model_name not in available_models:
            logger.error(
                f"Model '{self._model_name}' not found among locally available models: {available_models}."
                + " Please download the model or configure it correctly."
            )

    def start(self) -> None:
        """Loads the model by asking a trivial question, so that the first real decision is fast.

        Raises:
            DecisionModelError: If the runner cannot be reached or fails to load the model.
        """
        body: dict[str, Json] = {
            "model": self._model_name,
            "state": "warm-up",
            "questions": {"ready": {"type": "noul", "instructions": "Is the decision model ready?"}},
        }
        started_at = time.perf_counter()
        self._post(body, timeout=WARMUP_TIMEOUT_S)
        elapsed_s = time.perf_counter() - started_at
        logger.info(f"Decision model '{self._model_id}' ready in {elapsed_s:.2f} s")

    def decide(self, state: str | Mapping[str, Json] | Sequence[Json], questions: Mapping[str, Question]) -> Answers:
        """Answers one or more questions about a state in a single request.

        Args:
            state (str | Mapping[str, Json] | Sequence[Json]): What the questions are about: free text, or a
                JSON-serializable dict or list (e.g. sensor readings), sent to the model untouched.
            questions (Mapping[str, Question]): The questions keyed by an id of your choice; each value is a
                `Choice`, `Score` or `Noul`.

        Returns:
            Answers: One answer per question id: a `ChoiceAnswer` for a `Choice`, a `ScoreAnswer` for a `Score`
                and a `NoulAnswer` for a `Noul`. A dict, with the typed accessors `choice(id)`, `score(id)` and `noul(id)`.

        Raises:
            TypeError: If `state` or `questions` have the wrong type.
            ValueError: If `state` or `questions` are empty.
            DecisionModelError: If the runner cannot be reached, rejects the request or returns an unexpected answer.
        """
        _validate_state(state)
        validated = _validate_questions(questions)
        body: dict[str, Json] = {
            "model": self._model_name,
            "state": state,
            "questions": {question_id: _question_to_request(question) for question_id, question in validated.items()},
        }
        data = self._post(body, timeout=self._timeout)
        raw_answers = _as_mapping(data.get("answers"))
        if raw_answers is None:
            raise DecisionModelError("The decision model response has no 'answers' object.", hint=_RUNNER_RESPONSE_HINT)

        answers = Answers()
        for question_id, question in validated.items():
            if question_id not in raw_answers:
                raise DecisionModelError(f"The decision model did not answer question '{question_id}'.", hint=_RUNNER_RESPONSE_HINT)
            try:
                answers[question_id] = _parse_answer(question, raw_answers[question_id])
            except TypeError as e:
                raise DecisionModelError(f"Malformed answer to question '{question_id}': {e}.", hint=_RUNNER_RESPONSE_HINT) from e
        return answers

    def choose(
        self,
        state: str | Mapping[str, Json] | Sequence[Json],
        instructions: str | Mapping[str, Json] | Sequence[Json],
        options: Mapping[str, str | None] | Sequence[str] | type[Enum],
    ) -> ChoiceAnswer:
        """Asks a single multiple-choice question. Shortcut for `decide()` with one `Choice`.

        Args:
            state (str | Mapping[str, Json] | Sequence[Json]): What the question is about (text, dict or list).
            instructions (str | Mapping[str, Json] | Sequence[Json]): What to decide.
            options (Mapping[str, str | None] | Sequence[str] | type[Enum]): The options: a dict of option to description
                (or None), a list of option names, or an Enum class whose member values are the option names.

        Returns:
            ChoiceAnswer: The chosen option with the probability of every option.

        Raises:
            TypeError: If the arguments have the wrong type.
            ValueError: If the arguments are empty.
            DecisionModelError: If the runner cannot be reached, rejects the request or returns an unexpected answer.
        """
        answer = self._decide_one(state, Choice(instructions=instructions, options=options))
        if not isinstance(answer, ChoiceAnswer):
            raise DecisionModelError("The decision model returned an answer of the wrong type.", hint=_RUNNER_RESPONSE_HINT)
        return answer

    def score(
        self,
        state: str | Mapping[str, Json] | Sequence[Json],
        instructions: str | Mapping[str, Json] | Sequence[Json],
        levels: Sequence[str],
    ) -> ScoreAnswer:
        """Asks a single graded question. Shortcut for `decide()` with one `Score`.

        Args:
            state (str | Mapping[str, Json] | Sequence[Json]): What the question is about (text, dict or list).
            instructions (str | Mapping[str, Json] | Sequence[Json]): What to grade.
            levels (Sequence[str]): The descriptions of the levels, lowest first (between 2 and 10).

        Returns:
            ScoreAnswer: The expected level and the probability of every level.

        Raises:
            TypeError: If the arguments have the wrong type.
            ValueError: If the arguments are empty or the number of levels is not between 2 and 10.
            DecisionModelError: If the runner cannot be reached, rejects the request or returns an unexpected answer.
        """
        answer = self._decide_one(state, Score(instructions=instructions, levels=levels))
        if not isinstance(answer, ScoreAnswer):
            raise DecisionModelError("The decision model returned an answer of the wrong type.", hint=_RUNNER_RESPONSE_HINT)
        return answer

    def check(
        self,
        state: str | Mapping[str, Json] | Sequence[Json],
        instructions: str | Mapping[str, Json] | Sequence[Json],
        true_description: str | None = None,
        false_description: str | None = None,
    ) -> NoulAnswer:
        """Asks a single yes/no question. Shortcut for `decide()` with one `Noul`.

        Args:
            state (str | Mapping[str, Json] | Sequence[Json]): What the question is about (text, dict or list).
            instructions (str | Mapping[str, Json] | Sequence[Json]): The question.
            true_description (str | None): Optional description of when the answer is yes; requires `false_description`.
            false_description (str | None): Optional description of when the answer is no; requires `true_description`.

        Returns:
            NoulAnswer: The probability that the answer is yes.

        Raises:
            TypeError: If the arguments have the wrong type.
            ValueError: If the arguments are empty or only one of the two descriptions is given.
            DecisionModelError: If the runner cannot be reached, rejects the request or returns an unexpected answer.
        """
        answer = self._decide_one(state, Noul(instructions=instructions, true_description=true_description, false_description=false_description))
        if not isinstance(answer, NoulAnswer):
            raise DecisionModelError("The decision model returned an answer of the wrong type.", hint=_RUNNER_RESPONSE_HINT)
        return answer

    def list_models(self) -> list[str]:
        """Returns the identifiers of the models served by the local llama.cpp runner.

        Connection errors are retried for a few seconds, since the runner may still be starting up.

        Returns:
            list[str]: The served model names without the `llamacpp:` prefix (e.g. ["Laya-Q8_0"]), or an
                empty list if the runner cannot be reached.
        """
        url = f"{self._base_url}/v1/models"
        for attempt in range(1, LIST_MODELS_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(url, timeout=LIST_MODELS_TIMEOUT_S)
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}: {self._server_message(response)}")
                return _model_ids(response.json())
            except requests.ConnectionError as e:
                if attempt >= LIST_MODELS_MAX_ATTEMPTS:
                    logger.warning(f"Failed to list models after {attempt} attempts: {e}")
                    return []
                logger.debug(f"Models runner not reachable yet (attempt {attempt}/{LIST_MODELS_MAX_ATTEMPTS}): {e}. Retrying...")
                time.sleep(LIST_MODELS_RETRY_DELAY_S)
            except Exception as e:
                logger.warning(f"Failed to list models: {e}")
                return []
        return []

    def _decide_one(self, state: str | Mapping[str, Json] | Sequence[Json], question: Question) -> Answer:
        """Sends a single question and returns its answer.

        Args:
            state (str | Mapping[str, Json] | Sequence[Json]): What the question is about.
            question (Question): The question.

        Returns:
            Answer: The answer to the question.
        """
        return self.decide(state, {"answer": question})["answer"]

    def _post(self, body: dict[str, Json], timeout: float | None) -> Mapping[object, object]:
        """Posts a request to `/v1/systemone` and returns the decoded JSON body.

        Args:
            body (dict[str, Json]): The request body.
            timeout (float | None): The request timeout in seconds, None to wait forever.

        Returns:
            Mapping[object, object]: The decoded response.

        Raises:
            DecisionModelError: If the runner cannot be reached, times out, answers with an error or returns invalid JSON.
        """
        url = f"{self._base_url}/v1/systemone"
        try:
            response = requests.post(url, json=body, timeout=timeout)
        except requests.ConnectionError as e:
            raise DecisionModelError(
                f"Cannot reach the llama.cpp models runner at {url}",
                hint="Make sure the arduino:llamacpp service is running (it starts with the app when the brick is declared in app.yaml).",
            ) from e
        except requests.Timeout as e:
            raise DecisionModelError(
                f"The decision model did not answer within {timeout} s.",
                hint="Raise the `timeout` of the DecisionModel brick or shorten the state; the first request also loads the model.",
            ) from e
        except requests.RequestException as e:
            raise DecisionModelError(f"Request to the llama.cpp models runner failed: {e}", hint="Check the logs of the models runner.") from e

        if not 200 <= response.status_code < 300:
            message = self._server_message(response)
            raise DecisionModelError(
                f"The llama.cpp models runner rejected the request (HTTP {response.status_code}): {message}",
                hint=self._hint_for(response.status_code, message),
            )

        try:
            payload: object = response.json()
        except ValueError as e:
            raise DecisionModelError("The decision model response is not valid JSON.", hint=_RUNNER_RESPONSE_HINT) from e
        data = _as_mapping(payload)
        if data is None:
            raise DecisionModelError("The decision model response is not a JSON object.", hint=_RUNNER_RESPONSE_HINT)
        return data

    @staticmethod
    def _server_message(response: requests.Response) -> str:
        """Extracts the error message from a runner response.

        Args:
            response (requests.Response): The error response.

        Returns:
            str: The `error.message` of a JSON error body, else the response text, else the HTTP status.
        """
        try:
            payload: object = response.json()
        except ValueError:
            payload = None
        body = _as_mapping(payload)
        if body is not None:
            error = body.get("error")
            details = _as_mapping(error)
            if details is not None:
                message = details.get("message")
                if isinstance(message, str) and message:
                    return message
            elif isinstance(error, str) and error:
                return error
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    def _hint_for(self, status_code: int, message: str) -> str:
        """Picks the user-facing hint for a runner error.

        Args:
            status_code (int): The HTTP status of the error response.
            message (str): The error message returned by the runner.

        Returns:
            str: A suggestion on how to fix the problem.
        """
        if status_code == 404:
            # An older runner image without the /v1/systemone endpoint answers "File Not Found".
            return _RUNNER_RESPONSE_HINT
        lowered = message.lower()
        if "model is not loaded" in lowered or "failed to load" in lowered or "not found" in lowered:
            return f"Download the model '{self._model_id}' from Arduino App Lab and check the app.yaml configuration."
        if "too many options" in lowered:
            return "Reduce the number of options of the choice question: every decision model has a limit (255 for Laya and Kev, 20 for Julia-1)."
        if "decision model" in lowered:
            return f"'{self._model_id}' is not a decision model: pick one of the decision models of the catalog (e.g. llamacpp:Laya-Q8_0)."
        return "Check the request and the logs of the llama.cpp models runner."
