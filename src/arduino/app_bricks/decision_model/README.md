# Decision Model Brick

The Decision Model Brick asks a locally hosted decision model typed questions about a piece of text or a JSON document (the *state*): pick one option, grade on a scale, answer yes or no. Every answer comes back with calibrated probabilities in a single, fast inference step, with no text generation involved.

## Overview

A decision model (TypeSafe "System One" API, served by llama.cpp) is an encoder-based classifier, not a chat model. You describe the state and the questions; the model answers all of them in one forward pass and returns, for every question, a probability distribution over the options you provided plus a confidence. There is no prompt engineering, no parsing of free text and no risk of a malformed reply: the answer is always one of your options.

Three question types are supported:

- **`Choice`**: pick one option among many (routing, classification, intent detection). Options are a list of names, a dict of name to description, or an `Enum` class.
- **`Score`**: place the state on a scale of 2 to 10 described levels (urgency, severity, quality). The answer is the expected level plus the probability of every level.
- **`Noul`**: a yes/no question (checks, filters, flags). The answer is the probability that the answer is yes.

Available models (download them in Arduino App Lab): `llamacpp:Laya-Q8_0` (default, 0.42B parameters, up to 255 options), `llamacpp:Julia-1-Q8_0` (0.14B, the smallest and fastest, 2 to 20 options) and `llamacpp:Kev-0.8B-Q8_0` (0.8B, up to 255 options, best on long states). The model can be changed in `app.yaml` under `arduino:decision_model`.

## Features

- **Typed answers**: `Choice`, `Score` and `Noul` questions return `ChoiceAnswer`, `ScoreAnswer` and `NoulAnswer` data classes, with the probabilities of every option; `decide()` returns them in an `Answers` dict with typed accessors.
- **One request, many questions**: `decide()` answers any number of questions about the same state in a single call.
- **Text or JSON state**: pass free text, or a dict/list of readings and let the model reason on structured data.
- **Local and private**: the model runs on the board inside the `arduino:llamacpp` service, like the Large Language Model Brick.
- **Shortcuts**: `choose()`, `score()` and `check()` ask a single question and return its typed answer directly.

## Code Example and Usage

### Routing a Customer Message

This example asks three questions about the same message in one request: which department should handle it, how urgent it is and whether the customer is angry. Models must be downloaded and available locally.

```python
from arduino.app_bricks.decision_model import Choice, DecisionModel, Noul, Score
from arduino.app_utils import App

decision_model = DecisionModel()

MESSAGE = "Hi, I was charged twice for my last order and nobody answers the phone. Please fix this today!"


def route_message():
    answers = decision_model.decide(
        state=MESSAGE,
        questions={
            "department": Choice(
                instructions="Which department should handle this customer message?",
                options={
                    "billing": "Payments, invoices, refunds and double charges",
                    "shipping": "Delivery status, lost or damaged parcels",
                    "technical": "Product not working, setup and firmware problems",
                },
            ),
            "urgency": Score(
                instructions="How urgent is this message?",
                levels=["Can wait a week", "Should be handled in a few days", "Needs an answer today"],
            ),
            "angry": Noul(instructions="Is the customer angry?"),
        },
    )

    department = answers.choice("department")
    urgency = answers.score("urgency")
    angry = answers.noul("angry")
    print(f"Department: {department.choice} (confidence {department.confidence:.2f})")
    print(f"Urgency: {urgency.level} (score {urgency.score:.1f} of {len(urgency.legend) - 1})")
    print(f"Angry customer: {angry.is_true} (probability {angry.probability:.2f})")
    raise StopIteration


App.run(route_message)
```

### Classifying a JSON State with an Enum

The state can be a dict (or a list) instead of text, and the options can be an `Enum` class: the member values are the options and the chosen value converts back to the enum member.

```python
from enum import Enum

from arduino.app_bricks.decision_model import DecisionModel
from arduino.app_utils import App


class Team(Enum):
    HARDWARE = "hardware"
    FIRMWARE = "firmware"
    CLOUD = "cloud"


decision_model = DecisionModel()

TICKET = {
    "title": "Board not detected after update",
    "body": "Since the 2.1 update the board shows up for a second and then disappears from the USB devices.",
    "product": "UNO Q",
}


def triage_ticket():
    answer = decision_model.choose(
        state=TICKET,
        instructions="Which team should investigate this ticket?",
        options=Team,
    )
    team = Team(answer.choice)
    print(f"Assigned to {team.name} with probability {answer.probabilities[answer.choice]:.2f}")
    for option, probability in answer.probabilities.items():
        print(f"  {option}: {probability:.2f}")
    raise StopIteration


App.run(triage_ticket)
```

### Checking and Grading Sensor Readings

`check()` answers a yes/no question and `score()` grades the state on a scale you describe. Both accept the optional descriptions that help the model understand your domain.

```python
from arduino.app_bricks.decision_model import DecisionModel
from arduino.app_utils import App

decision_model = DecisionModel()

READINGS = {
    "room": "server closet",
    "temperature_c": 38.5,
    "humidity_pct": 71,
    "fan_rpm": 0,
    "door_open": False,
}


def assess_readings():
    overheating = decision_model.check(
        state=READINGS,
        instructions="Is the equipment in this room at risk of overheating?",
        true_description="Temperature or humidity are high and the cooling is not working",
        false_description="The environment is within normal operating conditions",
    )
    severity = decision_model.score(
        state=READINGS,
        instructions="How severe is the situation?",
        levels=["Normal", "Keep an eye on it", "Intervene soon", "Emergency"],
    )
    if overheating.is_true:
        print(f"Overheating risk (probability {overheating.probability:.2f}): {severity.level}")
    else:
        print(f"All good (probability of a problem {overheating.probability:.2f}): {severity.level}")
    raise StopIteration


App.run(assess_readings)
```

## Methods

- **`decide(state, questions)`**: Answers several questions about `state` in one request. `questions` maps an id of your choice to a `Choice`, `Score` or `Noul`; the result is an `Answers` dict mapping the same ids to a `ChoiceAnswer`, `ScoreAnswer` or `NoulAnswer`, with the typed accessors `answers.choice(id)`, `answers.score(id)` and `answers.noul(id)`.
- **`choose(state, instructions, options)`**: Asks a single multiple-choice question and returns a `ChoiceAnswer` (`choice`, `probabilities`, `confidence`). `options` is a list of names, a dict of name to description (or `None`), or an `Enum` class.
- **`score(state, instructions, levels)`**: Asks a single graded question and returns a `ScoreAnswer` (`score`, `legend`, `probabilities`, `confidence`, and the `level` property with the most probable level description). `levels` lists 2 to 10 level descriptions, lowest first.
- **`check(state, instructions, true_description=None, false_description=None)`**: Asks a single yes/no question and returns a `NoulAnswer` (`probability` of yes, and the `is_true` property). The two descriptions are optional but must be given together.
- **`list_models()`**: Returns the model identifiers available on the local inference service (without the `llamacpp:` prefix).
- **`start()`**: Loads the model with a warm-up request so that the first real decision is fast. Called automatically by `App.run()`.

All methods raise a `DecisionModelError` (with a hint on how to fix the problem) when the runner cannot be reached, the model is not downloaded or the request is rejected, and `TypeError`/`ValueError` for invalid questions (for example a `Score` with a single level or a `Choice` without options).

On the UNO Q the llama.cpp runner serves one model at a time (`LLAMA_ARG_MODELS_MAX=1`): an app that alternates `DecisionModel` and `LargeLanguageModel` requests reloads a model at every switch, so keep the two kinds of requests grouped when latency matters.
