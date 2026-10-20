# Multimodal AI in App Bricks: decision brief

Arduino App Bricks team, 2026-10-20. Working prototype on branch `llamacpp-moltimodal-support`,
validated on the VENTUNO Q. Nothing described here is released. Technical details and
measurements: [PROPOSAL-NOTES.md](PROPOSAL-NOTES.md).

## What we have: a working prototype

An Arduino App can talk to an AI model running on the board with text, pictures and voice, fully
offline, accelerated by the NPU. The prototype is built as a new, separate Brick ("Multimodal
Language Model") only because that was the fastest way to try it; whether it ships in that shape
is the decision below.

Verified on the VENTUNO Q with Gemma 4 E2B: describing a photo, answering a spoken question about
a photo, transcribing speech. Same output quality as on the CPU, 4 to 6 times faster: a spoken
command of 10-15 seconds gets its answer in about 3 seconds (12-15 s without the NPU). This builds
on the Hexagon multimodal work merged upstream in llama.cpp plus patches we have submitted and
are waiting to see merged.

## Why a decision is needed now

Today users see two Bricks: **LLM** (text) and **VLM** (text + pictures). That split came from
the models and runtimes available at the time. With the new generation of models the picture
changes: the *same* model does text, pictures and voice depending on which files are installed
on the board. The text model is one download (about 3 GB for Gemma 4 E2B); the "senses" are a
second, optional file of about 1 GB that also takes RAM and a few seconds at load time. An app
that only chats should not pay for it.

Two product decisions follow, independent of the engineering underneath:

1. **A third Brick, or extend the LLM Brick we already have?**
2. **How does the user get the right model files?**

## Decision 1: new Brick or extension of LLM

The same app, written the two ways:

```python
# A. A third Brick next to LLM and VLM
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel
model = MultimodalLanguageModel()
model.chat("Answer the spoken question about this picture.", images=["photo.jpg"], audio=recording)

# B. The existing LLM Brick, which accepts pictures and voice when the chosen model supports them
from arduino.app_bricks.llm import LargeLanguageModel
model = LargeLanguageModel()
model.chat("Answer the spoken question about this picture.", images=["photo.jpg"], audio=recording)
```

| | A. New third Brick | B. Extend the LLM Brick |
| :-- | :-- | :-- |
| Brick list in App Lab | LLM, VLM, Multimodal | LLM, VLM (unchanged) |
| How the user gets pictures and voice | adds the Multimodal Brick, which comes with a multimodal model | picks a multimodal model for the LLM Brick in App Lab; `images=` and `audio=` then work |
| Text-only users | untouched | untouched: the default LLM model stays text-only and costs nothing more |
| Mistakes | picking the wrong Brick for the model: three similar Bricks to tell apart | passing a picture to a text-only model: rejected with a clear message (already in place) |
| Documentation and examples | a third Brick to explain, with mostly the same API as LLM | one Brick, one page, new sections for pictures and voice |
| Future | VLM and Multimodal overlap as soon as VLM moves to the same runtime | VLM can later become "LLM with a vision model"; one Brick to maintain |
| Engineering | done in the prototype | small: the prototype code moves into LLM, the same tests apply |

A variant of B was proposed in review: declare the senses when creating the Brick,
`LargeLanguageModel(modalities=["image", "audio"])`. It reads well, but models are installed by
App Lab before the app runs, so the choice must be made in App Lab anyway; the parameter would
only repeat it. We suggest the model choice alone carries the modality.

**Recommendation: B.** Users think "I want to talk to a model and maybe show it a photo", not in
model architectures. One Brick keeps the list short, and the prototype already contains the code.

## Decision 2: how the user gets the model

| | 1. Two entries in the model list | 2. One entry with an optional "vision & audio" component | 3. One entry, always everything |
| :-- | :-- | :-- | :-- |
| In App Lab | "Gemma 4 E2B" (text, ~3 GB) and "Gemma 4 E2B Multimodal" (text + pictures + voice, ~4 GB) | "Gemma 4 E2B" with a second step for the senses (+1 GB) | "Gemma 4 E2B" always 4 GB |
| Pros | works today, nothing to build in App Lab; the download size is visible before choosing | no duplication, one name per model | simplest to explain |
| Cons | a user who wants both installs the 3 GB base twice unless the two entries share it | needs App Lab and CLI support for optional files | every user pays 1 GB of download and RAM, including text-only apps |

**Recommendation: 1 now, 2 later.** Two entries ship with what exists; sharing the base file
between them is a catalog refinement for when the CLI supports optional components, which is also
what option 2 needs.

One correction to make whatever we decide: in the prototype branch the current text-only Gemma 4
entries were given the extra file, so today a text-only app would download and load the senses it
does not use. The entries need to be split as in option 1.

## What this unlocks

Voice assistants that answer about what the camera sees, hands-free dictation and commands,
"what is this?" with a photo and a spoken question, all offline. VENTUNO Q only: the UNO Q has no
NPU and these models are too large for its CPU.

## Still open on the engineering side

- Robustness of spoken commands: on short clips the model sometimes ignores the audio if it is
  placed before the written prompt; placing the text first removed the failures in our tests.
  Internal detail, decided by measurements, no user impact.
- Which Gemma 4 build to publish as the multimodal default (two candidates, same quality class).
- Rebuild of the images after the merge, examples and documentation.

Effort for B plus catalog option 1: small, the prototype already contains the code and the tests;
to be estimated precisely once the two decisions are taken.

## Key numbers (VENTUNO Q, 2026-10-20)

| | With NPU | CPU only |
| :-- | --: | --: |
| Spoken question, 13 s of audio, short answer | 3.6 s | 15.4 s |
| Photo description, one sentence | 1.8 s | 8.4 s |
| Text-only question, one-line answer | 0.8 s | 1.0 s |
| Model download, text only / with pictures and voice | 2.8 GB / 3.8 GB | |

Outputs on NPU and CPU were identical in every test.
