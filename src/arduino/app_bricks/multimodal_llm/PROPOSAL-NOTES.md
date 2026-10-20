# Multimodal Language Model Brick: technical notes

Draft for the App Bricks team, 2026-10-20. Branch `llamacpp-moltimodal-support` (rjtokenring fork),
HEAD `a7b34507`. The user-facing contract of the Brick is [README.md](README.md); the decision brief for management is [PROPOSAL.md](PROPOSAL.md); this file is the
material for deciding whether, and in what shape, the Brick ships.

## 1. What is on the table

A new Brick, `arduino:multimodal_llm` (`MultimodalLanguageModel`), that chats with a local
llama.cpp model using text, images and audio in the same message:

```python
model = MultimodalLanguageModel(system_prompt="Answer briefly.")
model.chat("Answer the spoken question about this picture.", images=["desk.jpg"], audio=recording)
```

- Same API as `arduino:llm` (`chat`, `chat_stream`, `stop_stream`, `with_memory`, `clear_memory`,
  `get_client`, `tools`, `reasoning_effort`), plus an `audio` argument next to `images`.
- `audio` accepts `.wav`/`.mp3` paths, WAV/MP3 bytes, and the numpy arrays the `Microphone`
  peripheral returns (`record_wav()` whole file, `record_pcm()` raw int16/float samples, read as
  16 kHz mono). One clip or a list.
- Default model `llamacpp:gemma-4-E2B-it-Q4_0`: Unsloth GGUF of Gemma 4 E2B (Q4_0, 2.8 GB) plus
  `mmproj-F16.gguf` (0.9 GB, vision + audio encoders), served by the llama.cpp NPU runner.
  `supported_boards: ["ventunoq"]`, `LLAMA_ARG_CTX_SIZE` 16384.
- Temperature defaults to 0.0 (deterministic: counting, reading, transcription).

### What the branch contains (diff against `main`, 22 files, +1579/-93)

| Area | Change |
| :-- | :-- |
| Brick | `multimodal_llm/` (`__init__.py`, `local_multimodal_llm.py`, `brick_config.yaml`, `README.md`). Subclass of `LargeLanguageModel`. |
| Shared code | `cloud_llm/audio.py`: builds the OpenAI `input_audio` content part from path/bytes/array, wraps raw PCM in WAV, sniffs WAV vs MP3. `CloudLLM._get_message_with_history` takes `audio`; parts are ordered images, audio, text. |
| `llm` brick | Maps the two llama.cpp multimodal errors (`input is not supported`, `failed to process mtmd chunk`) to actionable messages; `<think>` stripping factored into `_strip_think` / `_filter_think`. |
| llama.cpp service | `service_compose.yaml`: `GGML_HEXAGON_MBUF` 256 to 1024 (the encoders need it, 512 crashes the runner), `LLAMA_ARG_IMAGE_MAX_TOKENS=280`, `LLAMA_ARG_FIT=off`. |
| NPU runner | llama.cpp package 20260918 to 20261019 (Hexagon multimodal support), `configure-llamacpp.py`: session budget 0.85, warning for a tensor larger than one MBUF chunk. `SESSION_ALLOCATION.md` updated. |
| Catalog | New entry `llamacpp:gemma-4-E2B-it-Q4_0` (brick `arduino:multimodal_llm`, 1 session). The Google QAT entries (`gemma-4-E2B_q4_0-it`, `gemma-4-E4B_q4_0-it`) gain `model_mmproj_url` and `arduino:vlm` compatibility; their `model_size_mb` now includes the mmproj. |
| Tests | `tests/arduino/app_bricks/multimodal_llm/` (API signatures, audio parts, message building) and `tests/scripts/test_configure_llamacpp.py`. 147 passed locally on 2026-10-20. |

Not in the branch yet: an example in `app-bricks-examples`, the generated `API.md`. On a board the
new model shows up in `arduino-app-cli model list` only when the CLI uses the branch's
models-downloader image (verified with the dev-image environment override; the release image does
not have it).

## 2. Verified on the VENTUNO Q (21q) on 2026-10-20

How: `arduino-app-cli app start ~/ArduinoApps/multimodal-llm` with the fork's dev images
`dev-llamacpp-moltimodal-support` (built by CI on 2026-10-19 from `985e6f38`; they differ from HEAD
only by the upstream shared-camera merge, the multimodal files are byte-identical). The model was
downloaded by hand into the served directory because the board's catalog does not list it yet. The
runner auto-detected 2 Hexagon sessions because Gemma 4 E4B is also installed on that board; a
board with only E2B gets 1. All times are wall clock of one call with the model already loaded.
The "App" and "Server" rows ran on an otherwise idle board; the "Brick" rows ran while a CPU-only
llama-server was transcribing on the same cores, so their times are upper bounds.

| Call | Result | Time |
| :-- | :-- | --: |
| App: describe `pianta.jpg` (first call, ~190 tokens out) | Correct (pothos in a black pot) | 12.1 s |
| App: image + `question.wav` ("Describe what you see in this image", 4 s) | Correct description | 3.1 s |
| App: `question.wav` alone, "Write what you have listen" | Model obeyed the spoken instruction instead ("Please provide the image or audio") | 2.4 s |
| Brick: transcribe `en_1522.wav` (13 s) from a file path | Correct transcription | 2.0 s |
| Brick: transcribe `it_1585.wav` (11 s) as an int16 PCM array | Correct transcription | 2.0 s |
| Brick: language of the same clip as a float32 array | "Italian" | 1.5 s |
| Brick: `chat_stream` transcribing `en_1583.wav` (16 s) | Correct transcription | 2.3 s |
| Brick: text only | "8" | 0.3 s |
| Server, direct: 13 s clip, NPU vs CPU-only llama-server | Same transcript, same token count | 3.5 s vs 16.0 s |

So the whole path works on the NPU: brick, OpenAI `input_audio` part, llama.cpp mtmd, Hexagon
encoders. Audio costs 25 prompt tokens per second of speech (4 s = 100 tokens), as Google documents.
The CPU-only comparison shows the Hexagon path is numerically faithful.

### Instruction order matters for audio

Current rule (from #430, measured on images): media first, text last. With audio, when the clip
*is itself an instruction* ("Describe what you see in this image") and comes before the written
prompt, Gemma 4 E2B follows the spoken one and ignores the written one. Putting the text first
fixes it (4 s clip: "Describe what you see in this image." transcribed in 1.3 s). On narrative
speech both orders transcribe correctly. The same happens with llama.cpp on the CPU, so it is the
model, not the Hexagon backend.

Aggregate benchmark over the 66 clips of the ASR test set on the board, full clip and first 3 s,
instruction before vs after the audio, temperature 0, no prompt cache, NPU runner:

| Variant | Instruction | n | Median WER | Clips with WER > 0.5 | Refusals ("no audio provided") | p50 / p95 time |
| :-- | :-- | --: | --: | --: | --: | --: |
| Full clip | after the audio (today's order) | 66 | 0.054 | 0 | 0 | 3.84 / 9.12 s |
| Full clip | before the audio | 66 | 0.060 | 0 | 0 | 3.76 / 9.23 s |
| First 3 s | after the audio (today's order) | 66 | 0.310 | 17 | 5 | 1.15 / 2.45 s |
| First 3 s | before the audio | 66 | 0.286 | 9 | 0 | 1.10 / 1.36 s |

Clips: 30 English (4-29 s), 30 Italian (6-31 s), 6 long Italian (33-39 s). WER against the
normalized reference; for the 3 s cuts against the reference prefix, so the absolute values there
are inflated by the cut word. The first 13 requests overlapped a CPU-only server run on the same
cores. Scripts and raw output in section 6.

Reading: on whole sentences the order does not matter. On short clips, the regime of a spoken
command to a voice assistant, the current order answers "no audio provided" in 5 of 66 cases and
the text-first order never does; outside those 5, the two orders fail on mostly the same hard cuts
(8 clips in common). The tail latency of the 3 s cuts is also lower text-first (p95 1.4 s vs
2.5 s): the refusals are long answers.

### Reasoning leaks into audio requests

With `LLAMA_ARG_REASONING=off` and `LLAMA_ARG_THINK_BUDGET=128`, some audio-first requests still
produced a `reasoning_content` of ~120 tokens ("Here's a thinking process...") before the answer:
10 s instead of 1.5 s for the same clip, and an empty answer when `max_tokens` is small. The Brick's
default `max_tokens=512` hides it as latency, not as a failure.

### Performance: NPU vs CPU, same binary, same model

Interleaved A/B on 21q, 2026-10-20: the app's NPU runner (llama.cpp 20261019, `--device
HTP0,HTP1`, mmproj on the NPU, ctx 16384) against a second instance of the same `llama-server`
started inside the same container with `--device none` (4 threads on cores 0-3, ctx 4096).
3 rounds, NPU/CPU order alternated per case, temperature 0, no prompt cache, no other load on
the board. Medians of 3; `prompt` is the server's own `prompt_ms` (media encoding + prefill),
`generation` its `predicted_ms`. Transcripts and descriptions were identical on NPU and CPU in
every round.

| Input | Prompt tokens | Prompt NPU | Prompt CPU | Generation NPU | Generation CPU | Wall NPU | Wall CPU |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Audio 9.5 s (en_1633) | 258 | 1.29 s | 10.45 s | 22 tok, 17.4 tok/s | 15.7 tok/s | 2.6 s | 11.9 s |
| Audio 13.4 s (en_1522) | 356 | 1.62 s | 13.25 s | 33 tok, 17.1 tok/s | 15.4 tok/s | 3.6 s | 15.4 s |
| Audio 16.5 s (en_1583) | 433 | 2.07 s | 19.04 s | 34 tok, 17.2 tok/s | 15.3 tok/s | 4.1 s | 21.3 s |
| Audio 19.9 s (it_1564) | 519 | 2.27 s | 24.98 s | 38 tok, 16.9 tok/s | 14.9 tok/s | 4.6 s | 27.6 s |
| Audio 37.5 s (itlong) | 959 | 4.28 s | 39.75 s | 80 tok, 16.3 tok/s | 14.1 tok/s | 9.3 s | 45.5 s |
| Image 256x197 JPEG, one sentence | 98 | 0.83 s | 7.37 s | 16 tok, 17.8 tok/s | 16.5 tok/s | 1.75 s | 8.4 s |
| Text only | 17 | 0.27 s | 0.40 s | 10 tok, 18.7 tok/s | 17.5 tok/s | 0.8 s | 1.0 s |

Reading:

- The NPU wins the prompt phase, which is where multimodal time goes: 200-230 prompt tokens/s
  against 21-28 on the CPU for audio (8-11x), 118 against 13 for the image (8.8x). Audio costs
  about 0.12 s of NPU time per second of speech, the image 0.8 s.
- Generation is memory-bound and nearly the same on both: 16-19 tok/s on the NPU, 14-18 on the
  CPU (1.1x). A long answer costs the same with or without the NPU; the Brick's `max_tokens=512`
  default is 30 s of generation either way.
- End to end a typical voice command (10-15 s of audio, 20-30 tokens of answer) is 2.6-3.6 s on
  the NPU and 12-15 s on the CPU. NPU times were stable (spread under 0.1 s over the 3 rounds);
  CPU times had outliers up to 140 s on the long clips.
- With only one session (a board without E4B) and 4 CPU threads shared with the app, numbers may
  differ; not measured.

## 3. Constraints to keep in mind

- VENTUNO Q only. On the UNO Q (CPU, no fp16) Gemma 4 E2B with encoders is not usable.
- RAM: a loaded model costs ~1.6x its GGUF size (DSP buffers are system RAM, outside the cgroup).
  E2B multimodal: 3.8 GB on disk, about 6 GB loaded by that rule (not re-measured today). E4B:
  6.3 GB on disk, 2 sessions.
- `GGML_HEXAGON_MBUF=1024` is required by the encoders; 512 crashes the runner at CLIP warm-up.
  This value is now the service default for every llama.cpp app, not only this Brick.
- Audio: 25 tokens per second; Google documents 30 s per clip, llama.cpp splits longer audio
  into 30 s chunks and the 33-39 s clips above were transcribed with WER 0.05. Images: capped at 280 tokens
  (`LLAMA_ARG_IMAGE_MAX_TOKENS`). Context 16384 shared by history, media and answer.
- Memory (`with_memory`) stores media as base64 in the history: a few images or clips fill the
  window and the RAM of the app.
- Encoder on the CPU (`--no-mmproj-offload`) works but images are 4x slower.

## 4. How to expose it to users

### Where the split comes from

`arduino:llm` and `arduino:vlm` were split for two reasons: two classes with different inputs,
and the Genie container, where a VLM only takes images placed in the prompt. With llama.cpp that
reasoning no longer holds. A multimodal model is two files, the text model and the encoder
(`mmproj`), and what gets loaded in RAM decides which modalities exist. Not every model has an
audio encoder, but Gemma 4 ships vision and audio in the same mmproj (verified on both the Google
and the Unsloth files), and the text/vision/omni classification is dissolving. Upstream llama.cpp has merged a good part of
the Hexagon multimodal work; the runner still ships a fork package (`llamacpp-hexagon-20261019`)
with patches that are submitted but not yet merged. With BF16 encoders Marco measured image and
audio processing at about 400-500 ms each, after which the usual text generation starts; the
table in section 2 is consistent (image 0.83 s for 98 prompt tokens, 4 s of audio about 0.5 s).

Two positions were put forward:

- **Marco's note.** Keep `llm` as pure text-to-text, so an app that does not need the encoder
  does not download or load its extra GB. Add a new Brick that does text, images and audio, and
  keep `vlm` for the pure vision models. This is what the branch implements.
- **Review comment.** A third Brick mirrors the internal architecture, not how users think: they
  do not know what an mmproj is, they want to talk to a model and maybe hand it a photo. One
  Brick, modality as a parameter, text-only by default so current users pay nothing:

  ```python
  model = LanguageModel()                               # text only, no encoder
  model = LanguageModel(modalities=["image"])           # downloads and loads the mmproj
  model = LanguageModel(modalities=["image", "audio"])
  model.chat("...", images=[...], audio=[...])
  ```

  Two fixed points: fail fast when `images=` or `audio=` goes to a model without that encoder
  (today the risk is silence: in the example above the model described the plant, and nothing
  shows it heard the question), and `model.capabilities` to know what the loaded model supports.
  `llm` and `vlm` would survive as deprecated aliases (`vlm` = `LanguageModel(modalities=["image"])`)
  for one release, so existing sketches and examples keep working.

### Facts that constrain the choice

1. **Provisioning happens before the app runs.** App Lab downloads models per Brick id from the
   catalog at prepare time; a Python constructor argument is invisible to it. `modalities=` can
   drive a download only if it is also declared in `app.yaml`: the Brick system already has
   `variables`, and `requires_services` can depend on the model id (`when: model: "llamacpp:*"`,
   as `llm` does today to pick Genie or llama.cpp). So the parameter is implementable, but as a
   Brick variable plus a catalog that marks the mmproj as optional, not as a plain argument.
2. **The runner decides the modality from the disk, not from the Brick.** `configure-llamacpp.py`
   attaches any `*mmproj*.gguf` found next to a model, and llama-server loads the encoder with it.
   In the branch the Google QAT entries gained `model_mmproj_url`, so a text-only `llm` app on
   Gemma 4 E2B now downloads 0.94 GB more and loads the encoder (the runner's sizing notes record
   552 MiB of HTP compute buffers for E2B with the mmproj attached, measured at a different
   micro-batch size than the 265 MiB without it, so the two are indicative, not a difference).
   Whatever the Brick layout, "text-only pays nothing" needs either separate catalog entries or a
   runner that attaches the mmproj on request. Under option C the Google and Unsloth E2B entries
   are the same weights for two purposes, see decision 4.
3. **The server already knows what the model can take.** llama-server reports
   `input_modalities` (text, image, audio, video) for each loaded model (checked in the fork's
   server source, not yet probed through the router on the board), and rejects media the model
   cannot encode ("image input is not supported"), which the branch maps to a clear error.
   So `capabilities` can come from the runner rather than from a hand-maintained table, and the
   "no encoder" case already fails fast. What cannot fail fast is media the model accepts and
   then ignores, which is the ordering finding above: that one is handled by the order of the
   parts and by the docs. A cheap self-check is possible, since audio adds 25 prompt tokens per
   second: a request whose `prompt_tokens` did not grow did not encode the clip.
4. **The Genie VLM.** `vlm` runs on the Genie container with `genie:*` models. A unified Brick
   could reach it through the same `when: model:` rules `llm` uses, so the alias looks feasible
   without migrating the VLM to llama.cpp first. The mechanism exists; it has not been tried with
   two backends and two `chat` signatures behind one Brick id.
5. **Defaults are per Brick id and per board.** `llm` picks a text model per board
   (`model_by_boards`); the multimodal default is a different catalog entry. A single Brick needs
   a rule for which entry is the default when the modality changes.

### Options

| | A: three Bricks (branch) | B: one Brick, `modalities=` parameter | C: one Brick, modality from the chosen model |
| :-- | :-- | :-- | :-- |
| User sees | `llm`, `vlm`, `multimodal_llm` | `LanguageModel`, aliases for one release | `llm` (and `vlm` until Genie migrates); picks "Gemma 4 E2B Multimodal" in App Lab |
| Text-only cost | zero by construction | zero by default | zero if text entries have no mmproj (fix point 2) |
| Needs outside this repo | catalog entry only (done) | CLI: Brick variable drives which files are downloaded; catalog: optional mmproj | catalog: text entries without mmproj (or the runner attaches the mmproj on request); the entry chosen in App Lab already lists its files |
| Fail fast / capabilities | server errors mapped (done) | as designed | same, `capabilities` read from the server at init |
| Risk | three similar Bricks to explain and document | the parameter promises a download it cannot trigger at runtime | `llm` README grows; default text model cannot take media, error only when media is passed |

Recommendation: C, as the smallest change that gives the user model of B. Add `audio=` and
`capabilities` to `LargeLanguageModel`, keep the content order from the benchmark, and let the
modality follow the catalog entry the user picks for the `llm` Brick. Keep `vlm` as it is while it
runs on Genie, and remove the mmproj from the text-only catalog entries. `multimodal_llm` as a
separate id then becomes unnecessary, and the code in the branch moves into `llm` almost
unchanged (the audio helper, the error mapping and the tests are already shared).

## 5. Decisions to take

1. **Brick layout.** A, B or C above. Recommendation: C.
2. **Name.** Only if A is chosen: "Multimodal Language Model" / `MultimodalLanguageModel` /
   `arduino:multimodal_llm`, or "Omni", or "Vision & Audio LM". Decide before it appears in App Lab.
3. **Order of content parts.** Keep media first (today), or put the text before the audio
   (images, text, audio), or text first for everything. The benchmark says the audio should come
   after the text on short clips and that it costs nothing on long ones; #430 chose images first
   on measurements with small VLMs, which this does not contradict. Recommendation: images, text,
   audio, with the message-building tests updated to pin it. Not changed in the branch yet.
4. **Default model and catalog entries.** Unsloth E2B Q4_0 + F16 mmproj (3.8 GB, in the branch)
   vs Google QAT `gemma-4-E2B_q4_0-it` + Google mmproj (4.4 GB, already in the catalog for
   `llm`/`vlm`, both mmproj files carry vision and audio). One model for every use would save a
   3-4 GB download per board; quality and speed of the two have not been compared. Independent
   of the layout: the Google entries should not pull the mmproj for text-only `llm` apps (one
   entry per purpose, or an optional mmproj in the catalog).
5. **E4B multimodal.** Offer it in the catalog for this Brick (2 sessions, 6.3 GB) or not.
6. **Thinking.** The Gemma 4 chat template defaults `enable_thinking` to false, so the thought
   block above was produced by the model on its own. In the one probe with a non-empty system
   prompt the same request answered in 1.6 s without it. Options: a default system prompt for the
   Brick, `LLAMA_ARG_THINK_BUDGET=0` for this model, or accept the occasional slow answer. To
   measure before choosing.
7. **Catalog `model_configuration` for a Brick default model.** `GGML_HEXAGON_DEVICES: 1` from the
   catalog did not reach the compose of the test app, although the CLI run with the dev images
   does list `gemma-4-E2B-it-Q4_0` in `model list`. Two candidates, not yet separated: the app
   leaves the model to `brick_config.yaml` instead of naming it in `app.yaml` (vlm-detector, which
   names it, got the variable), or the model was installed by hand and has no
   `.arduino_metadata.yaml`, so the CLI never "resolved" it. Check on the CLI side: start the same
   app once with the model named in `app.yaml`, and once after a download by the CLI.
8. **Packaging and docs.** `app-bricks-examples` entry, `API.md`, App Lab description, and whether
   the README's Microphone example (`record_pcm(5)`) is the first thing users should see.

## 6. How it was tested (for reproduction)

- Board: 21q, Ubuntu 24.04, CLI `0.0.0-04327324`. Start with the dev images:
  `DOCKER_REGISTRY_BASE=ghcr.io/rjtokenring/ DOCKER_PYTHON_BASE_IMAGE=app-bricks/python-apps-base:dev-llamacpp-moltimodal-support arduino-app-cli app start ~/ArduinoApps/multimodal-llm`.
- Model files in `/var/lib/arduino-app-cli/models/llamacpp/unsloth/gemma-4-E2B-it-GGUF/` (curl from
  the Hugging Face revision pinned in the catalog).
- Direct server probes and the ordering benchmark: `~/mm-app-test/*.py` on the board
  (`audio_ab.py`, `audio_matrix.py`, `order_bench.py`), results in `order_bench.jsonl`.
- NPU vs CPU A/B: `perf_ab.py` run inside the runner container against the router (:9999) and a
  CPU-only `llama-server --device none --port 9998` started in the same container; results in
  `~/mm-app-test/perf_ab.jsonl` (server `timings` per request).
- Brick end-to-end from the app image: `docker compose -f .cache/app-compose.yaml -p multimodal-llm run --rm --no-deps -v ~/asrtest/data:/clips:ro -v ~/mm-app-test:/harness:ro --entrypoint python -T main /harness/e2e_brick.py`.
