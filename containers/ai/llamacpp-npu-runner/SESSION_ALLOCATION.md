# Hexagon session allocation

How many Hexagon (NPU) sessions a model needs, how that was measured, and how
[`scripts/configure-llamacpp.py`](scripts/configure-llamacpp.py) decides it.

A session is one DSP protection domain — `HTP0`..`HTP3`, four of them per process.
`llama-server` spreads a model over the sessions it is given layer by layer: each session
holds the weights of its layers, the slice of the KV cache belonging to them, and its own
compute buffers. Too few sessions and the model fails to load; too many and each token
costs a little more. The runner has to choose the number before the server starts, from
the files on disk alone.

## Contents

- [In short](#in-short)
- [The matrix](#the-matrix)
- [How a model uses a session](#how-a-model-uses-a-session)
- [Where the limit is](#where-the-limit-is)
- [The sizing rule](#the-sizing-rule)
- [A quantized KV cache](#a-quantized-kv-cache)
- [Method](#method)

## In short

- **File size does not predict sessions.** What reaches the NPU ranges from 31% to 99% of
  the GGUF, and the same 4B model needs one session as Q4_0 and two as Q8_0. The sizing
  now reads the GGUF header instead: which tensors land on the NPU, and how big the KV
  cache is at the context the server runs at. Both reproduce what `llama-server` logs to
  within a MiB.
- **One session reliably holds about 1.8 GiB of weights plus KV cache.** Above that the
  outcome is not a function of size at all: loads succeeded holding 2.9 GiB and failed
  holding 1.8 GiB, and nine configurations loaded on some attempts and not on others.
- **The rule:** the smallest session count whose busiest session stays under 1800 MiB, then
  halve the context until no model needs all four sessions. Against 175 trials it never
  asks for fewer sessions than a model was measured to need, where the size tables it
  replaces did so twice.
- **A quantized KV cache would halve the context axis and make 16k reachable for 8B models,
  but it is not enabled:** on a model whose layers share a cache, splitting it across
  sessions corrupts the output silently.

## The matrix

Measured with the environment the service runs in ([Method](#method)) on a 16 GB board,
against the llama.cpp build shipped in this image (`0.3.0-dev`, build 10779) and DSP
firmware `DSP.AT.1.0.1-00170-LEMANS-1`: 175 trials over three runs, covering 87 distinct
(model, context, sessions) configurations.

| Model | Quant | GGUF | On NPU | KV 4k / 8k / 16k | Sessions 4k / 8k / 16k | Sized 4k / 8k / 16k |
|---|:---:|---:|---:|---:|:---:|:---:|
| SmolVLM2-500M-Video-Instruct | Q8_0 | 436 MB | 366 MiB | 160 / 320 / 640 MiB | **1 / 1 / 1** | 1 / 1 / 1 |
| Qwen3.5-0.8B | Q4_0 | 507 MB | 250 MiB | 48 / 96 / 192 MiB | **1 / 1 / 1** | 1 / 1 / 1 |
| gemma-3-1b-it | Q4_0 | 721 MB | 682 MiB | 66 / 82 / 114 MiB | **1 / 1 / 1** | 1 / 1 / 1 |
| granite-4.2-3b | Q4_0 | 2129 MB | 1687 MiB | 320 / 640 / 1280 MiB | **1 / 2 / 2** | 2 / 2 / 2 |
| Phi-4-mini-instruct | Q4_0 | 2331 MB | 1734 MiB | 512 / 1024 / 2048 MiB | **2 / 2 / 3** | 2 / 2 / 3 |
| Qwen3-4B-Instruct-2507 | Q4_0 | 2375 MB | 1954 MiB | 576 / 1152 / 2304 MiB | **2 / 2 / 3** | 2 / 2 / 3 |
| Qwen3.5-4B | Q4_0 pure | 2380 MB | 2258 MiB | 128 / 256 / 512 MiB | **1 / 1 / 2** | 2 / 2 / 2 |
| Nemotron-Mini-4B-Instruct | Q4_0 | 2574 MB | 1411 MiB | 512 / 1024 / 2048 MiB | **1 / 2 / 3** | 2 / 2 / 3 |
| Qwen3.5-4B | Q4_0 | 2583 MB | 1790 MiB | 128 / 256 / 512 MiB | **1 / 1 / 2** | 2 / 2 / 2 |
| gemma-4-E2B-it | Q4_0 | 3349 MB | 1026 MiB | 51 / 75 / 123 MiB | **1 / 1 / 1** | 1 / 1 / 1 |
| Qwen3.5-4B | Q8_0 | 4482 MB | 4263 MiB | 128 / 256 / 512 MiB | **2 / 2 / >2** | 3 / 3 / 4 |
| DeepSeek-R1-Distill-Llama-8B | Q4_0 | 4675 MB | 3758 MiB | 512 / 1024 / 2048 MiB | **2 / 3 / >4** | 3 / 3 / 4 |
| Qwen3-8B | Q4_0 | 4787 MB | 3737 MiB | 576 / 1152 / 2304 MiB | **2 / 3 / >4** | 3 / 3 / 4 |
| DeepSeek-R1-0528-Qwen3-8B | Q4_0 | 4787 MB | 3737 MiB | 576 / 1152 / 2304 MiB | **2 / 3 / >4** | 3 / 3 / 4 |
| LFM2.5-8B-A1B | Q4_0 | 4844 MB | 4406 MiB | 48 / 96 / 192 MiB | **2 / 3 / 2** | 3 / 3 / 4 |
| granite-4.2-8b | Q4_0 | 5055 MB | 4276 MiB | 640 / 1280 / 2560 MiB | **3 / 3 / 4** | 4 / 4 / 4 |
| gemma-4-E4B-it | Q4_0 | 5154 MB | 2171 MiB | 154 / 218 / 346 MiB | **1 / 1 / 1** | 2 / 2 / 2 |
| Qwen3.5-9B | Q4_0 | 5741 MB | 3771 MiB | 128 / 256 / 512 MiB | **2 / 2 / 2** | 3 / 3 / 3 |
| gemma-4-12b-it-qat | Q4_0 | 6975 MB | 5848 MiB | 1344 / 1488 / 1616 MiB | **3 / 3 / 4** | 4 / 4 / 4 |

### Reading the table

- **Sessions** is the smallest count that loaded and generated on *every* attempt; `>n`
  means nothing up to `n` did. Nine configurations loaded on some attempts and not on
  others (see [Where the limit is](#where-the-limit-is)), which is why the column reports
  what always worked rather than what once did. A cell tried only once is weaker evidence:
  LFM2.5-8B's 8k cell failed on two sessions once, while its 16k cell was tried on two only
  once and passed — sampling, not physics.
- **Sized** is what `configure-llamacpp.py` asks for.
- **On NPU** is the weight buffer `llama-server` reports, which is not the file size.
- **KV** is the whole cache, summed over the sessions — what the context size buys. It
  grows linearly with the context for a plain attention model, stops growing for a
  sliding-window one (the gemmas), and stays small for a hybrid one, which only caches on
  its attention layers (Qwen3.5-4B: 8 layers of 32).
- **Q4_0 pure** is `llama-quantize --pure`, every tensor forced to q4_0 rather than keeping
  the embeddings and a few sensitive layers wider
  ([MODEL_QUANTIZATION.md](MODEL_QUANTIZATION.md)). It is in the table because it differs
  from the plain Q4_0 of the same model by 469 MiB of NPU weights, which is a session at
  some contexts.
- **A 20B model (11.5 GB) was left out.** A loaded model costs about 1.6x its GGUF size in
  RAM, so it would need around 19 GB, and the harness refuses a model that would take the
  board down with it.

## How a model uses a session

Three things decide how much of a model lands on each session. All three are computable
from the GGUF header, and all three were checked against what `llama-server` logs.

### Weights: what reaches the NPU

Sizing by file size cannot work: gemma-4-E4B (5.15 GB) runs on one session while
Qwen3-4B-Instruct-2507 (2.38 GB) needs two. What reaches the NPU ranges from 31% of the
file (gemma-4-E2B) to 99% (Qwen3.5-4B pure), for two reasons:

- The Hexagon backend does not repack the K-quants, so those tensors stay on the CPU. The
  matformer gemmas keep two thirds of their bytes in q6_K, which is the whole of their
  apparent size problem: gemma-4-E2B is a 3.35 GB file that puts 1.03 GB on the NPU.
- The token embeddings of a model with a separate `output.weight` stay on the CPU too —
  they are only read by `get_rows`. With tied embeddings that same tensor is also the
  output projection, and then it does go to the NPU (gemma-3-1b: all 682 MiB of it).

Summing the tensor index under those two rules reproduces the `HTP model buffer size`
`llama-server` reports **to within a MiB on 18 of the 19 models** (Qwen3.5-9B is 6% over).

### KV cache: the context axis

Per layer that owns a cache, `n_head_kv x (key_length + value_length)` elements a token,
2 bytes each for an f16 cache — llama.cpp's default, and what the service runs.

- Which layers own a cache comes out of the tensor index: the recurrent layers of a hybrid
  model, and the layers of a gemma that share another layer's cache, have no K projection
  of their own.
- A layer flagged in `attention.sliding_window_pattern` holds only
  `n_seq_max x window + n_ubatch` cells, however large the context.

This reproduces every `llama_kv_cache: size =` line in the logs exactly, including the
two-cache split of the gemmas: gemma-4-12b at 16k has 256 MiB over its 8 full-attention
layers plus 1360 MiB over its 40 windowed ones.

### The split across sessions

The busiest session takes one layer's worth more than an even share of both weights and
KV cache:

```
share = (ceil(n_layer / sessions) + 1) / n_layer
```

granite-4.2-3b at 16k over two sessions: 21/40 of 1280 MiB = 672 MiB, and 672 MiB is what
the log says. It holds for every model and session count measured.

## Where the limit is

### The failure

When a session runs out, the load fails on the buffer that no longer fits:

```
ggml-hex: HTP0 buffer mapping failed : domain_id 3 size 536883200 fd 5 error 0x00000001
ggml-hex: HTP0 failed to allocate device buffer context: ggml-hex: fastrpc_mmap failed
alloc_tensor_range: failed to allocate HTP0 buffer of size 536870912
llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache
```

512 MiB is `GGML_HEXAGON_MBUF`, the largest buffer the backend maps at once, so the failure
lands on a whole chunk rather than on the byte that overflowed.

The DSP buffers come from plain system memory (`/dev/dma_heap/system`), so **the same
message appears when the board is simply out of RAM** — check free memory before reading a
load failure as a sizing problem. It was not the cause here: available memory never dropped
below 7.8 GB across the run.

### Not a single number

The backend reports about 3.1 GiB of DSP address space per session (`ggml-hex: HTP0 op
batching: ... vmem 3285700608`). Loads succeeded holding as much as 2.9 GiB and failed
holding as little as 1.8 GiB, and the measurements do not separate at any threshold:

| Configuration | Busiest session | Outcome |
|---|---|---|
| Qwen3.5-4B pure @8k, 1 session | 2259 MiB weights + 457 MiB context | loads |
| Phi-4-mini @8k, 1 session | 1734 MiB weights + 1024 MiB context | fails |
| Phi-4-mini @16k, 2 sessions | 924 MiB weights + 1088 MiB context | loads |
| Qwen3-8B @16k, 4 sessions | 931 MiB weights + 512 MiB context | fails |

The last two are the awkward pair: the configuration that fails asks for strictly less of
both than the one that loads, so **no rule monotone in (weights, KV) can separate them**.
The total mapped across all sessions does not separate them either — granite-4.2-8b maps
6.8 GiB over four sessions and loads, Qwen3-8B fails at 5.9 GiB over the same four. What
the failures have in common is the fourth session and a 16k context.

The per-session figures in these tables are `llama-server`'s own projection, compute buffers
included. The budget in [The sizing rule](#the-sizing-rule) is compared against weights
plus KV cache and state only, which is 30 to 900 MiB less depending on the model, so the two
sets of numbers are not directly comparable.

### Not reproducible near the edge

Of the 87 configurations, 63 were attempted more than once — a repeat run of the whole
matrix, plus nine boundary cases run three times each, interleaved, with a minute of idle
before every trial. **Nine of the 63 gave different verdicts on different attempts:**

| Configuration | Busiest session | Attempts |
|---|---|---|
| LFM2.5-8B-A1B @8k, 2 sessions | 3132 MiB | pass, fail |
| Qwen3.5-4B Q8_0 @16k, 2 sessions | 2891 MiB | pass, pass, pass, fail, pass |
| Nemotron-Mini-4B @16k, 2 sessions | 2514 MiB | pass, fail |
| granite-4.2-3b @8k, 1 session | 2361 MiB | fail, pass, pass, pass, fail |
| Qwen3-8B @16k, 3 sessions | 2281 MiB | fail, pass |
| Phi-4-mini @4k, 1 session | 2278 MiB | pass, pass, pass, pass, fail |
| Phi-4-mini @16k, 2 sessions | 2086 MiB | pass, fail |
| Qwen3-8B @16k, 4 sessions | 1779 MiB | fail, fail, fail, pass |
| DeepSeek-R1-0528-Qwen3-8B @16k, 4 sessions | 1779 MiB | pass, fail |

Four things follow:

- **The per-session total does not order the outcomes.** Among the configurations whose
  verdict never changed, one holding 2771 MiB loaded every time while one holding 2436 MiB
  failed every time, and the stable verdicts interleave with the unstable ones all the way
  up the range. Whatever the allocator runs out of is not a number these measurements can
  see, so a budget can only sit *below the lowest load ever measured to fail*, not near the
  highest that loaded.
- **The flakiness is not confined to the extremes** — it appears from 1779 to 3132 MiB — so
  a single measurement of a tight configuration means little. Every minimum in the matrix
  is the count that worked on *every* attempt.
- **Part of it is the trial order.** The first run walked the configurations back to back,
  about 3 seconds between a `SIGTERM` and the next load; granite-4.2-3b at 8k on one
  session failed that way and then loaded three times running when given a minute of idle
  first. RAM comes back within one 2-second sample of the kill, so what lingers is the
  protection domain rather than memory. Back-to-back numbers are the pessimistic reading —
  and the one that matters, since the router keeps `LLAMA_ARG_MODELS_MAX` models resident
  and swaps them, so a real load can land on a DSP that was busy moments ago.
- **The fourth session is the worst of it.** Two 8B models at 16k failed there at a
  per-session load nothing else fails at, one of them after passing once, and the two are
  the same architecture with identical tensor shapes. Nothing about a model predicts this,
  which is why needing four sessions is treated as a reason to shrink the context rather
  than as an allocation to make.

## The sizing rule

`configure-llamacpp.py` reads each GGUF header and asks for the smallest number of
sessions whose busiest session stays inside the budget:

```
npu_bytes(ctx) = npu_weight_bytes + kv_cache_bytes(ctx) + recurrent_state
session_bytes  = npu_bytes(ctx) * (ceil(n_layer / sessions) + 1) / n_layer
sessions       = min n <= 4 such that session_bytes <= 1800 MiB
```

Then, because needing all four sessions is where the allocator stops being dependable, it
halves the context until no model needs the fourth — down to a floor of 4096.

Compute buffers are left out on purpose. They range from 11 MiB to 863 MiB a session, and
llama.cpp retries a smaller graph when one does not fit, so they are elastic in a way
weights and KV are not: LFM2.5-8B loads happily with 863 MiB of compute buffers on top of
2260 MiB of weights and KV.

### The budget

1800 MiB is the largest budget that **never asks for fewer sessions than a configuration
was measured to need**, scoring every configuration by its worst attempt rather than its
luckiest. What binds it is Nemotron-Mini-4B at 16k, which wants 1838 MiB across two
sessions and failed there — so the margin is 2%, and it is thin on purpose. Dropping to
1700 MiB would cap the context of a board with an 8B model installed from 8192 to 4096, and
8192 on three sessions is measured to work in every run. A spare session costs a little
throughput; halving everyone's context does not.

Two consequences are worth naming:

- **Qwen3.5-4B Q8_0 gets four sessions at 16k** where two were measured to work four times
  in five. Giving it three would need a budget of 1887 MiB, above the 1838 MiB that fails,
  so no single budget serves both. The size tables this replaces also gave it four:
  nothing is lost, only not gained.
- **LFM2.5-8B gets four sessions at 16k** against a measured two — on the strength of one
  attempt at two sessions at 16k, and a failure at two sessions at 8k.

### Against the size tables it replaces

The previous sizing used two tables keyed on the GGUF size, one per context band, plus
per-model pins for the gemmas. Scored against the 53 measured cells that have a session
count that always worked:

| | Under-allocated (load fails) | Over-allocated | Exact |
|---|:---:|:---:|:---:|
| GGUF-size tables | 2 | 30 | 21 |
| GGUF header + budget | 0 | 23 | 30 |

The two under-allocations are real: Qwen3-4B-Instruct-2507 and Phi-4-mini at 4k are both
sized for one session and need two. Beyond the counts, the sizing now gets three things the
tables could not:

- **The gemma pins are gone.** gemma-4-E2B comes out at one session and gemma-4-E4B at two
  from their tensor types alone, and both keep the full context without an exemption.
- **Quantization is an axis.** Qwen3.5-4B is sized differently as Q4_0 and as Q8_0, because
  the sizing weighs its tensors instead of its file.
- **The context degrades gradually.** With an 8B model installed the tables cut straight
  from 16k to 4k; the budget lands on 8k, where that model was measured to run on three
  sessions in every attempt. A board carrying only 4B-class models keeps 16k on two.

### When llama.cpp changes

Reading the header means depending on llama.cpp's formats, so the sizing degrades in tiers
rather than guessing:

1. **The tensor type table comes from the shipped `libggml`.** A tensor's size is
   `elements / block size * bytes per block`, and the GGUF index carries neither, so
   `configure-llamacpp.py` asks the library in the image (`ggml_type_name`,
   `ggml_blck_size`, `ggml_type_size`) and only falls back to its own table if that cannot
   be loaded. A bump that adds a quantization needs no edit here — the built-in table was
   three types out of date one bump after being written, which is why it is the second
   source and not the first.
2. **A type neither source knows is refused, not assumed.** It would otherwise size as zero
   bytes and the model would look free, which is the one error that fails a load.
3. **A header that cannot be read, or that says nothing a model would say, falls back to
   sizing by file size** — the tables the runner used before it read headers, thresholds
   and gemma pins unchanged. Less accurate, but it always produces a number; a model left
   out of the count entirely is exactly how the server ends up with too few sessions.

Only a file that cannot be read at all is skipped. What no tier can catch is llama.cpp
changing a *meaning* rather than a name — the sliding-window cell count, or which layers own
a cache — so the matrix is worth re-measuring after a bump ([Running it
again](#running-it-again)).

## A quantized KV cache

The cache is f16 because that is llama.cpp's default, not because the DSP requires it:
`ggml_hexagon_supported_flash_attn_ext` takes K and V in f16 *or* q8_0, and `set_rows` can
write into a q8_0 destination when the head is at least 32 elements wide. Measured with
`LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q8_0`, it works and it is worth a lot:

| Configuration | KV f16 | KV q8_0 | Sessions f16 → q8_0 |
|---|---|---|:---:|
| Qwen3.5-4B @16k | 512 MiB, fails on 1 | 272 MiB, loads on 1 | 2 → 1 |
| Phi-4-mini @16k | 2048 MiB, needs 3 | 1088 MiB, loads on 2 | 3 → 2 |
| Qwen3-8B @16k | 2304 MiB, **fails on every count** | 1224 MiB, loads on 3 | none → 3 |

The cache costs 53% of f16 (34 bytes per 32 elements); every buffer stays on the NPU — the
logs show `HTP0 KV buffer size`, so attention did not fall back to the CPU; prefill is
unchanged and decode is within the run-to-run spread of this board. On a model set with an
8B model it would take the served context from 8k to 16k at the same three sessions.

### Why it stays off

The DSP's `CPY` op only handles f32 and f16. In ordinary graphs that is harmless — the host
refuses the other types in `supports_op` and runs the copy on the CPU — but the copies the
scheduler injects for inputs that cross a session boundary are not type-checked, and a DSP
error response to one is only logged. A model whose layers *share* a KV cache (the
matformer gemmas) therefore has a quantized cache view copied between sessions once per
token, silently and wrongly:

| Model | Sessions | KV | Outcome |
|---|:---:|---|---|
| gemma-4-E4B | 1 | q8_0 | correct |
| gemma-4-E4B | 2 | q8_0 | correct |
| gemma-4-E2B | 2 | q8_0 | output rejected by llama.cpp's gemma-4 parser (HTTP 500), then the server declined every later request |
| gemma-4-E2B | 3 | q8_0 | fluent nonsense: "capital of France" → "Rome Wait, I apologize. The answer is **Lapula**" |
| gemma-4-E4B | 2 | f16 | correct (control) |

E4B survives a two-way split and E2B does not, so whether it breaks depends on where the
split falls relative to the shared views, not on the model alone — which makes it
unpredictable from the header. Turning a quantized cache on would mean either keeping every
KV-sharing model on a single session, or waiting for the DSP's `CPY` to handle quantized
types (or for the host to fall back to a CPU copy for them).

Meanwhile `configure-llamacpp.py` reads `LLAMA_ARG_CACHE_TYPE_K` and `_V` and sizes the
cache from them, so anyone who does set them gets sessions sized for the cache they will
actually get. Unset — the default, and what the service ships — that is f16.

## Method

Each trial runs one model on a freshly started `llama-server`, with the environment
[`service_compose.yaml`](../../../src/arduino/app_services/llamacpp/service_compose.yaml)
gives the container (`LLAMA_ARG_BATCH=1024`, `LLAMA_ARG_UBATCH=256`,
`LLAMA_ARG_FLASH_ATTN=on`, `LLAMA_ARG_THREADS=4`, `LLAMA_ARG_CPU_MASK=0x0f`,
`GGML_HEXAGON_OPBATCH=2048`, `GGML_HEXAGON_MBUF=512`) and the arguments
[`run-model-router.sh`](scripts/run-model-router.sh) builds (`--device HTP0[,HTP1...]
-ngl 100 --load-mode none`), so that what is measured is what the service will do.

It waits for `/health`, then asks two questions over `/v1/chat/completions` at
temperature 0:

- `Tell me the steps to prepare and cook an apple pie. Use a max of 150 words.` — the
  generation load, and a check that the answer is on topic.
- `What is the capital of France? Answer with the city name only.` — a control question
  with one right answer, so that a model which loads but computes garbage is not counted
  as a success.

A trial passes when the server comes up and both answers arrive. Answer quality is recorded
separately, so that a reasoning model rambling past its token budget is not mistaken for an
allocation failure. Every trial that loaded answered both questions correctly — the one
thing that ever loaded and then degraded was the quantized cache on a KV-sharing model,
which is exactly what the control question caught.

For each model the search walks the session count up from one — seeded, for the larger
contexts, at what the smaller one needed, since the KV cache only grows — and stops at the
first count that passes. The per-session figures quoted throughout come from the projection
`llama-server` logs under `-v` before it allocates anything, so they exist for the loads
that fail too:

```
common_memory_breakdown_print: | - HTP0 (Hexagon) | 0 = 0 + (2361 = 1688 + 640 + 33) + -2361 |
```

### Running it again

Worth redoing after a llama.cpp bump, a DSP firmware change, or a change to
`GGML_HEXAGON_MBUF` — any of them can move the envelope — and after any change to which
tensor types the Hexagon backend keeps on the NPU, which would invalidate the K-quant rule.

[`tools/`](tools/) holds the harness: `session-trial.sh` runs one (model, context, sessions)
trial, `session-matrix.sh` walks the matrix, `repro.sh` repeats a configuration to test
whether its verdict is stable. They need `bash`, `curl`, `python3` and access to the DSP
devices, and they kill any `llama-server` from the same package before each trial, so run
them on a board where the service itself is stopped. Point `LLAMACPP_PREFIX` at the
unpacked llama.cpp package and run:

```bash
LLAMACPP_PREFIX=/opt/pkg-snapdragon ./session-matrix.sh /path/to/models
```

Results land in `./session-matrix/results.txt`, one line of key=value fields per trial,
with the trimmed server log beside it. `SETTLE` sets the idle time before each trial
(default 60 seconds; 0 reproduces what back-to-back loads see) and `CTX_SIZES` the contexts
to walk. The run is resumable: trials already in `results.txt` are not repeated.
