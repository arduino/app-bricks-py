# Multimodal Language Model Brick

The Multimodal Language Model Brick provides a simple Python® interface for chatting with a locally hosted AI model using text, images and audio. It lets Arduino® App Lab applications send a text prompt alone, together with images, together with audio clips, or with both, then receive generated text from the model running on the board.

## Overview

The Multimodal Language Model Brick is designed for applications that need to understand what the board sees and hears without sending data to the cloud. It uses the same chat-style API as the LLM and VLM Bricks, and adds an `audio` parameter next to `images`.

Use this Brick when your application needs to answer questions about a spoken request, transcribe or summarize a recording, describe a camera frame, or combine what is heard with what is seen. Recordings made with the `Microphone` peripheral can be passed directly.

The Brick uses the model configured for `arduino:multimodal_llm` in Arduino App Lab. The default model is Gemma 4 E2B, served by the local llama.cpp model runner.

## Features

- **Local multimodal AI**: Sends prompts to a model running on the board through the local llama.cpp service.
- **Text, image and audio prompts**: Accepts a text message with optional images and optional audio clips, alone or together in the same message.
- **Microphone ready**: Accepts the arrays returned by `Microphone.record_pcm()` and `Microphone.record_wav()` as they are.
- **Deterministic by default**: The temperature defaults to `0.0`, so the same input always gets the same answer.
- **Synchronous responses**: Uses `chat()` when the application needs the full answer before continuing.
- **Streaming responses**: Uses `chat_stream()` when the application should display generated text as it arrives.
- **Conversation memory**: Keeps recent chat history with `with_memory()` and can persist it across restarts with `persistence=True`.
- **Advanced access**: Exposes the underlying LangChain chat model through `get_client()` for custom integrations.

## Prerequisites

- A supported board. The current Brick configuration supports `ventunoq`.
- A compatible multimodal model downloaded and configured in Arduino App Lab.
- The `arduino:multimodal_llm` Brick added to the application from App Lab.
- For live audio, a microphone connected to the board and the `Microphone` peripheral.

**Note:** The model runs locally through the board model service, so cloud inference and cloud API keys are not required.

## Code example and usage

### Ask a Text Question

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel(system_prompt="You are a concise assistant.")

print(model.chat("How many legs does a spider have?"))
```

### Analyze an Image

`images` takes a list of image file paths or encoded image bytes, such as a JPEG camera frame.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel()

response = model.chat(
    message="How many cars are in this image? Answer with a number.",
    images=["street.jpg"],
)
print(response)
```

### Answer a Spoken Request from the Microphone

`record_pcm()` returns raw 16 kHz mono samples, the `Microphone` defaults. Pass them directly as `audio`.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel
from arduino.app_peripherals.microphone import Microphone

model = MultimodalLanguageModel(system_prompt="You are a helpful voice assistant. Answer briefly.")

mic = Microphone()
mic.start()
recording = mic.record_pcm(5)  # 5 seconds of audio
mic.stop()

print(model.chat("Answer the question asked in this recording.", audio=recording))
```

### Transcribe an Audio File

`audio` also takes `.wav` and `.mp3` file paths, or the bytes of a WAV or MP3 file.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel()

for chunk in model.chat_stream("Transcribe exactly what is said in this audio.", audio=["note.wav"]):
    print(chunk, end="", flush=True)
```

### Combine an Image and Audio

Images and audio can be sent in the same message.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel()

response = model.chat(
    message="Answer the spoken question about this picture.",
    images=["desk.jpg"],
    audio=["question.wav"],
)
print(response)
```

### Enable Conversation Memory

Memory is off by default. Use `with_memory()` when follow-up prompts should keep recent context. Images and audio are stored in the history too, so keep the window small.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel().with_memory(max_messages=6)

print(model.chat("Remember what is in this image.", images=["desk.jpg"]))
print(model.chat("What object did I show you earlier?"))
```

## Configuration

The Brick is initialized with the following parameters:

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `system_prompt` | `str` | `""` | System-level instruction that defines the assistant behavior. |
| `temperature` | `float` \| `None` | `0.0` | Controls randomness. `0.0` gives the same answer to the same input; raise it (e.g. `0.7`) for more varied, conversational replies. |
| `max_tokens` | `int` | `512` | Maximum number of tokens to generate in the response. |
| `timeout` | `int` \| `None` | `None` | Maximum time in seconds to wait for a response. |
| `tools` | `list[Callable]` | `None` | Optional tool functions the model can call, declared with the `@tool` decorator exported by the Brick. |
| `model` | `str` \| `None` | App Lab configured model | Local model identifier configured for `arduino:multimodal_llm` in App Lab, e.g. `llamacpp:gemma-4-E2B-it-Q4_0`. |
| `**kwargs` | `dict` | `{}` | Additional keyword arguments passed to the underlying model constructor. |

## Methods

- **`chat(message, images=None, audio=None, reasoning_effort=None)`**: Sends a prompt with optional images and audio, then returns the complete generated response as a string. `reasoning_effort` optionally sets how much a reasoning model thinks before answering, as a discrete level or an integer token budget, when the local runner supports it.
- **`chat_stream(message, images=None, audio=None)`**: Sends a prompt with optional images and audio, then yields generated text chunks as they arrive.
- **`stop_stream()`**: Requests cancellation of the active streaming response.
- **`with_memory(max_messages=0, persistence=None)`**: Enables conversational memory for the instance. `persistence=True` enables persistence with a default database/thread; pass a `MessagePersistence` (importable as `from arduino.app_bricks.cloud_llm.memory import MessagePersistence`) for full control. Pass `max_messages=0` to disable history.
- **`clear_memory()`**: Clears the active conversation history.
- **`get_client()`**: Returns the underlying LangChain `BaseChatModel` instance.

## Image Inputs

The `images` argument accepts a list containing:

- File paths, such as `"street.jpg"`.
- Encoded image bytes, such as a JPEG frame captured from a camera. Raw pixel arrays are not accepted: encode camera frames to JPEG first.

## Audio Inputs

The `audio` argument accepts one clip or a list of clips. Each clip can be:

- A `.wav` or `.mp3` file path, such as `"note.wav"`.
- The bytes of a WAV or MP3 file.
- The array returned by `Microphone.record_wav()`: a complete WAV file, whose header keeps the sample rate and channel count.
- The array returned by `Microphone.record_pcm()`: raw int16 samples, or float samples between -1.0 and 1.0. Raw samples carry no header and are read as 16 kHz mono, the `Microphone` defaults. If the microphone uses another sample rate or channel count, use `record_wav()` instead.

Keep clips short: a spoken request of a few seconds works best, while long recordings take longer to encode and need more memory.

## Troubleshooting

### The model does not accept this kind of input

**Fix:** The selected model has no encoder for images or audio. Select a multimodal model for `arduino:multimodal_llm` in App Lab.

### The model could not encode the image or audio input

**Fix:** The model runner ran out of memory while encoding the media. Use a shorter audio clip or a smaller image, close other running applications, and check the logs of the models runner.

### Audio file not found

**Fix:** Use a path that exists inside the application container, or pass the recording directly from the `Microphone` peripheral.

### Different answers to the same question

**Fix:** Set `temperature=0.0` (the default). Higher temperatures make the model sample a different answer each time.
