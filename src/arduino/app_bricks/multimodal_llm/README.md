# Large Language Model - Multimodal Brick

The Large Language Model - Multimodal Brick provides a simple Python® interface for chatting with a locally hosted AI model using text, images and audio. It lets Arduino® App Lab applications send a text prompt alone, together with images, together with audio clips, or with both, then receive generated text from the model running on the board.

## Overview

The Multimodal Language Model Brick is designed for applications that need to understand what the board sees and hears without sending camera frames or recordings to the cloud.

Use this Brick when your application needs to answer a spoken request, transcribe or summarize a recording, describe a camera frame, or combine what is heard with what is seen. Recordings made with the `Microphone` peripheral can be passed directly.

Brick requires a multimodal llm, like Gemma4 E2B/E4B.

## Features

- **Local multimodal AI**: Sends prompts to a model running on the board through the local llama.cpp model service.
- **Text, image and audio prompts**: Accepts a text message with optional images and optional audio clips, alone or together in the same message.
- **Microphone ready**: Accepts the arrays returned by `Microphone.record_pcm()` and `Microphone.record_wav()` as they are.
- **Synchronous responses**: Uses `chat()` when the application needs the full answer before continuing.
- **Streaming responses**: Uses `chat_stream()` when the application should display generated text as it arrives.
- **Conversation memory**: Keeps recent chat history with `with_memory()` and can persist it across restarts with `persistence=True`.
- **Configurable generation**: Supports system prompts, temperature, token limits, timeouts, tools and model overrides.
- **Advanced access**: Exposes the underlying LangChain chat model through `get_client()` for custom integrations.

## Prerequisites

- A supported board with NPU acceleration, like `ventunoq`.
- A compatible multimodal model downloaded and configured in Arduino App Lab.
- The `arduino:multimodal_llm` Brick added to the application from App Lab.
- Image input as a valid file path or encoded image bytes. Camera frames should be encoded as JPEG bytes before calling the Brick.
- Audio input as a WAV or MP3 file path, WAV or MP3 bytes, or a recording from the `Microphone` peripheral. For live audio, a microphone connected to the board.

## Code example and usage

### Ask a Text Question

This example sends a text-only prompt and waits for the complete model response.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel(system_prompt="You are a concise assistant.")

print(model.chat("How many legs does a spider have?"))
```

### Analyze an Image File

This example sends a local image file together with the prompt.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel(
    system_prompt="You are a concise visual assistant.",
    max_tokens=120,
)

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

### Stream a Transcription

Use `chat_stream()` when a web UI or terminal interface should show text as the model generates it. `audio` also takes `.wav` and `.mp3` file paths, or the bytes of a WAV or MP3 file.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel()

for chunk in model.chat_stream("Transcribe exactly what is said in this audio.", audio=["note.wav"]):
    print(chunk, end="", flush=True)
```

### Analyze a Camera Frame and a Recording Together

Images and audio can be sent in the same message. This pattern captures the latest camera frame as JPEG bytes and lets the user ask about it by voice.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel(
    system_prompt="You are a helpful visual assistant.",
    max_tokens=120,
)

def answer_about_frame(frame_bytes: bytes, question_wav: bytes) -> str:
    return model.chat(
        message="Answer the spoken question about this picture.",
        images=[frame_bytes],
        audio=question_wav,
    ).strip()
```

### Enable Conversation Memory

Memory is off by default. Use `with_memory()` when follow-up prompts should keep recent context across calls. Pass `persistence=True` to retain memory across restarts: this uses the `SQLStore` brick internally, so no extra setup is required. The text of every turn stays in the history, but only the latest message with images or audio keeps them: in earlier messages each image becomes the text `[image]` and each audio clip `[audio]`. Persisted history stores the placeholders only.

```python
from arduino.app_bricks.multimodal_llm import MultimodalLanguageModel

model = MultimodalLanguageModel(
    system_prompt="You remember relevant details of what you see and hear.",
).with_memory(
    max_messages=6,
    persistence=True,
)

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
| `tools` | `Sequence[ToolLike]` \| `None` | `None` | Optional tools the model can call: `BaseTool` objects declared with the `@tool` decorator exported by the Brick, or plain callables (auto-wrapped into tools). |
| `**kwargs` | `dict` | `{}` | Additional keyword arguments passed to the underlying model constructor. |

## Methods

- **`chat(message, images=None, audio=None, reasoning_effort=None)`**: Sends a prompt with optional images and audio, then returns the complete generated response as a string. `reasoning_effort` optionally sets how much a reasoning model thinks before answering, as a discrete level or an integer token budget, when the local runner supports it.
- **`chat_stream(message, images=None, audio=None)`**: Sends a prompt with optional images and audio, then yields generated text chunks as they arrive.
- **`stop_stream()`**: Requests cancellation of the active streaming response.
- **`with_memory(max_messages=0, persistence=None)`**: Enables conversational memory for the instance. `persistence=True` enables persistence with a default database/thread; pass a `MessagePersistence` (importable as `from arduino.app_bricks.cloud_llm.memory import MessagePersistence`) for full control. Pass `max_messages=0` to disable history. Only the latest message with images or audio keeps them; earlier ones keep `[image]` and `[audio]` placeholders.
- **`clear_memory()`**: Clears the active conversation history.
- **`list_models()`**: Returns the model identifiers available on the local inference service.
- **`get_client()`**: Returns the underlying LangChain chat model, or the `Runnable` that binds it to the registered tools when `tools` are set.

## Image Inputs

The `images` argument accepts a list containing:

- File paths, such as `"street.jpg"`.
- Encoded image bytes, such as a JPEG frame captured from a camera.

When an image path is used, the file must exist in the application runtime environment. When bytes are used, make sure they represent an encoded image format: raw pixel arrays are not accepted, so convert camera frames to JPEG bytes before passing them to the Brick.

## Audio Inputs

The `audio` argument accepts one clip or a list of clips. Each clip can be:

- A `.wav` or `.mp3` file path, such as `"note.wav"`.
- The bytes of a WAV or MP3 file.
- The array returned by `Microphone.record_wav()`: a complete WAV file, whose header keeps the sample rate and channel count.
- The array returned by `Microphone.record_pcm()`: raw int16 samples, or float samples between -1.0 and 1.0. Raw samples carry no header and are read as 16 kHz mono, the `Microphone` defaults. If the microphone uses another sample rate or channel count, use `record_wav()` instead.

Keep clips short: a spoken request of a few seconds works best, while long recordings take longer to encode and need more memory.

## Troubleshooting

### Model not found

**Fix:** Verify that the selected multimodal model is downloaded and available in App Lab. If you override the model, make sure the identifier matches a model exposed by the local `llamacpp` service.

### The model could not encode the image or audio input

**Fix:** The model runner ran out of memory while encoding the media. Use a shorter audio clip or a smaller image, close other running applications, and check the logs of the models runner.

### The model follows the spoken instruction instead of the prompt

**Fix:** The Brick sends images and audio before the text, and this order cannot be changed. When the recording itself contains a command, the model may obey it and ignore the written prompt. Keep the written prompt consistent with what the recording is for, and do not give it an instruction that competes with the spoken one.

### Image or audio file not found

**Fix:** Use a path that exists inside the application container, or pass image bytes or the recording from the `Microphone` peripheral instead of a file path.

### Different answers to the same question

**Fix:** Set `temperature=0.0` (the default). Higher temperatures make the model sample a different answer each time.

### Response generation fails with a memory error

**Fix:** Reduce `max_tokens`, use shorter clips and smaller images, close other running applications, and restart the app. The model and its encoders can require significant memory during loading and inference.
