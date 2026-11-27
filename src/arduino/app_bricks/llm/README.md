# Large Language Model - Text Brick

The Large Language Model - Text Brick provides functionality for interacting with locally-based LLMs such as Qwenm, LLama, Gemma. It offers a simplified, unified interface for chatting with the local models. It supports both synchronous 'one-shot' responses and streaming output, with optional conversational memory.

## Overview

This Brick acts as a gateway to powerful AI models hosted locally. Whether you need a simple one-off answer or a continuous conversation with memory, the LLM Brick provides a unified API for different providers.

## Features

- **Multi-LLM Support**: Compatible with multiple LLM models including Qwen and Gemma. The default model used is different depending on the board that is used. Other models can be set and downloaded in Arduino App Labs.
- **Conversational Memory**: Built-in support for windowed history, allowing the AI to remember context from previous exchanges.
- **Streaming Responses**: Receive text chunks in real-time as they are generated, ideal for responsive user interfaces.
- **Configurable Behavior**: Customize system prompts, temperature (creativity), and request timeouts.

## Code Example and Usage

### Basic Conversation

This example initializes the Brick with an local model and performs a simple chat interaction. Models must be downloaded and available locally.

```python
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

llm = LargeLanguageModel()


def ask_prompt():
    prompt = "Hi, what can you do as an AI assistant?"
    print(llm.chat(prompt))
    print()
    raise StopIteration


App.run(ask_prompt)
```

### Streaming with Memory

This example demonstrates how to start a Local LLM chat with persistent memory.

```python
from arduino.app_bricks.cloud_llm import SQLMessagePersistence
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

db = SQLStore("llm_persistent_demo.db")
db.start()

llm = LargeLanguageModel(
    system_prompt="You are a helpful assistant.",
).with_memory(
    max_messages=10,
    persistence=SQLMessagePersistence(sql_store=db, thread_id="llm-demo-conversation"),
)


def ask_prompt():
    prompt = "Hi, what can you do as an AI assistant?"
    print(llm.chat(prompt))
    raise StopIteration


App.run(ask_prompt)
```

## Methods

- **`chat(message, images=None, reasoning_effort=None)`**: Sends a message (with optional image file paths or raw bytes) and returns the complete response string. Blocks until generation is finished. `reasoning_effort` optionally sets how much a reasoning model thinks before answering, as a discrete level or an integer token budget, when the local runner supports it.
- **`chat_stream(message, images=None)`**: Returns a generator yielding response tokens as they arrive.
- **`stop_stream()`**: Interrupts an active streaming generation.
- **`with_memory(max_messages=10, persistence=None, on_context_full="drop_oldest")`**: Enables history tracking. `max_messages` is the window size sent to the model. `persistence=True` enables persistence with a dedicated default database/thread; pass a `MessagePersistence` (e.g. `SQLMessagePersistence`) for full control. A turn enters the memory once the model has answered it. `on_context_full` says what happens when a request does not fit in the context window of the model: `"drop_oldest"` (default) drops the oldest whole turns from the memory, down to about half of the context window, logs a warning and sends the request again; `"raise"` raises `ContextLengthExceededError` and leaves the memory as it is; a callable takes the messages of the memory window (oldest first, without the system prompt) and returns fewer messages to keep in their place, for example a summary of the oldest turns followed by the newest ones. The new message is never cut, and the persistence backend keeps the whole conversation.
- **`clear_memory()`**: Resets the conversation history (also deletes persisted rows for the active thread when a persistence backend is configured).
- **`list_models()`**: Returns the model identifiers available on the local inference service.

## Errors

- `ContextLengthExceededError`: raised by `chat()` and `chat_stream()` when the request does not fit in the context window of the model and the memory cannot make room for it: the new message alone does not fit, or `on_context_full` is `"raise"`. The request holds the system prompt, the messages in the memory window and the new message, with its images and tool results. `prompt_tokens` and `context_size` hold the sizes the llama.cpp runner reports. The turn that failed is not added to the memory, and the memory is left as it was. On a `llamacpp:` model, `LLAMA_ARG_CTX_SIZE` in the variables of the Brick in `app.yaml` sets a larger context window. It is a `RuntimeError`, like the other request failures.

```python
from arduino.app_bricks.llm import ContextLengthExceededError, LargeLanguageModel

llm = LargeLanguageModel().with_memory(on_context_full="raise")


def answer(question: str) -> str:
    try:
        return llm.chat(question)
    except ContextLengthExceededError:
        # The conversation filled the context: start a new one
        llm.clear_memory()
        return llm.chat(question)
```
