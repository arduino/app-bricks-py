# Large Language Model (LLMs) Brick

The Large Language Model (LLM) Brick provides functionality for interacting with locally-based LLMs such as Qwenm, LLama, Gemma. It wraps LangChain functionality to provide a simplified, unified interface for chatting with the local models. It supports both synchronous 'one-shot' responses and streaming output, with optional conversational memory.

## Overview

This Brick acts as a gateway to powerful AI models hosted locally. Whether you need a simple one-off answer or a continuous conversation with memory, the LLM Brick provides a unified API for different providers.

## Features

- **Multi-Provider Support**: Compatible with multiple LLM providers including Qwenm, LLama and Gemma.
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

- **`chat(message)`**: Sends a message and returns the complete response string. Blocks until generation is finished.
- **`chat_stream(message)`**: Returns a generator yielding response tokens as they arrive.
- **`stop_stream()`**: Interrupts an active streaming generation.
- **`with_memory(max_messages, persistence=None)`**: Enables history tracking. `max_messages` is the window size sent to the model. `persistence=True` enables persistence with a dedicated default database/thread; pass a `MessagePersistence` (e.g. `SQLMessagePersistence`) for full control.
