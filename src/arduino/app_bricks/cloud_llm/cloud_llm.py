# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import threading
from typing import Iterator, Optional, Union

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langsmith import uuid7

from arduino.app_utils import Logger, brick

from .models import CloudModel
from .memory import WindowedChatMessageHistory

logger = Logger("CloudLLM")
DEFAULT_MEMORY = 10


class AlreadyGenerating(Exception):
    """Exception raised when a generation is already in progress."""

    pass


@brick
class CloudLLM:
    """A Brick for interacting with cloud-based Large Language Models (LLMs).

    This class wraps LangChain functionality to provide a simplified, unified interface
    for chatting with models like Claude, GPT, and Gemini. It supports both synchronous
    'one-shot' responses and streaming output, with optional conversational memory.
    """

    def __init__(
        self,
        api_key: str = os.getenv("API_KEY", ""),
        model: Union[str, CloudModel] = CloudModel.ANTHROPIC_CLAUDE,
        system_prompt: str = "",
        temperature: Optional[float] = 0.7,
        timeout: int = 30,
    ):
        """Initializes the CloudLLM brick with the specified provider and configuration.

        Args:
            api_key (str): The API access key for the target LLM service. Defaults to the
                'API_KEY' environment variable.
            model (Union[str, CloudModel]): The model identifier. Accepts a `CloudModel`
                enum member (e.g., `CloudModel.OPENAI_GPT`) or its corresponding raw string
                value (e.g., `'gpt-4o-mini'`). Defaults to `CloudModel.ANTHROPIC_CLAUDE`.
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. Defaults to 0.7.
            timeout (int): The maximum duration in seconds to wait for a response before
                timing out. Defaults to 30.

        Raises:
            ValueError: If `api_key` is not provided (empty string).
        """
        if api_key == "":
            raise ValueError("API key is required to initialize CloudLLM brick.")

        self._api_key = api_key

        # Model configuration
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout = timeout

        # LangChain components
        self._prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])
        self._model = model_factory(
            model,
            api_key=self._api_key,
            temperature=self._temperature,
            timeout=self._timeout,
        )
        self._parser = StrOutputParser()
        self._history_cfg = {"configurable": {"session_id": uuid7()}}

        core_chain = self._prompt | self._model | self._parser
        self._chain = RunnableWithMessageHistory(
            core_chain,
            lambda session_id: self._get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

        # Memory management
        self._max_messages = DEFAULT_MEMORY
        self._history = None

        self._keep_streaming = threading.Event()

    def with_memory(self, max_messages: int = DEFAULT_MEMORY) -> "CloudLLM":
        """Enables conversational memory for this instance.

        Configures the Brick to retain a window of previous messages, allowing the
        AI to maintain context across multiple interactions.

        Args:
            max_messages (int): The maximum number of messages (user + AI) to keep
                in history. Older messages are discarded. Set to 0 to disable memory.
                Defaults to 10.

        Returns:
            CloudLLM: The current instance, allowing for method chaining.
        """
        self._max_messages = max_messages

        return self

    def chat(self, message: str) -> str:
        """Sends a message to the AI and blocks until the complete response is received.

        This method automatically manages conversation history if memory is enabled.

        Args:
            message (str): The input text prompt from the user.

        Returns:
            str: The complete text response generated by the AI.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
        """
        if self._chain is None:
            raise RuntimeError("CloudLLM brick is not started. Please call start() before generating text.")

        try:
            return self._chain.invoke({"input": message}, config=self._history_cfg)
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {e}")

    def chat_stream(self, message: str) -> Iterator[str]:
        """Sends a message to the AI and yields response tokens as they are generated.

        This allows for processing or displaying the response in real-time (streaming).
        The generation can be interrupted by calling `stop_stream()`.

        Args:
            message (str): The input text prompt from the user.

        Yields:
            str: Chunks of text (tokens) from the AI response.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            AlreadyGenerating: If a streaming session is already active.
        """
        if self._chain is None:
            raise RuntimeError("CloudLLM brick is not started. Please call start() before generating text.")
        if self._keep_streaming.is_set():
            raise AlreadyGenerating("A streaming response is already in progress. Please stop it before starting a new one.")

        try:
            self._keep_streaming.set()
            for token in self._chain.stream({"input": message}, config=self._history_cfg):
                if not self._keep_streaming.is_set():
                    break  # This stops the iteration and halts further token generation
                yield token
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {e}")
        finally:
            self._keep_streaming.clear()

    def stop_stream(self) -> None:
        """Signals the active streaming generation to stop.

        This sets an internal flag that causes the `chat_stream` iterator to break
        early. It has no effect if no stream is currently running.
        """
        self._keep_streaming.clear()

    def clear_memory(self) -> None:
        """Clears the conversational memory history.

        Resets the stored context. This is useful for starting a new conversation
        topic without previous context interfering. Only applies if memory is enabled.
        """
        if self._history:
            self._history.clear()

    def _get_session_history(self, session_id: str) -> WindowedChatMessageHistory:
        """Retrieves or creates the chat history for a given session.

        Internal callback used by LangChain's `RunnableWithMessageHistory`.

        Args:
            session_id (str): The unique identifier for the session.

        Returns:
            WindowedChatMessageHistory: The history object managing the message window.
        """
        if self._max_messages == 0:
            self._history = InMemoryChatMessageHistory()
        if self._history is None:
            self._history = WindowedChatMessageHistory(k=self._max_messages)
        return self._history


def model_factory(model_name: CloudModel, **kwargs) -> BaseChatModel:
    """Factory function to instantiate the specific LangChain chat model.

    This function maps the supported `CloudModel` enum values to their respective
    LangChain implementations.

    Args:
        model_name (CloudModel): The enum or string identifier for the model.
        **kwargs: Additional arguments passed to the model constructor (e.g., api_key, temperature).

    Returns:
        BaseChatModel: An instance of a LangChain chat model wrapper.

    Raises:
        ValueError: If `model_name` does not match one of the supported `CloudModel` options.
    """
    if model_name == CloudModel.ANTHROPIC_CLAUDE:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, **kwargs)
    elif model_name == CloudModel.OPENAI_GPT:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, **kwargs)
    elif model_name == CloudModel.GOOGLE_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, **kwargs)
    else:
        raise ValueError(f"Model not supported: {model_name}")
