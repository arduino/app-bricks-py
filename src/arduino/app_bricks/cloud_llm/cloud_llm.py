# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
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
    """A simplified, opinionated wrapper for common LangChain conversational patterns.

    This class provides a single interface to manage stateless chat and chat with memory.
    """

    def __init__(
        self,
        api_key: str = os.getenv("API_KEY", ""),
        model: Union[str, CloudModel] = CloudModel.ANTHROPIC_CLAUDE,
        system_prompt: str = "",
        temperature: Optional[float] = 0.7,
        timeout: int = 30,
    ):
        """Initializes the CloudLLM brick with the given configuration.

        Args:
            api_key: The API key for the LLM service.
            model: The model identifier as per LangChain specification (e.g., "anthropic:claude-3-sonnet-20240229")
                   or by using a CloudModels enum (e.g. CloudModels.OPENAI_GPT). Defaults to CloudModel.ANTHROPIC_CLAUDE.
            system_prompt: The global system-level instruction for the AI.
            temperature: The sampling temperature for response generation. Defaults to 0.7.
            timeout: The maximum time to wait for a response from the LLM service, in seconds. Defaults to 30 seconds.

        Raises:
            ValueError: If the API key is missing.
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
        self._history_cfg = {"configurable": {"session_id": "default_session"}}

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

        This allows the chatbot to remember previous user and AI messages.
        Calling this modifies the instance to be stateful.

        Args:
            max_messages: The total number of past messages (user + AI) to
                          keep in the conversation window. Set to 0 to disable memory.

        Returns:
            self: The current CloudLLM instance for method chaining.
        """
        self._max_messages = max_messages

        return self

    def chat(self, message: str) -> str:
        """Sends a single message to the AI and gets a complete response synchronously.

        This is the primary way to interact. It automatically handles memory
        based on how the instance was configured.

        Args:
            message: The user's message.

        Returns:
            The AI's complete response as a string.

        Raises:
            RuntimeError: If the chat model is not initialized or if text generation fails.
        """
        if self._chain is None:
            raise RuntimeError("CloudLLM brick is not started. Please call start() before generating text.")

        try:
            return self._chain.invoke({"input": message}, config=self._history_cfg)
        except Exception as e:
            raise RuntimeError(f"Response generation failed: {e}")

    def chat_stream(self, message: str) -> Iterator[str]:
        """Sends a single message to the AI and streams the response as a synchronous generator.

        Use this to get tokens as they are generated, perfect for a streaming UI.

        Args:
            message: The user's message.

        Yields:
            str: Chunks of the AI's response as they become available.

        Raises:
            RuntimeError: If the chat model is not initialized or if text generation fails.
            AlreadyGenerating: If the chat model is already streaming a response.
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
        """Signals the LLM to stop generating a response."""
        self._keep_streaming.clear()

    def clear_memory(self) -> None:
        """Clears the conversational memory.

        This only has an effect if with_memory() has been called.
        """
        if self._history:
            self._history.clear()

    def _get_session_history(self, session_id: str) -> WindowedChatMessageHistory:
        if self._max_messages == 0:
            self._history = InMemoryChatMessageHistory()
        if self._history is None:
            self._history = WindowedChatMessageHistory(k=self._max_messages)
        return self._history


def model_factory(model_name: CloudModel, **kwargs) -> BaseChatModel:
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
