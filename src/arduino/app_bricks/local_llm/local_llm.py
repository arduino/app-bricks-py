# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.cloud_llm import CloudLLM, CloudModelProvider
from arduino.app_bricks.cloud_llm.cloud_llm import DEFAULT_MEMORY
from arduino.app_utils import Logger, brick
from arduino.app_internal.core import resolve_address

import os
from typing import Iterator, List, Optional, Any, Callable

logger = Logger("LocalLLM")


@brick
class LocalLLM(CloudLLM):
    """A Brick for interacting with locally-based Large Language Models (LLMs).

    This class wraps LangChain functionality to provide a simplified, unified interface
    for chatting with models like Qwenm, LLama, Gemma. It supports both synchronous
    'one-shot' responses and streaming output, with optional conversational memory.
    """

    GENIE_MODEL = "genie"
    LLAMACPP_MODEL = "llamacpp"
    OLLAMA_MODEL = "ollama"

    def __init__(
        self,
        api_key: str = os.getenv("LOCAL_LLM_API_KEY", "api_key"),
        model: str = "genie:qwen2.5-7b",
        system_prompt: str = "",
        temperature: Optional[float] = 0.7,
        timeout: int = 30,
        tools: List[Callable[..., Any]] = None,
        **kwargs,
    ):
        """Initializes the CloudLLM brick with the specified provider and configuration.

        Args:
            api_key (str): The API access key for the target LLM service. Defaults to the
                'LOCAL_LLM_API_KEY' environment variable.
            model (str): The specific model name or identifier to use (e.g., "genie:qwen2.5-7b").
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. Defaults to 0.7.
            timeout (int): The maximum duration in seconds to wait for a response before
                timing out. Defaults to 30.
            tools (List[Callable[..., Any]]): A list of callable tool functions to register. Defaults to None.
            **kwargs: Additional arguments passed to the model constructor

        Raises:
            ValueError: If `api_key` is not provided (empty string).
        """

        host = "localhost"
        port = 0

        host = resolve_address(host)
        if not host:
            raise RuntimeError("Host address resolution failed for local LLM runner.")

        if "base_url" in kwargs:
            logger.warning("Overriding provided 'base_url' argument with resolved local address.")
            base_url = kwargs.pop("base_url")

            if base_url is None or base_url.strip() == "":
                raise ValueError("Empty or wrongly configured 'base_url")

        else:
            if model.startswith(self.GENIE_MODEL):
                port = 9001
                host = "genie-models-runner"
            elif model.startswith(self.LLAMACPP_MODEL):
                port = 9999
                host = "llamacpp-models-runner"
            elif model.startswith(self.OLLAMA_MODEL):
                port = 11434
                host = "ollama-models-runner"
            else:
                raise ValueError(f"Unsupported local model type: {model}")

            model = model.split(":", 1)[-1].strip()  # Remove prefix if any
            base_url = f"http://{host}:{port}/v1"

        logger.info(f"Initializing LocalLLM with model '{model}' at {base_url}")

        # Force OpenAI provider for local LLMs to force ChatCompletion APIs
        model = f"{CloudModelProvider.OPENAI}:{model}"

        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout=timeout,
            tools=tools,
            base_url=base_url,
            **kwargs,
        )

        self._list_supported_models()

    def _list_supported_models(self) -> List[str]:
        """Returns a list of supported local model identifiers.

        Note: LocalLLM supports OpenAI-compatible API. This methos uses the OpenAI client to query available models from the local server.
        LangChain's OpenAI wrapper does not provide a direct method to list models, so we need to use the underlying OpenAI client directly.

        Returns:
            List[str]: A list of supported model names (e.g., ["qwen2.5-7b", "vicuna-13b"]).
        """
        try:
            from openai import OpenAI

            with OpenAI(base_url=self._model.openai_api_base, api_key=self._model.openai_api_key) as openai_client:
                models_response = openai_client.models.list()
                model_list = [model.id for model in models_response.data]

                return model_list
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

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
        return super().with_memory(max_messages=max_messages)

    def chat(self, message: str, images: List[str | bytes] = None) -> str:
        """Sends a message to the AI and blocks until the complete response is received.

        This method automatically manages conversation history if memory is enabled.

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.

        Returns:
            str: The complete text response generated by the AI.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
        """
        return super().chat(message=message, images=images)

    def chat_stream(self, message: str, images: List[str | bytes] = None) -> Iterator[str]:
        """Sends a message to the AI and yields response tokens as they are generated.

        This allows for processing or displaying the response in real-time (streaming).
        The generation can be interrupted by calling `stop_stream()`.

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.

        Yields:
            str: Chunks of text (tokens) from the AI response.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            AlreadyGenerating: If a streaming session is already active.
        """
        return super().chat_stream(message=message, images=images)

    def stop_stream(self) -> None:
        """Signals the active streaming generation to stop.

        This sets an internal flag that causes the `chat_stream` iterator to break
        early. It has no effect if no stream is currently running.
        """
        super().stop_stream()

    def clear_memory(self) -> None:
        """Clears the conversational memory history.

        Resets the stored context. This is useful for starting a new conversation
        topic without previous context interfering. Only applies if memory is enabled.
        """
        super().clear_memory()
