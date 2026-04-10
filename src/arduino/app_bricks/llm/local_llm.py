# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.language_models import BaseChatModel

from arduino.app_bricks.cloud_llm import CloudLLM, CloudModelProvider
from arduino.app_bricks.cloud_llm.cloud_llm import DEFAULT_MEMORY
from arduino.app_utils import Logger, brick
from arduino.app_internal.core import resolve_address, get_brick_config, get_brick_configured_model

import os
from openai import OpenAI, APIError, BadRequestError
from typing import Iterator, List, Optional, Any, Callable

logger = Logger("LargeLanguageModel")


@brick
class LargeLanguageModel(CloudLLM):
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
        system_prompt: str = "",
        temperature: Optional[float] = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
        tools: List[Callable[..., Any]] = None,
        model: str = None,
        **kwargs,
    ):
        """Initializes the LargeLanguageModel brick with the specified provider and configuration.

        Args:
            api_key (str): The API access key for the target LLM service. Defaults to the
                'LOCAL_LLM_API_KEY' environment variable.
            model (str): The specific model name or identifier to use (e.g., "genie:qwen3-4b").
                If not provided, model will be determined from app configuration or default brick configuration.
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. Defaults to 0.7.
            max_tokens (int): The maximum number of tokens to generate in the response.
                Defaults to 512.
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

        if model is None:
            brick_config = get_brick_config(self.__class__)
            app_configured_model = get_brick_configured_model(brick_config.get("id") if brick_config else None)
            if app_configured_model:
                logger.debug(f"Using model: '{app_configured_model}'.")
                model = app_configured_model
            else:
                model = brick_config.get("model", None)
                logger.debug(f"Using default model: '{model}'.")
        else:
            logger.debug(f"Forcing use of model: '{model}'.")

        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")

            if base_url is None or base_url.strip() == "":
                raise ValueError("Empty or wrongly configured 'base_url'")

        else:
            if model.startswith(self.GENIE_MODEL):
                port = 9001
                host = "genie-models-runner"
            # elif model.startswith(self.LLAMACPP_MODEL):
            #     port = 9999
            #     host = "llamacpp-models-runner"
            # elif model.startswith(self.OLLAMA_MODEL):
            #     port = 11434
            #     host = "ollama-models-runner"
            else:
                raise ValueError(f"Unsupported local model type: {model}")

            base_url = f"http://{host}:{port}/v1"

        if model.startswith(self.GENIE_MODEL) or model.startswith(self.LLAMACPP_MODEL) or model.startswith(self.OLLAMA_MODEL):
            model = model.split(":")[-1]  # Extract model name without provider prefix

        logger.info(f"Initializing brick with model '{model}' at {base_url}")

        # Force OpenAI provider for local LLMs to force ChatCompletion APIs
        plain_model_name = model
        model = f"{CloudModelProvider.OPENAI}:{model}"

        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout=timeout,
            tools=tools,
            base_url=base_url,
            max_tokens=max_tokens,
            **kwargs,
        )

        available_models = self.list_models()
        if plain_model_name not in available_models:
            logger.error(
                f"Model '{plain_model_name}' not found among locally available models: {available_models}."
                + " Please download the model or configure it correctly."
            )

    def list_models(self) -> List[str]:
        """Returns a list of supported local model identifiers.

        Note: LargeLanguageModel supports OpenAI-compatible API. This method uses the OpenAI client to query available models from the local server.
        LangChain's OpenAI wrapper does not provide a direct method to list models, so we need to use the underlying OpenAI client directly.

        Returns:
            List[str]: A list of supported model names (e.g., ["qwen2.5-7b"]).
        """
        try:
            with OpenAI(base_url=self._model.openai_api_base, api_key=self._model.openai_api_key) as openai_client:
                models_response = openai_client.models.list()
                model_list = [model.id for model in models_response.data]

                return model_list
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def with_memory(self, max_messages: int = DEFAULT_MEMORY) -> "LargeLanguageModel":
        """Enables conversational memory for this instance.

        Configures the Brick to retain a window of previous messages, allowing the
        AI to maintain context across multiple interactions.

        Args:
            max_messages (int): The maximum number of messages (user + AI) to keep
                in history. Older messages are discarded. Set to 0 to disable memory.
                Defaults to 10.

        Returns:
            LargeLanguageModel: The current instance, allowing for method chaining.
        """
        return super().with_memory(max_messages=max_messages)

    def get_client(self) -> BaseChatModel:
        """Returns the underlying LangChain model instance.

        This allows for advanced users to access the full capabilities of the model
        directly, such as calling `generate()` or `stream()` with custom message formats.

        Returns:
            BaseChatModel: The LangChain chat model instance used internally.
        """
        return self._model

    def _handle_api_error(self, ilogger: Logger, e: Exception) -> None:
        """Handles OpenAI API errors by logging details and raising RuntimeError.

        Args:
            ilogger (Logger): The logger instance to use for logging errors.
            e: The exception to handle (BadRequestError or APIError)

        Raises:
            RuntimeError: Always raises with detailed error message and chained original exception
        """
        if isinstance(e, BadRequestError):
            error_msg = f"Bad request: {e.message if hasattr(e, 'message') else str(e)}"
            ilogger.error(error_msg)
            if hasattr(e, "response") and hasattr(e.response, "json"):
                try:
                    error_detail = e.response.json()
                    ilogger.error(f"Error details: {error_detail}")
                except Exception:
                    pass
            raise RuntimeError(error_msg) from e
        elif isinstance(e, APIError):
            if e.code == 503:
                error_msg = f"Cannot load model due to a potential memory exhaustion. message={e.message if hasattr(e, 'message') else str(e)}"
            else:
                error_msg = f"Error: status_code={e.code}, message={e.message if hasattr(e, 'message') else str(e)}"
            ilogger.error(error_msg)
            raise RuntimeError(error_msg) from e
        else:
            raise

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
        try:
            message = super()._chat_invoke(message=message, images=images)
            if "<think>" in message and "</think>" in message:
                splitted_message = message.split("<think>")[1].split("</think>")
                if len(splitted_message) > 1:
                    return splitted_message[1]  # Extract actual content
                else:
                    return message  # Fallback to full message if tags are not properly closed
            return message

        except (BadRequestError, APIError) as e:
            self._handle_api_error(logger, e)

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
        try:
            in_thininkg = False
            for chunk in super()._chat_stream_invoke(message=message, images=images):
                if in_thininkg:
                    if "</think>" in chunk:
                        in_thininkg = False
                        chunk = chunk.split("</think>")[-1]  # Take content after </think>
                        if chunk is not None and chunk.strip() != "":
                            yield chunk
                    continue

                if "<think>" in chunk:
                    in_thininkg = True
                    continue  # Skip the <think> tag itself
                else:
                    yield chunk
        except (BadRequestError, APIError) as e:
            self._handle_api_error(logger, e)

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
