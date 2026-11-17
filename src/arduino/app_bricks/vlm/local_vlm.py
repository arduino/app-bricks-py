# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from arduino.app_bricks.cloud_llm import ReasoningEffort
from arduino.app_bricks.cloud_llm.cloud_llm import ToolLike
from arduino.app_bricks.cloud_llm.memory import MessagePersistence
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import Logger, brick
from arduino.app_internal.core import get_brick_config, get_brick_configured_model

import base64
import io
import openai
from functools import lru_cache
from typing import Any, Self
from collections.abc import Iterator, Sequence

from PIL import Image

logger = Logger("VisionLanguageModel")

# Size of the synthetic image sent by ``init()``. Small enough to keep the warm-up call
# cheap, large enough to stay above the minimum resolution vision encoders accept.
_CANARY_IMAGE_SIZE = (224, 224)
_CANARY_PROMPT = "Is this image completely black? Answer YES or NO."


@lru_cache(maxsize=1)
def _canary_image_jpeg() -> bytes:
    """Returns a solid black JPEG built in memory (never written to disk).

    A VLM invoked without an image can misbehave or fail outright, so the warm-up call
    performed by ``init()`` always ships a real, if trivial, picture.
    """
    buffer = io.BytesIO()
    Image.new("RGB", _CANARY_IMAGE_SIZE, color="black").save(buffer, format="JPEG")
    return buffer.getvalue()


@brick
class VisionLanguageModel(LargeLanguageModel):
    """A Brick for interacting with locally-based Vision Language Models (VLMs).

    It provides a simplified, unified interface for chatting with models like Qwenm, LLama,
    Gemma. It supports both synchronous 'one-shot' responses and streaming output,
    with optional conversational memory.
    """

    def __init__(
        self,
        system_prompt: str = "",
        temperature: float | None = 0.0,
        max_tokens: int = 512,
        timeout: int | None = None,
        tools: Sequence[ToolLike] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the VisionLanguageModel brick with the specified provider and configuration.

        Args:
            model (str | None): The specific model name or identifier to use, served by the genie runner
                (e.g., "genie:qwen3_vl_4b_instruct") or by the llama.cpp runner (e.g.,
                "llamacpp:gemma-4-E2B_q4_0-it-multimodal"). If not provided, model will be determined
                from app configuration or default brick configuration.
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. Defaults to 0.0 (greedy decoding): typical VLM tasks are
                perception tasks (object detection, counting, OCR, constrained YES/NO
                answers) where reproducibility matters and sampling noise flips answers
                between runs, especially on small models. Pass a higher value (e.g. 0.7)
                for creative image descriptions.
            max_tokens (int): The maximum number of tokens to generate in the response.
                Defaults to 256.
            timeout (Optional[int]): The maximum duration in seconds to wait for a response before
                timing out. Defaults to None.
            tools (Sequence[ToolLike] | None): BaseTool objects (from @tool or MCPClient.get_tools()) or plain
                callables (auto-wrapped into tools). Defaults to None.
            **kwargs: Additional arguments passed to the model constructor
        """

        if model is None:
            brick_config = get_brick_config(self.__class__)
            # app.yaml override first, then the brick default (model_by_boards or model)
            model = get_brick_configured_model(brick_config.get("id") if brick_config else None, brick_config=brick_config)
            logger.debug(f"Using model: '{model}'.")

        super().__init__(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
            **kwargs,
        )
        super().with_memory(0)  # Initialize without memory enabled (0 means no history)

    def get_client(self) -> BaseChatModel | Runnable[LanguageModelInput, AIMessage]:
        """Returns the underlying LangChain model instance.

        This allows for advanced users to access the full capabilities of the model
        directly, such as calling `generate()` or `stream()` with custom message formats.

        Returns:
            BaseChatModel | Runnable[LanguageModelInput, AIMessage]: The LangChain chat model
                used internally, or the `Runnable` binding it to the registered tools.
        """
        return self._model

    def init(self) -> None:
        """Initializes the internal chain for the VLM.

        This method can be called before any chat or streaming operations.
        Pre load the model to ensure it's ready for use. Unlike a text-only LLM, a VLM is
        warmed up with a multimodal request: a solid black image generated in memory plus a
        constrained YES/NO question, so the vision encoder is exercised as well. The call is
        made on the base model and does not touch the conversation memory.
        If the model is not responsive or misconfigured, this method will raise a RuntimeError.

        Raises:
            RuntimeError: If initialization fails due to misconfiguration or API errors.
        """
        try:
            # Canary call to force the model load and ensure the runner is responsive.
            image_b64 = base64.b64encode(_canary_image_jpeg()).decode()
            # Image first, then text: same ordering used by chat() for multimodal messages.
            content: list[str | dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": _CANARY_PROMPT},
            ]
            self._base_model.invoke([HumanMessage(content=content)], max_tokens=1)
        except (openai.BadRequestError, openai.APIError) as e:
            self._handle_api_error(logger, e)

    def chat(
        self,
        message: str,
        images: Sequence[str | bytes] | None = None,
        reasoning_effort: ReasoningEffort | str | int | None = None,
    ) -> str:
        """Sends a message to the AI and blocks until the complete response is received.

        This method automatically manages conversation history if memory is enabled.

        Args:
            message (str): The input text prompt from the user.
            images (Sequence[str | bytes] | None): Optional sequence of image file paths or raw bytes to include in the prompt.
                None (default) sends a text-only message.
            reasoning_effort (ReasoningEffort | str | int | None): Optional effort level or token budget,
                as in `CloudLLM.chat`. None (default) keeps the default behavior; support depends on the local runner.

        Returns:
            str: The complete text response generated by the AI.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            ValueError: If `reasoning_effort` is not a supported level or budget.
            TypeError: If `reasoning_effort` is not a ReasoningEffort, str, int, or None.
        """
        try:
            return super()._chat_invoke(message=message, images=images, reasoning_effort=reasoning_effort)
        except (openai.BadRequestError, openai.APIError) as e:
            self._handle_api_error(logger, e)

    def chat_stream(self, message: str, images: Sequence[str | bytes] | None = None) -> Iterator[str]:
        """Sends a message to the AI and yields response tokens as they are generated.

        This allows for processing or displaying the response in real-time (streaming).
        The generation can be interrupted by calling `stop_stream()`.

        Args:
            message (str): The input text prompt from the user.
            images (Sequence[str | bytes] | None): Optional sequence of image file paths or raw bytes to include in the prompt.
                None (default) sends a text-only message.

        Yields:
            str: Chunks of text (tokens) from the AI response.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            AlreadyGenerating: If a streaming session is already active.
        """
        try:
            return super()._chat_stream_invoke(message=message, images=images)
        except (openai.BadRequestError, openai.APIError) as e:
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

    def with_memory(
        self,
        max_messages: int = 0,
        persistence: bool | MessagePersistence | None = None,
    ) -> Self:
        """Enables conversational memory for this instance.

        Configures the Brick to retain a window of previous messages, allowing the
        AI to maintain context across multiple interactions. An optional persistence
        backend stores the history so it can resume across restarts.

        Args:
            max_messages (int): The maximum number of messages.
            persistence (bool | MessagePersistence | None): Optional persistence backend.
                `None` or `False` keep history in memory only (default behavior).
                `True` instantiates the default SQL-backed store. Pass a
                `MessagePersistence` implementation directly for full control.

        Returns:
            Self: The current instance, allowing for method chaining.
        """
        return super().with_memory(max_messages=max_messages, persistence=persistence)
