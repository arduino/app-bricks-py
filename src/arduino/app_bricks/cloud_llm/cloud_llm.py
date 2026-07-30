# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import os
import re
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union, Any, Sequence, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, ToolCall
from langchain_core.tools import BaseTool, StructuredTool

from arduino.app_utils import brick

from .utils import logger
from .models import (
    CloudModel,
    CloudModelProvider,
    ReasoningEffort,
    EFFORT_TO_BUDGET,
    ANTHROPIC_MIN_THINKING_BUDGET,
    ANTHROPIC_DEFAULT_THINKING_BUDGET,
    ANTHROPIC_EFFORT_MAP,
)
from .memory import MessagePersistence, SQLMessagePersistence, WindowedChatMessageHistory

DEFAULT_MEMORY = 10

ToolLike = Union[BaseTool, Callable[..., Any]]


class AlreadyGenerating(Exception):
    """Exception raised when a generation is already in progress."""

    pass


@dataclass(frozen=True)
class ReasoningStreamChunk:
    """Base type for chunks yielded by `CloudLLM.chat_stream_reasoning`.

    Every chunk carries a `content` text fragment. Use `isinstance` checks to
    distinguish the model's reasoning from its final answer:

    ```python
    for chunk in llm.chat_stream_reasoning("..."):
        if isinstance(chunk, ReasoningChunk):
            ...  # chunk.content is part of the chain-of-thought
        elif isinstance(chunk, ContentChunk):
            ...  # chunk.content is part of the final answer
    ```
    """

    content: str


@dataclass(frozen=True)
class ReasoningChunk(ReasoningStreamChunk):
    """A fragment of the model's internal reasoning (chain-of-thought)."""


@dataclass(frozen=True)
class ContentChunk(ReasoningStreamChunk):
    """A fragment of the model's final answer."""


@brick
class CloudLLM:
    """A Brick for interacting with cloud-based Large Language Models (LLMs).

    This class wraps LangChain functionality to provide a simplified, unified interface
    for chatting with models like Claude, GPT, and Gemini. It supports both synchronous
    'one-shot' responses and streaming output, with optional conversational memory.
    """

    _logger = logger

    def __init__(
        self,
        api_key: str = os.getenv("API_KEY", ""),
        model: Union[str, CloudModel] = CloudModel.ANTHROPIC_CLAUDE,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        reasoning_effort: Union["ReasoningEffort", str, int, None] = None,
        max_tool_loops: int = 8,
        timeout: Optional[int] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        callbacks: Any = None,
        **kwargs,
    ):
        """Initializes the CloudLLM brick with the specified provider and configuration.

        Args:
            api_key (str): The API access key for the target LLM service. Defaults to the
                'API_KEY' environment variable.
            model (Union[str, CloudModel]): The model identifier. Accepts a `CloudModel`
                enum member (e.g., `CloudModel.OPENAI_GPT`) or its corresponding raw string
                value (e.g., `'openai:gpt-5-mini'`). Defaults to `CloudModel.ANTHROPIC_CLAUDE`.
                To identify the model provider, you need to use prefixes like 'openai:', 'anthropic:', or 'google:'.
                If no prefix is provided, the model will be defaulted to an OpenAI compatible model.
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. When ``None`` (default) no temperature is sent and each
                provider's own default is used; this also avoids errors on models that
                deprecated ``temperature`` (e.g. Anthropic Claude Sonnet 5+).
            reasoning_effort (ReasoningEffort | str | int | None): Optional default reasoning
                effort applied to every ``chat``/``chat_stream_reasoning`` call that does not
                pass its own. ``chat_stream`` is never affected: it always streams from the
                plain model (chat completions on OpenAI) and yields answer text only. When
                ``None`` (default) nothing is added and the plain model is used. When set, the
                reasoning-capable client (Responses API for OpenAI) is used so it works with
                tools. It is never forwarded as a raw model argument (which would break tool
                calling on OpenAI chat completions). Accepts a discrete level (`ReasoningEffort`
                / 'minimal'/'low'/'medium'/'high') or an integer token budget.
            max_tool_loops (int): The maximum number of consecutive tool-call loops
                allowed during a single chat interaction. Defaults to 8.
            timeout (Optional[int]): The maximum duration in seconds to wait for a response before
                timing out. Defaults to None.
            callbacks (Any): Optional callbacks for monitoring generation events.
            tools (Sequence[ToolLike]): BaseTool objects (from @tool or MCPClient.get_tools()) or plain
                callables (auto-wrapped into tools). Defaults to None.
            **kwargs: Additional arguments passed to the model constructor

        Raises:
            ValueError: If `api_key` is not provided (empty string).
        """
        if api_key == "" and (
            model.startswith(f"{CloudModelProvider.OPENAI}:")
            or model.startswith(f"{CloudModelProvider.ANTHROPIC}:")
            or model.startswith(f"{CloudModelProvider.GOOGLE}:")
        ):
            raise ValueError("API key is required to initialize CloudLLM brick.")

        self._api_key = api_key

        # Model configuration
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._validate_reasoning_effort(reasoning_effort)
        self._reasoning_effort_default = reasoning_effort
        self._max_tool_loops = max_tool_loops
        self._timeout = timeout
        self._callbacks = callbacks
        self._model_loaded = False
        self._model_name = model

        # Registered tools
        self._tools_map = {}
        if tools is None:
            self._tools = []
        else:
            self._tools = [t if isinstance(t, BaseTool) else StructuredTool.from_function(t) for t in tools]
            for tool_func in self._tools:
                self._tools_map[tool_func.name] = tool_func

        # Only forward ``temperature`` when explicitly set: passing ``None`` lets each
        # provider use its own default and, crucially, avoids sending the field to models
        # that deprecated it (Anthropic Sonnet 5+) or reject ``None`` for it (Gemini, whose
        # ``temperature`` is a strict float).
        # ``reasoning_effort`` is never forwarded to the base model: on OpenAI it would be
        # sent as a raw chat-completions field and break tool calling. It is applied only
        # through the reasoning flow (Responses API) when set.
        model_kwargs = dict(kwargs)
        model_kwargs.pop("reasoning_effort", None)
        if self._temperature is not None:
            model_kwargs["temperature"] = self._temperature

        self._model = model_factory(
            model,
            api_key=self._api_key,
            timeout=self._timeout,
            **model_kwargs,
        )

        # Keep a reference to the unbound model so a reasoning-capable client can
        # be derived lazily (see `_get_reasoning_model`).
        self._base_model = self._model
        self._reasoning_model = None
        self._reasoning_effort = None

        if self._tools and len(self._tools) > 0:
            logger.info(f"Binding {len(self._tools)} tool(s) to the model.")
            from .reasoning import ChatOpenAIReasoning

            # OpenAI reasoning models (e.g. gpt-5.x) reason by default and reject function
            # tools combined with reasoning in /v1/chat/completions. The Responses API
            # supports both, so switch the OpenAI client to it before binding tools.
            if isinstance(self._model, ChatOpenAIReasoning):
                self._model = self._model.model_copy(update={"use_responses_api": True, "output_version": "responses/v1"})
            self._model = self._model.bind_tools(tools=self._tools)

        # Memory management
        self.with_memory(DEFAULT_MEMORY)

        self._keep_streaming = threading.Event()

    def with_memory(
        self,
        max_messages: int = DEFAULT_MEMORY,
        persistence: Union[bool, MessagePersistence, None] = None,
    ) -> "CloudLLM":
        """Enables conversational memory for this instance.

        Configures the Brick to retain a window of previous messages, allowing the
        AI to maintain context across multiple interactions. An optional persistence
        backend stores the history so it can resume across restarts.

        Args:
            max_messages (int): The maximum number of messages.
            persistence (bool | MessagePersistence | None): Optional persistence backend.
                `None` or `False` keep history in memory only (default behavior).
                `True` instantiates a default `SQLMessagePersistence()`. Pass a
                `MessagePersistence` implementation directly (e.g.
                `SQLMessagePersistence(thread_id="user-42")`) for full control.

        Returns:
            CloudLLM: The current instance, allowing for method chaining.
        """
        if persistence is True:
            store: Optional[MessagePersistence] = SQLMessagePersistence()
        elif persistence in (False, None):
            store = None
        else:
            store = persistence

        self._max_messages = max_messages
        self._history = WindowedChatMessageHistory(
            k=max_messages,
            system_message=self._system_prompt,
            store=store,
        )

        return self

    def _get_message_with_history(self, user_input: str, images: List[str | bytes] = None) -> List[BaseMessage]:
        """Retrieves the current message history for the conversation, including the new user input.

        Args:
            user_input (str): The latest input message from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.

        Returns:
            List[BaseMessage]: The list of messages in the conversation history,
                including system prompt if set.
        """

        if self._model_loaded is False:
            logger.info(f"Initializing model {self._model_name}...")
            self._model_loaded = True

        messages = self._history.get_messages()
        message = None
        if images is not None and len(images) > 0:
            content = []
            content.append({"type": "text", "text": user_input})
            for img in images:
                image_b64 = self._image_to_base64(img)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

            message = HumanMessage(content=content)
        else:
            message = HumanMessage(content=user_input)

        if message is not None:
            messages.append(message)
            self._history.add_messages([message])

        return messages

    def _process_tool_calls(self, tool_calls: list[ToolCall], input_messages: List[BaseMessage]) -> List[BaseMessage]:
        """Processes any tool calls requested by the model in its response.

        Args:
            tool_calls (list[ToolCall]): The list of tool calls requested by the model.
            input_messages (List[BaseMessage]): The current message scope including history.

        Returns:
            List[BaseMessage]: Updated message scope after processing tool calls.
        """

        if len(tool_calls) == 0:
            return input_messages

        for tool_call in tool_calls:
            logger.debug(f"Calling tool: {tool_call['name']} with args: {tool_call['args']} with id: {tool_call['id']}")
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in self._tools_map:
                logger.debug(f"Invoking tool function for: {tool_name}")
                tool_func = self._tools_map[tool_name]
                tool_output = asyncio.run(
                    tool_func.ainvoke(
                        tool_args,
                        config={"callbacks": self._callbacks},
                    )
                )
                logger.debug(f"Tool '{tool_name}' returned: {tool_output}")

                # Append tool output message to current message scope
                input_messages.append(
                    ToolMessage(
                        tool_call_id=tool_id,
                        content=self._content_to_text(tool_output),
                    )
                )

        # Return updated message scope for further processing
        return input_messages

    def _image_to_base64(self, path: str | bytes) -> str:
        """Encodes an image file to a base64 string.
        Args:
            path (str | bytes): The file path to the image or raw bytes of the image
        Returns:
            str: The base64-encoded string of the image.
        Raises:
            FileNotFoundError: If the provided file path does not exist.
        """
        if isinstance(path, bytes):
            return base64.b64encode(path).decode()
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)

        return str(content)

    def get_client(self) -> BaseChatModel:
        """Returns the underlying LangChain model instance.

        This allows for advanced users to access the full capabilities of the model
        directly, such as calling `generate()` or `stream()` with custom message formats.

        Returns:
            BaseChatModel: The LangChain chat model instance used internally.
        """
        return self._model

    def chat(
        self,
        message: str,
        images: List[str | bytes] = None,
        reasoning_effort: Union["ReasoningEffort", str, int, None] = None,
    ) -> str:
        """Sends a message to the AI and blocks until the complete response is received.

        This method automatically manages conversation history if memory is enabled.

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.
            reasoning_effort (ReasoningEffort | str | int | None): Optional control over
                how much the model reasons before answering. When ``None`` (default) the
                behavior is unchanged and the base model is used. When provided, the
                reasoning-capable client is used so the effort is applied, but only the
                final answer text is returned (the chain-of-thought is not included). Pass
                a discrete level (`ReasoningEffort` or one of 'minimal'/'low'/'medium'/'high')
                or an explicit integer token budget (`-1` dynamic/unrestricted, `0` off,
                `N>0` token budget), mapped to the provider's native knob. Requires an
                OpenAI-compatible, Google Gemini, or Anthropic Claude reasoning model.

        Returns:
            str: The complete text response generated by the AI.

        Raises:
            RuntimeError: If the internal chain is not initialized, the model does not
                support reasoning, or the API request fails.
            ValueError: If `reasoning_effort` is not a supported level or budget.
            TypeError: If `reasoning_effort` is not a ReasoningEffort, str, int, or None.
        """
        if self._model is None:
            raise RuntimeError("Model has not been declared properly. Please check the model configuration.")

        try:
            return self._chat_invoke(message, images, reasoning_effort)
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise RuntimeError(f"Response generation failed: {e}")

    def _chat_invoke(
        self,
        message: str,
        images: List[str | bytes] = None,
        reasoning_effort: Union["ReasoningEffort", str, int, None] = None,
    ) -> str:
        """Internal method to perform the chat invocation with the model.

        This is separated from `chat()` to allow for better error handling and potential reuse
        in other contexts (e.g., within tool calls).

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.
            reasoning_effort (ReasoningEffort | str | int | None): Optional effort level or
                token budget. When ``None`` the base model is used unchanged; otherwise the
                reasoning-capable client is used and only the final answer text is returned.

        Returns:
            str: The complete text response generated by the AI.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            ValueError: If `reasoning_effort` is not a supported level or budget.
        """
        effort = reasoning_effort if reasoning_effort is not None else self._reasoning_effort_default
        model = self._model if effort is None else self._get_reasoning_model(effort)
        input_messages = self._get_message_with_history(message, images)
        loops = 0

        while True:
            message = model.invoke(input=input_messages, config={"callbacks": self._callbacks})
            if message is None:
                raise RuntimeError("Received empty response from the LLM.")

            logger.debug(f"Model invoked. Full response: {message}")

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                break

            loops += 1
            if loops > self._max_tool_loops:
                raise RuntimeError(f"Too many consecutive tool-call loops ({self._max_tool_loops}). Possible tool loop.")

            input_messages.append(message)
            input_messages = self._process_tool_calls(tool_calls, input_messages.copy())

        # Add the AI message to long term history
        self._history.add_messages([message])
        return self._content_to_text(message.content)

    def chat_stream(self, message: str, images: List[str | bytes] = None) -> Iterator[str]:
        """Sends a message to the AI and yields response tokens as they are generated.

        This allows for processing or displaying the response in real-time (streaming).
        The generation can be interrupted by calling `stop_stream()`.

        The stream always comes from the plain model (chat completions on OpenAI) and each
        yielded item is plain answer text: a `reasoning_effort` configured on the brick does
        not apply here. Use `chat_stream_reasoning` to stream the chain-of-thought.

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
            yield from self._chat_stream_invoke(message, images)

        except AlreadyGenerating:
            raise
        except Exception as e:
            self._handle_stream_error(e)

    def _handle_stream_error(self, e: Exception) -> None:
        """Handles stream errors and acts as an override hook for subclasses.

        Args:
            e (Exception): The exception that occurred during streaming.
        """
        self._logger.error(f"Response generation failed: {e}")
        raise RuntimeError(f"Response generation failed: {e}") from e

    def _chat_stream_invoke(self, message: str, images: List[str | bytes] = None) -> Iterator[str]:
        """Internal method to perform the chat streaming invocation with the model.

        This is separated from `chat_stream()` to allow for better error handling and potential reuse
        in other contexts (e.g., within tool calls).

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.

        Yields:
            str: Chunks of text (tokens) from the AI response.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            AlreadyGenerating: If a streaming session is already active.
        """
        if self._model is None:
            raise RuntimeError("Model has not been declared properly. Please check the model configuration.")
        if self._keep_streaming.is_set():
            raise AlreadyGenerating("A streaming response is already in progress. Please stop it before starting a new one.")
        assistant_chunks: list[str] = []

        try:
            self._keep_streaming.set()
            input_messages = self._get_message_with_history(message, images)

            tool_calls = []
            for token in self._model.stream(input_messages):
                if not self._keep_streaming.is_set():
                    break  # This stops the iteration and halts further token generation
                if token.tool_calls and len(token.tool_calls) > 0:
                    tool_calls.extend(token.tool_calls)
                else:
                    # Providers may stream content as a list of blocks (Gemini, and OpenAI
                    # via the Responses API) instead of a plain string
                    content = self._content_to_text(token.content)
                    if content:
                        assistant_chunks.append(content)
                        yield content

            # If there were tool calls, process them
            if len(tool_calls) > 0:
                input_messages = self._process_tool_calls(tool_calls, input_messages.copy())
                for token in self._model.stream(input=input_messages, config={"callbacks": self._callbacks}):
                    if not self._keep_streaming.is_set():
                        break
                    content = self._content_to_text(token.content)
                    if content:
                        assistant_chunks.append(content)
                        yield content

        finally:
            self._keep_streaming.clear()
            if len(assistant_chunks) > 0:
                full_response = "".join(assistant_chunks)
                self._history.add_messages([AIMessage(content=full_response)])

    def _get_reasoning_model(self, reasoning_effort: Union["ReasoningEffort", str, int, None] = None) -> BaseChatModel:
        """Returns a reasoning-capable client that streams reasoning tokens.

        The client is derived lazily from the base model depending on the provider:

        - OpenAI-compatible models enable the OpenAI Responses API, which exposes
          reasoning content while streaming.
        - Google Gemini models enable ``include_thoughts`` so the model streams its
          reasoning summaries as ``thinking`` content blocks.
        - Anthropic Claude models enable extended thinking (``thinking``) so the model
          streams its reasoning as ``thinking`` content blocks.

        The optional ``reasoning_effort`` controls how much the model reasons and is
        translated to each provider's native knob (see ``_reasoning_effort_update``).

        Args:
            reasoning_effort (ReasoningEffort | str | int | None): A discrete effort
                level or an explicit integer token budget. ``None`` uses the model
                default.

        Returns:
            BaseChatModel: A model configured to stream reasoning tokens.

        Raises:
            RuntimeError: If the underlying model does not support reasoning streaming.
            ValueError: If ``reasoning_effort`` is not a supported level or budget.
        """
        if self._reasoning_model is not None and self._reasoning_effort == reasoning_effort:
            return self._reasoning_model

        self._validate_reasoning_effort(reasoning_effort)

        from .reasoning import ChatOpenAIReasoning

        base_model = self._base_model
        if isinstance(base_model, ChatOpenAIReasoning):
            update = {"use_responses_api": True, "output_version": "responses/v1"}
            update.update(self._openai_effort_update(base_model, reasoning_effort))
            reasoning_model = base_model.model_copy(update=update)
        elif self._is_google_model(base_model):
            update = {"include_thoughts": True}
            update.update(self._gemini_effort_update(base_model, reasoning_effort))
            reasoning_model = base_model.model_copy(update=update)
        elif self._is_anthropic_model(base_model):
            reasoning_model = base_model.model_copy(update=self._anthropic_effort_update(base_model, reasoning_effort))
        else:
            raise RuntimeError("Reasoning streaming is only supported for OpenAI-compatible, Google Gemini, and Anthropic Claude models.")

        if self._tools and len(self._tools) > 0:
            reasoning_model = reasoning_model.bind_tools(tools=self._tools)

        self._reasoning_model = reasoning_model
        self._reasoning_effort = reasoning_effort
        return self._reasoning_model

    @staticmethod
    def _resolve_effort_level(reasoning_effort: Union["ReasoningEffort", str]) -> ReasoningEffort:
        """Validates and normalizes a discrete effort level.

        Args:
            reasoning_effort (ReasoningEffort | str): The effort level to validate.

        Returns:
            ReasoningEffort: The normalized effort level.

        Raises:
            ValueError: If the value is not a supported effort level.
        """
        try:
            return ReasoningEffort(reasoning_effort)
        except ValueError:
            allowed = ", ".join(e.value for e in ReasoningEffort)
            raise ValueError(f"Unsupported reasoning effort '{reasoning_effort}'. Expected one of: {allowed}, or an integer token budget.")

    @staticmethod
    def _validate_reasoning_effort(reasoning_effort: Union["ReasoningEffort", str, int, None]) -> None:
        """Guards the ``reasoning_effort`` argument to avoid level/budget confusion.

        Accepted forms:
        - ``None`` (use the model default),
        - a discrete level: ``ReasoningEffort`` or its string value ('minimal',
          'low', 'medium', 'high'),
        - an integer token budget.

        A ``bool`` and a numeric string (e.g. ``'64'``) are rejected explicitly
        because they are ambiguous with an integer budget.

        Args:
            reasoning_effort (ReasoningEffort | str | int | None): The value to validate.

        Raises:
            ValueError: If a bool is passed, a numeric string is passed, or a string
                is not a supported effort level.
            TypeError: If the value is not a ``ReasoningEffort``, ``str``, ``int``, or ``None``.
        """
        if reasoning_effort is None:
            return
        if isinstance(reasoning_effort, bool):
            raise ValueError("reasoning_effort must be an effort level (str) or an int token budget, not a bool.")
        if isinstance(reasoning_effort, int):
            return
        if isinstance(reasoning_effort, str):
            if reasoning_effort.strip().lstrip("-").isdigit():
                allowed = ", ".join(e.value for e in ReasoningEffort)
                raise ValueError(
                    f"reasoning_effort '{reasoning_effort}' is a numeric string. Pass an int "
                    f"(e.g. {int(reasoning_effort)}) for a token budget, or a level ({allowed})."
                )
            CloudLLM._resolve_effort_level(reasoning_effort)
            return
        raise TypeError(f"reasoning_effort must be ReasoningEffort, str, int, or None, got {type(reasoning_effort).__name__}.")

    def _openai_effort_update(self, model: BaseChatModel, reasoning_effort: Union["ReasoningEffort", str, int, None]) -> dict:
        """Builds the model-copy update applying reasoning effort for OpenAI models.

        Since reasoning streaming goes through the Responses API, effort and the
        reasoning summary are configured via the ``reasoning`` dict. ``summary``
        is always set to ``'auto'`` so OpenAI's proprietary reasoning models emit
        their ``response.reasoning_summary_text.delta`` events; without it these
        models reason silently and no reasoning is streamed. A discrete level maps
        to ``reasoning['effort']``; ``None`` keeps the model default effort while
        still requesting the summary.

        An integer instead maps to llama.cpp's ``thinking_budget_tokens``
        (``-1`` unrestricted, ``0`` off, ``N>0`` token budget), passed via
        ``extra_body`` since it is not a standard OpenAI field. Because llama.cpp only
        applies the budget when the model is actually thinking, ``enable_thinking`` is
        also set via ``chat_template_kwargs`` for templates that gate thinking behind it.
        llama.cpp/gpt-oss stream raw ``response.reasoning_text.delta`` events, so no
        summary is requested on that path.

        Args:
            model (BaseChatModel): The base OpenAI-compatible model.
            reasoning_effort (ReasoningEffort | str | int | None): Effort level or budget.

        Returns:
            dict: Fields to apply via ``model_copy``.
        """
        if isinstance(reasoning_effort, int) and not isinstance(reasoning_effort, bool):
            extra_body = dict(getattr(model, "extra_body", None) or {})
            extra_body["thinking_budget_tokens"] = reasoning_effort
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs.setdefault("enable_thinking", True)
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            return {"extra_body": extra_body}

        reasoning: dict = {"summary": "auto"}
        if reasoning_effort is not None:
            reasoning["effort"] = self._resolve_effort_level(reasoning_effort).value
        return {"reasoning": reasoning}

    def _gemini_effort_update(self, model: BaseChatModel, reasoning_effort: Union["ReasoningEffort", str, int, None]) -> dict:
        """Builds the model-copy update applying reasoning effort for Gemini models.

        An integer maps directly to ``thinking_budget`` (``-1`` dynamic, ``0`` off,
        ``N>0`` token budget). A discrete level maps to ``thinking_level`` on Gemini
        3+ models, or to a ``thinking_budget`` token count on Gemini 2.5 models,
        which do not support ``thinking_level``.

        Args:
            model (BaseChatModel): The base Gemini model.
            reasoning_effort (ReasoningEffort | str | int | None): Effort level or budget.

        Returns:
            dict: Fields to apply via ``model_copy``.
        """
        if reasoning_effort is None:
            return {}
        if isinstance(reasoning_effort, int) and not isinstance(reasoning_effort, bool):
            return {"thinking_budget": reasoning_effort}

        level = self._resolve_effort_level(reasoning_effort)

        from langchain_google_genai.chat_models import _is_gemini_3_or_later

        if _is_gemini_3_or_later(getattr(model, "model", "") or ""):
            # ``reasoning_effort`` is the field name (serialization alias ``thinking_level``);
            # ``model_copy(update=...)`` requires the field name, not the alias.
            return {"reasoning_effort": level.value}
        return {"thinking_budget": EFFORT_TO_BUDGET[level]}

    def _anthropic_effort_update(self, model: BaseChatModel, reasoning_effort: Union["ReasoningEffort", str, int, None]) -> dict:
        """Builds the model-copy update applying reasoning effort for Anthropic models.

        Anthropic exposes reasoning via extended thinking, but the API differs by model
        generation:

        - Legacy models (Sonnet 3.7 through 4.6, Opus 4.5/4.6) use
          ``thinking={"type": "enabled", "budget_tokens": N}``.
        - Newer models (Opus 4.7+ and Sonnet 5+) dropped ``budget_tokens`` and require
          ``thinking={"type": "adaptive"}`` with ``output_config.effort``
          (see ``_anthropic_requires_adaptive`` and ``_anthropic_adaptive_update``).

        The effort argument maps as follows (``0`` disables thinking, ``-1`` defers the
        budget to the model, ``N>0`` is a token budget, and a level is a discrete effort):

        - ``0`` -> thinking disabled (no reasoning).
        - ``-1`` -> adaptive thinking.
        - ``N>0`` -> legacy ``budget_tokens`` (clamped to the 1024 minimum); ignored on
          adaptive-only models (which use adaptive thinking without a discrete effort).
        - level -> legacy ``budget_tokens`` via ``EFFORT_TO_BUDGET`` (clamped), or
          ``output_config.effort`` via ``ANTHROPIC_EFFORT_MAP`` on adaptive-only models.
        - ``None`` -> a default budget (legacy) or plain adaptive thinking (newer).

        When thinking is enabled ``temperature`` is only sent when explicitly configured on
        the brick (``self._temperature``); otherwise it is left unset so the provider default
        applies (Anthropic defaults to ``1``, which is required while thinking is active). On
        legacy models ``max_tokens`` is raised above the budget when needed (``budget_tokens``
        must be ``< max_tokens``).

        Args:
            model (BaseChatModel): The base Anthropic model.
            reasoning_effort (ReasoningEffort | str | int | None): Effort level or budget.

        Returns:
            dict: Fields to apply via ``model_copy``.
        """
        model_name = getattr(model, "model", "") or ""
        is_budget = isinstance(reasoning_effort, int) and not isinstance(reasoning_effort, bool)

        # An explicit 0 disables thinking; a negative budget defers the amount to the model.
        if is_budget and reasoning_effort == 0:
            return {}
        if is_budget and reasoning_effort < 0:
            return self._anthropic_adaptive_update(model_name, level=None)

        # Newer models only accept adaptive thinking guided by ``output_config.effort``; an
        # explicit token budget is not supported, so only a discrete level (if any) is used.
        if self._anthropic_requires_adaptive(model_name):
            level = None if (is_budget or reasoning_effort is None) else self._resolve_effort_level(reasoning_effort)
            return self._anthropic_adaptive_update(model_name, level)

        # Legacy models use enabled thinking with an explicit token budget.
        if reasoning_effort is None:
            budget = ANTHROPIC_DEFAULT_THINKING_BUDGET
        elif is_budget:
            budget = max(reasoning_effort, ANTHROPIC_MIN_THINKING_BUDGET)
        else:
            budget = max(EFFORT_TO_BUDGET[self._resolve_effort_level(reasoning_effort)], ANTHROPIC_MIN_THINKING_BUDGET)

        update: dict = {"thinking": {"type": "enabled", "budget_tokens": budget}}
        if self._temperature is not None:
            update["temperature"] = self._temperature
        max_tokens = getattr(model, "max_tokens", None)
        if max_tokens is not None and max_tokens <= budget:
            update["max_tokens"] = budget + ANTHROPIC_MIN_THINKING_BUDGET
        return update

    def _anthropic_adaptive_update(self, model_name: str, level: Optional["ReasoningEffort"]) -> dict:
        """Builds the model-copy update for Anthropic adaptive thinking.

        Adaptive thinking (``{"type": "adaptive"}``) lets the model choose its own reasoning
        budget. On models that require it (Opus 4.7+/Sonnet 5+) the reasoning output is
        omitted by default, so ``display`` is set to ``"summarized"`` to surface the stream
        and a discrete ``level`` (if any) is forwarded as ``output_config.effort``. On legacy
        models reached via an explicit ``-1``, neither key applies.

        Args:
            model_name (str): The Anthropic model identifier.
            level (ReasoningEffort | None): Optional discrete effort to forward as
                ``output_config.effort`` (only on adaptive-only models).

        Returns:
            dict: Fields to apply via ``model_copy``.
        """
        thinking: dict = {"type": "adaptive"}
        update: dict = {"thinking": thinking}
        if self._temperature is not None:
            update["temperature"] = self._temperature
        if self._anthropic_requires_adaptive(model_name):
            thinking["display"] = "summarized"
            if level is not None:
                # ``reasoning_effort`` is the field name (serialization alias ``effort``,
                # sent as ``output_config.effort``); ``model_copy(update=...)`` requires the
                # field name, not the alias.
                update["reasoning_effort"] = ANTHROPIC_EFFORT_MAP[level]
        return update

    @staticmethod
    def _anthropic_requires_adaptive(model_name: str) -> bool:
        """Returns True when a Claude model requires adaptive thinking + effort.

        Newer Claude models (Opus 4.7+ and Sonnet 5+) dropped the
        ``thinking.type.enabled`` / ``budget_tokens`` API and only accept
        ``thinking.type.adaptive`` together with ``output_config.effort``. The version is
        parsed from the model identifier; older date-suffixed or legacy-named models
        (e.g. ``claude-3-7-sonnet``) fall back to the legacy path.

        Args:
            model_name (str): The Anthropic model identifier (e.g. ``claude-sonnet-4-6``).

        Returns:
            bool: True if the model requires the adaptive thinking API.
        """
        match = re.search(r"claude-(?:sonnet|opus|haiku)-(\d+)(?:-(\d+))?", (model_name or "").lower())
        if not match:
            return False
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        return major >= 5 or (major == 4 and minor >= 7)

    @staticmethod
    def _is_google_model(model: BaseChatModel) -> bool:
        """Returns True when ``model`` is a Google Gemini chat model.

        The import is performed lazily so the optional ``langchain-google-genai``
        dependency is only required when a Gemini model is actually in use.

        Args:
            model (BaseChatModel): The model instance to inspect.

        Returns:
            bool: True if ``model`` is a ``ChatGoogleGenerativeAI`` instance.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return False
        return isinstance(model, ChatGoogleGenerativeAI)

    @staticmethod
    def _is_anthropic_model(model: BaseChatModel) -> bool:
        """Returns True when ``model`` is an Anthropic Claude chat model.

        The import is performed lazily so the optional ``langchain-anthropic``
        dependency is only required when a Claude model is actually in use.

        Args:
            model (BaseChatModel): The model instance to inspect.

        Returns:
            bool: True if ``model`` is a ``ChatAnthropic`` instance.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            return False
        return isinstance(model, ChatAnthropic)

    def _extract_reasoning(self, token: BaseMessage) -> str:
        """Extracts reasoning (chain-of-thought) text from a streamed token.

        Supports both provider conventions:

        - OpenAI-compatible models surface reasoning via
          ``additional_kwargs['reasoning_content']``.
        - Google Gemini models surface reasoning as ``thinking`` content blocks
          within the message content list.

        Args:
            token (BaseMessage): A streamed message chunk.

        Returns:
            str: The reasoning text fragment, or an empty string if the token
                carries no reasoning.
        """
        reasoning = token.additional_kwargs.get("reasoning_content")
        if reasoning:
            return reasoning

        content = token.content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "thinking":
                    parts.append(part.get("thinking", ""))
            return "".join(parts)

        return ""

    def chat_stream_reasoning(
        self,
        message: str,
        images: List[str | bytes] = None,
        reasoning_effort: Union["ReasoningEffort", str, int, None] = None,
    ) -> Iterator[ReasoningStreamChunk]:
        """Sends a message and yields both reasoning and answer tokens as they are generated.

        Unlike `chat_stream`, this method separates the model's internal reasoning
        (chain-of-thought) from the final answer. Each yielded item is a
        `ReasoningStreamChunk`: either a `ReasoningChunk` (chain-of-thought) or a
        `ContentChunk` (final answer), both exposing a `content` text fragment.
        Branch on the concrete type with `isinstance`.

        This requires an OpenAI-compatible, Google Gemini, or Anthropic Claude
        reasoning model. The generation can be interrupted by calling `stop_stream()`.

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.
            reasoning_effort (ReasoningEffort | str | int | None): Controls how much
                the model reasons. Pass a discrete level (`ReasoningEffort` or one of
                'minimal'/'low'/'medium'/'high') or an explicit integer token budget
                (`-1` dynamic/unrestricted, `0` off, `N>0` token budget). The value
                is mapped to the provider's native knob (OpenAI `reasoning_effort`,
                Gemini `thinking_level`/`thinking_budget`, Anthropic `thinking` budget_tokens
                on legacy models or `output_config.effort` with adaptive thinking on newer
                ones (Opus 4.7+/Sonnet 5+), llama.cpp `thinking_budget_tokens`). `None` uses
                the model default (for Anthropic, a default thinking budget is applied since
                Claude does not think by default). A bool or a numeric string (e.g. '64') is
                rejected to avoid confusion with an integer budget.

        Yields:
            ReasoningStreamChunk: A `ReasoningChunk` or `ContentChunk` holding a `content` text fragment.

        Raises:
            RuntimeError: If the model is not initialized, does not support reasoning, or the API request fails.
            ValueError: If `reasoning_effort` is not a supported level or budget.
            TypeError: If `reasoning_effort` is not a ReasoningEffort, str, int, or None.
            AlreadyGenerating: If a streaming session is already active.
        """
        try:
            yield from self._chat_stream_reasoning_invoke(message, images, reasoning_effort)

        except (AlreadyGenerating, ValueError, TypeError):
            raise
        except Exception as e:
            self._handle_stream_error(e)

    def _chat_stream_reasoning_invoke(
        self,
        message: str,
        images: List[str | bytes] = None,
        reasoning_effort: Union["ReasoningEffort", str, int, None] = None,
    ) -> Iterator[ReasoningStreamChunk]:
        """Internal method to stream reasoning and answer tokens from the model.

        This is separated from `chat_stream_reasoning()` to allow for better error
        handling and potential reuse in subclasses.

        Args:
            message (str): The input text prompt from the user.
            images (List[str | bytes]): Optional list of image file paths or raw bytes to include in the prompt.
            reasoning_effort (ReasoningEffort | str | int | None): Effort level or token budget.

        Yields:
            ReasoningStreamChunk: A `ReasoningChunk` or `ContentChunk` holding a `content` text fragment.

        Raises:
            RuntimeError: If the internal chain is not initialized or if the API request fails.
            ValueError: If `reasoning_effort` is not a supported level or budget.
            AlreadyGenerating: If a streaming session is already active.
        """
        effort = reasoning_effort if reasoning_effort is not None else self._reasoning_effort_default
        reasoning_model = self._get_reasoning_model(effort)
        if self._keep_streaming.is_set():
            raise AlreadyGenerating("A streaming response is already in progress. Please stop it before starting a new one.")
        assistant_chunks: list[str] = []

        try:
            self._keep_streaming.set()
            input_messages = self._get_message_with_history(message, images)
            loops = 0

            while True:
                gathered = None
                for token in reasoning_model.stream(input_messages):
                    if not self._keep_streaming.is_set():
                        break  # This stops the iteration and halts further token generation

                    reasoning = self._extract_reasoning(token)
                    if reasoning:
                        yield ReasoningChunk(content=reasoning)

                    content = self._content_to_text(token.content)
                    if content:
                        assistant_chunks.append(content)
                        yield ContentChunk(content=content)

                    # Accumulate chunks so streamed tool calls can be assembled.
                    gathered = token if gathered is None else gathered + token

                if not self._keep_streaming.is_set():
                    break

                tool_calls = getattr(gathered, "tool_calls", None) or [] if gathered is not None else []
                if not tool_calls:
                    break

                loops += 1
                if loops > self._max_tool_loops:
                    raise RuntimeError(f"Too many consecutive tool-call loops ({self._max_tool_loops}). Possible tool loop.")

                input_messages.append(gathered)
                input_messages = self._process_tool_calls(tool_calls, input_messages.copy())

        finally:
            self._keep_streaming.clear()
            if len(assistant_chunks) > 0:
                full_response = "".join(assistant_chunks)
                self._history.add_messages([AIMessage(content=full_response)])

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


def model_factory(model_name: CloudModel, **kwargs) -> BaseChatModel:
    """Factory function to instantiate the specific LangChain chat model.

    This function maps the supported `CloudModel` enum values to their respective
    LangChain implementations. In case of prefix-based model identifiers (e.g., 'openai:gpt-5-mini'),
    it extracts the provider and model name accordingly.

    Args:
        model_name (CloudModel): The enum or string identifier for the model.
            Model name can include provider prefixes like 'openai:', 'anthropic:', or 'google:'
            to specify the provider. If no prefix is provided, the model will be defaulted to an OpenAI compatible model.
        **kwargs: Additional arguments passed to the model constructor (e.g., api_key, temperature).

    Returns:
        BaseChatModel: An instance of a LangChain chat model wrapper.

    Raises:
        ValueError: If `model_name` does not match one of the supported options.
    """

    if (
        "base_url" in kwargs
        and not model_name.startswith(f"{CloudModelProvider.OPENAI}:")
        and not model_name.startswith(f"{CloudModelProvider.ANTHROPIC}:")
        and not model_name.startswith(f"{CloudModelProvider.GOOGLE}:")
    ):
        logger.debug(f"Model name '{model_name}' does not specify a provider prefix, but 'base_url' is provided. Defaulting to OpenAI provider.")
        model_name = f"{CloudModelProvider.OPENAI}:{model_name}"

    if model_name == CloudModel.ANTHROPIC_CLAUDE or model_name.startswith(f"{CloudModelProvider.ANTHROPIC}:"):
        from langchain_anthropic import ChatAnthropic

        if model_name.startswith(f"{CloudModelProvider.ANTHROPIC}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatAnthropic(model=model_name, **kwargs)
    elif model_name == CloudModel.OPENAI_GPT or model_name.startswith(f"{CloudModelProvider.OPENAI}:"):
        from .reasoning import ChatOpenAIReasoning

        if model_name.startswith(f"{CloudModelProvider.OPENAI}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatOpenAIReasoning(model=model_name, **kwargs)
    elif model_name == CloudModel.GOOGLE_GEMINI or model_name.startswith(f"{CloudModelProvider.GOOGLE}:"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        if model_name.startswith(f"{CloudModelProvider.GOOGLE}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatGoogleGenerativeAI(model=model_name, **kwargs)
    else:
        raise ValueError(f"Model not supported: {model_name}")
