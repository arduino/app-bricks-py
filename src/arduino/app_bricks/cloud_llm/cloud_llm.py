# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import asyncio
import base64
import os
import threading
from typing import Iterator, List, Optional, Union, Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, ToolCall

from arduino.app_utils import brick

from .utils import logger
from .models import CloudModel, CloudModelProvider
from .memory import WindowedChatMessageHistory

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
        max_tool_loops: int = 8,
        timeout: int = 30,
        tools: List[Callable[..., Any]] = None,
        callbacks: Any = None,
        **kwargs,
    ):
        """Initializes the CloudLLM brick with the specified provider and configuration.

        Args:
            api_key (str): The API access key for the target LLM service. Defaults to the
                'API_KEY' environment variable.
            model (Union[str, CloudModel]): The model identifier. Accepts a `CloudModel`
                enum member (e.g., `CloudModel.OPENAI_GPT`) or its corresponding raw string
                value (e.g., `'openai:gpt-4o-mini'`). Defaults to `CloudModel.ANTHROPIC_CLAUDE`.
                To identify the model provider, you need to use prefixes like 'openai:', 'anthropic:', or 'google:'.
            system_prompt (str): A system-level instruction that defines the AI's persona
                and constraints (e.g., "You are a helpful assistant"). Defaults to empty.
            temperature (Optional[float]): The sampling temperature between 0.0 and 1.0.
                Higher values make output more random/creative; lower values make it more
                deterministic. Defaults to 0.7.
            max_tool_loops (int): The maximum number of consecutive tool-call loops
                allowed during a single chat interaction. Defaults to 8.
            timeout (int): The maximum duration in seconds to wait for a response before
                timing out. Defaults to 30.
            callbacks (Any): Optional callbacks for monitoring generation events.
            tools (List[Callable[..., Any]]): A list of callable tool functions to register. Defaults to None.
            **kwargs: Additional arguments passed to the model constructor

        Raises:
            ValueError: If `api_key` is not provided (empty string).
        """
        if api_key == "":
            raise ValueError("API key is required to initialize CloudLLM brick.")

        self._api_key = api_key

        # Model configuration
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tool_loops = max_tool_loops
        self._timeout = timeout
        self._callbacks = callbacks

        # Registered tools
        self._tools_map = {}
        if tools is None:
            self._tools = []
        else:
            self._tools = tools
            for tool_func in tools:
                self._tools_map[tool_func.name] = tool_func

        # LangChain components
        self._model = model_factory(
            model,
            api_key=self._api_key,
            temperature=self._temperature,
            timeout=self._timeout,
            **kwargs,
        )

        if self._tools and len(self._tools) > 0:
            logger.info(f"Binding {len(self._tools)} tool(s) to the model.")
            self._model = self._model.bind_tools(tools=self._tools)

        # Memory management
        self.with_memory(DEFAULT_MEMORY)

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
        self._history = WindowedChatMessageHistory(k=self._max_messages, system_message=self._system_prompt)

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
                        content=tool_output,
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
        if self._model is None:
            raise RuntimeError("CloudLLM brick is not started. Please call start() before generating text.")

        try:
            input_messages = self._get_message_with_history(message, images)
            loops = 0

            while True:
                message = self._model.invoke(input=input_messages, config={"callbacks": self._callbacks})
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

        except Exception as e:
            raise RuntimeError(f"Response generation failed: {e}")

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
        if self._model is None:
            raise RuntimeError("CloudLLM brick is not started. Please call start() before generating text.")
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
                    if token.content:
                        assistant_chunks.append(token.content)
                        yield token.content

            # If there were tool calls, process them
            if len(tool_calls) > 0:
                input_messages = self._process_tool_calls(tool_calls, input_messages.copy())
                for token in self._model.stream(input=input_messages, config={"callbacks": self._callbacks}):
                    if not self._keep_streaming.is_set():
                        break
                    if token.content:
                        assistant_chunks.append(token.content)
                        yield token.content

        except Exception as e:
            raise RuntimeError(f"Response generation failed: {e}")
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
    LangChain implementations.

    Args:
        model_name (CloudModel): The enum or string identifier for the model.
            Model name can include provider prefixes like 'openai:', 'anthropic:', or 'google:'
            to specify the provider.
        **kwargs: Additional arguments passed to the model constructor (e.g., api_key, temperature).

    Returns:
        BaseChatModel: An instance of a LangChain chat model wrapper.

    Raises:
        ValueError: If `model_name` does not match one of the supported `CloudModel` options.
    """
    if model_name == CloudModel.ANTHROPIC_CLAUDE or model_name.startswith(f"{CloudModelProvider.ANTHROPIC}:"):
        from langchain_anthropic import ChatAnthropic

        if model_name.startswith(f"{CloudModelProvider.ANTHROPIC}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatAnthropic(model=model_name, **kwargs)
    elif model_name == CloudModel.OPENAI_GPT or model_name.startswith(f"{CloudModelProvider.OPENAI}:"):
        from langchain_openai import ChatOpenAI

        if model_name.startswith(f"{CloudModelProvider.OPENAI}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatOpenAI(model=model_name, **kwargs)
    elif model_name == CloudModel.GOOGLE_GEMINI or model_name.startswith(f"{CloudModelProvider.GOOGLE}:"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        if model_name.startswith(f"{CloudModelProvider.GOOGLE}:"):
            model_name = model_name.split(":", 1)[1]

        return ChatGoogleGenerativeAI(model=model_name, **kwargs)
    else:
        raise ValueError(f"Model not supported: {model_name}")
