# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

# Standard OpenAI Responses API streaming events that carry reasoning text.
# - ``response.reasoning_text.delta`` is emitted by models that expose the raw
#   chain-of-thought (e.g. gpt-oss and llama.cpp).
# - ``response.reasoning_summary_text.delta`` is emitted by OpenAI's proprietary
#   reasoning models, which only expose a summary of the reasoning.
_REASONING_DELTA_EVENTS = (
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
)


class ChatOpenAIReasoning(ChatOpenAI):
    """A ``ChatOpenAI`` subclass that surfaces streamed reasoning tokens.

    The langchain-openai Responses API parser only recognizes OpenAI's
    ``response.reasoning_summary_text.delta`` events and silently drops the
    standard ``response.reasoning_text.delta`` events emitted by reasoning
    models such as gpt-oss and llama.cpp. As a result, reasoning tokens are lost
    when streaming through the stock client.

    This subclass intercepts both reasoning delta events during streaming and
    exposes their text via ``additional_kwargs['reasoning_content']`` on the
    yielded message chunks, mirroring the DeepSeek convention. All other events
    are delegated to the standard langchain-openai converter, so regular content
    and tool calls behave exactly as before.

    When the Responses API is not in use (the default Chat Completions path),
    this class behaves identically to ``ChatOpenAI``.
    """

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to our reasoning-aware Responses streaming when applicable."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            yield from self._stream_responses(*args, **kwargs)
        else:
            yield from super()._stream(*args, **kwargs)

    def _stream_responses(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream Responses API output, surfacing reasoning delta events.

        This mirrors ``BaseChatOpenAI._stream_responses`` but adds handling for
        the reasoning delta events, which are otherwise ignored by the
        langchain-openai converter.
        """
        import openai
        from langchain_openai.chat_models import base as _oai_base

        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        try:
            if self.include_response_headers:
                raw_context_manager = self.root_client.with_raw_response.responses.create(**payload)
                context_manager = raw_context_manager.parse()
                headers = {"headers": dict(raw_context_manager.headers)}
            else:
                context_manager = self.root_client.responses.create(**payload)
                headers = {}
            original_schema_obj = kwargs.get("response_format")

            with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                for chunk in response:
                    if getattr(chunk, "type", None) in _REASONING_DELTA_EVENTS:
                        delta = getattr(chunk, "delta", "") or ""
                        if delta:
                            generation_chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={"reasoning_content": delta},
                                )
                            )
                            if run_manager:
                                run_manager.on_llm_new_token("", chunk=generation_chunk)
                            is_first_chunk = False
                            yield generation_chunk
                        continue

                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _oai_base._convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if run_manager:
                            run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            _oai_base._handle_openai_bad_request(e)
        except openai.APIError as e:
            _oai_base._handle_openai_api_error(e)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        """Route to our reasoning-aware async Responses streaming when applicable."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in self._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    async def _astream_responses(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async counterpart of `_stream_responses`, surfacing reasoning deltas."""
        import openai
        from langchain_openai.chat_models import base as _oai_base

        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        try:
            if self.include_response_headers:
                raw_context_manager = await self.root_async_client.with_raw_response.responses.create(**payload)
                context_manager = raw_context_manager.parse()
                headers = {"headers": dict(raw_context_manager.headers)}
            else:
                context_manager = await self.root_async_client.responses.create(**payload)
                headers = {}
            original_schema_obj = kwargs.get("response_format")

            async with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                async for chunk in response:
                    if getattr(chunk, "type", None) in _REASONING_DELTA_EVENTS:
                        delta = getattr(chunk, "delta", "") or ""
                        if delta:
                            generation_chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    additional_kwargs={"reasoning_content": delta},
                                )
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token("", chunk=generation_chunk)
                            is_first_chunk = False
                            yield generation_chunk
                        continue

                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _oai_base._convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if run_manager:
                            await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk)
                        is_first_chunk = False
                        if "reasoning" in generation_chunk.message.additional_kwargs:
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            _oai_base._handle_openai_bad_request(e)
        except openai.APIError as e:
            _oai_base._handle_openai_api_error(e)
