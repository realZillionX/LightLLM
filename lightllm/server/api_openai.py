from fastapi.responses import Response, StreamingResponse, JSONResponse
from lightllm.server.core.objs.x2i_params import X2IParams
from .api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionLogprobs,
    CompletionStreamResponse,
    CompletionStreamChoice,
    FunctionResponse,
    ToolCall,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionRequestV2,
    MessageContent,
    ImageURL,
)
import asyncio
import collections
import time
import uvloop
import requests
import base64
import os
from io import BytesIO
import pickle
import uuid

from lightllm.server.reasoning_parser import ReasoningParser

from .function_call_parser import TOOLS_TAG_LIST, FunctionCallParser, ToolCallItem
from .build_prompt import build_prompt, init_tokenizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
from PIL import Image
import multiprocessing as mp
from typing import Any, AsyncGenerator, Optional, Union, List, Dict, Tuple
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from lightllm.server.core.objs.sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score
from lightllm.utils.envs_utils import get_env_start_args, get_lightllm_websocket_max_message_size
from lightllm.utils.error_utils import ClientDisconnected, InvalidRequestError, SERVER_BUSY_MESSAGE, ServerBusyError

from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from dataclasses import dataclass

from .api_errors import create_error_response
from .api_stream_obj import CustomStreamingResponse
from .api_models import (
    PromptTokensDetails,
    CompletionTokensDetails,
)

logger = init_logger(__name__)


async def _safe_stream_wrapper(stream_generator):
    """Convert generation errors to SSE events after the response has started."""
    first_chunk_sent = False
    try:
        async for item in stream_generator:
            first_chunk_sent = True
            yield item
    except InvalidRequestError as e:
        if not first_chunk_sent:
            raise
        error_data = json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
    except ValueError as e:
        if not first_chunk_sent:
            # Request setup is lazy for streaming responses, so validation
            # errors raised here bypass the endpoint's normal ValueError
            # handler. Convert them to the dedicated request-error type so
            # the application-level handler returns HTTP 400.
            raise InvalidRequestError(str(e)) from e
        error_data = json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}, ensure_ascii=False)
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
    except ServerBusyError as e:
        logger.debug("Server busy detail: %s", e.message)
        if not first_chunk_sent:
            raise
        error_data = json.dumps(
            {"error": {"message": SERVER_BUSY_MESSAGE, "type": "server_error", "code": "stream_error"}},
            ensure_ascii=False,
        )
        yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"
    except ClientDisconnected as e:
        logger.warning(str(e))
        # Client is gone — there's no point yielding more SSE chunks. Stop quietly.
        return


def _serialize_sse_chunk(chunk, choice_nulls=(), response_nulls=()):
    """Serialize a streaming chunk, explicitly including specified null fields."""
    d = chunk.model_dump(exclude_none=True)
    if choice_nulls and d.get("choices"):
        for choice in d["choices"]:
            for field in choice_nulls:
                choice[field] = None
    for field in response_nulls:
        d[field] = None
    return json.dumps(d, ensure_ascii=False)


def _process_tool_call_id(
    tool_call_parser,
    call_item: ToolCallItem,
    history_tool_calls_cnt: int,
) -> str:
    """Process for generating a new and unique `tool_call_id`"""
    if tool_call_parser != "kimi_k2":
        # A simple uuid is sufficient for all models except for Kimi-K2.
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        return tool_call_id
    else:
        # Align with Kimi-K2 format: functions.{name}:{index}
        # Kimi-K2 allows multiple tool_calls in one message;
        # SGLang sets call_item.tool_index to the *local* position inside that message.
        # Therefore, the index must be corrected by using
        # `history_tool_calls_cnt + call_item.tool_index` to ensure globally unique and properly ordered.
        tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt + call_item.tool_index}"
        logger.debug(
            f"Process tool call idx, parser: {tool_call_parser}, \
            tool_call_id: {tool_call_id}, \
            history_cnt: {history_tool_calls_cnt}"
        )
        return tool_call_id


def _get_history_tool_calls_cnt(request: ChatCompletionRequest) -> int:
    """Counts the number of tool calls in the request's message history.

    NOTE: This method is only useful for models that include self-increasing
    history tool call idx in tool calls id, such as kimi-k2

    Args:
        request: The chat completion request object.

    Returns:
        The total number of tool calls in the history, or 0 if not applicable.
    """
    messages = getattr(request, "messages", [])
    idx = 0
    for msg in messages:
        if msg.role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
    return idx


def _is_force_thinking_mode(request: ChatCompletionRequest) -> bool:
    """Whether this request uses forced thinking / reasoning (parser + template)."""
    from .build_prompt import tokenizer_supports_force_thinking

    if not tokenizer_supports_force_thinking():
        return False

    reasoning_parser = get_env_start_args().reasoning_parser
    if not reasoning_parser:
        return False
    if reasoning_parser in ["qwen3-thinking", "gpt-oss", "minimax"]:
        return True
    if reasoning_parser in ["deepseek-v3"]:
        return request.chat_template_kwargs is not None and request.chat_template_kwargs.get("thinking") is True
    if reasoning_parser in ["qwen3", "glm45", "nano_v3", "interns1", "gemma4"]:
        # qwen3, glm45, nano_v3, interns1, and gemma4 are reasoning by default;
        return not request.chat_template_kwargs or request.chat_template_kwargs.get("enable_thinking", True) is True
    return True  # default


def _process_reasoning_stream(
    index: int,
    delta: str,
    reasoning_parser_dict: Dict[int, ReasoningParser],
    metadata: Dict[str, Any],
    request: ChatCompletionRequest,
) -> tuple[Optional[str], str]:
    """Process reasoning content and update its token usage."""
    if index not in reasoning_parser_dict:
        reasoning_parser_dict[index] = ReasoningParser(
            get_env_start_args().reasoning_parser,
            request.stream_reasoning,
            _is_force_thinking_mode(request),
        )
    parser = reasoning_parser_dict[index]
    token_id = metadata.get("id")
    if token_id is not None:
        parser.update_reasoning_token_count(int(token_id))
    return parser.parse_stream_chunk(delta)


def _process_tools_stream(index: int, delta: str, parser_dict: Dict, request: ChatCompletionRequest):
    from .api_http import g_objs

    if index not in parser_dict:
        # 为 tool_call_parser 提供默认值
        tool_parser = getattr(g_objs.args, "tool_call_parser", None) or "llama3"
        parser_dict[index] = FunctionCallParser(
            tools=request.tools,
            tool_call_parser=tool_parser,
        )
    parser = parser_dict[index]

    # parse_increment => returns (normal_text, calls)
    normal_text, calls = parser.parse_stream_chunk(delta)
    return normal_text, calls


def _split_tool_argument_delta(arguments: Optional[str]) -> List[str]:
    """Split a complete JSON argument string into OpenAI-style deltas."""
    if not arguments:
        return []
    if len(arguments) <= 2:
        return [arguments]
    if arguments[0] in "{[" and arguments[-1] in "}]":
        middle = arguments[1:-1]
        chunks = [arguments[0]]
        if middle:
            chunks.append(middle)
        chunks.append(arguments[-1])
        return [chunk for chunk in chunks if chunk]
    return [arguments]


def _build_multimodal_params(request: ChatCompletionRequest) -> MultimodalParams:
    """Build LightLLM multimodal inputs from chat content parts."""
    multimodal_params_dict = {"images": [], "audios": []}
    for message in request.messages:
        if isinstance(message.content, list):
            texts = []
            for content in message.content:
                if content.type == "text" and content.text:
                    texts.append(content.text)
                elif content.type == "image_url" and content.image_url is not None:
                    img = content.image_url.url
                    if img.startswith("http://") or img.startswith("https://"):
                        multimodal_params_dict["images"].append({"type": "url", "data": img})
                    elif img.startswith("data:image"):
                        # "data:image/jpeg;base64,{base64_image}"
                        data_str = img.split(";", 1)[1]
                        if data_str.startswith("base64,"):
                            data = data_str[7:]
                            multimodal_params_dict["images"].append({"type": "base64", "data": data})
                        else:
                            raise ValueError("Unrecognized image input.")
                    elif img.startswith("file://"):
                        # Local file path with file:// prefix
                        file_path = img[7:]  # Remove "file://" prefix
                        with open(file_path, "rb") as f:
                            multimodal_params_dict["images"].append(
                                {"type": "base64", "data": base64.b64encode(f.read()).decode("utf-8")}
                            )
                    else:
                        raise ValueError(
                            "Unrecognized image input. Supports local path, http url, base64, and PIL.Image."
                        )
                elif content.type == "audio_url" and content.audio_url is not None:
                    audio = content.audio_url.url
                    if audio.startswith("http://") or audio.startswith("https://"):
                        multimodal_params_dict["audios"].append({"type": "url", "data": audio})
                    elif audio.startswith("data:audio"):
                        data_str = audio.split(";", 1)[1]
                        if data_str.startswith("base64,"):
                            data = data_str[7:]
                            multimodal_params_dict["audios"].append({"type": "base64", "data": data})
                        else:
                            raise ValueError("Unrecognized audio input.")
                    else:
                        raise ValueError("Unrecognized audio input. Supports local path, http url, base64.")

    return MultimodalParams(**multimodal_params_dict)


def _select_chat_template_tools(request: ChatCompletionRequest) -> Optional[List[dict]]:
    """Select the tool schemas exposed to the model's chat template."""
    tools = None
    if request.tools and request.tool_choice != "none":
        # request.skip_special_tokens = False
        # exclude_none=True so optional default-None fields (e.g. ``response``)
        # don't surface in the chat-template render — Function.model_dump()
        # otherwise emits {"response": None}, which chat.jinja's
        # render_extra_keys turns into ``<response>null</response>`` and adds
        # ~7 tokens per tool, drifting prompts away from other engines/clients
        # that pass tools without that field.
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump(exclude_none=True)
                for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump(exclude_none=True) for item in request.tools]

    return tools


async def chat_completions_impl(request: ChatCompletionRequest, raw_request: Request) -> Response:
    from .api_http import g_objs

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.function_call != "none":
        return create_error_response(HTTPStatus.BAD_REQUEST, "The function call feature is not supported")

    created_time = int(time.time())
    multimodal_params = _build_multimodal_params(request)
    tools = _select_chat_template_tools(request)
    prompt = await build_prompt(request, tools)
    sampling_params_dict = {
        "do_sample": request.do_sample,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "n": request.n,
        "best_of": request.n,
        "add_special_tokens": False,
        "seed": request.seed,
    }

    # Gemma-4's reasoning delimiters (<|channel>=100, <channel|>=101) are
    # special tokens. The default skip_special_tokens=True would drop them
    # from the decoded stream and the Gemma4Detector would be unable to
    # find the reasoning boundary. Mirrors vllm's
    # Gemma4ReasoningParser.adjust_request behaviour. Only applied when no
    # explicit value is supplied so callers can still opt back into the
    # default if they want.
    if get_env_start_args().reasoning_parser == "gemma4" and "skip_special_tokens" not in sampling_params_dict:
        sampling_params_dict["skip_special_tokens"] = False

    if request.max_completion_tokens is not None:
        sampling_params_dict["max_new_tokens"] = request.max_completion_tokens
    elif request.max_tokens is not None:
        sampling_params_dict["max_new_tokens"] = request.max_tokens
    if request.stop is not None:
        sampling_params_dict["stop_sequences"] = request.stop

    # Structured output handling
    if request.response_format:
        if request.response_format.type == "json_schema":
            obj = request.response_format.json_schema
            if obj:
                # guided_json takes str instead of dict obj
                sampling_params_dict["guided_json"] = json.dumps(obj.json_schema)
        elif request.response_format.type == "json_object":
            sampling_params_dict["guided_grammar"] = "json"

    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)

    sampling_params.verify()
    results_generator = g_objs.httpserver_manager.generate(
        prompt, sampling_params, multimodal_params, request=raw_request
    )

    # Non-streaming case
    if not request.stream:
        reasoning_parser = get_env_start_args().reasoning_parser
        request_enable_reasoning = _is_force_thinking_mode(request) if reasoning_parser else False
        final_output_dict = collections.defaultdict(list)
        count_output_tokens_dict = collections.defaultdict(lambda: 0)
        finish_reason_dict = {}
        prompt_tokens_dict = {}
        prompt_cache_len_dict = {}
        completion_tokens = 0
        reasoning_parser_dict: Dict[int, ReasoningParser] = {}
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            from .req_id_generator import convert_sub_id_to_group_id

            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            count_output_tokens_dict[sub_req_id] += 1
            final_output_dict[sub_req_id].append(request_output)
            if reasoning_parser:
                parser = reasoning_parser_dict.get(sub_req_id)
                if parser is None:
                    parser = ReasoningParser(
                        reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=request_enable_reasoning,
                    )
                    reasoning_parser_dict[sub_req_id] = parser
                token_id = metadata.get("id")
                if token_id is not None:
                    parser.update_reasoning_token_count(int(token_id))
            if finish_status.is_finished():
                finish_reason_dict[sub_req_id] = finish_status.get_finish_reason()
                prompt_tokens_dict[sub_req_id] = metadata["prompt_tokens"]
                prompt_cache_len_dict[sub_req_id] = metadata.get("prompt_cache_len", 0)
        choices = []
        sub_ids = list(final_output_dict.keys())[: request.n]
        prompt_tokens = prompt_tokens_dict[sub_ids[0]]
        completion_tokens = sum(count_output_tokens_dict[sub_req_id] for sub_req_id in sub_ids)
        cached_tokens = prompt_cache_len_dict.get(sub_ids[0], 0)
        reasoning_tokens = sum(reasoning_parser_dict[sub_req_id].reasoning_tokens for sub_req_id in sub_ids)

        for i in range(request.n):
            sub_req_id = sub_ids[i]
            finish_reason = finish_reason_dict[sub_req_id]
            text = "".join(final_output_dict[sub_req_id])

            # Handle reasoning content
            reasoning_text = None
            if reasoning_parser:
                try:
                    parser = reasoning_parser_dict[sub_req_id]
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST,
                        "Failed to parse fc related info to json format!",
                    )
                if not request.separate_reasoning:
                    text = (reasoning_text or "") + (text or "")
                    reasoning_text = None

            # Handle tool_calls parsing
            tool_calls = None
            tool_choice = request.tool_choice
            tools = request.tools
            if tool_choice != "none" and any([i in text for i in TOOLS_TAG_LIST]):
                try:
                    # 为 tool_call_parser 提供默认值
                    tool_parser = getattr(g_objs.args, "tool_call_parser", None) or "llama3"
                    parser = FunctionCallParser(tools, tool_parser)
                    text, call_info_list = parser.parse_non_stream(text)
                    tool_calls = []
                    history_tool_calls_cnt = _get_history_tool_calls_cnt(request)
                    for call_info in call_info_list:
                        tool_id = _process_tool_call_id(tool_parser, call_info, history_tool_calls_cnt)
                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                index=getattr(call_info, "tool_index", None),
                                type="function",
                                function=FunctionResponse(name=call_info.name, arguments=call_info.parameters),
                            )
                        )
                except Exception as e:
                    logger.error(f"Exception: {e}")
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST,
                        "Failed to parse fc related info to json format!",
                    )
            if tool_calls and finish_reason == "stop":
                finish_reason = "tool_calls"
            chat_message = ChatMessage(
                role="assistant",
                content=text if text else "",
                tool_calls=tool_calls,
                reasoning=reasoning_text if reasoning_text else "",
            )
            choice = ChatCompletionResponseChoice(
                index=i,
                message=chat_message,
                finish_reason=finish_reason,
            )
            choices.append(choice)
        completion_tokens_details = None
        if reasoning_parser:
            completion_tokens_details = CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
            completion_tokens_details=completion_tokens_details,
        )
        resp = ChatCompletionResponse(
            id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
        )
        return resp

    parser_dict = {}
    reasoning_parser_dict = {}

    # Pre-generate a UUID-style request ID (matching the 36888 service format)
    chat_completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Common null fields to include in every streamed choice chunk
    _choice_nulls = ("logprobs", "token_ids", "finish_reason")
    _first_choice_nulls = ("logprobs", "finish_reason")
    _final_choice_nulls = ("logprobs", "token_ids", "stop_reason")
    _first_resp_nulls = ("prompt_token_ids",)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        has_emitted_tool_calls: Dict[int, bool] = collections.defaultdict(bool)
        has_emitted_first_chunk: Dict[int, bool] = collections.defaultdict(bool)
        stream_tool_call_ids: Dict[Tuple[int, int], str] = {}
        from .req_id_generator import convert_sub_id_to_group_id

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            prompt_tokens = metadata["prompt_tokens"]
            cached_tokens = metadata.get("prompt_cache_len", 0)
            completion_tokens += 1
            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            choice_index = sub_req_id - group_request_id

            delta = request_output
            current_finish_reason = finish_status.get_finish_reason()

            # Emit the initial role-only chunk once per choice, as required by the
            # OpenAI SSE spec: role appears only in the first delta with content="".
            if not has_emitted_first_chunk[choice_index]:
                has_emitted_first_chunk[choice_index] = True
                first_choice = ChatCompletionStreamResponseChoice(
                    index=choice_index,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
                first_chunk = ChatCompletionStreamResponse(
                    id=chat_completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[first_choice],
                )
                yield f"data: {_serialize_sse_chunk(first_chunk, _first_choice_nulls, _first_resp_nulls)}\n\n"

            # Handle reasoning content
            if get_env_start_args().reasoning_parser:
                reasoning_text, delta = _process_reasoning_stream(
                    choice_index, delta, reasoning_parser_dict, metadata, request
                )
                if reasoning_text:
                    if request.separate_reasoning:
                        choice_data = ChatCompletionStreamResponseChoice(
                            index=choice_index,
                            delta=DeltaMessage(reasoning=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=chat_completion_id,
                            created=created_time,
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {_serialize_sse_chunk(chunk, _choice_nulls)}\n\n"
                    else:
                        delta = reasoning_text + (delta or "")

            if request.tool_choice != "none" and request.tools:
                # parse_increment => returns (normal_text, calls)
                normal_text, calls = _process_tools_stream(
                    index=choice_index, delta=delta, parser_dict=parser_dict, request=request
                )

                # 1) if there's normal_text, output it as normal content
                if normal_text and (normal_text.strip() or not has_emitted_tool_calls[sub_req_id]):
                    choice_data = ChatCompletionStreamResponseChoice(
                        index=choice_index,
                        delta=DeltaMessage(content=normal_text),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=chat_completion_id,
                        created=created_time,
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {_serialize_sse_chunk(chunk, _choice_nulls)}\n\n"

                # 2) if we found calls, we output them as separate chunk(s)
                history_tool_calls_cnt = _get_history_tool_calls_cnt(request)
                fc_parser = parser_dict[choice_index]
                for call_item in calls:
                    has_emitted_tool_calls[sub_req_id] = True
                    # transform call_item -> FunctionResponse + ToolCall
                    if current_finish_reason == "stop":
                        det = fc_parser.detector
                        ti = call_item.tool_index
                        if ti >= 0 and ti < len(det.prev_tool_call_arr) and ti < len(det.streamed_args_for_tool):
                            latest_delta_len = 0
                            if isinstance(call_item.parameters, str):
                                latest_delta_len = len(call_item.parameters)

                            expected_call = json.dumps(
                                det.prev_tool_call_arr[ti].get("arguments", {}),
                                ensure_ascii=False,
                            )
                            actual_call = det.streamed_args_for_tool[ti]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            call_item.parameters = remaining_call

                    tool_parser = getattr(g_objs.args, "tool_call_parser", None) or "llama3"
                    stream_index = getattr(call_item, "tool_index", None)
                    id_key = (choice_index, stream_index)
                    if call_item.name:
                        if id_key not in stream_tool_call_ids:
                            stream_tool_call_ids[id_key] = _process_tool_call_id(
                                tool_parser, call_item, history_tool_calls_cnt
                            )
                        tool_call_id = stream_tool_call_ids[id_key]
                        function_name = call_item.name
                    else:
                        tool_call_id = stream_tool_call_ids.get(id_key)
                        function_name = None

                    is_tool_head = call_item.name is not None

                    if is_tool_head and call_item.parameters:
                        head_tool_call = ToolCall(
                            id=tool_call_id,
                            index=stream_index,
                            type="function",
                            function=FunctionResponse(
                                name=function_name,
                                arguments="",
                            ),
                        )
                        head_choice = ChatCompletionStreamResponseChoice(
                            index=choice_index,
                            delta=DeltaMessage(tool_calls=[head_tool_call]),
                            finish_reason=None,
                        )
                        head_chunk = ChatCompletionStreamResponse(
                            id=chat_completion_id,
                            created=created_time,
                            choices=[head_choice],
                            model=request.model,
                        )
                        yield f"data: {_serialize_sse_chunk(head_chunk, _choice_nulls)}\n\n"

                        for arg_delta in _split_tool_argument_delta(call_item.parameters):
                            arg_tool_call = ToolCall(
                                index=stream_index,
                                function=FunctionResponse(arguments=arg_delta),
                            )
                            arg_choice = ChatCompletionStreamResponseChoice(
                                index=choice_index,
                                delta=DeltaMessage(tool_calls=[arg_tool_call]),
                                finish_reason=None,
                            )
                            arg_chunk = ChatCompletionStreamResponse(
                                id=chat_completion_id,
                                created=created_time,
                                choices=[arg_choice],
                                model=request.model,
                            )
                            yield f"data: {_serialize_sse_chunk(arg_chunk, _choice_nulls)}\n\n"
                    else:
                        tool_call = ToolCall(
                            id=tool_call_id if is_tool_head else None,
                            index=stream_index,
                            type="function" if is_tool_head else None,
                            function=FunctionResponse(
                                name=function_name,
                                arguments=(
                                    (call_item.parameters if call_item.parameters is not None else "")
                                    if is_tool_head
                                    else call_item.parameters
                                ),
                            ),
                        )
                        choice_data = ChatCompletionStreamResponseChoice(
                            index=choice_index,
                            delta=DeltaMessage(tool_calls=[tool_call]),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=chat_completion_id,
                            created=created_time,
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {_serialize_sse_chunk(chunk, _choice_nulls)}\n\n"
            else:
                if delta:
                    # If this is the final token, merge content with finish_reason
                    if current_finish_reason is not None:
                        if has_emitted_tool_calls[sub_req_id] and current_finish_reason == "stop":
                            current_finish_reason = "tool_calls"
                        delta_message = DeltaMessage(content=delta)
                        stream_choice = ChatCompletionStreamResponseChoice(
                            index=choice_index, delta=delta_message, finish_reason=current_finish_reason
                        )
                        stream_resp = ChatCompletionStreamResponse(
                            id=chat_completion_id,
                            created=created_time,
                            model=request.model,
                            choices=[stream_choice],
                        )
                        yield f"data: {_serialize_sse_chunk(stream_resp, _final_choice_nulls)}\n\n"
                        # Skip the separate final-chunk logic below
                        continue
                    else:
                        delta_message = DeltaMessage(content=delta)
                        stream_choice = ChatCompletionStreamResponseChoice(
                            index=choice_index, delta=delta_message, finish_reason=None
                        )
                        stream_resp = ChatCompletionStreamResponse(
                            id=chat_completion_id,
                            created=created_time,
                            model=request.model,
                            choices=[stream_choice],
                        )
                        yield f"data: {_serialize_sse_chunk(stream_resp, _choice_nulls)}\n\n"

            # Emit a per-choice final chunk with finish_reason (for tool_calls path
            # or when no delta was emitted alongside finish_reason).
            if current_finish_reason is not None:
                # Flush any buffered reasoning content that was never released
                # (e.g., max_completion_tokens hit before </think/> was seen).
                if get_env_start_args().reasoning_parser:
                    parser = reasoning_parser_dict.get(choice_index)
                    if parser is not None:
                        flush_reasoning, flush_text = parser.flush()
                        if flush_reasoning:
                            if request.separate_reasoning:
                                flush_choice = ChatCompletionStreamResponseChoice(
                                    index=choice_index,
                                    delta=DeltaMessage(reasoning=flush_reasoning),
                                    finish_reason=None,
                                )
                            else:
                                # vLLM compat: emit buffered thinking as content
                                flush_choice = ChatCompletionStreamResponseChoice(
                                    index=choice_index,
                                    delta=DeltaMessage(content=flush_reasoning),
                                    finish_reason=None,
                                )
                            flush_chunk = ChatCompletionStreamResponse(
                                id=chat_completion_id,
                                created=created_time,
                                model=request.model,
                                choices=[flush_choice],
                            )
                            yield f"data: {_serialize_sse_chunk(flush_chunk, _choice_nulls)}\n\n"
                        if flush_text:
                            flush_choice = ChatCompletionStreamResponseChoice(
                                index=choice_index,
                                delta=DeltaMessage(content=flush_text),
                                finish_reason=None,
                            )
                            flush_chunk = ChatCompletionStreamResponse(
                                id=chat_completion_id,
                                created=created_time,
                                model=request.model,
                                choices=[flush_choice],
                            )
                            yield f"data: {_serialize_sse_chunk(flush_chunk, _choice_nulls)}\n\n"
                if has_emitted_tool_calls[sub_req_id] and current_finish_reason == "stop":
                    current_finish_reason = "tool_calls"
                final_choice = ChatCompletionStreamResponseChoice(
                    index=choice_index,
                    delta=DeltaMessage(),
                    finish_reason=current_finish_reason,
                )
                final_chunk = ChatCompletionStreamResponse(
                    id=chat_completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[final_choice],
                )
                yield f"data: {_serialize_sse_chunk(final_chunk, _final_choice_nulls)}\n\n"

        reasoning_parser = get_env_start_args().reasoning_parser
        completion_tokens_details = None
        if reasoning_parser:
            reasoning_tokens = sum(parser.reasoning_tokens for parser in reasoning_parser_dict.values())
            completion_tokens_details = CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
            completion_tokens_details=completion_tokens_details,
        )
        usage_chunk = ChatCompletionStreamResponse(
            id=chat_completion_id,
            created=created_time,
            choices=[],  # Empty choices array as per OpenAI spec
            model=request.model,
            usage=usage,
        )
        yield f"data: {json.dumps(usage_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n".encode("utf-8")

    background_tasks = BackgroundTasks()
    return CustomStreamingResponse(
        _safe_stream_wrapper(stream_results()), media_type="text/event-stream", background=background_tasks
    )


async def completions_impl(request: CompletionRequest, raw_request: Request) -> Response:
    from .api_http import g_objs

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    created_time = int(time.time())

    # Parse and normalize prompts
    prompts = []
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "Prompt cannot be empty",
            )

        # Check if it's a list of integers (token IDs)
        if isinstance(request.prompt[0], int):
            prompts.append(request.prompt)
        elif isinstance(request.prompt[0], list):
            for token_list in request.prompt:
                prompts.append(token_list)
        else:
            # List of strings
            prompts = request.prompt
    else:
        # Single string
        prompts = [request.prompt]

    # Handle suffix for completion mode
    if request.suffix:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The suffix parameter is not currently supported",
        )

    # Prepare sampling parameters - same as g_generate_stream_func pattern
    sampling_params_dict = {
        "do_sample": request.do_sample,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "n": request.n,
        "best_of": request.best_of,
        "add_special_tokens": False,
        "seed": request.seed,
    }
    if request.max_completion_tokens is not None:
        sampling_params_dict["max_new_tokens"] = request.max_completion_tokens
    elif request.max_tokens is not None:
        sampling_params_dict["max_new_tokens"] = request.max_tokens
    if request.stop is not None:
        sampling_params_dict["stop_sequences"] = request.stop

    if request.response_format:
        if request.response_format.type == "json_schema":
            obj = request.response_format.json_schema
            if obj:
                # guided_json takes str instead of dict obj
                sampling_params_dict["guided_json"] = json.dumps(obj.json_schema)
        elif request.response_format.type == "json_object":
            sampling_params_dict["guided_grammar"] = "json"

    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)
    sampling_params.verify()

    # v1/completions does not support multimodal inputs, so we use an empty MultimodalParams
    multimodal_params = MultimodalParams()

    return await _process_prompts_completion(
        prompts, sampling_params, sampling_params_dict, multimodal_params, raw_request, request, created_time
    )


async def _process_prompts_completion(
    prompts: Union[List[str], List[List[int]]],
    sampling_params: SamplingParams,
    sampling_params_dict: Dict,
    multimodal_params: MultimodalParams,
    raw_request: Request,
    request: CompletionRequest,
    created_time: int,
) -> Response:
    from .api_http import g_objs
    import asyncio

    if request.stream:
        if len(prompts) > 1:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                "Streaming is not supported for batch requests",
            )

        return await _handle_streaming_completion(
            prompts[0], sampling_params, multimodal_params, raw_request, request, created_time
        )

    async def process_single_prompt(prompt: Union[str, List[int]]):
        if len(prompts) > 1:
            individual_sampling_params = SamplingParams()
            individual_sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)
            individual_sampling_params.verify()
        else:
            individual_sampling_params = sampling_params

        # Convert token array to string for _collect_generation_results
        prompt_str = prompt
        if isinstance(prompt, list):
            prompt_str = g_objs.httpserver_manager.tokenizer.decode(prompt, skip_special_tokens=False)

        generator = g_objs.httpserver_manager.generate(
            prompt, individual_sampling_params, multimodal_params, request=raw_request
        )

        return await _collect_generation_results(generator, request, prompt_str, individual_sampling_params)

    tasks = [asyncio.create_task(process_single_prompt(prompt)) for prompt in prompts]

    results_by_prompt = await asyncio.gather(*tasks)
    return _build_completion_response(results_by_prompt, request, created_time, len(prompts) > 1)


async def _handle_streaming_completion(
    prompt: Union[str, List[int]],
    sampling_params: SamplingParams,
    multimodal_params: MultimodalParams,
    raw_request: Request,
    request: CompletionRequest,
    created_time: int,
) -> Response:
    from .api_http import g_objs

    results_generator = g_objs.httpserver_manager.generate(
        prompt, sampling_params, multimodal_params, request=raw_request
    )

    async def stream_results() -> AsyncGenerator[bytes, None]:
        from .req_id_generator import convert_sub_id_to_group_id

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            choice_index = sub_req_id - group_request_id
            prompt_tokens = metadata["prompt_tokens"]
            cached_tokens = metadata.get("prompt_cache_len", 0)
            completion_tokens += 1
            current_finish_reason = None
            if finish_status.is_finished():
                current_finish_reason = finish_status.get_finish_reason()

            output_text = request_output
            if request.echo and metadata.get("is_first_token", False):
                prompt_str = prompt
                if isinstance(prompt, list):
                    prompt_str = g_objs.httpserver_manager.tokenizer.decode(prompt, skip_special_tokens=False)
                output_text = prompt_str + output_text

            stream_choice = CompletionStreamChoice(
                index=choice_index,
                text=output_text,
                finish_reason=current_finish_reason,
                logprobs=None if request.logprobs is None else {},
            )
            stream_resp = CompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield f"data: {json.dumps(stream_resp.model_dump(), ensure_ascii=False)}\n\n"

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        )
        usage_chunk = CompletionStreamResponse(
            id=group_request_id,
            created=created_time,
            choices=[],  # Empty choices array as per OpenAI spec
            model=request.model,
            usage=usage,
        )
        yield f"data: {json.dumps(usage_chunk.model_dump(), ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    background_tasks = BackgroundTasks()
    return CustomStreamingResponse(
        _safe_stream_wrapper(stream_results()), media_type="text/event-stream", background=background_tasks
    )


async def _collect_generation_results(
    generator, request: CompletionRequest, prompt: str, sampling_params: SamplingParams
):
    final_outputs = collections.defaultdict(list)
    output_token_counts = collections.defaultdict(int)
    finish_reasons = {}
    prompt_tokens = {}
    prompt_cache_lens = {}
    token_infos = collections.defaultdict(list)
    prompt_logprobs = {}
    prompt_token_ids = {}

    async for sub_req_id, request_output, metadata, finish_status in generator:
        if sub_req_id not in prompt_token_ids:
            prompt_logprobs[sub_req_id] = metadata.get("prompt_logprobs")
            prompt_token_ids[sub_req_id] = metadata.get("prompt_token_ids")

        output_token_counts[sub_req_id] += 1
        final_outputs[sub_req_id].append(request_output)

        if request.logprobs is not None:
            token_infos[sub_req_id].append(
                {
                    "text": request_output,
                    "logprob": metadata.get("logprob"),
                    "id": metadata.get("id"),
                }
            )

        if finish_status.is_finished():
            finish_reasons[sub_req_id] = finish_status.get_finish_reason()
            prompt_tokens[sub_req_id] = metadata["prompt_tokens"]
            prompt_cache_lens[sub_req_id] = metadata.get("prompt_cache_len", 0)

    results = []
    for sub_req_id in sorted(final_outputs)[: request.n]:
        final_text = "".join(final_outputs[sub_req_id])
        finish_reason = finish_reasons.get(sub_req_id)

        if finish_reason == "stop" and sampling_params.stop_sequences.size > 0:
            for stop_str in sampling_params.stop_sequences.to_strings():
                stop_index = final_text.rfind(
                    stop_str,
                    max(0, len(final_text) - len(stop_str) - 20),
                    len(final_text),
                )
                if stop_index != -1:
                    logger.debug("removed stop sequence in tail: '%s'", final_text[stop_index:])
                    final_text = final_text[:stop_index]
                    break

        results.append(
            {
                "text": final_text,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens.get(sub_req_id, 0),
                "prompt_cache_len": prompt_cache_lens.get(sub_req_id, 0),
                "completion_tokens": output_token_counts[sub_req_id],
                "token_infos": (token_infos[sub_req_id] if request.logprobs is not None else None),
                "prompt_logprobs": prompt_logprobs[sub_req_id],
                "prompt_token_ids": prompt_token_ids[sub_req_id],
                "prompt_text": prompt,
            }
        )

    return results


def _build_completion_response(
    results_by_prompt: List[List[Dict]], request: CompletionRequest, created_time: int, is_batch: bool
):
    from .api_http import g_objs

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0

    for prompt_results in results_by_prompt:
        if not prompt_results:
            continue

        total_prompt_tokens += prompt_results[0]["prompt_tokens"]
        total_cached_tokens += prompt_results[0].get("prompt_cache_len", 0)

        for result in prompt_results:
            text = result["text"]
            if request.echo:
                text = result["prompt_text"] + text

            logprobs_data = _build_logprobs_data(result, request, g_objs.httpserver_manager.tokenizer)
            choices.append(
                CompletionChoice(
                    index=len(choices),
                    text=text,
                    finish_reason=result["finish_reason"],
                    logprobs=CompletionLogprobs(**logprobs_data) if logprobs_data else None,
                )
            )
            total_completion_tokens += result["completion_tokens"]

    usage = UsageInfo(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=total_cached_tokens),
    )

    if is_batch:
        group_request_id = f"cmpl-batch-{uuid.uuid4().hex[:8]}"
    else:
        group_request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    return CompletionResponse(
        id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
    )


def _build_logprobs_data(result: Dict, request: CompletionRequest, tokenizer) -> Dict:
    if request.logprobs is None:
        return None

    all_tokens = []
    all_token_logprobs = []
    all_text_offsets = []
    offset = 0

    def add_tokens_to_logprobs(token_ids=None, token_infos=None, logprob_map=None):
        def add_single_token(token_text: str, logprob: float):
            nonlocal offset
            all_tokens.append(token_text)
            all_token_logprobs.append(logprob)
            all_text_offsets.append(offset)
            offset += len(token_text)

        if token_ids is not None:
            for token_id in token_ids:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                logprob = logprob_map.get(token_id, None) if logprob_map else None
                add_single_token(token_text, logprob)
        elif token_infos is not None:
            for token_info in token_infos:
                add_single_token(token_info["text"], token_info["logprob"])

    # 处理 echo 模式下的 prompt tokens
    if request.echo and result.get("prompt_logprobs") is not None:
        prompt_logprobs = result["prompt_logprobs"]
        prompt_token_ids = result.get("prompt_token_ids")

        # 创建 token_id 到 logprob 的映射
        logprob_map = {}
        for current_token_id, logprobs_dict in prompt_logprobs:
            for next_token_id, logprob in logprobs_dict.items():
                logprob_map[int(next_token_id)] = logprob

        # 处理所有 prompt tokens
        if prompt_token_ids is not None:
            add_tokens_to_logprobs(token_ids=prompt_token_ids, logprob_map=logprob_map)

    elif request.echo:
        # echo=True 但没有 prompt logprobs
        prompt_token_ids = result.get("prompt_token_ids")
        if prompt_token_ids is not None:
            add_tokens_to_logprobs(token_ids=prompt_token_ids)
        else:
            # 回退：重新 tokenize prompt
            prompt_tokens = tokenizer.encode(result["prompt_text"], add_special_tokens=False)
            add_tokens_to_logprobs(token_ids=prompt_tokens)

    # 添加生成的 tokens 和 logprobs
    if result.get("token_infos"):
        add_tokens_to_logprobs(token_infos=result["token_infos"])

    top_logprobs_list = []
    for i, (token, logprob) in enumerate(zip(all_tokens, all_token_logprobs)):
        if logprob is not None:
            # TODO: 标准实现需要从后端获取top-k个logprobs数据
            # 目前后端不支持，只能获取所选token的logprobs
            top_logprobs_list.append({token: logprob})
        else:
            top_logprobs_list.append(None)

    return {
        "tokens": all_tokens,
        "token_logprobs": all_token_logprobs,
        "top_logprobs": top_logprobs_list,
        "text_offset": all_text_offsets,
    }




def _get_reasoning_from_request(request: ChatCompletionRequest) -> bool:
    """Judge whether the request needs reasoning"""
    reasoning_parser = get_env_start_args().reasoning_parser
    if not reasoning_parser:
        return False
    if reasoning_parser in ["deepseek-v3"]:
        return request.chat_template_kwargs is not None and request.chat_template_kwargs.get("thinking") is True
    if reasoning_parser in ["qwen3", "glm45", "nano_v3", "interns1"]:
        # qwen3, glm45, nano_v3, and interns1 are reasoning by default
        return not request.chat_template_kwargs or request.chat_template_kwargs.get("enable_thinking", True) is True
    return True


def _get_images_and_audios(request: ChatCompletionRequest):
    images, audios = [], []
    for message in request.messages:
        if isinstance(message.content, list):
            for content in message.content:
                if content.type == "image_url" and content.image_url is not None:
                    img = content.image_url.url
                    if img.startswith("http://") or img.startswith("https://"):
                        images.append({"type": "url", "data": img})
                    elif img.startswith("data:image"):
                        # "data:image/jpeg;base64,{base64_image}"
                        data_str = img.split(";", 1)[1]
                        if data_str.startswith("base64,"):
                            data = data_str[7:]
                            images.append({"type": "base64", "data": data})
                        else:
                            raise ValueError("Unrecognized image input.")
                    elif img.startswith("file://"):
                        # Local file path with file:// prefix
                        file_path = img[7:]  # Remove "file://" prefix
                        with open(file_path, "rb") as f:
                            images.append({"type": "base64", "data": base64.b64encode(f.read()).decode("utf-8")})
                    else:
                        raise ValueError(
                            "Unrecognized image input. Supports local path, http url, base64, and PIL.Image."
                        )
                elif content.type == "audio_url" and content.audio_url is not None:
                    audio = content.audio_url.url
                    if audio.startswith("http://") or audio.startswith("https://"):
                        audios.append({"type": "url", "data": audio})
                    elif audio.startswith("data:audio"):
                        data_str = audio.split(";", 1)[1]
                        if data_str.startswith("base64,"):
                            data = data_str[7:]
                            audios.append({"type": "base64", "data": data})
                        else:
                            raise ValueError("Unrecognized audio input.")
                    else:
                        raise ValueError("Unrecognized audio input. Supports local path, http url, base64.")
    return images, audios


def _get_tools(request: ChatCompletionRequest):
    tools = None
    if request.tools and request.tool_choice != "none":
        # request.skip_special_tokens = False
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump()
                for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump() for item in request.tools]
    return tools


async def _get_text_generator_input(request: ChatCompletionRequest):
    from .api_http import g_objs

    images, audios = _get_images_and_audios(request)
    multimodal_params_dict = {"images": images, "audios": audios}

    tools = _get_tools(request)
    prompt = await build_prompt(request, tools)

    sampling_params_dict = {
        "do_sample": request.do_sample,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "max_new_tokens": request.max_tokens,
        "seed": request.seed,
        "stop_sequences": request.stop,
        "n": request.n,
        "best_of": request.n,
        "add_special_tokens": False,
    }
    # Structured output handling
    if request.response_format:
        if request.response_format.type == "json_schema":
            obj = request.response_format.json_schema
            if obj:
                # guided_json takes str instead of dict obj
                sampling_params_dict["guided_json"] = json.dumps(obj.json_schema)
        elif request.response_format.type == "json_object":
            sampling_params_dict["guided_grammar"] = "json"

    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)

    sampling_params.verify()
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    logger.info(f"call text generator with prompt: {prompt} and {sampling_params_dict}")

    return prompt, sampling_params, multimodal_params


def _raw_image_to_data_url(image: Union[str, bytes], image_type: str) -> str:
    mime = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[image_type]
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode("ascii")
    else:
        if image.startswith("data:"):
            return image
        b64 = image
    return f"data:{mime};base64,{b64}"


def _message_contents_from_raw_images(images: List[Any], image_type: str) -> List[MessageContent]:
    return [
        MessageContent(
            type="image_url",
            image_url=ImageURL(url=_raw_image_to_data_url(img, image_type)),
        )
        for img in images
    ]


def _normalize_image_b64_for_multimodal(image: Union[str, bytes]) -> str:
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    return image


async def chat_completions_impl_v2(request: ChatCompletionRequestV2, raw_request: Request) -> Response:
    """SenseNova inference with one total input/output context limit."""
    from .api_http import g_objs
    from .trajectory_budget import (require_server_limit, image_context_tokens,
                                    span_budget, prompt_token_count, TokenizedMultimodalPrompt)

    manager = g_objs.httpserver_manager
    serving_modality = getattr(manager.args, "sensenova_modality", None)
    if not manager.args.enable_multimodal_x2i and serving_modality != "ti2t":
        return await chat_completions_impl(ChatCompletionRequest(**request.model_dump()), raw_request)
    if serving_modality == "ti2t" and (request.modalities != ["text"] or request.max_images != 0):
        raise ValueError("TI2T serving accepts text output only and max_images=0")
    limit = require_server_limit(request.max_sequence_length, manager.max_req_total_len)
    if request.logit_bias is not None or request.function_call != "none":
        raise ValueError("SenseNova trajectory inference does not support logit bias or function calls")
    if request.n != 1:
        raise ValueError("SenseNova trajectory inference requires n=1")
    if {"max_tokens", "max_completion_tokens"} & request.model_fields_set:
        raise ValueError("Use max_sequence_length; separate completion limits are not supported")
    if request.image_config is not None and request.image_config.num_images not in (None, 1):
        raise ValueError("Each image action generates exactly one image")
    request.chat_template_kwargs = request.chat_template_kwargs or {}
    chat_request = ChatCompletionRequest(**request.model_dump())
    # This lower-level API requires a value; every span replaces it with the
    # exact remaining context after multimodal encoding.
    chat_request.max_tokens = limit
    prompt, sampling, multimodal = await _get_text_generator_input(chat_request)
    await multimodal.verify_and_preload(raw_request)
    initial_tokens = prompt_token_count(manager, prompt, multimodal, sampling)
    initial_image_tokens = sum(manager.tokenizer.get_image_token_length(image) for image in multimodal.images)
    prompt_usage = dict(prompt_text_tokens=initial_tokens - initial_image_tokens,
                        prompt_image_tokens=initial_image_tokens)
    if initial_tokens >= limit:
        raise ValueError(f"input uses {initial_tokens} tokens and leaves no room under total limit {limit}")
    input_image_num = len(multimodal.images)
    image_only = request.modalities == ["image"]
    image_enabled = "image" in request.modalities
    max_images = min(request.max_images, 1) if image_only else request.max_images
    image_start = manager.tokenizer.image_start_tag
    image_id = manager.tokenizer.image_start_id
    image_tag = manager.tokenizer.image_tag
    prompt_ids = manager.tokenizer.encode(
        prompt.replace(image_tag, image_start + manager.tokenizer.image_end_tag, input_image_num),
        None, add_special_tokens=False,
    )
    image_params = X2IParams()
    image_tokens = 0
    if image_enabled:
        image_params.init_from_image_config(request.image_config)
        if input_image_num:
            image_params.update_hw(multimodal.images[0].image_w, multimodal.images[0].image_h)
        image_tokens = image_context_tokens(
            manager.tokenizer, width=int(image_params.width), height=int(image_params.height)
        )
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def events():
        nonlocal prompt
        if image_only and not request.chat_template_kwargs.get("enable_thinking", False):
            # Direct image generation has no sampled <img> text action: both
            # image delimiters belong to the visual context in this case.
            visual_tokens = image_tokens + 1
            if max_images < 1 or initial_tokens + visual_tokens > limit:
                raise ValueError("the requested image does not fit the total trajectory budget")
            images = await manager.generate_image(prompt, image_params, multimodal.clone(),
                                                  request=raw_request, input_image_num=input_image_num)
            if not images or len(images) != 1:
                raise RuntimeError("SenseNova image action must return exactly one image")
            yield {"type": "image", "text": "", "images": _message_contents_from_raw_images(images, request.image_config.image_type)}
            yield {"type": "end", "finish_reason": "stop", "usage": UsageInfo(
                prompt_tokens=initial_tokens, completion_tokens=0, image_context_tokens=visual_tokens,
                total_tokens=initial_tokens + visual_tokens, max_sequence_length=limit, image_limit_hit=True,
                completion_token_ids=[], **prompt_usage)}
            return
        text_tokens = 0
        sampled_token_ids = []
        visual_tokens = 0
        images_used = 0
        context = initial_tokens
        text_only_tail = not image_enabled or max_images == 0
        finish_reason = "length"
        while context < limit:
            span = span_budget(limit, context, image_tokens=image_tokens,
                               allow_image=not text_only_tail and images_used < max_images)
            text_only_tail = not span.allow_image
            if span.max_tokens == 0:
                break
            sampling.max_new_tokens = span.max_tokens
            stops = list(request.stop or []) if not isinstance(request.stop, str) else [request.stop]
            if span.allow_image:
                stops.append(image_start)
            sampling.stop_sequences.initialize(stops, manager.tokenizer)
            sampling.invalid_token_ids.initialize([] if span.allow_image else [image_id])
            chunk = ""
            stopped_on_image = False
            emitted = 0
            span_finish = "length"
            async for _, text, metadata, status in manager.generate(
                TokenizedMultimodalPrompt(tuple(prompt_ids)), sampling, multimodal.clone(), request=raw_request,
                max_req_total_len=span.sequence_limit,
            ):
                emitted += 1
                text_tokens += 1
                sampled_token_ids.append(int(metadata["id"]))
                prompt_ids.append(int(metadata["id"]))
                stopped_on_image = int(metadata["id"]) == image_id
                if not stopped_on_image:
                    chunk += text
                    yield {"type": "text", "text": text}
                if status.is_finished():
                    span_finish = status.get_finish_reason()
                    actual_prompt = int(metadata["prompt_tokens"])
                    if actual_prompt != context:
                        raise RuntimeError("interleaved token prompt disagrees with the trajectory token count")
            if not emitted:
                raise RuntimeError("SenseNova returned an empty text span")
            context += emitted
            prompt += chunk
            if not stopped_on_image:
                if span_finish == "length" and span.allow_image and context < limit:
                    text_only_tail = True
                    continue
                finish_reason = span_finish
                break
            if not span.allow_image:
                raise RuntimeError("SenseNova emitted a masked image action")
            images = await manager.generate_image(
                prompt, image_params, multimodal.clone(), request=raw_request,
                input_image_num=input_image_num,
                conditional_prompt=TokenizedMultimodalPrompt(tuple(prompt_ids)),
            )
            if not images or len(images) != 1:
                raise RuntimeError("SenseNova image action must return exactly one image")
            actual_image_tokens = image_context_tokens(
                manager.tokenizer, width=int(image_params.width), height=int(image_params.height)
            )
            if actual_image_tokens != image_tokens or context + image_tokens > limit:
                raise RuntimeError("image geometry exceeded its reserved context")
            prompt += image_tag
            prompt_ids.append(manager.tokenizer.image_end_id)
            multimodal.add_image({"type": "base64", "data": _normalize_image_b64_for_multimodal(images[0])})
            images_used += 1
            visual_tokens += image_tokens
            context += image_tokens
            yield {"type": "image", "images": _message_contents_from_raw_images(images, request.image_config.image_type),
                   "text": image_tag if not image_only else ""}
            if image_only:
                finish_reason = "stop"
                break
        if context > limit or context != initial_tokens + text_tokens + visual_tokens:
            raise RuntimeError("SenseNova violated total trajectory accounting")
        yield {"type": "end", "finish_reason": finish_reason, "usage": UsageInfo(
            prompt_tokens=initial_tokens, completion_tokens=text_tokens,
            image_context_tokens=visual_tokens, total_tokens=context,
            max_sequence_length=limit, image_limit_hit=image_enabled and images_used == max_images,
            completion_token_ids=sampled_token_ids, **prompt_usage,
        )}

    if not request.stream:
        full_text = ""
        response_images = []
        final = None
        async for event in events():
            if event["type"] == "end":
                final = event
            else:
                full_text += event["text"]
                if event["type"] == "image":
                    response_images.extend(event["images"])
        reasoning_text = None
        text_out = full_text
        reasoning_parser = get_env_start_args().reasoning_parser
        if reasoning_parser and request.separate_reasoning:
            parser = ReasoningParser(model_type=reasoning_parser, stream_reasoning=False,
                                     force_reasoning=_get_reasoning_from_request(request))
            reasoning_text, text_out = parser.parse_non_stream(full_text)
        return ChatCompletionResponse(
            id=request_id, created=created, model=request.model, usage=final["usage"],
            choices=[ChatCompletionResponseChoice(index=0, finish_reason=final["finish_reason"],
                message=ChatMessage(role="assistant", content=text_out,
                    reasoning_content=reasoning_text or "", images=response_images or None))],
        )

    async def stream_result():
        parser_state = {}
        async for event in events():
            if event["type"] == "end":
                payload = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                           "model": request.model, "choices": [{"index": 0, "delta": {},
                           "finish_reason": event["finish_reason"]}], "usage": event["usage"].model_dump()}
            else:
                delta = {"content": event["text"]}
                if event["type"] == "image":
                    delta["images"] = [item.model_dump() for item in event["images"]]
                elif get_env_start_args().reasoning_parser and request.separate_reasoning:
                    reasoning, content = _process_reasoning_stream(0, event["text"], parser_state, event["text"])
                    delta = {"content": content}
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                payload = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                           "model": request.model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            yield ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_result(), media_type="text/event-stream")
