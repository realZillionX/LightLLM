# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import collections
import time

import uvloop
import requests
import base64
import os
import re
from io import BytesIO
import setproctitle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
import uuid
from PIL import Image
import multiprocessing as mp
from typing import AsyncGenerator, Union
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs import StartArgs
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.error_utils import ClientDisconnected, InvalidRequestError, SERVER_BUSY_MESSAGE, ServerBusyError
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_port_args import get_shm_port_args
from dataclasses import asdict, dataclass, is_dataclass

from .api_openai import chat_completions_impl, completions_impl, chat_completions_impl_v2
from .api_errors import create_error_response, create_server_busy_response
from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequestV2,
    ModelCard,
    ModelListResponse,
)
from .build_prompt import build_prompt, init_tokenizer
from .rl_models import (
    CommitWeightsUpdateRequest,
    DestroyWeightsUpdateGroupRequest,
    DistributedWeightsRequest,
    InitWeightsUpdateGroupRequest,
    RLRolloutRequest,
    TensorWeightsRequest,
)

logger = init_logger(__name__)


@dataclass
class G_Objs:
    app: FastAPI = None
    metric_client: MetricClient = None
    args: StartArgs = None
    g_generate_func: Callable = None
    g_generate_stream_func: Callable = None
    g_generate_image_func: Callable = None
    httpserver_manager: Union[HttpServerManager, HttpServerManagerForPDMaster] = None
    shared_token_load: TokenLoad = None
    # OpenAI-compatible "created" timestamp for /v1/models.
    # Should be stable for the lifetime of this server process.
    model_created: int = None

    def set_args(self, args: StartArgs):
        self.args = args
        from .api_lightllm import lightllm_generate, lightllm_generate_stream
        from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl

        if args.use_tgi_api:
            self.g_generate_func = tgi_generate_impl
            self.g_generate_stream_func = tgi_generate_stream_impl
        else:
            self.g_generate_func = lightllm_generate
            self.g_generate_stream_func = lightllm_generate_stream

        if args.enable_multimodal_x2i:
            from .api_lightllm import lightllm_generate_image

            self.g_generate_image_func = lightllm_generate_image

        setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::api_server")

        init_tokenizer(args)  # for openai api
        SamplingParams.load_generation_cfg(args.model_dir)
        CompletionRequest.load_generation_cfg(args.model_dir)
        ChatCompletionRequest.load_generation_cfg(args.model_dir)

        if self.model_created is None:
            self.model_created = int(time.time())

        if args.run_mode == "pd_master":
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
            self.httpserver_manager = HttpServerManagerForPDMaster(
                args=args,
            )
        else:
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
            self.httpserver_manager = HttpServerManager(args=args)
            dp_size_in_node = max(1, args.dp // args.nnodes)  # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
            self.shared_token_load = TokenLoad("shared_token_load", dp_size_in_node)


g_objs = G_Objs()

app = FastAPI()
g_objs.app = app

_ACCESS_LOG_STATUS_COLORS = {2: "\033[32m", 3: "\033[36m", 4: "\033[33m", 5: "\033[31m"}
_ACCESS_LOG_RESET = "\033[0m"


class _AccessLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["type"] == "http":
                status = status_holder["status"]
                msg = f"{scope['method']} {scope['path']} {status}"
                color = _ACCESS_LOG_STATUS_COLORS.get(status // 100, "")
                if color:
                    msg = color + msg + _ACCESS_LOG_RESET
                logger.info(msg)


app.add_middleware(_AccessLogMiddleware)


@app.exception_handler(ServerBusyError)
async def server_busy_exception_handler(request: Request, exc: ServerBusyError) -> JSONResponse:
    logger.warning("Server busy detail: %s", exc.message)

    # Streaming responses can raise during their first body iteration, after
    # the route handler has already returned. Preserve the Anthropic error
    # envelope for that deferred failure path as well.
    if request.url.path == "/v1/messages":
        from .api_anthropic import _anthropic_error_response

        g_objs.metric_client.counter_inc("lightllm_request_failure")
        return _anthropic_error_response(HTTPStatus(exc.status_code), SERVER_BUSY_MESSAGE)

    return create_server_busy_response(exc)


@app.exception_handler(InvalidRequestError)
async def invalid_request_exception_handler(request: Request, exc: InvalidRequestError) -> JSONResponse:
    if request.url.path == "/v1/messages":
        from .api_anthropic import _anthropic_error_response

        g_objs.metric_client.counter_inc("lightllm_request_failure")
        return _anthropic_error_response(HTTPStatus.BAD_REQUEST, str(exc))

    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    if g_objs.args.run_mode == "pd_master":
        pd_nodes_are_ready = g_objs.httpserver_manager.pd_manager.is_pd_nodes_ready()
        return JSONResponse(
            {"status": "ok" if pd_nodes_are_ready else "not ready"},
            status_code=200 if pd_nodes_are_ready else 503,
        )
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    return {"model_name": g_objs.args.model_name}


@app.get("/get_server_info")
@app.post("/get_server_info")
def get_server_info():
    if is_dataclass(g_objs.args):
        return asdict(g_objs.args)

    # HTTP workers restore StartArgs from the environment as an EasyDict.
    return dict(g_objs.args)


@app.get("/get_weight_version")
@app.post("/get_weight_version")
def get_weight_version():
    return {"weight_version": getattr(g_objs.httpserver_manager, "rl_active_policy_version", g_objs.args.weight_version)}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=503)

    if g_objs.args.run_mode == "pd_master":
        httpserver_manager = g_objs.httpserver_manager
        pd_manager = httpserver_manager.pd_manager
        if g_objs.args.pd_master_mode == "elastic":
            inference_is_healthy = httpserver_manager.is_healthy()
            pd_nodes_are_ready = pd_manager.is_pd_nodes_ready()
            is_healthy = inference_is_healthy and pd_nodes_are_ready
            health_info = {
                "inference_healthy": inference_is_healthy,
                "pd_nodes_ready": pd_nodes_are_ready,
            }
        else:
            inference_is_healthy = httpserver_manager.is_healthy()
            pd_nodes_are_ready = pd_manager.is_pd_nodes_ready()
            pd_nodes_are_healthy = (
                inference_is_healthy and pd_nodes_are_ready and await pd_manager.check_pd_nodes_health()
            )
            is_healthy = pd_nodes_are_healthy
            health_info = {
                "inference_healthy": inference_is_healthy,
                "pd_nodes_ready": pd_nodes_are_ready,
                "pd_nodes_healthy": pd_nodes_are_healthy,
            }

        health_info.update(
            {
                "message": "Ok" if is_healthy else "Error",
                "pd_master_mode": g_objs.args.pd_master_mode,
                "registered_prefill_nodes": len(pd_manager.prefill_nodes),
                "registered_decode_nodes": len(pd_manager.decode_nodes),
            }
        )
        return JSONResponse(health_info, status_code=200 if is_healthy else 503)

    from lightllm.utils.health_check import health_check

    is_healthy = health_check(g_objs.httpserver_manager.shm_req_manager)
    return JSONResponse(
        {"message": "Ok" if is_healthy else "Error"},
        status_code=200 if is_healthy else 503,
    )


@app.get("/token_load", summary="Get the current server's load of tokens")
async def token_load(request: Request):
    ans_dict = {
        # 当前使用 token 量，估计的负载
        "current_load": [
            float(g_objs.shared_token_load.get_current_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 朴素估计的负载，简单将当前请求的输入和输出长度想加得到,目前已未使用，其值与 dynamic_max_load 一样。
        "logical_max_load": [
            float(g_objs.shared_token_load.get_logical_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 动态估计的最大负载，考虑请求中途退出的情况的负载
        "dynamic_max_load": [
            float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
    }

    if g_objs.args.dp == 1:
        ans_dict = {k: v[0] for k, v in ans_dict.items()}

    return JSONResponse(ans_dict, status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_stream_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await lightllm_get_score(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        resp = await chat_completions_impl(request, raw_request)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        resp = await completions_impl(request, raw_request)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/generate_image")
async def generate_image(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    await g_objs.httpserver_manager.begin_generation_session(request)
    try:
        try:
            return await g_objs.g_generate_image_func(request, g_objs.httpserver_manager)
        except Exception as e:
            return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))
    finally:
        await g_objs.httpserver_manager.end_generation_session(request)


@app.post("/pause_generation")
async def pause_generation():
    await g_objs.httpserver_manager.pause_generation()
    return Response(content="Generation paused and drained.", status_code=200)


@app.post("/continue_generation")
async def continue_generation():
    await g_objs.httpserver_manager.continue_generation()
    return Response(content="Generation continued.", status_code=200)


@app.get("/v1/rl/status")
async def rl_status():
    return g_objs.httpserver_manager.rl_status()




@app.post("/init_weights_update_group")
async def init_weights_update_group(request: InitWeightsUpdateGroupRequest):
    try:
        return await g_objs.httpserver_manager.init_weights_update_group(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: DistributedWeightsRequest):
    try:
        return await g_objs.httpserver_manager.update_weights_from_distributed(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: TensorWeightsRequest):
    try:
        return await g_objs.httpserver_manager.update_weights_from_tensor(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/commit_weights_update")
async def commit_weights_update(request: CommitWeightsUpdateRequest):
    try:
        return await g_objs.httpserver_manager.commit_weights_update(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/destroy_weights_update_group")
async def destroy_weights_update_group(request: DestroyWeightsUpdateGroupRequest):
    try:
        return await g_objs.httpserver_manager.destroy_weights_update_group(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/rl/rollouts")
async def rl_rollouts(request: RLRolloutRequest, raw_request: Request):
    from .api_rl import rl_rollouts as impl

    await g_objs.httpserver_manager.begin_generation_session(raw_request)
    try:
        return await impl(request, raw_request, g_objs.httpserver_manager)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await g_objs.httpserver_manager.end_generation_session(raw_request)


def _trace_path(bundle_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", bundle_id):
        raise HTTPException(status_code=400, detail="invalid trace bundle id")
    path = os.path.join(
        os.getenv("MOVA_RL_TRACE_DIR", "/dev/shm/mova_rl_traces"),
        f"{bundle_id}.safetensors",
    )
    if os.path.isfile(path):
        ttl = int(os.getenv("MOVA_RL_TRACE_TTL", "3600"))
        if ttl <= 0:
            raise HTTPException(status_code=500, detail="invalid RL trace TTL")
        if os.path.getmtime(path) < time.time() - ttl:
            os.unlink(path)
    return path


@app.get("/v1/rl/traces/{bundle_id}")
async def get_rl_trace(bundle_id: str):
    path = _trace_path(bundle_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="trace bundle not found")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{bundle_id}.safetensors")


@app.delete("/v1/rl/traces/{bundle_id}")
async def delete_rl_trace(bundle_id: str):
    path = _trace_path(bundle_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="trace bundle not found")
    os.unlink(path)
    return {"deleted": bundle_id}


@app.websocket("/v1/rl/traces/ws")
async def stream_rl_traces(websocket: WebSocket):
    """Stream many SDE bundles as raw safetensors frames, then delete them.

    The JSON rollout response remains small and carries only bundle IDs.  A
    trainer opens one side-channel per rollout group, receives each bundle in
    bounded binary frames, and never performs one HTTP file download per image.
    The producer/consumer hand-off lives in ``/dev/shm`` by default, so neither
    side writes a persistent trace artifact.
    """

    await websocket.accept()
    pending: list[tuple[str, str]] = []
    try:
        request = await websocket.receive_json()
        bundle_ids = request.get("bundle_ids") if isinstance(request, dict) else None
        if (
            not isinstance(bundle_ids, list)
            or not bundle_ids
            or len(bundle_ids) > 256
            or len(set(bundle_ids)) != len(bundle_ids)
        ):
            raise ValueError("bundle_ids must contain 1..256 distinct trace IDs")
        for bundle_id in bundle_ids:
            if not isinstance(bundle_id, str):
                raise ValueError("trace bundle IDs must be strings")
            pending.append((bundle_id, _trace_path(bundle_id)))
        missing = [bundle_id for bundle_id, path in pending if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"trace bundles not found: {missing[:8]}")
        ttl_seconds = int(os.getenv("MOVA_RL_TRACE_TTL", "3600"))
        oldest_trace_age_seconds = max(
            time.time() - os.path.getmtime(path) for _bundle_id, path in pending
        )
        logger.info(
            json.dumps(
                {
                    "component": "lightllm.rl_trace_stream",
                    "event": "trace_age",
                    "bundle_count": len(pending),
                    "oldest_trace_age_seconds": oldest_trace_age_seconds,
                    "trace_ttl_seconds": ttl_seconds,
                }
            )
        )
        await websocket.send_json(
            {
                "schema": "mova.rl.sde_stream.v1",
                "bundle_count": len(pending),
                "oldest_trace_age_seconds": oldest_trace_age_seconds,
                "trace_ttl_seconds": ttl_seconds,
            }
        )
        chunk_bytes = 8 * 1024 * 1024
        for bundle_id, path in pending:
            size = os.path.getsize(path)
            if size <= 0:
                raise RuntimeError(f"trace bundle is empty: {bundle_id}")
            await websocket.send_json(
                {
                    "bundle_id": bundle_id,
                    "size": size,
                    "chunk_bytes": chunk_bytes,
                }
            )
            with open(path, "rb") as handle:
                sent = 0
                while block := handle.read(chunk_bytes):
                    await websocket.send_bytes(block)
                    sent += len(block)
            if sent != size:
                raise RuntimeError(f"trace bundle changed while streaming: {bundle_id}")
            os.unlink(path)
        pending.clear()
        await websocket.send_json({"complete": True})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        # A disconnected trainer must not strand high-frequency rollout state.
        for _bundle_id, path in pending:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def completions_v2(request: ChatCompletionRequestV2, raw_request: Request) -> Response:
    from lightllm.utils.config_utils import get_model_type_v1

    if get_model_type_v1() != "neo_chat":
        return await chat_completions(request, raw_request)
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    await g_objs.httpserver_manager.begin_generation_session(raw_request)
    try:
        resp = await chat_completions_impl_v2(request, raw_request)
    except ValueError as exc:
        await g_objs.httpserver_manager.end_generation_session(raw_request)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BaseException:
        await g_objs.httpserver_manager.end_generation_session(raw_request)
        raise
    if isinstance(resp, StreamingResponse):
        body_iterator = resp.body_iterator

        async def drain_session_after_stream():
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                await g_objs.httpserver_manager.end_generation_session(raw_request)

        resp.body_iterator = drain_session_after_stream()
        return resp
    await g_objs.httpserver_manager.end_generation_session(raw_request)
    return resp


@app.post("/v1/messages")
async def anthropic_messages(raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_anthropic import _anthropic_error_response, anthropic_messages_impl

    try:
        return await anthropic_messages_impl(raw_request)
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        g_objs.metric_client.counter_inc("lightllm_request_failure")
        return _anthropic_error_response(HTTPStatus(e.status_code), SERVER_BUSY_MESSAGE)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(raw_request: Request) -> Response:
    from .api_anthropic import _anthropic_error_response, anthropic_count_tokens_impl

    try:
        return await anthropic_count_tokens_impl(raw_request)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return _anthropic_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.post("/v1/responses")
async def openai_responses(raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_responses import responses_impl

    try:
        return await responses_impl(raw_request)
    except ServerBusyError as e:
        logger.warning("Server busy detail: %s", e.message)
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.get("/v1/models", response_model=ModelListResponse)
async def get_models(raw_request: Request):
    model_name = g_objs.args.model_name
    max_model_len = g_objs.httpserver_manager.get_real_supported_max_req_total_len()

    if model_name == "default_model_name" and g_objs.args.model_dir:
        model_name = os.path.basename(g_objs.args.model_dir.rstrip("/"))

    return ModelListResponse(
        data=[
            ModelCard(
                id=model_name,
                created=g_objs.model_created,
                max_model_len=max_model_len,
                owned_by=g_objs.args.model_owner or "lightllm",
            )
        ]
    )


@app.get("/tokens")
@app.post("/tokens")
async def tokens(request: Request):
    try:
        request_dict = await request.json()
        prompt = request_dict.pop("text")
        sample_params_dict = request_dict.pop("parameters", {})

        sampling_params = SamplingParams()
        sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sample_params_dict)
        sampling_params.verify()

        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)
        await multimodal_params.verify_and_preload(request)
        return JSONResponse(
            {
                "ntokens": g_objs.httpserver_manager.tokens(
                    prompt, multimodal_params, sampling_params, sample_params_dict
                )
            },
            status_code=200,
        )
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await g_objs.metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


# RL 控制面接口（abort / pause / flush / memory / weight update），见 api_http_rl.py
from .api_http_rl import router as rl_router

app.include_router(rl_router)

# PD 分离控制面接口（P/D 注册与 KV 状态上报），见 api_http_pd.py
from .api_http_pd import router as pd_router

app.include_router(pd_router)


@app.get("/profiler_start")
async def profiler_start() -> Response:
    if g_objs.args.enable_profiling:
        await g_objs.httpserver_manager.profiler_cmd("start")
        return JSONResponse({"status": "ok"})
    else:
        return JSONResponse({"message": "Profiling support not enabled"}, status_code=400)


@app.get("/profiler_stop")
async def profiler_stop() -> Response:
    if g_objs.args.enable_profiling:
        await g_objs.httpserver_manager.profiler_cmd("stop")
        return JSONResponse({"status": "ok"})
    else:
        return JSONResponse({"message": "Profiling support not enabled"}, status_code=400)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)
    logger.info("Graceful shutdown completed.")
    return


@app.on_event("startup")
async def startup_event():
    logger.info("server start up")
    loop = asyncio.get_event_loop()
    g_objs.set_args(get_env_start_args())
    loop.create_task(g_objs.httpserver_manager.handle_loop())
    logger.info(f"server start up ok, loop use is {asyncio.get_event_loop()}")
    return
