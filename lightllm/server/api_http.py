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
import pickle
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
from lightllm.utils.envs_utils import get_env_start_args, get_lightllm_websocket_max_message_size
from lightllm.utils.log_utils import init_logger
from lightllm.utils.error_utils import ServerBusyError
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from dataclasses import dataclass

from .api_openai import chat_completions_impl, completions_impl, chat_completions_impl_v2
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

        if args.run_mode == "pd_master":
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManagerForPDMaster(
                args=args,
            )
        else:
            init_tokenizer(args)  # for openai api
            SamplingParams.load_generation_cfg(args.model_dir)
            CompletionRequest.load_generation_cfg(args.model_dir)
            ChatCompletionRequest.load_generation_cfg(args.model_dir)
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManager(args=args)
            dp_size_in_node = max(1, args.dp // args.nnodes)  # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
            self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", dp_size_in_node)
            if self.model_created is None:
                self.model_created = int(time.time())


g_objs = G_Objs()

app = FastAPI()
g_objs.app = app


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    g_objs.metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    return {"model_name": g_objs.args.model_name}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    if g_objs.args.run_mode == "pd_master":
        return JSONResponse({"message": "Ok"}, status_code=200)

    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=503)
    from lightllm.utils.health_check import health_check, health_obj

    health_task = asyncio.create_task(health_check(g_objs.args, g_objs.httpserver_manager, None))
    if not health_obj.is_health():
        await health_task
    return JSONResponse(
        {"message": "Ok" if health_obj.is_health() else "Error"}, status_code=200 if health_obj.is_health() else 503
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
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.error("%s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_stream_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.error("%s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await lightllm_get_score(request, g_objs.httpserver_manager)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


# @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
#     if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
#         return create_error_response(
#             HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
#         )

#     resp = await chat_completions_impl(request, raw_request)
#     return resp


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    resp = await completions_impl(request, raw_request)
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


@app.get("/get_weight_version")
async def get_weight_version():
    return {"weight_version": g_objs.httpserver_manager.rl_active_policy_version}


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
    if get_env_start_args().run_mode in ["prefill", "decode", "nixl_prefill", "nixl_decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_anthropic import anthropic_messages_impl

    return await anthropic_messages_impl(raw_request)


@app.get("/v1/models", response_model=ModelListResponse)
@app.post("/v1/models", response_model=ModelListResponse)
async def get_models(raw_request: Request):
    model_name = g_objs.args.model_name
    max_model_len = g_objs.args.max_req_total_len
    if model_name == "default_model_name" and g_objs.args.model_dir:
        model_name = os.path.basename(g_objs.args.model_dir.rstrip("/"))

    return ModelListResponse(
        data=[
            ModelCard(
                id=model_name,
                created=g_objs.model_created,
                max_model_len=max_model_len,
                owned_by=g_objs.args.model_owner,
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
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await g_objs.metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


@app.websocket("/pd_register")
async def register_and_keep_alive(websocket: WebSocket):
    await websocket.accept()
    websocket._receive_bytes_max_size = get_lightllm_websocket_max_message_size()
    client_ip, client_port = websocket.client
    logger.info(f"Client connected from IP: {client_ip}, Port: {client_port}")
    regist_json = json.loads(await websocket.receive_text())
    logger.info(f"received regist_json {regist_json}")
    await g_objs.httpserver_manager.register_pd(regist_json, websocket)

    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_bytes()
            obj = pickle.loads(data)
            await g_objs.httpserver_manager.put_to_handle_queue(obj)

    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"client {regist_json} has error {str(e)}")
        logger.exception(str(e))
    finally:
        logger.error(f"client {regist_json} removed")
        await g_objs.httpserver_manager.remove_pd(regist_json)
    return


@app.websocket("/kv_move_status")
async def kv_move_status(websocket: WebSocket):
    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"kv_move_status Client connected from IP: {client_ip}, Port: {client_port}")
    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_bytes()
            upkv_status = pickle.loads(data)
            logger.info(f"received upkv_status {upkv_status} from {(client_ip, client_port)}")
            await g_objs.httpserver_manager.update_req_status(upkv_status)
    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"kv_move_status client {(client_ip, client_port)} has error {str(e)}")
        logger.exception(str(e))
    return


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
