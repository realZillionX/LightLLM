import zmq
import asyncio
import uvloop
import inspect
import setproctitle
import pickle
import torch
import time
import multiprocessing as mp
import os
from typing import List
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease, CfgNormType
from lightllm.utils.dist_utils import set_current_device_id
from lightllm.utils.start_utils import start_submodule_processes
from .past_kv_cache_client import PastKVCacheClient
from lightllm.server.core.objs.io_objs import RLControlRequest, RLControlResponse
from lightllm.utils.rl_weight_update import DistributedWeightReceiver

logger = init_logger(__name__)


"""
manage a generation service,
1. start x2v pipelines
2. receive generation request from http_server.
3. call llm gen to obtain past key values
4. call x2v to generate images and pass the key values to it
5. return the generated images.
"""


class X2IManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        context = zmq.Context(2)
        self.args = args

        # from http server
        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_port}")

        # to http server
        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"{args.zmq_mode}127.0.0.1:{args.http_server_port_for_x2i}")
        self.send_rl_control_response = context.socket(zmq.PUSH)
        self.send_rl_control_response.connect(
            f"{args.zmq_mode}127.0.0.1:{args.rl_control_response_port}"
        )

        self.use_naive_x2i = args.x2i_use_naive_impl
        self.world_size = args.x2i_server_used_gpus

        if not self.use_naive_x2i and self.world_size > 1:
            # send to workers
            self.worker_pub = context.socket(zmq.PUSH)
            self.worker_pub.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_worker_task_port}")

        self.waiting_reqs: List[X2IParams] = []

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=True)

    async def wait_to_model_ready(self):

        if self.world_size <= 1:
            if self.use_naive_x2i:
                from lightllm.server.x2i_server.naive.modeling_neo_chat import NEOX2I

                self.naive_x2i = NEOX2I(self.args.model_dir, torch.cuda.current_device())
            else:
                from lightx2v import LightX2VPipeline

                self.gen_pipe = LightX2VPipeline(
                    model_path=self.args.model_dir,
                    model_cls="neopp",
                    support_tasks=["t2i", "i2i"],
                )
                self.gen_pipe.create_generator(
                    config_json=self.args.x2v_gen_model_config,
                )
                self.gen_pipe.modify_config(
                    {"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": False}
                )
                from lightx2v.rl.trace_store import TraceStore

                self.rl_trace_store = TraceStore(
                    root=os.getenv("MOVA_RL_TRACE_DIR", "/dev/shm/mova_rl_traces"),
                    ttl_seconds=int(os.getenv("MOVA_RL_TRACE_TTL", "3600")),
                )
                self.rl_weight_receiver = DistributedWeightReceiver(
                    consumer="x2v", device=torch.device(f"cuda:{torch.cuda.current_device()}")
                )
        else:
            # distribted x2v
            from lightllm.server.x2i_server.lightx2v.adapter import start_x2v_process

            funcs = [start_x2v_process] * self.world_size
            args = [(self.args, rank, self.world_size) for rank in range(self.world_size)]
            start_submodule_processes(funcs, args)

    async def t2i_generate(self, past_kv_cache, past_kv_cache_text, param: X2IParams):
        if self.use_naive_x2i:
            images = self.naive_x2i.t2i(past_kv_cache, past_kv_cache_text, param)
            return images, []

        self.gen_pipe.runner.set_inference_params(
            index_offset_cond=param.past_kvcache.get_compressed_len(),
            index_offset_uncond=param.past_kvcache_text.get_compressed_len(),
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
        )
        images = []
        trace_bundles = []
        for i in range(param.num_images):
            self.gen_pipe.runner.set_kvcache(past_kv_cache, past_kv_cache_text)
            if hasattr(param, "rl_config"):
                self.gen_pipe.runner.scheduler.infer_steps = param.steps
                # X2IParams crosses shared memory with a signed C integer seed,
                # while NumPy's legacy RNG (used by LightX2V seed_all) accepts
                # only uint32 values. Preserve the seed's bit pattern so half
                # of concurrent RL image actions do not fail nondeterministically.
                rl_seed = (int(param.seed) & 0xFFFFFFFF) if param.first_image else None
                image, trace = self.gen_pipe.generate_rl(
                    rl_config=param.rl_config,
                    seed=rl_seed,
                    save_result_path="",
                    target_shape=[param.height, param.width],
                )
                trace_bundles.append(
                    self.rl_trace_store.put(
                        trace,
                        {"request_id": str(param.request_id), "image_index": str(i)},
                    )
                )
            else:
                image = self.gen_pipe.generate(
                    seed=param.seed if param.first_image else None,
                    save_result_path="",  # 返回base64，不需要指定路径了
                    target_shape=[param.height, param.width],  # Height, Width
                )
            images.append(image)
        return images, trace_bundles

    async def it2i_generate(self, past_kv_cache, past_kv_cache_text, past_kv_cache_img, param: X2IParams):
        if self.use_naive_x2i:
            images = self.naive_x2i.it2i(past_kv_cache, past_kv_cache_text, past_kv_cache_img, param)
            return images, []

        self.gen_pipe.runner.set_inference_params(
            index_offset_cond=param.past_kvcache.get_compressed_len(),
            index_offset_uncond=param.past_kvcache_text.get_compressed_len(),
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
        )
        images = []
        trace_bundles = []
        for i in range(param.num_images):
            self.gen_pipe.runner.set_kvcache_i2i(past_kv_cache, past_kv_cache_text, past_kv_cache_img)
            if hasattr(param, "rl_config"):
                self.gen_pipe.runner.scheduler.infer_steps = param.steps
                rl_seed = int(param.seed + param.past_kvcache_img.img_len + i) & 0xFFFFFFFF
                image, trace = self.gen_pipe.generate_rl(
                    rl_config=param.rl_config,
                    seed=rl_seed,
                    save_result_path="",
                    target_shape=[param.height, param.width],
                )
                trace_bundles.append(
                    self.rl_trace_store.put(
                        trace,
                        {"request_id": str(param.request_id), "image_index": str(i)},
                    )
                )
            else:
                image = self.gen_pipe.generate(
                    seed=param.seed + param.past_kvcache_img.img_len + i,
                    save_result_path="",  # 返回base64，不需要指定路径了
                    target_shape=[param.height, param.width],  # Height, Width
                )
            images.append(image)
        return images, trace_bundles

    async def loop_for_fwd(self):
        while True:
            try:
                if len(self.waiting_reqs) == 0:
                    await asyncio.sleep(0.01)
                    continue

                x2i_param = self.waiting_reqs.pop(0)

                if not self.use_naive_x2i and self.world_size > 1:
                    # broadcast to workers
                    self.worker_pub.send_pyobj(x2i_param, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache.get_all(), x2i_param.past_kvcache.token_len, self.use_naive_x2i
                    )

                    past_kv_cache_text = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache_text.get_all(), x2i_param.past_kvcache_text.token_len, self.use_naive_x2i
                    )
                    is_t2i = x2i_param.past_kvcache_img.is_empty()

                    past_kv_cache_img = None
                    if not is_t2i:  # t2i
                        past_kv_cache_img = self.past_kv_cache_client.get_kv_cache_for_x2i(
                            x2i_param.past_kvcache_img.get_all(),
                            x2i_param.past_kvcache_img.token_len,
                            self.use_naive_x2i,
                        )

                    # release
                    self.send_to_httpserver.send_pyobj(
                        X2ICacheRelease(request_id=x2i_param.request_id), protocol=pickle.HIGHEST_PROTOCOL
                    )

                    images = []
                    trace_bundles = []
                    logger.info(f"{'t2i' if is_t2i else 'it2i'} generate images with: {x2i_param}")
                    start_t = time.time()
                    if is_t2i:
                        images, trace_bundles = await self.t2i_generate(past_kv_cache, past_kv_cache_text, x2i_param)
                    else:
                        images, trace_bundles = await self.it2i_generate(
                            past_kv_cache, past_kv_cache_text, past_kv_cache_img, x2i_param
                        )
                    logger.info(f"generate {len(images)} images done, cost {time.time() - start_t:.2f}s")

                    self.send_to_httpserver.send_pyobj(
                        X2IResponse(request_id=x2i_param.request_id, images=images, trace_bundles=trace_bundles), protocol=pickle.HIGHEST_PROTOCOL
                    )

            except Exception as e:
                self.send_to_httpserver.send_pyobj(
                    X2IResponse(request_id=x2i_param.request_id, images=None), protocol=pickle.HIGHEST_PROTOCOL
                )
                logger.error(e, exc_info=e)

    async def loop_for_netio_req(self):
        while True:
            try:
                recv_req = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(recv_req, RLControlRequest):
                    await self.handle_rl_control(recv_req)
                else:
                    self.waiting_reqs.append(recv_req)

            except zmq.ZMQError:
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.01)

    async def handle_rl_control(self, request: RLControlRequest):
        try:
            if self.world_size != 1 or self.use_naive_x2i:
                raise RuntimeError("RL weight updates require one-GPU LightX2V separate mode")
            payload = dict(request.payload)
            payload["group_name"] = f"{payload.get('group_name', 'weight_update_group')}:x2v"
            if request.operation == "init_weights_update_group":
                payload["master_port"] = payload["master_ports"]["x2v"]
                payload["world_size"] = payload["x2v_world_size"]
                payload["rank_base"] = int(payload.get("x2v_rank_base", 1))
            if request.operation == "init_weights_update_group":
                data = self.rl_weight_receiver.init_group(payload, rank=payload["rank_base"])
                closure = self.gen_pipe.rl_weight_closure()
                data["closure_names"] = sorted(closure)
                data["closure_specs"] = closure
            elif request.operation == "destroy_weights_update_group":
                data = self.rl_weight_receiver.destroy_group(payload.get("group_name", "weight_update_group"))
            elif request.operation == "update_weights_from_distributed":
                tensors, data = self.rl_weight_receiver.receive(payload)
                closure = set(self.gen_pipe.rl_weight_closure())
                if payload.get("full_update", False) and set(tensors) != closure:
                    raise ValueError(
                        "X2V parameter closure mismatch: "
                        f"missing={sorted(closure - set(tensors))[:5]}, "
                        f"unexpected={sorted(set(tensors) - closure)[:5]}"
                    )
                data["apply"] = self.gen_pipe.update_rl_weights(tensors, strict=False)
                data["policy_version"] = payload["policy_version"]
            elif request.operation == "update_weights_from_tensor":
                tensors, data = self.rl_weight_receiver.decode_bundle(payload)
                closure = set(self.gen_pipe.rl_weight_closure())
                if payload.get("full_update", False) and set(tensors) != closure:
                    raise ValueError(
                        "X2V parameter closure mismatch: "
                        f"missing={sorted(closure - set(tensors))[:5]}, "
                        f"unexpected={sorted(set(tensors) - closure)[:5]}"
                    )
                data["apply"] = self.gen_pipe.update_rl_weights(tensors, strict=False)
                data["policy_version"] = payload["policy_version"]
            else:
                raise ValueError(f"unsupported X2V RL operation: {request.operation}")
            response = RLControlResponse(request.op_id, "x2v", True, data=data)
        except Exception as exc:
            logger.exception("X2V RL control failed")
            response = RLControlResponse(request.op_id, "x2v", False, message=str(exc))
        self.send_rl_control_response.send_pyobj(response, protocol=pickle.HIGHEST_PROTOCOL)

    def clean_up(self):
        pass


def setup_devices(args: StartArgs):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    logger.info(f"current devices: {devices} {torch.cuda.device_count()}")
    if not devices:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [int(x.strip()) for x in devices.split(",") if x.strip()]

    llm_need_gpus = 0 if args.x2i_server_deploy_mode == "colocate" else args.tp * args.dp
    x2i_need_gpus = args.x2i_server_used_gpus
    if len(devices) < llm_need_gpus + x2i_need_gpus:
        raise ValueError(f"devices {devices} not enough, need {llm_need_gpus} and {x2i_need_gpus}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices[llm_need_gpus : llm_need_gpus + x2i_need_gpus]))

    logger.info(
        f"setup devices for x2i server: {os.environ['CUDA_VISIBLE_DEVICES']}, "
        f"{torch.cuda.device_count()} {torch.cuda.current_device()}"
    )


def start_x2i_process(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2i_server")
    start_parent_check_thread()
    set_current_device_id(torch.cuda.current_device())
    try:
        x2iserver = X2IManager(
            args=args,
        )
        asyncio.run(x2iserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        x2iserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"X2IServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(x2iserver.loop_for_fwd())
    loop.run_until_complete(x2iserver.loop_for_netio_req())
    return
