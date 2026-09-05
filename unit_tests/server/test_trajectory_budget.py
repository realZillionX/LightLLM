"""CPU tests execute the serving loop with a scripted model and real budget math."""
import ast
import asyncio
import importlib.util
import json
import math
import re
import sys
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

SERVER = Path(__file__).resolve().parents[2] / "lightllm/server"
spec = importlib.util.spec_from_file_location("trajectory_budget_test_impl", SERVER / "trajectory_budget.py")
budget = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = budget
spec.loader.exec_module(budget)


class Box(types.SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()


class Tokens:
    patch_size, downsample_ratio = 16, 0.5
    image_start_id = 1
    image_tag, image_start_tag, image_end_tag = "<image>", "<img>", "</img>"

    def encode(self, text, *_args, **_kwargs):
        return re.findall(r"<img>|</img>|.", text)

    def init_imageitem_extral_params(self, *_args):
        pass

    def get_image_token_length(self, _image):
        return 1

    def decode(self, token_ids, **kwargs):
        return "x" * len(token_ids)


class MM:
    def __init__(self):
        self.images, self.audios = [], []

    async def verify_and_preload(self, _request):
        pass

    def clone(self):
        return self

    def add_image(self, _image):
        self.images.append(Box(image_w=32, image_h=32))


class Sampler:
    def __init__(self):
        self.stop_sequences = Box(initialize=lambda *args: None)
        self.invalid_token_ids = Box(initialize=self.set_invalid)
        self.invalid = []

    def set_invalid(self, ids):
        self.invalid = ids

    def init(self, **kwargs):
        self.max_new_tokens = kwargs["max_new_tokens"]
        self.invalid = kwargs.get("invalid_token_ids", [])

    def verify(self):
        pass


class ImageParams:
    width, height = 32, 32

    def init_from_image_config(self, _config):
        pass

    def update_hw(self, *_args):
        pass


class Model:
    def __init__(self, limit, *, images=False, eos=True):
        self.args = Box(enable_multimodal_x2i=True)
        self.max_req_total_len = limit
        self.tokenizer = Tokens()
        self.images = images
        self.eos = eos
        self.draws = 0
        self.spans = []

    async def generate(self, prompt, sampling, mm, *, max_req_total_len, **kwargs):
        context = budget.prompt_token_count(self, prompt, mm, sampling)
        self.spans.append((context, sampling.max_new_tokens, max_req_total_len, list(sampling.invalid)))
        assert context + sampling.max_new_tokens == max_req_total_len
        if self.images and 1 not in sampling.invalid:
            yield 0, "<img>", {"id": 1, "logprob": -1.0, "prompt_tokens": context}, Box(
                is_finished=lambda: True, get_finish_reason=lambda: "stop")
        elif self.eos:
            yield 0, "", {"id": 2, "logprob": -1.0, "prompt_tokens": context}, Box(
                is_finished=lambda: True, get_finish_reason=lambda: "stop")
        else:
            for i in range(sampling.max_new_tokens):
                yield 0, "x", {"id": 3, "logprob": -1.0, "prompt_tokens": context}, Box(
                    is_finished=lambda i=i: i + 1 == sampling.max_new_tokens,
                    get_finish_reason=lambda: "length")

    async def generate_image(self, *_args, **_kwargs):
        self.draws += 1
        return ["image-bytes"]


def run_request(model, *, requested=None, text_only=False, prompt="PPPP", max_images=10, stream=False):
    async def input_for(_request):
        return prompt, Sampler(), MM()

    tree = ast.parse((SERVER / "api_openai.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "chat_completions_impl_v2")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(__package__="trajectory_route_test", uuid=uuid, time=time, json=json,
                 ChatCompletionRequest=Box, X2IParams=ImageParams, UsageInfo=Box,
                 ChatCompletionResponse=Box, ChatCompletionResponseChoice=Box, ChatMessage=Box,
                 StreamingResponse=lambda it, **kw: it,
                 _get_text_generator_input=input_for,
                 get_env_start_args=lambda: Box(reasoning_parser=None),
                 _normalize_image_b64_for_multimodal=lambda x: x,
                 _message_contents_from_raw_images=lambda images, kind: [Box(data=x) for x in images])
    exec(compile(module, str(SERVER / "api_openai.py"), "exec"), scope)
    request = Box(max_sequence_length=requested, max_images=max_images, modalities=["text"] if text_only else ["text", "image"],
                  image_config=None if text_only else Box(num_images=1, image_type="png"),
                  model_fields_set=set(), logit_bias=None, function_call="none", n=1,
                  chat_template_kwargs={}, stream=stream, model="test", stop=None, separate_reasoning=False)
    modules = {"trajectory_route_test.api_http": Box(g_objs=Box(httpserver_manager=model)),
               "trajectory_route_test.trajectory_budget": budget}

    async def run():
        response = await scope["chat_completions_impl_v2"](request, None)
        return [chunk async for chunk in response] if stream else response

    with patch.dict(sys.modules, modules):
        return asyncio.run(run())


class TrajectoryBudgetTest(unittest.TestCase):
    def test_input_and_output_share_one_limit(self):
        response = run_request(Model(20, eos=False), requested=20, text_only=True)
        self.assertEqual(response.usage.prompt_tokens, 4)
        self.assertEqual(response.usage.completion_tokens, 16)
        self.assertEqual(response.usage.total_tokens, 20)
        self.assertEqual(response.choices[0].finish_reason, "length")

    def test_ten_images_then_text_can_end_normally(self):
        model = Model(100, images=True)
        response = run_request(model)
        self.assertEqual(model.draws, 10)
        self.assertIn(1, model.spans[-1][-1])
        self.assertEqual(response.usage.total_tokens, 4 + 11 + 10 * 2)
        self.assertTrue(response.usage.image_limit_hit)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_image_reserve_prevents_overrun_and_releases_text_tail(self):
        model = Model(12, images=True, eos=False)
        response = run_request(model)
        self.assertEqual(model.draws, 2)
        self.assertEqual(response.usage.total_tokens, 12)
        self.assertEqual(response.usage.image_context_tokens, 4)
        self.assertEqual(response.choices[0].finish_reason, "length")

    def test_no_image_action_does_not_waste_reserved_context(self):
        model = Model(20, eos=False)
        response = run_request(model)
        self.assertEqual(response.usage.completion_tokens, 16)
        self.assertEqual(len(model.spans), 2)

    def test_rejects_an_independent_server_capacity(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            run_request(Model(100), requested=20)

    def test_input_without_generation_room_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no room"):
            run_request(Model(4), requested=4)

    def test_streaming_obeys_the_same_budget(self):
        chunks = run_request(Model(20, eos=False), text_only=True, stream=True)
        last = json.loads(chunks[-2].decode().removeprefix("data: "))
        self.assertEqual(last["usage"]["total_tokens"], 20)
        self.assertEqual(last["choices"][0]["finish_reason"], "length")

    def test_rl_text_uses_all_remaining_context(self):
        path = SERVER / "api_rl.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if getattr(n, "name", None) == "_one_rollout")
        module = ast.parse("from __future__ import annotations")
        module.body.append(fn)

        async def build_prompt(*args, **kwargs):
            return "PPPP"

        scope = dict(math=math, require_server_limit=budget.require_server_limit,
                     prompt_token_count=budget.prompt_token_count, span_budget=budget.span_budget,
                     _image_context_tokens=budget.image_context_tokens,
                     ChatCompletionRequestV2=Box, ChatCompletionRequest=Box,
                     _get_images_and_audios=lambda req: ([], []),
                     MultimodalParams=lambda **kw: MM(), build_prompt=build_prompt,
                     SamplingParams=Sampler)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        request = Box(max_sequence_length=20, modality="ti2t", messages=[], max_images=0)
        response = asyncio.run(scope["_one_rollout"](request, 42, None, Model(20, eos=False)))
        self.assertEqual(response["usage"]["sequence_tokens"], 20)
        self.assertEqual(response["usage"]["completion_tokens"], 16)
        self.assertEqual(response["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
