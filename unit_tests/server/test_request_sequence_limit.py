import asyncio
import unittest
from types import SimpleNamespace

import ast
from pathlib import Path

# Exercise the real admission methods without importing CUDA inference workers.
source = Path(__file__).resolve().parents[2] / "lightllm/server/httpserver/manager.py"
manager = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "HttpServerManager")
methods = [node for node in manager.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name in {"get_real_supported_max_req_total_len", "_check_and_repair_length"}]
namespace = {"InvalidRequestError": ValueError}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="HttpServerManager", bases=[], keywords=[],
                                         body=methods, decorator_list=[])], type_ignores=[])), str(source), "exec"), namespace)
HttpServerManager = namespace["HttpServerManager"]


class RequestSequenceLimitTest(unittest.TestCase):
    def test_request_limit_clamps_completion_without_changing_server_capacity(self):
        manager = object.__new__(HttpServerManager)
        manager.max_req_total_len = 16384
        manager.shm_max_total_token_num = SimpleNamespace(get_value=lambda: manager.max_req_total_len + 36)
        sampling = SimpleNamespace(max_new_tokens=6144)

        prompt = list(range(3000))
        observed = asyncio.run(
            manager._check_and_repair_length(
                prompt,
                sampling,
                max_req_total_len=8192,
            )
        )

        self.assertEqual(observed, prompt)
        self.assertEqual(sampling.max_new_tokens, 5192)
        self.assertEqual(manager.max_req_total_len, 16384)

    def test_request_limit_cannot_exceed_server_capacity(self):
        manager = object.__new__(HttpServerManager)
        manager.max_req_total_len = 8192
        manager.shm_max_total_token_num = SimpleNamespace(get_value=lambda: manager.max_req_total_len + 36)
        sampling = SimpleNamespace(max_new_tokens=1)

        with self.assertRaisesRegex(ValueError, "exceeds server limit"):
            asyncio.run(
                manager._check_and_repair_length(
                    [1],
                    sampling,
                    max_req_total_len=8193,
                )
            )

    def test_request_limit_also_respects_available_kv_capacity(self):
        manager = object.__new__(HttpServerManager)
        manager.max_req_total_len = 8192
        manager.shm_max_total_token_num = SimpleNamespace(get_value=lambda: 4132)
        sampling = SimpleNamespace(max_new_tokens=8192)
        asyncio.run(manager._check_and_repair_length(list(range(3000)), sampling, max_req_total_len=8192))
        self.assertEqual(sampling.max_new_tokens, 1096)


if __name__ == "__main__":
    unittest.main()
