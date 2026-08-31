import asyncio
import unittest
from types import SimpleNamespace

from lightllm.server.httpserver.manager import HttpServerManager


class RequestSequenceLimitTest(unittest.TestCase):
    def test_request_limit_clamps_completion_without_changing_server_capacity(self):
        manager = object.__new__(HttpServerManager)
        manager.max_req_total_len = 16384
        manager.args = SimpleNamespace(long_truncation_mode=None)
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
        manager.args = SimpleNamespace(long_truncation_mode=None)
        sampling = SimpleNamespace(max_new_tokens=1)

        with self.assertRaisesRegex(ValueError, "exceeds server limit"):
            asyncio.run(
                manager._check_and_repair_length(
                    [1],
                    sampling,
                    max_req_total_len=8193,
                )
            )


if __name__ == "__main__":
    unittest.main()
