import asyncio
from types import SimpleNamespace

import pytest

from lightllm.server.httpserver.manager import HttpServerManager


def test_request_limit_clamps_completion_without_changing_server_capacity():
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

    assert observed == prompt
    assert sampling.max_new_tokens == 5192
    assert manager.max_req_total_len == 16384


def test_request_limit_cannot_exceed_server_capacity():
    manager = object.__new__(HttpServerManager)
    manager.max_req_total_len = 8192
    manager.args = SimpleNamespace(long_truncation_mode=None)
    sampling = SimpleNamespace(max_new_tokens=1)

    with pytest.raises(ValueError, match="exceeds server limit"):
        asyncio.run(
            manager._check_and_repair_length(
                [1],
                sampling,
                max_req_total_len=8193,
            )
        )
