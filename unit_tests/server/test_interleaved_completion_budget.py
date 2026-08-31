from types import SimpleNamespace

from lightllm.server.api_openai import _set_interleaved_completion_budget


def test_interleaved_completion_budget_is_request_wide() -> None:
    sampling_params = SimpleNamespace(max_new_tokens=8192)

    assert _set_interleaved_completion_budget(
        sampling_params,
        total_completion_tokens=8192,
        used_completion_tokens=0,
    )
    assert sampling_params.max_new_tokens == 8192

    assert _set_interleaved_completion_budget(
        sampling_params,
        total_completion_tokens=8192,
        used_completion_tokens=3072,
    )
    assert sampling_params.max_new_tokens == 5120

    assert not _set_interleaved_completion_budget(
        sampling_params,
        total_completion_tokens=8192,
        used_completion_tokens=8192,
    )
    assert not _set_interleaved_completion_budget(
        sampling_params,
        total_completion_tokens=8192,
        used_completion_tokens=9000,
    )
