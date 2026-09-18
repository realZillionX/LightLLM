"""Template thinking policy is independent of response-channel parsing."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class PromptThinkingPolicyTest(unittest.TestCase):
    def test_preserve_explicit_policy_and_parserless_template_defaults(self):
        source = Path(__file__).resolve().parents[2] / "lightllm/server/build_prompt.py"
        method = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.AsyncFunctionDef) and node.name == "build_prompt")
        for parser in (None, "qwen3"):
            for explicit in (None, True, False):
                with self.subTest(parser=parser, explicit=explicit):
                    captured = {}

                    def template(**kwargs):
                        captured.update(kwargs)
                        return "assistant" if kwargs.get("enable_thinking", True) else "assistant<think></think>"

                    namespace = dict(
                        __package__="lightllm.server", tokenizer=SimpleNamespace(apply_chat_template=template),
                        _normalize_tool_call_arguments=lambda _: None, _flatten_multimodal_content=lambda _: None,
                        _alias_reasoning_to_reasoning_content=lambda _: None,
                        get_model_type_v1=lambda: "neo_chat",
                        get_env_start_args=lambda: SimpleNamespace(reasoning_parser=parser),
                    )
                    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
                    request = SimpleNamespace(messages=[], character_settings=None, role_settings=None,
                                              reasoning_effort=None, chat_template_kwargs={} if explicit is None else {"enable_thinking": explicit})
                    api = SimpleNamespace(_is_force_thinking_mode=lambda _: bool(parser))
                    with patch.dict("sys.modules", {"lightllm.server.api_openai": api}):
                        actual = asyncio.run(namespace["build_prompt"](request, None))
                    self.assertEqual(actual, "assistant<think></think>" if explicit is False else "assistant")
                    if parser is None and explicit is None:
                        self.assertNotIn("enable_thinking", captured)
                        self.assertNotIn("thinking", captured)


if __name__ == "__main__":
    unittest.main()
