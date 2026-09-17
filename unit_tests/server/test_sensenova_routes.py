"""Route registration must not shadow interleaved generation or policy identity."""
import ast
import unittest
from pathlib import Path


class SenseNovaRoutesTest(unittest.TestCase):
    def test_public_paths_are_registered_once_per_http_method(self):
        source = Path(__file__).resolve().parents[2] / "lightllm/server/api_http.py"
        routes = {}
        for node in ast.parse(source.read_text()).body:
            for decorator in getattr(node, "decorator_list", []):
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                    continue
                route = (decorator.func.attr, decorator.args[0].value)
                self.assertNotIn(route, routes, f"route {route} is shadowed by {node.name}")
                routes[route] = node.name
        self.assertEqual(routes[("post", "/v1/chat/completions")], "completions_v2")
        self.assertEqual(routes[("get", "/get_weight_version")], "get_weight_version")
