"""The KV pinning API completes synchronously and returns a device pointer."""
import ast
from pathlib import Path
from threading import Condition, Lock
from types import SimpleNamespace
import unittest


class PastKvAttachTest(unittest.TestCase):
    def test_constructor_does_not_wait_on_the_returned_pointer(self):
        source = Path(__file__).resolve().parents[2] / "lightllm/server/x2i_server/past_kv_cache_client.py"
        method = next(node for node in ast.walk(ast.parse(source.read_text()))
                      if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        called = []
        namespace = dict(get_env_start_args=lambda: SimpleNamespace(),
                         calcu_cpu_cache_meta=lambda: SimpleNamespace(page_num=2, token_page_size=512),
                         Lock=Lock, Condition=Condition, List=list)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        instance = SimpleNamespace(_attach_shm_cpu_kv_cache=lambda: called.append(True) or 1234)
        namespace["__init__"](instance, only_create_meta_data=False, init_shm_data=False)
        self.assertEqual(called, [True])
        self.assertEqual(instance.free_pages, [0, 1])


if __name__ == "__main__":
    unittest.main()
