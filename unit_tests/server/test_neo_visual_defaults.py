"""Exercise NeoPP preprocessing for synthetic warmup and regular request images."""
import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest


class NeoVisualDefaultsTest(unittest.TestCase):
    def test_shm_probe_token_count_uses_the_same_pixel_defaults(self):
        source = Path(__file__).resolve().parents[2] / "lightllm/models/neo_chat_moe/model.py"
        method = next(node for node in ast.walk(ast.parse(source.read_text()))
                      if isinstance(node, ast.FunctionDef) and node.name == "get_image_token_length")
        captured = []

        def resize(**kwargs):
            captured.append(kwargs)
            return 64, 64

        namespace = dict(ImageItem=SimpleNamespace, smart_resize=resize)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        model = SimpleNamespace(patch_size=16, downsample_ratio=0.5, min_pixel=65536, max_pixel=2408448)
        for extra, expected in [({}, (65536, 2408448)), ({"min_pixels": 262144, "max_pixels": 1048576}, (262144, 1048576))]:
            item = SimpleNamespace(image_w=64, image_h=64, extra_params=extra)
            self.assertEqual(namespace["get_image_token_length"](model, item), 4)
            self.assertEqual((captured[-1]["min_pixels"], captured[-1]["max_pixels"]), expected)

    def test_warmup_uses_model_pixels_and_requests_keep_overrides(self):
        source = Path(__file__).resolve().parents[2] / "lightllm/models/neo_chat_moe/neo_visual.py"
        method = next(node for node in ast.walk(ast.parse(source.read_text()))
                      if isinstance(node, ast.FunctionDef) and node.name == "encode")
        captured = []

        class Preprocessed(Exception):
            pass

        def preprocess(image, **kwargs):
            captured.append(kwargs)
            raise Preprocessed

        namespace = dict(List=list, ImageItem=SimpleNamespace, BytesIO=BytesIO,
                         Image=SimpleNamespace(open=lambda _: "image"), read_shm=lambda _: b"image",
                         get_shm_name_data=lambda value: value, load_image_native=preprocess)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        model = SimpleNamespace(patch_size=16, downsample_ratio=0.5, min_pixels=65536, max_pixels=2408448)
        for extra, expected in [({}, (65536, 2408448)), ({"min_pixels": 262144, "max_pixels": 1048576}, (262144, 1048576))]:
            with self.assertRaises(Preprocessed):
                namespace["encode"](model, [SimpleNamespace(uuid="warmup", extra_params=extra)])
            self.assertEqual((captured[-1]["min_pixels"], captured[-1]["max_pixels"]), expected)


if __name__ == "__main__":
    unittest.main()
