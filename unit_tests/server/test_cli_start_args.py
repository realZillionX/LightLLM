"""Every public CLI configuration must construct the server's typed arguments."""
import argparse
import runpy
import unittest
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "lightllm/server"


class CliStartArgsTest(unittest.TestCase):
    def test_default_and_sensenova_cli_construct_start_args(self):
        add_cli_args = runpy.run_path(str(SERVER / "api_cli.py"))["add_cli_args"]
        start_args = runpy.run_path(str(SERVER / "core/objs/start_args_type.py"))["StartArgs"]
        for mode in (None, "separate", "colocate"):
            parser = add_cli_args(argparse.ArgumentParser())
            argv = [] if mode is None else [
                "--sensenova_modality", "ti2ti", "--enable_multimodal_x2i",
                "--x2i_server_deploy_mode", mode, "--x2v_gen_model_config", "/profiles/cfg0.json",
            ]
            value = start_args(**vars(parser.parse_args(argv)))
            self.assertEqual(value.x2v_gen_model_config, None if mode is None else "/profiles/cfg0.json")


if __name__ == "__main__":
    unittest.main()
