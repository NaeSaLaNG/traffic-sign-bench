"""Small boundary tests for the crosswalk operator command and LLM feedback."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from traffic_bench.policy_lab.agent import EvalConfig, SandboxTools
from traffic_bench.policy_lab.gauntlet import Entry
from traffic_bench.policy_lab.hooks import HarnessHooks, public_feedback
from traffic_bench.policy_lab.media import convert_gifs
from traffic_bench.policy_lab.run_crosswalk import main, parse_args, server_model, write_gif_index
from traffic_bench.policy_lab.sandbox_env import sandbox_env


class FeedbackBoundaryTests(unittest.TestCase):
    def test_only_numeric_aggregates_reach_builder_and_critic(self):
        private = "SECRET_SCENE_AND_TRACEBACK"
        raw = {
            "overall": {"episodes": 2, "clean_success_rate": 0.5, "leak": private},
            "by_family": {"crosswalk": {"episodes": 2, "collision_rate": 0.5,
                                         "diagnostic": private},
                          private: {"episodes": 1}},
            "policy_fault_examples": [private], "judge_log": private,
        }
        expected = {"overall": {"episodes": 2, "clean_success_rate": 0.5},
                    "by_family": {"crosswalk": {"episodes": 2, "collision_rate": 0.5}}}
        self.assertEqual(public_feedback(raw), expected)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            (sandbox / "policy.py").write_text("class Policy: pass\n")
            cfg = EvalConfig("python", "manifest.jsonl", 2, 3, "cpu")
            tools = SandboxTools(sandbox, root / "run", root / "checkpoint", cfg, hooks=False)
            tools.run_judge = lambda _: (root / "eval_001", raw, None)
            result = tools.evaluate("policy.py")
            self.assertEqual(json.loads(result.split("\n", 1)[1]), expected)
            entry = Entry("policy.py", root / "eval_001", raw)
            self.assertEqual(entry.public_metrics(), expected)
            self.assertNotIn(private, result)
            tools.run_judge = lambda _: (_ for _ in ()).throw(RuntimeError(private))
            self.assertNotIn(private, tools.evaluate("policy.py"))
            self.assertIn(private, (root / "run/evaluation_errors.log").read_text())

    def test_input_hooks_block_escapes_and_forbidden_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            hooks = HarnessHooks(sandbox, root / "run")
            self.assertIn("blocked", hooks.before("run", {"command": "cat ../secret.txt"}))
            self.assertIn("blocked", hooks.before("write_file", {
                "path": "policies/bad.py", "content": "import socket\n"}))
            self.assertFalse((sandbox / "policies/bad.py").exists())
            self.assertTrue((root / "run/hooks.jsonl").is_file())


class ClusterCommandTests(unittest.TestCase):
    def test_split_and_budget_in_dry_run(self):
        for split, manifest_split in (("valid", "train"), ("test", "test")):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    main(["--split", split, "--iterations", "2", "--work-dir", tmp,
                          "--model", "local-qwen", "--dry-run"])
                printed = output.getvalue()
                self.assertIn(f"/crosswalk/{manifest_split}/real_manifest.jsonl", printed)
                self.assertIn("--max-evals 3", printed)
                self.assertIn("--max-rounds 2", printed)
                self.assertIn("--save-gifs", printed)
                self.assertFalse(list(Path(tmp).iterdir()))

    def test_iterations_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--iterations", "0"])

    def test_sandbox_environment_has_private_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = sandbox_env(root, root / "checkpoint")
            self.assertEqual(env["HOME"], str(root.resolve() / ".home"))
            self.assertEqual(env["TMPDIR"], str(root.resolve() / ".tmp"))
            self.assertNotIn("POLICY_LAB_LLM_API_KEY", env)

    def test_gif_gallery_links_to_private_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            gifs = run_dir / "eval_001/gifs"
            gifs.mkdir(parents=True)
            (gifs / "scene.gif").write_bytes(b"GIF89a")
            (gifs / "scene.mp4").write_bytes(b"mp4")
            page = write_gif_index(run_dir).read_text()
            self.assertIn("eval_001/gifs/scene.gif", page)
            self.assertIn("eval_001/gifs/scene.mp4", page)
            self.assertIn("<video", page)

    def test_gifs_are_converted_beside_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            gif = Path(tmp) / "scene.gif"
            gif.write_bytes(b"GIF89a")

            def fake_ffmpeg(command, **kwargs):
                Path(command[-1]).write_bytes(b"mp4")

            with mock.patch("traffic_bench.policy_lab.media.shutil.which", return_value="/usr/bin/ffmpeg"), \
                 mock.patch("traffic_bench.policy_lab.media.subprocess.run", side_effect=fake_ffmpeg):
                count, errors = convert_gifs(Path(tmp))
            self.assertEqual((count, errors), (1, []))
            self.assertTrue(gif.is_file())
            self.assertEqual(gif.with_suffix(".mp4").read_bytes(), b"mp4")

    def test_server_model_auto_detection(self):
        response = io.BytesIO(b'{"data":[{"id":"qwen-local"}]}')
        with mock.patch("traffic_bench.policy_lab.run_crosswalk.urllib.request.urlopen",
                        return_value=contextlib.nullcontext(response)):
            self.assertEqual(server_model("http://localhost:8000/v1", None), "qwen-local")


if __name__ == "__main__":
    unittest.main()
