import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_batch_module():
    script_path = SCRIPTS_DIR / "eval_batch_binary_qwen_omni.py"
    spec = importlib.util.spec_from_file_location("eval_batch_binary_qwen_omni", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalBatchBinaryQwenOmniTests(unittest.TestCase):
    def setUp(self):
        self.batch_module = load_batch_module()

    def test_load_config_reads_yaml_batch_config(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        config_path = scratch / f"batch_eval_{uuid.uuid4().hex}.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model_path: /models/qwen",
                    "output_root: /tmp/batch_eval",
                    "defaults:",
                    "  batch_size: 1",
                    "runs:",
                    "  - name: fakeavceleb_stage1",
                    "    dataset: FakeAVCeleb",
                    "    adapter_path: /outputs/checkpoint-1",
                    "    jsonl: /data/fakeavceleb_eval.jsonl",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        config = self.batch_module.load_config(config_path)

        self.assertEqual(config["model_path"], "/models/qwen")
        self.assertEqual(config["runs"][0]["name"], "fakeavceleb_stage1")

    def test_resolve_run_merges_defaults_and_output_root(self):
        config = {
            "model_path": "/models/qwen",
            "output_root": "/tmp/batch_eval",
            "defaults": {
                "batch_size": 1,
                "fps": 1.0,
                "torch_dtype": "bfloat16",
                "device_map": "auto",
                "use_audio_in_video": True,
                "save_every": 100,
            },
            "runs": [
                {
                    "name": "fakeavceleb_stage1",
                    "adapter_path": "/outputs/checkpoint-1",
                    "jsonl": "/data/fakeavceleb_eval.jsonl",
                    "dataset": "FakeAVCeleb",
                }
            ],
        }

        resolved = self.batch_module.resolve_run(config, config["runs"][0], overrides={})

        self.assertEqual(resolved["model_path"], "/models/qwen")
        self.assertEqual(resolved["batch_size"], 1)
        self.assertEqual(resolved["output_dir"], "/tmp/batch_eval/fakeavceleb_stage1")
        self.assertEqual(resolved["dataset"], "FakeAVCeleb")

    def test_build_eval_command_includes_batch_and_audio_flags(self):
        run = {
            "name": "mavosdd_stage1",
            "model_path": "/models/qwen",
            "adapter_path": "/outputs/checkpoint-2",
            "jsonl": "/data/mavosdd_test.jsonl",
            "output_dir": "/tmp/out/mavosdd_stage1",
            "batch_size": 2,
            "fps": 0.5,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "use_audio_in_video": False,
            "save_every": 20,
            "fake_token_id": 52317,
            "real_token_id": 12768,
            "max_samples": 50,
        }

        command = self.batch_module.build_eval_command(
            run,
            python_executable="python",
            eval_script=Path("scripts/eval_binary_logits_qwen_omni.py"),
        )

        self.assertIn("--batch_size", command)
        self.assertIn("2", command)
        self.assertIn("--fps", command)
        self.assertIn("0.5", command)
        self.assertIn("--no_use_audio_in_video", command)
        self.assertIn("--max_samples", command)
        self.assertIn("50", command)

    def test_default_eval_script_points_to_script_entry(self):
        path = self.batch_module.default_eval_script()

        self.assertEqual(path.name, "eval_binary_logits_qwen_omni.py")
        self.assertEqual(path.parent.name, "scripts")

    def test_default_config_points_to_yaml_file(self):
        args = self.batch_module.parse_args([])

        self.assertTrue(str(args.config).endswith("qwen_omni_binary_batch_eval.yaml"))

    def test_summary_row_keeps_requested_metrics(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        output_dir = scratch / f"batch_metrics_{uuid.uuid4().hex}"
        output_dir.mkdir()
        metrics = {
            "num_predictions": 10,
            "num_bad_samples": 1,
            "accuracy": 0.7,
            "roc_auc": 0.8,
            "average_precision": 0.75,
            "map": 0.75,
            "fake_recall": 0.6,
            "real_recall": 0.9,
            "confusion_matrix_labels": ["Fake", "Real"],
            "confusion_matrix": [[3, 2], [1, 4]],
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        run = {
            "name": "fakeavceleb_stage1",
            "dataset": "FakeAVCeleb",
            "adapter_path": "/outputs/checkpoint-1",
            "jsonl": "/data/fakeavceleb_eval.jsonl",
            "output_dir": str(output_dir),
        }

        row = self.batch_module.build_summary_row(run, returncode=0, status="completed")

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["accuracy"], 0.7)
        self.assertEqual(row["auc"], 0.8)
        self.assertEqual(row["ap"], 0.75)
        self.assertEqual(row["map"], 0.75)
        self.assertEqual(row["fake_recall"], 0.6)
        self.assertEqual(row["real_recall"], 0.9)
        self.assertEqual(row["confusion_matrix"], [[3, 2], [1, 4]])


if __name__ == "__main__":
    unittest.main()
