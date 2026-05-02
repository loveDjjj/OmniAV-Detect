import json
import sys
import unittest
import uuid
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omniav_detect.evaluation.parallel_runner import (
    build_worker_command,
    resolve_worker_gpus,
    split_jsonl_to_shards,
    write_merged_outputs,
)


TMP_ROOT = Path(__file__).resolve().parent / ".tmp" / "parallel_eval"


def make_tmp_dir() -> Path:
    path = TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


class EvalParallelBinaryQwenOmniTests(unittest.TestCase):
    def test_resolve_worker_gpus_defaults_to_all_gpus(self):
        self.assertEqual(resolve_worker_gpus("0,1", None), ["0", "1"])

    def test_resolve_worker_gpus_can_limit_workers(self):
        self.assertEqual(resolve_worker_gpus("0,1,2", 2), ["0", "1"])

    def test_split_jsonl_to_shards_uses_round_robin_and_max_samples(self):
        root = make_tmp_dir()
        source = root / "eval.jsonl"
        source.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(5)), encoding="utf-8")

        shards = split_jsonl_to_shards(source, root / "shards", num_shards=2, max_samples=4)

        self.assertEqual(len(shards), 2)
        self.assertEqual([json.loads(line)["id"] for line in shards[0].read_text(encoding="utf-8").splitlines()], [0, 2])
        self.assertEqual([json.loads(line)["id"] for line in shards[1].read_text(encoding="utf-8").splitlines()], [1, 3])

    def test_build_worker_command_includes_single_gpu_shard_settings(self):
        command = build_worker_command(
            python_executable="python",
            eval_script=Path("scripts/eval_binary_logits_qwen_omni.py"),
            model_path="/model",
            adapter_path="/adapter",
            shard_jsonl=Path("/tmp/shard.jsonl"),
            shard_output_dir=Path("/tmp/out/shard_000"),
            batch_size=2,
            fps=1.0,
            save_every=10,
            torch_dtype="bfloat16",
            device_map="auto",
            fake_token_id=52317,
            real_token_id=12768,
            use_audio_in_video=True,
            extra_args=[],
        )

        self.assertIn("--jsonl", command)
        self.assertIn(str(Path("/tmp/shard.jsonl")), command)
        self.assertIn("--batch_size", command)
        self.assertIn("2", command)
        self.assertIn("--use_audio_in_video", command)

    def test_build_worker_command_omits_adapter_flag_for_base_model_eval(self):
        command = build_worker_command(
            python_executable="python",
            eval_script=Path("scripts/eval_binary_logits_qwen_omni.py"),
            model_path="/model",
            adapter_path=None,
            shard_jsonl=Path("/tmp/shard.jsonl"),
            shard_output_dir=Path("/tmp/out/shard_000"),
            batch_size=1,
            fps=1.0,
            save_every=10,
            torch_dtype="bfloat16",
            device_map="auto",
            fake_token_id=52317,
            real_token_id=12768,
            use_audio_in_video=True,
            extra_args=[],
        )

        self.assertNotIn("--adapter_path", command)

    def test_write_merged_outputs_recomputes_metrics_from_shards(self):
        root = make_tmp_dir()
        shard_a = root / "worker_000"
        shard_b = root / "worker_001"
        shard_a.mkdir()
        shard_b.mkdir()
        predictions_a = [
            {"label": "Fake", "pred": "Fake", "p_fake": 0.9, "p_real": 0.1},
            {"label": "Real", "pred": "Fake", "p_fake": 0.6, "p_real": 0.4},
        ]
        predictions_b = [
            {"label": "Real", "pred": "Real", "p_fake": 0.2, "p_real": 0.8},
        ]
        (shard_a / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions_a),
            encoding="utf-8",
        )
        (shard_b / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions_b),
            encoding="utf-8",
        )
        (shard_a / "bad_samples.jsonl").write_text("", encoding="utf-8")
        (shard_b / "bad_samples.jsonl").write_text("", encoding="utf-8")

        metrics = write_merged_outputs(root / "merged", [shard_a, shard_b], {"backend": "parallel"})

        self.assertEqual(metrics["num_predictions"], 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertAlmostEqual(metrics["fake_recall"], 1.0)
        self.assertAlmostEqual(metrics["real_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
