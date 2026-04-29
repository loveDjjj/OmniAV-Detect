import importlib.util
import json
import math
import sys
import unittest
import uuid
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_eval_module():
    script_path = SCRIPTS_DIR / "eval_binary_logits_qwen_omni.py"
    spec = importlib.util.spec_from_file_location("eval_binary_logits_qwen_omni", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalBinaryLogitsQwenOmniTests(unittest.TestCase):
    def setUp(self):
        self.eval_module = load_eval_module()

    def test_build_conversation_uses_expected_qwen_omni_prompt(self):
        video_path = str(Path("clip.mp4").resolve())

        conversation = self.eval_module.build_conversation(video_path)

        self.assertEqual(conversation[0]["role"], "system")
        self.assertEqual(conversation[0]["content"][0]["text"], self.eval_module.SYSTEM_PROMPT)
        self.assertEqual(conversation[1]["role"], "user")
        self.assertEqual(conversation[1]["content"][0], {"type": "video", "video": video_path})
        self.assertEqual(conversation[1]["content"][1]["text"], self.eval_module.USER_PROMPT_AFTER_VIDEO)
        self.assertEqual(
            self.eval_module.USER_PROMPT,
            "<video>\nGiven the video, please assess if it's Real or Fake? Only answer Real or Fake.",
        )

    def test_pair_softmax_predicts_fake_when_fake_probability_is_higher(self):
        result = self.eval_module.pair_softmax(real_logit=1.0, fake_logit=3.0)

        self.assertEqual(result["pred"], "Fake")
        self.assertGreater(result["p_fake"], result["p_real"])
        self.assertTrue(math.isclose(result["p_fake"] + result["p_real"], 1.0, rel_tol=1e-9))

    def test_load_jsonl_samples_reads_ms_swift_records_and_respects_max_samples(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        jsonl_path = scratch / f"eval_samples_{uuid.uuid4().hex}.jsonl"
        rows = [
            {
                "messages": [],
                "videos": [str((scratch / "a.mp4").resolve())],
                "meta": {"overall_label": "Real", "dataset": "FakeAVCeleb"},
            },
            {
                "messages": [],
                "videos": [str((scratch / "b.mp4").resolve())],
                "meta": {"overall_label": "Fake", "dataset": "FakeAVCeleb"},
            },
        ]
        jsonl_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        samples = self.eval_module.load_jsonl_samples(jsonl_path, max_samples=1)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], "Real")
        self.assertEqual(samples[0]["video_path"], rows[0]["videos"][0])
        self.assertEqual(samples[0]["source_record"], rows[0])

    def test_compute_metrics_uses_fake_as_positive_class(self):
        predictions = [
            {"label": "Real", "pred": "Real", "p_fake": 0.1},
            {"label": "Fake", "pred": "Fake", "p_fake": 0.9},
            {"label": "Fake", "pred": "Real", "p_fake": 0.4},
        ]

        metrics = self.eval_module.compute_metrics(predictions)

        self.assertTrue(math.isclose(metrics["accuracy"], 2 / 3, rel_tol=1e-9))
        self.assertEqual(metrics["label_distribution"], {"Real": 1, "Fake": 2})
        self.assertEqual(metrics["prediction_distribution"], {"Real": 2, "Fake": 1})
        self.assertEqual(metrics["confusion_matrix_labels"], ["Fake", "Real"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertTrue(math.isclose(metrics["roc_auc"], 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["average_precision"], 1.0, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
