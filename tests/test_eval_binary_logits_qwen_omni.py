import json
import math
import sys
import types
import unittest
import uuid
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omniav_detect.evaluation import binary_logits as eval_module


class EvalBinaryLogitsQwenOmniTests(unittest.TestCase):
    def test_build_conversation_uses_expected_qwen_omni_prompt(self):
        video_path = str(Path("clip.mp4").resolve())

        conversation = eval_module.build_conversation(video_path)

        self.assertEqual(conversation[0]["role"], "system")
        self.assertEqual(conversation[0]["content"][0]["text"], eval_module.SYSTEM_PROMPT)
        self.assertEqual(conversation[1]["role"], "user")
        self.assertEqual(conversation[1]["content"][0], {"type": "video", "video": video_path})
        self.assertEqual(conversation[1]["content"][1]["text"], eval_module.USER_PROMPT_AFTER_VIDEO)
        self.assertEqual(
            eval_module.USER_PROMPT,
            "<video>\nGiven the video, please assess if it's Real or Fake? Only answer Real or Fake.",
        )

    def test_pair_softmax_predicts_fake_when_fake_probability_is_higher(self):
        result = eval_module.pair_softmax(real_logit=1.0, fake_logit=3.0)

        self.assertEqual(result["pred"], "Fake")
        self.assertGreater(result["p_fake"], result["p_real"])
        self.assertTrue(math.isclose(result["p_fake"] + result["p_real"], 1.0, rel_tol=1e-9))

    def test_parse_args_defaults_to_single_sample_batches(self):
        args = eval_module.parse_args(["--jsonl", "eval.jsonl", "--output_dir", "out"])

        self.assertEqual(args.batch_size, 1)
        self.assertTrue(args.use_audio_in_video)
        self.assertIsNone(args.adapter_path)

    def test_load_model_and_processor_skips_peft_when_adapter_path_is_missing(self):
        calls = []

        class FakeBaseModel:
            def eval(self):
                calls.append(("eval", None))
                return self

        class FakeProcessor:
            pass

        class FakeQwenModelClass:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append(("base_model", {"args": args, "kwargs": kwargs}))
                return FakeBaseModel()

        class FakeProcessorClass:
            @staticmethod
            def from_pretrained(model_path):
                calls.append(("processor", model_path))
                return FakeProcessor()

        fake_torch = types.SimpleNamespace(bfloat16="bf16")
        fake_transformers = types.SimpleNamespace(
            Qwen2_5OmniForConditionalGeneration=FakeQwenModelClass,
            Qwen2_5OmniProcessor=FakeProcessorClass,
        )
        fake_peft = types.SimpleNamespace(
            PeftModel=types.SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PEFT should not be used"))
            )
        )
        old_torch = sys.modules.get("torch")
        old_peft = sys.modules.get("peft")
        old_transformers = sys.modules.get("transformers")
        sys.modules["torch"] = fake_torch
        sys.modules["peft"] = fake_peft
        sys.modules["transformers"] = fake_transformers
        try:
            args = types.SimpleNamespace(model_path="/model", adapter_path=None, torch_dtype="bfloat16", device_map="auto")
            model, processor = eval_module.load_model_and_processor(args)
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch
            if old_peft is None:
                sys.modules.pop("peft", None)
            else:
                sys.modules["peft"] = old_peft
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers

        self.assertIsInstance(model, FakeBaseModel)
        self.assertIsInstance(processor, FakeProcessor)
        self.assertEqual(calls[0][0], "base_model")
        self.assertEqual(calls[1][0], "eval")
        self.assertEqual(calls[2], ("processor", "/model"))

    def test_batch_samples_groups_samples_without_dropping_tail(self):
        samples = [{"index": idx} for idx in range(5)]

        batches = list(eval_module.batch_samples(samples, batch_size=2))

        self.assertEqual([[item["index"] for item in batch] for batch in batches], [[0, 1], [2, 3], [4]])

    def test_resolve_forward_model_prefers_peft_wrapped_thinker(self):
        thinker = object()

        class InnerModel:
            pass

        class BaseModel:
            pass

        class WrappedModel:
            pass

        inner = InnerModel()
        inner.thinker = thinker
        base = BaseModel()
        base.model = inner
        wrapped = WrappedModel()
        wrapped.base_model = base

        self.assertIs(eval_module.resolve_forward_model(wrapped), thinker)

    def test_prepare_inputs_prefers_qwen_omni_utils_decoder_path(self):
        calls = []

        class FakeProcessor:
            def apply_chat_template(self, conversations, **kwargs):
                calls.append(("apply_chat_template", kwargs))
                if kwargs.get("tokenize"):
                    raise AssertionError("prepare_inputs should not use the transformers torchvision video path first")
                return "chat text"

            def __call__(self, **kwargs):
                calls.append(("processor_call", kwargs))
                return {"input_ids": "ids"}

        def fake_process_mm_info(conversations, use_audio_in_video):
            calls.append(("process_mm_info", {"use_audio_in_video": use_audio_in_video}))
            return ["audio"], [], ["video"]

        fake_module = types.SimpleNamespace(process_mm_info=fake_process_mm_info)
        old_module = sys.modules.get("qwen_omni_utils")
        sys.modules["qwen_omni_utils"] = fake_module
        try:
            inputs = eval_module.prepare_inputs(
                FakeProcessor(),
                eval_module.build_conversation("/tmp/a.mp4"),
                device=None,
                use_audio_in_video=True,
                fps=1.0,
            )
        finally:
            if old_module is None:
                sys.modules.pop("qwen_omni_utils", None)
            else:
                sys.modules["qwen_omni_utils"] = old_module

        self.assertEqual(inputs, {"input_ids": "ids"})
        self.assertEqual(calls[0][0], "apply_chat_template")
        self.assertFalse(calls[0][1]["tokenize"])
        self.assertEqual(calls[1], ("process_mm_info", {"use_audio_in_video": True}))

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

        samples = eval_module.load_jsonl_samples(jsonl_path, max_samples=1)

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

        metrics = eval_module.compute_metrics(predictions)

        self.assertTrue(math.isclose(metrics["accuracy"], 2 / 3, rel_tol=1e-9))
        self.assertEqual(metrics["label_distribution"], {"Real": 1, "Fake": 2})
        self.assertEqual(metrics["prediction_distribution"], {"Real": 2, "Fake": 1})
        self.assertEqual(metrics["confusion_matrix_labels"], ["Fake", "Real"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertTrue(math.isclose(metrics["fake_recall"], 0.5, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["real_recall"], 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["roc_auc"], 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["average_precision"], 1.0, rel_tol=1e-9))

    def test_save_outputs_writes_visualization_artifacts(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        output_dir = scratch / f"eval_visuals_{uuid.uuid4().hex}"
        predictions = [
            {"label": "Real", "pred": "Real", "p_fake": 0.1, "p_real": 0.9},
            {"label": "Fake", "pred": "Fake", "p_fake": 0.9, "p_real": 0.1},
            {"label": "Fake", "pred": "Real", "p_fake": 0.4, "p_real": 0.6},
        ]

        metrics = eval_module.save_outputs(output_dir, predictions, [], {"run": "test"})

        visual_dir = output_dir / "visualizations"
        self.assertEqual(metrics["visualizations_dir"], str(visual_dir))
        self.assertTrue((visual_dir / "confusion_matrix.csv").exists())
        self.assertTrue((visual_dir / "score_distribution.csv").exists())
        self.assertTrue((visual_dir / "summary.html").exists())
        self.assertIn("Fake recall", (visual_dir / "summary.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
