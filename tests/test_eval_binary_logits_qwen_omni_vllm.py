"""
本文件功能：
- 覆盖 vLLM 后端评估入口的基础参数、prompt 与多模态输入构造逻辑。

主要内容：
- 测试 parse_args 默认值。
- 测试 build_multi_modal_data、build_prompt_text 与 dtype 解析。
"""

import importlib.util
import argparse
import sys
import types
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_eval_module():
    script_path = SCRIPTS_DIR / "eval_binary_logits_qwen_omni_vllm.py"
    spec = importlib.util.spec_from_file_location("eval_binary_logits_qwen_omni_vllm", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalBinaryLogitsQwenOmniVllmTests(unittest.TestCase):
    def setUp(self):
        self.eval_module = load_eval_module()

    def test_parse_args_defaults(self):
        args = self.eval_module.parse_args(
            [
                "--adapter_path",
                "adapter",
                "--jsonl",
                "eval.jsonl",
                "--output_dir",
                "out",
            ]
        )

        self.assertEqual(args.batch_size, 1)
        self.assertTrue(args.use_audio_in_video)
        self.assertEqual(args.mm_format, "omni_av")
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.logprobs, -1)

    def test_build_multi_modal_data_default_uses_qwen_omni_audio_video(self):
        conversation = self.eval_module.build_conversation("/tmp/a.mp4")

        def fake_process_mm_info(messages, use_audio_in_video):
            return ["audio_inputs"], [], ["video_inputs"]

        fake_module = types.SimpleNamespace(process_mm_info=fake_process_mm_info)
        old_module = sys.modules.get("qwen_omni_utils")
        sys.modules["qwen_omni_utils"] = fake_module
        try:
            mm_data = self.eval_module.build_multi_modal_data(conversation, True, "omni_av")
        finally:
            if old_module is None:
                sys.modules.pop("qwen_omni_utils", None)
            else:
                sys.modules["qwen_omni_utils"] = old_module

        self.assertEqual(mm_data, {"audio": ["audio_inputs"], "video": ["video_inputs"]})

    def test_build_multi_modal_data_requires_audio_for_omni_av_when_enabled(self):
        conversation = self.eval_module.build_conversation("/tmp/a.mp4")

        def fake_process_mm_info(messages, use_audio_in_video):
            return [], [], ["video_inputs"]

        fake_module = types.SimpleNamespace(process_mm_info=fake_process_mm_info)
        old_module = sys.modules.get("qwen_omni_utils")
        sys.modules["qwen_omni_utils"] = fake_module
        try:
            with self.assertRaisesRegex(ValueError, "audio"):
                self.eval_module.build_multi_modal_data(conversation, True, "omni_av")
        finally:
            if old_module is None:
                sys.modules.pop("qwen_omni_utils", None)
            else:
                sys.modules["qwen_omni_utils"] = old_module

    def test_build_multi_modal_data_video_only_variants(self):
        conversation = self.eval_module.build_conversation("/tmp/a.mp4")

        def fake_process_vision_info(messages):
            return ["image_inputs"], ["video_inputs"]

        fake_module = types.SimpleNamespace(process_vision_info=fake_process_vision_info)
        old_module = sys.modules.get("qwen_vl_utils")
        sys.modules["qwen_vl_utils"] = fake_module
        try:
            mm_data = self.eval_module.build_multi_modal_data(conversation, True, "video")
        finally:
            if old_module is None:
                sys.modules.pop("qwen_vl_utils", None)
            else:
                sys.modules["qwen_vl_utils"] = old_module

        self.assertEqual(mm_data, {"image": ["image_inputs"], "video": ["video_inputs"]})
        self.assertEqual(
            self.eval_module.build_multi_modal_data(conversation, True, "video_path"),
            {"video": "/tmp/a.mp4"},
        )
        self.assertEqual(
            self.eval_module.build_multi_modal_data(conversation, True, "videos_list"),
            {"videos": ["/tmp/a.mp4"]},
        )
        self.assertEqual(
            self.eval_module.build_multi_modal_data(conversation, False, "video_audio"),
            {"video": "/tmp/a.mp4"},
        )

    def test_build_sampling_params_omits_logprob_token_ids_when_unsupported(self):
        captured = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_module = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
        old_module = sys.modules.get("vllm")
        sys.modules["vllm"] = fake_module
        try:
            args = argparse.Namespace(
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=1,
                logprobs=-1,
                real_token_id=12768,
                fake_token_id=52317,
            )
            self.eval_module.build_sampling_params(args)
        finally:
            if old_module is None:
                sys.modules.pop("vllm", None)
            else:
                sys.modules["vllm"] = old_module

        self.assertEqual(captured["logprobs"], -1)
        self.assertNotIn("logprob_token_ids", captured)

    def test_build_sampling_params_uses_logprob_token_ids_when_supported(self):
        captured = {}

        class FakeSamplingParams:
            def __init__(self, temperature, top_p, max_tokens, logprobs, prompt_logprobs, logprob_token_ids):
                captured.update(
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "logprobs": logprobs,
                        "prompt_logprobs": prompt_logprobs,
                        "logprob_token_ids": logprob_token_ids,
                    }
                )

        fake_module = types.SimpleNamespace(SamplingParams=FakeSamplingParams)
        old_module = sys.modules.get("vllm")
        sys.modules["vllm"] = fake_module
        try:
            args = argparse.Namespace(
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=1,
                logprobs=2,
                real_token_id=12768,
                fake_token_id=52317,
            )
            self.eval_module.build_sampling_params(args)
        finally:
            if old_module is None:
                sys.modules.pop("vllm", None)
            else:
                sys.modules["vllm"] = old_module

        self.assertEqual(captured["logprob_token_ids"], [12768, 52317])

    def test_extract_binary_probs_accepts_string_token_id_keys(self):
        generation = types.SimpleNamespace(
            logprobs=[
                {
                    "12768": types.SimpleNamespace(logprob=-0.2),
                    "52317": types.SimpleNamespace(logprob=-1.3),
                }
            ]
        )
        output = types.SimpleNamespace(outputs=[generation])

        scores = self.eval_module.extract_binary_probs_from_output(output, 12768, 52317)

        self.assertEqual(scores["pred"], "Real")
        self.assertGreater(scores["p_real"], scores["p_fake"])

    def test_build_prompt_text_fallbacks_without_tokenizer(self):
        prompt = self.eval_module.build_prompt_text(None, self.eval_module.build_conversation("/tmp/a.mp4"))

        self.assertIn(self.eval_module.SYSTEM_PROMPT, prompt)
        self.assertIn(self.eval_module.USER_PROMPT, prompt)

    def test_resolve_vllm_dtype_aliases(self):
        self.assertEqual(self.eval_module.resolve_vllm_dtype("bf16"), "bfloat16")
        self.assertEqual(self.eval_module.resolve_vllm_dtype("fp16"), "float16")


if __name__ == "__main__":
    unittest.main()
