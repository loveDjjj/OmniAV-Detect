"""
本文件功能：
- 覆盖 vLLM 后端评估入口的基础参数、prompt 与多模态输入构造逻辑。

主要内容：
- 测试 parse_args 默认值。
- 测试 build_multi_modal_data、build_prompt_text 与 dtype 解析。
"""

import importlib.util
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
        self.assertEqual(args.mm_format, "video")
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.logprobs, -1)

    def test_build_multi_modal_data_variants(self):
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

    def test_build_prompt_text_fallbacks_without_tokenizer(self):
        prompt = self.eval_module.build_prompt_text(None, self.eval_module.build_conversation("/tmp/a.mp4"))

        self.assertIn(self.eval_module.SYSTEM_PROMPT, prompt)
        self.assertIn(self.eval_module.USER_PROMPT, prompt)

    def test_resolve_vllm_dtype_aliases(self):
        self.assertEqual(self.eval_module.resolve_vllm_dtype("bf16"), "bfloat16")
        self.assertEqual(self.eval_module.resolve_vllm_dtype("fp16"), "float16")


if __name__ == "__main__":
    unittest.main()
