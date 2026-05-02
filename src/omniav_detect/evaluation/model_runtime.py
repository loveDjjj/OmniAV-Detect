"""
本文件功能：
- 负责 Qwen2.5-Omni binary detector 的模型加载、多模态输入构造和 logits 推理。

主要内容：
- load_model_and_processor：加载 Qwen2.5-Omni 基座、可选 LoRA adapter 和 processor。
- prepare_inputs：优先使用 qwen_omni_utils/decord 路径处理视频和音频。
- evaluate_batch / evaluate_sample：执行 Real/Fake token logits 评估。

使用方式：
- 被 `binary_logits.py` 主流程调用，也通过脚本入口重新导出供测试使用。
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Sequence, Tuple

from omniav_detect.evaluation.constants import SYSTEM_PROMPT, USER_PROMPT_AFTER_VIDEO
from omniav_detect.evaluation.metrics import pair_softmax


def build_conversation(video_path: str) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 构造与训练一致的 Qwen2.5-Omni 多模态对话。

    参数：
    - video_path: 单个视频文件绝对路径。

    返回：
    - Qwen2.5-Omni processor 可读取的 conversation 列表。
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": USER_PROMPT_AFTER_VIDEO},
            ],
        },
    ]


def resolve_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    """将命令行 dtype 名称解析为 torch dtype 对象。"""
    normalized = str(dtype_name).strip().lower()
    if normalized in {"auto", "none"}:
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "float": "float32",
        "fp32": "float32",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(torch_module, attr):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch_module, attr)


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    """
    函数功能：
    - 加载 Qwen2.5-Omni 基座模型、可选 LoRA adapter 和 processor。

    参数：
    - args: 评估 CLI 参数，包含 model_path、adapter_path、dtype、device_map。

    返回：
    - 已切换 eval 模式的模型和 processor。
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Please install transformers, peft, torch, and qwen_omni_utils "
            "in the Qwen2.5-Omni evaluation environment."
        ) from exc

    dtype = resolve_torch_dtype(args.torch_dtype, torch)
    logging.info("Loading base model from %s", args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        enable_audio_output=False,
    )
    if args.adapter_path:
        logging.info("Loading LoRA adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        logging.info("Running base-model evaluation without LoRA adapter")
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    return model, processor


def resolve_forward_model(model: Any) -> Any:
    """
    函数功能：
    - 从 PEFT 包装后的 Qwen2.5-Omni 模型中解析真正负责 text logits 的 forward 模块。

    关键逻辑：
    - 优先返回 `base_model.model.thinker`，避免直接调用 PEFT 顶层 Omni wrapper。
    """
    candidate_paths = [
        ("base_model", "model", "thinker"),
        ("model", "thinker"),
        ("thinker",),
    ]
    for path in candidate_paths:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current
    return model


def infer_input_device(model: Any) -> Any:
    """从模型或参数中推断输入 tensor 应移动到的设备。"""
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    for parameter in model.parameters():
        return parameter.device
    return None


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    """递归地把 processor 输出移动到目标设备。"""
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: move_inputs_to_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, list):
        return [move_inputs_to_device(value, device) for value in inputs]
    if isinstance(inputs, tuple):
        return tuple(move_inputs_to_device(value, device) for value in inputs)
    return inputs


def prepare_inputs(
    processor: Any,
    conversations: List[Dict[str, Any]] | List[List[Dict[str, Any]]],
    device: Any,
    use_audio_in_video: bool,
    fps: float,
) -> Any:
    """
    函数功能：
    - 将 conversation 转换为 Qwen2.5-Omni forward 所需 tensor 输入。

    关键逻辑：
    - 优先使用 qwen_omni_utils.process_mm_info/decord 路径，规避服务器 torchvision read_video 缺失问题。
    """
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        try:
            inputs = processor.apply_chat_template(
                conversations,
                load_audio_from_video=use_audio_in_video,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                fps=fps,
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )
            return move_inputs_to_device(inputs, device)
        except Exception as fallback_exc:  # noqa: BLE001 - preserve actionable dependency hint.
            raise RuntimeError(
                "qwen_omni_utils is unavailable and the Transformers built-in video decoder failed. "
                "Install qwen_omni_utils/decord or fix the torchvision video backend."
            ) from fallback_exc

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)
    common_kwargs = {
        "text": text,
        "images": images,
        "videos": videos,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": use_audio_in_video,
    }
    try:
        inputs = processor(audio=audios, **common_kwargs)
    except TypeError:
        inputs = processor(audios=audios, **common_kwargs)
    return move_inputs_to_device(inputs, device)


def get_last_token_logits_batch(outputs: Any, inputs: Any) -> Any:
    """根据 attention_mask 找到每条样本最后一个有效位置的 logits。"""
    logits = outputs.logits
    attention_mask = None
    if isinstance(inputs, dict):
        attention_mask = inputs.get("attention_mask")
    elif hasattr(inputs, "get"):
        attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        last_indices = attention_mask.sum(dim=1).to(logits.device).long() - 1
        batch_indices = last_indices.new_tensor(range(logits.shape[0]))
        return logits[batch_indices, last_indices, :]
    return logits[:, -1, :]


def get_last_token_logits(outputs: Any, inputs: Any) -> Any:
    """返回单样本最后一个有效位置的 logits。"""
    return get_last_token_logits_batch(outputs, inputs)[0]


def extract_binary_probs(logits: Any, real_token_id: int, fake_token_id: int) -> Dict[str, float | str]:
    """从词表 logits 中提取 Real/Fake token 概率。"""
    vocab_size = int(logits.shape[-1])
    if real_token_id < 0 or real_token_id >= vocab_size:
        raise ValueError(f"real_token_id={real_token_id} is outside vocabulary size {vocab_size}")
    if fake_token_id < 0 or fake_token_id >= vocab_size:
        raise ValueError(f"fake_token_id={fake_token_id} is outside vocabulary size {vocab_size}")
    real_logit = float(logits[real_token_id].detach().float().cpu().item())
    fake_logit = float(logits[fake_token_id].detach().float().cpu().item())
    return pair_softmax(real_logit=real_logit, fake_logit=fake_logit)


def extract_binary_probs_batch(logits: Any, real_token_id: int, fake_token_id: int) -> List[Dict[str, float | str]]:
    """批量提取 Real/Fake token 概率。"""
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
    return [
        extract_binary_probs(row_logits, real_token_id=real_token_id, fake_token_id=fake_token_id)
        for row_logits in logits
    ]


def evaluate_batch(
    model: Any,
    processor: Any,
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 对一个 batch 的样本执行 Qwen2.5-Omni logits 评估。

    返回：
    - 每个样本的预测记录，包含 p_real、p_fake、pred、label 和 meta。
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing torch in evaluation environment.") from exc

    forward_model = resolve_forward_model(model)
    conversations = [build_conversation(sample["video_path"]) for sample in samples]
    processor_input = conversations[0] if len(conversations) == 1 else conversations
    device = infer_input_device(forward_model)
    inputs = prepare_inputs(processor, processor_input, device, args.use_audio_in_video, args.fps)
    with torch.inference_mode():
        try:
            outputs = forward_model(**inputs, use_audio_in_video=args.use_audio_in_video)
        except TypeError:
            outputs = forward_model(**inputs)
    logits = get_last_token_logits_batch(outputs, inputs)
    score_rows = extract_binary_probs_batch(logits, real_token_id=args.real_token_id, fake_token_id=args.fake_token_id)
    records = []
    for sample, scores in zip(samples, score_rows):
        records.append(
            {
                "index": sample["index"],
                "line_number": sample["line_number"],
                "video_path": sample["video_path"],
                "label": sample["label"],
                "pred": scores["pred"],
                "p_real": scores["p_real"],
                "p_fake": scores["p_fake"],
                "real_logit": scores["real_logit"],
                "fake_logit": scores["fake_logit"],
                "meta": sample.get("meta", {}),
            }
        )
    return records


def evaluate_sample(model: Any, processor: Any, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """单样本评估封装，用于 batch 失败后的样本级重试。"""
    return evaluate_batch(model, processor, [sample], args)[0]
