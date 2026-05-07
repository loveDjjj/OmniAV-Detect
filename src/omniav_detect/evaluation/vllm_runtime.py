"""
本文件功能：
- 负责基于 vLLM 的 Qwen2.5-Omni LoRA binary detector 推理与 Real/Fake token logits 评估。

主要内容：
- build_conversation：构造与训练一致的多模态对话结构。
- build_prompt_text：基于 tokenizer 或后备规则生成 prompt 文本。
- load_vllm_engine：加载 vLLM 引擎与 LoRA 请求对象。
- build_sampling_params：构造用于获取指定 token 概率的采样参数。
- evaluate_batch / evaluate_sample：执行批量/单样本评估并返回标准预测记录。
"""

from __future__ import annotations

import argparse
import inspect
import logging
from typing import Any, Dict, List, Sequence, Tuple

from omniav_detect.evaluation.constants import SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_AFTER_VIDEO
from omniav_detect.evaluation.metrics import pair_softmax


def build_conversation(
    video_path: str,
    fps: float | None = None,
    audio_paths: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 构造与训练一致的 Qwen2.5-Omni 多模态对话。

    参数：
    - video_path: 单个视频文件绝对路径。

    返回：
    - Qwen2.5-Omni 聊天模板可读取的 conversation 列表。
    """
    video_payload: Dict[str, Any] = {"type": "video", "video": video_path}
    if fps is not None:
        video_payload["fps"] = fps
    user_content: List[Dict[str, Any]] = [video_payload]
    for audio_path in audio_paths or []:
        if str(audio_path).strip():
            user_content.append({"type": "audio", "audio": str(audio_path)})
    user_content.append({"type": "text", "text": USER_PROMPT_AFTER_VIDEO})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def extract_video_path(conversation: List[Dict[str, Any]]) -> str:
    """从 conversation 中提取视频路径，用于兼容旧的 multi-modal 输入格式。"""
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                value = item.get("video")
                if value:
                    return str(value)
    return ""


def resolve_vllm_dtype(dtype_name: str) -> str:
    """将命令行 dtype 名称映射为 vLLM 可接受的字符串。"""
    normalized = str(dtype_name).strip().lower()
    if normalized in {"auto", "none"}:
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "float": "float32",
        "fp32": "float32",
    }
    return aliases.get(normalized, normalized)


def resolve_tokenizer(llm: Any) -> Any:
    """从 vLLM 引擎中解析 tokenizer 实例。"""
    getter = getattr(llm, "get_tokenizer", None)
    if callable(getter):
        return getter()
    return getattr(llm, "tokenizer", None)


def build_prompt_text(tokenizer: Any, conversation: List[Dict[str, Any]]) -> str:
    """
    函数功能：
    - 使用 tokenizer 的 chat template 构造 prompt 文本，失败时回退到固定拼接。

    参数：
    - tokenizer: vLLM 里的 tokenizer 或 None。
    - conversation: build_conversation 返回的对话结构。

    返回：
    - 可直接传给 vLLM 的 prompt 文本。
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(conversation, tokenize=False)
    return f"{SYSTEM_PROMPT}\n{USER_PROMPT}"


def build_multi_modal_data(
    conversation: List[Dict[str, Any]],
    use_audio_in_video: bool,
    mm_format: str,
) -> Dict[str, Any] | None:
    """
    函数功能：
    - 构造 vLLM multi-modal 输入数据结构。

    参数：
    - conversation: build_conversation 返回的对话结构。
    - use_audio_in_video: 是否使用视频里的音频。
    - mm_format: multi-modal 数据格式。

    返回：
    - multi_modal_data 字典，或 None 表示仅用纯文本 prompt。
    """
    normalized = str(mm_format).strip().lower()
    if normalized in {"none", "text"}:
        return None
    if normalized in {"omni_av", "qwen_omni", "audio_video"}:
        return build_omni_av_multi_modal_data(conversation, use_audio_in_video)
    if normalized in {"video", "qwen_vl", "qwen_vl_utils"}:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "qwen-vl-utils is required to build vLLM video inputs. "
                "Install it with: pip install qwen-vl-utils"
            ) from exc
        image_inputs, video_inputs = process_vision_info(conversation)
        mm_data: Dict[str, Any] = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs
        if not mm_data:
            raise ValueError("process_vision_info returned empty multi-modal inputs")
        return mm_data

    video_path = extract_video_path(conversation)
    if not video_path:
        raise ValueError("No video path found in conversation for multi-modal inputs")
    if normalized == "video_path":
        return {"video": video_path}
    if normalized == "video_dict":
        return {"video": {"path": video_path, "use_audio": use_audio_in_video}}
    if normalized == "videos_list":
        return {"videos": [video_path]}
    if normalized == "video_audio":
        data: Dict[str, Any] = {"video": video_path}
        if use_audio_in_video:
            data["audio"] = video_path
        return data
    raise ValueError(f"Unsupported mm_format: {mm_format}")


def build_omni_av_multi_modal_data(
    conversation: List[Dict[str, Any]],
    use_audio_in_video: bool,
) -> Dict[str, Any]:
    """
    函数功能：
    - 使用 Qwen2.5-Omni 官方工具构造 vLLM 的 audio+video multi_modal_data。

    参数：
    - conversation: build_conversation 返回的对话结构。
    - use_audio_in_video: 是否从视频中抽取音频。

    返回：
    - 包含 audio/video/image 的 multi_modal_data 字典。

    关键逻辑：
    - 默认 vLLM 加速路径要与 Transformers 路径一样使用 `process_mm_info`，避免只传视频帧。
    - 如果请求音频但没有得到 audio 输入，直接报错，避免误以为做了双模态评估。
    """
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise RuntimeError(
            "qwen_omni_utils is required for vLLM audio-video inputs. "
            "Install it before using --mm_format omni_av."
        ) from exc

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    mm_data: Dict[str, Any] = {}
    if audios:
        mm_data["audio"] = audios
    if images:
        mm_data["image"] = images
    if videos:
        mm_data["video"] = videos
    if use_audio_in_video and not audios:
        raise ValueError(
            "Audio was requested for vLLM evaluation but qwen_omni_utils returned no audio inputs. "
            "Use --no_use_audio_in_video for video-only evaluation or check that the source video has audio."
        )
    if not mm_data:
        raise ValueError("qwen_omni_utils.process_mm_info returned empty multi-modal inputs")
    return mm_data


def load_vllm_engine(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    """
    函数功能：
    - 初始化 vLLM LLM 引擎，并按需准备 LoRA 请求对象。

    参数：
    - args: 评估 CLI 参数。

    返回：
    - llm 引擎、tokenizer、lora_request。
    """
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("Missing vLLM dependency. Please install vllm before using the vLLM backend.") from exc

    lora_request = None
    enable_lora = False
    if args.adapter_path:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise RuntimeError("vLLM LoRA support is unavailable. Please install/upgrade vLLM with LoRA.") from exc
        # 关键逻辑：vLLM 需要 LoRARequest 对象才能在推理时启用 adapter。
        lora_request = LoRARequest("omniav", 1, args.adapter_path)
        enable_lora = True

    llm_kwargs: Dict[str, Any] = {
        "model": args.model_path,
        "dtype": resolve_vllm_dtype(args.torch_dtype),
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_lora": enable_lora,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    logging.info("Initializing vLLM engine with model %s", args.model_path)
    llm = LLM(**llm_kwargs)
    tokenizer = resolve_tokenizer(llm)
    return llm, tokenizer, lora_request


def build_sampling_params(args: argparse.Namespace) -> Any:
    """
    函数功能：
    - 构造 vLLM SamplingParams，强制输出单 token 并返回指定 token 的 logprob。

    参数：
    - args: 评估 CLI 参数。

    返回：
    - SamplingParams 实例。
    """
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("Missing vLLM dependency. Please install vllm before using the vLLM backend.") from exc

    kwargs: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "logprobs": args.logprobs,
        "prompt_logprobs": 0,
    }
    signature = inspect.signature(SamplingParams)
    if "logprob_token_ids" in signature.parameters:
        kwargs["logprob_token_ids"] = [args.real_token_id, args.fake_token_id]
    return SamplingParams(**kwargs)


def _read_logprob(value: Any) -> float:
    """解析 vLLM 返回的 logprob 值或对象。"""
    if value is None:
        raise ValueError("logprob value is missing")
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict) and "logprob" in value:
        return float(value["logprob"])
    return float(value)


def _extract_logprob_map(generation: Any) -> Dict[int, Any] | None:
    """从 vLLM generation 输出中提取 token_id -> logprob 的映射。"""
    logprob_token_ids = getattr(generation, "logprob_token_ids", None)
    if isinstance(logprob_token_ids, dict):
        return logprob_token_ids
    if isinstance(logprob_token_ids, list) and logprob_token_ids:
        if isinstance(logprob_token_ids[0], dict):
            return logprob_token_ids[0]

    logprobs = getattr(generation, "logprobs", None)
    if isinstance(logprobs, list) and logprobs:
        entry = logprobs[0]
        if isinstance(entry, dict):
            return entry

    token_logprobs = getattr(generation, "token_logprobs", None)
    if isinstance(token_logprobs, dict):
        return token_logprobs
    return None


def _lookup_logprob(logprob_map: Dict[Any, Any], token_id: int) -> Any:
    """兼容 int key、数字字符串 key 和对象 key 的 vLLM logprob map。"""
    if token_id in logprob_map:
        return logprob_map[token_id]
    text_key = str(token_id)
    if text_key in logprob_map:
        return logprob_map[text_key]
    for key, value in logprob_map.items():
        if str(key) == text_key:
            return value
    return None


def extract_binary_probs_from_output(output: Any, real_token_id: int, fake_token_id: int) -> Dict[str, float | str]:
    """
    函数功能：
    - 从 vLLM 输出中提取 Real/Fake token 的 logprob 并计算二分类概率。
    """
    generations = getattr(output, "outputs", None)
    if not generations:
        raise ValueError("vLLM output does not contain generation outputs")
    generation = generations[0]

    # 关键逻辑：兼容不同 vLLM 版本的 logprob 字段结构。
    logprob_map = _extract_logprob_map(generation)
    if not isinstance(logprob_map, dict):
        raise ValueError(
            "vLLM output does not include logprob_token_ids/logprobs data. "
            "Set --logprobs to a non-zero value (e.g., -1) to enable logprob outputs."
        )

    real_value = _lookup_logprob(logprob_map, real_token_id)
    fake_value = _lookup_logprob(logprob_map, fake_token_id)
    if real_value is None or fake_value is None:
        raise ValueError("vLLM output is missing requested token logprobs")

    real_logprob = _read_logprob(real_value)
    fake_logprob = _read_logprob(fake_value)
    return pair_softmax(real_logit=real_logprob, fake_logit=fake_logprob)


def generate_with_lora(llm: Any, prompts: List[Any], sampling_params: Any, lora_request: Any) -> Any:
    """封装 vLLM generate 并兼容 lora_request 参数差异。"""
    if lora_request is None:
        return llm.generate(prompts, sampling_params)
    try:
        return llm.generate(prompts, sampling_params, lora_request=lora_request)
    except TypeError:
        return llm.generate(prompts, sampling_params, lora_requests=[lora_request])


def evaluate_batch(
    llm: Any,
    tokenizer: Any,
    lora_request: Any,
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    函数功能：
    - 对一个 batch 的样本执行 vLLM logits 评估。

    返回：
    - 每个样本的预测记录，包含 p_real、p_fake、pred、label 和 meta。
    """
    conversations = [
        build_conversation(sample["video_path"], fps=args.fps, audio_paths=sample.get("audio_paths"))
        for sample in samples
    ]
    prompts = [build_prompt_text(tokenizer, conversation) for conversation in conversations]
    mm_data_list = [
        build_multi_modal_data(conversation, args.use_audio_in_video, args.mm_format)
        for conversation in conversations
    ]

    prompt_inputs: List[Any] = []
    for prompt, mm_data in zip(prompts, mm_data_list):
        if mm_data is None:
            prompt_inputs.append(prompt)
        else:
            prompt_inputs.append({"prompt": prompt, "multi_modal_data": mm_data})

    sampling_params = build_sampling_params(args)
    outputs = generate_with_lora(llm, prompt_inputs, sampling_params, lora_request)
    if len(outputs) != len(samples):
        raise ValueError(f"vLLM outputs length {len(outputs)} does not match samples {len(samples)}")

    records: List[Dict[str, Any]] = []
    for sample, output in zip(samples, outputs):
        scores = extract_binary_probs_from_output(output, args.real_token_id, args.fake_token_id)
        records.append(
            {
                "index": sample["index"],
                "line_number": sample["line_number"],
                "video_path": sample["video_path"],
                "audio_paths": sample.get("audio_paths", []),
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


def evaluate_sample(
    llm: Any,
    tokenizer: Any,
    lora_request: Any,
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """单样本评估封装，用于 batch 失败后的样本级重试。"""
    return evaluate_batch(llm, tokenizer, lora_request, [sample], args)[0]
