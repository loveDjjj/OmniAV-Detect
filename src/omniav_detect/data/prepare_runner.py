"""
本文件功能：
- 提供统一的数据准备命令行入口，通过 YAML 配置选择 FakeAVCeleb 或 MAVOS-DD。

主要内容：
- load_prepare_config：读取并校验数据准备配置。
- resolve_dataset_run：合并配置默认值、数据集配置和命令行覆盖参数。
- prepare_dataset：按数据集类型调用对应 builder，并复用公共写出逻辑。
- main：统一 CLI 入口。

使用方式：
- `python scripts/prepare_swift_av_sft.py --dataset fakeavceleb --config configs/data/swift_av_sft.yaml`
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omniav_detect.config import load_config_file
from omniav_detect.data import fakeavceleb, mavosdd
from omniav_detect.data.common import (
    Issue,
    Record,
    scan_video_files,
    setup_logging,
    write_output_jsonl,
    write_stats,
)


DEFAULT_PREPARE_CONFIG = "configs/data/swift_av_sft.yaml"


@dataclass
class PrepareRun:
    """单个数据集准备任务的最终运行参数。"""

    name: str
    dataset_type: str
    root: Path
    output_dir: Path
    seed: int
    num_preview: int
    max_samples_per_class: Optional[int]
    dry_run: bool
    mode: str = "binary"
    train_ratio: float = 0.7
    split_protocol: str = "random_stratified"
    folds_root: Optional[Path] = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one local AV deepfake dataset as ms-swift/Qwen2.5-Omni SFT JSONL."
    )
    parser.add_argument("--config", default=DEFAULT_PREPARE_CONFIG, help="YAML/JSON data prepare config.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset entry name in config, for example: fakeavceleb or mavosdd.",
    )
    parser.add_argument("--root", default=None, help="Override selected dataset root.")
    parser.add_argument("--output_dir", default=None, help="Override selected dataset output directory.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_preview", type=int, default=None)
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument("--mode", choices=["binary", "structured", "both"], default=None)
    parser.add_argument("--fakeavceleb_train_ratio", type=float, default=None)
    parser.add_argument(
        "--fakeavceleb_split_protocol",
        choices=["random_stratified", "mrdf_5fold"],
        default=None,
        help="FakeAVCeleb split protocol: default random stratified split or MRDF subject-independent 5-fold.",
    )
    parser.add_argument(
        "--fakeavceleb_folds_root",
        default=None,
        help="Directory that contains MRDF FakeAVCeleb train_*.txt / test_*.txt files.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fakeavceleb_root", default=None, help="Compatibility alias for --root.")
    parser.add_argument("--mavos_root", default=None, help="Compatibility alias for --root.")
    return parser.parse_args(argv)


def load_prepare_config(path: Path | str) -> Dict[str, Any]:
    """
    函数功能：
    - 读取数据准备配置，并确认存在 datasets 字段。

    参数：
    - path: YAML 或 JSON 配置路径。

    返回：
    - 配置字典。

    关键逻辑：
    - `datasets` 必须是非空 mapping，数据集选择由 `--dataset` 指定。
    """
    config = load_config_file(path)
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"{path} must contain a non-empty 'datasets' mapping")
    return config


def resolve_dataset_run(
    config: Dict[str, Any],
    dataset_name: str,
    overrides: Dict[str, Any],
) -> PrepareRun:
    """
    函数功能：
    - 合并数据准备配置默认值、单数据集配置和命令行覆盖参数。

    参数：
    - config: load_prepare_config 返回的整体配置。
    - dataset_name: `datasets` 下的数据集条目名称。
    - overrides: 命令行覆盖参数字典。

    返回：
    - PrepareRun 数据类。

    关键逻辑：
    - 只解析一个数据集，避免一个命令意外同时重写多个数据集输出目录。
    """
    defaults = dict(config.get("defaults", {}))
    datasets = config["datasets"]
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    resolved: Dict[str, Any] = {}
    resolved.update(defaults)
    resolved.update(datasets[dataset_name] or {})
    resolved.setdefault("type", dataset_name)
    resolved.setdefault("seed", 42)
    resolved.setdefault("num_preview", 10)
    resolved.setdefault("max_samples_per_class", None)
    resolved.setdefault("dry_run", False)
    resolved.setdefault("mode", "binary")
    resolved.setdefault("train_ratio", resolved.get("fakeavceleb_train_ratio", 0.7))
    resolved.setdefault("split_protocol", "random_stratified")
    resolved.setdefault("folds_root", None)

    root_override = overrides.get("root")
    if root_override:
        resolved["root"] = root_override
    if overrides.get("output_dir"):
        resolved["output_dir"] = overrides["output_dir"]
    for key in ["seed", "num_preview", "max_samples_per_class", "mode"]:
        if overrides.get(key) is not None:
            resolved[key] = overrides[key]
    if overrides.get("fakeavceleb_train_ratio") is not None:
        resolved["train_ratio"] = overrides["fakeavceleb_train_ratio"]
    if overrides.get("fakeavceleb_split_protocol") is not None:
        resolved["split_protocol"] = overrides["fakeavceleb_split_protocol"]
    if overrides.get("fakeavceleb_folds_root"):
        resolved["folds_root"] = overrides["fakeavceleb_folds_root"]
    if overrides.get("dry_run"):
        resolved["dry_run"] = True

    missing = [key for key in ["type", "root", "output_dir"] if resolved.get(key) in {None, ""}]
    if missing:
        raise ValueError(f"Dataset '{dataset_name}' is missing required fields: {missing}")

    run = PrepareRun(
        name=dataset_name,
        dataset_type=str(resolved["type"]).lower(),
        root=Path(str(resolved["root"])).expanduser(),
        output_dir=Path(str(resolved["output_dir"])).expanduser(),
        seed=int(resolved["seed"]),
        num_preview=int(resolved["num_preview"]),
        max_samples_per_class=(
            None
            if resolved.get("max_samples_per_class") is None
            else int(resolved["max_samples_per_class"])
        ),
        dry_run=bool(resolved.get("dry_run", False)),
        mode=str(resolved.get("mode", "binary")),
        train_ratio=float(resolved.get("train_ratio", 0.7)),
        split_protocol=str(resolved.get("split_protocol", "random_stratified")),
        folds_root=(
            None
            if resolved.get("folds_root") in {None, ""}
            else Path(str(resolved["folds_root"])).expanduser()
        ),
    )
    validate_prepare_run(run)
    return run


def validate_prepare_run(run: PrepareRun) -> None:
    """
    函数功能：
    - 校验单数据集准备任务中的通用参数和 FakeAVCeleb 专属参数。

    参数：
    - run: resolve_dataset_run 生成的运行参数。

    返回：
    - 无返回值，参数非法时抛出 ValueError。
    """
    if run.max_samples_per_class is not None and run.max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive when provided")
    if run.num_preview < 0:
        raise ValueError("num_preview must be non-negative")
    if run.dataset_type == "fakeavceleb" and not 0 < run.train_ratio < 1:
        raise ValueError("FakeAVCeleb train_ratio must be between 0 and 1")
    if run.dataset_type == "fakeavceleb" and run.mode not in {"binary", "structured", "both"}:
        raise ValueError("FakeAVCeleb mode must be one of: binary, structured, both")
    if run.dataset_type == "fakeavceleb" and run.split_protocol not in {"random_stratified", "mrdf_5fold"}:
        raise ValueError("FakeAVCeleb split_protocol must be one of: random_stratified, mrdf_5fold")
    if run.dataset_type == "fakeavceleb" and run.split_protocol == "mrdf_5fold" and run.folds_root is None:
        raise ValueError("FakeAVCeleb split_protocol=mrdf_5fold requires folds_root")


def build_dataset_outputs(run: PrepareRun, issues: List[Issue]) -> tuple[str, Dict[str, Any], Dict[str, List[Record]]]:
    """
    函数功能：
    - 根据数据集类型构建输出 JSONL 记录。

    参数：
    - run: 单数据集运行参数。
    - issues: 缺失或异常文件记录列表，builder 会在其中追加问题。

    返回：
    - 数据集显示名、扫描摘要、输出文件到记录列表的映射。

    关键逻辑：
    - FakeAVCeleb 和 MAVOS-DD 的差异只保留在各自 builder 中，写文件、统计和日志复用同一流程。
    """
    if run.dataset_type == "fakeavceleb":
        dataset_label = "FakeAVCeleb"
        samples = fakeavceleb.build_fakeavceleb_samples(
            root=run.root,
            max_samples_per_class=run.max_samples_per_class,
            seed=run.seed,
            missing_or_invalid=issues,
        )
        output_args = argparse.Namespace(
            fakeavceleb_train_ratio=run.train_ratio,
            seed=run.seed,
            mode=run.mode,
            split_protocol=run.split_protocol,
            folds_root=str(run.folds_root) if run.folds_root is not None else None,
        )
        if run.split_protocol == "mrdf_5fold":
            outputs = fakeavceleb.build_fakeavceleb_fold_output_records(samples, output_args)
        else:
            outputs = fakeavceleb.build_fakeavceleb_output_records(samples, output_args)
    elif run.dataset_type in {"mavosdd", "mavos-dd"}:
        dataset_label = "MAVOS-DD"
        split_samples = mavosdd.build_mavosdd_samples(
            root=run.root,
            max_samples_per_class=run.max_samples_per_class,
            seed=run.seed,
            missing_or_invalid=issues,
        )
        outputs = mavosdd.build_mavosdd_output_records(split_samples)
    else:
        raise ValueError(f"Unsupported dataset type: {run.dataset_type}")

    scan_summary = {dataset_label: scan_video_files(run.root)}
    return dataset_label, scan_summary, outputs


def prepare_dataset(run: PrepareRun) -> Dict[str, List[Record]]:
    """
    函数功能：
    - 执行单数据集扫描、记录构建、JSONL 写出和统计报告写出。

    参数：
    - run: 单数据集运行参数。

    返回：
    - 输出文件名到记录列表的映射。

    关键逻辑：
    - dry-run 只跳过训练 JSONL，统计、缺失异常报告和预览仍会写出。
    """
    run.output_dir.mkdir(parents=True, exist_ok=True)
    issues: List[Issue] = []
    dataset_label = "FakeAVCeleb" if run.dataset_type == "fakeavceleb" else "MAVOS-DD"
    logging.info("Preparing %s with dry_run=%s", dataset_label, run.dry_run)
    logging.info("%s root: %s", dataset_label, run.root)
    logging.info("Output dir: %s", run.output_dir)
    dataset_label, scan_summary, outputs = build_dataset_outputs(run, issues)

    write_output_jsonl(run.output_dir, outputs, run.dry_run)
    write_stats(run.output_dir, scan_summary, outputs, issues, run.num_preview, run.seed)
    logging.info("Prepared %d %s output jsonl definitions.", len(outputs), dataset_label)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    函数功能：
    - 统一数据准备 CLI 主流程。

    参数：
    - argv: 命令行参数列表，None 时读取真实命令行。

    返回：
    - 进程退出码，0 表示成功，2 表示参数或配置错误。
    """
    setup_logging()
    args = parse_args(argv)
    root_override = args.root
    if args.fakeavceleb_root and args.dataset == "fakeavceleb":
        root_override = args.fakeavceleb_root
    if args.mavos_root and args.dataset in {"mavosdd", "mavos-dd"}:
        root_override = args.mavos_root
    overrides = {
        "root": root_override,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "num_preview": args.num_preview,
        "max_samples_per_class": args.max_samples_per_class,
        "mode": args.mode,
        "fakeavceleb_train_ratio": args.fakeavceleb_train_ratio,
        "fakeavceleb_split_protocol": args.fakeavceleb_split_protocol,
        "fakeavceleb_folds_root": args.fakeavceleb_folds_root,
        "dry_run": args.dry_run,
    }
    try:
        config = load_prepare_config(args.config)
        run = resolve_dataset_run(config, args.dataset, overrides)
        prepare_dataset(run)
    except Exception as exc:
        logging.error("%s", exc)
        return 2
    return 0
