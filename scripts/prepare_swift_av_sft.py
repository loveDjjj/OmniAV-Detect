#!/usr/bin/env python3
"""Compatibility module for audio-video SFT preparation helpers.

Use the dataset-specific entry points instead:
  - scripts/prepare_fakeavceleb_swift_sft.py
  - scripts/prepare_mavosdd_swift_sft.py
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence

from prepare_fakeavceleb_swift_sft import (
    FAKEAVCELEB_CATEGORIES,
    build_fakeavceleb_samples,
    stratified_split,
)
from prepare_mavosdd_swift_sft import MAVOS_OUTPUT_SPLITS, build_mavosdd_samples
from prepare_swift_av_sft_common import (
    BINARY_USER_PROMPT,
    STRUCTURED_USER_PROMPT,
    SYSTEM_PROMPT,
    build_preview_samples,
    build_stats,
    clean_text,
    count_meta,
    make_binary_record,
    make_messages,
    make_structured_evidence,
    make_structured_record,
    scan_video_files,
    setup_logging,
    write_json,
    write_jsonl,
    write_missing_or_invalid,
    write_output_jsonl,
    write_stats,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    logging.error(
        "The combined prepare_swift_av_sft.py entry point no longer processes multiple datasets. "
        "Use scripts/prepare_fakeavceleb_swift_sft.py or scripts/prepare_mavosdd_swift_sft.py."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
