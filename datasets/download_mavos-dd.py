import os
import time
from pathlib import Path


os.environ["HF_HOME"] = "/data/OneDay/.hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/OneDay/.hf_cache/hub"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


from huggingface_hub import snapshot_download

repo_id = "unibuc-cs/MAVOS-DD"
local_dir = "/data/OneDay/MAVOS-DD"

allow_patterns = [
    "README.md",
    "dataset_info.json",
    "state.json",
    "data-00000-of-00001.arrow",
    "dataset.py",
    "metadata_generation.py",
    "arabic/**",
    "english/**",
    "german/**",
    "hindi/**",
    "mandarin/**",
    "romanian/**",
    "russian/**",
    "spanish/**",
]

for attempt in range(1, 20):
    try:
        print(f"[Attempt {attempt}] downloading MAVOS-DD...")
        out = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            max_workers=8,
            force_download=False,
            etag_timeout=120,
        )
        print(f"snapshot_download returned: {out}")

        # 简单检查每个语言目录是否有 mp4 文件
        root = Path(local_dir)
        languages = [
            "arabic", "english", "german", "hindi",
            "mandarin", "romanian", "russian", "spanish"
        ]

        bad_langs = []
        for lang in languages:
            lang_dir = root / lang
            mp4_count = len(list(lang_dir.rglob("*.mp4"))) if lang_dir.exists() else 0
            print(f"{lang}: {mp4_count} mp4 files")
            if mp4_count == 0:
                bad_langs.append(lang)

        if bad_langs:
            raise RuntimeError(f"These language dirs seem incomplete or empty: {bad_langs}")

        print("download/check done")
        break

    except Exception as e:
        print(f"[Attempt {attempt}] failed: {repr(e)}")
        print("sleep 60s and retry...")
        time.sleep(60)
else:
    raise RuntimeError("MAVOS-DD download failed after many retries.")