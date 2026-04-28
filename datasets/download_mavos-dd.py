import os
import time
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/OneDay/.hf_cache"
os.environ["HF_HUB_CACHE"] = "/data/OneDay/.hf_cache/hub"

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
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            endpoint="https://hf-mirror.com",
            allow_patterns=allow_patterns,
            max_workers=4,
            force_download=False,
        )
        print("done")
        break
    except Exception as e:
        print(f"[Attempt {attempt}] failed: {repr(e)}")
        print("sleep 60s and retry...")
        time.sleep(60)
else:
    raise RuntimeError("MAVOS-DD download failed after many retries.")