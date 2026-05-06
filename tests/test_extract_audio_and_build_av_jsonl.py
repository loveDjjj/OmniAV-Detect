import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_audio_script_module():
    script_path = SCRIPTS_DIR / "extract_audio_and_build_av_jsonl.py"
    spec = importlib.util.spec_from_file_location("extract_audio_and_build_av_jsonl", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExtractAudioAndBuildAvJsonlTests(unittest.TestCase):
    def setUp(self):
        self.module = load_audio_script_module()

    def test_iter_augmented_rows_adds_audios_field(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        video_path = scratch / f"video_{uuid.uuid4().hex}.mp4"
        video_path.write_bytes(b"video")
        input_jsonl = scratch / f"input_{uuid.uuid4().hex}.jsonl"
        row = {
            "messages": [],
            "videos": [str(video_path.resolve())],
            "meta": {"overall_label": "Real"},
            "_line_number": 1,
        }
        args = mock.Mock(
            audio_root=str(scratch / "audios"),
            audio_ext=".wav",
            sample_rate=16000,
            audio_channels=1,
            ffmpeg="ffmpeg",
            overwrite=False,
            dry_run=True,
        )

        augmented = self.module.iter_augmented_rows([row], args, input_jsonl)

        self.assertEqual(len(augmented), 1)
        self.assertIn("audios", augmented[0])
        self.assertTrue(augmented[0]["audios"][0].endswith(".wav"))
        self.assertEqual(augmented[0]["videos"], [str(video_path.resolve())])

    def test_main_writes_augmented_jsonl(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        video_path = scratch / f"video_{uuid.uuid4().hex}.mp4"
        video_path.write_bytes(b"video")
        input_jsonl = scratch / f"input_{uuid.uuid4().hex}.jsonl"
        output_jsonl = scratch / f"output_{uuid.uuid4().hex}.jsonl"
        input_jsonl.write_text(
            json.dumps(
                {
                    "messages": [],
                    "videos": [str(video_path.resolve())],
                    "meta": {"overall_label": "Fake"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        exit_code = self.module.main(
            [
                "--input_jsonl",
                str(input_jsonl),
                "--output_jsonl",
                str(output_jsonl),
                "--audio_root",
                str(scratch / "audios"),
                "--dry_run",
            ]
        )

        self.assertEqual(exit_code, 0)
        rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertIn("audios", rows[0])
        self.assertEqual(rows[0]["meta"]["overall_label"], "Fake")


if __name__ == "__main__":
    unittest.main()
