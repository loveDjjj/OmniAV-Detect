import importlib.util
import json
import unittest
import uuid
from pathlib import Path
from unittest import mock


def load_prepare_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_swift_av_sft.py"
    spec = importlib.util.spec_from_file_location("prepare_swift_av_sft", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareSwiftAvSftTests(unittest.TestCase):
    def setUp(self):
        self.prepare = load_prepare_module()

    def test_binary_record_matches_ms_swift_video_format(self):
        sample = {
            "video_path": str(Path("clip.mp4").resolve()),
            "meta": {
                "dataset": "FakeAVCeleb",
                "source_path": str(Path("clip.mp4").resolve()),
                "overall_label": "Real",
                "video_label": "Real",
                "audio_label": "Real",
                "modality_type": "R-R",
                "language": "",
                "generative_method": "",
                "original_split": "",
            },
        }

        record = self.prepare.make_binary_record(sample)

        self.assertEqual(record["messages"][0]["role"], "system")
        self.assertIn("<video>", record["messages"][1]["content"])
        self.assertEqual(record["messages"][2], {"role": "assistant", "content": "Real"})
        self.assertEqual(record["videos"], [sample["video_path"]])
        self.assertEqual(record["meta"]["modality_type"], "R-R")

    def test_structured_record_has_json_string_assistant_content(self):
        sample = {
            "video_path": str(Path("clip.mp4").resolve()),
            "meta": {
                "dataset": "FakeAVCeleb",
                "source_path": str(Path("clip.mp4").resolve()),
                "overall_label": "Fake",
                "video_label": "Real",
                "audio_label": "Fake",
                "modality_type": "R-F",
                "language": "",
                "generative_method": "",
                "original_split": "",
            },
        }

        record = self.prepare.make_structured_record(sample)
        assistant_content = record["messages"][2]["content"]
        parsed = json.loads(assistant_content)

        self.assertIsInstance(assistant_content, str)
        self.assertEqual(parsed["overall_label"], "Fake")
        self.assertEqual(parsed["video_label"], "Real")
        self.assertEqual(parsed["audio_label"], "Fake")
        self.assertEqual(parsed["modality_type"], "R-F")
        self.assertIn("audio modality is fake", parsed["evidence"])

    def test_stratified_split_preserves_overall_label_and_modality_groups(self):
        samples = []
        for label, modality in [("Real", "R-R"), ("Fake", "R-F"), ("Fake", "F-R")]:
            for idx in range(10):
                samples.append({"meta": {"overall_label": label, "modality_type": modality}, "id": idx})

        train, eval_samples = self.prepare.stratified_split(samples, 0.7, seed=7)

        def counts(items):
            result = {}
            for item in items:
                key = (item["meta"]["overall_label"], item["meta"]["modality_type"])
                result[key] = result.get(key, 0) + 1
            return result

        self.assertEqual(counts(train), {("Real", "R-R"): 7, ("Fake", "R-F"): 7, ("Fake", "F-R"): 7})
        self.assertEqual(counts(eval_samples), {("Real", "R-R"): 3, ("Fake", "R-F"): 3, ("Fake", "F-R"): 3})

    def test_build_fakeavceleb_samples_skips_zero_size_and_invalid_extension(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_{uuid.uuid4().hex}"
        real_dir = root / "RealVideo-RealAudio"
        fake_dir = root / "FakeVideo-FakeAudio"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        (root / "RealVideo-FakeAudio").mkdir()
        (root / "FakeVideo-RealAudio").mkdir()
        valid_real = real_dir / "real.mp4"
        valid_fake = fake_dir / "fake.webm"
        empty_video = fake_dir / "empty.mp4"
        invalid_ext = real_dir / "note.txt"
        valid_real.write_bytes(b"video")
        valid_fake.write_bytes(b"video")
        empty_video.write_bytes(b"")
        invalid_ext.write_text("not a video", encoding="utf-8")
        missing_or_invalid = []

        with mock.patch.dict("sys.modules", {"pandas": None}):
            samples = self.prepare.build_fakeavceleb_samples(
                root=root,
                max_samples_per_class=None,
                seed=42,
                missing_or_invalid=missing_or_invalid,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual({sample["meta"]["overall_label"] for sample in samples}, {"Real", "Fake"})
        self.assertTrue(any(item["reason"] == "zero_size_file" for item in missing_or_invalid))
        self.assertTrue(any(item["reason"] == "invalid_extension" for item in missing_or_invalid))

    def test_build_fakeavceleb_samples_reports_metadata_missing_files(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_meta_{uuid.uuid4().hex}"
        for dirname in self.prepare.FAKEAVCELEB_CATEGORIES:
            (root / dirname).mkdir(parents=True)
        valid_video = root / "RealVideo-RealAudio" / "present.mp4"
        valid_video.write_bytes(b"video")
        (root / "meta_data.csv").write_text(
            "video_path,label\n"
            "RealVideo-RealAudio/present.mp4,real\n"
            "RealVideo-RealAudio/missing.mp4,real\n",
            encoding="utf-8",
        )
        missing_or_invalid = []

        samples = self.prepare.build_fakeavceleb_samples(
            root=root,
            max_samples_per_class=None,
            seed=42,
            missing_or_invalid=missing_or_invalid,
        )

        self.assertEqual(len(samples), 1)
        self.assertTrue(
            any(
                item["reason"] == "missing_file"
                and Path(item["expected_path"]).name == "missing.mp4"
                and "RealVideo-RealAudio" in Path(item["expected_path"]).parts
                for item in missing_or_invalid
            )
        )


if __name__ == "__main__":
    unittest.main()
