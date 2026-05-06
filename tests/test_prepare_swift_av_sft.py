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


def load_prepare_module():
    script_path = SCRIPTS_DIR / "prepare_swift_av_sft.py"
    spec = importlib.util.spec_from_file_location("prepare_swift_av_sft", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareSwiftAvSftTests(unittest.TestCase):
    def setUp(self):
        self.prepare = load_prepare_module()

    def test_unified_entry_selects_fakeavceleb_from_yaml_config(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_unified_{uuid.uuid4().hex}"
        output_dir = scratch / f"fakeavceleb_output_{uuid.uuid4().hex}"
        config_path = scratch / f"prepare_config_{uuid.uuid4().hex}.yaml"
        for dirname in self.prepare.FAKEAVCELEB_CATEGORIES:
            (root / dirname).mkdir(parents=True)
        (root / "RealVideo-RealAudio" / "real.mp4").write_bytes(b"video")
        (root / "FakeVideo-FakeAudio" / "fake.mp4").write_bytes(b"video")
        config_path.write_text(
            "\n".join(
                [
                    "defaults:",
                    "  seed: 42",
                    "  num_preview: 2",
                    "datasets:",
                    "  fakeavceleb:",
                    "    type: fakeavceleb",
                    f"    root: {root.as_posix()}",
                    f"    output_dir: {output_dir.as_posix()}",
                    "    mode: both",
                    "    train_ratio: 0.7",
                    "  mavosdd:",
                    "    type: mavosdd",
                    "    root: /does/not/matter",
                    "    output_dir: /does/not/matter",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        exit_code = self.prepare.main(
            [
                "--config",
                str(config_path),
                "--dataset",
                "fakeavceleb",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "fakeavceleb_binary_train.jsonl").exists())
        self.assertTrue((output_dir / "fakeavceleb_structured_train.jsonl").exists())
        self.assertFalse((output_dir / "mavosdd_binary_train.jsonl").exists())
        stats = json.loads((output_dir / "dataset_stats.json").read_text(encoding="utf-8"))
        self.assertTrue(all(name.startswith("fakeavceleb") for name in stats["outputs"]))

    def test_unified_entry_selects_mavosdd_from_yaml_config(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"mavosdd_unified_{uuid.uuid4().hex}"
        output_dir = scratch / f"mavosdd_output_{uuid.uuid4().hex}"
        config_path = scratch / f"prepare_config_{uuid.uuid4().hex}.yaml"
        (root / "english").mkdir(parents=True)
        video_path = root / "english" / "real.mp4"
        video_path.write_bytes(b"video")
        config_path.write_text(
            "\n".join(
                [
                    "defaults:",
                    "  seed: 42",
                    "  num_preview: 2",
                    "datasets:",
                    "  fakeavceleb:",
                    "    type: fakeavceleb",
                    "    root: /does/not/matter",
                    "    output_dir: /does/not/matter",
                    "  mavosdd:",
                    "    type: mavosdd",
                    f"    root: {root.as_posix()}",
                    f"    output_dir: {output_dir.as_posix()}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "video_path": "english/real.mp4",
                "label": "real",
                "split": "train",
                "language": "english",
                "generative_method": "",
                "open_set_model": False,
                "open_set_language": False,
            }
        ]

        with mock.patch.object(self.prepare.mavosdd, "load_mavos_dataset", return_value=rows):
            exit_code = self.prepare.main(
                [
                    "--config",
                    str(config_path),
                    "--dataset",
                    "mavosdd",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_dir / "mavosdd_binary_train.jsonl").exists())
        self.assertFalse((output_dir / "fakeavceleb_binary_train.jsonl").exists())
        stats = json.loads((output_dir / "dataset_stats.json").read_text(encoding="utf-8"))
        self.assertTrue(all(name.startswith("mavosdd") for name in stats["outputs"]))

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

    def test_build_fakeavceleb_samples_resolves_metadata_directory_columns(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_meta_dirs_{uuid.uuid4().hex}"
        for dirname in self.prepare.FAKEAVCELEB_CATEGORIES:
            (root / dirname).mkdir(parents=True)
        video_dir = root / "RealVideo-RealAudio" / "African" / "men" / "id00076"
        video_dir.mkdir(parents=True, exist_ok=True)
        valid_video = video_dir / "00109.mp4"
        valid_video.write_bytes(b"video")
        (root / "meta_data.csv").write_text(
            "source,target1,target2,method,category,type,race,gender,path\n"
            "id00076,-,-,real,A,RealVideo-RealAudio,African,men,00109.mp4\n",
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
        self.assertEqual(samples[0]["meta"]["source_metadata"]["source"], "id00076")
        self.assertFalse(
            any(item["reason"] == "missing_file" for item in missing_or_invalid),
            missing_or_invalid,
        )

    def test_build_fakeavceleb_samples_stores_subject_id_for_fold_protocol(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_subject_{uuid.uuid4().hex}"
        for dirname in self.prepare.FAKEAVCELEB_CATEGORIES:
            (root / dirname).mkdir(parents=True)
        video_dir = root / "RealVideo-RealAudio" / "African" / "men" / "id00076"
        video_dir.mkdir(parents=True, exist_ok=True)
        valid_video = video_dir / "00109.mp4"
        valid_video.write_bytes(b"video")
        (root / "meta_data.csv").write_text(
            "source,target1,target2,method,category,type,race,gender,path\n"
            "id00076,-,-,real,A,RealVideo-RealAudio,African,men,00109.mp4\n",
            encoding="utf-8",
        )

        samples = self.prepare.build_fakeavceleb_samples(
            root=root,
            max_samples_per_class=None,
            seed=42,
            missing_or_invalid=[],
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["meta"]["subject_id"], "id00076")

    def test_build_fakeavceleb_fold_output_records_uses_subject_independent_split_files(self):
        scratch = Path(__file__).resolve().parent / ".tmp"
        scratch.mkdir(exist_ok=True)
        root = scratch / f"fakeavceleb_fold_{uuid.uuid4().hex}"
        folds_root = scratch / f"fakeavceleb_folds_{uuid.uuid4().hex}"
        folds_root.mkdir(parents=True, exist_ok=True)
        for dirname in self.prepare.FAKEAVCELEB_CATEGORIES:
            (root / dirname).mkdir(parents=True)

        subjects = {
            "id00076": ("RealVideo-RealAudio", "Real"),
            "id02005": ("FakeVideo-FakeAudio", "Fake"),
        }
        for subject_id, (dirname, _) in subjects.items():
            video_dir = root / dirname / "African" / "men" / subject_id
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / "00001.mp4").write_bytes(b"video")

        (root / "meta_data.csv").write_text(
            "\n".join(
                [
                    "source,target1,target2,method,category,type,race,gender,path",
                    "id00076,-,-,real,A,RealVideo-RealAudio,African,men,00001.mp4",
                    "id02005,-,-,fake,A,FakeVideo-FakeAudio,African,men,00001.mp4",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        for fold_idx in range(1, 6):
            train_ids = ["id00076"] if fold_idx == 1 else ["id02005"]
            test_ids = ["id02005"] if fold_idx == 1 else ["id00076"]
            (folds_root / f"train_{fold_idx}.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
            (folds_root / f"test_{fold_idx}.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

        samples = self.prepare.build_fakeavceleb_samples(
            root=root,
            max_samples_per_class=None,
            seed=42,
            missing_or_invalid=[],
        )
        outputs = self.prepare.build_fakeavceleb_fold_output_records(
            samples,
            mock.Mock(folds_root=str(folds_root), mode="binary"),
        )

        self.assertIn("fakeavceleb_mrdf5fold_fold1_binary_train.jsonl", outputs)
        self.assertIn("fakeavceleb_mrdf5fold_fold1_binary_test.jsonl", outputs)
        self.assertEqual(len(outputs["fakeavceleb_mrdf5fold_fold1_binary_train.jsonl"]), 1)
        self.assertEqual(len(outputs["fakeavceleb_mrdf5fold_fold1_binary_test.jsonl"]), 1)
        train_meta = outputs["fakeavceleb_mrdf5fold_fold1_binary_train.jsonl"][0]["meta"]
        test_meta = outputs["fakeavceleb_mrdf5fold_fold1_binary_test.jsonl"][0]["meta"]
        self.assertEqual(train_meta["subject_id"], "id00076")
        self.assertEqual(test_meta["subject_id"], "id02005")

    def test_make_binary_record_includes_audios_when_sample_has_audio_path(self):
        sample = {
            "video_path": str(Path("clip.mp4").resolve()),
            "audio_path": str(Path("clip.wav").resolve()),
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

        self.assertEqual(record["audios"], [sample["audio_path"]])
        self.assertIn("<audio>", record["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
