import json
import shutil
import sys
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class MvadPrepareTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parent / ".tmp" / f"mvad_{uuid.uuid4().hex}"
        self.scratch.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def test_parse_mvad_video_path_maps_four_modalities(self):
        from mvad.common import parse_video_sample

        cases = {
            "train/real_real/talkvid/talk001.mp4": ("Real", "Real", "Real", "R-R"),
            "train/real_fake/msrvtt/msrvtt_AudioX/clip_001_AudioX.mp4": ("Fake", "Real", "Fake", "R-F"),
            "train/fake_real/Humo/humo_001.mp4": ("Fake", "Fake", "Real", "F-R"),
            "train/fake_fake/indirect/pika/pika_AudioX/clip_001_AudioX.mp4": ("Fake", "Fake", "Fake", "F-F"),
        }

        for relative_path, expected in cases.items():
            sample = parse_video_sample(Path("/data/MVAD/unpacked") / relative_path, Path("/data/MVAD/unpacked"))
            meta = sample["meta"]
            self.assertEqual(
                (meta["overall_label"], meta["video_label"], meta["audio_label"], meta["modality_type"]),
                expected,
            )

    def test_group_id_removes_audio_generator_suffix_for_derived_audio(self):
        from mvad.common import parse_video_sample

        root = Path("/data/MVAD/unpacked")
        first = parse_video_sample(
            root / "train" / "real_fake" / "msrvtt" / "msrvtt_AudioX" / "clip_001_AudioX.mp4",
            root,
        )
        second = parse_video_sample(
            root / "train" / "real_fake" / "msrvtt" / "msrvtt_MMAudio" / "clip_001_MMAudio.mp4",
            root,
        )
        direct = parse_video_sample(
            root / "train" / "fake_fake" / "direct" / "Sora2" / "sora_001.mp4",
            root,
        )

        self.assertEqual(first["meta"]["group_id"], second["meta"]["group_id"])
        self.assertNotEqual(first["meta"]["group_id"], direct["meta"]["group_id"])

    def test_group_aware_split_keeps_group_on_one_side(self):
        from mvad.build_index_and_split import group_aware_split

        samples = []
        for group_idx in range(10):
            for variant in ("AudioX", "MMAudio"):
                samples.append(
                    {
                        "video_path": f"/videos/g{group_idx}_{variant}.mp4",
                        "meta": {
                            "group_id": f"group-{group_idx}",
                            "modality_type": "R-F",
                            "overall_label": "Fake",
                        },
                    }
                )

        train, val = group_aware_split(samples, val_ratio=0.3, seed=7)
        train_groups = {item["meta"]["group_id"] for item in train}
        val_groups = {item["meta"]["group_id"] for item in val}

        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_build_records_adds_audios_and_explicit_audio_prompt(self):
        from mvad.build_av_jsonl import build_jsonl_records

        video_path = self.scratch / "train" / "real_real" / "talkvid" / "talk001.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video")
        samples = [
            {
                "video_path": str(video_path.resolve()),
                "meta": {
                    "dataset": "MVAD",
                    "overall_label": "Real",
                    "video_label": "Real",
                    "audio_label": "Real",
                    "modality_type": "R-R",
                    "relative_path": "train/real_real/talkvid/talk001.mp4",
                },
            }
        ]

        records = build_jsonl_records(samples, self.scratch / "audio", dry_run=True)

        self.assertEqual(len(records), 1)
        self.assertIn("audios", records[0])
        self.assertIn("<audio>", records[0]["messages"][1]["content"])
        self.assertEqual(records[0]["messages"][2]["content"], "Real")

    def test_unzip_archives_writes_manifest(self):
        from mvad.unzip_archives import unpack_archives

        source_root = self.scratch / "raw"
        archive_dir = source_root / "train" / "real_real" / "talkvid"
        archive_dir.mkdir(parents=True)
        archive_path = archive_dir / "talkvid.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("talk001.mp4", b"video")

        manifest = unpack_archives(source_root, self.scratch / "unpacked", overwrite=False)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["status"], "extracted")
        self.assertTrue((self.scratch / "unpacked" / "train" / "real_real" / "talkvid" / "talkvid" / "talk001.mp4").exists())

    def test_unzip_archives_uses_7z_extractor(self):
        from mvad.unzip_archives import unpack_archives

        source_root = self.scratch / "raw"
        archive_dir = source_root / "train" / "real_real" / "talkvid"
        archive_dir.mkdir(parents=True)
        archive_path = archive_dir / "talkvid.zip"
        archive_path.write_bytes(b"zip")

        def fake_run(command, check, stdout, stderr, text):
            target_arg = next(item for item in command if item.startswith("-o"))
            target_dir = Path(target_arg[2:])
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "talk001.mp4").write_bytes(b"video")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run) as run_mock:
            manifest = unpack_archives(source_root, self.scratch / "unpacked", overwrite=False, extractor="7z")

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "7z")
        self.assertIn("x", command)
        self.assertEqual(manifest[0]["extractor"], "7z")


if __name__ == "__main__":
    unittest.main()
