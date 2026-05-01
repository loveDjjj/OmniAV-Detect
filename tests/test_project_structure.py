import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_src_python_files_do_not_exceed_500_lines(self):
        oversized = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 500:
                oversized.append((path.relative_to(REPO_ROOT).as_posix(), line_count))

        self.assertEqual(oversized, [])

    def test_agents_documents_python_file_line_limit(self):
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("单个 Python 代码文件不应超过 500 行", text)


if __name__ == "__main__":
    unittest.main()
