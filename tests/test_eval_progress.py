import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from omniav_detect.evaluation.progress import ProgressProxy, count_batches, create_progress


class EvalProgressTests(unittest.TestCase):
    def test_count_batches_handles_tail_batch(self):
        self.assertEqual(count_batches(0, 4), 0)
        self.assertEqual(count_batches(1, 4), 1)
        self.assertEqual(count_batches(8, 4), 2)
        self.assertEqual(count_batches(9, 4), 3)

    def test_create_progress_falls_back_when_tqdm_is_missing(self):
        old_tqdm = sys.modules.get("tqdm.auto")
        sys.modules["tqdm.auto"] = None
        try:
            progress = create_progress(total=3, desc="Eval", unit="batch")
        finally:
            if old_tqdm is None:
                sys.modules.pop("tqdm.auto", None)
            else:
                sys.modules["tqdm.auto"] = old_tqdm

        self.assertIsInstance(progress, ProgressProxy)
        self.assertEqual(progress.n, 0)
        progress.update(2)
        self.assertEqual(progress.n, 2)
        progress.close()


if __name__ == "__main__":
    unittest.main()
