import json
import tempfile
import unittest
from pathlib import Path

from .analyze_residency import analyze


class ResidencyAnalysisTests(unittest.TestCase):
    def fixture(self, root):
        for rank in range(2):
            rows = []
            for repeat in range(-1, 3):
                for enabled in (False, True):
                    rows.append({"rank": rank, "repeat": repeat, "warmup": repeat < 0,
                                 "residency": {"requested": enabled, "admitted": enabled},
                                 "exact_trajectories_and_log_probs": True,
                                 "seconds": 1000 if repeat < 0 else (1 if enabled else 2),
                                 "peak_allocated_bytes": 1024, "max_new_tokens": 128,
                                 "details": {"trajectories": [{"response_tokens": 4, "text": "42"}],
                                             "decode_steps": 4, "wasted_decode_rows": 0,
                                             "sampling": {"temperature": 1}, "reward_rule": "exact"}})
            (root / f"rank{rank}.json").write_text(json.dumps(rows))
            for enabled in (False, True):
                path = root / f"profile-resident{enabled}-rank{rank}"
                path.mkdir()
                (path / "operators.json").write_text(json.dumps({
                    "rank": rank, "identity": {"resident": enabled, "max_new_tokens": 16},
                    "update_result": {"decode_steps": 16}, "operators": [],
                }))

    def test_warmup_exclusion_and_complete_pair_requirement(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            result = analyze(root, 2)
            self.assertEqual(result["median_paired_speedup"], 2)
            self.assertEqual(result["paired_samples"][0]["resident"]["tokens_per_second"], 8)
            path = root / "rank1.json"
            rows = json.loads(path.read_text())
            path.write_text(json.dumps(rows[:-1]))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                analyze(root, 2)

    def test_true_flag_does_not_override_observed_workload_change(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            path = root / "rank1.json"
            rows = json.loads(path.read_text())
            rows[-1]["details"]["trajectories"][0]["response_tokens"] = 3
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "workload differs"):
                analyze(root, 2)


if __name__ == "__main__":
    unittest.main()
