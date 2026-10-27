"""Reject incomplete or internally inconsistent evidence before reporting speed."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from .analyze_replay import analyze
from .contracts import Trajectory, UpdateBatch
from .planner import plan_update


class AnalysisTests(unittest.TestCase):
    def fixture(self, root):
        batch = UpdateBatch(tuple(
            Trajectory(str(i), str(i // 2), 1, (2, 3, 4), (0, 1), (0.0, -1.0), float(i % 2))
            for i in range(8)
        ), 1, 1, 2)
        (root / "batch-1.json").write_text(json.dumps(asdict(batch)))
        for rank in range(2):
            rows = []
            for repeat in range(3):
                for strategy in ("verl", "slot_sorted"):
                    for micro in (1, 2):
                        plan = plan_update(batch, 2, strategy=strategy, micro_batch=micro)
                        rows.append({
                            "repeat": repeat, "strategy": strategy, "micro_batch": micro,
                            "plan": asdict(plan), "plan_sha256": plan.sha256, "batch_sha256": batch.sha256,
                            "seconds": 1.0, "total_seconds": 1.0, "peak_allocated_bytes": 1024,
                            "input_tokens": 24, "response_tokens": 8,
                            "reference": repeat == 0 and strategy == "verl" and micro == 1,
                            "numerically_accepted": True,
                            "errors": {k: {"relative_l2": 0.0, "max_abs": 0.0}
                                       for k in ("gradient", "exp_avg", "exp_avg_sq", "parameter")},
                        })
            for name, data in (
                (f"environment-rank{rank}.json", {"world": 2, "seed_per_rank": rank}),
                (f"batch-diagnostics-1-rank{rank}.json", {
                    "batch_sha256": batch.sha256, "signal_groups": 4, "groups": 4, "mean_reward": 0.5,
                }),
                (f"history-rank{rank}.json", [{"update": 0}, {"update": 1}]),
                (f"replay-rank{rank}.json", rows),
            ):
                (root / name).write_text(json.dumps(data))

    def test_complete_fixture_and_shared_missing_measurement(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            self.assertTrue(analyze(root)["useful_gradient_test"])
            for rank in range(2):
                path = root / f"replay-rank{rank}.json"
                path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
            with self.assertRaisesRegex(ValueError, "matrix"):
                analyze(root)

    def test_restricted_matrix_requires_explicit_environment(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            for rank in range(2):
                path = root / f"replay-rank{rank}.json"
                rows = json.loads(path.read_text())
                path.write_text(json.dumps([r for r in rows if r["micro_batch"] == 1]))
            with self.assertRaisesRegex(ValueError, "prescribed"):
                analyze(root)
            for rank in range(2):
                path = root / f"environment-rank{rank}.json"
                data = json.loads(path.read_text())
                data["replay_microbatches"] = [1]
                path.write_text(json.dumps(data))
            self.assertEqual(len(analyze(root)["paired_comparisons"]), 1)

    def test_accept_flag_cannot_hide_nonfinite_errors(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            path = root / "replay-rank1.json"
            rows = json.loads(path.read_text())
            rows[1]["errors"]["gradient"]["relative_l2"] = float("nan")
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "error metrics"):
                analyze(root)

    def test_finished_replay_without_final_update_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            (root / "history-rank1.json").write_text('[{"update": 0}]')
            with self.assertRaisesRegex(ValueError, "synchronous updates"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
