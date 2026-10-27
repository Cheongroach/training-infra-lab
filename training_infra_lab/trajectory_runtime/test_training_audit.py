import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from .audit_training import analyze
from .contracts import Trajectory, UpdateBatch


class TrainingAuditTests(unittest.TestCase):
    def fixture(self, root, wrong_finish="length"):
        batch = UpdateBatch(tuple(
            Trajectory(str(i), "prompt", 0, (1, 2, 3), (0, 1), (0., -1.), float(i == 0))
            for i in range(2)
        ), 0, 0, 2)
        diagnostics = {"batch_sha256": batch.sha256, "signal_groups": 1,
                       "probability_replay": {"rms": 0., "p99_abs": 0., "outside_clip_fraction": 0.}}
        history = dict(diagnostics, update=0, update_result={
            "rank": 0, "update_id": 0, "batch_sha256": batch.sha256,
            "grad_norm": 1., "total_seconds": 2., "peak_allocated_bytes": 1024})
        rows = [{"trajectory_id": str(i), "reward": float(i == 0), "prompt_tokens": 2,
                 "response_tokens": 1, "finish_reason": "eos" if i == 0 else wrong_finish}
                for i in range(2)]
        for name, data in {
            "environment-rank0.json": {"world": 1, "updates": 1, "groups_per_rank": 1,
                                        "resident_rollout": True},
            "batch-0.json": asdict(batch),
            "batch-diagnostics-0-rank0.json": diagnostics,
            "history-rank0.json": [history],
            "rollout-0-rank0.json": [{"phase_seconds": 5., "residency": {"admitted": True},
                                     "trajectories": rows}],
        }.items():
            (root / name).write_text(json.dumps(data))

    def test_truncated_wrong_is_not_completed_wrong_signal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            report = analyze(root)
            self.assertTrue(report["all_updates_have_nonzero_gradients"])
            self.assertFalse(report["all_updates_have_completed_wrong_signal"])
            self.assertEqual(report["updates"][0]["termination_and_reward"]["length_wrong"], 1)
            self.fixture(root, wrong_finish="eos")
            self.assertTrue(analyze(root)["all_updates_have_completed_wrong_signal"])

    def test_observed_zero_gradient_is_not_overridden_by_advantage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            path = root / "history-rank0.json"
            rows = json.loads(path.read_text())
            rows[0]["update_result"]["grad_norm"] = 0.
            path.write_text(json.dumps(rows))
            self.assertFalse(analyze(root)["all_updates_have_nonzero_gradients"])

    def test_strict_reward_is_recomputed_and_format_failure_is_distinct(self):
        from .rewards import STRICT_RULE

        with tempfile.TemporaryDirectory() as name:
            artifact = Path(name)
            root = artifact / "results/grpo"
            root.mkdir(parents=True)
            (artifact / "code").mkdir()
            (artifact / "code/data.jsonl").write_text(json.dumps({
                "source_id": "prompt", "answer": "#### 42"}) + "\n")
            self.fixture(root, wrong_finish="eos")
            path = root / "environment-rank0.json"
            env = json.loads(path.read_text())
            env["reward_rule"] = STRICT_RULE
            path.write_text(json.dumps(env))
            path = root / "rollout-0-rank0.json"
            groups = json.loads(path.read_text())
            for row, answer in zip(groups[0]["trajectories"], (42, 41)):
                row.update(text=f"Final answer: {answer}", parsed_answer=str(answer),
                           reward_outcome="correct" if answer == 42 else "incorrect")
            path.write_text(json.dumps(groups))
            self.assertTrue(analyze(root)["all_updates_have_completed_wrong_signal"])
            groups[0]["trajectories"][1].update(text="Intermediate value 42", parsed_answer=None,
                                               reward_outcome="missing_final_answer")
            path.write_text(json.dumps(groups))
            self.assertFalse(analyze(root)["all_updates_have_completed_wrong_signal"])
            groups[0]["trajectories"][1]["reward"] = 1.
            path.write_text(json.dumps(groups))
            with self.assertRaisesRegex(ValueError, "independently recomputed"):
                analyze(root)

    def test_misaligned_rollout_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            path = root / "rollout-0-rank0.json"
            rows = json.loads(path.read_text())
            rows[0]["trajectories"][0]["trajectory_id"] = "unknown"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "identities differ"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
