"""Reject altered real publication evidence while retaining hash-algorithm boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from audit_publication_benchmark import audit

RESULTS = Path(__file__).resolve().parent / "runs/publication-01/shared-results"


@unittest.skipUnless(RESULTS.is_dir(), "Requires collected publication-01 artifacts")
class PublicationAuditTest(unittest.TestCase):
    def test_original_evidence_and_distinct_hash_algorithms(self):
        report = audit(RESULTS)
        self.assertTrue(report["passed"], report["errors"])
        self.assertFalse(report["engine_hash_binding"]["cross_algorithm_digest_equality_claim"])

    def altered(self, relative, mutate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in RESULTS.rglob("*"):
                target = root / source.relative_to(RESULTS)
                if source.is_dir():
                    target.mkdir()
                else:
                    target.symlink_to(source)
            target = root / relative
            value = json.loads(target.read_text())
            mutate(value)
            target.unlink()
            target.write_text(json.dumps(value))
            report = audit(root)
            self.assertFalse(report["passed"])
            return report["errors"]

    def test_changed_terminal_learner_state_is_rejected(self):
        def change(state):
            name = next(iter(state["param_hashes"]))
            state["param_hashes"][name] = "0" * 64
        errors = self.altered("publication-benchmark/local-reference-rank-0.json", change)
        self.assertIn("Benchmark changed final model/Adam state 0", errors)

    def test_changed_raw_engine_digest_is_rejected(self):
        def change(states):
            name = next(iter(states[0]))
            for state in states:
                state[name] = "0" * 64
        errors = self.altered("publication-benchmark/engine-reference.json", change)
        self.assertTrue(any("State content changed" in error for error in errors))

    def test_missing_engine_tensor_coverage_is_rejected(self):
        def change(states):
            name = next(iter(states[0]))
            for state in states:
                del state[name]
        errors = self.altered("publication-benchmark/engine-reference.json", change)
        self.assertIn("Final publication differs from benchmark tensor-name coverage", errors)

    def test_altered_critical_rank_timing_is_rejected(self):
        def change(report):
            report["iterations"][0]["measured_seconds"] += 10
        errors = self.altered("publication-benchmark/rank-0.json", change)
        self.assertIn("Recomputed critical-rank timing 0", errors)


if __name__ == "__main__":
    unittest.main()
