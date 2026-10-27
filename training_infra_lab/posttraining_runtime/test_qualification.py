"""CPU rejection/cap evidence; no claims about the GPU runtime."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from qualification import assess_candidate, qualify


def candidate(offset=0, mixed=False, version=0):
    samples = [
        {"index": offset + i, "group_index": (offset + i) // 4,
         "response_length": 2, "tokens": [1, 2, 3], "status": "completed",
         "reward": {"score": 1. if mixed and i == 0 else -1., "pred": "1"},
         "rollout_log_probs": [-1., -2.],
         "lineage": {"source_sample_index": offset + i, "output_index": 0, "output_count": 1},
         "weight_versions": [{"spans": [{"version": str(version), "abs_start": 1, "abs_end": 3}]}]}
        for i in range(16)
    ]
    issued = [{k: s[k] for k in ("index", "group_index")} for s in samples]
    return samples, issued


class QualificationTest(unittest.IsolatedAsyncioTestCase):
    async def run_candidates(self, batches):
        self.calls = 0
        self.dropped = []
        self.objects = []
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        async def next_batch():
            batch = batches[self.calls]
            self.calls += 1
            output = object()
            self.objects.append(output)
            return output, *batch

        return await qualify(next_batch, root=self.root, rollout_id=0, weight_version=0,
                             seen=set(), on_drop=self.dropped.append)

    async def test_reject_then_accept_preserves_output(self):
        result = await self.run_candidates([candidate(), candidate(16, mixed=True)])
        self.assertIs(result, self.objects[1])
        self.assertEqual(self.dropped, self.objects[:1])
        manifest = json.loads((self.root / "qualification/step-0.json").read_text())
        self.assertTrue(manifest["passed"])
        self.assertEqual([a["decision"] for a in manifest["attempts"]], ["rejected", "admitted"])
        self.assertEqual(manifest["attempts"][0]["sample_indices"], list(range(16)))

    async def test_cap_is_exactly_four(self):
        with self.assertRaisesRegex(RuntimeError, "EXHAUSTED"):
            await self.run_candidates([candidate(i * 16) for i in range(5)])
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(self.dropped), 4)

    async def test_nan_aborts_without_resampling_and_is_archived(self):
        batch = candidate(mixed=True)
        batch[0][0]["rollout_log_probs"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "FATAL"):
            await self.run_candidates([batch, candidate(16, mixed=True)])
        self.assertEqual(self.calls, 1)
        self.assertTrue((self.root / "qualification/candidate-0-0.json").is_file())

    async def test_reused_identity_aborts_second_batch(self):
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            await self.run_candidates([candidate(), candidate(mixed=True)])
        self.assertEqual(self.calls, 2)

    async def test_stale_policy_never_resamples(self):
        with self.assertRaisesRegex(RuntimeError, "behavior version"):
            await self.run_candidates([candidate(mixed=True, version=1)])
        self.assertEqual(self.calls, 1)

    def test_issued_identity_and_lineage(self):
        samples, issued = candidate(mixed=True)
        issued[0]["index"] = 999
        report = assess_candidate(samples, issued, set(), rollout_id=0, weight_version=0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["retryable"])
        samples, issued = candidate(mixed=True)
        samples[0]["lineage"]["source_sample_index"] = 5
        self.assertFalse(assess_candidate(samples, issued, set(),
                                         rollout_id=0, weight_version=0)["retryable"])

    def test_probability_and_span_coverage_are_fatal(self):
        samples, issued = candidate(mixed=True)
        samples[0]["weight_versions"][0]["spans"][0]["abs_end"] = 2
        self.assertFalse(assess_candidate(samples, issued, set(),
                                         rollout_id=0, weight_version=0)["retryable"])


if __name__ == "__main__":
    unittest.main()
