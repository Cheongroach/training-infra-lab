import unittest

from .rewards import LEGACY_RULE, score_answer


class RewardTests(unittest.TestCase):
    def test_observed_intermediate_number_false_positive(self):
        text = "s = |(-3 + sqrt{21} -"
        self.assertEqual(score_answer(text, "#### 21", "length", LEGACY_RULE)["reward"], 1)
        self.assertEqual(score_answer(text, "#### 21", "length")["reward_outcome"], "truncated")
        self.assertEqual(score_answer("Final answer: 21", "#### 21", "length")["reward"], 0)

    def test_explicit_answer_and_normal_completion(self):
        good = score_answer("Reasoning 123.\nFinal answer: -3.0", "#### -3", "eos")
        self.assertEqual(good["reward"], 1)
        self.assertEqual(good["parsed_answer"], "-3.0")
        bad = score_answer("Final answer: 4", "#### -3", "eos")
        self.assertEqual(bad["reward_outcome"], "incorrect")
        self.assertEqual(bad["reward"], 0)

    def test_intermediate_or_missing_final_answer_is_reported_separately(self):
        for text in ("The exponent is 21", "Final answer: 21\nBut there is more",
                     "Final answer: 21 or 22", "Final answer: NaN", ""):
            self.assertEqual(score_answer(text, "#### 21", "eos")["reward_outcome"],
                             "missing_final_answer")

    def test_invalid_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            score_answer("Final answer: 1", "#### 1", "cancelled")
        with self.assertRaises(ValueError):
            score_answer("Final answer: 1", "no numeric reference", "eos")
        with self.assertRaises(ValueError):
            score_answer("Final answer: 1", "#### 1", "length", "unknown")


if __name__ == "__main__":
    unittest.main()
