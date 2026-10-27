"""Versioned numeric rewards; termination and answer extraction are explicit."""

import re
from decimal import Decimal, InvalidOperation

LEGACY_RULE = "last_decimal_exact_v1"
STRICT_RULE = "eos_final_line_decimal_v2"
RULES = (LEGACY_RULE, STRICT_RULE)


def final_number(text):
    """Legacy extraction, also used for the pinned numeric reference answers."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").split("####")[-1])
    try:
        return Decimal(numbers[-1]) if numbers else None
    except InvalidOperation:
        return None


def instruction(rule):
    if rule == LEGACY_RULE:
        return "Solve the problem. Put the final numeric answer last."
    if rule == STRICT_RULE:
        return ("Solve the problem with concise reasoning. End with exactly one final line "
                "in this format: Final answer: <number>. Replace <number> with a signed "
                "integer or decimal, without units, commas, Markdown or a trailing period.")
    raise ValueError("Unknown reward rule")


def score_answer(text, reference, finish_reason, rule=STRICT_RULE):
    instruction(rule)  # Validate even when termination prevents parsing.
    if finish_reason not in ("eos", "length"):
        raise ValueError("Unknown termination reason")
    expected = final_number(reference)
    if expected is None:
        raise ValueError("Reference has no valid final numeric answer")
    if rule == STRICT_RULE:
        if finish_reason != "eos":
            return {"reward": 0.0, "reward_outcome": "truncated", "parsed_answer": None}
        lines = text.strip().splitlines()
        match = re.fullmatch(r"Final answer:\s*([+-]?\d+(?:\.\d+)?)", lines[-1].strip(),
                             flags=re.IGNORECASE) if lines else None
        if match is None:
            return {"reward": 0.0, "reward_outcome": "missing_final_answer", "parsed_answer": None}
        actual = Decimal(match.group(1))
    else:
        actual = final_number(text)
    correct = actual == expected
    return {"reward": float(correct), "reward_outcome": "correct" if correct else "incorrect",
            "parsed_answer": str(actual) if actual is not None else None}
