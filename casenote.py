"""casenote.py - Part C: policy-grounded case notes with validation.

Run:  python casenote.py       (resets the mock, runs the 10 scenarios)

    from casenote import run_case
    result = run_case("T900001", 0.96, 82000, "card payment above INR 50000 from a new device")

Pipeline: retrieve the policy -> refuse if below MIN_SCORE -> build the prompt ->
call mock_llm -> validate in three gates -> retry the same prompt -> safe default
-> output guardrail. A ConnectionError from the model is NOT caught here.
"""

import json

import mock_llm
from retrieval import MIN_SCORE, retrieve

SCENARIOS_PATH = "case_scenarios.json"
MAX_ATTEMPTS = 3
OVERRIDE_PROBABILITY = 0.5

# The case-note rules, written once in JSON Schema vocabulary; the gates read from it.
CASE_NOTE_SCHEMA = {
    "type": "object",
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
}

SAFE_DEFAULT_SUMMARY = "Automated case note unavailable; manual review required."


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text) -> str:
    """Role -> task -> constraints -> output format, plus the four labelled lines."""
    props = CASE_NOTE_SCHEMA["properties"]
    allowed = "\n".join(
        f"- {key}: " + (" | ".join(props[key]["enum"]) if "enum" in props[key] else "string")
        for key in CASE_NOTE_SCHEMA["required"]
    )
    return (
        "ROLE\n"
        "You are a fraud analyst assistant on a merchant risk desk.\n\n"
        "TASK\n"
        "Write a short case note for the held payment below, citing the fraud policy that applies.\n\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n\n"
        f"POLICY TEXT\n{policy_text}\n\n"
        "CONSTRAINTS\n"
        "- Use only the policy text above; do not invent rules, teams or amounts.\n"
        f"- Set policy_id to {policy_id}.\n"
        "- Reply with one JSON object and nothing else: no prose, no markdown.\n\n"
        "OUTPUT FORMAT\n"
        "A JSON object with exactly these five keys and allowed values:\n"
        f"{allowed}\n"
    )


def validate(text, retrieved_policy_id):
    """Return (note, gate): gate 0 = passed, else the first gate that failed."""
    # Gate 1: valid JSON and a JSON object.
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1
    if not isinstance(note, dict):
        return None, 1

    # Gate 2: every required key present and a string.
    props = CASE_NOTE_SCHEMA["properties"]
    for key in CASE_NOTE_SCHEMA["required"]:
        if key not in note or not isinstance(note[key], str):
            return note, 2

    # Gate 3: allowed values, and the policy id must be the retrieved one.
    for key in CASE_NOTE_SCHEMA["required"]:
        if "enum" in props[key] and note[key] not in props[key]["enum"]:
            return note, 3
    if note["policy_id"] != retrieved_policy_id:
        return note, 3
    return note, 0


def safe_default(policy_id) -> dict:
    return {
        "summary": SAFE_DEFAULT_SUMMARY,
        "priority": "high",
        "recommended_team": "customer_care",
        "recommended_action": "hold_for_review",
        "policy_id": policy_id,
    }


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=MAX_ATTEMPTS) -> dict:
    """Retrieve, prompt, validate, retry, fall back safely, then apply the guardrail."""
    policy_id, policy_text, score = retrieve(question)
    result = {
        "status": "ok", "attempts": 0, "failures": [], "case_note": None,
        "safe_default": False, "guardrail_override": False,
    }
    if score < MIN_SCORE:
        result["status"] = "no_policy_match"
        return result

    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    note = None
    while result["attempts"] < max_attempts:
        result["attempts"] += 1
        text = mock_llm.generate(prompt)          # ConnectionError propagates to the caller
        candidate, gate = validate(text, policy_id)
        if gate == 0:
            note = candidate
            break
        result["failures"].append(gate)

    if note is None:
        note = safe_default(policy_id)
        result["safe_default"] = True

    # Output guardrail: a fluent, well-formed note can still be wrong.
    if note["recommended_action"] == "approve" and fraud_probability >= OVERRIDE_PROBABILITY:
        note = {**note, "recommended_action": "hold_for_review"}
        result["guardrail_override"] = True

    result["case_note"] = note
    return result


def yes_no(flag) -> str:
    return "yes" if flag else "no"


def main():
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        scenarios = json.load(f)

    mock_llm.reset()
    results = []
    print("PART C - CASE-NOTE SCENARIOS")
    print(f"  {'scenario':<14}{'policy':<8}{'attempts':>8}  {'failures':<11}{'status':<8}"
          f"{'safe default':<14}{'final action':<18}{'override':<8}")
    for i, s in enumerate(scenarios, start=1):
        r = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        results.append(r)
        action = r["case_note"]["recommended_action"] if r["case_note"] else "-"
        print(f"  {'S' + str(i) + ' (' + s['txn_id'] + ')':<14}{r['case_note']['policy_id']:<8}{r['attempts']:>8}  "
              f"{str(r['failures']):<11}{r['status']:<8}{yes_no(r['safe_default']):<14}"
              f"{action:<18}{yes_no(r['guardrail_override']):<8}")

    calls = sum(r["attempts"] for r in results)
    first = [f"S{i}" for i, r in enumerate(results, 1) if r["attempts"] == 1 and not r["failures"]]
    retried = [f"S{i}" for i, r in enumerate(results, 1) if r["failures"] and not r["safe_default"]]
    defaults = [f"S{i}" for i, r in enumerate(results, 1) if r["safe_default"]]
    overrides = [f"S{i}" for i, r in enumerate(results, 1) if r["guardrail_override"]]
    print("\nTOTALS")
    print(f"  Model calls: {calls}")
    print(f"  Valid on first call: {len(first)} {first}")
    print(f"  Recovered by retry: {len(retried)} {retried}")
    print(f"  Safe default: {len(defaults)} {defaults}")
    print(f"  Guardrail override: {len(overrides)} {overrides}")

    # Extra checks from the acceptance criteria.
    try:
        run_case("T900099", 0.9, 1000, "card payment above INR 50000 from a new device")
        raise AssertionError("ConnectionError was swallowed")
    except ConnectionError:
        print("\n  T900099: ConnectionError raised to the caller (not swallowed)")
    miss = run_case("T800000", 0.9, 1000, "what is the cafeteria menu today")
    print(f"  Out-of-scope question: status {miss['status']}, attempts {miss['attempts']}, "
          f"case_note {miss['case_note']}")

    expected = [
        ("P1", 1, [], "hold_for_review", False, False),
        ("P2", 2, [1], "hold_for_review", False, False),
        ("P3", 2, [2], "hold_for_review", False, False),
        ("P4", 2, [3], "block_and_notify", False, False),
        ("P5", 3, [1, 1, 1], "hold_for_review", True, False),
        ("P2", 1, [], "hold_for_review", False, True),
        ("P1", 2, [3], "hold_for_review", False, False),
        ("P4", 1, [], "block_and_notify", False, False),
        ("P3", 3, [1, 2], "hold_for_review", False, False),
        ("P2", 2, [3], "hold_for_review", False, False),
    ]
    observed = [(r["case_note"]["policy_id"], r["attempts"], r["failures"], r["case_note"]["recommended_action"],
                 r["safe_default"], r["guardrail_override"]) for r in results]
    assert observed == expected
    assert all(r["status"] == "ok" for r in results)
    assert calls == 19 and len(first) == 3 and len(retried) == 6
    assert len(defaults) == 1 and len(overrides) == 1
    assert all(set(r) == {"status", "attempts", "failures", "case_note", "safe_default",
                          "guardrail_override"} for r in results + [miss])
    assert miss["status"] == "no_policy_match" and miss["attempts"] == 0 and miss["case_note"] is None
    assert validate("not json", "P1")[1] == 1 and validate("[1, 2]", "P1")[1] == 1
    print("\nAll Part C checkpoints passed.")


if __name__ == "__main__":
    main()