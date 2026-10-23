"""retrieval.py - Part C: retrieve the fraud policy that matches a question.

Run:  python retrieval.py      (prints the retrieval table)

    from retrieval import retrieve, MIN_SCORE
    policy_id, policy_text, score = retrieve("card payment above INR 50000")

TF-IDF with default settings, fitted on the text field of the six policies;
the best policy is the one with the highest cosine similarity to the query.
"""

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICIES_PATH = "fraud_policies.json"
QUERIES_PATH = "retrieval_queries.json"

# Highest out-of-scope score is 0.4251; lowest in-scope score is 0.4862.
# A query scoring below MIN_SCORE has no matching policy and is never sent to the model.
MIN_SCORE = 0.45

with open(POLICIES_PATH, encoding="utf-8") as f:
    POLICIES = json.load(f)

_VECTORIZER = TfidfVectorizer()
_POLICY_MATRIX = _VECTORIZER.fit_transform([p["text"] for p in POLICIES])


def retrieve(query: str) -> tuple[str, str, float]:
    """Return (policy_id, policy_text, score) of the most similar policy."""
    scores = cosine_similarity(_VECTORIZER.transform([query]), _POLICY_MATRIX)[0]
    best = int(scores.argmax())
    return POLICIES[best]["policy_id"], POLICIES[best]["text"], float(scores[best])


def main():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    print("PART C - RETRIEVAL")
    print(f"  {'query':<66}{'expected':<10}{'retrieved':<11}{'score':>7}  decision")
    hits, in_scope, in_scores, out_scores = 0, 0, [], []
    for q in queries:
        pid, _, score = retrieve(q["query"])
        expected = q["expected_policy"]
        answered = score >= MIN_SCORE
        if expected is None:
            out_scores.append(score)
        else:
            in_scope += 1
            in_scores.append(score)
            hits += int(pid == expected)
        decision = "answer" if answered else "no_policy_match"
        print(f"  {q['query']:<66}{str(expected):<10}{pid:<11}{score:>7.4f}  {decision}")

    print(f"\n  In-scope top-1 hits: {hits} of {in_scope}")
    print(f"  Highest out-of-scope score: {max(out_scores):.4f}")
    print(f"  Lowest in-scope score:      {min(in_scores):.4f}")
    print(f"  MIN_SCORE = {MIN_SCORE}")

    assert hits == in_scope == 8
    assert max(out_scores) < MIN_SCORE <= min(in_scores)
    print("\nAll retrieval checkpoints passed.")


if __name__ == "__main__":
    main()