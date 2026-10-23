"""smoke_test.py - Part D: send Q1-Q11 to a freshly started service.
Run:  uvicorn app:app --port 8000      (terminal 1, restart before every run)
      python smoke_test.py             (terminal 2)

      python smoke_test.py --extra     (on its own fresh server: T900006 guardrail check)

mock_llm remembers call counts per txn_id, so a second run against the same
server gives different attempts. Restart the server first."""

import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"
WAIT_SECONDS = 120

BENIGN = {
    "txn_id": "T100001", "merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI",
    "amount_inr": 850.0, "hour_of_day": 14, "txn_count_last_24h": 2, "avg_amount_30d_inr": 900.0,
    "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900,
}
RISKY = {
    "txn_id": "T100002", "merchant_category": " electronics ", "city": "MUMBAI", "payment_mode": "Card",
    "amount_inr": 48000.0, "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 4000.0,
    "distance_from_home_km": 250.0, "is_new_device": 1, "account_age_days": 12,
}
# Not flagged by the model, but INR 150,000 on a new device -> large-payment hold.
BIG = {**BENIGN, "txn_id": "T100003", "amount_inr": 150000.0, "avg_amount_30d_inr": 120000.0,
       "is_new_device": 1, "distance_from_home_km": None}

UPI_QUESTION = "more than 6 UPI payments in 24 hours velocity alert"
ELECTRONICS_QUESTION = "electronics merchant payment on an account younger than 30 days"

# (id, category, method, path, body, check) - check(status, json) -> True when it passes.
REQUESTS = [
    ("Q1", "happy", "GET", "/health", None,
     lambda s, j: s == 200 and j == {"status": "ok"}),
    ("Q2", "happy", "POST", "/score", BENIGN,
     lambda s, j: s == 200 and not j["flagged"] and j["decision"] == "approve" and not j["human_review"]),
    ("Q3", "escalation", "POST", "/score", RISKY,
     lambda s, j: s == 200 and j["flagged"] and j["decision"] == "hold_for_review" and j["human_review"]),
    ("Q4", "escalation", "POST", "/score", BIG,
     lambda s, j: s == 200 and not j["flagged"] and j["decision"] == "hold_for_review" and j["human_review"]),
    ("Q5", "edge", "POST", "/score", {**BENIGN, "amount_inr": -5},
     lambda s, j: s == 400 and j.get("field") == "amount_inr"),
    ("Q6", "edge", "POST", "/score", {k: v for k, v in BENIGN.items() if k != "hour_of_day"},
     lambda s, j: s == 422),
    ("Q7", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900002", "question": UPI_QUESTION},
     lambda s, j: s == 200 and j["attempts"] == 2 and j["case_note"]["policy_id"] == "P2"
     and j["human_review"]),
    ("Q8", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900005", "question": ELECTRONICS_QUESTION},
     lambda s, j: s == 200 and j["attempts"] == 3 and j["safe_default"]
     and j["decision"] == "hold_for_review" and j["human_review"]),
    ("Q9", "edge", "POST", "/case", {**RISKY, "txn_id": "T900099", "question": UPI_QUESTION},
     lambda s, j: s == 503 and set(j) == {"error"}),
    ("Q10", "happy", "POST", "/case", {**BENIGN, "question": UPI_QUESTION},
     lambda s, j: s == 200 and j["case_note"] is None and j["decision"] == "approve"
     and not j["human_review"]),
    ("Q11", "edge", "POST", "/case", {**RISKY, "txn_id": "T100004", "question": "what is the cafeteria menu today"},
     lambda s, j: s == 200 and j["case_note"] is None and j["attempts"] == 0
     and j["decision"] == "hold_for_review" and j["human_review"]),
]

EXPECTED_METRICS = {"requests_total": 10, "success_total": 7, "error_total": 3, "success_rate": 0.7,
                    "error_rate": 0.3, "human_review_total": 5, "human_review_rate": 0.5}

# Flagged (p about 0.43) but below casenote's 0.5 guardrail, so the app's own rule must catch "approve".
T900006 = {**BENIGN, "txn_id": "T900006", "amount_inr": 2000.0, "avg_amount_30d_inr": 1500.0,
           "txn_count_last_24h": 3, "question": "temporary hold on repeated UPI payments within a day"}


def wait_for_health():
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    sys.exit(f"Service did not answer /health within {WAIT_SECONDS} s")


def send(method, path, body):
    response = requests.request(method, f"{BASE}{path}", json=body, timeout=60)
    return response.status_code, response.json()


def run_smoke():
    passed, totals = {}, {}
    for qid, category, method, path, body, check in REQUESTS:
        status, body_json = send(method, path, body)
        try:
            ok = bool(check(status, body_json))
        except (KeyError, TypeError):
            ok = False
        passed[category] = passed.get(category, 0) + ok
        totals[category] = totals.get(category, 0) + 1
        print(f"{qid:<4}{'PASS' if ok else 'FAIL':<6}{method:<5}{path:<9}{status:<5}{json.dumps(body_json)}")

    print()
    for category in ("happy", "escalation", "edge"):
        print(f"{category:<11}{passed[category]}/{totals[category]}")
    n_pass = sum(passed.values())
    print(f"Total      {n_pass}/{len(REQUESTS)}")

    metrics = requests.get(f"{BASE}/metrics", timeout=5).json()
    print(f"\n/metrics {json.dumps(metrics, indent=2)}")
    metrics_ok = all(metrics[k] == v for k, v in EXPECTED_METRICS.items())
    print(f"Metrics match expected: {metrics_ok}")
    return n_pass == len(REQUESTS) and metrics_ok


def run_extra():
    status, body = send("POST", "/case", T900006)
    print(json.dumps(body, indent=2))
    ok = (status == 200 and body["flagged"] and body["fraud_probability"] < 0.5
          and body["decision"] == "hold_for_review" and body["guardrail_override"]
          and body["case_note"]["recommended_action"] == "hold_for_review")
    print(f"T900006 guardrail check: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    wait_for_health()
    ok = run_extra() if "--extra" in sys.argv else run_smoke()
    sys.exit(0 if ok else 1)
