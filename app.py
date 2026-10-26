"""app.py - Part D: fraud triage service.
Run:  uvicorn app:app --port 8000      (no --reload: restart to reset the mock's memory)

    GET  /health    liveness
    POST /score     fraud probability, flag and decision for one payment
    POST /case      /score plus a validated, policy-grounded case note when flagged
    GET  /metrics   request, error and human-review counters

The model is rebuilt once at start-up (build_final_model); nothing is loaded from disk.
One middleware writes exactly one service.log line per POST and updates the counters.
"""

import logging
import time

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casenote import run_case
from prepare import MAX_VALID_AMOUNT, MIN_VALID_AMOUNT, TEXT_COLUMNS, add_features
from train import DISTANCE, build_final_model

LOG_PATH = "service.log"
CATEGORIES = {"Grocery", "Food Delivery", "Fashion", "Electronics", "Utilities", "Travel"}
LARGE_PAYMENT_INR = 100_000
UNAVAILABLE = "case-note service temporarily unavailable, try again later"


def _service_logger():
    """Own logger and FileHandler, so uvicorn's logging setup can't interfere."""
    logger = logging.getLogger("triage_service")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = _service_logger()
app = FastAPI(title="Fraud triage service")
MODEL = build_final_model()
COUNTERS = {"requests": 0, "success": 0, "error": 0, "latency_ms": 0.0, "human_review": 0}


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


# ---------- scoring ----------

def check_input(txn: Txn):
    """Business-rule checks -> 400 response, or None when the payment is valid."""
    category = txn.merchant_category.strip().title()
    problems = [
        ("amount_inr", not MIN_VALID_AMOUNT <= txn.amount_inr <= MAX_VALID_AMOUNT,
         f"amount_inr must be between {MIN_VALID_AMOUNT} and {MAX_VALID_AMOUNT:,}"),
        ("hour_of_day", not 0 <= txn.hour_of_day <= 23, "hour_of_day must be between 0 and 23"),
        ("txn_count_last_24h", txn.txn_count_last_24h < 0, "txn_count_last_24h must not be negative"),
        ("avg_amount_30d_inr", txn.avg_amount_30d_inr <= 0, "avg_amount_30d_inr must be above 0"),
        ("merchant_category", category not in CATEGORIES,
         f"merchant_category must be one of {sorted(CATEGORIES)}"),
    ]
    for field, failed, message in problems:
        if failed:
            return JSONResponse(status_code=400, content={"error": message, "field": field})
    return None


def fraud_probability(txn: Txn) -> float:
    df = pd.DataFrame([txn.model_dump(include=set(Txn.model_fields))])
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().str.title()
    X = pd.get_dummies(add_features(df).drop(columns=["txn_id"]), dtype=int)
    X = X.reindex(columns=MODEL["columns"], fill_value=0).fillna({DISTANCE: MODEL["median"]})
    return float(MODEL["model"].predict_proba(X)[0, 1])


def score(txn: Txn, request: Request) -> dict:
    """Probability, flag and decision; records txn_id and probability for the log line."""
    proba = fraud_probability(txn)
    flagged = proba >= MODEL["threshold"]
    large_hold = not flagged and txn.amount_inr >= LARGE_PAYMENT_INR and txn.is_new_device == 1
    decision = "hold_for_review" if flagged or large_hold else "approve"
    request.state.txn_id, request.state.probability = txn.txn_id, proba
    request.state.override = large_hold
    return {"txn_id": txn.txn_id, "fraud_probability": round(proba, 4), "flagged": flagged,
            "decision": decision, "human_review": decision != "approve"}


# ---------- routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score_route(txn: Txn, request: Request):
    if (rejected := check_input(txn)) is not None:
        return rejected
    result = score(txn, request)
    request.state.human_review = result["human_review"]
    return result


@app.post("/case")
def case_route(case: CaseIn, request: Request):
    if (rejected := check_input(case)) is not None:
        return rejected
    result = score(case, request)
    extra = {"case_note": None, "attempts": 0, "safe_default": False,
             "guardrail_override": request.state.override}

    if result["flagged"]:
        try:
            outcome = run_case(case.txn_id, result["fraud_probability"], case.amount_inr, case.question)
        except ConnectionError:
            return JSONResponse(status_code=503, content={"error": UNAVAILABLE})
        note = outcome["case_note"]
        override = outcome["guardrail_override"]
        if note is None:                                   # no_policy_match: model never called
            decision = "hold_for_review"
        else:
            decision = note["recommended_action"]
            if decision == "approve":                      # never auto-approve a flagged payment
                decision = "hold_for_review"
                note = {**note, "recommended_action": decision}
                override = True
        result.update(decision=decision, human_review=True)
        extra = {"case_note": note, "attempts": outcome["attempts"],
                 "safe_default": outcome["safe_default"], "guardrail_override": override}
        request.state.safe_default = outcome["safe_default"]
        request.state.override = override

    request.state.human_review = result["human_review"]
    return {**result, **extra}


@app.get("/metrics")
def metrics():
    n = COUNTERS["requests"]
    rate = lambda count: round(count / n, 4) if n else 0.0
    return {
        "requests_total": n,
        "success_total": COUNTERS["success"],
        "error_total": COUNTERS["error"],
        "success_rate": rate(COUNTERS["success"]),
        "error_rate": rate(COUNTERS["error"]),
        "avg_latency_ms": round(COUNTERS["latency_ms"] / n, 2) if n else 0.0,
        "human_review_total": COUNTERS["human_review"],
        "human_review_rate": rate(COUNTERS["human_review"]),
    }


# ---------- logging and counters ----------

@app.middleware("http")
async def log_and_count(request: Request, call_next):
    if request.method != "POST":
        return await call_next(request)

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:                               # last resort: still log, never leak a trace
        response = JSONResponse(status_code=500, content={"error": "internal error"})
        request.state.failure = type(exc).__name__
    latency = (time.perf_counter() - start) * 1000

    state = request.state
    status = response.status_code
    ok = status < 400
    human_review = ok and getattr(state, "human_review", False)
    COUNTERS["requests"] += 1
    COUNTERS["success" if ok else "error"] += 1
    COUNTERS["latency_ms"] += latency
    COUNTERS["human_review"] += int(human_review)

    parts = [f"{request.url.path}", f"status={status}", f"latency_ms={latency:.1f}"]
    if hasattr(state, "txn_id"):
        parts += [f"txn_id={state.txn_id}", f"probability={state.probability:.4f}"]
    if ok:
        parts.append(f"human_review={human_review}")
    if getattr(state, "safe_default", False):
        parts.append("safe_default=True")
    if getattr(state, "override", False):
        parts.append("override=True")
    if hasattr(state, "failure"):
        parts.append(f"failure={state.failure}")

    if status >= 500:
        level = logging.ERROR
    elif status >= 400 or getattr(state, "safe_default", False) or getattr(state, "override", False):
        level = logging.WARNING
    else:
        level = logging.INFO
    log.log(level, " ".join(parts))
    return response
