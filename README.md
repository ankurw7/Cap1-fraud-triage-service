# Razorpay Merchant Risk Desk — Fraud Triage Service

A fraud triage service built in four connected Parts: data preparation, a tuned fraud model with an alert threshold, policy-grounded case notes from a validated LLM, and a FastAPI service. Risky payments go to human review. A flagged payment is never approved automatically.

## Overview

**Key capabilities**
- Audit and clean the raw transactions, and engineer three risk features.
- Tune a random forest for class imbalance and choose an alert threshold `t*` on validation.
- Retrieve the matching fraud policy with TF-IDF and refuse off-topic questions.
- Validate LLM case notes with three gates, retries, a safe default and an approve guardrail.
- Serve `/health`, `/score`, `/case` and `/metrics` with input checks, 503 handling, logging and metrics.

**Important:** this is an educational capstone project on synthetic data. It is not affiliated with Razorpay and is not a production fraud system.

## Project Structure

All files live in the repo root, and every command is run from there. No cleaned data or model is written to disk.

```text
Cap1-fraud-triage-service/
├── generate_data.py        # Provided: builds transactions_raw.csv (not modified)
├── mock_llm.py             # Provided: offline stand-in for the hosted LLM (not modified)
├── transactions_raw.csv    # Provided: raw transactions (6,090 rows × 14 columns)
├── fraud_policies.json     # Provided: the six fraud policies P1–P6
├── retrieval_queries.json  # Provided: the 10 retrieval test queries
├── case_scenarios.json     # Provided: case-note scenarios S1–S10
│
├── prepare.py              # Part A: clean_transactions(), add_features(), audit
├── train.py                # Part B: model comparison, tuning, t*, build_final_model()
├── retrieval.py            # Part C: TF-IDF policy retrieval, retrieve(), MIN_SCORE
├── casenote.py             # Part C: build_prompt(), validate(), run_case()
├── app.py                  # Part D: FastAPI service (/health, /score, /case, /metrics)
├── smoke_test.py           # Part D: sends Q1–Q11 and prints PASS/FAIL + /metrics
├── service.log             # Part D: log from one fresh smoke run
├── requirements.txt        # pandas==3.*, scikit-learn==1.9.*, numpy, fastapi, uvicorn, pydantic, requests
└── README.md
```

## Reproduce

Tested with Python 3.14.6, pandas 3.1 and scikit-learn 1.9.1. No paid APIs or network access are needed. `transactions_raw.csv` is committed (MD5 `8716797e8bc1a477df5f52cbc8aae3b8`), so you don't need to regenerate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python prepare.py        # Part A audit
python train.py          # Part B tables (~40 s)
python retrieval.py      # Part C retrieval table
python casenote.py       # Part C scenario table
uvicorn app:app --port 8000
```

When the server reports `Application startup complete`, open a second terminal, activate the virtual environment and run `python smoke_test.py`. It should end with `Total 11/11` and `Metrics match expected: True`. Restart the server before each smoke run, because the mock LLM counts how many times it has seen each transaction ID. Every script asserts its own checkpoints, so a wrong result fails loudly.

## Part A — Data Audit and Feature Engineering

| Audit item | Result |
|---|---:|
| Raw rows × columns | 6,090 × 14 |
| Exact duplicate rows | 90 |
| Spellings of `merchant_category` / `city` (raw → cleaned) | 24 → 6 each |
| Missing `amount_inr` (after removing duplicates) | 76 |
| Impossible amounts (< 1 or > 1,000,000) | 5 |
| **Rows after cleaning** | **5,919** |
| Missing distance after cleaning (left unfilled) | 191 |
| Fraud prevalence | 366 rows (6.18%) |

**Cleaning order:** drop duplicates → strip and title-case the text → drop missing amounts → keep amounts between 1 and 1,000,000. Missing distances are filled in Part B with the training-split median, so no statistic comes from validation or test rows.

**Features:** `amount_vs_avg_ratio` (max 13.11), `is_night` for hour ≤ 5 (365 rows) and `high_value` for amounts above INR 50,000 (17 rows).

**Imbalance:** always predicting "not fraud" gives 93.82% accuracy but 0 recall, so the model is judged on precision, recall and F1, not accuracy.

**Leakage:** 304 of the 373 rows with a chargeback are fraud, compared with 62 of the 5,546 rows without one. A chargeback is only filed after settlement, so `chargeback_filed` is excluded from the inputs, along with `txn_id`, `customer_id` and the target `is_fraud`.

## Part B — Model Training and Threshold Selection

There are 26 input columns: 7 raw numeric columns, the 3 engineered features and 16 one-hot columns. The split is 60/20/20, stratified, giving 3,551 / 1,184 / 1,184 rows with 220 / 73 / 73 fraud cases. Missing distance is filled with the training median (12.6 km).

**Imbalance handling (logistic regression, validation, threshold 0.5)**

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Plain | 0.625 | 0.342 | 0.442 |
| `class_weight="balanced"` | 0.252 | 0.781 | 0.381 |
| Over-sampled (6,662 rows) | 0.257 | 0.781 | 0.386 |

Balancing more than doubles recall, at the cost of many more false alarms. Both methods give almost the same result, so I used class weights because they don't change the data.

**Model choice (5-fold CV F1 on the training split):** balanced LR scored 0.402 and an untuned balanced random forest 0.453. A grid search picked the tuned forest (`max_depth=None`, `min_samples_leaf=5`) at **0.465**. Shallow trees underfit, and larger leaves smooth away the small fraud groups.

**Threshold selection (validation)**

| t | Flagged | Precision | Recall |
|---|---:|---:|---:|
| 0.3 | 302 | 0.209 | 0.863 |
| **0.4** | **191** | **0.272** | **0.712** |
| 0.5 | 122 | 0.352 | 0.589 |
| 0.6 | 73 | 0.507 | 0.507 |

`t*` is the highest threshold that keeps validation recall ≥ 0.65, so **t\* = 0.4**.

**Test set (used once, at 0.4):** precision **0.281**, recall **0.781**, F1 **0.413**, with 203 of 1,184 payments flagged.

```text
                 Predicted 0   Predicted 1
Actual 0              965           146
Actual 1               16            57
```

The brief gives 202 flagged. The difference is one legitimate transaction that scores 0.4005 on my machine. All the validation numbers match the brief exactly.

**Note for the risk committee:** the model catches about 78% of fraud by holding about 17% of payments. Only 1 in 4 held payments is really fraud, so the review team has to be sized for the false alarms. Raising the threshold to 0.5 would cut the queue by about a third but miss noticeably more fraud. I'd start at 0.4 and review it once real chargeback data comes in.

**Leakage check:** adding `chargeback_filed` raises validation F1 from 0.441 to 0.797. A model that uses it would look great in testing and fail in production.

**Top features:** `amount_vs_avg_ratio`, `txn_count_last_24h`, `amount_inr`, `account_age_days` and `distance_from_home_km`. These are an unusually large amount, a burst of transactions, a new account and an unusual location.

## Part C — Policy Retrieval and Case Notes

**Retrieval:** default `TfidfVectorizer()` fitted on the six policy texts, matched by cosine similarity. 8/8 in-scope queries match and 2/2 off-topic queries are refused.

| Query | Expected | Retrieved | Score |
|---|---|---|---:|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | none | refused | 0.4251 |
| how do I reset my email password | none | refused | 0.0000 |

`MIN_SCORE = 0.45` sits between the highest off-topic score (0.4251) and the lowest on-topic score (0.4862). A refused query never calls the model.

**Case notes:** each reply must pass three gates. Gate 1 checks that it is a JSON object. Gate 2 checks that all five keys are present and are strings. Gate 3 checks that every value is allowed and that `policy_id` matches the retrieved policy. A failed reply is retried, up to 3 calls in total. After that the case gets a safe default (high, `customer_care`, `hold_for_review`). An `approve` with probability ≥ 0.5 is changed to `hold_for_review` by the guardrail.

| Scenario | Policy | Attempts | Failures | Final action | Note |
|---|---|---:|---|---|---|
| S1 T900001 | P1 | 1 | [] | hold_for_review | |
| S2 T900002 | P2 | 2 | [1] | hold_for_review | |
| S3 T900003 | P3 | 2 | [2] | hold_for_review | |
| S4 T900004 | P4 | 2 | [3] | block_and_notify | |
| S5 T900005 | P5 | 3 | [1, 1, 1] | hold_for_review | safe default |
| S6 T900006 | P2 | 1 | [] | hold_for_review | guardrail override |
| S7 T900007 | P1 | 2 | [3] | hold_for_review | |
| S8 T900008 | P4 | 1 | [] | block_and_notify | |
| S9 T900009 | P3 | 3 | [1, 2] | hold_for_review | |
| S10 T900010 | P2 | 2 | [3] | hold_for_review | |

That is 19 model calls in total: 3 notes were valid first time, 6 recovered after a retry and 1 needed the safe default. For T900099 the `ConnectionError` is passed up for Part D to turn into a 503.

**Why this matters:** a hosted model can wrap JSON in prose, cut it off, drop a field or make up a value, and the same prompt can fail and then succeed (S9). S7 was valid JSON but cited the wrong policy, which only Gate 3 catches.

## Part D — FastAPI Service and Smoke Tests

| Situation | Response |
|---|---|
| Impossible amount, hour, txn count, avg amount or category | **400** `{"error", "field"}`, before the model is called |
| Missing or wrongly typed field | **422** from FastAPI |
| Not flagged, but amount ≥ INR 100,000 on a new device | `hold_for_review` |
| No policy matches the question | No note; flagged payments are held |
| Case-note model unavailable (T900099) | **503**; the service keeps running |

One middleware writes one `service.log` line per POST and updates the `/metrics` counters.

**Smoke test (fresh server): 11/11 pass.**

| Id | Request | Result |
|---|---|---|
| Q1 | GET /health | 200 |
| Q2 | /score BENIGN | 200, p 0.1312, approve |
| Q3 | /score RISKY | 200, p 0.9596, flagged, hold |
| Q4 | /score BIG (INR 150,000, new device) | 200, p 0.3224, not flagged, hold |
| Q5 | /score amount -5 | 400, field `amount_inr` |
| Q6 | /score without `hour_of_day` | 422 |
| Q7 | /case T900002 | 200, 2 attempts, P2 |
| Q8 | /case T900005 | 200, 3 attempts, safe default |
| Q9 | /case T900099 | 503 |
| Q10 | /case BENIGN | 200, no note, approve |
| Q11 | /case cafeteria question | 200, no note, hold |

```json
{"requests_total": 10, "success_total": 7, "error_total": 3, "success_rate": 0.7, "error_rate": 0.3,
 "avg_latency_ms": 11.32, "human_review_total": 5, "human_review_rate": 0.5}
```

`service.log` records every run, with one line per POST and no stack traces:

| Run | Time | Lines | INFO | WARNING | ERROR |
|---|---|---:|---:|---:|---:|
| Smoke test 1 | 2026-10-23 21:11 | 10 | 5 | 4 | 1 |
| Smoke test 2 | 2026-10-24 00:05 | 10 | 5 | 4 | 1 |
| Extra check T900006 (×2) | 2026-10-24 00:07 | 2 | 1 | 1 | 0 |
| Smoke test 3 | 2026-10-24 00:08 | 10 | 5 | 4 | 1 |
| **Total** | | **32** | **16** | **13** | **3** |

Each smoke run gives the expected 10 lines (5 INFO, 4 WARNING, 1 ERROR). Example:

```text
2026-10-23 21:11:30,815 WARNING /case status=200 latency_ms=14.6 txn_id=T900005 probability=0.9596 human_review=True safe_default=True
```

**Extra check (`python smoke_test.py --extra`, fresh server):** T900006 scores 0.4272, so it is flagged but below the 0.5 guardrail in `run_case`. The mock answers `approve`, and the app's own rule changes it to `hold_for_review` with `guardrail_override: true`. That's why the app checks every flagged case itself.

**Trying it by hand: while the server is running, open http://127.0.0.1:8000/docs in a browser. FastAPI's built-in page lets you send /score and /case requests and see the responses. Those requests also count towards /metrics, so restart before the smoke test.

## Limitations

1. **Synthetic data:** the metrics and threshold won't carry over to real traffic without retraining.
2. **Review workload:** only about 28% of held payments are fraud, so analysts mostly clear false alarms.
3. **Lexical retrieval:** TF-IDF only matches words, so a differently phrased question (e.g. "late-night" vs "midnight") can be refused.
4. **Mock LLM:** a real hosted model would add latency, cost, timeouts and failure types these tests don't cover.
5. **In-memory metrics:** the counters reset on restart and aren't shared across workers.
6. **Retraining at start-up:** fine for 6,000 rows, but too slow at real data volumes.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- TF-IDF and cosine similarity
- FastAPI
- Pydantic
- Uvicorn
- Git and GitHub

## Disclaimer

This repository is an independent educational portfolio project. It is not affiliated with, endorsed by, or an official product of Razorpay. The service and results are for demonstration purposes only and should not be used as the sole basis for real financial or fraud decisions.