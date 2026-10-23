# Cap1-fraud-triage-service
Fraud triage service: data prep, fraud model, policy-grounded case notes, FastAPI service

## Part A: Data cleaning (`prepare.py`)

Run `python prepare.py`. Each number below is printed by the script and checked with an assert.

**Raw data:** 6,090 rows × 14 columns. 90 rows are exact duplicates. `merchant_category` and `city` each have 24 spellings, because of mixed case and extra spaces, but only 6 real values. After removing duplicates there are 6,000 rows: 76 have no amount, 193 have no distance and 5 have impossible amounts (below 1 or above 1,000,000).

**Cleaning:** drop duplicates → fix the text → drop rows with a missing or impossible amount. This leaves **5,919 rows**. I left the 191 missing distances unfilled. They will be filled in Part B using the training-split median, to avoid leakage.

**Features:** `amount_vs_avg_ratio` (max 13.11), `is_night` for hour ≤ 5 (365 rows) and `high_value` for amounts above INR 50,000 (17 rows).

**Imbalance:** 366 of the 5,919 rows are fraud (6.18%). Predicting "never fraud" gives 93.82% accuracy but 0 recall, so I'll judge the model with ROC-AUC, PR-AUC, precision and recall.

**Leakage:** 304 of the 373 chargeback rows are fraud, compared with 62 of the 5,546 rows without a chargeback. A chargeback is only filed after settlement, so I excluded `chargeback_filed`. I also excluded `txn_id`, `customer_id` and `is_fraud`.

## Part B: Fraud model (`train.py`)

Run `python train.py` (about 40 seconds). Each number below is printed by the script and checked with an assert.

**Inputs:** 26 columns. These are the 7 raw numeric columns (`amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`), the 3 engineered features and 16 one-hot columns for category, city and payment mode.

**Splits:** 60/20/20, stratified, so train/validation/test have 3,551 / 1,184 / 1,184 rows and 220 / 73 / 73 fraud cases. I filled missing distance in all three splits with the training median (12.6 km).

**Imbalance (logistic regression, validation, threshold 0.5):**

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Plain | 0.625 | 0.342 | 0.442 |
| `class_weight="balanced"` | 0.252 | 0.781 | 0.381 |
| Over-sampled (6,662 rows) | 0.257 | 0.781 | 0.386 |

The plain model misses about two-thirds of the fraud. Balancing the classes, either with class weights or by over-sampling, more than doubles recall, at the cost of many more false alarms. Both balancing methods give almost the same result, so I used class weights because they don't change the data.

**Model choice:** with 5-fold CV on the training split, balanced LR scored F1 0.402 (sd 0.015) and an untuned balanced random forest scored 0.453. A grid search over `max_depth` × `min_samples_leaf` (12 combinations) picked `max_depth=None, min_samples_leaf=5`, with CV F1 0.465 (the brief gives 0.464; the unrounded value is 0.4646). Shallow trees (depth 4–6) underfit, and larger leaves smooth away the small fraud groups, so the deep forest with small leaves won. This tuned forest is my final model.

**Threshold (validation):**

| t | Flagged | Precision | Recall |
|---|---:|---:|---:|
| 0.3 | 302 | 0.209 | 0.863 |
| 0.4 | 191 | 0.272 | 0.712 |
| 0.5 | 122 | 0.352 | 0.589 |
| 0.6 | 73 | 0.507 | 0.507 |

I picked the highest threshold that still catches at least 65% of fraud, so **t\* = 0.4**.

**Test (used once, at 0.4):** precision 0.281, recall 0.781, F1 0.413, with 203 of 1,184 flagged (TN 965, FP 146, FN 16, TP 57). The brief gives 202 flagged and 145 FP. The difference is one legitimate transaction that scores 0.4005 on my machine, and all the validation numbers match exactly.

*Note for the risk committee:* at this threshold the model catches about 78% of fraud (57 of 73) by holding about 17% of payments. A little over 1 in 4 held payments is really fraud, so analysts will mostly be clearing false alarms, and the review team has to be sized for that. 16 fraud cases still get through. Raising the threshold to 0.5 would cut the queue by about a third but miss noticeably more fraud. I'd start at 0.4 and review it once real chargeback data comes in.

**Leakage check:** adding `chargeback_filed` as a 27th column raises validation F1 from 0.441 to 0.797. That jump is the leak. The column is only known after settlement, so a model that uses it would look great in testing and fail in production.

**Top 5 features:** `amount_vs_avg_ratio`, `txn_count_last_24h`, `amount_inr`, `account_age_days` and `distance_from_home_km`. These are the signals I'd expect: an amount far above the customer's usual spend, a burst of transactions, a new account and an unusual location.

`build_final_model()` rebuilds the model in memory and returns `{"model", "columns", "median", "threshold"}` for Part D. Nothing is saved to disk.

## Part C: Policy-grounded case notes (`retrieval.py`, `casenote.py`)

Run `python retrieval.py`, then `python casenote.py`. Both scripts check their own results with asserts.

**Retrieval:** TF-IDF with default settings, fitted on the six policy texts, matched by cosine similarity. All 8 in-scope queries hit.

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

I set `MIN_SCORE = 0.45`. It sits between the highest off-topic score (0.4251) and the lowest on-topic score (0.4862), so both off-topic questions are refused without calling the model.

**Case notes:** each reply has to pass three gates, all driven by one JSON-Schema-style dict. Gate 1 checks that the reply is a JSON object. Gate 2 checks that all five keys are present and are strings. Gate 3 checks that every value is allowed and that `policy_id` matches the policy I retrieved. A failed reply is retried up to 3 times. If all three fail, the case gets a safe default (high priority, `hold_for_review`), and an `approve` with fraud probability ≥ 0.5 is always changed to `hold_for_review`.

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

All ten have status `ok`. In total there were 19 model calls: 3 notes were valid on the first call, 6 recovered after a retry, 1 needed the safe default and 1 was overridden by the guardrail. For T900099 the `ConnectionError` is passed up to the caller, and Part D turns it into a 503.

**Why the retries and checks are needed:** a hosted model is not reliable. It can wrap the JSON in prose, cut it off, drop a field or make up a value, and the same prompt can fail twice in a row and then succeed (S9). Without the gates, a bad note would reach an analyst or crash the service. S7 shows the policy check matters: that reply was valid JSON but cited the wrong policy. The guardrail and the safe default mean a flagged payment is never approved automatically.

## Part D: Triage service (`app.py`, `smoke_test.py`)

Run `uvicorn app:app --port 8000` in one terminal and `python smoke_test.py` in another. Restart the server before every smoke run, because the mock model remembers how many times it has seen each transaction ID.

The service rebuilds the model once at start-up and exposes `/health`, `/score`, `/case` and `/metrics`. My own checks reject bad business values with a 400 (`{"error", "field"}`) before the model is called. FastAPI rejects missing or wrongly typed fields with a 422. A payment the model doesn't flag is still held if it's INR 100,000 or more on a new device. If the case-note model is down, `/case` returns a 503 with a fixed message and the service keeps running. One middleware writes one `service.log` line per POST and updates the counters.

**Smoke test (fresh server): 11/11 pass** (happy 3/3, escalation 4/4, edge 4/4).

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

`service.log` has exactly 10 lines (5 INFO, 4 WARNING, 1 ERROR) and no stack traces. Examples:
```
2026-10-23 21:11:30,682 INFO /score status=200 latency_ms=13.4 txn_id=T100001 probability=0.1312 human_review=False
2026-10-23 21:11:30,815 WARNING /case status=200 latency_ms=14.6 txn_id=T900005 probability=0.9596 human_review=True safe_default=True
2026-10-23 21:11:30,849 ERROR /case status=503 latency_ms=14.4 txn_id=T900099 probability=0.9596
```

**Extra check (separate fresh server, `python smoke_test.py --extra`):** T900006 scores 0.4272. That is flagged, but it's below the 0.5 guardrail inside `run_case`. The mock answers `approve`, so the app's own rule changes it to `hold_for_review` and sets `guardrail_override: true`. That's why the app checks every flagged case itself, not just cases above 0.5. The brief's payload scores about 0.4391; I built my own, also between 0.4 and 0.5.

## Limitations

- The data is synthetic, so the precision, recall and threshold won't carry over to real Razorpay traffic without retraining.
- At t\* = 0.4 only about 28% of held payments are fraud, so most of the analysts' work is clearing false alarms.
- TF-IDF only matches words. A question phrased with different words than the policy (e.g. "late-night" vs "midnight") can be refused even when a policy applies.
- The counters live in memory, so they reset on restart and aren't shared if the service runs several workers.
- `mock_llm` is deterministic and offline. A real hosted model would add latency, cost, timeouts and new failure types that these tests don't cover.
- The model is retrained at every start-up, which is fine for 6,000 rows but would be too slow on real data volumes.
