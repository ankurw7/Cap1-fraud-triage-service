"""prepare.py - Part A: audit, clean and enrich transactions_raw.csv.
Run:  python prepare.py

    from prepare import load_raw, clean_transactions, add_features
    df = add_features(clean_transactions(load_raw()))
"""

import numpy as np
import pandas as pd

#the constants
RAW_PATH = "transactions_raw.csv"
TEXT_COLUMNS = ["merchant_category", "city"]
MIN_VALID_AMOUNT = 1
MAX_VALID_AMOUNT = 1_000_000      # genuine amounts top out near INR 106k
NIGHT_LAST_HOUR = 5               # hour_of_day <= 5 counts as night
HIGH_VALUE_INR = 50_000
EXCLUDED_COLUMNS = ["chargeback_filed", "txn_id", "customer_id", "is_fraud"]

#Loading the raw CSV
def load_raw(path=RAW_PATH):
    return pd.read_csv(path)

#Detecting invalid transaction amounts
def out_of_range(amount):
    return (amount < MIN_VALID_AMOUNT) | (amount > MAX_VALID_AMOUNT)

#Cleaning the transactions
def clean_transactions(raw_df):
   
    df = raw_df.copy()                             #Make a copy  
    df = df.drop_duplicates()                      #Remove exact duplicates
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().str.title()  #Normalize text columns
    df = df[df["amount_inr"].notna()]              #Remove missing amounts
    df = df[~out_of_range(df["amount_inr"])]       #Remove invalid amounts
    return df.reset_index(drop=True)               #Reset the index

#Feature engineering
def add_features(df):
    df = df.copy()

    df["amount_vs_avg_ratio"] = df["amount_inr"] / df["avg_amount_30d_inr"]             #feature1: amount vs average ratio
    df["is_night"] = (df["hour_of_day"] <= NIGHT_LAST_HOUR).astype(int)                 #feature2: is night transaction
    df["high_value"] = (df["amount_inr"] > HIGH_VALUE_INR).astype(int)                  #feature3: is high value transaction
    return df


clean = clean_transactions

#Leakage table for chargeback_filed 
def leakage_table(df):   
    table = df.groupby("chargeback_filed")["is_fraud"].agg(rows="size", fraud="sum")
    table["fraud_rate"] = table["fraud"] / table["rows"]
    return table

#"Never fraud" baseline: accuracy and recall
def never_fraud_baseline(df):
    y = df["is_fraud"]
    accuracy = (y == 0).mean()
    recall = 0.0                        # it never flags anything, so it catches no fraud
    return accuracy, recall

#Both functions on a one-row frame: correct output, input unchanged.
def self_test():    
    row = pd.DataFrame([{
        "txn_id": "T1", "customer_id": "C1", "merchant_category": "  electronics ",
        "city": "PUNE ", "payment_mode": "UPI", "amount_inr": 60_000.0,
        "hour_of_day": 3, "txn_count_last_24h": 4, "avg_amount_30d_inr": 6_000.0,
        "distance_from_home_km": np.nan, "is_new_device": 1, "account_age_days": 10,
        "chargeback_filed": 0, "is_fraud": 1,
    }])

    before = row.copy()
    cleaned = clean_transactions(row)
    featured = add_features(cleaned)
    pd.testing.assert_frame_equal(row, before)
    assert "amount_vs_avg_ratio" not in cleaned.columns
    assert cleaned.loc[0, "merchant_category"] == "Electronics"
    assert cleaned.loc[0, "city"] == "Pune"
    assert pd.isna(cleaned.loc[0, "distance_from_home_km"])
    assert featured.loc[0, "amount_vs_avg_ratio"] == 10.0
    assert featured.loc[0, "is_night"] == 1
    assert featured.loc[0, "high_value"] == 1
    bad = row.assign(amount_inr=0.5)
    assert len(clean_transactions(bad)) == 0
    print("Self-test passed: one-row frame cleaned and enriched, input unchanged")

#Auditing the dataset in main()
def main():
    raw = load_raw()

    print("== Audit (raw file) ==")
#Check the dataset size and duplicates
    n_dupes = int(raw.duplicated().sum())
    n_cat_raw = raw["merchant_category"].nunique()
    n_city_raw = raw["city"].nunique()
    
    print(f"Raw shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print(f"Exact duplicates: {n_dupes}")
    print(f"Distinct raw spellings: merchant_category {n_cat_raw}, city {n_city_raw}")
#assertions: to check that the condition is true, if not it raises an AssertionError
    assert raw.shape == (6_090, 14)
    assert n_dupes == 90
    assert (n_cat_raw, n_city_raw) == (24, 24)

#Count missing and invalid amounts
    dedup = raw.drop_duplicates()

    n_miss_amt = int(dedup["amount_inr"].isna().sum())
    n_miss_dist = int(dedup["distance_from_home_km"].isna().sum())
    n_bad_amt = int(out_of_range(dedup["amount_inr"]).sum())

    print(f"After de-dup: {len(dedup):,} rows; {n_miss_amt} missing amount; "
          f"{n_miss_dist} missing distance; {n_bad_amt} out-of-range amounts")
#assertions
    assert (len(dedup), n_miss_amt, n_miss_dist, n_bad_amt) == (6_000, 76, 193, 5)

#Verify the cleaned data
    print("\n== Clean ==")
    df = clean_transactions(raw)

#The script also checks the merchant category, city count and payment-mode distribution.
    n_dist_left = int(df["distance_from_home_km"].isna().sum())
    print(f"Clean rows: {len(df):,}; missing distance remaining (imputed in Stage 2): {n_dist_left}")
    assert len(df) == 5_919
    assert n_dist_left == 191

    cats = df["merchant_category"].value_counts()
    print("Categories:", ", ".join(f"{k} {v:,}" for k, v in cats.items()))
    assert cats.to_dict() == {"Grocery": 1_406, "Food Delivery": 1_326, "Fashion": 976,
                              "Electronics": 845, "Utilities": 685, "Travel": 681}
    print("Cities:", ", ".join(sorted(df["city"].unique())))
    assert df["city"].nunique() == 6

    modes = df["payment_mode"].value_counts()
    print("Payment modes:", ", ".join(f"{k} {v:,}" for k, v in modes.items()))
    assert modes.to_dict() == {"UPI": 3_221, "Card": 1_450, "NetBanking": 731, "Wallet": 517}

    amt_min, amt_max = df["amount_inr"].min(), df["amount_inr"].max()
    print(f"Amount range: min {amt_min:,.2f}, max {amt_max:,.2f}")
    assert (round(amt_min, 2), round(amt_max, 2)) == (141.37, 105_883.95)

#Verify feature engineering
    print("\n== Features ==")
    df = add_features(df)
    n_night, n_high = int(df["is_night"].sum()), int(df["high_value"].sum())
    max_ratio = df["amount_vs_avg_ratio"].max()
    print(f"is_night: {n_night}; high_value: {n_high}; max amount_vs_avg_ratio: {max_ratio:.2f}")
    assert (n_night, n_high, round(max_ratio, 2)) == (365, 17, 13.11)

#Understanding the fraud baseline
    print("\n== Target and baseline ==")
    n_fraud = int(df["is_fraud"].sum())
    accuracy, recall = never_fraud_baseline(df)
    print(f"Fraud: {n_fraud} of {len(df):,} ({n_fraud / len(df):.2%})")
    print(f"'Never fraud' baseline: accuracy {accuracy:.2%}, recall {recall:.0f}")
    assert n_fraud == 366
    assert round(accuracy * 100, 2) == 93.82

    print("\n== Leakage: chargeback_filed ==")
    table = leakage_table(df)
    for flag, r in table.iterrows():
        print(f"chargeback {flag}: {int(r['rows']):,} rows, {int(r['fraud'])} fraud ({r['fraud_rate']:.1%})")
    assert table.loc[1, ["rows", "fraud"]].tolist() == [373, 304]
    assert table.loc[0, ["rows", "fraud"]].tolist() == [5_546, 62]
    print(f"Excluded from features: {', '.join(EXCLUDED_COLUMNS)}")

    print()
    self_test()


if __name__ == "__main__":
    main()
