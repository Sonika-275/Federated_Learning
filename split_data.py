# split_data.py
# PURPOSE: Take the raw Kaggle CSV and split into 3 bank datasets

import pandas as pd
import numpy as np
import os

# ── STEP 1: Load the raw dataset ──────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')
print(f"Total rows loaded: {len(df)}")
print(f"Total fraud cases: {df['Class'].sum()}")
print(f"Total legit cases: {len(df) - df['Class'].sum()}")

# ── STEP 2: Shuffle the data ──────────────────────────────────
# Why shuffle? So each bank gets a mix of fraud and legit cases
# Without shuffling, fraud cases might all land in one bank
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# random_state=42 means shuffle is reproducible — same result every run

# ── STEP 3: Split into 3 equal parts ─────────────────────────
total = len(df)
split1 = total // 3        # first cut point
split2 = 2 * total // 3   # second cut point

bank_a = df.iloc[:split1]           # rows 0 to split1
bank_b = df.iloc[split1:split2]     # rows split1 to split2
bank_c = df.iloc[split2:]           # rows split2 to end

# ── STEP 4: Save each bank's data ────────────────────────────
bank_a.to_csv('data/bank_a.csv', index=False)
bank_b.to_csv('data/bank_b.csv', index=False)
bank_c.to_csv('data/bank_c.csv', index=False)

# ── STEP 5: Print summary ─────────────────────────────────────
print("\n── Split Complete ──")
print(f"Bank A (SBI)   → {len(bank_a)} rows | {bank_a['Class'].sum()} fraud cases")
print(f"Bank B (HDFC)  → {len(bank_b)} rows | {bank_b['Class'].sum()} fraud cases")
print(f"Bank C (ICICI) → {len(bank_c)} rows | {bank_c['Class'].sum()} fraud cases")
print("\nFiles saved to data/ folder")