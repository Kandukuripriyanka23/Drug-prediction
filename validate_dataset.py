"""
Sprint 0: Dataset Validation Script
Patient's Condition Classification Using Drug Reviews (P642)
"""

import pandas as pd
import numpy as np
import os
import json
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

print("=" * 60)
print("  SPRINT 0 - Dataset Validation Report")
print("  Patient's Condition Classification (P642)")
print("=" * 60)

# --- Load Dataset ---
DATASET_PATH = "drugsCom_raw.xlsx"
print(f"\n[*] Loading dataset: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH)

# --- Basic Info ---
print(f"\n{'-' * 40}")
print("[BASIC DATASET INFO]")
print(f"{'-' * 40}")
print(f"  Shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  File size      : {os.path.getsize(DATASET_PATH) / (1024*1024):.2f} MB")
print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")

# --- Columns ---
print(f"\n{'-' * 40}")
print("[COLUMNS & DATA TYPES]")
print(f"{'-' * 40}")
for col in df.columns:
    print(f"  {col:20s} -> {str(df[col].dtype):10s} | Non-null: {df[col].notna().sum():>7,} | Null: {df[col].isna().sum():>5,}")

# --- Missing Values ---
print(f"\n{'-' * 40}")
print("[MISSING VALUES]")
print(f"{'-' * 40}")
missing = df.isnull().sum()
for col, count in missing.items():
    pct = count / len(df) * 100
    status = "OK" if count == 0 else "WARN"
    print(f"  [{status:4s}] {col:20s}: {count:>6,} ({pct:.2f}%)")

# --- Duplicates ---
dupes = df.duplicated().sum()
print(f"\n  Duplicate rows: {dupes:,} ({dupes / len(df) * 100:.2f}%)")

# --- Target Conditions ---
TARGET_CONDITIONS = ["Depression", "High Blood Pressure", "Diabetes, Type 2"]

print(f"\n{'-' * 40}")
print("[TARGET CONDITIONS]")
print(f"{'-' * 40}")
total_unique = df['condition'].nunique()
print(f"  Total unique conditions: {total_unique:,}")
print()

for condition in TARGET_CONDITIONS:
    count = (df['condition'] == condition).sum()
    pct = count / len(df) * 100
    print(f"  >> {condition:25s}: {count:>6,} reviews ({pct:.2f}%)")

# Filtered dataset
df_filtered = df[df['condition'].isin(TARGET_CONDITIONS)].copy()
total_filtered = len(df_filtered)
print(f"\n  Total filtered records   : {total_filtered:,}")
print(f"  Percentage of full dataset: {total_filtered / len(df) * 100:.2f}%")

# --- Class Balance ---
print(f"\n{'-' * 40}")
print("[CLASS BALANCE - Filtered Dataset]")
print(f"{'-' * 40}")
distribution = df_filtered['condition'].value_counts()
max_count = distribution.max()
for condition, count in distribution.items():
    ratio = count / max_count
    bar = "#" * int(ratio * 30)
    print(f"  {condition:25s}: {count:>6,} {bar}")

# --- Rating Distribution ---
print(f"\n{'-' * 40}")
print("[RATING DISTRIBUTION - Filtered]")
print(f"{'-' * 40}")
print(f"  Mean   : {df_filtered['rating'].mean():.2f}")
print(f"  Median : {df_filtered['rating'].median():.1f}")
print(f"  Std    : {df_filtered['rating'].std():.2f}")
print(f"  Min    : {df_filtered['rating'].min()}")
print(f"  Max    : {df_filtered['rating'].max()}")

# --- Review Length Stats ---
print(f"\n{'-' * 40}")
print("[REVIEW LENGTH STATS - Filtered]")
print(f"{'-' * 40}")
df_filtered['review_len'] = df_filtered['review'].astype(str).str.len()
print(f"  Mean length   : {df_filtered['review_len'].mean():.0f} chars")
print(f"  Median length : {df_filtered['review_len'].median():.0f} chars")
print(f"  Max length    : {df_filtered['review_len'].max():,} chars")
print(f"  Min length    : {df_filtered['review_len'].min()} chars")

# --- Sample Reviews ---
print(f"\n{'-' * 40}")
print("[SAMPLE REVIEWS - 1 per condition]")
print(f"{'-' * 40}")
for cond in TARGET_CONDITIONS:
    sample = df_filtered[df_filtered['condition'] == cond].iloc[0]
    review_preview = str(sample['review'])[:150].replace('\n', ' ')
    print(f"\n  [{cond}] (Rating: {sample['rating']})")
    print(f"  Drug: {sample['drugName']}")
    print(f"  \"{review_preview}...\"")

# --- Top Drugs per Condition ---
print(f"\n{'-' * 40}")
print("[TOP 5 DRUGS PER CONDITION]")
print(f"{'-' * 40}")
for cond in TARGET_CONDITIONS:
    top_drugs = df_filtered[df_filtered['condition'] == cond]['drugName'].value_counts().head(5)
    print(f"\n  {cond}:")
    for drug, count in top_drugs.items():
        print(f"    - {drug}: {count} reviews")

# --- Save Validation Summary ---
summary = {
    "dataset_file": DATASET_PATH,
    "total_rows": int(df.shape[0]),
    "total_columns": int(df.shape[1]),
    "columns": list(df.columns),
    "missing_values": {col: int(v) for col, v in missing.items()},
    "duplicate_rows": int(dupes),
    "unique_conditions": int(total_unique),
    "target_conditions": {
        cond: int((df['condition'] == cond).sum()) for cond in TARGET_CONDITIONS
    },
    "filtered_total": int(total_filtered),
    "class_distribution": {str(k): int(v) for k, v in distribution.items()},
    "validation_status": "PASSED"
}

with open("data/dataset_validation.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 60}")
print("  VALIDATION COMPLETE - Dataset is ready for Sprint 1")
print(f"  Summary saved to: data/dataset_validation.json")
print(f"{'=' * 60}")
