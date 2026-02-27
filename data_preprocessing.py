"""
Sprint 1: Data Preprocessing Pipeline
Patient's Condition Classification Using Drug Reviews (P642)

This script:
1. Loads the raw dataset
2. Filters to 3 target conditions (Depression, High Blood Pressure, Diabetes Type 2)
3. Cleans and preprocesses review text
4. Engineers new features
5. Saves the cleaned dataset
"""

import pandas as pd
import numpy as np
import re
import sys
import time
import warnings
from html import unescape

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DATASET = "drugsCom_raw.xlsx"
CLEANED_DATASET = "data/drugsCom_cleaned.csv"
TARGET_CONDITIONS = ["Depression", "High Blood Pressure", "Diabetes, Type 2"]
CONDITION_LABELS = {
    "Depression": 0,
    "High Blood Pressure": 1,
    "Diabetes, Type 2": 2
}
LABEL_NAMES = {v: k for k, v in CONDITION_LABELS.items()}

# ============================================================
# STEP 1: LOAD & FILTER
# ============================================================
print("=" * 60)
print("  SPRINT 1 - Data Preprocessing Pipeline")
print("=" * 60)

print("\n[STEP 1/5] Loading raw dataset...")
start = time.time()
df_raw = pd.read_excel(RAW_DATASET)
print(f"  Loaded {df_raw.shape[0]:,} rows in {time.time()-start:.1f}s")

# Drop the unnamed index column if present
if 'Unnamed: 0' in df_raw.columns:
    df_raw.drop('Unnamed: 0', axis=1, inplace=True)
    print("  Dropped 'Unnamed: 0' index column")

# Filter to target conditions
print(f"\n[STEP 1/5] Filtering to target conditions...")
df = df_raw[df_raw['condition'].isin(TARGET_CONDITIONS)].copy()
df.reset_index(drop=True, inplace=True)
print(f"  Filtered: {df_raw.shape[0]:,} -> {df.shape[0]:,} rows")
print(f"  Columns: {list(df.columns)}")

# Add label column
df['condition_label'] = df['condition'].map(CONDITION_LABELS)
print(f"\n  Class distribution:")
for cond, label in CONDITION_LABELS.items():
    count = (df['condition_label'] == label).sum()
    pct = count / len(df) * 100
    print(f"    [{label}] {cond:25s}: {count:>6,} ({pct:.1f}%)")


# ============================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================
print(f"\n[STEP 2/5] Text Preprocessing...")
start = time.time()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
# Add custom medical/drug-related stopwords to keep domain-relevant terms
custom_stopwords = {'drug', 'take', 'taking', 'took', 'taken', 'mg', 'pill',
                    'pills', 'tablet', 'dose', 'day', 'week', 'month', 'year',
                    'doctor', 'prescribed', 'medication', 'medicine', 'started',
                    'one', 'would', 'could', 'also', 'get', 'got', 'like',
                    'really', 'much', 'even', 'still', 'back', 'going', 'went',
                    'time', 'first', 'two', 'three', 'well', 'since', 'put',
                    'made', 'make', 'using', 'used', 'use'}
stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()


def clean_html(text):
    """Remove HTML entities and tags"""
    text = unescape(text)                           # &#039; -> '
    text = re.sub(r'<[^>]+>', '', text)             # Remove HTML tags
    text = re.sub(r'&[a-zA-Z]+;', '', text)         # Remove remaining HTML entities
    return text


def preprocess_review(text):
    """
    Full preprocessing pipeline for a single review.
    
    Steps:
    1. Handle NaN / non-string
    2. Clean HTML entities and tags
    3. Lowercase
    4. Remove URLs
    5. Remove numbers (but keep words with numbers like '10mg' initially)
    6. Remove special characters, keep only letters and spaces
    7. Tokenize
    8. Remove stopwords
    9. Lemmatize
    10. Remove very short tokens (len < 3)
    11. Rejoin
    """
    # Handle NaN
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Clean HTML
    text = clean_html(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()  # Faster than word_tokenize for simple space-split
    
    # Remove stopwords and short tokens, then lemmatize
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) >= 3
    ]
    
    return ' '.join(tokens)


# Apply preprocessing with progress tracking
total = len(df)
batch_size = total // 10

print(f"  Processing {total:,} reviews...")
cleaned_reviews = []
for i, review in enumerate(df['review']):
    cleaned_reviews.append(preprocess_review(review))
    if (i + 1) % batch_size == 0:
        pct = (i + 1) / total * 100
        print(f"    Progress: {pct:.0f}% ({i+1:,}/{total:,})")

df['clean_review'] = cleaned_reviews
elapsed = time.time() - start
print(f"  Text preprocessing complete in {elapsed:.1f}s")

# Show sample before/after
print(f"\n  Sample preprocessing (first review):")
print(f"    BEFORE: \"{str(df['review'].iloc[0])[:120]}...\"")
print(f"    AFTER : \"{str(df['clean_review'].iloc[0])[:120]}...\"")


# ============================================================
# STEP 3: HANDLE EMPTY REVIEWS
# ============================================================
print(f"\n[STEP 3/5] Handling empty reviews after preprocessing...")
empty_count = (df['clean_review'].str.strip() == '').sum()
print(f"  Empty reviews after cleaning: {empty_count}")
if empty_count > 0:
    df = df[df['clean_review'].str.strip() != ''].reset_index(drop=True)
    print(f"  Removed {empty_count} empty reviews. New shape: {df.shape[0]:,}")


# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print(f"\n[STEP 4/5] Feature Engineering...")
start = time.time()

# Original review features (from raw text)
df['review_length'] = df['review'].astype(str).str.len()
df['word_count'] = df['review'].astype(str).str.split().str.len()
df['avg_word_length'] = df['review'].astype(str).apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
)

# Clean review features
df['clean_word_count'] = df['clean_review'].str.split().str.len()
df['clean_review_length'] = df['clean_review'].str.len()

# Sentiment analysis using TextBlob
print("  Computing sentiment scores...")
df['sentiment_polarity'] = df['review'].astype(str).apply(
    lambda x: TextBlob(x).sentiment.polarity
)
df['sentiment_subjectivity'] = df['review'].astype(str).apply(
    lambda x: TextBlob(x).sentiment.subjectivity
)

# Sentiment category
df['sentiment_category'] = pd.cut(
    df['sentiment_polarity'],
    bins=[-1.01, -0.1, 0.1, 1.01],
    labels=['Negative', 'Neutral', 'Positive']
)

# Rating category
df['rating_category'] = pd.cut(
    df['rating'],
    bins=[0, 3, 6, 10],
    labels=['Low', 'Medium', 'High']
)

# Exclamation & question marks count (from original review)
df['exclamation_count'] = df['review'].astype(str).str.count('!')
df['question_count'] = df['review'].astype(str).str.count(r'\?')

# Has uppercase emphasis
df['uppercase_ratio'] = df['review'].astype(str).apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
)

elapsed = time.time() - start
print(f"  Feature engineering complete in {elapsed:.1f}s")

# Feature summary
print(f"\n  Engineered Features:")
eng_features = ['review_length', 'word_count', 'avg_word_length', 'clean_word_count',
                'clean_review_length', 'sentiment_polarity', 'sentiment_subjectivity',
                'sentiment_category', 'rating_category', 'exclamation_count',
                'question_count', 'uppercase_ratio']
for feat in eng_features:
    if df[feat].dtype in ['float64', 'int64']:
        print(f"    {feat:30s}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")
    else:
        print(f"    {feat:30s}: {df[feat].value_counts().to_dict()}")


# ============================================================
# STEP 5: SAVE CLEANED DATASET
# ============================================================
print(f"\n[STEP 5/5] Saving cleaned dataset...")

# Final column order
final_columns = [
    'drugName', 'condition', 'condition_label',
    'review', 'clean_review',
    'rating', 'rating_category',
    'date', 'usefulCount',
    'review_length', 'word_count', 'avg_word_length',
    'clean_word_count', 'clean_review_length',
    'sentiment_polarity', 'sentiment_subjectivity', 'sentiment_category',
    'exclamation_count', 'question_count', 'uppercase_ratio'
]

df = df[final_columns]
df.to_csv(CLEANED_DATASET, index=False, encoding='utf-8')
file_size = pd.io.common.file_exists(CLEANED_DATASET)

print(f"  Saved to: {CLEANED_DATASET}")
print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Final Summary
print(f"\n{'=' * 60}")
print("  SPRINT 1 COMPLETE - Preprocessing Pipeline Done")
print(f"{'=' * 60}")
print(f"\n  Dataset Summary:")
print(f"    Raw records      : {df_raw.shape[0]:,}")
print(f"    Filtered records : {df.shape[0]:,}")
print(f"    Features         : {df.shape[1]} columns")
print(f"    Clean reviews    : {(df['clean_review'].str.len() > 0).sum():,} non-empty")
print(f"\n  Class Distribution:")
for cond, label in CONDITION_LABELS.items():
    count = (df['condition_label'] == label).sum()
    print(f"    [{label}] {cond:25s}: {count:>6,}")
print(f"\n  Output: {CLEANED_DATASET}")
print(f"{'=' * 60}")
