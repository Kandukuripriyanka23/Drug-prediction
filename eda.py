"""
Sprint 2: Exploratory Data Analysis (EDA)
Patient's Condition Classification Using Drug Reviews (P642)

Generates all visualizations and saves to plots/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import sys
import warnings
import os

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURATION
# ============================================================
CLEANED_DATASET = "data/drugsCom_cleaned.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style configuration
plt.style.use('seaborn-darkgrid')
COLORS = {'Depression': '#E74C3C', 'High Blood Pressure': '#3498DB', 'Diabetes, Type 2': '#2ECC71'}
CONDITION_ORDER = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']
sns.set_palette(list(COLORS.values()))

print("=" * 60)
print("  SPRINT 2 - Exploratory Data Analysis")
print("=" * 60)

# Load cleaned dataset
print("\n[*] Loading cleaned dataset...")
df = pd.read_csv(CLEANED_DATASET)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ============================================================
# PLOT 1: CLASS DISTRIBUTION
# ============================================================
print("\n[1/15] Class Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
counts = df['condition'].value_counts().reindex(CONDITION_ORDER)
bars = axes[0].bar(range(len(counts)), counts.values,
                   color=[COLORS[c] for c in counts.index], edgecolor='white', linewidth=1.5)
axes[0].set_xticks(range(len(counts)))
axes[0].set_xticklabels(counts.index, fontsize=10)
axes[0].set_ylabel('Number of Reviews', fontsize=11)
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
                 f'{val:,}', ha='center', fontsize=11, fontweight='bold')

# Pie chart
axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=[COLORS[c] for c in counts.index], startangle=90,
            textprops={'fontsize': 10}, pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white'))
axes[1].set_title('Class Distribution (Donut Chart)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_class_distribution.png")


# ============================================================
# PLOT 2: RATING DISTRIBUTION
# ============================================================
print("[2/15] Rating Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
axes[0].hist(df['rating'], bins=10, range=(0.5, 10.5), color='#5DADE2',
             edgecolor='white', linewidth=1.2, alpha=0.85)
axes[0].set_xlabel('Rating', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Overall Rating Distribution', fontsize=13, fontweight='bold')
axes[0].axvline(df['rating'].mean(), color='#E74C3C', linestyle='--', linewidth=2,
                label=f"Mean: {df['rating'].mean():.2f}")
axes[0].legend(fontsize=10)

# Per condition
for cond in CONDITION_ORDER:
    subset = df[df['condition'] == cond]['rating']
    axes[1].hist(subset, bins=10, range=(0.5, 10.5), alpha=0.5,
                 label=f"{cond} (mean={subset.mean():.1f})", color=COLORS[cond])
axes[1].set_xlabel('Rating', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Rating Distribution by Condition', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_rating_distribution.png")


# ============================================================
# PLOT 3: RATING BY CONDITION (BOXPLOT + VIOLIN)
# ============================================================
print("[3/15] Rating by Condition (Box + Violin)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='condition', y='rating', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[0], width=0.5)
axes[0].set_title('Rating by Condition (Box Plot)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Rating', fontsize=11)

sns.violinplot(data=df, x='condition', y='rating', order=CONDITION_ORDER,
               palette=COLORS, ax=axes[1], inner='quartile')
axes[1].set_title('Rating by Condition (Violin Plot)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Rating', fontsize=11)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_rating_by_condition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_rating_by_condition.png")


# ============================================================
# PLOT 4: USEFUL COUNT DISTRIBUTION
# ============================================================
print("[4/15] Useful Count Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Capped at 100 for better visualization
capped = df['usefulCount'].clip(upper=100)
axes[0].hist(capped, bins=50, color='#AF7AC5', edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Useful Count (capped at 100)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Useful Count Distribution', fontsize=13, fontweight='bold')

sns.violinplot(data=df, x='condition', y='usefulCount', order=CONDITION_ORDER,
               palette=COLORS, ax=axes[1], inner='quartile', cut=0)
axes[1].set_ylim(0, df['usefulCount'].quantile(0.95))
axes[1].set_title('Useful Count by Condition', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_useful_count.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_useful_count.png")


# ============================================================
# PLOT 5: REVIEW LENGTH & WORD COUNT
# ============================================================
print("[5/15] Review Length & Word Count...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Review length histogram
axes[0, 0].hist(df['review_length'], bins=50, color='#48C9B0', edgecolor='white', alpha=0.85)
axes[0, 0].set_xlabel('Review Length (chars)')
axes[0, 0].set_title('Review Length Distribution', fontsize=12, fontweight='bold')

# Word count histogram
axes[0, 1].hist(df['word_count'], bins=50, color='#F39C12', edgecolor='white', alpha=0.85)
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')

# Review length by condition
sns.boxplot(data=df, x='condition', y='review_length', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[1, 0])
axes[1, 0].set_title('Review Length by Condition', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('')

# Word count by condition
sns.boxplot(data=df, x='condition', y='word_count', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[1, 1])
axes[1, 1].set_title('Word Count by Condition', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_review_length_wordcount.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_review_length_wordcount.png")


# ============================================================
# PLOT 6: SENTIMENT ANALYSIS
# ============================================================
print("[6/15] Sentiment Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Polarity distribution
axes[0, 0].hist(df['sentiment_polarity'], bins=50, color='#5DADE2', edgecolor='white', alpha=0.85)
axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
axes[0, 0].set_xlabel('Sentiment Polarity')
axes[0, 0].set_title('Sentiment Polarity Distribution', fontsize=12, fontweight='bold')

# Polarity by condition
sns.boxplot(data=df, x='condition', y='sentiment_polarity', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[0, 1])
axes[0, 1].set_title('Sentiment Polarity by Condition', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('')

# Sentiment category distribution per condition
sent_counts = df.groupby(['condition', 'sentiment_category']).size().unstack(fill_value=0)
sent_counts = sent_counts.reindex(CONDITION_ORDER)
sent_pct = sent_counts.div(sent_counts.sum(axis=1), axis=0) * 100
sent_pct[['Negative', 'Neutral', 'Positive']].plot(
    kind='bar', stacked=True, ax=axes[1, 0],
    color=['#E74C3C', '#95A5A6', '#2ECC71'], edgecolor='white'
)
axes[1, 0].set_title('Sentiment Category by Condition (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].legend(title='Sentiment', fontsize=9)
axes[1, 0].tick_params(axis='x', rotation=15)

# Polarity vs Rating scatter
scatter = axes[1, 1].scatter(df['rating'], df['sentiment_polarity'], alpha=0.1,
                              c=df['condition_label'], cmap='Set1', s=5)
axes[1, 1].set_xlabel('Rating')
axes[1, 1].set_ylabel('Sentiment Polarity')
axes[1, 1].set_title('Rating vs Sentiment Polarity', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/06_sentiment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_sentiment_analysis.png")


# ============================================================
# PLOT 7: WORD CLOUDS PER CONDITION
# ============================================================
print("[7/15] Word Clouds per Condition...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, cond in enumerate(CONDITION_ORDER):
    text = ' '.join(df[df['condition'] == cond]['clean_review'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=150, colormap='viridis',
                   contour_color=COLORS[cond], contour_width=2,
                   random_state=42).generate(text)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(cond, fontsize=14, fontweight='bold', color=COLORS[cond])
    axes[i].axis('off')

plt.suptitle('Word Clouds by Condition', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/07_wordclouds.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_wordclouds.png")


# ============================================================
# PLOT 8: TOP 20 WORDS PER CONDITION
# ============================================================
print("[8/15] Top 20 Words per Condition...")
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for i, cond in enumerate(CONDITION_ORDER):
    text = ' '.join(df[df['condition'] == cond]['clean_review'].dropna().astype(str))
    words = text.split()
    word_freq = Counter(words).most_common(20)
    words_list, counts_list = zip(*word_freq)

    bars = axes[i].barh(range(len(words_list)), counts_list, color=COLORS[cond],
                         edgecolor='white', alpha=0.85)
    axes[i].set_yticks(range(len(words_list)))
    axes[i].set_yticklabels(words_list, fontsize=9)
    axes[i].invert_yaxis()
    axes[i].set_xlabel('Frequency', fontsize=10)
    axes[i].set_title(f'Top 20 Words: {cond}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/08_top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08_top_words.png")


# ============================================================
# PLOT 9: BIGRAM & TRIGRAM ANALYSIS
# ============================================================
print("[9/15] N-gram Analysis (Bigrams & Trigrams)...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, cond in enumerate(CONDITION_ORDER):
    reviews = df[df['condition'] == cond]['clean_review'].dropna().astype(str)

    # Bigrams
    bigram_vec = CountVectorizer(ngram_range=(2, 2), max_features=15)
    bigrams = bigram_vec.fit_transform(reviews)
    bigram_freq = dict(zip(bigram_vec.get_feature_names_out(),
                           bigrams.sum(axis=0).A1))
    bigram_sorted = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    bg_words, bg_counts = zip(*bigram_sorted)

    axes[0, i].barh(range(len(bg_words)), bg_counts, color=COLORS[cond], alpha=0.85)
    axes[0, i].set_yticks(range(len(bg_words)))
    axes[0, i].set_yticklabels(bg_words, fontsize=9)
    axes[0, i].invert_yaxis()
    axes[0, i].set_title(f'Top Bigrams: {cond}', fontsize=11, fontweight='bold')

    # Trigrams
    trigram_vec = CountVectorizer(ngram_range=(3, 3), max_features=15)
    trigrams = trigram_vec.fit_transform(reviews)
    trigram_freq = dict(zip(trigram_vec.get_feature_names_out(),
                            trigrams.sum(axis=0).A1))
    trigram_sorted = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    tg_words, tg_counts = zip(*trigram_sorted)

    axes[1, i].barh(range(len(tg_words)), tg_counts, color=COLORS[cond], alpha=0.7)
    axes[1, i].set_yticks(range(len(tg_words)))
    axes[1, i].set_yticklabels(tg_words, fontsize=8)
    axes[1, i].invert_yaxis()
    axes[1, i].set_title(f'Top Trigrams: {cond}', fontsize=11, fontweight='bold')

plt.suptitle('N-gram Analysis by Condition', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/09_ngrams.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 09_ngrams.png")


# ============================================================
# PLOT 10: TF-IDF TOP FEATURES PER CONDITION
# ============================================================
print("[10/15] TF-IDF Top Features...")
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df['clean_review'].fillna(''))
feature_names = tfidf.get_feature_names_out()

for i, cond in enumerate(CONDITION_ORDER):
    mask = df['condition'] == cond
    cond_mean_tfidf = tfidf_matrix[mask].mean(axis=0).A1
    other_mean_tfidf = tfidf_matrix[~mask].mean(axis=0).A1

    # Discriminative features: high in this condition, low in others
    diff = cond_mean_tfidf - other_mean_tfidf
    top_idx = diff.argsort()[-20:][::-1]
    top_features = [(feature_names[j], diff[j]) for j in top_idx]
    feat_names, feat_scores = zip(*top_features)

    axes[i].barh(range(len(feat_names)), feat_scores, color=COLORS[cond], alpha=0.85)
    axes[i].set_yticks(range(len(feat_names)))
    axes[i].set_yticklabels(feat_names, fontsize=9)
    axes[i].invert_yaxis()
    axes[i].set_xlabel('TF-IDF Diff Score', fontsize=10)
    axes[i].set_title(f'Discriminative Features: {cond}', fontsize=11, fontweight='bold')

plt.suptitle('TF-IDF Discriminative Features by Condition', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_tfidf_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 10_tfidf_features.png")


# ============================================================
# PLOT 11: CORRELATION HEATMAP
# ============================================================
print("[11/15] Correlation Heatmap...")
numeric_cols = ['rating', 'usefulCount', 'review_length', 'word_count', 'avg_word_length',
                'clean_word_count', 'sentiment_polarity', 'sentiment_subjectivity',
                'exclamation_count', 'question_count', 'uppercase_ratio', 'condition_label']

fig, ax = plt.subplots(figsize=(12, 9))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/11_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 11_correlation_heatmap.png")


# ============================================================
# PLOT 12: RATING DISTRIBUTION PER CONDITION (KDE)
# ============================================================
print("[12/15] Rating KDE by Condition...")
fig, ax = plt.subplots(figsize=(10, 6))

for cond in CONDITION_ORDER:
    subset = df[df['condition'] == cond]['rating']
    ax.hist(subset, bins=10, range=(0.5, 10.5), alpha=0.3, color=COLORS[cond], density=True)
    subset.plot.kde(ax=ax, color=COLORS[cond], linewidth=2.5,
                    label=f"{cond} (mean={subset.mean():.1f})")

ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Rating Distribution (KDE) by Condition', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 11)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/12_rating_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 12_rating_kde.png")


# ============================================================
# PLOT 13: TEMPORAL TRENDS
# ============================================================
print("[13/15] Temporal Trends...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reviews over time (monthly)
df['year_month'] = df['date'].dt.to_period('M')
for cond in CONDITION_ORDER:
    monthly = df[df['condition'] == cond].groupby('year_month').size()
    monthly.index = monthly.index.to_timestamp()
    axes[0].plot(monthly.index, monthly.values, color=COLORS[cond],
                 label=cond, linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Date', fontsize=11)
axes[0].set_ylabel('Number of Reviews', fontsize=11)
axes[0].set_title('Monthly Review Count by Condition', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', rotation=30)

# Average rating over time
for cond in CONDITION_ORDER:
    monthly_rating = df[df['condition'] == cond].groupby('year_month')['rating'].mean()
    monthly_rating.index = monthly_rating.index.to_timestamp()
    axes[1].plot(monthly_rating.index, monthly_rating.values, color=COLORS[cond],
                 label=cond, linewidth=1.5, alpha=0.8)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Average Rating', fontsize=11)
axes[1].set_title('Average Rating Over Time', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/13_temporal_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 13_temporal_trends.png")


# ============================================================
# PLOT 14: TOP DRUGS PER CONDITION
# ============================================================
print("[14/15] Top Drugs per Condition...")
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for i, cond in enumerate(CONDITION_ORDER):
    subset = df[df['condition'] == cond]
    top_drugs = subset['drugName'].value_counts().head(10)
    avg_ratings = subset.groupby('drugName')['rating'].mean().reindex(top_drugs.index)

    ax2 = axes[i].twinx()
    bars = axes[i].barh(range(len(top_drugs)), top_drugs.values, color=COLORS[cond],
                         alpha=0.7, edgecolor='white')
    axes[i].set_yticks(range(len(top_drugs)))
    axes[i].set_yticklabels(top_drugs.index, fontsize=9)
    axes[i].invert_yaxis()
    axes[i].set_xlabel('Number of Reviews', fontsize=10, color=COLORS[cond])

    # Overlay avg rating as markers
    ax2.plot(avg_ratings.values, range(len(avg_ratings)), 'D',
             color='#2C3E50', markersize=6)
    ax2.set_xlabel('Avg Rating', fontsize=10, color='#2C3E50')
    ax2.set_xlim(0, 10.5)
    ax2.invert_yaxis()

    axes[i].set_title(f'Top 10 Drugs: {cond}', fontsize=11, fontweight='bold')

plt.suptitle('Top Drugs by Condition (bars=reviews, diamonds=avg rating)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/14_top_drugs.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 14_top_drugs.png")


# ============================================================
# PLOT 15: COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================
print("[15/15] Summary Dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Class distribution
counts = df['condition'].value_counts().reindex(CONDITION_ORDER)
axes[0, 0].bar(range(len(counts)), counts.values,
               color=[COLORS[c] for c in counts.index], edgecolor='white')
axes[0, 0].set_xticks(range(len(counts)))
axes[0, 0].set_xticklabels([c[:12] for c in counts.index], fontsize=8)
axes[0, 0].set_title('Class Distribution', fontsize=11, fontweight='bold')

# (0,1) Rating by condition
sns.boxplot(data=df, x='condition', y='rating', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[0, 1], width=0.5)
axes[0, 1].set_xticklabels([c[:12] for c in CONDITION_ORDER], fontsize=8)
axes[0, 1].set_title('Rating by Condition', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('')

# (0,2) Sentiment by condition
sns.boxplot(data=df, x='condition', y='sentiment_polarity', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[0, 2], width=0.5)
axes[0, 2].set_xticklabels([c[:12] for c in CONDITION_ORDER], fontsize=8)
axes[0, 2].set_title('Sentiment by Condition', fontsize=11, fontweight='bold')
axes[0, 2].set_xlabel('')

# (1,0) Review length by condition
sns.violinplot(data=df, x='condition', y='word_count', order=CONDITION_ORDER,
               palette=COLORS, ax=axes[1, 0], inner='quartile')
axes[1, 0].set_xticklabels([c[:12] for c in CONDITION_ORDER], fontsize=8)
axes[1, 0].set_title('Word Count by Condition', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('')

# (1,1) Rating category distribution
rating_counts = df.groupby(['condition', 'rating_category']).size().unstack(fill_value=0)
rating_counts = rating_counts.reindex(CONDITION_ORDER)
rating_pct = rating_counts.div(rating_counts.sum(axis=1), axis=0) * 100
rating_pct.plot(kind='bar', stacked=True, ax=axes[1, 1],
                color=['#E74C3C', '#F39C12', '#2ECC71'], edgecolor='white')
axes[1, 1].set_title('Rating Category (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_xticklabels([c[:12] for c in CONDITION_ORDER], fontsize=8, rotation=15)
axes[1, 1].legend(title='Rating', fontsize=8, title_fontsize=9)

# (1,2) Useful count by condition
sns.boxplot(data=df, x='condition', y='usefulCount', order=CONDITION_ORDER,
            palette=COLORS, ax=axes[1, 2], width=0.5)
axes[1, 2].set_ylim(0, df['usefulCount'].quantile(0.90))
axes[1, 2].set_xticklabels([c[:12] for c in CONDITION_ORDER], fontsize=8)
axes[1, 2].set_title('Useful Count by Condition', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('')

plt.suptitle('EDA Summary Dashboard - Drug Review Classification',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/15_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 15_summary_dashboard.png")


# ============================================================
# PRINT KEY INSIGHTS
# ============================================================
print(f"\n{'=' * 60}")
print("  KEY EDA INSIGHTS")
print(f"{'=' * 60}")

print(f"""
1. CLASS IMBALANCE:
   - Depression dominates with {counts['Depression']:,} reviews (65%)
   - HBP ({counts['High Blood Pressure']:,}) and Diabetes ({counts['Diabetes, Type 2']:,}) are minorities
   - Ratio: Depression is ~3.9x larger than HBP
   - ACTION: Use SMOTE or class_weight='balanced' in Sprint 3

2. RATINGS:
   - Overall mean rating: {df['rating'].mean():.2f} (positively skewed)
   - Depression mean: {df[df['condition']=='Depression']['rating'].mean():.2f}
   - HBP mean: {df[df['condition']=='High Blood Pressure']['rating'].mean():.2f}
   - Diabetes mean: {df[df['condition']=='Diabetes, Type 2']['rating'].mean():.2f}

3. SENTIMENT:
   - Reviews are slightly positive on average (polarity={df['sentiment_polarity'].mean():.3f})
   - {(df['sentiment_category']=='Positive').sum()} positive, {(df['sentiment_category']=='Neutral').sum()} neutral, {(df['sentiment_category']=='Negative').sum()} negative

4. REVIEW LENGTH:
   - Mean: {df['review_length'].mean():.0f} chars / {df['word_count'].mean():.0f} words
   - After cleaning: {df['clean_word_count'].mean():.0f} words avg

5. DISCRIMINATIVE TEXT FEATURES:
   - Depression: anxiety, mood, antidepressant, sleep
   - High Blood Pressure: blood pressure, bp, hypertension, amlodipine
   - Diabetes Type 2: blood sugar, insulin, metformin, a1c

6. TOP DRUGS:
   - Depression: Bupropion, Sertraline, Venlafaxine
   - HBP: Lisinopril, Losartan, Amlodipine
   - Diabetes: Liraglutide, Victoza, Dulaglutide
""")

# Count total plots
plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
print(f"\n  Total plots saved: {len(plot_files)} in {PLOTS_DIR}/")
for f in sorted(plot_files):
    print(f"    - {f}")

print(f"\n{'=' * 60}")
print("  SPRINT 2 COMPLETE - EDA Done")
print(f"{'=' * 60}")
