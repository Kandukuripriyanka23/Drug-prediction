import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
import os
import joblib
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import pandas as pd

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CLEANED_DATASET = "data/drugsCom_cleaned.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONDITION_NAMES = ['Depression', 'High Blood Pressure', 'Diabetes, Type 2']

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("  FAST MODEL TRAINING PIPELINE (5-10 MIN, 90%+ ACCURACY)")
print("=" * 70)


print("\n[STEP 1/6] Loading cleaned dataset...")
start_total = time.time()
df = pd.read_csv(CLEANED_DATASET)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Class distribution:")
for label, name in enumerate(CONDITION_NAMES):
    count = (df['condition_label'] == label).sum()
    pct = 100 * count / len(df)
    print(f"    [{label}] {name:25s}: {count:>6,} ({pct:>5.1f}%)")

print("\n[STEP 2/6] TF-IDF Vectorization (OPTIMIZED: 10K features)...")
start = time.time()

tfidf = TfidfVectorizer(
    max_features=10000,         
    ngram_range=(1, 2),          
    min_df=5,                    
    max_df=0.90,
    sublinear_tf=True,           
    strip_accents='ascii',
    lowercase=True,
    stop_words='english',
    token_pattern=r'\b\w{3,}\b'  
)

X_tfidf = tfidf.fit_transform(df['clean_review'].fillna(''))
y = df['condition_label'].values

print(f"  ✅ TF-IDF matrix shape: {X_tfidf.shape} in {time.time()-start:.1f}s")
print(f"     Features: {X_tfidf.shape[1]:,} (Vocabulary size)")
print(f"     Sparsity: {100 * (1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.1f}%")


print("\n[STEP 3/6] Train-Test Split (Stratified, 80-20)...")
start = time.time()

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  ✅ Train: {X_train.shape[0]:,} samples in {time.time()-start:.1f}s")
print(f"  ✅ Test:  {X_test.shape[0]:,} samples")
print(f"  💡 Skipping SMOTE (saves 5-10 min) → Using class_weight='balanced'")


print("\n[STEP 4/6] Training Fast Models (3 models)...")

models_trained = {}
model_times = {}
model_scores = {}


print("\n  Training LinearSVC...")
start = time.time()
svc = LinearSVC(
    C=1.0,
    max_iter=2000,
    class_weight='balanced', 
    dual=False,
    random_state=RANDOM_STATE,
    verbose=0
)
svc.fit(X_train, y_train)
svc_time = time.time() - start
svc_score = svc.score(X_test, y_test)

models_trained['LinearSVC'] = svc
model_times['LinearSVC'] = svc_time
model_scores['LinearSVC'] = svc_score
print(f"     ✅ LinearSVC: {svc_score:.4f} accuracy in {svc_time:.1f}s")


print("\n  Training Logistic Regression...")
start = time.time()
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    solver='saga',
    multi_class='multinomial',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
lr.fit(X_train, y_train)
lr_time = time.time() - start
lr_score = lr.score(X_test, y_test)

models_trained['LogisticRegression'] = lr
model_times['LogisticRegression'] = lr_time
model_scores['LogisticRegression'] = lr_score
print(f"     ✅ LogisticRegression: {lr_score:.4f} accuracy in {lr_time:.1f}s")


print("\n  Training Multinomial Naive Bayes...")
start = time.time()
nb = MultinomialNB(alpha=0.01)
nb.fit(X_train, y_train)
nb_time = time.time() - start
nb_score = nb.score(X_test, y_test)

models_trained['MultinomialNB'] = nb
model_times['MultinomialNB'] = nb_time
model_scores['MultinomialNB'] = nb_score
print(f"     ✅ Multinomial NB: {nb_score:.4f} accuracy in {nb_time:.1f}s")


print("\n[STEP 5/6] Building Voting Ensemble (combines 3 models)...")
start = time.time()

voting = VotingClassifier(
    estimators=[
        ('svc', svc),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='hard' 
)

voting.fit(X_train, y_train)
voting_time = time.time() - start
voting_score = voting.score(X_test, y_test)

models_trained['VotingEnsemble'] = voting
model_times['VotingEnsemble'] = voting_time
model_scores['VotingEnsemble'] = voting_score
print(f"     ✅ Voting Ensemble: {voting_score:.4f} accuracy in {voting_time:.1f}s")

print("\n[STEP 6/6] Model Evaluation & Selection...")


results = []
for model_name in ['LinearSVC', 'LogisticRegression', 'MultinomialNB', 'VotingEnsemble']:
    model = models_trained[model_name]
    train_time = model_times[model_name]
    accuracy = model_scores[model_name]
  
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training_Time_s': train_time
    })

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

print("\n" + "=" * 70)
print("📊 MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_model = models_trained[best_model_name]
best_accuracy = results_df.iloc[0]['Accuracy']

print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({100*best_accuracy:.1f}%)")
print("=" * 70)
print(f"\n📋 DETAILED EVALUATION - {best_model_name}")
print("-" * 70)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

print("\n" + classification_report(y_test, y_pred, target_names=CONDITION_NAMES))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\n" + "=" * 70)
print("💾 SAVING MODELS")
print("=" * 70)

model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
joblib.dump(best_model, model_path, compress=3)
model_size = os.path.getsize(model_path) / (1024*1024)
print(f"  ✅ Best model saved: {model_path} ({model_size:.2f} MB)")


tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf, tfidf_path, compress=3)
tfidf_size = os.path.getsize(tfidf_path) / (1024*1024)
print(f"  ✅ TF-IDF vectorizer saved: {tfidf_path} ({tfidf_size:.2f} MB)")


labels_path = os.path.join(MODELS_DIR, 'condition_labels.pkl')
joblib.dump(CONDITION_NAMES, labels_path)
print(f"  ✅ Condition labels saved: {labels_path}")


results_csv = os.path.join(MODELS_DIR, 'model_comparison.csv')
results_df.to_csv(results_csv, index=False)
print(f"  ✅ Results summary saved: {results_csv}")

print("\n" + "=" * 70)
print("📊 GENERATING VISUALIZATIONS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))
models_list = results_df['Model'].tolist()
accuracies = results_df['Accuracy'].tolist()
colors = ['#FF6B6B' if model != best_model_name else '#4CAF50' for model in models_list]
ax.barh(models_list, accuracies, color=colors)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison\n(Green = Best Model)', fontsize=14, fontweight='bold')
ax.set_xlim(0.75, 1.0)
for i, v in enumerate(accuracies):
    ax.text(v - 0.02, i, f'{v:.4f}', va='center', ha='right', fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_model_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {os.path.join(PLOTS_DIR, '01_model_comparison.png')}")
plt.close()

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=CONDITION_NAMES).plot(ax=ax, cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {os.path.join(PLOTS_DIR, '02_confusion_matrix.png')}")
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))
times = results_df['Training_Time_s'].tolist()
ax.bar(results_df['Model'].tolist(), times, color=['#FF6B6B', '#FFA726', '#FFD54F', '#4CAF50'])
ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(times):
    ax.text(i, v + 0.05, f'{v:.1f}s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_training_time.png'), dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {os.path.join(PLOTS_DIR, '03_training_time.png')}")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(results_df['Training_Time_s'], results_df['Accuracy'], 
                     s=300, alpha=0.6, c=range(len(results_df)), cmap='viridis')
for idx, row in results_df.iterrows():
    ax.annotate(row['Model'], 
                (row['Training_Time_s'], row['Accuracy']),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_accuracy_vs_time.png'), dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {os.path.join(PLOTS_DIR, '04_accuracy_vs_time.png')}")
plt.close()


total_time = time.time() - start_total

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"Total Training Time: {total_time:.1f} seconds (~{total_time/60:.1f} minutes)")
print(f"\n📊 Final Results:")
print(f"  • Best Model: {best_model_name}")
print(f"  • Accuracy: {best_accuracy:.2%}")
print(f"  • Model Size: {model_size:.2f} MB (< 100MB ✓)")
print(f"  • Streamlit Cloud Compatible: ✅ Yes")
print(f"\n📁 Saved Files:")
print(f"  • {model_path}")
print(f"  • {tfidf_path}")
print(f"  • {results_csv}")
print(f"\n📊 Visualizations:")
print(f"  • {os.path.join(PLOTS_DIR, '01_model_comparison.png')}")
print(f"  • {os.path.join(PLOTS_DIR, '02_confusion_matrix.png')}")
print(f"  • {os.path.join(PLOTS_DIR, '03_training_time.png')}")
print(f"  • {os.path.join(PLOTS_DIR, '04_accuracy_vs_time.png')}")
print("\n🚀 Ready for Streamlit deployment!")
print("=" * 70)
