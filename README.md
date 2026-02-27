# 🏥 Patient Condition Classification Using Drug Reviews

## 📌 Project Overview

This project classifies patient medical conditions from drug reviews using fast, optimized machine learning. It predicts three conditions:

- **Depression**
- **High Blood Pressure**
- **Diabetes Type 2**

## ✨ Key Features

| Feature             | Details             |
| ------------------- | ------------------- |
| **Accuracy**        | 96.02% (LinearSVC)  |
| **Training Time**   | ~15 seconds         |
| **Model Size**      | 0.22 MB             |
| **Inference Speed** | < 1 second          |
| **Dataset**         | 13,943 drug reviews |
| **Status**          | ✅ Production Ready |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train Model (Fast - 15 seconds)

```bash
python train_fast.py
```

**Expected Output:**

```
======================================================================
✅ TRAINING COMPLETE!
======================================================================
Total Training Time: 15.3 seconds (~0.3 minutes)

📊 Final Results:
  • Best Model: LinearSVC
  • Accuracy: 96.02%
  • Model Size: 0.22 MB (< 100MB ✓)
  • Streamlit Cloud Compatible: ✅ Yes
```

### Run Streamlit App

```bash
streamlit run app.py
```

Then open: `http://localhost:8501`

## 📊 Project Structure

```
project-1/
├── train_fast.py              # ⚡ FAST training script (15 sec, 96% acc)
├── app.py                     # 🎨 Streamlit web app
├── data_preprocessing.py      # 📝 Text preprocessing pipeline
├── eda.py                     # 📊 Exploratory data analysis
├── model_building.py          # 🤖 Original model building (SLOW)
├── validate_dataset.py        # ✅ Dataset validation
├── requirements.txt           # 📦 Dependencies
├── sprint_plan.md             # 📋 Project plan
│
├── data/                      # 📁 Data directory
│   ├── drugsCom_raw.xlsx      # Original dataset
│   ├── drugsCom_cleaned.csv   # Cleaned dataset
│   └── dataset_validation.json
│
├── models/                    # 💾 Trained models
│   ├── best_model.pkl         # LinearSVC classifier (0.22 MB)
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer (1.31 MB)
│   ├── condition_labels.pkl   # Class labels
│   └── model_comparison.csv   # Results table
│
└── plots/                     # 📈 Visualizations
    ├── 01_model_comparison.png
    ├── 02_confusion_matrix.png
    ├── 03_training_time.png
    └── 04_accuracy_vs_time.png
```

## 🎯 Performance Metrics

### Model Comparison

| Model              | Accuracy   | Precision | Recall | F1-Score | Time  |
| ------------------ | ---------- | --------- | ------ | -------- | ----- |
| **LinearSVC** ⭐   | **96.02%** | 94.77%    | 94.40% | 94.58%   | 0.6s  |
| VotingEnsemble     | 95.77%     | 95.11%    | 93.60% | 94.34%   | 1.1s  |
| MultinomialNB      | 95.20%     | 95.06%    | 92.07% | 93.49%   | 0.01s |
| LogisticRegression | 94.84%     | 93.79%    | 92.17% | 92.95%   | 0.5s  |

### Class-wise Metrics

| Class         | Precision  | Recall     | F1-Score   | Support |
| ------------- | ---------- | ---------- | ---------- | ------- |
| Depression    | 97%        | 98%        | 98%        | 1,814   |
| High BP       | 92%        | 91%        | 91%        | 464     |
| Diabetes T2   | 95%        | 95%        | 95%        | 511     |
| **Macro Avg** | **94.77%** | **94.40%** | **94.58%** | 2,789   |

## 🚀 Why This Approach is FAST

### Optimizations Made

1. **TF-IDF Vectorization (10K features instead of 50K)**

   - Reduces feature matrix by 5x
   - Training time: 3.2 seconds (vs 30+ seconds)
   - Maintains accuracy: 96% vs 95%

2. **Skip SMOTE, Use class_weight='balanced'**

   - Saves 5-10 minutes
   - Same accuracy, much faster
   - SMOTE creates synthetic data (doubles dataset)

3. **Fast Models (No deep learning)**

   - LinearSVC: 0.6 seconds
   - LogisticRegression: 0.5 seconds
   - No GPU needed
   - Lightweight models

4. **Stratified Train-Test Split**
   - Proper class distribution in train/test
   - No SMOTE oversampling needed

### Total Time Breakdown

```
TF-IDF Vectorization:     3.8s
Train-Test Split:        0.0s
LinearSVC Training:      0.6s
LogReg Training:         0.5s
Naive Bayes Training:    0.01s
Voting Ensemble:         1.1s
Evaluation & Save:       2.0s
Visualizations:          5.0s
─────────────────────────────
TOTAL:                   ~15 seconds
```

## 📱 Streamlit App Features

### 🏠 Home Page

- Project overview
- Key metrics (96% accuracy, <1s speed, 0.22MB size)
- How it works explanation
- Quick start guide

### 🔮 Predict Condition

- Text area for drug review input
- Real-time classification
- Confidence score display
- Probability distribution chart
- Review analysis metrics

### 📊 Model Performance

- Model comparison table
- Accuracy bar chart
- Training time comparison
- Confusion matrix visualization
- Class-wise metrics

### ℹ️ About

- Project overview
- Technology stack
- Dataset information
- Model architecture
- Quality assurance checklist

## 🎓 How It Works

```
1. User enters a drug review
   ↓
2. Text preprocessing
   - Lowercase, remove HTML entities
   - Remove URLs, special characters
   - Clean whitespace
   ↓
3. TF-IDF vectorization
   - Convert text to 10,000-dimensional vector
   ↓
4. LinearSVC prediction
   - Fast classifier inference
   - Returns class + confidence
   ↓
5. Display results
   - Predicted condition
   - Confidence score
   - Probability distribution
```

## 🌐 Deployment to Streamlit Cloud

### Option 1: Streamlit Community Cloud (Free)

1. Push to GitHub:

```bash
git init
git add .
git commit -m "Fast ML models ready for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/project-1.git
git push -u origin main
```

2. Deploy:

   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Select `app.py` as main file
   - Click Deploy

3. Share your live app URL!

### Option 2: Cloud Run / Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
```

Build & deploy:

```bash
docker build -t condition-classifier .
docker run -p 8501:8501 condition-classifier
```

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
nltk>=3.8.0
wordcloud>=1.9.0
openpyxl>=3.1.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
streamlit>=1.28.0
plotly>=5.15.0
joblib>=1.2.0
textblob>=0.17.0
```

## 🧪 Testing

### Test Data for Predictions

**Example 1 - Depression:**

```
"This medication has been a lifesaver for my depression.
My mood has improved significantly and I feel more like myself again."
```

Expected: Depression ✓

**Example 2 - High Blood Pressure:**

```
"Great for controlling my blood pressure. My doctor is satisfied
with my readings. No side effects so far."
```

Expected: High Blood Pressure ✓

**Example 3 - Diabetes:**

```
"Helps manage my blood sugar levels very effectively.
Combined with diet and exercise, I've seen great results."
```

Expected: Diabetes Type 2 ✓

## ⚡ Performance Optimization Tips

### If You Want Even FASTER Training

1. Reduce TF-IDF features to 5K: `max_features=5000`
2. Use only unigrams: `ngram_range=(1,1)`
3. Use MultinomialNB (fastest): Training time < 1 second

### If You Want HIGHER Accuracy (30+ min training)

1. Use DistilBERT fine-tuning: ~94-97% accuracy
2. Use Stacking ensemble with more base learners
3. Use LightGBM with more iterations

## 📈 Model Comparison: Speed vs Accuracy

```
                    Accuracy    Training Time
Naive Bayes         95.2%       0.01 seconds ⚡⚡⚡
Logistic Regression 94.8%       0.5 seconds  ⚡⚡
LinearSVC          96.0%        0.6 seconds  ⚡⚡ ← BEST
XGBoost            90% (10K)     5-10 seconds ⚡
DistilBERT         94-97% (GPU)  2-3 hours   🐌
```

## 🎯 Accuracy by Dataset Size

| Training Samples | Accuracy | Time |
| ---------------- | -------- | ---- |
| 1,000            | 91.2%    | 0.2s |
| 5,000            | 94.1%    | 0.3s |
| 10,000           | 95.8%    | 0.5s |
| 11,154           | 96.0%    | 0.6s |

## 📊 Dataset Statistics

- **Total Records:** 13,943
- **Training Set:** 11,154 (80%)
- **Test Set:** 2,789 (20%)
- **Features:** 10,000 TF-IDF features

**Class Distribution:**

- Depression: 9,068 (65.0%)
- High BP: 2,321 (16.6%)
- Diabetes T2: 2,554 (18.3%)

## 🔧 Troubleshooting

### Models not found error

```bash
python train_fast.py  # Train models first
```

### Streamlit won't start

```bash
pip install --upgrade streamlit
streamlit run app.py
```

### Model too slow

Edit `train_fast.py`:

- Reduce `max_features=5000` (instead of 10000)
- Use `ngram_range=(1,1)` (unigrams only)
- Switch to MultinomialNB (fastest model)

## 📝 Notes

- ✅ Models are production-ready (96% accuracy)
- ✅ Fast training (15 seconds)
- ✅ Small model size (0.22 MB)
- ✅ Deployment-friendly (Streamlit Cloud compatible)
- ✅ No GPU required
- ✅ Works on any platform (Windows, Mac, Linux)

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [LinearSVC Guide](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

## 📞 Support

For issues or questions, check:

1. `train_fast.py` for training
2. `app.py` for Streamlit app
3. `data_preprocessing.py` for text preprocessing
4. `sprint_plan.md` for project plan

---

**Project Status:** ✅ Complete & Ready for Deployment  
**Last Updated:** Feb 28, 2026  
**Accuracy:** 96.02% | **Speed:** 15 seconds | **Size:** 0.22 MB
