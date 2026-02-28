# 🏥 Patient Condition Classification Using Drug Reviews

## 📌 Project Overview

This project classifies patient medical conditions from drug reviews using machine learning. It predicts three conditions:

- **Depression**
- **High Blood Pressure**
- **Diabetes Type 2**

## ✨ Key Features

| Feature        | Details              |
| -------------- | -------------------- |
| **Accuracy**   | 96.02%               |
| **Model Type** | LinearSVC Classifier |
| **Dataset**    | 13,943 drug reviews  |
| **Classes**    | 3 conditions         |
| **Status**     | ✅ Production Ready  |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_fast.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

Then open: `http://localhost:8501`

## 📊 Project Structure

```
project-1/
├── app.py                     # Streamlit web application
├── train_fast.py              # Model training script
├── data_preprocessing.py       # Text preprocessing
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── models/                     # Trained models
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── condition_labels.pkl
│   └── model_comparison.csv
│
└── data/                       # Dataset
    └── drugsCom_cleaned.csv
```

## 🎯 Model Performance

### Accuracy Metrics

| Metric                | Value  |
| --------------------- | ------ |
| **Overall Accuracy**  | 96.02% |
| **Precision (Macro)** | 94.77% |
| **Recall (Macro)**    | 94.40% |
| **F1-Score (Macro)**  | 94.58% |

### Class-wise Performance

| Class       | Precision | Recall | F1-Score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| Depression  | 97%       | 98%    | 98%      | 1,814   |
| High BP     | 92%       | 91%    | 91%      | 464     |
| Diabetes T2 | 95%       | 95%    | 95%      | 511     |

## 🌐 Streamlit App

The app has 4 interactive pages:

### 🏠 Home Page

- Project overview
- Key statistics
- How the system works

### 🔮 Predict Condition

- Enter drug review text
- Get classification result
- View confidence scores
- See probability distribution

### 📊 Model Performance

- Accuracy metrics
- Confusion matrix
- Class-wise performance
- Model comparison table

### ℹ️ About

- Project details
- Technology stack
- Dataset information
- Usage examples

## 📦 Dataset

- **Total Samples:** 13,943
- **Training Set:** 11,154 (80%)
- **Test Set:** 2,789 (20%)
- **Features:** TF-IDF vectorization (10,000 features)

### Class Distribution

- Depression: 9,068 (65.0%)
- High BP: 2,321 (16.6%)
- Diabetes T2: 2,554 (18.3%)

## 🛠️ Technology Stack

| Component              | Technology                  |
| ---------------------- | --------------------------- |
| **ML Framework**       | scikit-learn                |
| **Classifier**         | LinearSVC                   |
| **Feature Extraction** | TF-IDF Vectorizer           |
| **Web App**            | Streamlit                   |
| **Text Processing**    | NLTK                        |
| **Visualization**      | Matplotlib, Seaborn, Plotly |
| **Data Processing**    | Pandas, NumPy               |

## 📝 How It Works

1. **Input:** User enters a drug review
2. **Preprocessing:** Text is cleaned and normalized
3. **Vectorization:** Review converted to TF-IDF features
4. **Prediction:** LinearSVC classifier predicts condition
5. **Output:** Condition and confidence score displayed

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as main file
5. Click Deploy

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 🧪 Testing

### Test Locally

```bash
streamlit run app.py
```

### Try These Examples

**Depression:**

```
This medication has been a lifesaver for managing my depression symptoms.
I've noticed significant improvement in my mood and overall quality of life.
```

**High Blood Pressure:**

```
Great for controlling my blood pressure. My doctor said my readings are now
well-regulated. No major side effects so far.
```

**Diabetes Type 2:**

```
Helps manage my blood sugar levels effectively. Combined with diet and exercise,
I've seen great results in my glucose monitoring.
```

## 📋 Requirements

```
pandas
numpy
scikit-learn
nltk
joblib
streamlit
matplotlib
seaborn
plotly
textblob
```

## 📖 Files Description

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `app.py`                      | Streamlit web application with 4 pages |
| `train_fast.py`               | Model training and evaluation script   |
| `data_preprocessing.py`       | Text cleaning and feature engineering  |
| `requirements.txt`            | Python package dependencies            |
| `models/best_model.pkl`       | Trained classifier model               |
| `models/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer               |
| `data/drugsCom_cleaned.csv`   | Preprocessed dataset                   |

## 🎓 Model Details

### LinearSVC Classifier

- **Type:** Support Vector Machine (Linear Kernel)
- **Regularization:** C=1.0
- **Max Iterations:** 2000
- **Class Weight:** Balanced (handles class imbalance)
- **Training Samples:** 11,154
- **Test Samples:** 2,789

### TF-IDF Vectorizer

- **Max Features:** 10,000
- **N-gram Range:** (1, 2) - unigrams and bigrams
- **Min Document Frequency:** 5
- **Max Document Frequency:** 0.90
- **Sublinear TF Scaling:** Enabled

## 🚀 Performance

- **Prediction Speed:** < 100ms per request
- **Model Size:** 0.22 MB
- **Memory Usage:** ~50 MB with Streamlit
- **Supported Conditions:** 3

## 📞 Support

For issues or questions:

1. Check `DEPLOYMENT_GUIDE.md` for deployment help
2. Review `app.py` code comments
3. Check Streamlit documentation

## 🎉 Summary

This project successfully classifies patient medical conditions from drug reviews with 96% accuracy using a LinearSVC classifier and TF-IDF vectorization. The model is deployed as an interactive Streamlit web application ready for production use.

**Ready to deploy and submit!** ✅

---

**Last Updated:** February 28, 2026  
**Status:** ✅ Production Ready  
**Accuracy:** 96.02%  
**License:** MIT
