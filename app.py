"""
🚀 Patient Condition Classifier - Streamlit Web App
Using optimized fast ML models trained in <15 seconds
Deployed for Streamlit Cloud (small model, fast inference)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from html import unescape

# Page config
st.set_page_config(
    page_title="🔮 Condition Classifier",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    .big-font {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS & PREPROCESSING FUNCTION
# ============================================================

@st.cache_resource
def load_models():
    """Load pre-trained model and vectorizer"""
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    condition_names = joblib.load('models/condition_labels.pkl')
    return model, vectorizer, condition_names

def preprocess_review(text):
    """
    Clean and preprocess review text
    Mirrors the preprocessing used in training
    """
    # Convert to lowercase
    text = text.lower()
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load models
try:
    model, vectorizer, condition_names = load_models()
except FileNotFoundError:
    st.error("❌ Models not found! Please train the model first using `python train_fast.py`")
    st.stop()

# ============================================================
# SIDEBAR - NAVIGATION & INFO
# ============================================================
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select Page:", [
    "🏠 Home",
    "🔮 Predict Condition",
    "📊 Model Performance",
    "ℹ️ About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Model Information
- **Type:** LinearSVC Classifier
- **Features:** 10,000 TF-IDF features
- **Accuracy:** 96.02%
- **Training Time:** ~15 seconds
- **Model Size:** 0.22 MB
- **Status:** ✅ Production Ready
""")

# ============================================================
# PAGE 1: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown('<p class="big-font">🏥 Patient Condition Classifier</p>', unsafe_allow_html=True)
    st.markdown("""
    Classify patient medical conditions from drug review text using advanced machine learning.
    
    This system predicts whether a patient is being treated for one of three conditions:
    - **Depression**
    - **High Blood Pressure**
    - **Diabetes Type 2**
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <strong>📈 Accuracy</strong><br/>
            <span style="font-size: 28px; color: #28a745;">96.02%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <strong>⚡ Speed</strong><br/>
            <span style="font-size: 28px; color: #28a745;">< 1 second</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <strong>💾 Model Size</strong><br/>
            <span style="font-size: 28px; color: #28a745;">0.22 MB</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("✨ Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ✅ **Fast & Accurate**
        - 96% accuracy on test data
        - Inference in milliseconds
        
        ✅ **Lightweight**
        - Model size < 1MB
        - Deployable anywhere
        """)
    
    with col2:
        st.markdown("""
        ✅ **Production Ready**
        - Trained on 13,943+ reviews
        - 3-class multi-label classification
        - Optimized for accuracy
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 How It Works")
    st.markdown("""
    1. **Enter a drug review** describing patient experiences
    2. **AI analyzes** the text for medical indicators
    3. **Predicts** the most likely condition
    4. **Shows confidence** and probability distribution
    """)

# ============================================================
# PAGE 2: PREDICT CONDITION
# ============================================================
elif page == "🔮 Predict Condition":
    st.markdown('<p class="big-font">🔮 Predict Patient Condition</p>', unsafe_allow_html=True)
    
    st.markdown("Enter a drug review to classify the patient's condition:")
    
    # Text input
    review_text = st.text_area(
        "📝 Enter or paste drug review:",
        height=200,
        placeholder="Example: 'This medication has helped me manage my blood pressure very well. I take it twice daily and haven't experienced any significant side effects...'"
    )
    
    # Prediction button
    if st.button("🚀 Classify", use_container_width=True, type="primary"):
        if review_text.strip():
            with st.spinner("🔄 Analyzing review..."):
                # Preprocess
                cleaned_text = preprocess_review(review_text)
                
                # Vectorize
                X = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = model.predict(X)[0]
                distances = model.decision_function(X)[0]  # Get decision scores
                
                # Get confidence (normalize scores)
                confidence_scores = 1 / (1 + np.exp(-distances))  # sigmoid
                confidence_scores = confidence_scores / confidence_scores.sum()  # normalize
                
                predicted_condition = condition_names[prediction]
                confidence = confidence_scores[prediction]
            
            # Display results
            st.markdown("""
            <div class="success-box">
                <strong>✅ Prediction Complete</strong>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <strong>🎯 Predicted Condition</strong><br/>
                    <span style="font-size: 24px; color: #28a745; font-weight: bold;">{predicted_condition}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 20px; border-radius: 8px; border-left: 4px solid #1976d2;">
                    <strong>📊 Confidence</strong><br/>
                    <span style="font-size: 24px; color: #1976d2; font-weight: bold;">{confidence:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution
            st.subheader("📈 Probability Distribution")
            prob_data = {
                'Condition': condition_names,
                'Probability': confidence_scores
            }
            prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
            
            # Bar chart
            st.bar_chart(prob_df.set_index('Condition')['Probability'], height=300)
            
            # Detailed breakdown
            st.subheader("📋 Detailed Breakdown")
            for idx, row in prob_df.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.metric(row['Condition'], f"{row['Probability']:.1%}")
                with col2:
                    st.progress(row['Probability'])
            
            # Input text info
            st.subheader("📝 Review Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{len(review_text)} chars")
            with col2:
                st.metric("Cleaned Length", f"{len(cleaned_text)} chars")
            with col3:
                st.metric("Word Count", len(cleaned_text.split()))
        
        else:
            st.warning("⚠️ Please enter a review text!")

# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":
    st.markdown('<p class="big-font">📊 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Load comparison results if available
    try:
        results_df = pd.read_csv('models/model_comparison.csv')
        
        st.subheader("🏆 Model Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            accuracy_chart = results_df.sort_values('Accuracy', ascending=True)
            st.bar_chart(accuracy_chart.set_index('Model')['Accuracy'])
        
        with col2:
            st.subheader("Training Time (seconds)")
            time_chart = results_df.sort_values('Training_Time_s', ascending=True)
            st.bar_chart(time_chart.set_index('Model')['Training_Time_s'])
        
    except FileNotFoundError:
        st.info("📌 Model comparison data not available")
    
    # Display confusion matrix if available
    if os.path.exists('plots/02_confusion_matrix.png'):
        st.subheader("Confusion Matrix")
        from PIL import Image
        img = Image.open('plots/02_confusion_matrix.png')
        st.image(img, use_column_width=True)
    
    # Detailed metrics
    st.subheader("📈 Detailed Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("""
        <div class="info-box">
            <strong>Precision (Macro)</strong><br/>
            <span style="font-size: 22px; color: #0c5460;">94.77%</span><br/>
            <small>How accurate positive predictions are</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="info-box">
            <strong>Recall (Macro)</strong><br/>
            <span style="font-size: 22px; color: #0c5460;">94.40%</span><br/>
            <small>How many positives were found</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div class="info-box">
            <strong>F1-Score (Macro)</strong><br/>
            <span style="font-size: 22px; color: #0c5460;">94.58%</span><br/>
            <small>Balance of precision & recall</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Class-wise performance
    st.subheader("📊 Class-wise Performance")
    class_metrics = pd.DataFrame({
        'Class': condition_names,
        'Precision': [0.97, 0.92, 0.95],
        'Recall': [0.98, 0.91, 0.95],
        'F1-Score': [0.98, 0.91, 0.95],
        'Support': [1814, 464, 511]
    })
    st.dataframe(class_metrics, use_container_width=True)

# ============================================================
# PAGE 4: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown('<p class="big-font">ℹ️ About This Project</p>', unsafe_allow_html=True)
    
    st.subheader("📚 Project Overview")
    st.markdown("""
    **Patient Condition Classification Using Drug Reviews (P642)**
    
    This project uses natural language processing (NLP) and machine learning to automatically
    classify patient medical conditions based on drug review text. It's trained on 13,943+ 
    real drug reviews and achieves 96% accuracy.
    """)
    
    st.subheader("🛠️ Technology Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Data Processing**
        - Pandas
        - NumPy
        - NLTK
        """)
    with col2:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn
        - LinearSVC Classifier
        - TF-IDF Vectorizer
        """)
    with col3:
        st.markdown("""
        **Deployment**
        - Streamlit
        - Joblib
        - Python 3.8+
        """)
    
    st.subheader("📊 Dataset Information")
    dataset_info = pd.DataFrame({
        'Metric': ['Total Reviews', 'Depression', 'High BP', 'Diabetes T2', 'Features (TF-IDF)', 'Training Time'],
        'Value': ['13,943', '9,068 (65.0%)', '2,321 (16.6%)', '2,554 (18.3%)', '10,000', '15 seconds']
    })
    st.dataframe(dataset_info, use_container_width=True, hide_index=True)
    
    st.subheader("🎯 Model Architecture")
    st.markdown("""
    ```
    Input Review Text
         ↓
    Text Preprocessing (cleaning, tokenization)
         ↓
    TF-IDF Vectorization (10,000 features)
         ↓
    LinearSVC Classifier
         ↓
    Prediction: [Condition, Confidence Score]
    ```
    """)
    
    st.subheader("✅ Quality Assurance")
    st.markdown("""
    - ✅ Trained on 11,154 samples
    - ✅ Tested on 2,789 samples
    - ✅ 96.02% accuracy
    - ✅ Cross-validated
    - ✅ Production-ready
    """)
    
    st.subheader("📖 How to Use")
    st.markdown("""
    1. Go to **🔮 Predict Condition** tab
    2. Enter or paste a drug review
    3. Click **Classify** button
    4. View prediction and confidence scores
    5. Check detailed probability breakdown
    """)
    
    st.subheader("📝 Example Reviews")
    with st.expander("Click to see example reviews"):
        st.markdown("""
        **Depression Example:**
        *"This medication has been a lifesaver for managing my depression symptoms. 
        I've noticed significant improvement in my mood and overall quality of life."*
        
        **High Blood Pressure Example:**
        *"Great for controlling my blood pressure. My doctor said my readings are now 
        well-regulated. No major side effects so far."*
        
        **Diabetes Example:**
        *"Helps manage my blood sugar levels effectively. Combined with diet and exercise, 
        I've seen great results in my glucose monitoring."*
        """)
    
    st.markdown("---")
    st.markdown("""
    **Project Code:** P642  
    **Status:** ✅ Production Ready  
    **Last Updated:** Feb 28, 2026
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>🏥 Patient Condition Classifier | Powered by Machine Learning | 
    <strong>96% Accuracy</strong> | <strong>< 1s Inference</strong></small>
</div>
""", unsafe_allow_html=True)
