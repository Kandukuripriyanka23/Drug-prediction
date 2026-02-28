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
# CUSTOM STYLING - Dark Mode Compatible
# ============================================================
st.markdown("""
<style>
    /* Main fonts and colors */
    .big-font {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Result boxes - High contrast for dark mode */
    .prediction-result {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 30px;
        border-radius: 12px;
        border: 3px solid #60a5fa;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .condition-label {
        color: #e0e7ff;
        font-size: 16px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .condition-name {
        color: #fbbf24;
        font-size: 36px;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .confidence-box {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        padding: 25px;
        border-radius: 12px;
        border: 3px solid #34d399;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .confidence-label {
        color: #d1fae5;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .confidence-value {
        color: #fbbf24;
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Probability distribution */
    .prob-header {
        color: #60a5fa;
        font-size: 20px;
        font-weight: bold;
        margin: 30px 0 20px 0;
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 10px;
    }
    
    .prob-item {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #60a5fa;
    }
    
    .prob-condition {
        color: #e0e7ff;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .prob-percentage {
        color: #fbbf24;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* Analysis section */
    .analysis-box {
        background-color: rgba(59, 130, 246, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #3b82f6;
        margin: 20px 0;
    }
    
    .analysis-title {
        color: #60a5fa;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 15px 0;
    }
    
    .metric-item {
        background-color: rgba(107, 114, 128, 0.2);
        padding: 15px;
        border-radius: 8px;
        flex: 1;
        margin: 0 5px;
        text-align: center;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .metric-value {
        color: #60a5fa;
        font-size: 24px;
        font-weight: bold;
        margin-top: 5px;
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
# PAGE 2: PREDICT CONDITION - NEW DARK MODE FRIENDLY DESIGN
# ============================================================
elif page == "🔮 Predict Condition":
    st.markdown('<p class="big-font">🔮 Predict Patient Condition</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #60a5fa;">
        <strong style="color: #60a5fa;">💡 Enter a drug review below to classify the patient's medical condition</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Text input with better styling - LARGER HEIGHT
    review_text = st.text_area(
        "📝 Drug Review:",
        height=350,
        placeholder="Example: 'This medication has helped me manage my blood pressure very well. I take it twice daily and haven't experienced any significant side effects...'",
        help="Paste or type the patient's drug review here"
    )
    
    st.markdown("")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        predict_button = st.button("🚀 CLASSIFY NOW", use_container_width=True, type="primary")
    
    if predict_button:
        if review_text.strip():
            with st.spinner("⏳ Analyzing review... This may take a few seconds"):
                # Preprocess
                cleaned_text = preprocess_review(review_text)
                
                # Vectorize
                X = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = model.predict(X)[0]
                distances = model.decision_function(X)[0]
                
                # Get confidence (normalize scores)
                confidence_scores = 1 / (1 + np.exp(-distances))
                confidence_scores = confidence_scores / confidence_scores.sum()
                
                predicted_condition = condition_names[prediction]
                confidence = confidence_scores[prediction]
            
            # ========== RESULT DISPLAY ==========
            st.markdown("")
            st.markdown("---")
            st.markdown("")
            
            # Success message
            st.markdown("""
            <div style="background-color: #10b981; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                <span style="color: #d1fae5; font-size: 18px; font-weight: bold;">✅ PREDICTION COMPLETE</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Main prediction result
            st.markdown(f"""
            <div class="prediction-result">
                <div class="condition-label">🎯 Predicted Condition</div>
                <div class="condition-name">{predicted_condition}</div>
                <div style="color: #93c5fd; font-size: 14px;">Classification Result</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Confidence score
            st.markdown(f"""
            <div class="confidence-box">
                <div class="confidence-label">📊 Confidence Score</div>
                <div class="confidence-value">{confidence:.1%}</div>
                <div style="color: #d1fae5; font-size: 14px;">Model Certainty Level</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Probability distribution
            st.markdown("""
            <div class="prob-header">📈 PROBABILITY DISTRIBUTION</div>
            """, unsafe_allow_html=True)
            
            prob_data = {
                'Condition': condition_names,
                'Probability': confidence_scores
            }
            prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
            
            # Display each probability
            for idx, row in prob_df.iterrows():
                prob_pct = row['Probability'] * 100
                condition = row['Condition']
                
                # Color coding based on probability
                if prob_pct >= 60:
                    color = "#10b981"  # Green
                    bg_color = "rgba(16, 185, 129, 0.1)"
                elif prob_pct >= 30:
                    color = "#f59e0b"  # Amber
                    bg_color = "rgba(245, 158, 11, 0.1)"
                else:
                    color = "#6366f1"  # Indigo
                    bg_color = "rgba(99, 102, 241, 0.1)"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #e0e7ff; font-weight: 600; font-size: 16px;">{condition}</span>
                        <span style="color: {color}; font-weight: bold; font-size: 18px;">{prob_pct:.1f}%</span>
                    </div>
                    <div style="background-color: rgba(107, 114, 128, 0.3); height: 8px; border-radius: 4px; margin-top: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, {color}, {color}aa); height: 100%; width: {prob_pct}%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Visual chart
            st.markdown("**Probability Chart:**")
            st.bar_chart(prob_df.set_index('Condition')['Probability'], height=300)
            
            st.markdown("")
            
            # Input analysis
            st.markdown("""
            <div class="analysis-box">
                <div class="analysis-title">📝 Review Analysis</div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Original Length",
                    value=f"{len(review_text)} chars",
                    delta=None
                )
            with col2:
                st.metric(
                    label="Cleaned Length",
                    value=f"{len(cleaned_text)} chars",
                    delta=None
                )
            with col3:
                st.metric(
                    label="Word Count",
                    value=len(cleaned_text.split()),
                    delta=None
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("")
            
            # Prediction details
            st.info(f"""
            **Prediction Details:**
            - Selected condition: **{predicted_condition}**
            - Confidence level: **{confidence:.1%}**
            - Analysis completed successfully
            """)
            
        else:
            st.warning("⚠️ Please enter a drug review to classify!", icon="⚠️")

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
