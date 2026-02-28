# 🚀 DEPLOYMENT GUIDE - Streamlit Cloud

## Quick Deployment (5 minutes)

Deploy your Patient Condition Classifier to Streamlit Cloud in just 5 minutes!

---

## 📋 Prerequisites

✅ GitHub account (create at [github.com](https://github.com) if needed)  
✅ Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))  
✅ Project files pushed to GitHub repository

---

## 🎯 Step 1: Prepare Your GitHub Repository

### Option A: First Time Setup

```bash
# Initialize git repository
cd d:\downloads\project-1
git init

# Add all files
git add .

# Create first commit
git commit -m "Patient Condition Classifier - Ready for deployment"

# Rename branch to main
git branch -m main

# Add remote repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/project-1.git

# Push to GitHub
git push -u origin main
```

### Option B: Already Have Git Repository

```bash
# Navigate to project
cd d:\downloads\project-1

# Commit changes
git add .
git commit -m "Patient Condition Classifier - Updated for submission"

# Push to GitHub
git push origin main
```

---

## ✅ Step 2: Verify Files on GitHub

Ensure your GitHub repository has these files:

```
project-1/
├── app.py                    ✅ Main Streamlit application
├── train_fast.py             ✅ Model training script
├── data_preprocessing.py      ✅ Text preprocessing
├── requirements.txt           ✅ Python dependencies
├── README.md                  ✅ Documentation
│
├── models/
│   ├── best_model.pkl         ✅ Trained classifier
│   ├── tfidf_vectorizer.pkl   ✅ Text vectorizer
│   ├── condition_labels.pkl   ✅ Class labels
│   └── model_comparison.csv   ✅ Results table
│
└── data/
    └── drugsCom_cleaned.csv   ✅ Dataset
```

---

## 🌐 Step 3: Deploy to Streamlit Cloud

### Step 3.1: Go to Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"** button

### Step 3.2: Configure Deployment

Fill in the deployment form:

| Field              | Value                     |
| ------------------ | ------------------------- |
| **Repository**     | `YOUR_USERNAME/project-1` |
| **Branch**         | `main`                    |
| **Main file path** | `app.py`                  |

### Step 3.3: Deploy

Click **"Deploy"** button

**Expected:** App will deploy in 1-3 minutes

---

## 🎉 Step 4: Your App is Live!

After deployment completes, you'll get a public URL:

```
https://share.streamlit.io/YOUR_USERNAME/project-1/app.py
```

**Share this link with:**

- Team members
- Professors/Instructors
- Stakeholders
- Anyone who wants to test your app!

---

## 📱 Test Your Deployed App

### On the Streamlit Cloud App:

1. **Go to Home page** → View project overview
2. **Go to Predict page** → Enter a drug review
3. **Click "Classify"** → See prediction and confidence
4. **Go to Performance** → View model metrics
5. **Go to About** → Read project details

---

## ⚙️ Troubleshooting Deployment

### Issue: "Module not found" error

**Solution:** Make sure `requirements.txt` has all dependencies:

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

Then redeploy:

1. Go to your Streamlit Cloud app
2. Click **"Rerun"** or **"Settings"** → **"Reboot app"**

### Issue: "Models not found" error

**Solution:** Ensure `models/` folder is pushed to GitHub:

```bash
# Check if models folder is in git
git status

# Add models folder
git add models/
git commit -m "Add trained models"
git push origin main

# Redeploy on Streamlit Cloud
```

### Issue: App takes too long to load

**Solution:** This is normal on first load. Wait 30-60 seconds.

After first load, subsequent requests will be fast (< 1 second).

### Issue: Can't deploy

**Solution:** Check error message:

- Missing files? Add them: `git add .`
- Requirements outdated? Update: `pip freeze > requirements.txt`
- Python version? Make sure using 3.8+

---

## 🔄 Update Your App After Deployment

### To make changes to your app:

```bash
# Make changes to files (e.g., app.py, train_fast.py)

# Commit changes
git add .
git commit -m "Updated app features"

# Push to GitHub
git push origin main

# Streamlit Cloud will auto-redeploy within 1-2 minutes
```

**Note:** Streamlit Cloud auto-detects changes and redeploys automatically!

---

## 📊 Monitor Your Deployed App

### View App Metrics:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click your app in the list
3. View:
   - App status (deployed/running)
   - Number of users
   - Last deployment time
   - View logs

### Check Logs for Errors:

1. Click your app
2. Click **"Settings"** (gear icon)
3. Click **"View logs"**
4. Check for any error messages

---

## 🔐 Security Considerations

### Your App is Safe Because:

✅ No sensitive data in code  
✅ Models are static (no real PII)  
✅ HTTPS enforced automatically  
✅ Streamlit Cloud manages security  
✅ No database credentials needed

---

## 💾 Backup Your Models

Your trained models are stored in:

```
d:\downloads\project-1\models\
├── best_model.pkl
├── tfidf_vectorizer.pkl
├── condition_labels.pkl
└── model_comparison.csv
```

**Keep these files safe!** They're what powers your app.

---

## 🚀 Advanced: Retraining Models

If you want to retrain models with new data:

```bash
# 1. Update your dataset (if needed)
# Place new data in data/drugsCom_cleaned.csv

# 2. Retrain model
python train_fast.py

# 3. Commit and push
git add models/
git commit -m "Retrained models with new data"
git push origin main

# 4. Streamlit Cloud automatically redeploys
```

---

## 📋 Deployment Checklist

Before submitting, verify:

- [ ] GitHub repository created
- [ ] All files pushed to GitHub
- [ ] requirements.txt has all packages
- [ ] models/ folder is in repository
- [ ] App deployed to Streamlit Cloud
- [ ] Deployment URL works
- [ ] Home page displays correctly
- [ ] Prediction page works
- [ ] Performance page shows metrics
- [ ] Can make predictions
- [ ] Confidence scores display
- [ ] Charts and visualizations render
- [ ] No errors in app logs
- [ ] App responds in < 2 seconds
- [ ] Ready for submission ✅

---

## 📝 Submission Information

### For Submission, Provide:

1. **Live App URL:**

   ```
   https://share.streamlit.io/YOUR_USERNAME/project-1/app.py
   ```

2. **GitHub Repository:**

   ```
   https://github.com/YOUR_USERNAME/project-1
   ```

3. **Model Performance:**

   - Accuracy: 96.02%
   - Test samples: 2,789
   - Classes: 3 (Depression, High BP, Diabetes T2)

4. **Key Features:**
   - Real-time prediction from drug reviews
   - Confidence scores with probability distribution
   - Interactive visualizations
   - Model performance metrics
   - Fully automated pipeline

---

## 🎯 What Your Deployed App Does

### 🏠 Home Page

- Shows project overview
- Displays key statistics
- Explains how the system works

### 🔮 Predict Page

- Accept drug review as input
- Classify into one of 3 conditions
- Show prediction confidence
- Display probability distribution
- Provide review analysis

### 📊 Performance Page

- Model accuracy metrics
- Confusion matrix
- Class-wise performance
- Comparison table

### ℹ️ About Page

- Project details
- Technology stack
- Dataset information
- Usage examples

---

## ✅ Success Indicators

Your deployment is successful when:

✅ App URL is accessible  
✅ All pages load without errors  
✅ Predictions work correctly  
✅ Visualizations display properly  
✅ Metrics are accurate  
✅ Response time < 2 seconds  
✅ Can share URL with others

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Streamlit Logs:**

   - Go to app settings → View logs

2. **Review Documentation:**

   - README.md for project info
   - app.py for code explanation

3. **Test Locally First:**

   - Run: `streamlit run app.py`
   - Verify app works before redeploying

4. **Streamlit Support:**
   - [Streamlit Docs](https://docs.streamlit.io)
   - [Community Forum](https://discuss.streamlit.io)

---

## 📞 Quick Reference

| Task           | Command                                                |
| -------------- | ------------------------------------------------------ |
| Push to GitHub | `git push origin main`                                 |
| View logs      | Streamlit Cloud → Settings → View logs                 |
| Redeploy       | Click "Rerun" on Streamlit Cloud                       |
| Update app     | Edit files → `git push` → Auto redeploys               |
| Check status   | Go to [share.streamlit.io](https://share.streamlit.io) |

---

## 🎉 You're Ready!

Your Patient Condition Classifier is ready for deployment!

**Next steps:**

1. Push code to GitHub
2. Deploy on Streamlit Cloud
3. Share your live URL
4. Submit for evaluation

**Live App URL (after deployment):**

```
https://share.streamlit.io/YOUR_USERNAME/project-1/app.py
```

---

**Status:** ✅ Ready for Deployment  
**Platform:** Streamlit Cloud (Free)  
**Deployment Time:** 5 minutes  
**Maintenance:** Automatic (auto-redeploys on push)
