# Sentiment Analysis Web Application

Welcome to the Sentiment Analysis Web App! This project lets you enter any text review and instantly see if it’s **Positive** or **Negative**—using both a Support Vector Machine (SVM) and a Neural Network (NN) model.

---

## 🚀 Features

- Dual Model Predictions: See results from both SVM and Neural Network models.
- Automatic Text Cleaning: Your input is cleaned and preprocessed for best results.
- Easy Web Interface: Just type and click—no coding needed!
- Reusable Models: Pre-trained models load instantly for fast predictions.

---

## 🗂️ Project Structure

```
.
├── app.py                  # Flask web app
├── data_analysis.ipynb     # Data exploration, preprocessing, model training
├── nn_model.h5             # Trained Neural Network (Keras)
├── svm_model.pkl           # Trained SVM (scikit-learn)
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── templates/
│   └── index.html          # Web interface template
└── Advanced_Data_Analysis.pdf # Project report/documentation
```

---

## 🧑‍💻 How It Works

1. **Preprocessing:**  
   - Removes HTML, URLs, mentions, hashtags, punctuation, and non-letters  
   - Converts to lowercase  
   - Tokenizes and removes stopwords (using NLTK)  
   - Transforms text with a TF-IDF vectorizer

2. **Model Training:**  
   - SVM: LinearSVC from scikit-learn  
   - Neural Network: Keras Sequential model

3. **Prediction:**  
   - Both models predict sentiment for your input  
   - Results are shown on the web page

---

## 🏁 Getting Started

1. **Install dependencies:**
   ```sh
   pip install flask numpy tensorflow scikit-learn nltk beautifulsoup4
   ```

2. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

3. **Run the app:**
   ```sh
   python app.py
   ```

4. **Open in your browser:**  
   [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📊 Model Performance

- SVM Accuracy: ~89.6%
- Neural Network Accuracy: See `data_analysis.ipynb` for details

---

## 📚 Files & Documentation

- Data analysis & training: `data_analysis.ipynb`
- Web app code: `app.py`
- Web interface: `templates/index.html`
- Project report: `Advanced_Data_Analysis.pdf`

---
