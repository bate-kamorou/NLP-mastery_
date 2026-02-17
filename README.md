# 🎬 IMDb Movie Review Sentiment AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://movies-review-sentiment-analysis-1.streamlit.app/)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.94%25-green.svg)]

An end-to-end NLP solution that classifies movie reviews into **Positive** or **Negative** sentiments with high precision. Trained on 50,000 reviews and deployed as a live web application.

🚀 **Live App:** [View Sentiment AI Dashboard](https://movies-review-sentiment-analysis-1.streamlit.app/)

---

## 🛠️ Project Architecture

This project evolved from a baseline Naive Bayes model (84%) to a production-ready Logistic Regression pipeline, achieving a final accuracy of **88.94%**.

### 1. Data Processing Layer

To handle the "noise" inherent in web-scraped IMDb reviews, a custom `TextCleaner` class was developed:

* **HTML Stripping**: Used `BeautifulSoup` to remove residual tags.
* **Normalization**: Implemented Lemmatization via NLTK's `WordNetLemmatizer` to reduce words to their linguistic roots.
* **Noise Reduction**: Leveraged Regex to eliminate non-alphabetic characters.

### 2. Feature Engineering

Instead of simple word counts, the model utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)**:

* **Bigram Integration**: By setting $ngram\_range=(1, 2)$, the model captures context (e.g., "not good" or "fall asleep"), which was critical for passing complex sarcasm tests.
* **Dimensionality**: Optimized at 5,000 max features to balance model complexity with performance.

### 3. Model Development

* **The Pipeline**: Integrated the vectorizer and classifier into a Scikit-Learn `Pipeline`. This ensures that the same transformations applied to training data are perfectly replicated during real-time inference.
* **Optimization**: Used `tempfile.mkdtemp` for memory caching during training to handle the 50,000-review dataset efficiently on limited hardware.

### 4. Evaluation

Achieved an impressive [![Accuracy](https://img.shields.io/badge/Accuracy-88.94%25-green.svg)] on the test set, with a precision of 0.89 and recall of 0.88, demonstrating a well-balanced model.

### 5. Deployment & DevOps

* **Web Framework**: Streamlit for the interactive UI.
* **Model Versioning**: Git LFS (Large File Storage) was utilized to track and push the trained `.joblib` model artifacts (>100MB) to GitHub safely.
* **Cloud Hosting**: Deployed on Streamlit Cloud with a modular dependency structure.

---

## 🚀 How to Run Locally

1. **Clone the repo:**

   ```bash
   git clone [ https://github.com/bate-kamorou/NLP-mastery_.git]( https://github.com/bate-kamorou/NLP-mastery_.git)
    cd IMDb-Sentiment-Analysis-Pipeline
    ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Lunch the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

4. **Access the app:**  
   Open [http://localhost:8501](http://localhost:8501) in your browser
