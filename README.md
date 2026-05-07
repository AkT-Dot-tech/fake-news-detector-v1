# 🔍 Fake News Detector

A machine learning project that classifies news articles as **Real** or **Fake**
using NLP techniques and Logistic Regression.

---

## 📌 What It Does

- Takes a news article as input
- Cleans and preprocesses the text (removes stopwords, URLs, source bias)
- Converts text to numbers using TF-IDF vectorization
- Predicts whether the article is Real or Fake using Logistic Regression
- Shows confidence percentage for each prediction
- Runs as a web app built with Streamlit

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Accuracy | 98.85% |
| Features | 10,000 TF-IDF features |
| Training Data | 44,898 news articles |
| Dataset | Fake and Real News Dataset (Kaggle) |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data loading and manipulation |
| NLTK | Stopword removal |
| Scikit-learn | TF-IDF + Logistic Regression |
| Streamlit | Web app interface |
| Pickle | Saving trained model |
---

## 📂 Project Structure

fake-news-detector/
│
├── data/
│   ├── Fake.csv          # Fake news articles
│   └── True.csv          # Real news articles
│
├── src/
│   └── train_model.py    # Training pipeline
│
├── app/
│   └── streamlit_app.py  # Streamlit web app
│
├── model/
│   ├── model.pkl         # Saved trained model
│   └── vectorizer.pkl    # Saved TF-IDF vectorizer
│
└── README.md
---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn nltk streamlit
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
and place `Fake.csv` and `True.csv` inside the `data/` folder.

### 4. Train the model
```bash
python src/train_model.py
```

### 5. Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## ⚠️ Known Limitations

- Model works best on full news articles (2+ paragraphs), not short phrases
- Trained on 2016–2017 news data, so very recent events may not be handled well
- TF-IDF looks at word patterns, not meaning — a future improvement would be using BERT

---
## 🤖 About the Model

This project is a classical Machine Learning approach to fake news detection — it does **not**
use any neural network or transformer architecture.

The model works by learning **text patterns** from thousands of labelled news articles.
It uses TF-IDF to convert words into numbers, and Logistic Regression to find patterns
that separate fake articles from real ones. It does not understand the *meaning* of text —
it recognises *which words and phrases* tend to appear in fake vs real news.

This is intentional. The goal of this version is to build a working, explainable baseline
using fundamental ML techniques before moving to more complex architectures.

**Future plan:** The next version will use **BERT** (Bidirectional Encoder Representations
from Transformers), which uses a mechanism called **self-attention** to understand the
meaning and context of words — not just their frequency. This will allow the model to
handle short texts, sarcasm, and out-of-domain news articles far more accurately.

---

## 👨‍💻 Author

**Akshat** — B.Tech CSE (AI/ML), IILM University, Greater Noida