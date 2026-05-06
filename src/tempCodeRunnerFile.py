import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# ---------------------------
# LOAD DATA
# ---------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

data = data[["text", "label"]]

# ---------------------------
# CLEAN
# ---------------------------
data["text"] = data["text"].apply(clean_text)

# ---------------------------
# SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ---------------------------
# TF-IDF (IMPROVED)
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# MODEL (choose one)
# ---------------------------

# Option 1: Logistic Regression
model = LogisticRegression(max_iter=1000)

# Option 2: Naive Bayes (try this too)
# model = MultinomialNB()

model.fit(X_train_vec, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# SAVE
# ---------------------------
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nSaved successfully.")