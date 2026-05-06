import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

import pickle
import os

# ---------------------------
# SETUP
# ---------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------------------
# CLEAN TEXT FUNCTION
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    if len(words) == 0:
        return "empty"

    return " ".join(words)

# ---------------------------
# LOAD DATA
# ---------------------------
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

# merge + shuffle
data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# use only needed columns
data = data[["text", "label"]]

# ---------------------------
# CLEAN DATA
# ---------------------------
data["text"] = data["text"].apply(clean_text)

# ---------------------------
# CHECK DATA BALANCE
# ---------------------------
print("Label distribution:")
print(data["label"].value_counts())

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# ---------------------------
# TF-IDF VECTORIZER
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# MODEL (NAIVE BAYES)
# ---------------------------
model = MultinomialNB()

model.fit(X_train_vec, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully.")