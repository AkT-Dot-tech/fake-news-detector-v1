import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── TEXT CLEANING ─────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'\(reuters\)', '', text)       # remove Reuters tag
    text = re.sub(r'reuters', '', text)            # remove word 'reuters'
    text = re.sub(r'[^a-zA-Z]', ' ', text)         # keep only letters
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words) if words else 'empty'

# ── LOAD DATA ─────────────────────────────────────────────────────────
print("Step 1: Loading data...")
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")
fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data = data[["text", "label"]].dropna()
print(f"  Loaded: {len(data)} articles  |  Fake: {(data.label==0).sum()}  Real: {(data.label==1).sum()}")

# ── CLEAN TEXT ────────────────────────────────────────────────────────
print("Step 2: Cleaning text (Reuters bias removed)...")
data["text"] = data["text"].apply(clean_text)

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────
print("Step 3: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── TF-IDF ────────────────────────────────────────────────────────────
print("Step 4: Applying TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2               # ignore very rare words
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── TRAIN MODEL ───────────────────────────────────────────────────────
print("Step 5: Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ── EVALUATE ──────────────────────────────────────────────────────────
print("Step 6: Evaluating...")
y_pred   = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {accuracy * 100:.2f}%")
print("\n  Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# ── SAVE ──────────────────────────────────────────────────────────────
print("Step 7: Saving model...")
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("  Saved: model/model.pkl  +  model/vectorizer.pkl")
print("\nDone!")