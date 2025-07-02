import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# === Load Cleaned Email Data ===
df = pd.read_csv("data/emails_clean.csv")

# === Labeling ===
df["label"] = df["message"].apply(lambda x: 1 if "unsubscribe" in x.lower() or "click here" in x.lower() else 0)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# === TF-IDF Vectorizer ===
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# === Train Model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# === Evaluate ===
y_pred = model.predict(X_test_tfidf)
print("\n📊 Email Classification Report:")
print(classification_report(y_test, y_pred))

# === Save Model & Vectorizer ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/email_baseline_model.joblib")
joblib.dump(tfidf, "model/email_tfidf_vectorizer.joblib")
print("\n✅ Email model and vectorizer saved to /model")
