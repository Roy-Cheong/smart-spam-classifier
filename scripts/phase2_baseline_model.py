import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# === Load Cleaned Dataset ===
df = pd.read_csv("data/spam_clean.csv")

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# === Vectorize Text ===
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# === Train Logistic Regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# === Predict & Report ===
y_pred = model.predict(X_test_tfidf)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# === Save Model & Vectorizer ===
joblib.dump(model, "model/baseline_model.joblib")
joblib.dump(tfidf, "model/tfidf_vectorizer.joblib")
print("\n✅ Baseline model and vectorizer saved to /model")