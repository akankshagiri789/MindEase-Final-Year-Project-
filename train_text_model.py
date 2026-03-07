"""
Training script for text-based stress detection model.
Optimized for the Flask app's text-only inference path.
"""
import pickle
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Try to import joblib for better sklearn model saving
try:
    import joblib

    USE_JOBLIB = True
except ImportError:
    USE_JOBLIB = False
    print("joblib not available, using pickle instead.")

RANDOM_STATE = 42

print("Loading dataset...")
df = pd.read_csv("dreaddit-train.csv", encoding="ISO-8859-1")
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"Dataset shape after cleanup: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nVectorizing text (word + character features)...")
vectorizer = FeatureUnion(
    [
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                sublinear_tf=True,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                max_features=200000,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=2,
                max_df=0.99,
                max_features=300000,
            ),
        ),
    ]
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Combined text feature matrix shape: {X_train_vec.shape}")

print("Training classifier...")
model = LogisticRegression(
    C=10.0,
    solver="liblinear",
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_iter=5000,
)
model.fit(X_train_vec, y_train)

print("\nEvaluating model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Stress", "Stress"]))

print("\nSaving model artifacts...")
if USE_JOBLIB:
    joblib.dump(model, "stresslevel_text_model.pkl")
    joblib.dump(vectorizer, "stresslevel_text_vectorizer.pkl")
else:
    with open("stresslevel_text_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("stresslevel_text_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

print("Training complete. Saved:")
print("- stresslevel_text_model.pkl")
print("- stresslevel_text_vectorizer.pkl")
