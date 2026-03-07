from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Training Text-Based Stress Detection Model with Pipeline...")

# Training data: stress and non-stress examples
stress_texts = [
    "i am so stressed", "i feel anxious", "i am overwhelmed",
    "i have too much work", "i am worried", "i cannot sleep",
    "i feel depressed", "everything is terrible", "i am panicking",
    "i have anxiety attacks", "i feel helpless", "nothing works",
    "i am exhausted", "i feel alone", "i'm having a breakdown",
    "my heart is racing", "i cannot focus", "i'm so tired",
    "everything hurts", "i don't know what to do"
]

non_stress_texts = [
    "i am feeling great", "i am happy", "i feel peaceful",
    "everything is wonderful", "i love my life", "i am relaxed",
    "i feel calm", "i am confident", "life is good",
    "i feel motivated", "i am excited", "i feel grateful",
    "i am energized", "i feel loved", "i'm doing well",
    "i feel safe", "i can focus", "i'm refreshed",
    "everything is fine", "i feel positive"
]

# Combine data
texts = stress_texts + non_stress_texts
labels = [1] * len(stress_texts) + [0] * len(non_stress_texts)

# Create Pipeline: TfidfVectorizer + LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')),
    ('logistic', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

# Train
pipeline.fit(texts, labels)

# Evaluate
train_acc = pipeline.score(texts, labels)
print(f"Training Accuracy: {train_acc*100:.2f}%")

# Test on examples
test_examples = [
    "i am sad",
    "I'm feeling happy",
    "I have anxiety",
    "Life is good"
]

print("\nTest predictions:")
for text in test_examples:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    stress_prob = proba[1] * 100
    result = "STRESS" if pred == 1 else "NON-STRESS"
    print(f"  '{text}' → {result} ({stress_prob:.1f}% confidence)")

# Save model
joblib.dump(pipeline, 'stresslevel_text_model.pkl')
print("\n✅ Text model saved: stresslevel_text_model.pkl")
