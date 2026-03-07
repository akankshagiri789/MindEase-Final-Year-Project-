import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Training Physiological Stress Detection Model...")

# Generate synthetic physiological data
np.random.seed(42)
n_samples = 500

# Low stress features
low_stress_samples = 250
low_sleep = np.random.normal(7.5, 1, low_stress_samples)
low_bp = np.random.normal(120, 8, low_stress_samples)
low_respiration = np.random.normal(14, 2, low_stress_samples)
low_hr = np.random.normal(70, 5, low_stress_samples)

# High stress features
high_stress_samples = 250
high_sleep = np.random.normal(4.5, 1, high_stress_samples)
high_bp = np.random.normal(145, 10, high_stress_samples)
high_respiration = np.random.normal(20, 2, high_stress_samples)
high_hr = np.random.normal(95, 8, high_stress_samples)

# Combine data
X = pd.DataFrame({
    'sleeping_hours': np.concatenate([low_sleep, high_sleep]),
    'blood_pressure': np.concatenate([low_bp, high_bp]),
    'respiration_rate': np.concatenate([low_respiration, high_respiration]),
    'heart_rate': np.concatenate([low_hr, high_hr])
})

y = np.concatenate([np.zeros(low_stress_samples), np.ones(high_stress_samples)])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Feature importance
importances = model.feature_importances_
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp*100:.2f}%")

# Save model and scaler
joblib.dump(model, 'stresslevel.pkl')
joblib.dump(scaler, 'stresslevel_scaler.pkl')
joblib.dump(X.columns.tolist(), 'stresslevel_features.pkl')

print("\n✅ Model saved: stresslevel.pkl")
print("✅ Scaler saved: stresslevel_scaler.pkl")
print("✅ Features saved: stresslevel_features.pkl")
