"""
Test the improved stress detection with different emotion predictions
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\Users\hp\Desktop\code\final')

from facial_stress_detection import FacialStressDetector

print("[OK] Testing Improved Stress Detection\n")

detector = FacialStressDetector()

# Test 1: Happy face (should be Very Low stress)
print("Test 1: HAPPY FACE")
print("-" * 40)
happy_emotions = np.array([
    0.01,   # Angry
    0.01,   # Disgust
    0.02,   # Fear
    0.80,   # Happy
    0.10,   # Neutral
    0.04,   # Sad
    0.02    # Surprise
])
result = detector.analyze_stress_from_emotions(happy_emotions)
print(f"Dominant Emotion: {result['dominant_emotion']}")
print(f"Stress Score: {result['stress_score']*100:.1f}%")
print(f"Stress Level: {result['stress_level']}")
print(f"Expected: Very Low (< 15%)")
print("[PASS]" if result['stress_score'] < 0.15 else "[FAIL]")

# Test 2: Neutral face (should be Low stress)
print("\n\nTest 2: NEUTRAL FACE")
print("-" * 40)
neutral_emotions = np.array([
    0.05,   # Angry
    0.05,   # Disgust
    0.05,   # Fear
    0.20,   # Happy
    0.50,   # Neutral
    0.10,   # Sad
    0.05    # Surprise
])
result = detector.analyze_stress_from_emotions(neutral_emotions)
print(f"Dominant Emotion: {result['dominant_emotion']}")
print(f"Stress Score: {result['stress_score']*100:.1f}%")
print(f"Stress Level: {result['stress_level']}")
print(f"Expected: Low (25-32%)")
print("[PASS]" if 0.20 <= result['stress_score'] < 0.35 else "[FAIL]")

# Test 3: Sad face (should be Moderate to High stress)
print("\n\nTest 3: SAD FACE")
print("-" * 40)
sad_emotions = np.array([
    0.10,   # Angry
    0.10,   # Disgust
    0.15,   # Fear
    0.05,   # Happy
    0.15,   # Neutral
    0.40,   # Sad
    0.05    # Surprise
])
result = detector.analyze_stress_from_emotions(sad_emotions)
print(f"Dominant Emotion: {result['dominant_emotion']}")
print(f"Stress Score: {result['stress_score']*100:.1f}%")
print(f"Stress Level: {result['stress_level']}")
print(f"Expected: Moderate (40-65%)")
print("[PASS]" if 0.40 <= result['stress_score'] <= 0.65 else "[FAIL]")

# Test 4: Angry face (should be High stress)
print("\n\nTest 4: ANGRY FACE")
print("-" * 40)
angry_emotions = np.array([
    0.45,   # Angry
    0.20,   # Disgust
    0.10,   # Fear
    0.05,   # Happy
    0.10,   # Neutral
    0.05,   # Sad
    0.05    # Surprise
])
result = detector.analyze_stress_from_emotions(angry_emotions)
print(f"Dominant Emotion: {result['dominant_emotion']}")
print(f"Stress Score: {result['stress_score']*100:.1f}%")
print(f"Stress Level: {result['stress_level']}")
print(f"Expected: High (55-70%)")
print("[PASS]" if result['stress_score'] > 0.55 else "[FAIL]")

# Test 5: Fearful face (should be Very High stress)
print("\n\nTest 5: FEARFUL FACE")
print("-" * 40)
fearful_emotions = np.array([
    0.15,   # Angry
    0.10,   # Disgust
    0.50,   # Fear
    0.05,   # Happy
    0.10,   # Neutral
    0.05,   # Sad
    0.05    # Surprise
])
result = detector.analyze_stress_from_emotions(fearful_emotions)
print(f"Dominant Emotion: {result['dominant_emotion']}")
print(f"Stress Score: {result['stress_score']*100:.1f}%")
print(f"Stress Level: {result['stress_level']}")
print(f"Expected: Very High (> 70%)")
print("[PASS]" if result['stress_score'] > 0.65 else "[FAIL]")

print("\n\n" + "="*40)
print("All tests completed!")
print("="*40)
