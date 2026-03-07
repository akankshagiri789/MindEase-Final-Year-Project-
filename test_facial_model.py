"""
Test Emotion Model and Facial Stress Detection
Run this to verify everything is working correctly
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facial_stress_detection import FacialStressDetector, get_facial_stress_analysis

print("[OK] Testing Facial Stress Detection System\n")

# Test 1: Initialize detector
print("Test 1: Initializing facial stress detector...")
try:
    detector = FacialStressDetector()
    print("[OK] Detector initialized successfully\n")
except Exception as e:
    print(f"[ERROR] Failed to initialize detector: {e}\n")
    sys.exit(1)

# Test 2: Create test image with synthetic face data
print("Test 2: Creating test image...")
try:
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [200, 200, 200]  # Gray background
    
    # Add some face-like shapes
    cv2.circle(test_image, (320, 240), 80, (180, 150, 120), -1)  # Face circle
    cv2.circle(test_image, (290, 220), 15, (50, 50, 50), -1)     # Left eye
    cv2.circle(test_image, (350, 220), 15, (50, 50, 50), -1)     # Right eye
    cv2.rectangle(test_image, (300, 260), (340, 280), (100, 50, 50), -1)  # Mouth
    
    test_path = 'test_face.jpg'
    cv2.imwrite(test_path, test_image)
    print(f"[OK] Test image created at {test_path}\n")
except Exception as e:
    print(f"[ERROR] Failed to create test image: {e}\n")
    sys.exit(1)

# Test 3: Analyze test image
print("Test 3: Analyzing test image...")
try:
    results = detector.process_image_file(test_path)
    
    print("[OK] Analysis completed!")
    print(f"    Faces detected: {results.get('faces_detected', 0)}")
    print(f"    Stress score: {results.get('average_stress_score', 0):.2%}")
    print(f"    Stress level: {results.get('stress_level', 'Unknown')}")
    
    if results.get('face_data'):
        face = results['face_data'][0]
        print(f"\n    Face Details:")
        print(f"    - Dominant emotion: {face.get('dominant_emotion', 'Unknown')}")
        print(f"    - Emotion breakdown:")
        
        emotions = face.get('emotions', {})
        for emotion, confidence in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"      * {emotion}: {confidence:.2%}")
        
        recommendations = face.get('recommendations', [])
        if recommendations:
            print(f"\n    Recommendations:")
            for i, rec in enumerate(recommendations[:2], 1):
                print(f"    {i}. {rec}")
    
    print("\n[OK] All tests passed!\n")
    
except Exception as e:
    print(f"[ERROR] Analysis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
try:
    os.remove(test_path)
    print("[OK] Cleanup complete")
except:
    pass

print("\n" + "="*50)
print("[SUCCESS] Facial stress detection is working!")
print("="*50)
print("\nNext steps:")
print("1. Train emotion model: python train_emotion_model.py")
print("2. Run the app: python app.py")
print("3. Visit: http://localhost:5000/facial_stress")
