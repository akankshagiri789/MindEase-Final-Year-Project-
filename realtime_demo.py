"""
Real-time Facial Stress Detection Demo
Captures from webcam and shows live stress detection
"""

import cv2
import numpy as np
import sys
import os
from facial_stress_detection import FacialStressDetector

print("[OK] Real-time Facial Stress Detection Demo")
print("[INFO] Press 'q' to quit, 's' to save analysis\n")

try:
    # Initialize detector
    detector = FacialStressDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        sys.exit(1)
    
    print("[OK] Camera opened successfully")
    print("[INFO] Starting real-time detection...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        
        # Analyze frame every 5 frames to reduce CPU usage
        if frame_count % 5 == 0:
            results = detector.process_frame(frame)
            
            # Draw results on frame
            if results.get('faces_detected', 0) > 0:
                # Draw face rectangles and stress info
                for i, face_data in enumerate(results.get('face_data', [])):
                    # Get face location
                    location = face_data.get('location', [])
                    if len(location) == 4:
                        x, y, w, h = location
                        
                        # Draw rectangle
                        stress_score = face_data.get('stress_score', 0)
                        color = (0, 255, 0) if stress_score < 0.4 else (0, 255, 255) if stress_score < 0.7 else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Put text
                        stress_level = face_data.get('stress_level', 'Unknown')
                        text = f"{stress_level} ({stress_score*100:.0f}%)"
                        cv2.putText(frame, text, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                cv2.putText(frame, "No faces detected", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show overall stress
            avg_stress = results.get('average_stress_score', 0)
            overall_level = results.get('stress_level', 'Unknown')
            cv2.putText(frame, f"Overall: {overall_level} ({avg_stress*100:.0f}%)", 
                      (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Facial Stress Detection", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[OK] Exiting...")
            break
        elif key == ord('s'):
            # Save frame and analysis
            filename = f"stress_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[OK] Saved frame to {filename}")
            
            # Get detailed analysis
            results = detector.process_frame(frame)
            if results.get('faces_detected', 0) > 0:
                for face_data in results.get('face_data', []):
                    print(f"  Stress Level: {face_data['stress_level']}")
                    print(f"  Stress Score: {face_data['stress_score']:.2%}")
                    print(f"  Emotions:")
                    for emotion, confidence in sorted(face_data['emotions'].items(), 
                                                     key=lambda x: x[1], reverse=True):
                        print(f"    - {emotion}: {confidence:.1%}")
            else:
                print("[WARNING] No faces in frame to analyze")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Demo completed")
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
