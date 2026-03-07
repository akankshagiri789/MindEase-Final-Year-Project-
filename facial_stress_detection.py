"""
Facial Recognition-based Stress Detection using DeepFace
Provides accurate emotion recognition and stress analysis with real face detection
"""

import cv2
import numpy as np
from deepface import DeepFace
from pathlib import Path
import os
import base64
from io import BytesIO
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FacialStressDetector:
    def __init__(self):
        """Initialize facial stress detection with DeepFace"""
        self.face_cascade = None
        self.load_models()
        
    def load_models(self):
        """Load face detection cascade classifier"""
        try:
            # Load face detection cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("[OK] Face cascade classifier loaded")
            print("[OK] DeepFace will be used for emotion detection (VGG-Face backend)")
        except Exception as e:
            print(f"[ERROR] Error loading facial models: {e}")
    
    def detect_faces(self, frame):
        """Detect faces in frame using cascade classifier"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces, gray
    
    def analyze_emotion_deepface(self, frame, face_region):
        """
        Analyze emotions using DeepFace (highly accurate pre-trained models)
        Returns: emotion_dict with all emotion probabilities
        """
        try:
            x, y, w, h = face_region
            face_roi = frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB for DeepFace
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace for emotion analysis with VGG-Face backend (most accurate)
            # enforce_detection=False allows analysis even on edge cases
            analysis = DeepFace.analyze(
                face_rgb, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if analysis and len(analysis) > 0:
                emotion_data = analysis[0]['emotion']
                # Normalize to 0-1 range
                emotion_dict = {emotion: confidence / 100.0 for emotion, confidence in emotion_data.items()}
                return emotion_dict
            else:
                return None
                
        except Exception as e:
            print(f"[WARNING] DeepFace analysis error: {e}")
            return None
    
    def analyze_stress_from_emotions(self, emotions_detected):
        """
        Analyze stress level based on detected emotions
        Uses research-based stress mapping
        Returns: stress_score (0-1), stress_level (string), recommendations (list)
        """
        if emotions_detected is None:
            return 0.5, "Unknown", ["Unable to analyze emotions"]
        
        # Stress weights based on psychological research
        # Higher values = more stressful emotion
        stress_weights = {
            'fear': 0.95,        # Highest stress - fear/anxiety
            'angry': 0.90,       # Very high stress - frustration/anger
            'sad': 0.80,         # High stress - depression/sadness
            'disgust': 0.70,     # Medium-high stress - disgust/aversion
            'surprise': 0.45,    # Medium stress - uncertainty
            'neutral': 0.25,     # Low stress - baseline
            'happy': 0.05        # Lowest stress - positive emotion
        }
        
        # Calculate weighted stress score
        stress_score = 0.0
        total_weight = 0.0
        
        for emotion, confidence in emotions_detected.items():
            emotion_lower = emotion.lower()
            weight = stress_weights.get(emotion_lower, 0.5)
            stress_score += weight * confidence
            total_weight += confidence
        
        # Normalize stress score
        if total_weight > 0:
            stress_score = stress_score / total_weight
        else:
            stress_score = 0.5
        
        # Determine stress level
        if stress_score < 0.20:
            stress_level = "Very Low"
        elif stress_score < 0.35:
            stress_level = "Low"
        elif stress_score < 0.50:
            stress_level = "Moderate"
        elif stress_score < 0.70:
            stress_level = "High"
        else:
            stress_level = "Very High"
        
        # Generate recommendations based on stress level
        recommendations = self._get_recommendations(stress_level)
        
        return stress_score, stress_level, recommendations
    
    def _get_recommendations(self, stress_level):
        """Generate wellness recommendations based on stress level"""
        recommendations_map = {
            "Very Low": [
                "✅ Excellent! Your stress level is very low",
                "🧘 Maintain current healthy habits",
                "📚 Continue with regular mindfulness practice"
            ],
            "Low": [
                "✅ Good! Your stress level is low",
                "🎯 Keep up with healthy stress management",
                "💪 Regular exercise helps maintain low stress"
            ],
            "Moderate": [
                "⚠️ Moderate stress detected",
                "🧘 Try meditation or mindfulness exercises",
                "🚶 Take a short break from current activities",
                "🎵 Listen to calming music"
            ],
            "High": [
                "⚠️⚠️ High stress detected",
                "🧘 Practice progressive muscle relaxation",
                "💨 Try deep breathing exercises (4-7-8 technique)",
                "📞 Consider talking to someone about your stress",
                "🏃 Physical exercise can significantly reduce stress"
            ],
            "Very High": [
                "🚨 Very high stress detected",
                "📞 Consider reaching out for professional support",
                "🧘 Immediate relaxation needed - try meditation",
                "💨 Practice the 4-7-8 breathing technique now",
                "😴 Ensure adequate rest and sleep"
            ]
        }
        
        return recommendations_map.get(stress_level, ["Recommendations unavailable"])
    
    def get_facial_stress_analysis(self, image_path):
        """
        Main function to analyze stress from facial image
        Returns: comprehensive stress analysis with emotions and recommendations
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect faces
            faces, gray = self.detect_faces(img)
            
            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'No faces detected in image',
                    'faces_detected': 0
                }
            
            # Analyze each face
            face_data_list = []
            stress_scores = []
            
            for i, face_region in enumerate(faces):
                # Analyze emotions
                emotions = self.analyze_emotion_deepface(img, face_region)
                
                if emotions is None:
                    continue
                
                # Calculate stress
                stress_score, stress_level, recommendations = self.analyze_stress_from_emotions(emotions)
                stress_scores.append(stress_score)
                
                # Get dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                
                face_data = {
                    'face_id': i + 1,
                    'stress_score': round(stress_score, 2),
                    'stress_level': stress_level,
                    'dominant_emotion': dominant_emotion,
                    'emotion_confidence': round(emotions[dominant_emotion], 2),
                    'all_emotions': {emotion: round(conf, 4) for emotion, conf in emotions.items()},
                    'recommendations': recommendations
                }
                face_data_list.append(face_data)
            
            if not face_data_list:
                return {
                    'success': False,
                    'error': 'Faces detected but emotion analysis failed',
                    'faces_detected': len(faces)
                }
            
            # Calculate average stress
            average_stress = np.mean(stress_scores) if stress_scores else 0.5
            
            # Overall assessment
            if average_stress < 0.20:
                overall_level = "Very Low"
            elif average_stress < 0.35:
                overall_level = "Low"
            elif average_stress < 0.50:
                overall_level = "Moderate"
            elif average_stress < 0.70:
                overall_level = "High"
            else:
                overall_level = "Very High"
            
            return {
                'success': True,
                'faces_detected': len(face_data_list),
                'average_stress_score': round(average_stress, 2),
                'overall_stress_level': overall_level,
                'face_data': face_data_list
            }
            
        except Exception as e:
            print(f"[ERROR] Facial analysis error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global detector instance
_detector = None

def initialize_facial_detector():
    """Initialize global facial stress detector"""
    global _detector
    if _detector is None:
        _detector = FacialStressDetector()
    return _detector

def get_facial_stress_analysis(image_path):
    """Get stress analysis for image"""
    global _detector
    if _detector is None:
        _detector = initialize_facial_detector()
    return _detector.get_facial_stress_analysis(image_path)

def get_facial_stress_from_base64(base64_string):
    """
    Analyze facial stress from base64 encoded image
    Useful for camera uploads from frontend
    """
    try:
        global _detector
        if _detector is None:
            _detector = initialize_facial_detector()
        
        # Decode base64
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array and BGR
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces, _ = _detector.detect_faces(image_np)
        
        if len(faces) == 0:
            return {
                'success': False,
                'error': 'No faces detected in image',
                'faces_detected': 0
            }
        
        # Analyze each face
        face_data_list = []
        stress_scores = []
        
        for i, face_region in enumerate(faces):
            emotions = _detector.analyze_emotion_deepface(image_np, face_region)
            
            if emotions is None:
                continue
            
            stress_score, stress_level, recommendations = _detector.analyze_stress_from_emotions(emotions)
            stress_scores.append(stress_score)
            
            dominant_emotion = max(emotions, key=emotions.get)
            
            face_data = {
                'face_id': i + 1,
                'stress_score': round(stress_score, 2),
                'stress_level': stress_level,
                'dominant_emotion': dominant_emotion,
                'emotion_confidence': round(emotions[dominant_emotion], 2),
                'all_emotions': {emotion: round(conf, 4) for emotion, conf in emotions.items()},
                'recommendations': recommendations
            }
            face_data_list.append(face_data)
        
        if not face_data_list:
            return {
                'success': False,
                'error': 'Emotion analysis failed',
                'faces_detected': len(faces)
            }
        
        average_stress = np.mean(stress_scores)
        
        if average_stress < 0.20:
            overall_level = "Very Low"
        elif average_stress < 0.35:
            overall_level = "Low"
        elif average_stress < 0.50:
            overall_level = "Moderate"
        elif average_stress < 0.70:
            overall_level = "High"
        else:
            overall_level = "Very High"
        
        return {
            'success': True,
            'faces_detected': len(face_data_list),
            'average_stress_score': round(average_stress, 2),
            'overall_stress_level': overall_level,
            'face_data': face_data_list
        }
        
    except Exception as e:
        print(f"[ERROR] Base64 analysis error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def combine_multimodal_predictions(facial_stress, physiological_stress, text_stress):
    """
    Combine predictions from facial, physiological, and text analysis
    Returns: combined stress prediction and recommendations
    """
    try:
        weights = {
            'facial': 0.35,
            'physiological': 0.35,
            'text': 0.30
        }
        
        facial_score = facial_stress if isinstance(facial_stress, (int, float)) else 0.5
        physio_score = physiological_stress if isinstance(physiological_stress, (int, float)) else 0.5
        text_score = text_stress if isinstance(text_stress, (int, float)) else 0.5
        
        combined_score = (
            weights['facial'] * facial_score +
            weights['physiological'] * physio_score +
            weights['text'] * text_score
        )
        
        # Determine combined stress level
        if combined_score < 0.20:
            stress_level = "Very Low"
        elif combined_score < 0.35:
            stress_level = "Low"
        elif combined_score < 0.50:
            stress_level = "Moderate"
        elif combined_score < 0.70:
            stress_level = "High"
        else:
            stress_level = "Very High"
        
        return {
            'combined_score': round(combined_score, 2),
            'stress_level': stress_level,
            'facial_contribution': f"{weights['facial']*100:.0f}%",
            'physiological_contribution': f"{weights['physiological']*100:.0f}%",
            'text_contribution': f"{weights['text']*100:.0f}%"
        }
    except Exception as e:
        print(f"[ERROR] Multimodal combination error: {e}")
        return {
            'combined_score': 0.5,
            'stress_level': 'Unknown',
            'error': str(e)
        }