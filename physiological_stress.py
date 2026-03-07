"""
Physiological Stress Detection
Analyzes heart rate, blood pressure, respiration, and sleep for stress levels
Improved version that doesn't rely on problematic pickle models
"""

import numpy as np
from typing import Dict, Tuple, List

class PhysiologicalStressAnalyzer:
    """Analyze stress from physiological signals"""
    
    def __init__(self):
        """Initialize with stress thresholds and weights"""
        self.thresholds = {
            'heart_rate': {
                'low': (60, 80),           # BPM normal
                'moderate': (80, 100),     # Elevated
                'high': (100, 120),        # Very elevated
                'critical': (120, 200)     # Dangerous
            },
            'blood_pressure': {
                'low': (90, 120, 60, 80),           # Systolic/Diastolic normal
                'moderate': (120, 140, 80, 90),     # Elevated
                'high': (140, 160, 90, 100),        # High
                'critical': (160, 200, 100, 120)    # Critical
            },
            'respiration_rate': {
                'low': (12, 16),           # Breaths/min normal
                'moderate': (16, 20),      # Elevated
                'high': (20, 25),          # High
                'critical': (25, 40)       # Critical
            },
            'sleep_hours': {
                'good': (7, 9),            # Optimal sleep
                'moderate': (6, 7),        # Adequate
                'poor': (4, 6),            # Insufficient
                'critical': (0, 4)         # Severe deprivation
            }
        }
    
    def analyze_heart_rate(self, hr: float) -> Dict:
        """Analyze heart rate stress level"""
        if not 40 <= hr <= 200:
            return {'valid': False, 'error': 'Invalid heart rate'}
        
        stress_score = 0.0
        
        if 40 <= hr < 60:
            stress_score = 0.1  # Very low (resting)
            level = 'Very Low'
        elif 60 <= hr < 80:
            stress_score = 0.15  # Low (normal)
            level = 'Low'
        elif 80 <= hr < 100:
            stress_score = 0.4   # Moderate
            level = 'Moderate'
        elif 100 <= hr < 120:
            stress_score = 0.7   # High
            level = 'High'
        else:  # >= 120
            stress_score = 0.95  # Very high
            level = 'Very High'
        
        return {
            'valid': True,
            'heart_rate': hr,
            'stress_score': stress_score,
            'stress_level': level,
            'interpretation': f"{int(hr)} BPM is {level.lower()}"
        }
    
    def analyze_blood_pressure(self, systolic: float, diastolic: float) -> Dict:
        """Analyze blood pressure stress level"""
        if not (80 <= systolic <= 220 and 40 <= diastolic <= 150):
            return {'valid': False, 'error': 'Invalid blood pressure'}
        
        stress_score = 0.0
        
        if systolic < 120 and diastolic < 80:
            stress_score = 0.15  # Normal
            level = 'Low'
        elif systolic < 140 and diastolic < 90:
            stress_score = 0.4   # Elevated
            level = 'Moderate'
        elif systolic < 160 and diastolic < 100:
            stress_score = 0.7   # High
            level = 'High'
        else:
            stress_score = 0.95  # Very high
            level = 'Very High'
        
        return {
            'valid': True,
            'blood_pressure': f"{int(systolic)}/{int(diastolic)}",
            'stress_score': stress_score,
            'stress_level': level,
            'interpretation': f"{int(systolic)}/{int(diastolic)} is {level.lower()}"
        }
    
    def analyze_respiration_rate(self, rr: float) -> Dict:
        """Analyze respiration rate stress level"""
        if not 8 <= rr <= 40:
            return {'valid': False, 'error': 'Invalid respiration rate'}
        
        stress_score = 0.0
        
        if 12 <= rr < 16:
            stress_score = 0.15  # Normal
            level = 'Low'
        elif 16 <= rr < 20:
            stress_score = 0.4   # Elevated
            level = 'Moderate'
        elif 20 <= rr < 25:
            stress_score = 0.7   # High
            level = 'High'
        else:
            stress_score = 0.95  # Very high
            level = 'Very High'
        
        return {
            'valid': True,
            'respiration_rate': rr,
            'stress_score': stress_score,
            'stress_level': level,
            'interpretation': f"{rr} breaths/min is {level.lower()}"
        }
    
    def analyze_sleep(self, sleep_hours: float) -> Dict:
        """Analyze sleep quality/duration stress level"""
        if not 0 <= sleep_hours <= 12:
            return {'valid': False, 'error': 'Invalid sleep hours'}
        
        stress_score = 0.0
        
        if 7 <= sleep_hours <= 9:
            stress_score = 0.1   # Optimal
            level = 'Very Low'
        elif 6 <= sleep_hours < 7:
            stress_score = 0.25  # Adequate
            level = 'Low'
        elif 5 <= sleep_hours < 6:
            stress_score = 0.5   # Insufficient
            level = 'Moderate'
        elif 4 <= sleep_hours < 5:
            stress_score = 0.75  # Poor
            level = 'High'
        else:
            stress_score = 0.95  # Severe deprivation
            level = 'Very High'
        
        return {
            'valid': True,
            'sleep_hours': sleep_hours,
            'stress_score': stress_score,
            'stress_level': level,
            'interpretation': f"{sleep_hours}h sleep is {level.lower()}"
        }
    
    def combine_physiological_signals(
        self,
        heart_rate: float = None,
        systolic: float = None,
        diastolic: float = None,
        respiration_rate: float = None,
        sleep_hours: float = None
    ) -> Dict:
        """
        Combine multiple physiological signals for overall stress assessment
        
        Parameters:
        - heart_rate: Beats per minute (60-100 normal)
        - systolic: Systolic BP (120 normal)
        - diastolic: Diastolic BP (80 normal)
        - respiration_rate: Breaths per minute (12-16 normal)
        - sleep_hours: Hours of sleep (7-9 optimal)
        
        Returns: Combined stress analysis
        """
        scores = []
        analyses = {}
        
        # Analyze each signal if provided
        if heart_rate is not None:
            hr_analysis = self.analyze_heart_rate(heart_rate)
            if hr_analysis.get('valid'):
                scores.append(hr_analysis['stress_score'])
                analyses['heart_rate'] = hr_analysis
        
        if systolic is not None and diastolic is not None:
            bp_analysis = self.analyze_blood_pressure(systolic, diastolic)
            if bp_analysis.get('valid'):
                scores.append(bp_analysis['stress_score'])
                analyses['blood_pressure'] = bp_analysis
        
        if respiration_rate is not None:
            rr_analysis = self.analyze_respiration_rate(respiration_rate)
            if rr_analysis.get('valid'):
                scores.append(rr_analysis['stress_score'])
                analyses['respiration_rate'] = rr_analysis
        
        if sleep_hours is not None:
            sleep_analysis = self.analyze_sleep(sleep_hours)
            if sleep_analysis.get('valid'):
                scores.append(sleep_analysis['stress_score'])
                analyses['sleep'] = sleep_analysis
        
        # Calculate combined score
        if not scores:
            return {
                'success': False,
                'error': 'No valid physiological signals provided'
            }
        
        combined_score = np.mean(scores)
        
        # Determine overall level
        if combined_score < 0.25:
            overall_level = 'Very Low'
        elif combined_score < 0.4:
            overall_level = 'Low'
        elif combined_score < 0.6:
            overall_level = 'Moderate'
        elif combined_score < 0.8:
            overall_level = 'High'
        else:
            overall_level = 'Very High'
        
        # Generate recommendations
        recommendations = self._get_physiological_recommendations(overall_level, analyses)
        
        return {
            'success': True,
            'physiological_stress_score': round(combined_score, 2),
            'overall_stress_level': overall_level,
            'signals_analyzed': len(analyses),
            'individual_analyses': analyses,
            'recommendations': recommendations
        }
    
    def _get_physiological_recommendations(self, level: str, analyses: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if level == 'Very Low':
            recommendations.append('✅ Excellent physiological health indicators')
            recommendations.append('💚 Keep maintaining your healthy lifestyle')
        
        elif level == 'Low':
            recommendations.append('✅ Good physiological indicators')
            recommendations.append('🏃 Continue regular exercise and sleep')
        
        elif level == 'Moderate':
            recommendations.append('⚠️ Some physiological stress indicators detected')
            
            # Check specific signals
            if 'heart_rate' in analyses and analyses['heart_rate']['stress_score'] > 0.5:
                recommendations.append('💓 Try calming techniques to lower heart rate')
            
            if 'blood_pressure' in analyses and analyses['blood_pressure']['stress_score'] > 0.5:
                recommendations.append('🩹 Monitor blood pressure and reduce sodium intake')
            
            if 'respiration_rate' in analyses and analyses['respiration_rate']['stress_score'] > 0.5:
                recommendations.append('💨 Practice slow, deep breathing exercises')
            
            if 'sleep' in analyses and analyses['sleep']['stress_score'] > 0.5:
                recommendations.append('😴 Prioritize getting 7-9 hours of quality sleep')
        
        elif level == 'High':
            recommendations.append('⚠️⚠️ Multiple high stress indicators detected')
            recommendations.append('🏥 Consider consulting a healthcare professional')
            
            if 'heart_rate' in analyses:
                recommendations.append('💓 Take breaks and practice relaxation techniques')
            
            if 'sleep' in analyses and analyses['sleep']['stress_score'] > 0.7:
                recommendations.append('😴 Sleep deprivation is critical - establish sleep schedule')
            
            recommendations.append('🧘 Daily meditation or yoga recommended')
        
        else:  # Very High
            recommendations.append('🚨 Critical physiological stress levels')
            recommendations.append('📞 Seek immediate medical attention if symptoms persist')
            recommendations.append('🏥 Professional healthcare consultation strongly recommended')
            recommendations.append('🚨 If experiencing chest pain or severe symptoms, call emergency services')
        
        return recommendations


# Global analyzer instance
_analyzer = None

def get_physiological_analyzer():
    """Get or create the analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = PhysiologicalStressAnalyzer()
    return _analyzer

def analyze_physiological_stress(
    heart_rate: float = None,
    systolic: float = None,
    diastolic: float = None,
    respiration_rate: float = None,
    sleep_hours: float = None
) -> Dict:
    """
    Analyze physiological stress
    
    Example:
    result = analyze_physiological_stress(
        heart_rate=95,
        systolic=140,
        diastolic=90,
        respiration_rate=18,
        sleep_hours=5
    )
    """
    analyzer = get_physiological_analyzer()
    return analyzer.combine_physiological_signals(
        heart_rate=heart_rate,
        systolic=systolic,
        diastolic=diastolic,
        respiration_rate=respiration_rate,
        sleep_hours=sleep_hours
    )
