import os
import json
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, redirect, render_template, flash, request, jsonify, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_required, logout_user, login_user, LoginManager, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq
from werkzeug.utils import secure_filename
from facial_stress_detection import initialize_facial_detector, get_facial_stress_analysis, combine_multimodal_predictions
from physiological_stress import analyze_physiological_stress

# Try to import joblib for better sklearn model loading
try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    USE_JOBLIB = False 

# --- App Configuration ---
app = Flask(__name__, 
            static_url_path='', 
            static_folder='static', 
            template_folder='templates')

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = "tandrima"
import os
from dotenv import load_dotenv

# Load .env from this file's directory so behavior is stable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")

# --- Groq Configuration ---
client = None


def get_groq_client():
    """Lazily initialize Groq client to avoid app startup failure when key is missing."""
    global client
    if client is not None:
        return client

    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY is missing. Chat feature will be unavailable.")
        return None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"[WARNING] Failed to initialize Groq client: {e}")
        return None


def build_local_support_reply(user_message):
    """
    Offline fallback response when Groq is unavailable.
    Keeps chat functional with simple, supportive guidance.
    """
    text = (user_message or "").strip().lower()
    if not text:
        return "Please share what is on your mind, and I will help step by step."

    high_risk_words = ["suicide", "kill myself", "end my life", "self-harm", "self harm"]
    if any(word in text for word in high_risk_words):
        return "If you are in immediate danger, call emergency services now. You can also contact 988 (Suicide & Crisis Lifeline) right away."

    stress_words = ["stress", "anxious", "anxiety", "panic", "overwhelmed", "worried", "tension"]
    sad_words = ["sad", "depressed", "hopeless", "lonely", "empty", "crying"]
    sleep_words = ["sleep", "insomnia", "can't sleep", "cant sleep", "tired"]

    if any(word in text for word in stress_words):
        return (
            "It sounds like stress is high right now. Try this 60-second reset: "
            "inhale 4s, hold 4s, exhale 6s for 6 rounds. Then pick one small next task and do only that."
        )
    if any(word in text for word in sad_words):
        return (
            "I hear you. For the next 10 minutes, do one grounding action: drink water, wash your face, "
            "and message one trusted person. Small steps help when emotions feel heavy."
        )
    if any(word in text for word in sleep_words):
        return (
            "For tonight, keep lights low for 30 minutes before bed, avoid phone scrolling, and do slow breathing (4-6 pace). "
            "If awake after 20 minutes, get up briefly and read something calm."
        )

    return (
        "I am here with you. Tell me what happened today in 2-3 lines, and I will help you create a simple plan for the next hour."
    )

# --- Initialize Facial Stress Detector ---
initialize_facial_detector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Load Machine Learning Models ---
# Numerical/Physiological stress detection (now improved with direct calculation)
print("[OK] Physiological stress analyzer loaded (no pickle required)")
model = None  # Not needed anymore, using direct calculation

try:
    if os.path.exists('stresslevel_text_model.pkl') and os.path.exists('stresslevel_text_vectorizer.pkl'):
        if USE_JOBLIB:
            try:
                text_model = joblib.load('stresslevel_text_model.pkl')
                text_vectorizer = joblib.load('stresslevel_text_vectorizer.pkl')
                print("[OK] Text-based stress model loaded successfully (joblib)")
            except:
                text_model = pickle.load(open('stresslevel_text_model.pkl', 'rb'))
                text_vectorizer = pickle.load(open('stresslevel_text_vectorizer.pkl', 'rb'))
                print("[OK] Text-based stress model loaded successfully (pickle)")
        else:
            text_model = pickle.load(open('stresslevel_text_model.pkl', 'rb'))
            text_vectorizer = pickle.load(open('stresslevel_text_vectorizer.pkl', 'rb'))
            print("[OK] Text-based stress model loaded successfully (pickle)")
    else:
        print("[INFO] Text-based stress model files not found. Run 'python train_text_model.py' to create them.")
        text_model = None
        text_vectorizer = None
except Exception as e:
    print(f"[WARNING] Error loading text-based stress model: {e}")
    text_model = None
    text_vectorizer = None

# --- Database Model ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    usn = db.Column(db.String(20), unique=True)
    pas = db.Column(db.String(1000))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Authentication Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == "POST":
        usn = request.form.get('usn')
        pas = request.form.get('pas')
        if User.query.filter_by(usn=usn).first():
            flash("UserID is already taken", "warning")
            return render_template("usersignup.html")
        new_user = User(usn=usn, pas=generate_password_hash(pas))
        db.session.add(new_user)
        db.session.commit()
        return render_template("userlogin.html")
    return render_template("usersignup.html")

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        usn, pas = request.form.get('usn'), request.form.get('pas')
        user = User.query.filter_by(usn=usn).first()
        if user and check_password_hash(user.pas, pas):
            login_user(user)
            return redirect(url_for('home'))
        flash("Invalid Credentials", "danger")
    return render_template("userlogin.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Feature Routes ---
@app.route('/music')
@login_required
def music(): return render_template('music.html')

@app.route('/quizandgame')
@login_required
def quizandgame(): return render_template('quizandgame.html')

@app.route('/exercises')
@login_required
def exercises(): return render_template('exercises.html')

@app.route('/analysis')
@login_required
def analysis():
    try:
        train_df = pd.read_csv('dreaddit-train.csv', encoding='ISO-8859-1')
        fig = px.pie(train_df, names='subreddit', title='Subreddit Distribution')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('analysis.html', graphJSON=graphJSON)
    except:
        return "Analysis data not found."

@app.route('/facial')
@login_required
def facial():
    """Route to display facial stress detection page"""
    return render_template('facial_stress.html', prediction_text3="")

# --- Stress Detection Routes ---
@app.route('/i', methods=['GET', 'POST'])
@login_required
def i():
    """Route for numerical stress detection (GET: show form, POST: process prediction)"""
    if request.method == 'POST':
        try:
            # Get form inputs
            sleep_hours = request.form.get('rr')  # Sleeping hours
            systolic_bp = request.form.get('bp')  # Systolic Blood pressure
            diastolic_bp = request.form.get('bo')  # Diastolic Blood pressure / Respiration rate
            heart_rate = request.form.get('hr')  # Heart rate
            
            # Validate inputs
            if not all([sleep_hours, systolic_bp, diastolic_bp, heart_rate]):
                return render_template('stress.html', prediction_text3="Please fill in all fields.")
            
            # Convert to float and validate ranges
            try:
                sleep_hours = float(sleep_hours)
                systolic_bp = float(systolic_bp)
                diastolic_bp = float(diastolic_bp)
                heart_rate = float(heart_rate)
            except ValueError:
                return render_template('stress.html', prediction_text3="Please enter valid numbers.")
            
            # Use physiological stress analyzer
            result_dict = analyze_physiological_stress(
                heart_rate=heart_rate,
                systolic=systolic_bp,
                diastolic=diastolic_bp,
                respiration_rate=diastolic_bp,
                sleep_hours=sleep_hours
            )
            
            stress_score = result_dict.get('physiological_stress_score', 0.5)
            stress_level = result_dict.get('overall_stress_level', 'Unknown')
            
            # Format result
            if stress_score < 0.4:
                result = "✅ Stress Level: Low/No Stress\n\nYou seem to be managing stress well. Keep up the good habits!"
            elif stress_score < 0.7:
                result = f"⚠️ Stress Level: Moderate Stress\n\nYour physiological indicators suggest moderate stress levels ({stress_score*100:.1f}%).\n\nConsider:\n• Taking regular breaks\n• Practicing relaxation techniques\n• Getting more sleep\n• Using our Music Therapy and Exercises features"
            else:
                result = f"🔴 Stress Level: High Stress\n\nYour physiological indicators suggest elevated stress ({stress_score*100:.1f}%).\n\nConsider:\n• Taking breaks and practicing relaxation\n• Getting adequate sleep (recommend 7-9 hours)\n• Consulting with a healthcare professional\n• Using our Music Therapy and Exercises features"
            
            return render_template('stress.html', prediction_text3=result)
            
        except Exception as e:
            print(f"Error in stress prediction: {e}")
            return render_template('stress.html', prediction_text3=f"Error processing request: {str(e)}")
    
    # GET request - show form
    return render_template('stress.html', prediction_text3="")

@app.route('/stress_text', methods=['GET', 'POST'])
@login_required
def stress_text():
    """Route for text-based stress detection (GET: show form, POST: process prediction)"""
    if request.method == 'POST':
        try:
            # Get text input
            text = request.form.get('text', '').strip()
            
            if not text:
                return render_template('stress_text.html', prediction_text3="Please enter some text to analyze.")
            
            # Make prediction if model exists
            if text_model is None or text_vectorizer is None:
                return render_template('stress_text.html', prediction_text3="Text-based stress detection model not available. Please use the numerical input method or contact administrator.")
            
            # Vectorize the text
            text_vectorized = text_vectorizer.transform([text])
            
            # Make prediction
            prediction = text_model.predict(text_vectorized)[0]
            
            # Get prediction probability if available
            try:
                prediction_proba = text_model.predict_proba(text_vectorized)[0]
                stress_prob = prediction_proba[1] if len(prediction_proba) > 1 else 0.5
            except:
                stress_prob = 0.5
            
            # Format result
            if prediction == 0 or prediction == '0':
                result = f"✅ Stress Level: Low/No Stress\n\nBased on your text, you appear to be managing stress well. The model detected {stress_prob*100:.1f}% probability of stress indicators.\n\nKeep up the positive coping strategies!"
            else:
                result = f"⚠️ Stress Level: High Stress Detected\n\nThe model detected {stress_prob*100:.1f}% probability of stress indicators in your text.\n\nSuggestions:\n• Consider our Music Therapy feature for relaxation\n• Try our Exercises section for stress relief\n• Practice mindfulness and deep breathing\n• Consider speaking with a healthcare professional\n• Remember: It's okay to ask for help"
            
            return render_template('stress_text.html', prediction_text3=result)
            
        except Exception as e:
            print(f"Error in text-based stress prediction: {e}")
            return render_template('stress_text.html', prediction_text3=f"Error processing request: {str(e)}")
    
    # GET request - show form
    return render_template('stress_text.html', prediction_text3="")

@app.route('/stressdetect', methods=['POST'])
@login_required
def stressdetect():
    """Alternative route name for POST requests from stress.html form"""
    return i()

@app.route('/stressdetect_text', methods=['POST'])
@login_required
def stressdetect_text():
    """Route name for POST requests from stress_text.html form"""
    return stress_text()

@app.route('/facial_stress', methods=['GET', 'POST'])
@login_required
def facial_stress_page():
    """Route for facial recognition-based stress detection"""
    if request.method == 'POST':
        try:
            # Check if file is uploaded
            if 'image' not in request.files:
                return render_template('facial_stress.html', prediction_text3="Please upload an image.")
            
            file = request.files['image']
            
            if file.filename == '':
                return render_template('facial_stress.html', prediction_text3="Please select a file.")
            
            if not allowed_file(file.filename):
                return render_template('facial_stress.html', prediction_text3="Only image files (PNG, JPG, JPEG, GIF, BMP) are allowed.")
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
            file.save(filepath)
            
            # Analyze facial stress
            facial_result = get_facial_stress_analysis(filepath)
            
            # Format result with better display
            if facial_result.get('error'):
                result = f"⚠️ Analysis Error: {facial_result['error']}\n\nTip: Ensure your face is clearly visible in the image."
            else:
                faces_detected = facial_result.get('faces_detected', 0)
                stress_score = facial_result.get('average_stress_score', 0.5)
                stress_level = facial_result.get('stress_level', 'Unknown')
                
                if faces_detected == 0:
                    result = "⚠️ No faces detected.\nPlease upload a clear image with your face visible."
                else:
                    result = f"✅ FACIAL STRESS ANALYSIS COMPLETE\n"
                    result += f"{'='*40}\n\n"
                    result += f"📊 Stress Level: {stress_level}\n"
                    result += f"📈 Stress Score: {stress_score*100:.1f}%\n"
                    result += f"👤 Faces Detected: {faces_detected}\n\n"
                    
                    # Add detailed emotion breakdown
                    if facial_result.get('face_data'):
                        face_data = facial_result['face_data'][0]
                        emotions = face_data.get('emotions', {})
                        dominant_emotion = face_data.get('dominant_emotion', 'Unknown')
                        
                        if emotions:
                            result += f"{'Emotion Analysis':^40}\n"
                            result += f"{'-'*40}\n"
                            result += f"Primary: {dominant_emotion}\n\n"
                            
                            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                            for emotion, confidence in sorted_emotions:
                                bar_len = int(confidence * 30)
                                bar = '█' * bar_len + '░' * (30 - bar_len)
                                result += f"{emotion:10s} {bar} {confidence*100:5.1f}%\n"
                    
                    # Add recommendations
                    if facial_result.get('face_data'):
                        face_data = facial_result['face_data'][0]
                        recommendations = face_data.get('recommendations', [])
                        if recommendations:
                            result += f"\n{'Recommendations':^40}\n"
                            result += f"{'-'*40}\n"
                            for i, rec in enumerate(recommendations[:4], 1):
                                result += f"{i}. {rec}\n"
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return render_template('facial_stress.html', prediction_text3=result)
            
        except Exception as e:
            print(f"[ERROR] Error in facial stress detection: {e}")
            return render_template('facial_stress.html', prediction_text3=f"Error processing facial image: {str(e)}")
    
    return render_template('facial_stress.html', prediction_text3="")

@app.route('/multimodal_stress', methods=['GET', 'POST'])
@login_required
def multimodal_stress():
    """Combined multimodal stress detection (numerical, text, and facial)"""
    if request.method == 'GET':
        return render_template('multimodal_stress.html', prediction_text3="")
    
    try:
        # Get numerical data
        sleep_hours = request.form.get('rr', '')
        systolic_bp = request.form.get('bp', '')
        diastolic_bp = request.form.get('bo', '')
        heart_rate = request.form.get('hr', '')
        
        # Get text input
        text = request.form.get('text', '').strip()
        
        # Get facial image
        facial_file = request.files.get('image')
        
        facial_stress_score = None
        text_stress_score = None
        physiological_stress_score = None
        
        # Process physiological data using new analyzer
        if sleep_hours and systolic_bp and diastolic_bp and heart_rate:
            try:
                sleep_hours = float(sleep_hours)
                systolic_bp = float(systolic_bp)
                diastolic_bp = float(diastolic_bp)
                heart_rate = float(heart_rate)
                
                phys_result = analyze_physiological_stress(
                    heart_rate=heart_rate,
                    systolic=systolic_bp,
                    diastolic=diastolic_bp,
                    respiration_rate=diastolic_bp,
                    sleep_hours=sleep_hours
                )
                physiological_stress_score = phys_result.get('physiological_stress_score', 0.5)
            except Exception as e:
                print(f"Error processing physiological data: {e}")
        
        # Process text data
        if text:
            if text_model and text_vectorizer:
                try:
                    text_vec = text_vectorizer.transform([text])
                    pred = text_model.predict(text_vec)[0]
                    try:
                        prob = text_model.predict_proba(text_vec)[0]
                        text_stress_score = prob[1] if len(prob) > 1 else 0.5
                    except:
                        text_stress_score = 0.5 if pred == 1 else 0.2
                except Exception as e:
                    print(f"Error processing text data: {e}")
        
        # Process facial image
        facial_result = None
        if facial_file and facial_file.filename != '':
            if allowed_file(facial_file.filename):
                try:
                    filename = secure_filename(facial_file.filename)
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_')
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp + filename)
                    facial_file.save(filepath)
                    
                    facial_result = get_facial_stress_analysis(filepath)
                    if not facial_result.get('error'):
                        facial_stress_score = facial_result.get('average_stress_score', 0.5)
                    
                    # Clean up
                    try:
                        os.remove(filepath)
                    except:
                        pass
                except Exception as e:
                    print(f"Error processing facial data: {e}")
        
        # Combine results using proper scores
        multimodal = combine_multimodal_predictions(
            facial_stress_score if facial_stress_score is not None else 0.5,
            physiological_stress_score if physiological_stress_score is not None else 0.5,
            text_stress_score if text_stress_score is not None else 0.5
        )
        
        result = f"🎯 MULTIMODAL STRESS ANALYSIS\n"
        result += f"=" * 40 + "\n\n"
        
        if physiological_stress_score is not None:
            phys_level = "High" if physiological_stress_score > 0.7 else ("Moderate" if physiological_stress_score > 0.4 else "Low")
            result += f"📊 Physiological Data: {phys_level} Stress ({physiological_stress_score*100:.1f}%)\n"
        if text_stress_score is not None:
            text_level = "High Stress" if text_stress_score > 0.6 else "Low Stress"
            result += f"📝 Text Analysis: {text_level} ({text_stress_score*100:.1f}% confidence)\n"
        if facial_stress_score is not None:
            facial_level = "High" if facial_stress_score > 0.7 else ("Moderate" if facial_stress_score > 0.4 else "Low")
            result += f"👤 Facial Analysis: {facial_level} Stress ({facial_stress_score*100:.1f}%)\n"
        
        result += f"\n" + "=" * 40 + "\n"
        result += f"🎯 Overall Stress Level: {multimodal.get('stress_level', 'Unknown')}\n"
        result += f"📈 Combined Score: {multimodal.get('combined_score', 0)*100:.1f}%\n"
        
        return render_template('multimodal_stress.html', prediction_text3=result)
        
    except Exception as e:
        print(f"Error in multimodal stress detection: {e}")
        return render_template('stress.html', prediction_text3=f"Error in multimodal analysis: {str(e)}")

# --- CHAT ROUTE (Fixed 302 and AI logic) ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get('message', '')).strip()
        
        if not user_message:
            return jsonify({'reply': 'Empty message.'})

        # Safety Check
        if any(word in user_message.lower() for word in ['suicide', 'kill']):
            return jsonify({'reply': "Please contact 988 Crisis Line immediately."})

        # Groq Call
        try:
            groq_client = get_groq_client()
            if groq_client is None:
                return jsonify({'reply': build_local_support_reply(user_message)})

            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are MindEase, a supportive mental health assistant. Give short responses."},
                    {"role": "user", "content": user_message}
                ]
            )
            reply = completion.choices[0].message.content
            return jsonify({'reply': reply})
        except Exception as api_err:
            print(f"API Error: {api_err}")
            return jsonify({'reply': build_local_support_reply(user_message)})

    except Exception as e:
        return jsonify({'reply': "System error. Please try again."})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
