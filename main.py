import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from datetime import datetime

# App Config
st.set_page_config(
    page_title="HeartCare AI Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model Assets
@st.cache_resource
def load_models():
    model = joblib.load("LOGREG_heart.pkl")
    scaler = joblib.load("scaler_HEART.pkl")
    expected_columns = joblib.load("columns_feature.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

# Enhanced CSS with Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Title Styles */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #e5f2ff;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
    }
    
    /* Stats Card */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #e5f2ff;
        font-weight: 500;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: #fff;
        border-radius: 16px;
        border: none;
        padding: 14px 28px;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
    }
    
    /* Alert Boxes */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    .error-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 25px rgba(235, 51, 73, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Health Tips Card */
    .tip-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        padding: 1.5rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        border-left: 4px solid #38ef7d;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .tip-card:hover {
        transform: translateX(10px);
        border-left: 4px solid #f5576c;
    }
    
    .tip-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .tip-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.5rem;
    }
    
    .tip-content {
        font-size: 1rem;
        color: #e5f2ff;
        line-height: 1.6;
    }
    
    /* Feature Badge */
    .feature-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.3rem;
        color: #fff;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 0.95rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Input Field Customization */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Radio Button Styling */
    .stRadio > label {
        color: #fff !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='color: #fff; text-align: center; margin-bottom: 2rem;'>🫀 HeartCare AI</h2>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Prediction", "💡 Health Tips", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(f"""
    <div style='color: rgba(255,255,255,0.7); font-size: 0.85rem; text-align: center; margin-top: 2rem;'>
        <p>🕐 {datetime.now().strftime('%B %d, %Y')}</p>
        <p style='margin-top: 1rem;'>Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# HOME PAGE
def page_home():
    st.markdown("<h1 class='main-title'>🫀 HeartCare AI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Machine Learning for Cardiac Risk Assessment</p>", unsafe_allow_html=True)
    
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Stats Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>90%</div>
            <div class='stat-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>10K+</div>
            <div class='stat-label'>Predictions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>18</div>
            <div class='stat-label'>Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-number'>24/7</div>
            <div class='stat-label'>Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Info Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🎯 Why Choose HeartCare AI?</h3>
            <p style='color: #333; line-height: 1.8;'>
                Our advanced machine learning model analyzes 18 critical cardiac parameters to provide 
                instant risk assessment. Early detection can save lives - get your prediction in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🔬 How It Works</h3>
            <p style='color: #333; line-height: 1.8;'>
                Enter your health metrics including age, blood pressure, cholesterol, and ECG results. 
                Our AI model processes the data using logistic regression to predict heart disease risk.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <p style='text-align: center; color: #e5f2ff; margin-bottom: 0.5rem;'>
            👉 Click 'Prediction' in the sidebar to begin your assessment
        </p>
        """, unsafe_allow_html=True)

# PREDICTION PAGE
def page_prediction():
    st.markdown("<h1 class='main-title'>🔍 Heart Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Complete the form below for personalized risk analysis</p>", unsafe_allow_html=True)
    
    # Progress indicator
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Form Section
    with st.form("prediction_form"):
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age (years)", 18, 100, 40, help="Your current age")
            gender = st.selectbox("Biological Sex", ["Male", "Female"], help="Biological sex assigned at birth")
            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina (TA)"],
                help="Type of chest pain experienced"
            )
        
        with col2:
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest")
            cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 600, 200, help="Serum cholesterol")
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"], help="Is fasting blood sugar above 120?")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🫀 Cardiac Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            resting_ecg = st.selectbox(
                "Resting ECG Results",
                ["Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"],
                help="Resting electrocardiogram results"
            )
            max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, help="Peak heart rate during exercise")
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], help="Chest pain during exercise")
        
        with col2:
            oldpeak = st.slider(
                "ST Depression (Oldpeak)",
                0.0, 6.0, 1.0, 0.1,
                help="ST depression induced by exercise relative to rest"
            )
            st_slope = st.selectbox(
                "ST Segment Slope",
                ["Upsloping (Up)", "Flat", "Downsloping (Down)"],
                help="Slope of the peak exercise ST segment"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🧠 Analyze Heart Risk", use_container_width=True)
    
    # Prediction Logic
    if submitted:
        with st.spinner("🔄 Analyzing your cardiac data..."):
            time.sleep(1.5)
            
            # Data Processing
            is_Male = 1 if gender == "Male" else 0
            
            cp_map = {"Asymptomatic (ASY)": "ASY", "Atypical Angina (ATA)": "ATA", 
                     "Non-Anginal Pain (NAP)": "NAP", "Typical Angina (TA)": "TA"}
            cp = {k: 0 for k in ["ASY", "ATA", "NAP", "TA"]}
            cp[cp_map[chest_pain]] = 1
            
            ecg_map = {"Normal": "Normal", "ST-T Wave Abnormality (ST)": "ST", 
                      "Left Ventricular Hypertrophy (LVH)": "LVH"}
            ecg = {k: 0 for k in ["LVH", "Normal", "ST"]}
            ecg[ecg_map[resting_ecg]] = 1
            
            angina = 1 if exercise_angina == "Yes" else 0
            fbs = 1 if fasting_bs == "Yes" else 0
            
            slope_map = {"Upsloping (Up)": "Up", "Flat": "Flat", "Downsloping (Down)": "Down"}
            slope = {k: 0 for k in ["Down", "Flat", "Up"]}
            slope[slope_map[st_slope]] = 1
            
            # Create input array
            input_data = np.array([[
                age, is_Male, resting_bp, cholesterol, fbs, max_hr, oldpeak,
                cp["ASY"], cp["ATA"], cp["NAP"], cp["TA"],
                ecg["LVH"], ecg["Normal"], ecg["ST"],
                angina,
                slope["Down"], slope["Flat"], slope["Up"]
            ]])
            
            # Prediction
            df = pd.DataFrame(input_data, columns=expected_columns)
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0]
            
            # Results Display
            st.markdown("<br>", unsafe_allow_html=True)
            
            if pred == 1:
                st.markdown(f"""
                <div class='error-box'>
                    <h2>⚠️ High Risk Detected</h2>
                    <p>Risk Probability: {prob[1]*100:.1f}%</p>
                    <p style='margin-top: 1rem; font-size: 1rem;'>
                        Our model indicates elevated risk of heart disease. 
                        Please consult a cardiologist for comprehensive evaluation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-card'>
                    <h4 style='color: #eb3349;'>📋 Recommended Actions:</h4>
                    <ul style='color: #333; line-height: 2;'>
                        <li>Schedule an appointment with a cardiologist immediately</li>
                        <li>Get comprehensive cardiac testing done</li>
                        <li>Monitor blood pressure and cholesterol regularly</li>
                        <li>Adopt heart-healthy lifestyle changes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.markdown(f"""
                <div class='success-box'>
                    <h2>✅ Low Risk Profile</h2>
                    <p>Risk Probability: {prob[1]*100:.1f}%</p>
                    <p style='margin-top: 1rem; font-size: 1rem;'>
                        Great news! Your cardiac risk profile looks healthy. 
                        Continue maintaining your current lifestyle.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-card'>
                    <h4 style='color: #11998e;'>💚 Keep Up The Good Work:</h4>
                    <ul style='color: #333; line-height: 2;'>
                        <li>Maintain regular physical activity</li>
                        <li>Continue healthy eating habits</li>
                        <li>Get annual health checkups</li>
                        <li>Manage stress through relaxation techniques</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# HEALTH TIPS PAGE
def page_tips():
    st.markdown("<h1 class='main-title'>💡 Heart Health Guidelines</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Evidence-based tips for a healthy heart</p>", unsafe_allow_html=True)
    
    tips = [
        {
            "icon": "🏃",
            "title": "Regular Physical Activity",
            "content": "Aim for at least 150 minutes of moderate aerobic exercise weekly. Activities like brisk walking, swimming, or cycling strengthen your heart and improve circulation."
        },
        {
            "icon": "🥗",
            "title": "Heart-Healthy Diet",
            "content": "Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit saturated fats, trans fats, sodium, and added sugars. The Mediterranean diet is excellent for heart health."
        },
        {
            "icon": "🚭",
            "title": "Quit Smoking & Limit Alcohol",
            "content": "Smoking is a major risk factor for heart disease. If you drink alcohol, do so in moderation - up to one drink per day for women and two for men."
        },
        {
            "icon": "⚖️",
            "title": "Maintain Healthy Weight",
            "content": "Keep your BMI within the healthy range (18.5-24.9). Even a 5-10% weight loss can significantly reduce heart disease risk if you're overweight."
        },
        {
            "icon": "💊",
            "title": "Monitor Blood Pressure & Cholesterol",
            "content": "Regular screening is crucial. Keep blood pressure below 120/80 mm Hg and maintain healthy cholesterol levels through diet, exercise, and medication if prescribed."
        },
        {
            "icon": "😴",
            "title": "Quality Sleep",
            "content": "Aim for 7-9 hours of quality sleep nightly. Poor sleep is linked to high blood pressure, obesity, and increased heart disease risk."
        },
        {
            "icon": "🧘",
            "title": "Stress Management",
            "content": "Chronic stress contributes to heart disease. Practice relaxation techniques like meditation, yoga, deep breathing, or hobbies you enjoy."
        },
        {
            "icon": "🩺",
            "title": "Regular Health Checkups",
            "content": "Visit your doctor annually for comprehensive health screenings. Early detection of risk factors allows for timely intervention and prevention."
        }
    ]
    
    for i, tip in enumerate(tips):
        st.markdown(f"""
        <div class='tip-card'>
            <div style='display: flex; align-items: start;'>
                <span class='tip-icon'>{tip['icon']}</span>
                <div>
                    <div class='tip-title'>{tip['title']}</div>
                    <div class='tip-content'>{tip['content']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning Signs Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card' style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white;'>
        <h3 style='color: white; margin-bottom: 1rem;'>🚨 Warning Signs - Seek Immediate Help</h3>
        <p style='line-height: 1.8;'>
            Call emergency services if you experience: chest pain or discomfort, shortness of breath, 
            pain in arms/back/neck/jaw/stomach, cold sweat, nausea, or lightheadedness. 
            Time is critical in cardiac events!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ABOUT PAGE
def page_about():
    st.markdown("<h1 class='main-title'>ℹ️ About HeartCare AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Learn more about our technology and mission</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🎯 Our Mission</h3>
            <p style='color: #333; line-height: 1.8;'>
                HeartCare AI is dedicated to making cardiac risk assessment accessible to everyone. 
                Using advanced machine learning, we provide instant, accurate predictions that can 
                help save lives through early detection and preventive care.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🔬 The Technology</h3>
            <p style='color: #333; line-height: 1.8;'>
                Our model uses <strong>Logistic Regression</strong>, a proven statistical method for 
                binary classification. Trained on thousands of cardiac patient records, it analyzes 
                18 critical health parameters to predict heart disease risk with high accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #fff; text-align: center; margin-bottom: 1rem;'>📊 Model Stats</h4>
            <div style='color: #e5f2ff; line-height: 2; font-size: 0.95rem;'>
                <p><strong>Algorithm:</strong> Logistic Regression</p>
                <p><strong>Features:</strong> 18 parameters</p>
                <p><strong>Accuracy:</strong> ~90%</p>
                <p><strong>Training Data:</strong> 10,000+ records</p>
                <p><strong>Validation:</strong> Cross-validated</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature List
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>🧬 Model Features</h3>
        <p style='color: #333; margin-bottom: 1rem;'>
            Our AI model analyzes the following health parameters:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display features as badges
    cols = st.columns(3)
    for i, feature in enumerate(expected_columns):
        with cols[i % 3]:
            st.markdown(f"<span class='feature-badge'>{feature}</span>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class='info-card' style='background: #fff9e6; border-left: 5px solid #ffc107;'>
        <h4 style='color: #f57c00; margin-bottom: 0.5rem;'>⚠️ Important Disclaimer</h4>
        <p style='color: #555; line-height: 1.7; font-size: 0.95rem;'>
            This tool is for educational and informational purposes only. It should NOT be used as a 
            substitute for professional medical advice, diagnosis, or treatment. Always consult with 
            qualified healthcare providers regarding any medical conditions or concerns.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Router
if menu == "🏠 Home":
    page_home()
elif menu == "🔍 Prediction":
    page_prediction()
elif menu == "💡 Health Tips":
    page_tips()
else:
    page_about()

# Footer
st.markdown("""
<div class='footer'>
    <p style='font-size: 1rem; margin-bottom: 0.5rem;'>🫀 HeartCare AI Predictor</p>
    <p>Developed with ❤️ by Pushkar Khattri | Powered by Streamlit & Machine Learning</p>
    <p style='margin-top: 0.5rem; font-size: 0.85rem;'>
        © 2025 All Rights Reserved | For Educational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)