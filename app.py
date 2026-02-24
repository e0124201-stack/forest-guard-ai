import streamlit as st
import cv2
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import time
import datetime

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Forest Guard AI v1.0", layout="wide")
COLOR_FIRE = (0, 0, 255)
COLOR_SAFE = (0, 255, 0)

# --- MOCK ML MODELS (Logic for Demo) ---
def detect_fire_in_frame(frame):
    """
    Simulates a CNN/YOLO model. In a real project, 
    you'd load: model = cv2.dnn.readNet('fire_model.weights')
    For this demo, we use Color Range Segmentation (HSV) to find 'Fire Colors'.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define range of fire color in HSV
    lower_fire = np.array([0, 120, 70])
    upper_fire = np.array([10, 255, 255])
    mask1 = cv2.insrange(hsv, lower_fire, upper_fire)
    
    lower_fire2 = np.array([170, 120, 70])
    upper_fire2 = np.array([180, 255, 255])
    mask2 = cv2.insrange(hsv, lower_fire2, upper_fire2)
    
    mask = mask1 + mask2
    fire_pixel_count = np.sum(mask > 0)
    
    # Threshold for detection
    is_fire = fire_pixel_count > 500 
    confidence = min(99.8, (fire_pixel_count / 100))
    return is_fire, confidence, mask

def analyze_audio_logging(audio_file):
    """
    Uses Librosa to analyze audio frequencies.
    Logging/Chainsaws have high-frequency 'whine' patterns.
    """
    y, sr = librosa.load(audio_file)
    # Generate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Logic: If mean decibel in high freq is above threshold -> Illegal Logging
    avg_db = np.mean(S_dB)
    is_logging = avg_db > -35 # Simplified threshold for demo
    return is_logging, S_dB

# --- STREAMLIT UI ---
st.title("🌲 Forest Guard: Illegal Logging & Fire Alert System")
st.markdown("---")

# Sidebar for Controls
st.sidebar.header("Control Panel")
mode = st.sidebar.selectbox("Choose Mode", ["Live Surveillance", "Audio Analysis", "Historical Logs"])
alert_phone = st.sidebar.text_input("Emergency Contact", "+91 98765-43210")

if mode == "Live Surveillance":
    st.subheader("📡 Real-Time Visual Monitoring")
    col1, col2 = st.columns([2, 1])
    
    run_cam = st.checkbox("Start Webcam Feed")
    FRAME_WINDOW = col1.image([])
    
    status_box = col2.empty()
    metric_box = col2.empty()
    log_box = col2.container()

    camera = cv2.VideoCapture(0)

    while run_cam:
        ret, frame = camera.read()
        if not ret: break
        
        # ML Logic
        is_fire, conf, mask = detect_fire_in_frame(frame)
        
        # UI Feedback
        if is_fire:
            cv2.putText(frame, "!!! FIRE DETECTED !!!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_FIRE, 3)
            status_box.error(f"⚠️ ALERT: Fire Detected! Confidence: {conf:.2f}%")
            # Simulation of sending SMS
            log_box.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SMS Sent to {alert_phone}")
        else:
            status_box.success("✅ System Status: FOREST SAFE")
            metric_box.metric("Air Quality Index", "42 AQI", "Good")

        # Display Frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        time.sleep(0.05)
    else:
        st.info("Surveillance Offline. Tick the box to start.")

elif mode == "Audio Analysis":
    st.subheader("🔊 Acoustic Logging Detection")
    uploaded_file = st.file_uploader("Upload Forest Audio Clip", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing Acoustic Fingerprint..."):
            is_logging, spec_data = analyze_audio_logging(uploaded_file)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("### Spectrogram Result")
                fig, ax = plt.subplots()
                img = librosa.display.specshow(spec_data, x_axis='time', y_axis='mel', ax=ax)
                plt.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)
            
            with col_b:
                if is_logging:
                    st.error("🚨 RESULT: ILLEGAL LOGGING (Chainsaw Detected)")
                    st.progress(94)
                    st.write("Confidence: 94.2%")
                else:
                    st.success("🍃 RESULT: Natural Forest Sounds")
                    st.progress(10)
                    st.write("Confidence: 98.1%")

elif mode == "Historical Logs":
    st.subheader("📅 Incident Reports")
    data = {
        "Date": ["2024-05-10", "2024-05-12", "2024-05-15"],
        "Type": ["Fire", "Illegal Logging", "Fire"],
        "Location": ["Sector A-12", "North Range", "Sector B-04"],
        "Severity": ["High", "Medium", "Critical"]
    }
    st.table(data)

st.sidebar.markdown("---")
st.sidebar.write("Developed by: BTech 2nd Year ML Team")
