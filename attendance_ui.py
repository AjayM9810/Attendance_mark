import tensorflow as tf
import streamlit as st
import base64
import cv2
import pickle
import pandas as pd
import numpy as np
from mark_excel import predict_face, log_attendance, init_csv, update_break_time
from datetime import datetime
import requests

url = "https://github.com/AjayM9810/Attendance_mark/fine_tuned_model.h5"
r = requests.get(url)
with open("real_names.pkl", "rb") as f:
    class_names = pickle.load(f)
model = tf.keras.models.load_model("fine_tuned_model.h5")

def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img_base64 = get_base64_of_image("light_purple_sky_above_beach_rock_4k_hd_nature-1920x1080.jpg")

st.markdown(
    f"""
    <style>
    /*Background image*/
    .stApp{{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        position: relative;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.3);  /* Light overlay */
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* Title styling */
    h1 {{
        color: white;
        font-size: 2.5em;
        font-weight: 800;
        text-align: center;
        text-shadow: 1px 1px 4px black;

    }}    

    button[data-testid="start"] {{
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 12em;
        font-size: 16px;
        font-weight: bold;
    }}
    button[data-testid="start"]:hover {{
        background-color: #45a049;
    }}

    button[data-testid="stop"] {{
        background-color: #f44336;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 12em;
        font-size: 16px;
        font-weight: bold;
    }}
    button[data-testid="stop"]:hover {{
        background-color: #da190b;
    }}
    div.stAlert {{
        background-color: transparent;  /* Dark semi-transparent background */
        color: #4CAF50;
        font-weight: bold;
        border-radius: 0;
    p   adding: 0.5em;
    }}
    </style>
    """,
    unsafe_allow_html=True)


st.title("📸 Face Recognition Attendance System")

file = init_csv("attendance.csv")
start_button = st.button("▶️ Mark Attendance")
stop_button = st.button("⏹️ Quit")
FRAME = st.image([])

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "cap" not in st.session_state:
    st.session_state.cap = None

if start_button and not st.session_state.camera_running:
    st.session_state.camera_running = True
    # st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.cap = "browser"

if stop_button and st.session_state.camera_running:
    st.session_state.camera_running = False
    if isinstance (st.session_state.cap, cv2.VideoCapture):
        st.session_state.cap.release()
    st.session_state.cap = None

if st.session_state.camera_running and st.session_state.cap == "browser":
    img_file = st.camera_input("📷 Capture your face", label_visibility = "collapsed")
    if img_file is not None:
        bytes_data = img_file.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # ret, frame = st.session_state.cap.read()
    # if not ret:
    #     st.error('Camera not available')
    # else:
        label, confidence = predict_face(frame)

        if label is not None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            st.info(f"ℹ️Detected: {label} at {time_str} ({confidence:.2f})")
            st.warning("⚠️Please press ✅Confirm to mark or press 🔄Recapture to try again")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm"):
                    log_attendance(label, confidence,file)
                    status = "Login" if "09:00" <= time_str[:5] <= "09:30" else "Logout" if time_str[:5]>= "18:00" else "Break"
                    st.success(f"{status} for {label} at {time_str}")
            # with col2:
                # if st.button("🔄 Recapture"):
                    # st.warning("Please Confirm if Verified✅")
        # FRAME.image(frame, channels= "BGR")
# else:
#     st.info("Camera Stopped")

if st.button("📊 Finalize Day"):
    update_break_time(file)

    st.info("Break time updated")







