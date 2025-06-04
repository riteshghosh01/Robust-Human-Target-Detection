import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from collections import deque, defaultdict

st.set_page_config(page_title="Abnormal Activity Detection", layout="wide")
# Load the model once
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #00ffff;
        text-shadow: 1px 1px 2px #000;
    }

    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        background-color: #00cccc;
        color: black;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #00ffff;
        color: black;
        transform: scale(1.05);
    }

    .css-1aumxhk {
        background: rgba(255, 255, 255, 0.05); /* transparent cards for sidebar */
    }

    .css-1d391kg {  /* sidebar background */
        background: linear-gradient(to bottom right, #111, #222);
    }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    return YOLO("bestt.pt")

model = load_model()
class_names = model.names

# Streamlit UI
#st.set_page_config(layout="wide")
#st.title("🚨 Abnormal Activity Detection Dashboard")
st.markdown("""
    <h1 style='
        font-family: Orbitron, sans-serif;
        font-size: 48px;
        text-align: center;
        color: #00ffff;
        text-shadow: 2px 2px 4px #000000;
        margin-top: 10px;
        margin-bottom: 30px;
    '>
        🚨 Abnormal Activity Detection Dashboard
    </h1>
""", unsafe_allow_html=True)
st.markdown("Upload a video and click **Start Detection** to analyze abnormal human activities using YOLO.")

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
start_button = st.button("▶️ Start Detection")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# Temporary video path
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
else:
    video_path = "sample.mp4"  # default fallback video

# Detection logic
if start_button:
    cap = cv2.VideoCapture(video_path)
    count = 0
    prev_boxes = []
    prev_gray = None
    motion_history = deque(maxlen=30)
    trajectories = defaultdict(list)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    abnormal_threshold = 0.3
    proximity_threshold = 100

    def analyze_crowd_behavior(frame):
        fgMask = backSub.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        motion_pixels = cv2.countNonZero(fgMask)
        total_pixels = frame.shape[0] * frame.shape[1]
        crowd_density = motion_pixels / total_pixels
        return crowd_density, fgMask

    def is_activity_abnormal(activity, crowd_density):
        always_abnormal = {"fighting", "robbery", "armed", "lying_down"}
        return activity in always_abnormal or crowd_density > abnormal_threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crowd_density, fg_mask = analyze_crowd_behavior(frame)

        results = model.predict(frame)
        current_boxes = []
        abnormal_activities = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls)
                class_label = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                is_abnormal = is_activity_abnormal(class_label, crowd_density)

                color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                label = f"{class_label} ({'ABNORMAL' if is_abnormal else 'Normal'})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_abnormal:
                    abnormal_activities.append(class_label)

                current_boxes.append((x1, y1, x2, y2, class_label))

        overall_status = "ABNORMAL" if abnormal_activities else "NORMAL"
        status_color = (0, 0, 255) if overall_status == "ABNORMAL" else (0, 255, 0)

        cv2.putText(frame, f"Density: {crowd_density:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {overall_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Streamlit image display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)#use_column_width=True)

    cap.release()
    st.success("✅ Detection completed!")
