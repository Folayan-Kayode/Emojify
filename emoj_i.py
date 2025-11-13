import time
import streamlit as st
import cv2
import numpy as np
from fer.fer import FER

st.set_page_config(page_title="Facial Emotion Detector", page_icon="🎭", layout="centered")
st.title("🎭 Real-Time Facial Emotion Detector")
st.write("Detect emotions from your webcam in real time using FER and OpenCV.")

detector = FER(mtcnn=True)

if 'running' not in st.session_state:
    st.session_state.running = False

# Toggle control
st.session_state.running = st.checkbox("Start Webcam", value=st.session_state.running)

FRAME_WINDOW = st.image([])

if st.session_state.running:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # recommended on Windows
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible!")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = detector.detect_emotions(rgb)

            for faces in result:
                x, y, w, h = map(int, faces["box"])
                expression = faces["emotions"]
                emotion = max(expression, key=expression.get)

                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(rgb, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            FRAME_WINDOW.image(rgb)

            # small sleep to avoid a tight busy loop; Streamlit UI still limited for long loops
            time.sleep(0.03)
    finally:
        cap.release()
else:
    st.info("Check the box above to start the webcam.")
