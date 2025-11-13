import streamlit as st
import cv2
import numpy as np
from fer.fer import FER

st.set_page_config(page_title="Facial Emotion Detector", page_icon="🎭", layout="centered")
st.title("🎭 Real-Time Facial Emotion Detector")
st.write("Detect emotions from your webcam in real time using FER and OpenCV.")

# Initialize FER detector
detector = FER(mtcnn=True)

# Checkbox to start webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible!")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)

        result = detector.detect_emotions(frame)

        # Draw boxes and labels
        for emotions in result:
            (x, y, w, h) = emotions["box"]
            expression = emotions["emotions"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            emotion = max(expression, key=expression.get)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("Check the box above to start the webcam.")
