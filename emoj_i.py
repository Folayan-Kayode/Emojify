from fer.fer import FER
import cv2
from fer.utils import draw_annotations
import numpy as np

capture = cv2.VideoCapture(0)
detector = FER(mtcnn=True)



while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    result = detector.detect_emotions(frame)

    if not ret:
        print('Webcam unable to open!')

    for emotions in result:
        x, y, w, h = emotions['box']
        expression = emotions['emotions']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        emotion = max(expression, key=expression.get)
        cv2.putText(frame, emotion,(x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Webcam", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()