import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load trained model
model = load_model("landmark_model.h5")

# Load labels (A–Z)
labels = sorted(os.listdir("LandmarkDataset"))
labels_dict = {i: label for i, label in enumerate(labels)}

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_predicted_label = ""
last_spoken_time = 0
displayed_letter = ""
displayed_time = 0

CONFIDENCE_THRESHOLD = 0.85
DISPLAY_DURATION = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            # Get hand bounding box
            h, w, _ = frame.shape
            landmarks = hand_landmark.landmark
            x_vals = [int(lm.x * w) for lm in landmarks]
            y_vals = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            # Draw landmarks on webcam frame
            mp_draw.draw_landmarks(
                frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            # Draw bounding box
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)

            # Also draw landmarks on a white canvas for model input
            white = 255 * np.ones((400, 400, 3), dtype=np.uint8)
            mp_draw.draw_landmarks(
                white, hand_landmark, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            # Preprocess input
            resized = cv2.resize(white, (64, 64))
            normalized = resized / 255.0
            reshaped = np.expand_dims(normalized, axis=0)

            # Predict letter
            prediction = model.predict(reshaped)[0]
            class_index = np.argmax(prediction)
            confidence = prediction[class_index]

            if confidence > CONFIDENCE_THRESHOLD:
                predicted_label = labels_dict[class_index]
                current_time = time.time()

                # Speak and display if new or time passed
                if predicted_label != last_predicted_label or current_time - last_spoken_time > DISPLAY_DURATION:
                    engine.say(predicted_label)
                    engine.runAndWait()
                    last_spoken_time = current_time
                    last_predicted_label = predicted_label
                    displayed_letter = predicted_label
                    displayed_time = current_time

    # Keep predicted letter visible for 5 seconds
    if displayed_letter and (time.time() - displayed_time <= DISPLAY_DURATION):
        cv2.putText(frame, f"Prediction: {displayed_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Real-Time Sign Recognition (Full View)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
