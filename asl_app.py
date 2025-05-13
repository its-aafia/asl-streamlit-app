import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt

# Streamlit page config
st.set_page_config(page_title="🧠 ASL Real-time Recognizer", layout="centered")

# Load model and labels
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('asl_model.h5')

@st.cache_data
def load_labels():
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space']

model = load_model()
labels = load_labels()

# Constants
img_size = 64

# Title
st.title("🤖 Real-time ASL Sign Language Detection")
st.markdown("Show a hand sign in front of your webcam. The model will predict it in real-time!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam start
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
stop_btn = st.button("🛑 Stop Webcam")

while cap.isOpened() and not stop_btn:
    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Could not read from webcam.")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Get bounding box
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for landmark in results.multi_hand_landmarks[0].landmark:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        hand_crop = frame[y_min:y_max, x_min:x_max]

        # Preprocess
        try:
            resized = cv2.resize(hand_crop, (img_size, img_size))
            normalized = resized / 255.0
            reshaped = np.expand_dims(normalized, axis=0)  # Shape: (1, 64, 64, 3)

            # Predict
            prediction = model.predict(reshaped, verbose=0)
            class_idx = np.argmax(prediction)
            class_label = labels[class_idx]
            confidence = prediction[0][class_idx]

            if confidence > 0.7:
                cv2.putText(frame, f"{class_label} ({confidence*100:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Low confidence", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Show bar chart
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(labels, prediction[0], color='skyblue')
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)

        except Exception as e:
            cv2.putText(frame, "Hand too small/large", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)

    # Display frame
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Cleanup
cap.release()
st.success("✅ Webcam stopped.")
