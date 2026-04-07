import os
import json
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import serial

# ========== SERIAL (Arduino) CONNECTION ==========
# Change COM port if needed (COM3, COM4, etc.)
ser = serial.Serial("COM5", 9600)
time.sleep(2)  # allow Arduino to reset


# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cashew_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

IMG_SIZE = (160, 160)
CAMERA_SOURCE = 0
PREDICTION_INTERVAL = 0.5


# ========== SERVO SIGNAL FUNCTION ==========
def send_servo_signal(label):
    """
    Sends 'G' or 'B' to Arduino based on cashew grade.
    """
    if label == "good":
        ser.write(b'G')
        print("[SERVO] Sent G (GOOD → LEFT)")
    else:
        ser.write(b'B')
        print("[SERVO] Sent B (BAD → RIGHT)")


# ========== LOAD MODEL ==========
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Load class mapping
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
print("Class mapping:", index_to_class)


# ========== VIDEO CAPTURE ==========
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check CAMERA_SOURCE.")

last_prediction_time = 0
last_label = None

print("Starting real-time cashew sorting. Press 'q' to quit.")


# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    display_frame = frame.copy()
    current_time = time.time()

    if current_time - last_prediction_time >= PREDICTION_INTERVAL:
        img = cv2.resize(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img_rgb, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        prob = float(preds[0][0])

        label_index = 1 if prob >= 0.5 else 0
        label = index_to_class[label_index]

        last_prediction_time = current_time
        last_label = label

        print(f"Prediction: {label} (prob={prob:.2f})")

        send_servo_signal(label)
    # Draw GOOD / BAD text
    if last_label is not None:
        text = last_label.upper()
        color = (0, 255, 0) if last_label == "good" else (0, 0, 255)
        cv2.putText(display_frame, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Cashew Grading (GOOD / BAD)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
