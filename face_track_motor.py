"""
face_track_motor.py
Tracks a face using webcam and rotates a stepper motor (via Arduino) Left or Right
based on detected face movement direction.

Requirements:
    pip install opencv-python pyserial
Run:
    python face_track_motor.py
"""

import cv2
import time
import math
import serial
from collections import deque

# === Serial Setup ===
# Change COM port to your Arduino’s (check Arduino IDE -> Tools -> Port)
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # wait for connection

# === Tracking Parameters ===
MIN_FACE_SIZE = (60, 60)
SMOOTHING_WINDOW = 6
DIRECTION_THRESHOLD = 6
SPEED_SMOOTHING = 0.6

# === Load Face Detector ===
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade")

# === Helper Functions ===
def bbox_centroid(bbox):
    x, y, w, h = bbox
    return (int(x + w/2), int(y + h/2))

def get_direction(dx, thresh=DIRECTION_THRESHOLD):
    if abs(dx) >= thresh:
        return "R" if dx > 0 else "L"
    else:
        return "S"

# === Webcam Capture ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = time.time()
prev_centroid = None
centroid_history = deque(maxlen=SMOOTHING_WINDOW)
ema_speed = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_time if prev_time else 0
        prev_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=MIN_FACE_SIZE)

        if len(faces) > 0:
            face = max(faces, key=lambda b: b[2] * b[3])
            (x, y, w, h) = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 200, 0), 2)
            centroid = bbox_centroid(face)
            cv2.circle(frame, centroid, 3, (0, 255, 255), -1)

            if prev_centroid is not None and dt > 0:
                dx = centroid[0] - prev_centroid[0]
                dy = centroid[1] - prev_centroid[1]

                inst_speed = math.hypot(dx, dy) / dt
                if ema_speed is None:
                    ema_speed = inst_speed
                else:
                    ema_speed = SPEED_SMOOTHING * inst_speed + (1 - SPEED_SMOOTHING) * ema_speed

                centroid_history.append((dx, dt))
            else:
                dx = 0

            prev_centroid = centroid

            sum_dx = sum(item[0] for item in centroid_history)
            direction = get_direction(sum_dx)

            # Send command to Arduino
            if direction in ['L', 'R']:
                arduino.write(direction.encode())

            # Display info
            cv2.putText(frame, f"Direction: {'Left' if direction=='L' else 'Right' if direction=='R' else 'Still'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed: {0 if ema_speed is None else int(ema_speed)} px/s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            prev_centroid = None
            centroid_history.clear()

        cv2.imshow("Face Tracker + Motor Control", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

except KeyboardInterrupt:
    print("Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
