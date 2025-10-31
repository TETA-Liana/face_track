"""
face_track_speed.py
Detects a face from webcam, draws bounding box, shows movement direction (Left/Right/Up/Down)
and estimated speed in pixels/second.

Requirements:
    pip install opencv-python

Run:
    python face_track_speed.py
"""

import cv2
import time
import math
from collections import deque

# Parameters (tweak if needed)
MIN_FACE_SIZE = (60, 60)       # minimum face size to consider
SMOOTHING_WINDOW = 6          # frames to average for smoother direction/speed
DIRECTION_THRESHOLD = 6       # pixels — minimum delta to consider as movement
SPEED_SMOOTHING = 0.6         # EMA factor for speed smoothing (0..1)

# Load Haar cascade (path included with OpenCV)
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Check your OpenCV install.")

# Helper: centroid from bounding box
def bbox_centroid(bbox):
    x, y, w, h = bbox
    return (int(x + w/2), int(y + h/2))

# Helper: compute direction string from dx, dy
def get_direction(dx, dy, thresh=DIRECTION_THRESHOLD):
    dir_x = ""
    dir_y = ""

    if abs(dx) >= thresh:
        dir_x = "Right" if dx > 0 else "Left"
    if abs(dy) >= thresh:
        dir_y = "Down" if dy > 0 else "Up"

    if dir_x and dir_y:
        return f"{dir_x}-{dir_y}"
    elif dir_x:
        return dir_x
    elif dir_y:
        return dir_y
    else:
        return "Still"

# webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try a different camera index or check permissions.")

# For timing and smoothing
prev_time = time.time()
prev_centroid = None
centroid_history = deque(maxlen=SMOOTHING_WINDOW)  # store tuples (dx, dy, dt)
ema_speed = None  # exponential moving average for speed

# main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        now = time.time()
        dt = now - prev_time if prev_time else 0.0
        prev_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (scaleFactor and minNeighbors tuned for speed/accuracy)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE
        )

        # Choose the largest face (most likely the user)
        face = None
        if len(faces) > 0:
            # pick the face with largest area
            face = max(faces, key=lambda b: b[2] * b[3])

            (x, y, w, h) = face
            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 200, 0), 2)

            # centroid
            centroid = bbox_centroid(face)
            cv2.circle(frame, centroid, 3, (0, 255, 255), -1)

            # compute delta from previous centroid
            if prev_centroid is not None and dt > 0:
                dx = centroid[0] - prev_centroid[0]
                dy = centroid[1] - prev_centroid[1]

                # instantaneous speed (pixels per second)
                inst_speed = math.hypot(dx, dy) / dt

                # smooth speed using EMA
                if ema_speed is None:
                    ema_speed = inst_speed
                else:
                    ema_speed = SPEED_SMOOTHING * inst_speed + (1 - SPEED_SMOOTHING) * ema_speed

                # store history for smoother direction / noise reduction
                centroid_history.append((dx, dy, dt))
            else:
                dx = dy = 0
                inst_speed = 0.0
                # do not update ema_speed here

            prev_centroid = centroid

            # Compute aggregated dx,dy across history to decide direction
            sum_dx = sum(item[0] for item in centroid_history) if centroid_history else 0
            sum_dy = sum(item[1] for item in centroid_history) if centroid_history else 0

            direction = get_direction(sum_dx, sum_dy)

            # overlay text: direction and speed
            speed_text = f"Speed: {0 if ema_speed is None else int(ema_speed)} px/s"
            dir_text = f"Direction: {direction}"

            # Put texts on frame
            cv2.putText(frame, dir_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, speed_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)
            # also show raw centroid coordinates
            cv2.putText(frame, f"Centroid: {prev_centroid}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        else:
            # No face detected: show status and slowly decay EMA
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2, cv2.LINE_AA)
            # optionally decay ema_speed to zero to reflect no motion
            if ema_speed is not None:
                ema_speed = ema_speed * 0.95
                cv2.putText(frame, f"Speed: {int(ema_speed)} px/s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)

            prev_centroid = None
            centroid_history.clear()

        # show frame
        cv2.imshow("Face Track & Speed", frame)

        # quit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
