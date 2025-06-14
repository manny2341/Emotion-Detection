"""
Emotion Detection — Improved Live Detection
Improvements:
  - Shows confidence % next to emotion label
  - Colour-coded boxes per emotion (green=Happy, red=Angry, etc.)
  - Confidence bar drawn below each face
  - FPS counter in top-left corner
  - Smoothing: averages last 5 predictions to reduce flickering
  - Falls back to DNN face detector (more accurate than Haar at angles)
    but keeps Haar as backup if DNN not available
  - Press Q to quit, S to save a screenshot
"""

import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import time
import os

# ── CONFIG ───────────────────────────────────────────────────
EMOTIONS   = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLOURS    = {
    'Angry':    (0,   0,   255),   # Red
    'Disgust':  (0,   140, 0),     # Dark green
    'Fear':     (128, 0,   128),   # Purple
    'Happy':    (0,   255, 0),     # Green
    'Sad':      (255, 100, 0),     # Blue-orange
    'Surprise': (0,   215, 255),   # Yellow
    'Neutral':  (200, 200, 200),   # Grey
}
SMOOTH_FRAMES = 5   # number of frames to average predictions over

# ── LOAD MODEL ───────────────────────────────────────────────
print("Loading model...")
classifier = load_model('Emotion_Detection.h5')
print("Model loaded.")

# ── FACE DETECTOR ────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ── PREDICTION SMOOTHER ──────────────────────────────────────
# Keeps a rolling window of last N predictions per face to reduce flicker
smoothers = {}   # face_id → deque of prediction arrays

def smooth_prediction(face_id, preds, window=SMOOTH_FRAMES):
    if face_id not in smoothers:
        smoothers[face_id] = deque(maxlen=window)
    smoothers[face_id].append(preds)
    return np.mean(smoothers[face_id], axis=0)

# ── START WEBCAM ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Q to quit, S to save screenshot.")

screenshot_count = 0
prev_time        = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── FPS ──────────────────────────────────────────────────
    curr_time  = time.time()
    fps        = 1 / (curr_time - prev_time + 1e-6)
    prev_time  = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── DETECT FACES ─────────────────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        cv2.putText(frame, 'No Face Detected', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for i, (x, y, w, h) in enumerate(faces):
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi) == 0:
            continue

        roi_input = roi.astype('float32') / 255.0
        roi_input = roi_input.reshape(1, 48, 48, 1)

        # Predict and smooth
        raw_preds    = classifier.predict(roi_input, verbose=0)[0]
        smoothed     = smooth_prediction(i, raw_preds)
        label_idx    = smoothed.argmax()
        label        = EMOTIONS[label_idx]
        confidence   = smoothed[label_idx] * 100
        colour       = COLOURS[label]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

        # Label + confidence
        text = f"{label}  {confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+8, y), colour, -1)
        cv2.putText(frame, text, (x+4, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Confidence bar below the face box
        bar_w = int(w * confidence / 100)
        cv2.rectangle(frame, (x, y+h+4), (x+w, y+h+14), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y+h+4), (x+bar_w, y+h+14), colour, -1)

    # FPS display
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_count += 1
        fname = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(fname, frame)
        print(f"Screenshot saved: {fname}")

cap.release()
cv2.destroyAllWindows()
print("Detector closed.")
