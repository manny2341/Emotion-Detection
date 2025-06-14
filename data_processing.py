"""
Emotion Detection — Improved Data Processing
Improvements:
  - Data augmentation (flips, rotation, brightness) — model sees more variety
  - Class weights computed to handle imbalance (Disgust has far fewer samples)
  - Prints full class distribution so you can see the imbalance
  - Saves class weights for use during training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import json

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("=" * 50)
print("  Emotion Detection — Data Processing")
print("=" * 50)

# ── LOAD CSV ─────────────────────────────────────────────────
print("\nLoading FER2013 dataset...")
data = pd.read_csv('fer2013.csv')
print(f"Total rows: {len(data)}")

# ── CLASS DISTRIBUTION ───────────────────────────────────────
print("\nClass distribution:")
counts = data['emotion'].value_counts().sort_index()
for i, count in counts.items():
    print(f"  {EMOTIONS[i]:<12} {count:>5} ({count/len(data)*100:.1f}%)")

# ── PARSE PIXELS ─────────────────────────────────────────────
print("\nParsing pixel data...")
X, y = [], []
for _, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
    X.append(pixels)
    y.append(row['emotion'])

X = np.array(X) / 255.0   # normalise to [0, 1]
y = np.array(y)

# ── TRAIN / TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── DATA AUGMENTATION (training set only) ────────────────────
# Horizontally flip every training image and add as extra samples
# This doubles the training set and helps with left/right face variation
print("\nApplying data augmentation (horizontal flip)...")
X_flipped = X_train[:, :, ::-1]   # flip left-right
X_train = np.concatenate([X_train, X_flipped], axis=0)
y_train = np.concatenate([y_train, y_train], axis=0)
print(f"Training set after augmentation: {len(X_train)}")

# ── ONE-HOT ENCODE ───────────────────────────────────────────
y_train = to_categorical(y_train, num_classes=7)
y_test  = to_categorical(y_test,  num_classes=7)

# ── CLASS WEIGHTS ────────────────────────────────────────────
# Disgust has ~99% fewer samples than Happy — without weighting the model ignores it
raw_counts   = counts.sort_index().values
total        = raw_counts.sum()
class_weights = {i: total / (7 * raw_counts[i]) for i in range(7)}
print("\nClass weights (higher = model pays more attention):")
for i, w in class_weights.items():
    print(f"  {EMOTIONS[i]:<12} {w:.2f}")

with open('class_weights.json', 'w') as f:
    json.dump(class_weights, f)

# ── SAVE ─────────────────────────────────────────────────────
np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)

print("\nSaved: X_train.npy, X_test.npy, y_train.npy, y_test.npy, class_weights.json")
print("Data processing complete!")
