"""
Emotion Detection — Improved Model Training
Improvements:
  - BatchNormalization after every conv block — faster, more stable training
  - More filters (64, 128, 256) — learns richer facial features
  - Validation split so you can see if overfitting during training
  - EarlyStopping — stops automatically when validation stops improving
  - ReduceLROnPlateau — cuts learning rate when stuck, squeezes more accuracy
  - Class weights — forces model to learn Disgust/Fear not just Happy/Neutral
  - Prints full test accuracy + per-class report after training
  - Saves training history chart
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten,
                          Dense, Dropout, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("=" * 50)
print("  Emotion Detection — Model Training")
print("=" * 50)

# ── LOAD DATA ─────────────────────────────────────────────────
X_train = np.load('X_train.npy').reshape(-1, 48, 48, 1)
X_test  = np.load('X_test.npy').reshape(-1, 48, 48, 1)
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

with open('class_weights.json') as f:
    class_weights = {int(k): v for k, v in json.load(f).items()}

# ── MODEL ARCHITECTURE ───────────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Classifier head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── CALLBACKS ────────────────────────────────────────────────
callbacks = [
    # Stop training when validation accuracy stops improving for 10 epochs
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, verbose=1),
    # Halve learning rate when validation loss plateaus for 5 epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1),
    # Always keep the best model on disk
    ModelCheckpoint('Emotion_Detection.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
]

# ── TRAIN ────────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,        # 10% of training set used for validation
    class_weight=class_weights,  # compensate for class imbalance
    callbacks=callbacks,
    verbose=1,
)

# ── EVALUATE ─────────────────────────────────────────────────
print("\nEvaluating on test set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Test Loss:     {loss:.4f}")

y_pred  = model.predict(X_test, verbose=0).argmax(axis=1)
y_true  = y_test.argmax(axis=1)
print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# ── TRAINING HISTORY CHART ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Training History', fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: training_history.png")
print("Model saved: Emotion_Detection.h5")
print("Training complete!")
