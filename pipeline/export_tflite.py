"""
SCRIPT 3 — Export to TFLite (Surrogate Model Workaround)
=========================================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

PATHS:
  Reads:  models/isolation_forest.pkl
          models/scaler.pkl
  Writes: models/behavior_model.tflite

React Native (TFLite) cannot run scikit-learn models natively. We train a
small Keras neural network to mimic the Isolation Forest's predictions,
then export that as .tflite for mobile inference.

HOW TO RUN:
  python pipeline/export_tflite.py
"""

import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ISO_FOREST_PKL = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PKL     = os.path.join(MODELS_DIR, "scaler.pkl")
TFLITE_OUT     = os.path.join(MODELS_DIR, "behavior_model.tflite")

# ── Step 1 — Load sklearn artifacts ─────────────────────────────────────────
print("=" * 55)
print("STEP 1 — Loading sklearn artifacts")
print("=" * 55)

if not (os.path.exists(ISO_FOREST_PKL) and os.path.exists(SCALER_PKL)):
    raise FileNotFoundError(
        "Required .pkl files not found. Run pipeline/train_model.py first."
    )

with open(ISO_FOREST_PKL, "rb") as f:
    iso_forest = pickle.load(f)
with open(SCALER_PKL, "rb") as f:
    scaler = pickle.load(f)
print(f"  ✅  Loaded {ISO_FOREST_PKL}")
print(f"  ✅  Loaded {SCALER_PKL}")

# ── Step 2 — Generate surrogate training data ───────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Generating surrogate dataset (uniform in scaled space)")
print("=" * 55)

X_random_scaled = np.random.uniform(low=-4.0, high=4.0, size=(50000, 5))
raw_preds = iso_forest.predict(X_random_scaled)
y_surrogate = (raw_preds == -1).astype(np.float32)
print(f"  Total samples : {len(X_random_scaled)}")
print(f"  Anomalies     : {int(y_surrogate.sum())}")

X_train, X_test, y_train, y_test = train_test_split(
    X_random_scaled, y_surrogate, test_size=0.2, random_state=42
)

# ── Step 3 — Train Keras surrogate ──────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — Training Keras surrogate")
print("=" * 55)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("  Training neural network...")
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test), verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"  ✅  Surrogate accuracy: {accuracy * 100:.2f}%")

# ── Step 4 — Convert and save TFLite ────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — Converting to TFLite")
print("=" * 55)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_OUT) / 1024
print(f"  ✅  {TFLITE_OUT}  ({size_kb:.1f} KB)")
print("\nNEXT STEP: python pipeline/get_scalar_value.py")