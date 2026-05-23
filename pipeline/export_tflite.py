"""
SCRIPT 3 — Export to TFLite (The Surrogate Model Workaround)
=============================================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

WHAT THIS SCRIPT DOES:
----------------------
1. Loads the trained Isolation Forest and Scaler.
2. Generates a large randomized dataset.
3. Uses the Isolation Forest to label this dataset (Normal vs Anomaly).
4. Trains a lightweight TensorFlow Keras Neural Network to mimic those labels.
5. Exports the Keras model as a mobile-ready .tflite file.

WHY WE DO THIS:
---------------
React Native (via TensorFlow Lite) cannot run scikit-learn models natively. 
By training a tiny neural network to "copy" the Isolation Forest, we get the 
exact same decision logic in a format that runs natively on iOS and Android.

OUTPUT:
-------
behavior_model.tflite — Hand this file to Shashank for the mobile app!
"""

import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("=" * 55)
print("STEP 1 — Loading Scikit-Learn Models")
print("=" * 55)

# Load the models trained in Script 2
try:
    with open("isolation_forest.pkl", "rb") as f:
        iso_forest = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("✅ Successfully loaded isolation_forest.pkl and scaler.pkl")
except FileNotFoundError:
    print("❌ ERROR: Could not find .pkl files. Did you run 02_train_model.py?")
    exit()

print("\n" + "=" * 55)
print("STEP 2 — Generating Data for the Surrogate Model")
print("=" * 55)

# Generate 50,000 random samples across the scaled feature space
# We use a uniform distribution to ensure the neural net learns the boundaries
X_random_scaled = np.random.uniform(low=-4.0, high=4.0, size=(50000, 5))

# Have the Isolation Forest label the random data
# predict() returns 1 for normal, -1 for anomaly.
raw_preds = iso_forest.predict(X_random_scaled)

# Convert labels for Neural Network: 0.0 = Normal, 1.0 = Anomaly
y_surrogate = (raw_preds == -1).astype(np.float32)

print(f"Generated {len(X_random_scaled)} samples for distillation.")
print(f"Anomalies detected in sample space: {int(sum(y_surrogate))}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_random_scaled, y_surrogate, test_size=0.2, random_state=42)

print("\n" + "=" * 55)
print("STEP 3 — Training the TensorFlow Surrogate Model")
print("=" * 55)

# Build a tiny, lightning-fast Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Outputs a score between 0.0 and 1.0
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model silently (verbose=0)
print("Training neural network to mimic Isolation Forest... (this takes a few seconds)")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Check accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Surrogate Model Accuracy: {accuracy * 100:.2f}%")

print("\n" + "=" * 55)
print("STEP 4 — Exporting to TensorFlow Lite")
print("=" * 55)

# Convert the Keras model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for mobile (quantization reduces file size and speeds up inference)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the file
with open('behavior_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ behavior_model.tflite saved successfully!")
print("=" * 55)
