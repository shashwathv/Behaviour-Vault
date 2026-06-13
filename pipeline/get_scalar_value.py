"""
SCRIPT 5 — Extract scaler MEAN / STD for the API
=================================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

PATHS:
  Reads:  models/scaler.pkl

Prints the exact MEAN and STD arrays the API needs. Copy these into
api/app.py and replace the MEAN and STD constants there.

HOW TO RUN:
  python pipeline/get_scalar_value.py
"""

import os
import pickle
import json

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCALER_PKL   = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

if not os.path.exists(SCALER_PKL):
    raise FileNotFoundError(
        f"{SCALER_PKL} not found. Run pipeline/train_model.py first."
    )

with open(SCALER_PKL, "rb") as f:
    scaler = pickle.load(f)

export_data = {
    "mean": [round(x, 4) for x in scaler.mean_.tolist()],
    "std":  [round(x, 4) for x in scaler.scale_.tolist()],
}

print("\n" + "=" * 55)
print("  Copy these into api/app.py (MEAN and STD constants):")
print("=" * 55)
print(f"\nMEAN = {export_data['mean']}")
print(f"STD  = {export_data['std']}")
print("\n--- JSON form (for mobile app handoff) ---")
print(json.dumps(export_data, indent=2))
print("------------------------------------------")