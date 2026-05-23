import pickle
import json

# 1. Load the fitted scaler from Script 2
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2. Extract and print the exact arrays Shashank needs
export_data = {
    "mean": [round(x, 4) for x in scaler.mean_.tolist()],
    "std":  [round(x, 4) for x in scaler.scale_.tolist()]
}

print("\n--- COPY AND PASTE THIS TO SHASHANK ---")
print(json.dumps(export_data, indent=2))
print("---------------------------------------")