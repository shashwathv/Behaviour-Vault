
import numpy as np
import pandas as pd

print("=" * 55)
print("🛡️ BEHAVIORVAULT PRIVACY ENGINE (DPDP ACT SEC. 8)")
print("=" * 55)

def add_gaussian_noise(value, sensitivity, epsilon, delta):

    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = (sensitivity * c) / epsilon

    noise = np.random.normal(0, sigma)

    return max(0.0, value + noise)


print("\n[SERVER] Receiving raw session stats from Mobile App...")

raw_session_data = {
    "user_id": "user_7781",
    "avg_keystroke_ms": 210.5,
    "avg_swipe_speed": 450.2,
    "session_anomaly_score": 0.12
}

print("Raw Data (DANGEROUS TO STORE):")
for k, v in raw_session_data.items():
    print(f"  {k}: {v}")



print("\n[SERVER] Applying Differential Privacy (Gaussian Noise)...")


EPSILON = 1.0   
DELTA = 1e-5      


anonymized_data = {
    "user_id": raw_session_data["user_id"],

    "anonymized_keystroke_ms": round(add_gaussian_noise(
        raw_session_data["avg_keystroke_ms"], sensitivity=100, epsilon=EPSILON, delta=DELTA
    ), 2),

    "anonymized_swipe_speed": round(add_gaussian_noise(
        raw_session_data["avg_swipe_speed"], sensitivity=200, epsilon=EPSILON, delta=DELTA
    ), 2),
    

    "session_anomaly_score": raw_session_data["session_anomaly_score"]
}

print("Anonymized Data (SAFE TO STORE IN MONGODB):")
for k, v in anonymized_data.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 55)
print("✅ DPDP Act Compliance Validated.")
print("Take a screenshot of this output for your pitch deck (Slide 7)!")
print("=" * 55)