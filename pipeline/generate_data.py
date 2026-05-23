"""
SCRIPT 1 — Generate Synthetic Behavioral Data
==============================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

WHAT THIS SCRIPT DOES:
----------------------
Generates a CSV file of synthetic behavioral sessions.
Each row = one login/transaction session with 5 behavioral signals.

The dataset has two types of rows:
  - label=0 → NORMAL   (the real user's typical behavior)
  - label=1 → ANOMALY  (attacker, coerced user, or bot)

WHY SYNTHETIC DATA:
-------------------
You have no real user data yet. Synthetic data lets you:
  1. Train and validate the model before the real app is built
  2. Control exactly how "different" anomalies look
  3. Test edge cases (e.g. duress-style behavior)

NOTE ON LABELS:
---------------
Isolation Forest is UNSUPERVISED — it does NOT use labels during training.
Labels are only kept here so you can evaluate accuracy later (Script 2).

THE 5 FEATURES:
---------------
1. keystroke_timing_ms   — avg ms between key presses
                           Normal: 150–250ms | Anomaly: <60ms or >700ms
2. touch_pressure        — normalized 0.0–1.0 (iOS force / Android pressure)
                           Normal: 0.2–0.6   | Anomaly: >0.8 (stressed) or <0.1
3. swipe_speed_px_s      — pixels per second on swipe gestures
                           Normal: 300–600   | Anomaly: <100 (hesitant) or >1000
4. scroll_rhythm_ms      — ms between scroll events (consistency)
                           Normal: 100–300   | Anomaly: <30 or >800
5. accel_variance        — accelerometer variance over 5s window (DuressSense)
                           Normal: 0.01–0.1  | Anomaly: >0.5 (shaking hands)

HOW TO RUN:
-----------
  python 01_generate_data.py

OUTPUT:
-------
  behavioral_data.csv   — the full dataset (normal + anomaly rows)
  data_summary.txt      — quick stats so you can verify the data looks right
"""

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
# Setting a seed means you get the SAME data every time you run this.
# Remove or change the seed if you want different random data each run.
SEED = 42
np.random.seed(SEED)

# ── How many sessions to generate ────────────────────────────────────────────
# 1000 normal + 200 anomaly = realistic 17% anomaly rate
# Real-world fraud rates are 1–5%, but 17% gives the model enough anomaly
# examples to evaluate against. Isolation Forest trains on NORMAL only anyway.
N_NORMAL  = 1000
N_ANOMALY = 200


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL USER SESSIONS
# Each signal uses np.random.normal(mean, std_dev, count)
# The mean/std values are based on the signal ranges in your project plan.
# ─────────────────────────────────────────────────────────────────────────────
normal_sessions = pd.DataFrame({

    # Keystroke timing: centered at 200ms, std=30ms → most values 140–260ms
    "keystroke_timing_ms": np.random.normal(200, 30, N_NORMAL).clip(80, 400),

    # Touch pressure: centered at 0.4, std=0.08 → most values 0.24–0.56
    "touch_pressure": np.random.normal(0.4, 0.08, N_NORMAL).clip(0.1, 0.9),

    # Swipe speed: centered at 450 px/s, std=80 → most values 290–610
    "swipe_speed_px_s": np.random.normal(450, 80, N_NORMAL).clip(100, 900),

    # Scroll rhythm: centered at 180ms, std=40ms → most values 100–260ms
    "scroll_rhythm_ms": np.random.normal(180, 40, N_NORMAL).clip(50, 500),

    # Accelerometer variance: low for a steady hand
    "accel_variance": np.random.normal(0.05, 0.02, N_NORMAL).clip(0.01, 0.2),

    # Label: 0 = normal
    "label": 0
})


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALOUS SESSIONS — 3 sub-types mixed together
#
# Type A (40%): ATTACKER — fast typing, different pressure/swipe pattern
# Type B (40%): DURESS   — shaky hands (high accel), hesitant swipes
# Type C (20%): BOT      — unnaturally fast/uniform behavior
# ─────────────────────────────────────────────────────────────────────────────
n_attacker = int(N_ANOMALY * 0.4)   # 80 sessions
n_duress   = int(N_ANOMALY * 0.4)   # 80 sessions
n_bot      = N_ANOMALY - n_attacker - n_duress  # 40 sessions

# Type A — Attacker: different person, unfamiliar with the interface
attacker = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(350, 60, n_attacker).clip(200, 700),
    "touch_pressure":      np.random.normal(0.75, 0.1, n_attacker).clip(0.5, 1.0),
    "swipe_speed_px_s":    np.random.normal(200, 60, n_attacker).clip(80, 400),
    "scroll_rhythm_ms":    np.random.normal(500, 80, n_attacker).clip(300, 900),
    "accel_variance":      np.random.normal(0.15, 0.05, n_attacker).clip(0.05, 0.4),
    "label": 1
})

# Type B — Duress: user is physically coerced → shaking hands, hesitant
duress = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(600, 80, n_duress).clip(400, 900),
    "touch_pressure":      np.random.normal(0.85, 0.08, n_duress).clip(0.7, 1.0),
    "swipe_speed_px_s":    np.random.normal(120, 30, n_duress).clip(50, 220),
    "scroll_rhythm_ms":    np.random.normal(700, 100, n_duress).clip(400, 1000),
    # KEY SIGNAL for DuressSense: high accelerometer variance = shaking hands
    "accel_variance":      np.random.normal(0.65, 0.1, n_duress).clip(0.4, 1.0),
    "label": 1
})

# Type C — Bot: unnaturally fast and uniform (low variance across all signals)
bot = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(55, 5, n_bot).clip(40, 75),
    "touch_pressure":      np.random.normal(0.5, 0.01, n_bot).clip(0.48, 0.52),
    "swipe_speed_px_s":    np.random.normal(1100, 20, n_bot).clip(1000, 1200),
    "scroll_rhythm_ms":    np.random.normal(25, 3, n_bot).clip(15, 40),
    "accel_variance":      np.random.normal(0.005, 0.001, n_bot).clip(0.001, 0.01),
    "label": 1
})


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE + SHUFFLE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.concat([normal_sessions, attacker, duress, bot], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle rows

# Round to sensible decimal places for realism
df["keystroke_timing_ms"] = df["keystroke_timing_ms"].round(1)
df["touch_pressure"]       = df["touch_pressure"].round(4)
df["swipe_speed_px_s"]     = df["swipe_speed_px_s"].round(1)
df["scroll_rhythm_ms"]     = df["scroll_rhythm_ms"].round(1)
df["accel_variance"]        = df["accel_variance"].round(5)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv("behavioral_data.csv", index=False)
print(f"✅  behavioral_data.csv saved — {len(df)} rows total")
print(f"    Normal sessions  : {(df['label']==0).sum()}")
print(f"    Anomaly sessions : {(df['label']==1).sum()}")
print(f"      ↳ Attacker type: {n_attacker}")
print(f"      ↳ Duress type  : {n_duress}")
print(f"      ↳ Bot type     : {n_bot}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE SUMMARY STATS — so you can verify the data distribution looks correct
# ─────────────────────────────────────────────────────────────────────────────
features = ["keystroke_timing_ms", "touch_pressure", "swipe_speed_px_s",
            "scroll_rhythm_ms", "accel_variance"]

with open("data_summary.txt", "w") as f:
    f.write("BEHAVIORVAULT 2.0 — Synthetic Data Summary\n")
    f.write("=" * 50 + "\n\n")

    for label_val, label_name in [(0, "NORMAL"), (1, "ANOMALY")]:
        subset = df[df["label"] == label_val]
        f.write(f"{label_name} sessions ({len(subset)} rows):\n")
        f.write(subset[features].describe().to_string())
        f.write("\n\n")

print("✅  data_summary.txt saved — check this to verify distributions look right")
print("\nNEXT STEP: Run  python 02_train_model.py")