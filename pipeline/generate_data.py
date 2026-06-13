"""
SCRIPT 1 — Generate Synthetic Behavioral Data  (REALISTIC v2)
==============================================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

PATHS:
  Reads:  (nothing)
  Writes: data/behavioral_data.csv
          reports/data_summary.txt

CHANGES FROM v1:
  Normal distribution widened to match real mobile usage:
    keystroke      : mean=600ms, std=200
    touch_pressure : mean=0.50,  std=0.10
    swipe_speed    : mean=450,   std=150
    scroll_rhythm  : mean=179,   std=80
    accel_variance : mean=0.05,  std=0.02

  Anomaly sub-populations pushed clearly outside the new normal envelope
  so the Isolation Forest still gets a clean separation signal.

HOW TO RUN:
  python pipeline/generate_data.py        # from project root
  python generate_data.py                 # from inside pipeline/
"""

import os
import numpy as np
import pandas as pd

# ── Resolve project paths regardless of where the script is invoked from ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_CSV    = os.path.join(DATA_DIR, "behavioral_data.csv")
SUMMARY_TXT = os.path.join(REPORTS_DIR, "data_summary.txt")

SEED = 42
np.random.seed(SEED)

N_NORMAL  = 1000
N_ANOMALY = 200


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL USER SESSIONS — realistic mobile behavior
# ─────────────────────────────────────────────────────────────────────────────
normal_sessions = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(600, 200, N_NORMAL).clip(200, 1100),
    "touch_pressure":      np.random.normal(0.50, 0.10, N_NORMAL).clip(0.2, 0.9),
    "swipe_speed_px_s":    np.random.normal(450, 150, N_NORMAL).clip(150, 900),
    "scroll_rhythm_ms":    np.random.normal(179, 80,  N_NORMAL).clip(50, 500),
    "accel_variance":      np.random.normal(0.05, 0.02, N_NORMAL).clip(0.01, 0.2),
    "label": 0
})


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY SUB-POPULATIONS
# ─────────────────────────────────────────────────────────────────────────────
n_attacker = int(N_ANOMALY * 0.4)   # 80
n_duress   = int(N_ANOMALY * 0.4)   # 80
n_bot      = N_ANOMALY - n_attacker - n_duress  # 40

# Type A — Attacker: faster than legitimate user, harder presses, fast swipes
attacker = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(140, 30, n_attacker).clip(60, 200),
    "touch_pressure":      np.random.normal(0.85, 0.05, n_attacker).clip(0.7, 1.0),
    "swipe_speed_px_s":    np.random.normal(1050, 80, n_attacker).clip(900, 1300),
    "scroll_rhythm_ms":    np.random.normal(550, 60, n_attacker).clip(400, 800),
    "accel_variance":      np.random.normal(0.25, 0.05, n_attacker).clip(0.15, 0.4),
    "label": 1
})

# Type B — Duress: physically coerced, shaking, hesitant
duress = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(1500, 200, n_duress).clip(1200, 2000),
    "touch_pressure":      np.random.normal(0.90, 0.05, n_duress).clip(0.75, 1.0),
    "swipe_speed_px_s":    np.random.normal(100, 30, n_duress).clip(50, 200),
    "scroll_rhythm_ms":    np.random.normal(750, 100, n_duress).clip(500, 1000),
    "accel_variance":      np.random.normal(0.65, 0.10, n_duress).clip(0.4, 1.0),
    "label": 1
})

# Type C — Bot: impossibly fast and unnaturally uniform
bot = pd.DataFrame({
    "keystroke_timing_ms": np.random.normal(45, 5, n_bot).clip(30, 70),
    "touch_pressure":      np.random.normal(0.50, 0.005, n_bot).clip(0.49, 0.51),
    "swipe_speed_px_s":    np.random.normal(1250, 15, n_bot).clip(1200, 1300),
    "scroll_rhythm_ms":    np.random.normal(20, 3, n_bot).clip(15, 35),
    "accel_variance":      np.random.normal(0.003, 0.0005, n_bot).clip(0.001, 0.005),
    "label": 1
})


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE + SHUFFLE + ROUND + SAVE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.concat([normal_sessions, attacker, duress, bot], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

df["keystroke_timing_ms"] = df["keystroke_timing_ms"].round(1)
df["touch_pressure"]       = df["touch_pressure"].round(4)
df["swipe_speed_px_s"]     = df["swipe_speed_px_s"].round(1)
df["scroll_rhythm_ms"]     = df["scroll_rhythm_ms"].round(1)
df["accel_variance"]       = df["accel_variance"].round(5)

df.to_csv(DATA_CSV, index=False)
print(f"✅  {DATA_CSV} — {len(df)} rows total")
print(f"    Normal sessions  : {(df['label']==0).sum()}")
print(f"    Anomaly sessions : {(df['label']==1).sum()}")
print(f"      ↳ Attacker     : {n_attacker}")
print(f"      ↳ Duress       : {n_duress}")
print(f"      ↳ Bot          : {n_bot}")


# ── Summary report ──────────────────────────────────────────────────────────
features = ["keystroke_timing_ms", "touch_pressure", "swipe_speed_px_s",
            "scroll_rhythm_ms", "accel_variance"]

with open(SUMMARY_TXT, "w") as f:
    f.write("BEHAVIORVAULT 2.0 — Synthetic Data Summary (REALISTIC v2)\n")
    f.write("=" * 55 + "\n\n")
    for label_val, label_name in [(0, "NORMAL"), (1, "ANOMALY")]:
        subset = df[df["label"] == label_val]
        f.write(f"{label_name} sessions ({len(subset)} rows):\n")
        f.write(subset[features].describe().to_string())
        f.write("\n\n")

print(f"✅  {SUMMARY_TXT}")
print("\nNEXT STEP: python pipeline/train_model.py")