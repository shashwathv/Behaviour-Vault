"""
SCRIPT 2 — Train Isolation Forest + Save Model
===============================================
BehaviorVault 2.0 | SuRaksha Cyber Hackathon 2.0

PATHS:
  Reads:  data/behavioral_data.csv
  Writes: models/isolation_forest.pkl
          models/scaler.pkl
          data/scored_sessions.csv
          reports/evaluation_report.txt

The Isolation Forest is UNSUPERVISED — it trains only on label=0 rows.
Labels are only used for the post-hoc evaluation in Step 4.

HOW TO RUN:
  python pipeline/train_model.py
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DATA_CSV         = os.path.join(DATA_DIR,    "behavioral_data.csv")
SCORED_CSV       = os.path.join(DATA_DIR,    "scored_sessions.csv")
ISO_FOREST_PKL   = os.path.join(MODELS_DIR,  "isolation_forest.pkl")
SCALER_PKL       = os.path.join(MODELS_DIR,  "scaler.pkl")
EVAL_REPORT_TXT  = os.path.join(REPORTS_DIR, "evaluation_report.txt")

FEATURES = [
    "keystroke_timing_ms",
    "touch_pressure",
    "swipe_speed_px_s",
    "scroll_rhythm_ms",
    "accel_variance",
]

# ── Step 1 — Load data ──────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1 — Loading data")
print("=" * 55)

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(
        f"{DATA_CSV} not found. Run pipeline/generate_data.py first."
    )

df = pd.read_csv(DATA_CSV)
print(f"  Loaded {len(df)} rows from {DATA_CSV}")
print(f"  Normal  : {(df['label']==0).sum()}")
print(f"  Anomaly : {(df['label']==1).sum()}")

X        = df[FEATURES].values
y        = df["label"].values
X_normal = df[df["label"] == 0][FEATURES].values
print(f"  Training set (normal-only): {len(X_normal)} rows")

# ── Step 2 — Scale features ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2 — Scaling features (fitted on NORMAL only)")
print("=" * 55)

scaler = StandardScaler()
scaler.fit(X_normal)
X_scaled        = scaler.transform(X)
X_normal_scaled = scaler.transform(X_normal)

print("  Scaler stats on normal data after transform:")
for i, feat in enumerate(FEATURES):
    print(f"    {feat:<25} mean={X_normal_scaled[:, i].mean():+.4f}  "
          f"std={X_normal_scaled[:, i].std():.4f}")

# ── Step 3 — Train Isolation Forest ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3 — Training Isolation Forest")
print("=" * 55)

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    max_features=1.0,
    random_state=42,
)
model.fit(X_normal_scaled)
print(f"  ✅  n_estimators={model.n_estimators}  "
      f"contamination={model.contamination}  "
      f"n_features={model.n_features_in_}")

# ── Step 4 — Evaluate ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4 — Evaluation")
print("=" * 55)

raw_predictions = model.predict(X_scaled)
y_pred          = (raw_predictions == -1).astype(int)
anomaly_scores  = -model.decision_function(X_scaled)

df["anomaly_score"] = anomaly_scores.round(5)
df["predicted"]     = y_pred

report = classification_report(y, y_pred,
                                target_names=["Normal", "Anomaly"],
                                digits=3)
print("\n  Classification Report:")
for line in report.splitlines():
    print("  " + line)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n  Confusion Matrix:")
print(f"                    Pred Normal   Pred Anomaly")
print(f"  Actual Normal   :    {tn:<6}        {fp}")
print(f"  Actual Anomaly  :    {fn:<6}        {tp}")

auc = roc_auc_score(y, anomaly_scores)
print(f"\n  AUC-ROC: {auc:.4f}")

normal_scores  = anomaly_scores[y == 0]
anomaly_only   = anomaly_scores[y == 1]
print(f"\n  Score distribution:")
print(f"    Normal  mean={normal_scores.mean():.4f}  max={normal_scores.max():.4f}")
print(f"    Anomaly mean={anomaly_only.mean():.4f}  min={anomaly_only.min():.4f}")

# ── Step 5 — Save artifacts ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5 — Saving artifacts")
print("=" * 55)

with open(ISO_FOREST_PKL, "wb") as f:
    pickle.dump(model, f)
print(f"  ✅  {ISO_FOREST_PKL}")

with open(SCALER_PKL, "wb") as f:
    pickle.dump(scaler, f)
print(f"  ✅  {SCALER_PKL}")

df.to_csv(SCORED_CSV, index=False)
print(f"  ✅  {SCORED_CSV}")

with open(EVAL_REPORT_TXT, "w") as f:
    f.write("BEHAVIORVAULT 2.0 — Isolation Forest Evaluation\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Training set : {len(X_normal)} normal sessions\n")
    f.write(f"Eval set     : {len(X)} total sessions\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write(f"AUC-ROC: {auc:.4f}\n\n")
    f.write(f"Confusion Matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}\n\n")
    f.write(f"Score distribution:\n")
    f.write(f"  Normal  mean={normal_scores.mean():.4f}  "
            f"max={normal_scores.max():.4f}\n")
    f.write(f"  Anomaly mean={anomaly_only.mean():.4f}  "
            f"min={anomaly_only.min():.4f}\n")
print(f"  ✅  {EVAL_REPORT_TXT}")

print("\nNEXT STEP: python pipeline/export_tflite.py")