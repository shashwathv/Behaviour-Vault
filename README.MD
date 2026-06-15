# BehaviorVault 2.0 — ML Inference Service

> Behavioral biometric authentication for mobile banking. The ML model, API, and infrastructure half of the project.

This repository contains the Python ML pipeline, the FastAPI inference service, and the self-hosted infrastructure. The mobile app + Node.js backend + admin dashboard live in a separate repository — see [Project Architecture](#project-architecture) below.

---

## What it does

Most fraud detection runs at the login screen — passwords, OTPs, 2FA. Once an attacker gets past that, nothing stops them. BehaviorVault adds a second authentication layer that runs silently for the entire session, scoring user behavior continuously on five signals:

- Keystroke timing (ms between key presses)
- Touch pressure (normalized 0–1)
- Swipe velocity (px/sec)
- Scroll rhythm (ms between scroll events)
- Accelerometer variance (motion of the device while held)

The system detects three threat archetypes:

| Threat | What it looks like |
|---|---|
| **Attacker** | Different person using the device — unfamiliar typing rhythm, harder presses, faster swipes |
| **Duress** | Legitimate user under coercion — shaky hands (high accelerometer variance), hesitant gestures, slow keystrokes |
| **Bot / ADB injection** | Automated input — keystroke variance under 15ms (humans never), unnaturally uniform pressure |

---

## Project architecture

The system has two services that communicate over HTTPS:

```
┌────────────────────┐     ┌────────────────────┐     ┌──────────────────────────┐
│ React Native App   │ →   │ Node.js Backend    │ →   │ FastAPI ML Service       │
│ (Expo, on phone)   │     │ (Render)           │     │ (this repo, on homelab)  │
│                    │     │                    │     │                          │
│ BehaviorTracker    │     │ HMAC validator,    │     │ TFLite Isolation Forest, │
│ captures gestures  │     │ shadow escrow,     │     │ EWMA personal baseline,  │
│                    │     │ admin cockpit      │     │ Cloudflare Zero Trust    │
└────────────────────┘     └────────────────────┘     └──────────────────────────┘
   Mobile-side repo                                       This repo (ML-side)
```

**Mobile + backend repository** (built by [@Shashankgyaplar](https://github.com/Shashankgyaplar)):
👉 https://github.com/Shashankgyaplar/behaviorValut

**Live demos:**
- 📱 [Android APK](https://expo.dev/artifacts/eas/8gLYnWTD8aI9_smfvXoJ43BKz6pZEvlbXi40nTpARgE.apk)
- 🖥️ [Admin Dashboard](https://cockpitbv.onrender.com)

---

## What's in this repo

```
Behaviour-Vault/
├── api/
│   └── app.py                  # FastAPI inference service
├── pipeline/
│   ├── generate_data.py        # Synthetic behavioral dataset generator
│   ├── train_model.py          # Isolation Forest training
│   ├── export_tflite.py        # Keras surrogate → .tflite conversion
│   ├── get_scalar_value.py     # Extracts scaler MEAN/STD for the API
│   └── differential_privacy.py # Differential-privacy layer demo
├── models/
│   ├── isolation_forest.pkl    # Trained scikit-learn model
│   ├── scaler.pkl              # StandardScaler fitted on normal data
│   └── behavior_model.tflite   # 3KB TFLite model for on-device inference
├── data/
│   ├── behavioral_data.csv     # 1200 synthetic sessions
│   └── scored_sessions.csv     # Sessions with anomaly scores
├── reports/
│   ├── data_summary.txt        # Feature distribution sanity check
│   └── evaluation_report.txt   # Classification metrics + confusion matrix
├── Dockerfile
└── requirements.txt
```

---

## The ML model

**Algorithm:** Isolation Forest (scikit-learn), trained unsupervised on the "normal" class only.

**Why Isolation Forest?** It builds a picture of "normal behavior" by randomly partitioning the feature space; sessions that take few partitions to isolate are flagged as anomalies. Works without labeled attack data, which a real bank wouldn't have at launch.

**Training data:** 1200 synthetic sessions across realistic mobile behavior distributions. Normal sessions sit at keystroke ~600ms, swipe ~450 px/s, with three anomaly archetypes (attacker, duress, bot) constructed to lie clearly outside the normal envelope.

**Evaluation:**
- AUC-ROC: 1.0000 on the synthetic test set
- 100% recall on anomalies
- 5% false-positive rate (controlled by `contamination=0.05`)

The model's tightness was an interesting build problem. My first version trained on keystroke means of 200ms — fine for desktop typing, way off for mobile. Real users on phones type at 400–900ms, so the first model flagged every real user as an anomaly (z-score of 13+ against the training mean). Retraining with realistic distributions fixed it. Lesson: synthetic data is only useful if it reflects the production distribution.

---

## The TFLite distillation

scikit-learn doesn't run on mobile, and the goal was on-device scoring without sending raw behavioral data over the network. So I distilled the Isolation Forest into a small TensorFlow Lite model:

1. Generate 50,000 random points in scaled feature space
2. Label them using the trained Isolation Forest (1 if anomaly, 0 if normal)
3. Train a tiny Keras network (16 → 8 → 1 sigmoid) to match those labels
4. Export to TFLite with default optimizations

**Result:** 3KB model, ~10ms inference on a phone, ~97% agreement with the Isolation Forest on held-out test data. Distillation works.

---

## The API

**Stack:** FastAPI + Pydantic + uvicorn

```
POST   /predict                   — Score a behavioral session
GET    /baseline/{user_id}        — View a user's current EWMA baseline
DELETE /baseline/{user_id}        — Reset a user's baseline
GET    /health                    — Health check (open, for uptime monitors)
GET    /stats                     — Live request stats + recent activity
GET    /docs                      — Auto-generated OpenAPI / Swagger UI
```

All routes except `/health` require an `X-API-Key` header. From the Swagger UI at `/docs`, click "Authorize" and paste your key — every request fired from the page then includes it automatically.

### Personalized EWMA baselines

The global model handles cold-start (sessions 1–3). After that, each user's behavior is tracked with an Exponentially Weighted Moving Average (α=0.15) that adapts to their natural drift. The combined score weights 30% global + 70% personal baseline once warmup completes — the model that knows the user dominates the model that knows everyone.

### Zero-imputation for missing features

If a feature comes in as `0` (meaning "user didn't perform this gesture this session"), it's replaced with the training mean before model inference. Otherwise the global model treats missing data as anomalous, which produces nonsense scores for legitimate users who just didn't swipe that session. The personal baseline still tracks raw values so user-specific patterns learn correctly.

### Lockout prevention

The naive rule "freeze baseline on every anomaly" locks out legitimate users whose behavior naturally varies — one weird session permanently distrusts the user. The fix: only freeze the baseline on *severe* anomalies (score > 0.95), not borderline ones. Personal baselines stay adaptive while genuinely attack-like patterns still get blocked.

---

## Security

The API is publicly addressable but protected by three layers:

1. **Cloudflare Zero Trust Service Tokens** at the edge — requests without valid Service Token headers are blocked before reaching the origin.
2. **API key validation** in FastAPI (`X-API-Key` header against `BV_API_KEYS` env var).
3. **Pydantic schema validation** with per-feature range bounds — rejects NaN, negative values, out-of-range inputs before they hit the model.

Combined with WAF rules that bypass Bot Fight Mode only for authenticated service-to-service traffic, this gives the same protection pattern that B2B SaaS APIs ship with on day one.

---

## Infrastructure

Self-hosted on a personal Linux homelab — a Dell Inspiron 15 running **Arch Linux** + Docker + Nginx. Deployed via **Coolify** (open-source Heroku alternative) with GitHub auto-deploy. Public reachability via **Cloudflare Tunnel** — no exposed ports, no static IP, no port forwarding, no monthly cloud bill.

```
GitHub push
   ↓
Coolify webhook
   ↓
Docker build + container restart
   ↓
Nginx (host) → Container (port 5000)
   ↓
Cloudflare Tunnel ←→ bhv-api.nw-right.dev
```

---

## Observability

The API ships with a live terminal dashboard built on `rich` that logs every request inline:

- Per-request panel showing input features, all three scores (global, baseline, combined), confidence_pct, decision (color-coded NORMAL/ELEVATED/ANOMALY), client IP, latency
- Auto-printed summary every 25 requests with p50/p95 latency, unique users, anomaly rate, unauthorized attempts
- Compact log lines for baseline lookups and resets
- Silent on `/health` (uptime monitors hit it constantly)

`docker logs -f <container-name>` gives a real-time view of every request hitting the API.

---

## Run it locally

```bash
git clone https://github.com/shashwathv/Behaviour-Vault.git
cd Behaviour-Vault
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate the model from scratch (optional — pre-trained artifacts are committed)
python pipeline/generate_data.py
python pipeline/train_model.py
python pipeline/export_tflite.py
python pipeline/get_scalar_value.py

# Run the API
BV_API_KEYS=dev-key uvicorn api.app:app --reload --port 5000
```

Open http://localhost:5000/docs for the interactive Swagger UI. Click the **Authorize** button and paste `dev-key` to make requests from the page.

```bash
curl -s http://localhost:5000/predict \
  -H "X-API-Key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "demo",
    "keystroke_avg_ms": 600,
    "touch_pressure_avg": 0.45,
    "swipe_avg_px_per_sec": 420,
    "scroll_rhythm_ms": 180,
    "accelerometer_avg_variance": 0.04
  }' | jq
```

---

## Tech stack

- **ML:** Python, scikit-learn, TensorFlow Lite
- **API:** FastAPI, Pydantic, uvicorn
- **Infra:** Docker, Coolify, Nginx, Cloudflare Tunnel, Cloudflare Zero Trust, Arch Linux
- **Observability:** `rich` (terminal dashboards)

---

## Team

**Shashwath V** — ML model, API, infrastructure
Built the synthetic data pipeline, trained the Isolation Forest, distilled it to TFLite, designed the FastAPI inference service with personalized baselines, set up the multi-layer security, and self-hosted the whole stack on a homelab.

**Shashank G Yaplar** ([@Shashankgyaplar](https://github.com/Shashankgyaplar/behaviorValut)) — Mobile app, backend, admin dashboard
Built the React Native + Expo app, the behavioral telemetry capture layer, the Node.js + Express + MongoDB backend with HMAC-signed logging and shadow-escrow duress handling, and the React.js admin dashboard.

---

## License

MIT — see [LICENSE](./LICENSE).