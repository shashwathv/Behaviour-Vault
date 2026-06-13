"""
BehaviorVault 2.0 — Inference API (v3 with live terminal dashboard)
====================================================================

PATHS:
  Reads:        models/behavior_model.tflite
  Reads/writes: baselines/<user_id>.json

WHAT'S NEW IN v3:
  • Rich-formatted terminal dashboard for every /predict request
  • Startup banner showing config
  • Auto-printed summary every 25 requests, on /stats endpoint
  • Compact log lines for /baseline GET/DELETE
  • /health stays silent (uptime monitors hit it frequently)
  • Tracks client IP (Cloudflare-aware), unique users, p50/p95 latency

VIEW IT:
  docker logs -f behaviour-vault20         ← live tail
  docker logs --tail 200 behaviour-vault20

REQUIREMENTS:
  Add to requirements.txt:
    rich>=13.0
"""

# Suppress TensorFlow's startup noise so the dashboard reads cleanly.
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import re
import time
import threading
from collections import deque
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ── Console ────────────────────────────────────────────────────────────────
# force_terminal=True keeps ANSI colors flowing through docker logs/journald.
console = Console(
    force_terminal=True,
    width=int(os.environ.get('CONSOLE_WIDTH', '100')),
    log_path=False,
)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior_model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS  = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()

# ── Scaler values (from get_scalar_value.py output) ────────────────────────
MEAN = [604.4399, 0.5071, 451.8634, 179.5553, 0.0493]
STD  = [191.6922, 0.0997, 144.1839,  78.1840, 0.0192]

# ── Config ─────────────────────────────────────────────────────────────────
EWMA_ALPHA              = 0.15
WARMUP_SESSIONS         = 3
ANOMALY_THRESHOLD       = 0.8     # is_anomaly returned to client at >this
SEVERE_ANOMALY_THRESHOLD = 0.95   # baseline is FROZEN only at >this
ELEVATED_THRESHOLD      = 0.5
GLOBAL_WEIGHT           = 0.6
BASELINE_WEIGHT         = 0.4
BASELINE_NORMALIZER     = 3.0
SUMMARY_EVERY           = 25

BASELINE_DIR = os.path.join(BASE_DIR, "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)

FEATURE_KEYS = [
    'keystroke_avg_ms',
    'touch_pressure_avg',
    'swipe_avg_px_per_sec',
    'scroll_rhythm_ms',
    'accelerometer_avg_variance',
]

LEVEL_STYLE = {
    "NORMAL":   ("green",  "✓"),
    "ELEVATED": ("yellow", "!"),
    "ANOMALY":  ("red",    "✗"),
}

# ── Live stats ─────────────────────────────────────────────────────────────
STATS_LOCK = threading.Lock()
STATS = {
    'started_at':  time.time(),
    'request_id':  0,
    'total':       0,
    'NORMAL':      0,
    'ELEVATED':    0,
    'ANOMALY':     0,
    'errors':      0,
    'users':       set(),
    'latencies':   deque(maxlen=200),
}
RECENT = deque(maxlen=50)


# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
_USER_ID_RE = re.compile(r'[^A-Za-z0-9_\-]')

def safe_user_id(user_id: str) -> str:
    cleaned = _USER_ID_RE.sub('', str(user_id))
    return cleaned or 'anonymous'

def get_baseline_path(user_id: str) -> str:
    return os.path.join(BASELINE_DIR, f"{safe_user_id(user_id)}.json")

def load_baseline(user_id: str):
    path = get_baseline_path(user_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def save_baseline(user_id: str, baseline: dict):
    baseline["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(get_baseline_path(user_id), "w") as f:
        json.dump(baseline, f)

def update_ewma_baseline(user_id, current_features, existing):
    if existing is None:
        new_values = list(current_features)
        session_count = 1
    else:
        old_values = existing["values"]
        new_values = [
            EWMA_ALPHA * curr + (1 - EWMA_ALPHA) * old
            for curr, old in zip(current_features, old_values)
        ]
        session_count = existing.get("session_count", 0) + 1
    baseline = {"values": new_values, "session_count": session_count}
    save_baseline(user_id, baseline)
    return baseline

def score_against_baseline(raw, baseline_values):
    deviations = [abs(raw[i] - baseline_values[i]) / STD[i] for i in range(len(raw))]
    return float(np.mean(deviations))


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_tflite(raw):
    normalized = [(raw[i] - MEAN[i]) / STD[i] for i in range(5)]
    input_data = np.array([normalized], dtype=np.float32)
    interpreter.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    interpreter.invoke()
    return float(interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])[0][0])

def classify_level(score):
    if score > ANOMALY_THRESHOLD:    return 'ANOMALY'
    if score > ELEVATED_THRESHOLD:   return 'ELEVATED'
    return 'NORMAL'


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def get_client_ip():
    """Real client IP, accounting for Cloudflare tunnel headers."""
    return (request.headers.get('CF-Connecting-IP')
            or request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
            or request.remote_addr
            or 'unknown')

def ts():
    return datetime.now().strftime('%H:%M:%S')

def print_banner():
    info = Table(show_header=False, box=None, padding=(0, 1))
    info.add_column(style="dim", min_width=22)
    info.add_column()
    info.add_row("Status",             "[green bold]RUNNING[/]")
    info.add_row("Port",               "5000")
    info.add_row("Model",              os.path.relpath(MODEL_PATH, BASE_DIR))
    info.add_row("Warmup sessions",    str(WARMUP_SESSIONS))
    info.add_row("Anomaly threshold",  f"score > {ANOMALY_THRESHOLD}")
    info.add_row("Elevated threshold", f"score > {ELEVATED_THRESHOLD}")
    info.add_row("EWMA α",             str(EWMA_ALPHA))
    info.add_row("Score blend",        f"{GLOBAL_WEIGHT}·global + {BASELINE_WEIGHT}·baseline")
    info.add_row("Summary every",      f"{SUMMARY_EVERY} requests")
    console.print()
    console.print(Panel(info,
                        title="[bold cyan]🛡  BehaviorVault 2.0 API[/]",
                        title_align="left",
                        border_style="cyan",
                        box=box.HEAVY))

def print_request_panel(req_id, ip, user_id, raw, result, latency_ms):
    color, icon = LEVEL_STYLE[result['level']]

    body = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    body.add_column(style="dim", min_width=32)
    body.add_column(justify="right")

    body.add_row("[bold cyan]INPUT[/]", "")
    for k, v in zip(FEATURE_KEYS, raw):
        body.add_row(f"  {k}", f"{v}")

    body.add_row("", "")

    body.add_row("[bold cyan]OUTPUT[/]", "")
    body.add_row("  confidence_pct", f"[bold]{result['confidence_pct']}%[/]")
    body.add_row("  global_score",   f"{result['global_score']:.4f}")
    bs = (f"{result['baseline_score']:.4f}"
          if result['baseline_score'] is not None else "[dim]—[/]")
    body.add_row("  baseline_score", bs)
    body.add_row("  combined_score", f"[bold]{result['score']:.4f}[/]")
    body.add_row("  session_count",  str(result['session_count']))
    if result['warmup_remaining'] > 0:
        body.add_row("  warmup_remaining",
                     f"[yellow]{result['warmup_remaining']} session(s)[/]")
    body.add_row("  using_baseline", "yes" if result['using_baseline'] else "no")
    body.add_row("  decision",       f"[{color} bold]{icon} {result['level']}[/]")

    title = (f"[bold]#{req_id}[/]  ·  [dim]{ts()}[/]  ·  "
             f"user=[bold]{user_id}[/]  ·  [dim]{ip}[/]  ·  {latency_ms}ms")
    console.print(Panel(body, title=title, title_align="left",
                        border_style=color, box=box.ROUNDED))

def print_summary():
    with STATS_LOCK:
        uptime = int(time.time() - STATS['started_at'])
        h, rem = divmod(uptime, 3600); m, s = divmod(rem, 60)
        total = STATS['total']
        rate  = (STATS['ANOMALY'] / total * 100) if total else 0
        users = len(STATS['users'])
        lats  = sorted(STATS['latencies'])
        p50   = lats[len(lats)//2]            if lats else 0
        p95   = lats[int(len(lats)*0.95)]     if lats else 0
        norm, elev, anom = STATS['NORMAL'], STATS['ELEVATED'], STATS['ANOMALY']
        errs  = STATS['errors']

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", min_width=18)
    t.add_column(justify="right")
    t.add_row("Uptime",          f"{h:02d}h {m:02d}m {s:02d}s")
    t.add_row("Total requests",  str(total))
    t.add_row("[green]Normal[/]",     f"[green]{norm}[/]")
    t.add_row("[yellow]Elevated[/]",  f"[yellow]{elev}[/]")
    t.add_row("[red]Anomaly[/]",      f"[red]{anom}[/]")
    t.add_row("Errors (4xx)",    f"[red]{errs}[/]" if errs else "0")
    t.add_row("Unique users",    str(users))
    t.add_row("Anomaly rate",    f"{rate:.1f}%")
    t.add_row("Latency p50",     f"{p50}ms")
    t.add_row("Latency p95",     f"{p95}ms")

    console.print(Panel(t,
                        title="[bold magenta]📊  SUMMARY[/]",
                        title_align="left",
                        border_style="magenta",
                        box=box.DOUBLE))

def print_compact(ip, method, path, status, extra=""):
    color = "green" if 200 <= status < 300 else "red" if status >= 400 else "yellow"
    console.print(
        f"[dim]{ts()}[/]  [{color}]{status}[/]  "
        f"[bold]{method}[/] {path}  [dim]{ip}[/]  [dim]{extra}[/]"
    )

def print_error(ip, status, msg):
    console.print(
        f"[dim]{ts()}[/]  [red bold]✗ {status}[/]  [dim]{ip}[/]  [red]{msg}[/]"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    # Silent — uptime monitors hit this often.
    return jsonify({
        'status':          'ok',
        'model_loaded':    True,
        'warmup_sessions': WARMUP_SESSIONS,
        'feature_keys':    FEATURE_KEYS,
    })


@app.route('/predict', methods=['POST'])
def predict():
    start = time.perf_counter()
    ip = get_client_ip()

    with STATS_LOCK:
        STATS['request_id'] += 1
        req_id = STATS['request_id']

    # ── Validate ──────────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        with STATS_LOCK: STATS['errors'] += 1
        print_error(ip, 400, "body is not a JSON object")
        return jsonify({'error': 'Body must be a JSON object'}), 400

    user_id = data.get('userId') or data.get('user_id') or 'anonymous'

    missing = [k for k in FEATURE_KEYS if k not in data]
    if missing:
        with STATS_LOCK: STATS['errors'] += 1
        print_error(ip, 400, f"user={user_id}  missing features: {missing}")
        return jsonify({
            'error': 'Missing features', 'missing': missing,
            'expected': FEATURE_KEYS,
        }), 400

    try:
        raw = [float(data[k]) for k in FEATURE_KEYS]
    except (TypeError, ValueError):
        with STATS_LOCK: STATS['errors'] += 1
        print_error(ip, 400, f"user={user_id}  non-numeric feature")
        return jsonify({'error': 'All features must be numeric'}), 400

    # ── Global model ──────────────────────────────────────────────────────
    global_score = run_tflite(raw)

    # ── Personal baseline ─────────────────────────────────────────────────
    baseline = load_baseline(user_id)
    prior_session_count = baseline['session_count'] if baseline else 0
    baseline_score = (
        score_against_baseline(raw, baseline['values']) if baseline else None
    )

    using_baseline = (
        baseline_score is not None and prior_session_count >= WARMUP_SESSIONS
    )
    if using_baseline:
        norm_baseline = min(baseline_score / BASELINE_NORMALIZER, 1.0)
        combined_score = (GLOBAL_WEIGHT * global_score
                          + BASELINE_WEIGHT * norm_baseline)
    else:
        combined_score = global_score

    is_anomaly = combined_score > ANOMALY_THRESHOLD

    # Update baseline unless the score is *extreme*, not just over the
    # anomaly threshold. This prevents the lockout pattern where one
    # borderline anomaly freezes the baseline forever — legitimate users
    # whose behavior naturally varies would otherwise be locked out.
    # An attacker producing >0.95 scores still cannot pollute the baseline.
    freeze_baseline = combined_score > SEVERE_ANOMALY_THRESHOLD

    if not freeze_baseline:
        updated = update_ewma_baseline(user_id, raw, baseline)
        session_count = updated['session_count']
    else:
        session_count = prior_session_count

    level = classify_level(combined_score)
    latency_ms = int((time.perf_counter() - start) * 1000)

    # confidence_pct = how confident we are this IS the real user (0..100).
    # Pre-computed so clients don't have to do (1 - score) * 100 themselves,
    # which is bug-prone if they accidentally read baseline_score (unbounded).
    confidence_pct = int(round((1.0 - combined_score) * 100))

    result = {
        'user_id':          user_id,
        'confidence_pct':   confidence_pct,   # 0..100 — safe to display directly
        'score':            round(combined_score, 4),    # 0..1 — anomaly score
        'global_score':     round(global_score, 4),      # 0..1 — model only
        'baseline_score':   round(baseline_score, 4) if baseline_score is not None else None,  # UNBOUNDED z-score; do not display raw
        'using_baseline':   using_baseline,
        'warmup_remaining': max(0, WARMUP_SESSIONS - session_count),
        'session_count':    session_count,
        'is_anomaly':       is_anomaly,
        'level':            level,
        'baseline_updated': not freeze_baseline,
    }

    # ── Update stats + log ────────────────────────────────────────────────
    with STATS_LOCK:
        STATS['total'] += 1
        STATS[level] += 1
        STATS['users'].add(user_id)
        STATS['latencies'].append(latency_ms)
        total_now = STATS['total']
        RECENT.append({
            'id':         req_id,
            'ts':         time.time(),
            'ip':         ip,
            'user':       user_id,
            'level':      level,
            'score':      result['score'],
            'latency_ms': latency_ms,
        })

    print_request_panel(req_id, ip, user_id, raw, result, latency_ms)
    if total_now % SUMMARY_EVERY == 0:
        print_summary()

    return jsonify(result)


@app.route('/baseline/<user_id>', methods=['GET'])
def get_baseline_endpoint(user_id):
    ip = get_client_ip()
    baseline = load_baseline(user_id)
    if baseline is None:
        print_compact(ip, "GET", f"/baseline/{user_id}", 404, "no baseline")
        return jsonify({'error': 'No baseline found'}), 404
    print_compact(ip, "GET", f"/baseline/{user_id}", 200,
                  f"sessions={baseline['session_count']}")
    return jsonify({
        'user_id':       user_id,
        'session_count': baseline['session_count'],
        'baseline':      dict(zip(FEATURE_KEYS, baseline['values'])),
        'updated_at':    baseline.get('updated_at'),
    })


@app.route('/baseline/<user_id>', methods=['DELETE'])
def reset_baseline(user_id):
    ip = get_client_ip()
    path = get_baseline_path(user_id)
    if os.path.exists(path):
        os.remove(path)
        print_compact(ip, "DELETE", f"/baseline/{user_id}", 200, "reset")
        return jsonify({'message': f'Baseline for {user_id} reset.'})
    print_compact(ip, "DELETE", f"/baseline/{user_id}", 404, "no baseline")
    return jsonify({'error': 'No baseline found'}), 404


@app.route('/stats', methods=['GET'])
def get_stats():
    """JSON snapshot of live stats. Also triggers a terminal summary print."""
    print_compact(get_client_ip(), "GET", "/stats", 200, "summary requested")
    print_summary()
    with STATS_LOCK:
        uptime = time.time() - STATS['started_at']
        return jsonify({
            'uptime_seconds':  round(uptime, 2),
            'total_requests':  STATS['total'],
            'normal':          STATS['NORMAL'],
            'elevated':        STATS['ELEVATED'],
            'anomaly':         STATS['ANOMALY'],
            'errors':          STATS['errors'],
            'unique_users':    len(STATS['users']),
            'anomaly_rate':    (STATS['ANOMALY'] / max(1, STATS['total'])),
            'recent_requests': list(RECENT),
        })


# ── Startup banner — prints once on module load (works under gunicorn) ─────
print_banner()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)