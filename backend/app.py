import sys
import sqlite3
import datetime
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

# Project root:
# AI-Phishing-Guardian-Web/
# ├── backend/
# │   └── app.py
# ├── frontend/
# ├── ml/
# └── data/

BASE = Path(__file__).resolve().parent.parent

# Allow Python to import from the project root.
# This fixes:
# ModuleNotFoundError: No module named 'ml'
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


# ============================================================
# IMPORTS
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from ml.detector import PhishingDetector


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

FRONTEND_DIR = BASE / "frontend"
ML_DIR = BASE / "ml"
DATA_DIR = BASE / "data"

MODEL_PATH = ML_DIR / "model.joblib"
DATABASE_PATH = DATA_DIR / "scans.db"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

# Allow frontend requests to Flask API
CORS(app)


# ============================================================
# MACHINE LEARNING DETECTOR
# ============================================================

# The model must exist before the detector can be created.
# We handle the missing-model situation cleanly so the server
# can still start while you are preparing/training the model.

detector = None

if MODEL_PATH.exists():
    try:
        detector = PhishingDetector(MODEL_PATH)
        print(f"[OK] ML model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARNING] Could not load ML model: {e}")
else:
    print("[WARNING] ML model not found.")
    print(f"[INFO] Expected model location: {MODEL_PATH}")
    print("[INFO] Train the model before using /api/analyze.")


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Create and return a SQLite database connection.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """
    Create the scans table if it does not already exist.
    """
    with get_db() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                verdict TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                confidence REAL NOT NULL,
                reasons TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    print(f"[OK] Database ready: {DATABASE_PATH}")


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def index():
    """
    Serve the main website.
    """
    return send_from_directory(
        str(FRONTEND_DIR),
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Check whether the backend is running and whether
    the ML model is available.
    """

    return jsonify({
        "status": "running",
        "service": "AI Phishing Guardian API",
        "model_loaded": detector is not None
    })


# ============================================================
# URL ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze():
    """
    Analyze a URL using the phishing detection engine.
    """

    # Check whether the ML model is available
    if detector is None:
        return jsonify({
            "error": "ML model is not available. Please train the model first.",
            "model_path": str(MODEL_PATH)
        }), 503

    # Safely read JSON request
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        data = {}

    # Get URL
    url = str(data.get("url") or "").strip()

    if not url:
        return jsonify({
            "error": "URL is required."
        }), 400

    # Limit unnecessarily huge inputs
    if len(url) > 2048:
        return jsonify({
            "error": "URL is too long."
        }), 400

    # Run ML prediction
    try:
        result = detector.predict(url)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")

        return jsonify({
            "error": "Unable to analyze this URL.",
            "details": str(e)
        }), 500

    # Timestamp
    now = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()

    # Extract result fields safely
    verdict = result.get("verdict", "UNKNOWN")
    risk_score = int(result.get("risk_score", 0))
    confidence = float(result.get("confidence", 0))
    reasons = result.get("reasons", [])

    # Make sure reasons is a list
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    reasons_text = " | ".join(
        str(reason) for reason in reasons
    )

    # Save scan to database
    try:

        with get_db() as connection:

            connection.execute(
                """
                INSERT INTO scans
                (
                    url,
                    verdict,
                    risk_score,
                    confidence,
                    reasons,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    verdict,
                    risk_score,
                    confidence,
                    reasons_text,
                    now
                )
            )

            connection.commit()

    except sqlite3.Error as e:

        print(f"[ERROR] Database error: {e}")

        return jsonify({
            "error": "Analysis completed, but scan could not be saved."
        }), 500

    # Add timestamp to response
    result["created_at"] = now

    return jsonify(result)


# ============================================================
# SCAN HISTORY
# ============================================================

@app.get("/api/history")
def history():
    """
    Return recent scan history.
    """

    # Safely parse limit
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50

    # Keep limit within safe range
    limit = max(1, min(limit, 200))

    try:

        with get_db() as connection:

            rows = connection.execute(
                """
                SELECT
                    id,
                    url,
                    verdict,
                    risk_score,
                    confidence,
                    reasons,
                    created_at
                FROM scans
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return jsonify([
            dict(row)
            for row in rows
        ])

    except sqlite3.Error as e:

        print(f"[ERROR] History database error: {e}")

        return jsonify({
            "error": "Unable to retrieve scan history."
        }), 500


# ============================================================
# STATISTICS
# ============================================================

@app.get("/api/stats")
def stats():
    """
    Return dashboard statistics.
    """

    try:

        with get_db() as connection:

            total_scans = connection.execute(
                "SELECT COUNT(*) FROM scans"
            ).fetchone()[0]

            phishing_detected = connection.execute(
                """
                SELECT COUNT(*)
                FROM scans
                WHERE verdict = 'PHISHING'
                """
            ).fetchone()[0]

            suspicious = connection.execute(
                """
                SELECT COUNT(*)
                FROM scans
                WHERE verdict = 'SUSPICIOUS'
                """
            ).fetchone()[0]

            safe = connection.execute(
                """
                SELECT COUNT(*)
                FROM scans
                WHERE verdict = 'SAFE'
                """
            ).fetchone()[0]

        return jsonify({
            "total_scans": total_scans,
            "threats_detected": phishing_detected,
            "suspicious": suspicious,
            "safe": safe
        })

    except sqlite3.Error as e:

        print(f"[ERROR] Statistics database error: {e}")

        return jsonify({
            "error": "Unable to retrieve statistics."
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       AI PHISHING GUARDIAN")
    print("       Full-Stack Phishing Detection System")
    print("=" * 60)

    print(f"Project root : {BASE}")
    print(f"Frontend     : {FRONTEND_DIR}")
    print(f"Database     : {DATABASE_PATH}")
    print(f"ML model     : {MODEL_PATH}")

    # Initialize database
    init_db()

    print()
    print("Backend starting...")
    print("Website : http://127.0.0.1:5000")
    print("Health  : http://127.0.0.1:5000/api/health")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )