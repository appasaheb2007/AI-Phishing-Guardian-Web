# 🛡️ AI Phishing Guardian — Full-Stack ML Web App

A full-stack evolution of my original browser-extension prototype. The project provides a web interface for URL analysis, a Flask API, persistent scan history, statistics, and an optional scikit-learn phishing classifier.

## Architecture

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python + Flask
- **Database:** SQLite
- **ML:** scikit-learn Random Forest baseline
- **Model input:** lexical/structural URL features

## Important accuracy note

The application does **not** claim 100% real-world accuracy. A security classifier should be evaluated on a held-out test set using accuracy, precision, recall and F1. The training script prints these metrics. Never use a hard-coded 100% value as proof of security.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/app.py
```

Open `http://127.0.0.1:5000`.

## Train the ML model

Create `data/urls.csv` with:

```csv
url,label
https://legitimate-site.example,0
http://phishing-site.example/login,1
```

Then run:

```bash
cd ml
python train.py
```

The model is saved as `ml/model.joblib`. Do not commit a model trained on sensitive/private data.

## Features

- URL risk scoring
- Safe / Suspicious / Phishing classification
- ML probability when a trained model is available
- Explainable URL indicators
- Scan history
- Statistics dashboard
- Responsive UI
- Dark theme
- Demo URLs

## From the original extension

This web app keeps the core concept of the original extension—URL analysis, risk scoring, history and security indicators—while moving the user interface and analysis service into a full-stack architecture.

## Future upgrades

- Threat-intelligence/reputation API integration
- Domain age/DNS/WHOIS signals
- Redirect analysis
- HTML/DOM feature extraction in a sandbox
- Authentication and user-specific history
- PostgreSQL for production
- Docker deployment
- Model versioning and continuous evaluation
