# Bike Demand Auto-Retraining MLOps Pipeline

An end-to-end MLOps pipeline that automatically detects data drift,
retrains a machine learning model, and promotes it to production —
all without manual intervention.

## Architecture

```
New monthly data
      │
      ▼
drift_check   ──── no drift ──▶  keep current model
      │
   drift!
      │
      ▼
train_model   ──▶  MLflow experiment + Model Registry
      │
      ▼
evaluate      ──▶  challenger vs champion (RMSE comparison)
      │
   better?
      │
      ▼
promote       ──▶  alias 'champion' updated in MLflow Registry
      │
      ▼
FastAPI       ──▶  always serves current @champion model
```

## Stack

| Component | Tool |
|---|---|
| Orchestration | Prefect |
| Experiment tracking | MLflow |
| Model registry | MLflow Registry |
| Drift detection | Evidently AI |
| API | FastAPI + Uvicorn |
| Operational monitoring | Prometheus + Grafana |
| Testing | pytest |
| CI | GitHub Actions |
| Containerization | Docker + Docker Compose |

## Project Structure

```
bike-retrain-mlops/
├── data/
│   ├── raw/                     # Original hour.csv dataset
│   └── monthly_chunks/          # Dataset split by month (simulates incremental data)
├── src/
│   ├── data/data_prep.py        # Load, clean, and split dataset
│   ├── training/
│   │   ├── train.py             # Train model + log to MLflow
│   │   └── evaluate.py          # Compare challenger vs champion
│   ├── registry/promote.py      # Promote or reject challenger
│   ├── monitoring/
│   │   ├── drift_check.py       # Evidently drift detection
│   │   ├── prediction_logger.py # Log predictions to CSV
│   │   └── metrics.py           # Prometheus metrics
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   └── schemas.py           # Pydantic request/response models
│   ├── flows/retrain_flow.py    # Prefect flow (full pipeline)
│   └── tests/                   # pytest test suite
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
git clone <your-repo-url>
cd bike-retrain-mlops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

```bash
curl -L -o data/raw/hour.csv \
  "https://raw.githubusercontent.com/muditp19/UCI_Bike-sharing-dataset/master/hour.csv"

python -m src.data.data_prep
```

### 3. Start MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5001
```

### 4. Run the full pipeline

```bash
python -m src.flows.retrain_flow
```

### 5. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 6. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2, "yr": 0, "mnth": 6, "hr": 17,
    "holiday": 0, "weekday": 2, "workingday": 1,
    "weathersit": 1, "temp": 0.68, "atemp": 0.6364,
    "hum": 0.79, "windspeed": 0.1343
  }'
```

### 7. Run with Docker

```bash
docker compose up
```

Services:
- API: http://localhost:8000
- MLflow UI: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Running Tests

```bash
python -m pytest src/tests/ -v
```

## How the Pipeline Decides to Retrain

1. **Drift check**: Evidently compares feature distributions between
   training data and new month data using statistical tests (KS test).
   If more than 50% of features have drifted → trigger retraining.

2. **Train**: New model trained on all data up to the new month.

3. **Evaluate**: Both models predict on the new month data.
   If challenger RMSE is at least 5% better → promote.

4. **Promote**: MLflow alias `@champion` is updated to point
   to the new model version. FastAPI picks it up on next restart.