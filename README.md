# ML Insight Platform

A churn-prediction platform for a telecom customer base, built around an
event-driven data pipeline rather than a static training script. Synthetic
customer behavioral events (logins, support tickets, feature usage) are
streamed through RabbitMQ into Postgres, joined with the customer record
into a model-ready feature table, and served through a FastAPI endpoint
backed by a Redis cache. Model training is tracked in MLflow, feature
drift is monitored with Evidently, and retraining runs asynchronously via
Celery — the goal is a working end-to-end ML system with real
infrastructure seams (queue, cache, DB, async worker), not just a
notebook that outputs a metric.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        GEN["generate_events.py"] --> CSV[("data/synthetic/*.csv")]
        CSV --> PROD["event_producer.py"]
        PROD -->|publish JSON| MQ{{"RabbitMQ<br/>login_events / support_tickets<br/>feature_usage_logs"}}
        MQ --> CONS["event_consumer.py"]
    end

    subgraph Storage
        RAW[("telco_churn.csv")] --> LOAD["load_static_data.py"]
        LOAD --> PGC[("Postgres<br/>customers")]
        CONS --> PGE[("Postgres<br/>event tables")]
        PGC --> FEAT["build_features.py"]
        PGE --> FEAT
        FEAT --> PGF[("Postgres<br/>customer_features")]
    end

    subgraph Serving
        PGF --> CACHE["feature_cache.py<br/>(cache-aside)"]
        CACHE <--> REDIS[("Redis<br/>feature cache, TTL 300s")]
        CACHE --> API["FastAPI /predict"]
    end

    subgraph "ML Ops"
        PGF --> TRAIN["train_baseline.py<br/>LogReg + XGBoost"]
        TRAIN --> MLF[("MLflow tracking<br/>sqlite mlruns.db")]
        MLF --> API
        PGF -.reference vs current.-> DRIFT["monitor_drift.py"]
        DRIFT --> REPORT[["reports/drift_report.html"]]
        TRIG["trigger_retrain.py"] --> BROKER[("Redis<br/>Celery broker")]
        BROKER --> WORKER["Celery worker<br/>retrain_model task"]
        WORKER --> TRAIN
    end
```

## Tech stack

| Tool | Role |
|---|---|
| **RabbitMQ** | Event-driven ingestion backbone — three durable queues (`login_events`, `support_tickets`, `feature_usage_logs`) decouple the event producer from the consumer and simulate a live stream instead of a batch load. |
| **PostgreSQL** | System of record for everything relational: raw customer records, ingested event tables, and the computed `customer_features` table the models and API both read from. |
| **Redis** | Two jobs: cache-aside store in front of Postgres for `/predict` feature lookups (300s TTL), and the broker + result backend for Celery. |
| **pandas / numpy** | Synthetic event generation with per-customer randomized noise, and all feature-engineering aggregation (login trends, ticket resolution stats, usage rates). |
| **scikit-learn** | Preprocessing pipeline (one-hot encoding + scaling) and the Logistic Regression baseline; both models are persisted as a single fitted `Pipeline` so serving doesn't need to re-implement preprocessing. |
| **XGBoost** | Second baseline classifier, currently the stronger of the two on held-out F1/PR-AUC. |
| **MLflow** | Experiment tracking — params, metrics, the model artifact (full pipeline, not just the bare classifier), and feature-importance plots per run. The serving API loads the latest `xgboost` run directly from the tracking store at startup. |
| **FastAPI + Uvicorn** | Serving layer: `GET /health` and `POST /predict`. Model is loaded once at process startup, not per-request. |
| **Evidently** | Data drift detection between the training-time reference distribution and a simulated "current" sample, with an HTML report and per-column drift scores. |
| **Celery** | Decouples retraining from the request path — `trigger_retrain.py` enqueues a job, a worker process runs the actual training. |
| **Docker** | Runs RabbitMQ, Postgres, and Redis as isolated local services (`rabbitmq:3-management`, `postgres:16`, `redis:7`). |

## How to run locally

**1. Start the infrastructure containers**

```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
docker run -d --name postgres-ml -p 5432:5432 -e POSTGRES_PASSWORD=devpassword -e POSTGRES_DB=ml_insight -v pgdata:/var/lib/postgresql/data postgres:16
docker run -d --name redis-ml -p 6379:6379 redis:7
```

**2. Install Python dependencies**

```bash
python -m venv venv
venv\Scripts\activate        # Windows; source venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

**3. Generate synthetic data and load Postgres**

```bash
python src/data_gen/generate_events.py       # writes data/synthetic/*.csv
python src/pipelines/load_static_data.py     # loads data/raw/telco_churn.csv.csv -> customers table
```

**4. Stream the synthetic events through RabbitMQ into Postgres**

```bash
python src/ingestion/event_consumer.py       # run in its own terminal, keeps running
python src/ingestion/event_producer.py       # publishes the CSVs, then exits
```

**5. Build the model-ready feature table**

```bash
python src/pipelines/build_features.py       # writes customer_features table + data/processed/features.csv
```

**6. Train the baseline models (tracked in MLflow)**

```bash
python src/models/train_baseline.py
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000   # http://localhost:5000
```

**7. Check for feature drift**

```bash
python src/models/monitor_drift.py           # writes reports/drift_report.html
```

**8. Serve predictions**

```bash
uvicorn src.serving.api:app --reload --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"customer_id\": \"7590-VHVEG\"}"
```

**9. Trigger an async retrain**

```bash
celery -A src.pipelines.retrain_task worker --loglevel=info --pool=solo   # --pool=solo is required on Windows
python src/pipelines/trigger_retrain.py
```

## Current status

**Built:**
- Synthetic event generation with per-customer randomized, directionally-correlated (not deterministic) drift toward churn behavior across logins, support tickets, and feature usage
- RabbitMQ ingestion: producer/consumer pair, 3 durable queues, batched + idle-flushed commits into Postgres
- Postgres schema: `customers`, `login_events`, `support_tickets`, `feature_usage_logs`, `customer_features`
- Feature engineering pipeline joining customer attributes with event-derived aggregates
- Two baseline models (Logistic Regression, XGBoost) tracked in MLflow, with feature-importance plots logged per run. Current test-set metrics: **F1 0.69 / PR-AUC 0.78 (XGBoost)**, **F1 0.66 / PR-AUC 0.77 (Logistic Regression)**
- Redis cache-aside layer for feature lookups, with a connection-pooled Postgres fallback
- FastAPI serving endpoint (`/health`, `/predict`) loading the model once at startup
- Evidently drift monitoring comparing training-time vs. simulated current feature distributions
- Celery + Redis async retraining, triggered independently of the request path

**Planned:**
- LangChain agent (natural-language interface over the churn data / predictions)
- Automated tests (unit + integration)
- CI/CD pipeline
- Deployment (currently local-only, no cloud target set up)
