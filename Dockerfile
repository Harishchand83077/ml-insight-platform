# FastAPI churn-prediction + agent-chat service (src/serving/api.py).
# Runtime config (Postgres/Redis/MLflow/GROQ_API_KEY) is supplied via
# environment variables at `docker run` time, never baked into the image.
#
# Uses requirements-docker.txt (a curated subset of the full dev
# requirements.txt - just what this service actually imports) rather than
# the full requirements.txt, which is the whole local dev environment
# (Jupyter, pytest, faker, ...) and would otherwise pull in ~280 packages
# instead of ~50.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first, so sentence-transformers' unpinned "torch"
# requirement is already satisfied and pip never reaches for the default
# CUDA-enabled wheel (which drags in ~15 NVIDIA packages, multiple GB,
# useless for CPU-only inference in this container).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
