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

# Pre-download and cache the RAG/semantic-cache embedding model at build
# time, not on the container's first real request. We measured this cause
# a multi-minute stall on a fresh process in production-like testing:
# sentence-transformers checks the HuggingFace Hub for the latest model
# revision even when a local cache exists, unless told not to. HF_HUB_OFFLINE
# (set only after the download, so the download itself can still reach the
# Hub) stops the running container from ever making that check again once
# the model is already baked into this layer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1

COPY src/ src/
# The trained model, exported from the MLflow tracking store to a plain
# local model directory by scripts/export_model_for_deployment.py - the
# tracking store itself (mlruns.db/mlruns/) is dev-only and never shipped.
COPY models/production_model/ models/production_model/

# Build the RAG knowledge base (the Chroma vector index query_project_docs_tool
# reads) at image-build time, not on the container's first request - same
# build-time-not-runtime principle as the embedding-model pre-download
# above, and the actual fix for a real production gap: this step didn't
# exist before, so query_project_docs_tool silently had no index to query
# in every deployed container (data/ is .dockerignore'd, and nothing ever
# ran build_knowledge_base.py in the image - it always returned "No
# relevant context found", no error, no indication anything was wrong).
# docs/ is copied in just for this step; nothing at serve time reads
# docs/ directly, only the data/chroma_db/ index this produces.
COPY docs/ docs/
RUN python src/agent/build_knowledge_base.py

# Fail the build - not just at runtime - if the index wasn't actually
# produced. A build that silently ships an empty or missing index would
# otherwise look identical to a working one right up until the first real
# RAG query in production came back empty, which is exactly the failure
# mode this build step exists to catch. 50 is comfortably below the 128
# chunks the current 13 source docs produce, leaving room for future doc
# edits without this check becoming the thing that breaks on a harmless
# content change.
RUN python -c "import chromadb; c = chromadb.PersistentClient(path='data/chroma_db').get_collection('project_docs'); n = c.count(); print(f'RAG index check: {n} chunks in project_docs'); assert n >= 50, f'expected at least 50 chunks, got {n}'"

EXPOSE 8000

# Render (and most PaaS Docker hosts) require the app to bind to the $PORT
# they inject, not a fixed port - falls back to 8000 for local `docker run`
# where $PORT isn't set. Exec-form JSON can't expand ${PORT}, so the array
# runs /bin/sh only to read $PORT, and `exec` replaces that shell with
# uvicorn. uvicorn is then PID 1 and receives SIGTERM directly on
# `docker stop` / a Render restart, so it can drain in-flight requests.
CMD ["/bin/sh", "-c", "exec uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
