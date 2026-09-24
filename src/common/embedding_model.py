"""
Holds the one shared sentence-transformers embedding model instance
(BAAI/bge-small-en-v1.5), used by both the RAG tool
(src/agent/tools.py:query_project_docs_tool) and the semantic cache
(src/agent/semantic_cache.py) to embed short pieces of text.

Both call sites used to lazily instantiate their own HuggingFaceEmbeddings
(and therefore their own SentenceTransformer) on first use. On Render's
free tier (512MB RAM), a single live request that touched both code paths
could momentarily hold two full copies of the model in memory - confirmed
to be enough to crash the instance ("Instance failed", health check
timeout). Loading exactly one instance, once, removes that spike
entirely.

This module is a plain, dependency-free leaf: it doesn't import
src.serving.api or src.agent.agent, so src/agent/tools.py and
src/agent/semantic_cache.py can both import it without a circular import
(they're each imported *by* api.py/agent.py, directly or transitively).
src/serving/api.py's lifespan handler calls set_embedder() exactly once,
at startup, before the app accepts any requests; tools.py and
semantic_cache.py call get_embedder() per-request, which by then is
always already populated.
"""

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedder = None


def set_embedder(embedder):
    global _embedder
    _embedder = embedder


def get_embedder():
    if _embedder is None:
        raise RuntimeError(
            "Embedding model not initialized - src.serving.api's lifespan "
            "handler must call set_embedder() before any request runs."
        )
    return _embedder
