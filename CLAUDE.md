# Testing notes for this repo

## Semantic cache bypass (read this before testing /chat)

`/chat` short-circuits the first message of each session through a semantic
cache (`src/agent/semantic_cache.py`, cosine similarity >= 0.90 over the stored
questions). Two differently worded test questions can match each other, so a
test can receive a replayed answer and pass or fail for the wrong reason.
Changing the wording is not a reliable fix: a previously asked question can
still match above the threshold.

To bypass the cache for a test, send the header `X-Cache-Bypass` with the value
of the server's `CACHE_BYPASS_TOKEN`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "X-Cache-Bypass: $CACHE_BYPASS_TOKEN" \
  -d '{"session_id":"t1","message":"..."}'
```

Behavior:

- A bypassed request skips both the cache read and the cache write, so test
  runs don't add entries to the shared cache.
- If `CACHE_BYPASS_TOKEN` is unset on the server, the bypass is disabled. Any
  `X-Cache-Bypass` header then returns 403.
- A wrong token returns 403. It is not silently ignored.
- Set `CACHE_BYPASS_TOKEN` only in your local shell or a local `.env`. Never
  commit its value or put it in a file in this repo.

To confirm whether a request hit the cache, check the server log for
`SEMANTIC CACHE HIT`.

## Other test notes

- Unit tests (`pytest tests/unit`) mock Redis, Postgres, and Groq. They don't
  need any of those services.
- `tests/eval/rag_eval.py` needs a built Chroma index. Rebuild it with
  `python src/agent/build_knowledge_base.py` first.
- The Docker image is built with `docker build -t ml-insight-api .`. Its build
  step asserts at least 50 chunks in the RAG index.
