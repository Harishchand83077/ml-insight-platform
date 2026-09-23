# here we will note down every non obvious decisions 

## EDA findings (Day 1)

- Churn rate is 26.5% — imbalanced. Will use precision/recall/F1/PR-AUC
  instead of accuracy as primary model metrics in Week 3.

- TotalCharges has 11 blank values, all with tenure=0 (new customers,
  no billing cycle yet). Decision: impute as 0, not drop or mean-impute —
  these are legitimately zero-charge customers, not missing data.

- Strongest churn signals found: Contract type (month-to-month 42.7%
  churn vs two-year 2.8%), tenure (newest customers churn most, 47%
  in first year vs 9.5% after 4+ years), and lack of TechSupport/
  OnlineSecurity add-ons.

- Counterintuitive finding: Fiber optic customers churn more (42%)
  than DSL (19%) despite being a "premium" service — flagging for
  further investigation rather than assuming data error. Possible
  pricing/competition effect.

  ## EDA findings (Day 1, continued)

- No duplicate customerIDs — data integrity confirmed.

- TotalCharges is highly collinear with tenure (r=0.83) and
  MonthlyCharges (r=0.65) — expected, since it's roughly their product.
  Decision: [pick one — e.g. "drop TotalCharges from modeling features,
  derive tenure and MonthlyCharges as the independent signals instead"]

- All 6 add-on service columns (OnlineSecurity, OnlineBackup, etc.)
  share the same 3-value pattern, with "No internet service" being a
  redundant restatement of InternetService=No. Decision: engineer
  num_addons_active (count of Yes across the 6) and drop redundant
  categorical duplication of "No internet service" per column.

- Gender shows ~no churn gap (27% vs 26%) — sanity check confirming
  earlier signals (contract, tenure, add-ons) are real, not noise
  from an overly permissive method.

- MonthlyCharges boxplot: churners have higher median charge, but
  non-churners show a much wider low-end spread — likely a sticky
  low-cost/single-service segment. Worth a targeted look at customers
  with MonthlyCharges < 30 in feature engineering.


  ## rebbitMQ vs reddis for live streming synthetic data 
  I'm going with RabbitMQ here, and reserve Redis for the caching/rate-limiting/semantic-cache jobs  Reasoning: RabbitMQ is a proper message broker (durable queues, exchanges, routing keys, ack/nack, dead-letter queues) — (how do you guarantee delivery, what happens on consumer crash, how do you route by event type). 
  Redis Streams can do pub/sub, but it's lighter-weight and you'd be underusing Redis's actual strength (caching) while missing out on RabbitMQ's richer failure-handling story. Using both, each for its natural job, also just looks more deliberate in an interview than cramming everything into one tool.
  now to setup and use rabbit mq i used docker for this
  docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
  Create and run a RabbitMQ server inside a Docker container, and make it accessible from my Windows machine
  1. docker run- it tells docker to create a new container from an image and start it(a container is an isolated env where an application can run like here we run rabbit mq)
  2. -d means detach mode without this our terminal will occupied by rabbit mq logs so by this rabbit mq will run in bg and we will get our terminal back(you can open docker desktop and see container id)
  3. --name rabbitmq - just give this container a name so taht in docker commands we have not to use that long container id
  4. -p 5672:5672 - this is port mapping implies host machine is running on local host 5672 and docker container also on 5672  also 5672 is RabbitMQ's normal AMQP messaging port.
  5. -p 15672:15672-This exposes RabbitMQ's management web interface.so that we can open this http://localhost:15672 and see the rabbit mq ui use guest as login and password

  6. rabbitmq:3-management this is docker image we are asking docker to use


  ## Synthetic event generation (Day 2)

- Generated 3 correlated event types (login, support tickets, feature 
  usage) seeded with np.random.default_rng(42) for reproducibility.
- Validated correlations hold as designed: churned customers show 
  declining login trend (0.31 vs 0.62 recent/older ratio), lower 
  ticket resolution rate (55% vs 89%), lower feature usage (3.0 vs 
  7.0 avg count) vs non-churned.
- Generated to CSV first (not streamed) to validate data quality 
  independent of the messaging layer.



## Event streaming (Day 2, continued)

- Migrated synthetic CSVs through RabbitMQ (durable queues) into SQLite as a placeholder store, validating the full ingestion pipeline before introducing Postgres.
- All row counts matched exactly (139,973 / 12,799 / 120,178) — no data loss or duplication through the queue.
- Found and fixed a commit-per-row performance bug (~100x slower)(hotspot or high load on db so use batch processing took 100 messages from queue and commit all at a time ) and a related bug where a final batch smaller than the commit threshold never flushed. Switched to batched commits with an idle-timeout flush.so let say our consumer process messages in batches and only  commit after  100 rows  and let say last batch has only 37 so consumer waiting to complete it to 100 with infinite time so we add time threshold also

## Postgres migration (Day 2, continued)

- Replaced SQLite with Postgres 16 (Docker, named volume for persistence) 
  as the durable event store. Same producer/consumer logic, proper column 
  types (TIMESTAMP, BOOLEAN, DOUBLE PRECISION) instead of TEXT.
- Re-verified full pipeline end-to-end: exact row-count match across all 
  three tables, no duplicates or loss.
- Kept SQLite version as a reference checkpoint rather than deleting it.


## Feature engineering (Week 2)

- Built customer_features table (7043 rows, 17 columns) joining static 
  customer data with 3 event tables via left joins (all customers kept, 
  zero-events filled as 0, not dropped/imputed as mean).
- num_addons_active engineered from 6 raw addon columns; raw addon 
  columns and total_charges dropped (redundant/collinear, per EDA).
- Unresolved tickets have NULL resolution_time_hours by design — mean 
  naturally excludes them rather than treating as 0, which would 
  understate resolution time.


  ## Data leakage caught and fixed (Week 2)

- Initial baseline models scored unrealistically high (F1 ~0.96-0.99, 
  PR-AUC ~0.99). Root cause: synthetic event generator made churned 
  customers near-deterministically have declining logins/low usage/
  unresolved tickets, so features nearly encoded the label directly.
- Fixed by widening the distributions so churned/non-churned behavior 
  overlaps significantly while preserving correct directional 
  correlation. Re-trained; metrics now in a realistic range.



  ## Leakage fix validated (Week 2)

- Reworked synthetic generator to use per-customer randomized trends/noise 
  (churn-conditional probabilities with wide overlap) instead of 
  deterministic group-level rules.
- Re-trained on rebuilt features: F1 dropped from ~0.97 to 0.66-0.69, 
  PR-AUC from ~0.99 to ~0.77-0.78 — now in a realistic range for churn 
  prediction.
- Feature importance shifted from event-derived features dominating to 
  Contract/InternetService dominating, matching original EDA findings — 
  good sign the pipeline is measuring real signal, not an artifact.


  ## Redis caching (Week 2)

- Implemented cache-aside pattern for customer feature lookups: check 
  Redis first, fall back to Postgres on miss, populate cache on the way 
  back. TTL set to [X]s — [your stated reasoning].
- Measured [X]ms cache hit vs [X]ms cache miss latency on local setup.

## Redis caching validated (Week 2)

- Cache-aside pattern confirmed working: 10-92x speedup on cache hits 
  (2-5ms) vs misses (45-194ms). TTL 300s.
- Miss latency inflated by opening a fresh Postgres connection per call 
  (no pooling) — acceptable for validation, flagged as a real 
  improvement for the serving layer (connection pooling via psycopg2 
  pool or SQLAlchemy engine).

  ## Serving layer + bug fixes (Week 3)

- Found and fixed a serialization bug: MLflow was logging only the bare 
  XGBClassifier, not the fitted preprocessing pipeline (one-hot encoding). 
  Serving would have silently produced wrong predictions on raw input. 
  Fixed by logging the full sklearn Pipeline.
- Built /predict endpoint (FastAPI): model loaded once at startup via 
  lifespan hook, not per-request.
- Replaced per-call Postgres connections with a connection pool 
  (SimpleConnectionPool). Result: hit/miss gap narrowed from 10-90x 
  (standalone cache test) to near-identical (~200-240ms both) once 
  pooled — remaining latency is now FastAPI/HTTP overhead and 
  predict_proba, not the DB round trip. Honest finding: at this scale, 
  connection pooling mattered more than the cache itself for reducing 
  end-to-end latency.

  ## Drift monitoring + async retraining (Week 3 close)

- Evidently drift report: injected shifts in 3 numeric columns, 2/3 
  correctly flagged (avg_usage_count, avg_session_duration_recent_30d). 
  avg_resolution_time_hours (+20% shift) did NOT trigger — its high-
  variance/right-skewed distribution dampens normalized drift score for 
  a proportional shift. Dataset-level drift not flagged (12.5% of 
  columns, below 50% threshold) — a reminder that per-column and 
  dataset-level drift are different questions.
- Celery worker required --pool=solo on Windows (prefork pool relies on 
  fork(), unsupported natively). This means sequential task execution, 
  not real concurrency — acceptable for demo, would need Linux deployment 
  or --pool=threads for production concurrency.
- Async retrain verified: task triggered via .delay(), completed in 156s, 
  confirmed via fresh MLflow run (not a no-op).

  ## Dependency drift encountered (Week 5)

- LangChain 1.0 removed create_tool_calling_agent/AgentExecutor mid-build; 
  adopted current create_agent (LangGraph-based) pattern instead of 
  pinning to deprecated 0.3.x.
- Originally planned Groq model (llama-3.3-70b-versatile) no longer 
  available on the API; switched to openai/gpt-oss-120b, the largest 
  available general-purpose model on the free tier.


  ## First working agent tool (Week 5)

- predict_churn_tool verified end-to-end: agent correctly extracts 
  customer_id from natural language, calls /predict, synthesizes a 
  clean answer from the JSON response. Second call confirmed cache_hit, 
  validating the agent exercises the full serving stack, not a shortcut.
- Fixed a UTF-8/cp1252 console encoding crash on Windows when printing 
  LLM output containing non-ASCII characters (e.g. non-breaking hyphens) 
  — common real-world issue serving LLM output on Windows consoles.



  ## SQL tool with injection defense (Week 5)

- Added get_churn_rate_by_column and get_customer_count as constrained 
  tools instead of freeform text-to-SQL. Column names (identifiers) 
  validated against a static allowlist and interpolated directly — 
  parameterization only works for values, not identifiers, so this is 
  the correct defense, not just an extra check. Filter values go through 
  psycopg2 parameterization.
- Verified injection attempt ('DROP TABLE...') is rejected by the 
  allowlist.
- Agent correctly mapped "month-to-month" (a value) to the contract 
  column (schema-grounded reasoning, not keyword matching) and reported 
  42.7% without fabrication.


  ## RAG knowledge base (Week 6)

- Built local RAG pipeline: glossary.md + decisions.md → chunked (~500 
  tokens) → embedded with local sentence-transformers (all-MiniLM-L6-v2) 
  → stored in Chroma. Zero-cost, fully local except the LLM call itself.
- Found: embedding model truncates at 256 tokens, so 500-token chunks 
  lose the tail from the embedding vector (full text still stored/
  retrieved). Documented; [decide: left as-is / reduced chunk size / 
  switched to bge-small-en-v1.5].
- query_project_docs_tool returns raw retrieved context, not a 
  synthesized answer — keeps grounding visible to the main agent.

  ## Embedding model fix (Week 6)

- Switched from all-MiniLM-L6-v2 (256-token limit, truncated our 500-token 
  chunks) to BAAI/bge-small-en-v1.5 (512-token window, full chunk coverage). 
  Rebuilt Chroma store from scratch — embeddings across models aren't 
  compatible, can't be merged into an existing index.



  ## Chat endpoint (Week 6)

- Fixed a same-directory vs package import mismatch (agent.py assumed 
  standalone execution, broke when imported via uvicorn's package path) 
  with a try/except import fallback.
- POST /chat + DELETE /chat/{session_id}: in-memory per-session history 
  (explicitly not production-durable — no restart survival, not 
  multi-instance safe; Redis-backed sessions would be the real fix).
- Verified multi-turn memory: agent recalled a specific customer's churn 
  probability across turns without re-querying, correctly declined to 
  fabricate data it had no tool for, and tool_calls extraction correctly 
  scopes to only the current turn, not cumulative history.


  ## Semantic response caching (Week 6 close)

- Implemented brute-force cosine similarity cache (not a vector DB — 
  unnecessary at this scale), using the same embedding model as RAG. 
  Capped at 200 entries in Redis, threshold 0.90.
- Restricted to first-message-of-session only: multi-turn context 
  changes question meaning, so caching mid-conversation would risk 
  returning wrong cached answers. Verified this holds even for a 
  near-identical rephrase in an ongoing session (correctly bypassed).
- Verified via logs + Redis LLEN, not response inspection alone: 
  confirmed genuine cache hit (0.9502 similarity, zero Groq calls) vs. 
  genuine miss (real Groq call logged) in both directions.

  ## Testing (Week 7)

- 38 unit tests, all mocked (no live Postgres/Redis/Groq needed), 
  0 failed, 4 integration tests correctly gated behind --run-integration.
- Refactored build_features.py and load_static_data.py to extract pure 
  functions (build_feature_table, clean_total_charges) from I/O-bound 
  main() functions — writing tests surfaced logic/I/O coupling that 
  needed separating.
- Coverage: 18% of full src/ (expected — most modules need live infra, 
  covered by integration tests instead), but 49-88% on the pure-logic 
  modules unit tests actually target (semantic_cache, build_features, 
  SQL allowlist, load_static_data).
- SQL injection tests verify rejection happens before any DB connection 
  is attempted (psycopg2.connect patched to raise if reached), not just 
  that the final query string looks safe.


  ## CI/CD (Week 7 close)

- GitHub Actions: lint (ruff, errors only) + unit tests + coverage, 
  plus a separate Docker build-verification job. Both passing 
  (run 35891156103).
- Split requirements.txt (full dev environment) from 
  requirements-docker.txt (curated runtime-only deps, ~50 vs ~280 
  packages, CPU-only torch) — smaller, faster, more correct serving 
  image; dev tools have no business in a production container.
- Real incident during this work: force-killing Docker's processes to 
  clear a stuck build left an orphaned WSL2 instance holding the data 
  disk, breaking Docker Desktop entirely. Fixed non-destructively 
  (stopped Docker Desktop's app first so it wouldn't keep relaunching 
  the stuck WSL instance, then wsl --shutdown, then relaunched) — 
  verified zero data loss (all 7,043 Postgres rows, all containers/
  volumes intact) before proceeding.
  

## Monitoring (Week 7 close)

- Prometheus + Grafana: automatic HTTP metrics (via 
  prometheus-fastapi-instrumentator) plus 3 custom metrics placed at 
  their true source (not endpoint handlers) — feature cache hit/miss, 
  semantic cache hit rate, agent response time (isolated from raw HTTP 
  latency, excludes cache-hit shortcuts).
- Measured agent latency: p50 ~1.4s, p95/p99 ~2.5s — real Groq round-trip 
  variance, not application overhead.
- Distinction from Evidently: Prometheus/Grafana = system health (is 
  the service fast/up), Evidently = model health (is the data still 
  valid) — two different monitoring questions, two different tools.

## Load testing (Week 8 start)

- Locust: 20 users, 2min, 60/40 /predict-/chat split. Found two real 
  issues: (1) /chat 46.75% failures under load — Groq free-tier rate 
  limit propagating as unhandled 500s, no graceful degradation; 
  (2) /predict 0% failures but p99=11s despite p50=33ms — classic 
  queuing signature from POOL_MAX_CONN=5 exhaustion under concurrent load.
- Fixed: wrapped agent invocation in try/except returning 503 on 
  rate-limit; raised connection pool size. Re-tested: [fill in new 
  numbers after re-run].
- Semantic + feature caches both climbed to 80-90% hit rate under load, 
  measurably absorbing pressure that would otherwise hit Groq/Postgres 
  directly.