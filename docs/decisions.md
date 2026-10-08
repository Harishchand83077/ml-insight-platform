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

  ## Load test fixes verified (Week 8)

- Fixed 413/429 handling (checked actual Groq SDK exception hierarchy — 
  429 maps to RateLimitError but 413 falls through to generic 
  APIStatusError, so check .status_code directly rather than relying on 
  named exception types alone) and raised connection pool 5→20.
- Re-test: throughput 3.16→7.00 req/s (2.2x), /chat failures 46.75%→6.42%, 
  /predict p99 11s→6.7s, /chat p99 32s→24s. Remaining /chat failures are 
  now clean 503s (working as designed), not opaque 500s.
- New finding at higher throughput: ~10 ConnectionResetErrors, likely 
  uvicorn's single-process dev server (--reload) becoming the next 
  bottleneck. Not fixed — noted as the natural next constraint a 
  production deployment (multi-worker uvicorn/gunicorn) would need to 
  address, out of scope for this load-testing pass.

  ## Supabase migration (Week 8)

- Migrated customers + customer_features only (not raw event tables — 
  deployed app only queries customer_features; event tables already 
  did their job locally).
- Centralized load_dotenv() in db.py so all callers get env vars 
  automatically, fixing a silent gap where migrate script couldn't 
  see SUPABASE_DB_URL.
- Verified: 7043/7043 rows matched exactly on both tables.

## Upstash migration (Week 8)

- Centralized Redis connection in redis_client.py; rediss:// URL 
  auto-configures TLS via redis-py's from_url() — verified from source, 
  not assumed.
- Latency shift confirmed as expected: local hits 2-5ms → Upstash 
  ~28-32ms (network round-trip), still well under Postgres miss latency 
  (~71-78ms). Caching logic unchanged — only connection config differed.
- Found: first /chat call after a fresh process start took several 
  minutes — sentence-transformers model cold-loading, likely a slow 
  HuggingFace Hub connectivity check on first use. Flagged as a real 
  deployment risk (Render's free tier also cold-starts); fix is to bake 
  the embedding model into the Docker image at build time rather than 
  pulling it at first runtime use.


  ## Render deployment prep (Week 8)

- Dockerfile pre-downloads BAAI/bge-small-en-v1.5 at build time + sets 
  HF_HUB_OFFLINE=1, eliminating both the cold-load AND any runtime 
  network check against HuggingFace Hub — fixes the multi-minute 
  first-request stall found during Upstash migration.
- CORS uses allow_origin_regex scoped to *.vercel.app rather than a 
  bare wildcard — with allow_credentials=True, browsers reject a 
  literal "*" anyway, so a scoped regex is both more correct and more 
  secure.
- render.yaml declares secrets as sync: false (no values committed) — 
  set directly in Render's dashboard.
  ## Render + Supabase IPv6 issue (Week 8)

- Direct Supabase connection (db.xxx.supabase.co) resolves IPv6-only; 
  Render's free tier has no outbound IPv6 support, causing "Network is 
  unreachable" at container startup. Fixed by switching to Supabase's 
  Session Pooler connection string (IPv4-compatible, via Supavisor) — 
  no code changes needed, only the connection string.


  ## Model export for deployment (Week 8)

- Found: deployed API tried to load models from MLflow's tracking store 
  (sqlite:///mlruns.db), which only existed locally, never shipped to 
  Render. Fixed by exporting the chosen production run as a standalone 
  model directory (models/production_model/), committed to git and 
  baked into the Docker image — decoupling "what ships" from "the full 
  experiment tracking history."
- mlruns.db and mlruns/ are local development artifacts (every experiment
  run, not just the deployed one), not deployment artifacts. mlruns.db was
  never tracked. mlruns/ had 66 files still tracked in git until the change
  that adds it to .gitignore and untracks it. Only the exported production model 
  belongs in the repo/image.

  ## Backend live on Render (Week 8)

- https://ml-insight-platform.onrender.com — FastAPI + agent + Supabase 
  (Session Pooler) + Upstash + exported production model, all live.
- Two real deployment-specific bugs found and fixed along the way: 
  Supabase's direct connection is IPv6-only (Render's free tier has no 
  outbound IPv6) — fixed via Session Pooler; MLflow tracking store 
  never shipped to the container — fixed by exporting a frozen model 
  artifact instead of querying the tracking store at runtime.

  ## Backend fully live and verified (Week 8)

- https://ml-insight-platform.onrender.com/predict confirmed working 
  end-to-end in production: Supabase (Session Pooler) → feature lookup 
  → Redis/Upstash cache check → model inference → correct response 
  (matches local: 7590-VHVEG, 2.53% churn probability).
- Third deployment bug found/fixed: REDIS_URL env var was empty/unsaved 
  in Render's dashboard (separate issue from the earlier Supabase IPv6 
  problem) — Redis client's from_url() failed with a clear scheme-
  validation error, quickly diagnosed from the traceback.

  ## Full backend verified live (Week 8)

- /predict and /chat both confirmed working end-to-end in production: 
  Supabase, Upstash (feature + semantic cache), Groq, the agent's 
  tool-calling, and RAG all functioning together on Render.
- Semantic cache hit confirmed live (similarity=0.9985) on a repeated 
  question — full caching behavior verified in production, not just locally.

  ## Full system live (Week 8 close)

- https://ml-insight-platform.vercel.app (frontend) → 
  https://ml-insight-platform.onrender.com (backend) → Supabase + 
  Upstash + Groq, fully public and functional.
- CORS locked to the exact production domain, no wildcards remaining.

## Fixed OOM crash from redundant model loading (Week 8)

- Root cause: sentence-transformers embedding model was lazily 
  instantiated separately in 3 places (semantic_cache.py, tools.py, 
  build_knowledge_base.py) — under live request load, this caused a 
  memory spike that crashed the Render instance (confirmed: "Instance 
  failed", health check timeout).
- Fixed by loading the embedding model once at FastAPI startup 
  (lifespan handler, same pattern as the XGBoost model), shared via a 
  dependency-free leaf module (src/common/embedding_model.py) to avoid 
  a circular import between api.py and agent/tools.py.
- Added torch.set_num_threads(1) to reduce CPU/memory contention on 
  Render's constrained free-tier instance.
- Result: repeated model loads eliminated (1 load at startup vs. N per 
  request), per-request latency dropped from ~40s (cold embedding load) 
  to ~0.3s.

  ## Deployment complete and verified stable (Week 8)

- Full stack confirmed live and crash-free under real use: 
  https://ml-insight-platform.vercel.app → 
  https://ml-insight-platform.onrender.com → Supabase, Upstash, Groq.
- Post-fix verification: /chat request completed successfully, semantic 
  cache hit correctly logged, no instance failure or health check 
  timeout — confirms the embedding-model consolidation fix resolved 
  the earlier OOM crash.


  ## Authentication (Week 9)

- JWT auth added: POST /auth/signup, POST /auth/login, get_current_user 
  dependency protecting /predict, /chat, DELETE /chat/{session_id}.
- Security details: generic 401 on login failure (prevents user 
  enumeration via distinguishing error messages), race-guarded duplicate 
  signup (pre-check + unique-constraint fallback, avoiding a TOCTOU 
  race), correct 401 vs 403 semantics (missing token vs bad token).
- audit_logs table records every /predict and /chat call per user, 
  best-effort (logging failure warns, doesn't fail the request).
- Real dependency bug found: passlib's bcrypt backend breaks on 
  bcrypt>=4.1 (probes a removed attribute) — pinned bcrypt==4.0.1.
- Deliberately deferred frontend wiring — Vercel frontend will 401 
  until login UI is built (next task), backend verified standalone first.

  ## Frontend auth (Week 9)

- Login/signup UI added; JWT held in React state only (not localStorage) 
  — deliberate XSS-mitigation trade-off, accepting logout-on-refresh; a 
  production version would use httpOnly cookies instead.
- Axios interceptors centralize auth: one attaches the Bearer token to 
  every request, one handles 401 globally (clears session, returns to 
  login) — new endpoints get both behaviors automatically.
- Verified live against the real deployed backend: signup → chat → 
  refresh (correctly logged out) → re-login → logout.



  ## Liveness/readiness separation (Week 9)

- Fixed repeated deploy failures: Render's port-scan timeout was killing 
  deploys because /health didn't respond until all model loading 
  finished (XGBoost + embedding model, sequential, blocking).
- Fixed by loading models in a background asyncio task (via 
  asyncio.to_thread, since both loaders are blocking I/O) while lifespan 
  yields immediately — /health always returns 200 instantly; /predict 
  and /chat return 503 "still loading" until model_state["ready"] flips 
  true.
- Kept the startup task in a module-level reference — asyncio holds only 
  a weak reference to tasks from create_task(), so an unreferenced task 
  can be garbage-collected mid-execution.

  ## Liveness/readiness fix verified live (Week 9)

- Confirmed in production: /health responded 200 within 1s of container 
  start, well before model loading began — Render marked the service 
  live immediately, no more port-scan timeouts on slow model loads.

  ## Feedback loop (Week 9)

- POST /feedback (auth-protected) stores thumbs up/down per assistant 
  message, tied to user_id and session_id. The rating is validated twice: 
  Literal["up","down"] at the API layer and a CHECK constraint in the DB.
- The UI disables voting after one click per message. Verified by 
  querying Supabase directly with a join back to users, not just by 
  the 200 response.


  ## Production instability under real agent load (Week 9, open)

- Symptom: on Render free tier (0.1 CPU / 512MB), light requests (/predict, 
  semantic-cache hits) work, but a real /chat agent call sometimes kills 
  the instance. Render's stated reason: HTTP health check timed out 
  after 5s.
- Earlier OOM hypothesis is unconfirmed (Render did not report an OOM). 
  Candidate causes: blocked event loop, CPU starvation, memory. 
  Diagnosing by reproducing limits locally (--memory=512m --cpus=0.1) 
  before changing code.
- Frontend now treats 502/503/504/timeouts as "server waking up", 
  polls /health before enabling chat, and preserves input on retryable 
  failures.

## Root cause confirmed: OOM, not CPU (Week 9)

- Reproduced Render's limits locally (--memory=512m --cpus=0.1). At the 
  literal limit, the container OOM-killed during startup (never reached 
  models_ready) after 22m of memory thrashing (497-510MB plateau, block 
  I/O climbing to 5.38GB from page re-faulting). With headroom (900MB), 
  idle post-startup baseline was 749MB - 46% over the real 512MB budget, 
  before serving any request.
- Separate, independent finding: CPU starvation under a real agent call 
  (embedding-heavy work) caused /health latency spikes up to a full 
  timeout, though loop-blocking was ruled out (all handlers are plain 
  def, correctly threadpooled by Starlette).
- Fix: reduce baseline memory (mlflow -> direct skops.io load first, 
  audit serve-time imports; torch/sentence-transformers swap to fastembed 
  only if still needed) and move embedding calls to asyncio.to_thread to 
  stop them starving /health under CPU pressure. Two independent fixes 
  for two independent problems - fixing one doesn't fix the other.


## Memory fix validated, real bug found (Week 9)

- Heaviest-case Docker test (two tool calls, 512MB/0.1 CPU): no OOM, no 
  memory leak across 3 repeated calls (489.7-489.8-489.5 MiB), survived 
  with ~22MB margin. Some /health latency spikes up to 9.2s persist - 
  attributed to OS-level CPU quota throttling (cgroup CFS bandwidth), 
  not something async/threading changes can fully resolve.
- Decision: stop memory optimization here rather than proceed to the 
  fastembed/ONNX rewrite - passed its hardest test twice, additional 
  work carries more risk than the current thin-but-real margin justifies.
- Found: predict_churn_tool has been calling /predict over HTTP with no 
  auth token since JWT auth was added - silently broken in production. 
  Fixing by calling the prediction logic directly in-process instead of 
  over HTTP (removes an unnecessary network hop and the auth mismatch 
  entirely).


  ## predict_churn_tool auth bug fixed (Week 9)

- Fixed: predict_churn_tool called /predict over HTTP with no JWT, 
  broken since auth was added - every agent churn-risk question failed 
  silently. Fixed by calling prediction logic in-process 
  (src/serving/prediction.py) instead of over HTTP, via a shared leaf 
  module (src/common/production_model.py) - same pattern used for the 
  embedder fix, avoids both the auth mismatch and an unnecessary network 
  hop.
- Verified live: agent question for a real customer_id returns a real 
  prediction with tool_calls confirming predict_churn_tool executed.


  ## RAG deepened, deployment gap fixed (Week 9)

- Root cause of the empty-RAG-answers bug: Chroma was never built or 
  shipped in Docker at all (not specific to the deepening work - this 
  was broken since the very first deploy). Chroma silently creates an 
  empty collection for a missing path rather than erroring, so the tool 
  never raised - it just always returned "no relevant context."
- Fixed: Dockerfile now builds the index at build time (COPY docs/, run 
  build_knowledge_base.py, assert >=50 chunks or fail the build) and 
  .dockerignore scoped with an explicit exception rather than a blanket 
  data/ exclusion.
- Expanded to 11 synthetic docs (refund/contract/SLA/pricing policies, 
  retention playbooks, model card, data dictionary), citation format 
  [source: file.md > Section], agent explicitly declines rather than 
  hallucinating when nothing relevant retrieves.
- Eval baseline (17 questions): hit@1 0.79, hit@3 1.0 overall. Exact-term 
  queries weaker (0.67) than paraphrase (0.875) - expected for pure 
  vector search, a natural case for hybrid search (BM25 + vector) as a 
  next step. One unanswerable question scored near the suspicious 
  threshold (0.60) - a legitimate, explainable near-miss (topically 
  adjacent, not actually answering).
- data/chroma_db is a reproducible build artifact, not source, but it is
  still tracked in git (5 files). The intent to ignore it isn't in effect:
  the .gitignore line for it was written as UTF-16 bytes into a UTF-8 file,
  so git doesn't match it. Not yet untracked.


  ## Hybrid search: tested, not adopted (Week 9)

- Built hybrid BM25+vector retrieval, tuned two weightings against the 
  17-question eval set.
- 0.5/0.5 (reports/rag_eval_hybrid_50_50.json): exact-term hit@1 unchanged
  at 0.67, paraphrase hit@1 0.875->0.75, and unanswerable questions flagged
  as suspicious (score above 0.5) rose from 1/3 to 2/3.
- 0.7/0.3 (reports/rag_eval_hybrid.json): exact-term hit@1 0.67->1.0, but
  paraphrase hit@1 dropped 0.875->0.625, and unanswerable questions flagged
  as suspicious rose from 1/3 to 3/3 (all three) - worse, not better.
  BM25's keyword-overlap confidence made the retriever hand back
  confident-looking wrong chunks on questions it should have declined.
- Decision: reverted to vector-only. The regression on unanswerable 
  questions directly undermines the "decline rather than hallucinate" 
  behavior verified in the previous RAG work - not worth trading for 
  gains on a minority of exact-term queries. Hybrid module kept, 
  documented, unused - a real option if the corpus or query mix changes.

## CI broken by eager DB connection, fixed (Week 9)

- The predict_churn_tool in-process fix introduced a new import chain 
  (tools.py -> prediction.py -> feature_cache.py) that connects to 
  Postgres at module import time, breaking CI (no DB available there) 
  even though local/production both have real DB access.
- Fixed: feature_cache.py's connection pool is now lazily initialized 
  on first use, matching the pattern already used for the embedder and 
  model singletons - importing the module no longer requires a live DB.

  ## Import-time environment dependencies fixed across the board (Week 9)

- Found the same bug pattern in three places: feature_cache.py, auth.py, 
  and feedback.py all created their Postgres pool at module import time, 
  plus auth.py raised at import if JWT_SECRET_KEY was unset. Anything 
  importing these modules - including CI's test collection - required 
  live DB access and a real secret just to import, not just to actually 
  use them.
- Fixed: all three pools and the JWT secret check are now lazy, created/
  validated on first real use, matching the pattern already used for the 
  embedder and model singletons. Importing any of these modules now 
  requires nothing; using them still fails clearly if the real 
  dependency (DB, secret) is missing.
- Verified with a strict check that patches psycopg2.connect, 
  SimpleConnectionPool, and the JWT secret lookup to raise if touched 
  during import - confirms the fix at the mechanism level, not just by 
  absence of an error.


  ## Explainability tool (Week 9)

- Built explain_churn_tool (top-5 SHAP-equivalent feature contributions 
  per prediction). Initially used shap.TreeExplainer; measured it adding 
  ~64MB on first call on top of an already-tight 512MB budget - switched 
  to XGBoost's native booster.predict(pred_contribs=True), which gives 
  identical values (verified: max difference 0.0) with no new dependency 
  and no extra memory cost.
- Found and fixed a correctness issue: the LLM's narration added 
  unsupported causal/emotional language ("frustrated customers") not 
  justified by the contribution values, which are associational, not 
  causal. Constrained the system prompt to factual framing only.
- Contributions are in log-odds units, additive across features 
  (verified against the model's raw margin to 1.7e-5) - not directly 
  interpretable as "probability points."


  ## Explainability tool, finalized (Week 9)

- Switched shap.TreeExplainer -> XGBoost native booster.predict(pred_contribs=True): 
  identical values (max diff 0.0 across all features/totals/base on both 
  test customers), memory impact 64MB -> 2.8MB, cold-start latency 5.4s -> 322ms, 
  warm median 52ms (dominated by Redis lookup, not computation).
- Narration constrained to factual framing via system prompt; verified on 
  fresh (non-cached) test questions - speculative causal language 
  ("frustrated", "higher expectations") eliminated on tested inputs. 
  Caveat: this is prompt-level guidance, not a hard guarantee - could 
  still drift on untested phrasings.
- Caught own testing error mid-verification: initial "fix confirmed" 
  re-test was actually a semantic cache hit returning pre-fix cached 
  text, not fresh output - re-tested with varied wording before trusting 
  the result.


  ## What-if tool (Week 9)

- simulate_churn_tool: re-scores a customer with overridden account-level 
  fields (contract, payment method, etc. - not behavioral event features, 
  which aren't user-controllable inputs). Validates against observed 
  training ranges/allowed values before any model call - rejects 
  nonsense inputs (invalid category, out-of-range numeric, wrong type) 
  with a clear error instead of a silent bad prediction.
- Verified: 2691-NZETQ's risk drops 0.9989->0.9415 on a hypothetical 
  2-year contract switch, direction consistent with known EDA (month-to-
  month 42.7% vs two-year 2.8% churn) - the modest size of the drop makes 
  sense given this customer's score is dominated by a different feature 
  (unresolved ticket time), a good example of why single-feature 
  counterfactuals don't always move risk as much as intuition suggests.
- Testing gap found twice now: semantic cache silently invalidates 
  "verification" done through near-identical /chat questions. Added 
  [cache bypass mechanism] to prevent a third recurrence.

  ## Semantic cache test bypass (Week 9)

- Added X-Cache-Bypass header, gated by a server-side CACHE_BYPASS_TOKEN 
  env var (unset by default - disabled unless explicitly configured). 
  Constant-time comparison, wrong token returns 403 rather than silently 
  ignoring the header. Bypassed requests skip both cache read and write, 
  so test traffic never pollutes the shared cache.
- Not set on Render - local/testing convenience only, no reason to exist 
  in production.
- Documented in CLAUDE.md for future sessions, since this testing gap 
  had already caused two invalid verification rounds before being caught.


  ## Fourth instance of import-time env dependency, fixed (Week 9)

- Same bug class as feature_cache/auth/feedback: agent.py instantiated 
  ChatGroq with GROQ_API_KEY at module import time, breaking any test 
  that imports it (api.py -> agent.py chain) without a real key set - 
  this is what broke CI this round (test_cache_bypass.py's collection).
- Fixed with the same lazy-getter pattern used for the other three. 
  Four occurrences of the same mistake in one codebase is worth noting 
  as a lesson: anything doing I/O or reading a required secret at module 
  scope should be treated as suspect by default, not just caught 
  reactively.
- Dockerfile CMD switched to exec form - uvicorn is now PID 1 and 
  receives SIGTERM directly (verified: clean exit 0 vs. prior SIGKILL 
  137 after models are ready). Known gap, not fixed: a stop signal 
  during model loading still doesn't shut down cleanly, since the 
  embedding-model worker thread keeps the process alive - unlikely in 
  practice since Render restarts happen after readiness, not during 
  startup, but a real gap if that assumption ever breaks.


## agent.py import-time fix verified in CI-equivalent environment (Week 9)

- Same fix as the other three (lazy get_agent(), RuntimeError instead of 
  KeyError on missing key). Verified properly this time: simulated CI's 
  actual conditions (empty environment, no .env, copied source to a 
  clean directory) rather than relying on a strict-import check alone - 
  this is what caught test_agent.py's stale import before it became a 
  fifth break.
- Dockerfile CMD in exec form - uvicorn is PID 1, clean SIGTERM/exit 0 
  after startup completes.


## Repo hygiene pass (Week 9 close)

- Found committed local artifacts: 7 log/test files at the repo root, plus 
  a SQLite DB, generated CSVs, and HTML reports. Scanned the logs for 
  secrets first (none found; only localhost Redis URIs) before untracking.
- Untracked everything generated or local; kept inputs and deployment 
  artifacts (raw dataset, models/production_model, RAG eval JSONs).
- Verified with a clean-checkout simulation (no .env, no data/): unit 
  suite and lint pass, and the Dockerfile copies no untracked path.
- .gitignore had UTF-16 corruption in three places (found by checking 
  the file byte by byte after the first instance).

  ## Semantic cache correctness (Week 9)

- Hypothesis: identical questions about different customers collide. 
  Measured: they don't (0.83-0.84 vs 0.90 threshold). The real collision 
  was a what-if pair differing only in contract term (0.975): bge-small 
  barely moves on category words, so a similarity threshold can't 
  separate them from legitimate rewordings (0.91).
- Fix: skip the cache for ID-bearing questions; 24h TTL so entries 
  expire after data or KB changes. 55 unit tests.
- Open: ID-less data-dependent questions (e.g. churn rate by contract) 
  may still collide. Measuring next.
- Found during live testing: pooled connections go stale when Supabase 
  closes idle ones (intermittent 500 in auth). Fixing with keepalives, 
  discard-on-error, and retry for reads only.

  ## Pool resilience and cache scope (Week 9)

- Supabase closes idle connections; the shared pool handed them back out 
  (intermittent 500). Fix: TCP keepalives, discard connections on 
  error, retry once for idempotent reads only, ping-before-write for 
  writes (a blind retry could double-apply a write whose ack was lost). 
  Verified live by terminating the backend pid mid-session.
- Measured ID-less collisions: worst 0.907 vs lowest legitimate hit 
  0.910, so no threshold separates them. Changed what's cached instead: 
  only answers that used no data tools. Trade-off: fewer cache hits, 
  no wrong cached numbers.
- CI run #29 failed on GitHub infrastructure (runner never acquired, 
  server error), not on code; re-ran.

  ## Atomic feature-table rebuild (Week 9)

- Hypothesis: /predict could read an empty or partial customer_features 
  during a rebuild. Measured with a 100ms reader loop: no (0 zero-row 
  reads) because Postgres TRUNCATE is transactional.
- Real cost: TRUNCATE's exclusive lock blocked readers for the whole 
  insert (3 reads over 250ms, max 648ms locally).
- Fix: build customer_features_new, swap by rename in one transaction. 
  0 blocked reads, max 11ms; a forced mid-insert failure left the old 
  table untouched (same checksum, no staging table left behind).
- Not changed: feature cache TTL, so features can be up to 300s stale 
  after a rebuild.

  ## Production-readiness audit (Week 9)

- Read-only audit of the deployed path, with file:line evidence. Must-have 
  gaps: no Redis timeouts or fallback (a Redis outage would 500 every 
  /predict), no effective agent iteration limit (default 10007), Groq 
  timeouts and SQL tool errors surfacing as 500s, unpooled SQL tool 
  connections, no wait on pool exhaustion, untested failure paths.
- Noted a contradiction in the audit itself: /health shares the 
  40-thread pool, so a hung Redis could stall it and fail Render's health 
  check.
- Deliberately not built: negative caching, stampede coalescing, 
  partitioning, circuit breaker, Redlock. Can explain when each matters.

  ## Connection budget and failure handling (Week 9)

- Supabase pooler limits: pool size 15, 200 max clients, session mode 
  (port 5432). Our pools could hold up to 50 connections, plus local runs 
  sharing the same 15. Consolidating to one pool and testing transaction 
  mode (evidence before choosing).
- Deployed code had no Postgres connect timeout; a stalled connect 
  blocks a request for minutes. Fixed with connect_timeout=5.
- Empty final model turn on a two-tool question (5 of 6 runs, not caused 
  by the recursion-limit change; finish_reason stop, 101 tokens). Added 
  retry-once plus fallback, a log of response metadata, and a counter.

## Connection budget, resolved (Week 9)

- Measured Supabase poolers with 30 concurrent connections: session mode 
  (5432) rejected 15 of 30 at exactly pool_size 15 (EMAXCONNSESSION), 
  instantly, no queueing. Transaction mode (6543) accepted 28-29, with 1-2 
  transient SSL drops under the simultaneous-connect burst.
- Grepped for what transaction mode breaks (session SET, prepared 
  statements, advisory locks, LISTEN/NOTIFY): nothing found; the only SET 
  is SET LOCAL inside a transaction.
- Consolidated four pools (up to 50 connections) into one bounded pool of 
  10. A checkout waits up to 2s, then raises DBBusyError (HTTP 503).
- Local Locust before/after: no failures either way, p95 90→83ms, p99 
  300→150ms. Did not exercise exhaustion; unit tests do.
- Retracted a hypothesis: I suspected pool exhaustion caused the 
  Supabase stalls seen in testing. Exhaustion fails instantly; the 
  stalls were connect hangs, now bounded by connect_timeout=5.
- Empty model answers: 5/6 earlier, 0/30 later; cause unknown. Added 
  retry, fallback and a counter instead of a guess.

## CI caught a mock gap hidden by local services (Week 9)

- After consolidating to a shared pool, 22 tests failed in CI: they 
  patched run_read, but the tools fetch the pool first, so the real pool 
  tried to connect. Passed locally only because Postgres was running on 
  localhost. 
- Fix: autouse guard in tests/unit/conftest.py that fails any unit test 
  attempting a real DB or Redis connection, identically on every machine. 
  Verified with services stopped and in a clean environment before pushing.
- Lesson: "passes on my machine" is meaningless when the machine has live 
  services. The clean-environment simulation has to be re-run after every 
  change that touches connections, not just once.
  Update README.md and docs/ARCHITECTURE_SUMMARY.md to reflect the final 
state. Take every number from docs/decisions.md and reports/, not from 
memory:

1. Known limitations: replace stale items. Add what's now true: Redis 
   fails open (0.5s/1.0s timeouts), shared DB pool of 10 under Supabase's 
   session limit of 15 with a 2s bounded wait returning 503, agent 
   recursion limit 12, Groq timeout 30s. Keep what's still limitations: 
   512MB headroom, in-memory chat sessions, ingestion not deployed.
2. Add a short "Reliability" section: the failure modes tested and the 
   measured results (OOM before/after, session-pooler exhaustion test, 
   semantic-cache collision measurement, atomic rebuild).
3. Confirm the tool count and the live URLs are correct.
4. Don't add any claim that isn't backed by a decisions.md entry.

## Live stream, stage 1 (Week 10)

- Local generator plus control API publish churn-conditioned events 
  (with an optional drift mode) into RabbitMQ and local Postgres. A 
  guard refuses to run against any non-local database host, so the 
  stream can't touch production.
- Consumer is now idempotent: event_id unique index, INSERT ON CONFLICT 
  DO NOTHING, ack only after commit. Verified by killing the consumer 
  mid-stream: 235 redelivered messages, published == landed on all 
  three tables, zero duplicates.
- Refactored the batch generator into pure functions; regenerated 
  CSVs were byte-identical to the originals.
- Open risk found before stage 2: live events use wall-clock time while 
  historical data is from 2024, which can distort windowed features. 
  Fix: a simulated clock and an as_of parameter.

  ## Live stream, stage 2 (Week 10)

- Simulated clock, repeatable reset (two resets give identical table 
  checksums), atomic feature rebuild, drift endpoint.
- Drift-off control run flagged two columns (recent_30d_vs_older_60d_ratio, 
  avg_session_duration_recent_30d): as the simulated clock advances 
  past the historical data they become constant 0 for all customers 
  (the window needs ~1,167 events/sec at 1 sim-day/sec; the generator 
  does ~30). Excluded from the drift view with the reason visible in 
  the response. The other 14 columns stayed under 0.0087 vs the 0.10 
  threshold.
- Open risk for stage 3: those two columns are model inputs, so scoring 
  or retraining on the rebuilt table could mislead the promotion gate. 
  Measuring before building.

  ## Live stream, stage 3 pre-check (Week 10)

- Measured whether the simulated clock's collapsed window columns 
  distort model comparison: no. Two 60s drift-off trials moved PR-AUC by 
  -0.0040 and -0.0067 and F1 by +0.0108 and +0.0051; the columns rank 
  17th and 20th of 23 (1.87% of gain). Baseline reproduced the trained 
  model's PR-AUC (0.7819) exactly.
- Consequence: the model relies on contract (45%) and internet service 
  (19%); event-feature drift is unlikely to degrade it. Data drift is 
  not performance drift, so the promotion gate measures performance.
- Plan: two scenarios (covariate drift: rejected by the gate; concept 
  drift with simulated delayed labels: candidate wins, promoted). 
  Gate uses a paired bootstrap interval plus a minimum margin.


  ## Live stream, stage 3a: scenarios and gate (Week 10)

- Covariate drift: production PR-AUC barely moved at any strength 
  (+0.0006, -0.0069, +0.0104, all intervals include zero); retraining 
  doesn't help because the X->Y relationship is unchanged.
- Concept drift (simulated labels, local-only table): a full-population 
  rule change makes the candidate win (+0.105, interval [0.065, 0.142]). 
  Partial relabeling results (-0.134, -0.0376) are unverified.
- Gate: paired bootstrap interval must exclude zero and the point 
  difference must be >= 0.02. Paired SE measured 0.013-0.020; the single 
  model's PR-AUC SE on the 1,409-row holdout is 0.0215.
- Known weakness: the drift view's detections came from total_logins_90d, 
  a volume-sensitive count, with no equal-volume drift-off control. 
  Replacing it with event-level distribution drift.

  ## Live stream, stage 3b: promotion gate built (Week 10)

- Gate implemented: frozen shared holdout, paired bootstrap interval 
  must exclude zero, point difference >= 0.02, plus a serving-compat 
  check that reloads the exported model through the API's own loader 
  before any promotion.
- The compat check initially failed on rounding alone (API output is 
  rounded to 4 decimals; tolerance was 1e-6). Caught by the live check, 
  not unit tests; fixed by comparing full-precision outputs.
- Verified live against a temp model directory: Scenario B at 100% 
  promoted (version 1 visible in /health); baseline-vs-baseline rejected 
  with reasons, /promote returned 409. Real model directory untouched.
- Promotion is a manual git push by design; rollback script restores 
  models/production_model from any commit.