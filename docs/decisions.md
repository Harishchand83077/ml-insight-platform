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

