we have done basic eda on data and now we have to add synthetic data to it we have 2 options for this either use one time simpler add synthetic data or use live streaming synthetic data so that it looks like real time thing -so i am going with 2nd option and for this a am using rabbit mq as if arriving live (matches the event-driven design goal)





initially we tested things on sqlite but now moving to postgresql
|                    | SQLite                    | PostgreSQL                   |
| ------------------ | ------------------------- | ---------------------------- |
| Type               | Relational DB             | Relational DB                |
| Architecture       | Embedded                  | Client-server                |
| Data storage       | `.db` file                | Database server storage      |
| Server required    | ❌ No                      | ✅ Yes                        |
| Typical use        | Local/small applications  | Production/backend systems   |
| Concurrent users   | Limited                   | Much better                  |
| Deployment         | Extremely simple          | More infrastructure          |
| Data types         | Relatively flexible       | Stronger typing              |
| Networking         | Not designed as DB server | Designed for network clients |
| Scaling            | Limited                   | Much better                  |
| Your current setup | `events.db`               | Docker PostgreSQL            |

Docker PostgreSQL
We run PostgreSQL inside a Docker container instead of installing PostgreSQL directly on the computer.
postgres-ml = name of the PostgreSQL container.
5432 = default PostgreSQL port.
ml_insight = PostgreSQL database we create.
pgdata = Docker named volume used to persist PostgreSQL data.
Why -v pgdata:/var/lib/postgresql/data?
PostgreSQL stores its database files inside the container.
A container can be deleted/recreated.
Without a volume, database data could be lost when the container is removed.
The volume stores the data outside the container, so the data survives container recreation.
Container = PostgreSQL environment/process; Volume = persistent data.


docker exec postgres-ml psql -U postgres -d ml_insight -c

Means:

docker exec → Run a command inside a running Docker container.
postgres-ml → The name of your PostgreSQL container.
psql → PostgreSQL's command-line tool.
-U postgres → Connect as the postgres user.
-d ml_insight → Connect to the ml_insight database.
-c → Execute the SQL/psql command that comes after it.
docker exec -it postgres-ml psql -U postgres -d ml_insight in this we used -it to open an interactive terminal session inside docker container so that we can run multiple commands


docker exec <container> <command>

# CI/CD, Redis & Production Engineering Notes

## 1. What is CI?

**CI = Continuous Integration**

CI automatically checks code whenever developers push code or create a Pull Request.

Typical CI pipeline:

Developer pushes code
        ↓
GitHub Actions triggered
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Lint
        ↓
Run unit tests
        ↓
Measure test coverage
        ↓
Build Docker image
        ↓
PASS / FAIL

### Why CI?

CI catches bugs and broken builds before code is merged or deployed.

### Interview answer

> "Continuous Integration automatically builds and tests code whenever changes are pushed or submitted through a pull request. It helps catch regressions early and ensures that the codebase remains in a working state."

---

# 2. GitHub Actions

GitHub Actions is a CI/CD platform integrated with GitHub.

Workflow files are stored under:

.github/workflows/

Example:

.github/workflows/ci.yml

GitHub automatically detects workflow files in this directory.

### Typical triggers

- `push` → run when code is pushed
- `pull_request` → run when a PR is created/updated

Example concept:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]



CI runs on a clean machine, so it does not automatically have the packages installed on your computer.

Therefore CI installs:

pip install -r requirements.txt

The requirements file should contain the project's required Python dependencies.
Python Version Consistency

Keep Python versions consistent across:

Local development
CI
Docke

Linting

A linter analyzes source code for potential errors, bugs, bad practices, and sometimes style problems.

For modern Python projects, Ruff is a good simple choice.

Example:

ruff check src tests

Linting is different from testing.

Linting

Checks the source code statically.

Testing

Actually executes code and checks behavior.

I keep unit tests separate from integration tests because unit tests should be isolated and reproducible. Integration tests require external services and credentials, so I run them separately unless those dependencies are explicitly provisioned in C

pytest is a Python testing framework.

Example:

pytest tests/unit

means:

Run the tests inside tests/unit.

A failing test should cause CI to fail.

GitHub Actions Artifacts

An artifact is a file generated during a CI run that can be stored and inspected later.

Examples:

Coverage HTML report
Test report
Build output
Logs

Docker Build in CI

Passing unit tests does not guarantee that the Docker image can be built.the application may still fail when deploy so ci should also run docker build


Connection Pooling

Connection pooling means maintaining a group of reusable database connections instead of creating a new connection for every database operation.

Without pooling:

Event
 ↓
Create DB connection
 ↓
Query
 ↓
Close

Repeated thousands of times.

With pooling:

             Connection Pool
          ┌────┬────┬────┬────┐
          │ C1 │ C2 │ C3 │ C4 │
          └────┴────┴────┴────┘
                  ↓
              PostgreSQL

Application:
Get connection
 ↓
Use connection
 ↓
Return connection to pool



# notes for understanding main topics
1.rabbit mq-
so basically we have used this to work on live stream of customer event data (logins,tickets,feature usage) instead of loading simple csv  we connect producer to live stream of data it take and sent it to queue and rabbit mq push that data to consumer/workers do processing on it ..it is event driven architecture
i am using pika - Python library that allows Python to communicate with RabbitMQ
core concept-
1.we have producer and consumer ..here producer publish messages and consumer recieve messages and process them and they never talk to each other directly -rabbitmq sits bw them 
2.connection vs channel -basically we will setyup one real tcp connection using blockingconnection to the broker(rabbit mq) and then use multiple lightweight virtual paths multiplexed over that one tcp connection so that we don't have to pay the cost of new connection per operation
3.basic rabbit mq flow-  
  Producer → Exchange → Routing Key → Queue → Consumer → ACK → Retry/DLQ → Scaling
An exchange receives messages from producers and routes them to queues.and which is decided by routing key so that which messahe goes to which queue .we have used default exchaneg(exchange="") in both event_producer.py and event_producer.py which route message to queue acc to name matching here we does not apply fanout /topic logic as we only have 3 queues (one per event type)
4.Durable queue vs Persistent message — two separate guarantees that both matter together:
channel.queue_declare(queue=queue_name, durable=True) → the queue itself survives a RabbitMQ restart.
pika.BasicProperties(delivery_mode=2) → the message is written to disk, not just kept in memory.
If you only had one of the two, a restart could still lose data.
5. ack/nack-acknowledgment and non acknowledge the consumer tells rabbitmq -i sucessfully processed this message -ack or i could not-nack so until ack rabbitmq keeps this message ..and if the consumer crashes before ack rabbitmq redeliver it to another consumer this is how rabbit mq prevent to loose messahe when a worker dies
6.Prefetch (QoS) — controls how many unacked messages RabbitMQ will hand a consumer at once, so one slow consumer doesn't get flooded while others starve.
 
our implimentatio- in our event_producer.py file basically we are generating synthetic data ffor 3 part and store them in csv file with their queue names and then in event producer it takes csv rows and convert strings to json messages and then sent them to producer row wise 
we had declared all 3 queues durable and then publish with delivery mode =2 and then publisher sent message to queues and here one queue per event type instead of one queue with type feild so from this we have simpler consumer logic as each queue maps to one postgres table 

in event_consumer.py -this is important part(it's work is to take message from arbbit mq and process them bascally upload them to repective postgreq table)
Receive event-->Process event-->Insert into PostgreSQL-->Commit transaction-->Acknowledge RabbitMQ message
-for cold start it assumes that postgres tables already exist and then setup a connection with rabbitmq and then call-channel.basic_qos(prefetch_count=COMMIT_EVERY * 2) basically let say we have a batch of commit_everty=200 then prefetch will be 400 and then 200 sends and 200 waiting and we are sending these in batch so that we don't have to do all the process like processing one by one sequentially and then return ack one by one for every mssg higher throughput ..instead process a batch and commit and use return multiple=True so all batch ack true instead of one by one ack and we also use flush ideal sec let say in some batch we only left 37 mssg then instead of waiting to complete 200 mssges batch size ..will waiting forever we will setup idel time to wait and if that's time over then also process those 37 message 
we speciall use ack after commit let say we do first ack and then commit and let say for any message we ack and then consumer crashes then mssg lost 
here idempotency also matter let say a message processed but before ack the consumer crashes so the message remains in queue and agian send to consumer and it agin process and duplication occur so we take care of this by idempotancy and don't again reprocess when same message came

connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))-BlockingConnection means the Python program maintains a connection and waits for operations/messages.
channel = connection.channel() channel is a lightweight connection path inside rabbit maq connection
Python
  │
  │ TCP connection
  ▼
RabbitMQ
  │
  ├── Channel 1
  ├── Channel 2
  └── Channel 3

  imp exchange types-
  | Type    | Basic idea                     |
| ------- | ------------------------------ |
| Direct  | Exact routing key match        |
| Topic   | Pattern-based routing          |
| Fanout  | Broadcast to all bound queues  |
| Headers | Route based on message headers |
A routing key is information used by the exchange to decide which queue should receive the message.

With auto_ack=True, RabbitMQ considers the message handled immediately, which can cause message loss if processing fails afterward.

Suppose a message repeatedly fails:DLQ stores messages that couldn't be successfully processed Useful for debugging and preventing one bad message from continuously blocking processing.


--multiple consumers at a time 
RabbitMQ can distribute messages among consumers.This allows parallel processing and scaling.
Example:If you have 10,000 customer events, multiple consumers can process them instead of one consumer doing everything sequentially.

| RabbitMQ                                    | Kafka                                |
| ------------------------------------------- | ------------------------------------ |
| Message broker                              | Distributed event streaming platform |
| Queue-based messaging                       | Log/partition-based streaming        |
| Strong routing capabilities                 | Very high-throughput streaming       |
| Great for task/event messaging              | Great for large-scale event streams  |
| Messages are typically consumed and removed | Events can be retained and replayed  |

what to look in rabbit mq ui-
| Metric                | What to ask                                                |
| --------------------- | ---------------------------------------------------------- |
| **Ready**             | Is backlog growing or shrinking?                           |
| **Unacked**           | Are consumers processing/ACKing normally?                  |
| **Incoming**          | How fast are messages arriving?                            |
| **Deliver**           | How fast are consumers receiving them?                     |
| **ACK**               | How fast are they successfully completing them?            |
| **Redelivered**       | Are messages failing/retrying?                             |
| **Consumer capacity** | Are consumers being kept busy?                             |
| **Consumers**         | Do we have enough processing workers?                      |
| **Prefetch**          | Are we giving consumers too many/few outstanding messages? |
| **Durable**           | Will the queue survive broker restart?                     |
| **Persistent**        | Will messages be stored for restart durability?            |
| **DLQ**               | Where do repeatedly failed messages go?                    |

---What you'd say in an interview, in one breath
"I used RabbitMQ to simulate a live event stream instead of a static batch load — producer publishes durable, persistent JSON messages per event type; the consumer batches inserts and acks in groups of 200 (or on a 2-second idle flush) for throughput, and only acks after the Postgres commit succeeds, so a crash mid-batch causes safe redelivery instead of data loss, never double-counted-and-lost data."


--push vs pull modal-
push modal -in this th ebroker decide when to hand a consumer a message ,the consumer just register a callback and waits
pull modal-here the consumer explicitely ask the broker -give me the next messages ,the broker never initiates
in rabbitmq we used push mechanism ,in event_consumer.py we call channel.basic_consume(queue=..., on_message_callback=...)  this register a callback and rabbitmq pushes messages to it as they arrive ,restricted only by prefetch count ,it is a cap on how much unack message the broker will push ahead of the processing ,consumer never ask for messages ,broker decides when to send one
(Pika does have basic_get for a true pull — call it once, get one message or nothing, no callback. You didn't use it, and it's rarely used because it doesn't scale — no continuous flow, one round-trip per message.)
Kafka (pull) — the contrast
A Kafka consumer calls poll() in a loop and the client library fetches a batch from the broker at a time and offset it chooses. The consumer tracks its own offset, so it can replay from any earlier point.


### kafka vs rabbit mq
Pros / cons
                    Push (RabbitMQ)	                               Pull (Kafka)
Latency	Lower — broker sends as soon as ready	Slightly higher — bounded by poll interval
Backpressure	Needs manual tuning (prefetch_count), broker can still overwhelm a slow consumer if misconfigured	Natural — consumer only fetches what it can handle, at its own pace
Replay	Not really — message is gone once acked	Yes — offset-based, can rewind and reread
Multiple independent readers of same stream	Awkward — a queue's messages are removed once consumed	Natural — each consumer group has its own offset
Broker responsibility	Broker tracks per-message ack/retry/DLQ state	Broker just stores a log; consumer tracks progress
Which is "better"
Neither — it matches the job:

Push/RabbitMQ fits task distribution and command/event processing where you want the broker to manage per-message delivery guarantees (ack, retry, DLQ) — which is exactly your use case: each event needs to land in Postgres exactly-ish once, and you want RabbitMQ handling redelivery on crash.
Pull/Kafka fits high-throughput streaming and analytics where replay and multiple independent consumers reading history matter more than per-message delivery tracking.



### postgres
in start we used sqlite to validate the ingestion and rabbit mq producer and consumer works fine or not and then we migrate to postgres ,SQLite is embedded (one file, one process) — fine for validation, but no real concurrency and not how you'd run a production backend
postgres is like a client server multiple processes(our consumer,api,feature build script) can all connect to same running db
in our case we are doing 3 jobs in postgres
1. event store-we ahve login event ,support tickets ,feature usage logs..written by event consumer
2. feature store -customer_features built by build_features.py in this we simply left join of static customers table with 3 event tables
3. app tables -auth(auth.py) and feedback.py each keep their small pooled connection to postgres

in our codebase we have src/common/db.py -this is shared helper file and every module directly import this instead of each building it's own connection dict . as get_database_url() return one connection string not a dict as psycopg2 .connect() and pool.SimpleConnectionPool() both accept the same DSN string as their first argument. One code path serves both single-connection and pooled callers.
it checks both db url and supabase db url and if both not then fallback to building a url from local pg_host/PG_port 