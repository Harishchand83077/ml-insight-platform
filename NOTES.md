we have done basic eda on data and now we have to add synthetic data to it we have 2 options for this either use one time simpler add synthetic data or use live streaming synthetic data so that it looks like real time thing -so i am going with 2nd option and for this a am using rabbit mq as if arriving live (matches the event-driven design goal)
i am using pika - Python library that allows Python to communicate with RabbitMQ
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
  we generally don't create tcp connection for every message instead  Channels allow RabbitMQ clients to multiplex communication over a connection.

  basic rabbit mq flow-  
  Producer → Exchange → Routing Key → Queue → Consumer → ACK → Retry/DLQ → Scaling
  An exchange receives messages from producers and routes them to queues.
  imp exchange types-
  | Type    | Basic idea                     |
| ------- | ------------------------------ |
| Direct  | Exact routing key match        |
| Topic   | Pattern-based routing          |
| Fanout  | Broadcast to all bound queues  |
| Headers | Route based on message headers |
A routing key is information used by the exchange to decide which queue should receive the message.
ACK- means sucessfully processed the message ,NACK- could not process.
With auto_ack=True, RabbitMQ considers the message handled immediately, which can cause message loss if processing fails afterward.
Durable queue: Queue survives RabbitMQ restart.
Persistent message: Message is stored in a way intended to survive broker restart.
Suppose a message repeatedly fails:DLQ stores messages that couldn't be successfully processed Useful for debugging and preventing one bad message from continuously blocking processing.

--Consumer Failure / Redelivery
Suppose:Queue → Consumer
Consumer receives a message but crashes before ACK.RabbitMQ can make that unacknowledged message available again.So:ACK is important because it tells RabbitMQ the message has been successfully processed.

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