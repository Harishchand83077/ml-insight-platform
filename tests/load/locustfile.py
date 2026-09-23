"""
Load test for the FastAPI service.

- 60% of requests: POST /predict with a random customer_id from a
  pre-loaded list of real customer_ids (queried from customer_features
  once at startup).
- 40% of requests: POST /chat, each with a brand-new session_id (so
  every call is the first message of its session) and a question drawn
  from a rotating set that deliberately exercises all three agent tools,
  not just prediction lookups:
    - predict_churn_tool  (churn risk for a specific customer)
    - the SQL-style query tools (get_customer_count / get_churn_rate_by_column)
    - the RAG tool over the project's own docs

Note: because every /chat call is a fresh session (first message), the
semantic cache is in play - after the first occurrence of each question
text gets cached, later repeats of the same question (there are only 5
in rotation) should increasingly hit the semantic cache instead of
calling the agent/LLM, which is itself a useful thing to observe under
load.

Run:
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
    --users 20 --spawn-rate 2 --run-time 2m --headless \
    --html=reports/load_test_report.html
"""

import random
import uuid

import psycopg2
from locust import HttpUser, between, task

PG_DSN = {
    "host": "localhost",
    "port": "5432",
    "dbname": "ml_insight",
    "user": "postgres",
    "password": "devpassword",
}


def load_customer_ids(limit=100):
    conn = psycopg2.connect(**PG_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT customer_id FROM customer_features ORDER BY random() LIMIT %s", (limit,))
        ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return ids


CUSTOMER_IDS = load_customer_ids()

CHAT_QUESTIONS = [
    "What is the churn risk for customer {customer_id}?",  # predict_churn_tool
    "How many customers are on a two year contract?",  # SQL tool: get_customer_count
    "What percentage of month-to-month customers churn?",  # SQL tool: get_churn_rate_by_column
    "Why did you choose PR-AUC over accuracy for this project?",  # RAG tool
    "What does num_addons_active mean?",  # RAG tool
]


class ChurnApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(60)
    def predict(self):
        customer_id = random.choice(CUSTOMER_IDS)
        self.client.post("/predict", json={"customer_id": customer_id}, name="/predict")

    @task(40)
    def chat(self):
        question = random.choice(CHAT_QUESTIONS).format(customer_id=random.choice(CUSTOMER_IDS))
        self.client.post(
            "/chat",
            json={"session_id": str(uuid.uuid4()), "message": question},
            name="/chat",
        )
