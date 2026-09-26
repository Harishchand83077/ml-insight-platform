"""
Integration tests against the real FastAPI app: GET /health, the
signup/login/JWT auth flow, and POST /predict (both unauthenticated -
expect 401 - and with a real token). Requires postgres-ml and redis-ml
running (docker-compose or the individual `docker run` commands in
README.md); skips with a clear message otherwise rather than silently
passing.
"""

import uuid

import pytest

pytestmark = pytest.mark.integration

KNOWN_CUSTOMER_ID = "7590-VHVEG"


@pytest.fixture(scope="module")
def client(postgres_up, redis_up):
    if not postgres_up:
        pytest.skip("Postgres not reachable on localhost:5432 - start the postgres-ml container first")
    if not redis_up:
        pytest.skip("Redis not reachable on localhost:6379 - start the redis-ml container first")

    try:
        from src.serving.api import app
    except Exception as e:
        pytest.skip(f"Could not import src.serving.api (missing .env, mlruns.db, or a logged xgboost run?): {e}")

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_without_token_returns_401(client):
    response = client.post("/predict", json={"customer_id": KNOWN_CUSTOMER_ID})
    assert response.status_code == 401


def test_predict_returns_valid_response_shape(client, auth_headers):
    response = client.post("/predict", json={"customer_id": KNOWN_CUSTOMER_ID}, headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"customer_id", "churn_probability", "prediction", "cache_hit"}
    assert data["customer_id"] == KNOWN_CUSTOMER_ID
    assert isinstance(data["churn_probability"], (int, float))
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in ("Yes", "No")
    assert isinstance(data["cache_hit"], bool)


def test_predict_unknown_customer_returns_404(client, auth_headers):
    response = client.post("/predict", json={"customer_id": "not-a-real-customer-id"}, headers=auth_headers)
    assert response.status_code == 404


def test_signup_login_and_predict_flow(client):
    email = f"pytest-flow-{uuid.uuid4()}@example.test"
    password = "a-perfectly-fine-test-password"

    signup = client.post("/auth/signup", json={"email": email, "password": password})
    assert signup.status_code == 201
    assert signup.json()["token_type"] == "bearer"

    # signing up again with the same email is rejected
    dup_signup = client.post("/auth/signup", json={"email": email, "password": password})
    assert dup_signup.status_code == 409

    login = client.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    token = login.json()["access_token"]

    wrong_password = client.post("/auth/login", json={"email": email, "password": "not-the-right-password"})
    assert wrong_password.status_code == 401

    predict = client.post(
        "/predict", json={"customer_id": KNOWN_CUSTOMER_ID}, headers={"Authorization": f"Bearer {token}"}
    )
    assert predict.status_code == 200
