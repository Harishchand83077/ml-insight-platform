import os
import socket
import uuid

import pytest


def _port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def postgres_up():
    return _port_open("localhost", 5432)


@pytest.fixture(scope="session")
def redis_up():
    return _port_open("localhost", 6379)


@pytest.fixture(scope="session")
def groq_key_present():
    from dotenv import load_dotenv

    load_dotenv()
    return bool(os.environ.get("GROQ_API_KEY"))


@pytest.fixture(scope="module")
def auth_headers(client):
    """Signs up a throwaway test user (unique email per test run, so
    re-running the suite never collides with a previous run's user) and
    returns an Authorization header for it - /predict and /chat both
    require this. Depends on a `client` fixture, which each integration
    test module that uses this one defines for itself."""
    email = f"pytest-{uuid.uuid4()}@example.test"
    signup = client.post("/auth/signup", json={"email": email, "password": "pytest-integration-test-pw"})
    assert signup.status_code == 201, signup.text
    return {"Authorization": f"Bearer {signup.json()['access_token']}"}
