import os
import socket

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
