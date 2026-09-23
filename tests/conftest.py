"""
Makes integration tests opt-in: without --run-integration, anything marked
@pytest.mark.integration is skipped (not silently passed, not run). Unit
tests are unaffected and always run.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="also run tests marked @pytest.mark.integration (require live Postgres/Redis/MLflow/Groq)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="requires --run-integration (needs live external services)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
