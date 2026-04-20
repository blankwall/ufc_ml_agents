"""Shared pytest config for the integration tests.

Adds --base-url CLI flag so you can target localhost vs production:

    pytest tests/ --base-url http://localhost:8765
    pytest tests/ --base-url http://107.175.94.166:8002

Falls back to SITE_URL env var, then to the production URL.
"""
import os


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        action="store",
        default=None,
        help="Base URL of the FastAPI server to test against "
             "(overrides SITE_URL env var). e.g. http://localhost:8765",
    )


def pytest_configure(config):
    """Set SITE_URL before test modules import — they read it at module load."""
    cli = config.getoption("--base-url")
    if cli:
        os.environ["SITE_URL"] = cli
