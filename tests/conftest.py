"""Shared test fixtures."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so tests do not depend on the directory pytest is started from."""
    return REPO_ROOT
