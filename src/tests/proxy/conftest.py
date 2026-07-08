import pytest
from fastapi.testclient import TestClient

from horde_openai_proxy.proxy import app


@pytest.fixture
def app_server():
    return TestClient(app)
