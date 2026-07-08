from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host != "huggingface.co"
)
def test_get_chat_models(
    app_server: "TestClient", mock_horde_model, mock_text_model_reference
):
    models = app_server.get("/v1/chat/models").json()
    assert models == [
        {
            "name": "Henk717/airochronos-33B",
            "clean_name": "airochronos",
            "base_model": "alpaca",
            "template": "alpaca",
            "backend": "Henk717",
            "quant": "full",
            "size": 33.0,
            "known_to_horde": True,
        },
    ]


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host != "huggingface.co"
)
def test_post_chat_completion(app_server: "TestClient", httpx_mock: "HTTPXMock"):
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/async",
        method="POST",
        json={"id": "test-uuid-123"},
    )
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/status/test-uuid-123",
        method="GET",
        json={
            "done": False,
            "is_possible": True,
            "faulted": False,
            "kudos": 0,
            "generations": [],
        },
    )
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/generate/text/status/test-uuid-123",
        method="GET",
        json={
            "done": True,
            "is_possible": True,
            "faulted": False,
            "kudos": 42,
            "generations": [
                {
                    "text": "Hello, how can I help you?",
                    "model": "Henk717/airochronos-33B",
                }
            ],
        },
    )
    resp = app_server.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "Henk717/airochronos-33B",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "created": ANY,
        "id": "test-uuid-123",
        "usage": {"kudos": 42},
        "model": "Henk717/airochronos-33B",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you?",
                },
            },
        ],
    }
