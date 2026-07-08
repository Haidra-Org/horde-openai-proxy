import pytest
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_horde_model(httpx_mock: "HTTPXMock"):
    httpx_mock.add_response(
        url="https://stablehorde.net/api/v2/status/models",
        match_params={"type": "text", "min_count": "1"},
        json=[
            {
                "performance": 10,
                "queued": 0,
                "jobs": 0,
                "eta": 0,
                "type": "text",
                "name": "Henk717/airochronos-33B",
                "count": 1,
            }
        ],
        is_optional=True,
    )


@pytest.fixture
def mock_text_model_reference(httpx_mock: "HTTPXMock"):
    httpx_mock.add_response(
        url="https://raw.githubusercontent.com/db0/AI-Horde-text-model-reference/main/db.json",
        json={
            "Henk717/airochronos-33B": {
                "name": "Henk717/airochronos-33B",
                "model_name": "airochronos-33B",
                "baseline": "",
                "parameters": 33000000000,
                "description": "",
                "version": "1",
                "style": "generalist",
                "nsfw": False,
                "display_name": "airochronos 33B",
                "url": "https://huggingface.co/Henk717/airochronos-33B",
                "tags": ["33B"],
            },
        },
        is_optional=True,
    )
