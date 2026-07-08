from typing import List

from fastapi import FastAPI, HTTPException, Request

from horde_openai_proxy import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Model,
    openai_to_horde_async,
    completions_to_openai_response,
    get_horde_completion_async,
)
from horde_openai_proxy.utils import filter_models

app = FastAPI()


@app.get("/v1/chat/models")
async def get_chat_models(
    names: str = "",
    clean_names: str = "",
    base_models: str = "",
    templates: str = "",
    min_size: float = 0,
    max_size: float = -1,
    quant: str = "",
    backends: str = "",
) -> List[Model]:
    return await filter_models(
        set(n.strip() for n in names.split(",") if n.strip()),
        set(n.strip() for n in clean_names.split(",") if n.strip()),
        set(n.strip() for n in base_models.split(",") if n.strip()),
        set(n.strip() for n in templates.split(",") if n.strip()),
        set(n.strip() for n in backends.split(",") if n.strip()),
        set(n.strip() for n in quant.split(",") if n.strip()),
        min_size=min_size,
        max_size=max_size,
    )


@app.post("/v1/chat/completions")
async def post_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    token = request.headers["authorization"].lstrip("Bearer ").lstrip("sk-")

    try:
        horde_request = await openai_to_horde_async(body)
        # TODO: Pass req params into this
        completions = await get_horde_completion_async(token, horde_request)
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e))

    return completions_to_openai_response(completions)
