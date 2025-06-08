from typing import List
from typing import Annotated
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from loguru import logger
from fastapi import Header
from horde_openai_proxy import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelResponseRequest,
    ModelResponse,
    Model,
    get_horde_completion,
    openai_to_horde,
    openai_to_horde_model_response,
    completions_to_openai_response,
    horde_response_to_openai_model_response,
    filter_models,
)

app = FastAPI()


@app.get("/v1/models")
def get_chat_models(
    names: str = "",
    clean_names: str = "",
    base_models: str = "",
    templates: str = "",
    min_size: float = 0,
    max_size: float = -1,
    quant: str = "",
    backends: str = "",
) -> List[Model]:
    return filter_models(
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
def post_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    authorization:  Annotated[str | None, Header()] = None,
) -> ChatCompletionResponse:
    logger.debug(authorization)
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = authorization.lstrip("Bearer ")
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token missing")
    try:
        horde_request = openai_to_horde(body)
        logger.debug(horde_request)
        completions = get_horde_completion(token, horde_request)
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e))

    return completions_to_openai_response(completions)

@app.post("/v1/responses")
def post_chat_completion(
    request: Request,
    body: ModelResponseRequest,
    authorization:  Annotated[str | None, Header()] = None,
) -> ModelResponse:
    logger.debug(authorization)
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = authorization.lstrip("Bearer ")
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token missing")
    try:
        horde_request = openai_to_horde_model_response(body)
        completions = get_horde_completion(token, horde_request)
    except ValueError as e:
        raise HTTPException(status_code=406, detail=str(e))

    return horde_response_to_openai_model_response(completions)
