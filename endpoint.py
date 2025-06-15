from typing import List
from typing import Annotated
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from loguru import logger
from fastapi import Header
from fastapi import Response
from horde_openai_proxy import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelResponseRequest,
    ModelResponse,
    Model,
    HeartbeatResponse,
    get_horde_completion,
    openai_to_horde,
    openai_to_horde_model_response,
    completions_to_openai_response,
    horde_response_to_openai_model_response,
    filter_models,
)
from starlette.middleware.cors import CORSMiddleware # Import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import requests
from fastapi.logger import logger
from fastapi.responses import HTMLResponse
import markdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # Add CORSMiddleware as the first argument
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["*"] # You'll likely want to specify origins, not just allow all
)

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
) -> dict:
    return {
        "object": "list",
        "data": filter_models(
        set(n.strip() for n in names.split(",") if n.strip()),
        set(n.strip() for n in clean_names.split(",") if n.strip()),
        set(n.strip() for n in base_models.split(",") if n.strip()),
        set(n.strip() for n in templates.split(",") if n.strip()),
        set(n.strip() for n in backends.split(",") if n.strip()),
        set(n.strip() for n in quant.split(",") if n.strip()),
        min_size=min_size,
        max_size=max_size,
    )}


@app.post("/v1/chat/completions")
def post_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
    authorization:  Annotated[str | None, Header()] = None,
):  
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    token = authorization.lstrip("Bearer ")
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token missing")
    try:
        horde_request = openai_to_horde(body, origin_ip=request.client.host, apikey=token)
        completions = get_horde_completion(token, horde_request)
    except ValueError as err:
        logger.error(f"Error processing request: {err}")
        raise HTTPException(status_code=406, detail=str(err))

    if body.stream:
        logger.debug("Faking Streaming response")
        async def event_generator():
            import time
            import json


            openai_response = completions_to_openai_response(completions)
            if hasattr(openai_response, "model_dump"):
                gen = openai_response.model_dump()
            else:
                gen = openai_response

            now = int(time.time())
            friendlymodelname = gen.get("model", "unknown")
            message = gen['choices'][0]['message']
            content = message.get('content', '')

            # First chunk: send the content
            content_chunk = json.dumps({
                "id": gen.get("id", "koboldcpp"),
                "object": "chat.completion.chunk",
                "created": now,
                "model": friendlymodelname,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": None
                }]
            })
            yield {"data": content_chunk}

            # Tool calls, if present
            toolsdata_res = []
            try:
                toolsdata_res = message.get('tool_calls', [])
                if toolsdata_res and len(toolsdata_res) > 0:
                    toolsdata_res[0]["index"] = 0
            except Exception:
                toolsdata_res = []

            if toolsdata_res:
                toolsdata_p1 = json.dumps({
                    "id": gen.get("id", "koboldcpp"),
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": friendlymodelname,
                    "choices": [{
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": toolsdata_res
                        }
                    }]
                })

                toolsdata_p2 = json.dumps({
                    "id": gen.get("id", "koboldcpp"),
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": friendlymodelname,
                    "choices": [{
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {}
                    }]
                })

                yield {"data": toolsdata_p1}
                yield {"data": toolsdata_p2}
            else:
                # If no tool calls, send finish_reason
                done_chunk = json.dumps({
                    "id": gen.get("id", "koboldcpp"),
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": friendlymodelname,
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "delta": {}
                    }]
                })
                yield {"data": done_chunk}

            yield {"data": "[DONE]"}
        return EventSourceResponse(event_generator())
    else:
        return completions_to_openai_response(completions)

@app.post("/v1/responses")
def post_model_response(
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
        horde_request = openai_to_horde_model_response(body, origin_ip=request.client.host)
        completions = get_horde_completion(token, horde_request)
    except ValueError as err:
        logger.error(f"Error processing request: {err}")
        raise HTTPException(status_code=406, detail=str(err))

    return horde_response_to_openai_model_response(completions)

@app.get("/heartbeat")
def heartbeat() -> HeartbeatResponse:
    """
    Simple heartbeat endpoint to check if the service is running.
    :return: True if the service is running
    """
    hb = requests.get("https://aihorde.net/api/v2/status/heartbeat", timeout=5)
    if hb.status_code != 200:
        return HeartbeatResponse(message="AI Horde Error")
    if hb.json().get("message") != "OK" or hb.json().get("db_connection") != True:
        return HeartbeatResponse(message="AI Horde DB Error")
    return HeartbeatResponse(message="OK")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.md", "r") as f:
        md_content = f.read()
    style = """<style>
        body {
            max-width: 120ex;
            margin: 0 auto;
            color: #333333;
            line-height: 1.4;
            font-family: sans-serif;
            padding: 1em;
        }
        </style>
    """
    html_content = f"{style}{markdown.markdown(md_content)}"
    return HTMLResponse(content=html_content)