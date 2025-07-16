import time
from typing import List
from loguru import logger
from .model import get_models
from .template import apply_template, prompt_to_messages, get_tokenizer_config
from .types import (
    ChatCompletionRequest,
    HordeRequest,
    ModelGenerationInput,
    TextGeneration,
    ChatCompletionResponse,
    ModelResponseRequest,
    ModelResponse
)
from .workers import horde_workers

def openai_to_horde(
    request: ChatCompletionRequest,
    origin_ip: str,
    apikey: str,
    max_context_length: int = 2048,
) -> HordeRequest:
    """
    Convert an OpenAI request to a Horde request.

    :param request: The OpenAI request
    :param max_context_length: The maximum context length (not applicable to OpenAI and thus a constant)
    :return: The Horde request
    """
    model_names = [m.strip() for m in request.model.split(",")]
    known_models = get_models()
    primary_model = None
    for model_name in model_names:
        if model_name in known_models:
            primary_model = model_name
            break
    if primary_model is None:
        raise ValueError(f"Model {primary_model} not known!")
    base_model = known_models[primary_model].base_model

    # Fetch all stop words which may be used
    # One should not mix base_models, but if one does, at least stop works
    all_stops = set()
    for model_name in model_names:        
        tokenizer_config = get_tokenizer_config(model_name)
        # TODO: Set all_stops
    max_available_context_length = horde_workers.get_max_context_length_for_model(primary_model)
    max_available_tokens = horde_workers.get_max_tokens_for_model(primary_model)
    if max_available_tokens > 1024:
        max_available_tokens = 1024
    if apikey == "0000000000" and max_available_tokens > 256:
        max_available_tokens = 256
    return HordeRequest(
        prompt=apply_template(request.messages, model_name),
        models=model_names,
        timeout=300 if request.timeout is None else int(request.timeout),
        params=ModelGenerationInput(
            max_context_length=max_context_length if max_context_length is not None and max_context_length <= max_available_context_length else max_available_context_length,
            max_length=request.max_tokens if request.max_tokens <= max_available_tokens else max_available_tokens,
            n=request.n,
            rep_pen=request.frequency_penalty+1 if request.frequency_penalty is not None else 1.0,
            stop_sequence=([] if request.stop is None else request.stop)
            + list(all_stops),
            temperature=request.temperature,
            top_p=request.top_p,
        ),
        origin_ip=origin_ip,
    )

def openai_to_horde_model_response(
    request: ModelResponseRequest,
    origin_ip: str,
    apikey: str,
    max_context_length: int = 2048,
) -> HordeRequest:
    """
    Convert an OpenAI model response request to a Horde request.

    :param request: The OpenAI request
    :param max_context_length: The maximum context length (not applicable to OpenAI and thus a constant)
    :return: The Horde request
    """
    model_names = [m.strip() for m in request.model.split(",")]
    models = get_models()
    primary_model = model_names[0]
    if primary_model not in models:
        raise ValueError(f"Model {primary_model} not known!")

    max_available_context_length = horde_workers.get_max_context_length_for_model(primary_model)
    max_available_tokens = horde_workers.get_max_tokens_for_model(primary_model)
    if max_available_tokens > 1024:
        max_available_tokens = 1024
    if apikey == "0000000000" and max_available_tokens > 256:
        max_available_tokens = 256
    return HordeRequest(
        prompt=request.input,
        models=model_names,
        timeout=300 if request.timeout is None else int(request.timeout),
        params=ModelGenerationInput(
            max_context_length=max_context_length if max_context_length is not None and max_context_length <= max_available_context_length else max_available_context_length,
            max_length=request.max_tokens if request.max_tokens <= max_available_tokens else max_available_tokens,
            n=request.n,
            rep_pen=request.frequency_penalty+1 if request.frequency_penalty is not None else 1.0,
            temperature=request.temperature,
            top_p=request.top_p,
        ),
        origin_ip=origin_ip,
    )


def horde_to_openai(
    request: HordeRequest, *, include_base_stops: bool = True
) -> ChatCompletionRequest:
    """
    Convert a Horde request to an OpenAI request.

    :param request: The Horde request
    :param include_base_stops: Whether to include base stops
    :return: The OpenAI request
    """
    params = request.params
    if params is None:
        raise ValueError("Request params are required")

    base_stops = [
        "<|",
        "<eos>",
        "</s>",
    ]

    return ChatCompletionRequest(
        messages=prompt_to_messages(request.prompt),
        model=request.models[0],
        frequency_penalty=params.rep_pen,
        presence_penalty=None,
        max_tokens=params.max_length,
        n=params.n,
        stop=(
            ([] if params.stop_sequence is None else params.stop_sequence)
            + (base_stops if include_base_stops else [])
        )[:4],
        temperature=params.temperature,
        top_p=None if params.top_p == 1.0 else params.top_p,
        timeout=request.timeout,
    )


def completions_to_openai_response(
    completions: List[TextGeneration],
) -> ChatCompletionResponse:
    """
    Convert a list of completions to an OpenAI response.
    :param completions: List of completions
    :return: OpenAI response
    """
    return ChatCompletionResponse(
        id=completions[0].uuid,
        choices=[
            {
                "finish_reason": "stop",
                "index": index,
                "message": {"role": "assistant", "content": completion.text},
            }
            for index, completion in enumerate(completions)
        ],
        created=int(time.time()),
        model=completions[0].model,
        usage={
            "kudos": completions[0].kudos,
        },
    )

def horde_response_to_openai_model_response(
    completions: List[TextGeneration],
) -> ModelResponse:
    """
    Convert a list of horde responses to an OpenAI model response.
    :param completions: List of completions
    :return: OpenAI response
    """
    return ModelResponse(
        id=completions[0].uuid,
        output=[
            {
                "id": index,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": completion.text, "annotations": []}
                ],
            }
            for index, completion in enumerate(completions)
        ],
        created_at=int(time.time()),
        model=completions[0].model,
        usage={
            "kudos": completions[0].kudos,
        },
    )
