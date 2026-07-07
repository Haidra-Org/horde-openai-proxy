import time
from typing import List

from .model import get_models
from .template import apply_template, get_generation_config, prompt_to_messages
from .types import (
    ChatCompletionRequest,
    HordeRequest,
    ModelGenerationInput,
    TextGeneration,
    ChatCompletionResponse,
)


def openai_to_horde(
    request: ChatCompletionRequest,
    max_context_length: int = 2048,
) -> HordeRequest:
    """
    Convert an OpenAI request to a Horde request.

    :param request: The OpenAI request
    :param max_context_length: The maximum context length (not applicable to OpenAI and thus a constant)
    :return: The Horde request
    """
    model_names = [m.strip() for m in request.model.split(",")]
    models = get_models()
    primary_model = model_names[0]
    if primary_model not in models:
        raise ValueError(f"Model {primary_model} not known!")
    base_model = models[primary_model].base_model

    # Fetch all stop words which may be used
    # One should not mix base_models, but if one does, at least stop works
    all_stops = set()
    for model_name in model_names:
        if model_name in models:
            all_stops.update(
                get_generation_config(models[model_name].base_model).stop_words
            )

    return HordeRequest(
        prompt=apply_template(request.messages, base_model),
        models=model_names,
        timeout=300 if request.timeout is None else int(request.timeout),
        params=ModelGenerationInput(
            max_context_length=max_context_length,
            max_length=request.max_tokens,
            n=request.n,
            rep_pen=request.frequency_penalty,
            stop_sequence=([] if request.stop is None else request.stop)
            + list(all_stops),
            temperature=request.temperature,
            top_p=request.top_p,
        ),
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
