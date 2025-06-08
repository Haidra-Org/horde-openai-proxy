__all__ = [
    "openai_to_horde",
    "horde_to_openai",
    "get_horde_completion",
    "completions_to_openai_response",
    "TextGeneration",
    "get_models",
    "Model",
    "apply_template",
    "get_generation_config",
    "GenerationConfig",
    "ChatCompletionRequest",
    "HordeRequest",
    "ChatCompletionResponse",
    "ModelResponseRequest",
    "ModelResponse",
    "get_horde_models",
    "ModelGenerationInput",
    "filter_models",
    "openai_to_horde_model_response",
    "horde_response_to_openai_model_response",
]

import os

import huggingface_hub
from dotenv import load_dotenv

from .conversion import completions_to_openai_response, horde_to_openai, openai_to_horde, openai_to_horde_model_response, horde_response_to_openai_model_response
from .horde import TextGeneration, get_horde_completion, get_horde_models
from .model import Model, get_models
from .template import (
    GenerationConfig,
    apply_template,
    get_generation_config,
    
)
from .types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelResponseRequest,
    ModelResponse,
    HordeRequest,
    ModelGenerationInput,
)
from .utils import filter_models

load_dotenv()
print('aaaa')
print(os.getenv("HF_TOKEN"))
if os.getenv("HF_TOKEN"):
    huggingface_hub.login(token=os.getenv("HF_TOKEN"))
