from dataclasses import dataclass
from typing import Optional

import requests
from cachetools import TTLCache, cached

from .data import MODEL_SIZES, MODEL_TO_BASE_MODEL, BASE_MODELS
from .horde import get_horde_models
from loguru import logger


QUANTS = {
    "q2_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_0",
    "q4_k_s",
    "q4_k_m",
    "q5_0",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
    "q8_0",
    # I-Quants
    "iq1_m",
    "iq2_m",
    "iq2_s",
    "iq2_xs",
    "iq2_xxs",
    "iq3_m",
    "iq3_xs",
    "iq4_xs",
}

IGNORED_WORDS = {"GGUF"}


def estimate_clean_name(name: str) -> str:
    """Estimate the clean name of a model, without quant, size, backend, ..."""
    name = name.split("/")[-1]
    filtered = []
    for part in name.split("-"):
        if len(part) <= 4 and part.lower().endswith("b"):
            continue
        if part.lower() in QUANTS:
            continue
        if part.lower() in IGNORED_WORDS:
            continue
        filtered.append(part)
    return "-".join(filtered)


def find_similar_reference(name: str) -> str:
    """Estimate another name which might appear on the reference"""
    name = name.split("/")[-1]
    return "-".join(name.split("-")[:2])


def estimate_quant(name: str) -> str:
    """Estimate the quantization of a model based on the name."""
    name = name.lower()
    for q in QUANTS:
        if q in name:
            return q
    return "full"


def estimate_size(name: str) -> float:
    """Estimate the size of a model based on the name."""
    short_name = name.split("/")[-1]
    if name in MODEL_SIZES:
        return MODEL_SIZES[name]
    if short_name in MODEL_SIZES:
        return MODEL_SIZES[short_name]
    for segment in name.lower().split("-"):
        for s in segment.split("."):
            if s.endswith("b"):
                try:
                    return float(s[:-1])
                except ValueError:
                    pass
    logger.warning(f"Unknown size: {name}")
    return 0


def guess_base_model(name: str) -> Optional[str]:
    """Guess the base model of a model based on the name."""
    name = name.lower()
    if name in MODEL_TO_BASE_MODEL:
        return MODEL_TO_BASE_MODEL[name]
    for k, v in MODEL_TO_BASE_MODEL.items():
        if k in name:
            return v
    return None


@dataclass
class Model:
    id: str
    name: str
    clean_name: str
    base_model: str
    template: str
    backend: str
    quant: str
    size: float
    url: str
    known_to_horde: bool
    worker_threads: int = 1


@cached(TTLCache(maxsize=1, ttl=86400))
def get_references():
    """The references are known models, with usually more accurate information than the guesses."""
    return requests.get(
        "https://raw.githubusercontent.com/db0/AI-Horde-text-model-reference/main/db.json"
    ).json()


@cached(TTLCache(maxsize=1, ttl=3600))
def get_models() -> dict[str, Model]:
    """
    Get all models from the Horde API, with estimated sizes, base models, templates, etc.
    """
    references = get_references()

    models = {}
    for model in get_horde_models():
        name = model["name"]
        if "/" in name:
            model_id = name
            reference = references.get(name, {})
            backend = name.split("/", 1)[0].strip()
            clean_name = estimate_clean_name(name).strip()            
            base_model = guess_base_model(clean_name)
            template = (
                BASE_MODELS[base_model]["template"]
                if base_model in BASE_MODELS
                else "unknown"
            )

            if base_model is None:
                logger.warning(f"Unknown model: {clean_name} ({name}).")
                base_model = "unknown"
            url=reference.get("url")
            if url:
                url = url + '/raw/main/tokenizer_config.json'
            else:
                purename = None
                if 'koboldcpp' in name.lower():
                    purename = name.replace("koboldcpp/", "")
                elif 'aphrodite' in name.lower():
                    purename = name.replace("aphrodite/", "")
                if purename:
                    for model_name in references:
                        if not model_name.startswith("koboldcpp/") and not model_name.startswith("aphrodite/") and purename in model_name:
                            url = f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json"
                            break
                    if not url:
                        similar_name = find_similar_reference(name)
                        for model_name in references:
                            if not model_name.startswith("koboldcpp/") and not model_name.startswith("aphrodite/") and similar_name in model_name.lower():
                                url = references[model_name].get("url", f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json")
                                break
                else:
                    url = f"https://huggingface.co/{name}/raw/main/tokenizer_config.json"
            models[name] = Model(
                id=model_id,
                name=name,
                clean_name=clean_name,
                base_model=base_model,
                template=template,
                backend=backend,
                quant=estimate_quant(name),
                url=url,
                size="parameters" in reference
                and reference["parameters"] / 10**9
                or estimate_size(name),
                known_to_horde="name" in reference,
                worker_threads=model.get("count", 1),
            )
    return models
