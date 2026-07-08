"""
Mapping between partial model names and base models.
Move generic names down to the bottom to a void conflicts.
"""

# noinspection SpellCheckingInspection
MODEL_TO_BASE_MODEL = {
    "dusk_rainbow": "llama-3-instruct",
    "airochronos": "alpaca",
    "arliai-rpmax-v1.1": "mistral-instruct",
    "crestfall-mythomax-l2": "alpaca",
    "fimbulvetr": "alpaca",
    "gemmasutra": "gemma-it",
    "hathor_stable": "llama-3-instruct",
    "kobbletiny": "alpaca",
    "l3-lunaris": "llama-3-instruct",
    "l3-stheno": "llama-3-instruct",
    "moistral": "alpaca",
    "nemomix": "mistral-instruct",
    "rocinante": "mistral-instruct",  # Trained on ChatML, Alpaca, and Mistral
    "cydonia": "mistral-instruct",
    "chronos-gold": "custom-chatml",
    "2b_or_not_2b": "custom-chatml",
    "neuraldaredevil-abliterated": "llama-3-instruct",
    "psyonic-cetacean": "orca-2",
    "pygmalion": "alpaca",
    "thegreenlion-sft-v0.1.2": "qwen2-instruct",
    "tinyllama": "llama-2-chat",
    "magnum": "qwen2-instruct",
    # Defaults
    "llama-3.1": "llama-3.1-instruct",
    "llama3.1": "llama-3.1-instruct",
    "mistral": "mistral-instruct",
    "gemma-2": "gemma-2-it",
    "command-r": "openchat-3.5",
    "llama2": "llama-2-chat",
    "llama-3": "llama-3-instruct",
    "phi-3": "phi-3-instruct",
    # Too generic
    "ad": "custom-chatml",
}
