"""
A mapping from base model to tokenizer, template, and config.
"""

BASE_MODELS = {
    "llama-3.1-instruct": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "template": "llama-3-instruct",
        "config": "llama-3.1-instruct",
    },
    "llama-3-instruct": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "template": "llama-3-instruct",
        "config": "llama-3-instruct",
    },
    "llama-2-chat": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "template": "llama-2-chat",
        "config": "llama-2-chat",
    },
    "codellama-instruct": {
        "model": "meta-llama/CodeLlama-7b-Instruct-hf",
        "template": "llama-2-chat",
        "config": "llama-2-chat",
    },
    "qwen2-instruct": {
        "model": "Qwen/Qwen2-7B-Instruct",
        "template": "chatml",
        "config": "qwen2-instruct",
    },
    "qwen1.5-chat": {
        "model": "Qwen/Qwen1.5-7B-Chat",
        "template": "chatml",
        "config": "qwen2-instruct",
    },
    "mistral-instruct": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "template": "mistral-instruct",
        "config": "mistral-instruct",
    },
    "phi-3-instruct": {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "template": "phi-3",
        "config": "phi-3",
    },
    "yi-1.5-chat": {
        "model": "01-ai/Yi-1.5-6B-Chat",
        "template": "chatml",
        "config": "yi-chat",
    },
    "yi-chat": {
        "model": "01-ai/Yi-6B-Chat",
        "template": "chatml",
        "config": "yi-chat",
    },
    "gemma-it": {
        "model": "google/gemma-7b-it",
        "template": "gemma-it",
        "config": "gemma-it",
    },
    "gemma-2-it": {
        "model": "google/gemma-2-9b-it",
        "template": "gemma-it",
        "config": "gemma-it",
    },
    "llama3-chatqa-1.5": {
        "model": "nvidia/Llama3-ChatQA-1.5-8B",
        "template": "chatqa",
        "config": "chatqa",
    },
    "openchat-3.5": {
        "model": "openchat/openchat_3.5",
        "template": "openchat-3.5",
        "config": "openchat-3.5",
    },
    "starling-lm": {
        "model": "berkeley-nest/Starling-LM-7B-alpha",
        "template": "openchat-3.5",
        "config": "openchat-3.5",
    },
    "zephyr": {
        "model": "HuggingFaceH4/zephyr-7b-alpha",
        "template": "zephyr",
        "config": "zephyr",
    },
    "vicuna": {
        "model": "lmsys/vicuna-7b-v1.5",
        "template": "vicuna",
        "config": "vicuna",
    },
    "orca-2": {
        "model": "microsoft/Orca-2-7b",
        "template": "chatml",
        "config": "orca-2",
    },
    "falcon-instruct": {
        "model": "tiiuae/falcon-7b-instruct",
        "template": "falcon-instruct",
        "config": "",
    },
    "solar-instruct": {
        "model": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "template": "solar-instruct",
        "config": "solar-instruct",
    },
    "alpaca": {
        "model": "tatsu-lab/alpaca-7b-wdiff",
        "template": "alpaca",
        "config": "alpaca",
    },
    "amberchat": {
        "model": "LLM360/AmberChat",
        "template": "amberchat",
        "config": "amberchat",
    },
    "saiga": {
        "model": "IlyaGusev/saiga_mistral_7b_lora",
        "template": "saiga",
        "config": "saiga",
    },
    # Odd models, e.g, fine-tuned on another template than the baseline
    "custom-chatml": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "template": "chatml",
        "config": "mistral-instruct",
    },
}
