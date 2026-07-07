from pydantic import BaseModel

from horde_openai_proxy import (
    horde_to_openai,
    openai_to_horde,
    HordeRequest,
    ModelGenerationInput,
    apply_template,
)
from horde_openai_proxy.data import BASE_MODELS
from horde_openai_proxy.template import prompt_to_messages


def compare_dict(a: dict, b: dict) -> dict:
    differences = {}
    for k, v in a.items():
        if isinstance(v, dict):
            differences.update(compare_dict(v, b[k]))
        elif v != b[k]:
            differences[k] = (v, b[k])
    return differences


def safe_del(d: dict, k: str) -> None:
    if k in d:
        del d[k]


def compare_models(a: BaseModel, b: BaseModel) -> None:
    differences = compare_dict(a.model_dump(), b.model_dump())
    safe_del(differences, "messages")
    safe_del(differences, "prompt")
    safe_del(differences, "stop_sequence")
    safe_del(differences, "stop")
    assert not differences, differences


def test_from_horde():
    h1 = HordeRequest(
        prompt="Test prompt!",
        models=["Henk717/airochronos-33B"],
        timeout=7,
        params=ModelGenerationInput(
            max_context_length=2048,  # Not supported
            max_length=77,
            n=7,
            rep_pen=1.7,
            stop_sequence=["test"],
            temperature=0.6,
            top_p=0.5,
        ),
    )

    o1 = horde_to_openai(h1)
    h2 = openai_to_horde(o1)
    o2 = horde_to_openai(h2)

    compare_models(h1, h2)
    compare_models(o1, o2)


def test_from_prompt():
    templates = set()
    for base_model in BASE_MODELS:
        template = BASE_MODELS[base_model]["template"]
        if template not in templates:
            templates.add(template)

            messages = [
                {"role": "system", "content": "The system prompt"},
                {"role": "user", "content": "I'm a user"},
                {"role": "assistant", "content": "I'm a bot"},
                {"role": "user", "content": "I'm a user"},
            ]

            for _ in range(2):
                prompt = apply_template(messages, base_model)
                restored = prompt_to_messages(prompt)

                # Some templates treat systems different, it's not a bug
                if template == "gemma-it" and len(messages) == 4:
                    messages.pop(0)
                    messages.pop(0)
                    restored.pop(0)
                elif len(restored) == 4 and len(messages) == 3:
                    restored.pop(0)

                # Compare
                assert len(messages) == len(restored)
                for m1, m2 in zip(messages, restored):
                    assert m1["role"] == m2["role"], base_model
                    assert m1["content"].strip() == m2["content"].strip(), base_model
                messages.pop(0)
