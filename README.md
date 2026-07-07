# Horde-OpenAI-Proxy

Utilities to convert from the Horde LLM endpoint format to the OpenAI Chat API format. This allows us to

* use the AI Horde in existing applications that use the OpenAI API
* connect backends offering the OpenAI API to the AI Horde (e.g., [Ollama](https://ollama.com/))

## Differences

Features are restricted to the intersection of both APIs:

### Horde to OpenAI

* Formatting is not supported (`frmtadsnsp`, `frmtrmblln`, `frmtrmspch`, `frmttriminc`, `singleline`)
    * There are utility functions for this tho and can be done by the proxy.
* `max_context_length` is a constant passed to the converter.
* Not supported at all are a, k, tfs, and typical sampling, and `rep_pen_range`, `rep_pen_slope`,
  `use_default_badwordsids`, `smoothing_factor`, `dynatemp_range`, `dynatemp_exponent`
* Only the first `model` is used.

### OpenAI to Horde

* System message is not always supported, e.g., on Gemma based models.
* `logprobs`, `logit_bias` is not supported.
* Any response format or structured format is not supported.
* No tools, functions, seed, or streaming.
* `model` is a comma-separated list of models.

## Convert formats

Convert freely between the two formats with the following functions:

```py
from horde_openai_proxy import horde_to_openai, openai_to_horde, ChatCompletionRequest, HordeRequest

a: HordeRequest = ...

b: ChatCompletionRequest = horde_to_openai(a)
a: HordeRequest = openai_to_horde(b)
```

## As proxy

[`horde_openai_proxy/endpoint.py`](examples/endpoint.py) provides a FastAPI example on how to access the AI Horde via
the OpenAI API, with a utility endpoint to retrieve filtered models.

## As bridge

[`horde_openai_proxy/bridge.py`](examples/bridge.py) provides an example bridge, however it heavily struggles with keeping character.

To run the bridge:

```shell
uv sync --group fastapi
uv run fastapi dev -e examples.endpoint:app
```

## Utilities

Included are also utilities to clean up the model zoo mess:

* [get_models()](horde_openai_proxy/model.py) to fetch currently active models with name, parameters, quantization, and
  base model.
* [apply_template()](horde_openai_proxy/template.py) to convert a list of messages into a prompt.

## Setup

Login or provide your HF token as `HF_TOKEN` env var to access restricted models.
You also have to manually accept all baseline models.

## Maintenance

* [data](horde_openai_proxy/data/__init__.py) contains mappings between (partial) names and base model, param sizes, and
  base models and their template
* [chat_templates](horde_openai_proxy/chat_templates) needs to be updated once in a while
