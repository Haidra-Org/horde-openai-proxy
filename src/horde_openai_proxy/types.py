from typing import Optional, Union, List

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """An OpenAI Chat Completion request."""

    model_config = {"extra": "ignore"}

    messages: list[dict]
    model: str
    frequency_penalty: Optional[float] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    max_tokens: Optional[int] = Field(None)
    n: Optional[int] = Field(None)
    stop: List[str] = Field(None)
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    timeout: Union[float, None] = Field(None)


class ChatCompletionResponse(BaseModel):
    """An OpenAI Chat Completion response."""

    id: str
    choices: list[dict]
    created: int
    model: str
    usage: dict


class ModelGenerationInput(BaseModel):
    """A (partial) KoboldAI generation input."""

    model_config = {"extra": "ignore"}

    max_context_length: int = 2048
    max_length: Optional[int] = 512
    n: Optional[int] = None
    rep_pen: Optional[float] = None
    stop_sequence: List[str] = []
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    # Additional formatting options, handled by the proxy. It uses KoboldAI's GUI defaults!
    frmtadsnsp: bool = True
    frmtrmblln: bool = False
    frmtrmspch: bool = False
    frmttriminc: bool = True
    singleline: bool = False


class HordeRequest(BaseModel):
    """A request to the Horde API."""

    prompt: str = Field(...)
    models: List[str] = Field(...)
    timeout: int = Field((60 * 20) - 30)
    params: ModelGenerationInput


class TextGeneration(BaseModel):
    """A generated text returned from the Horde."""

    uuid: str
    model: str
    text: str
    kudos: int
