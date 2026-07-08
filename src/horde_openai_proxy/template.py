import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Optional

from transformers import PreTrainedTokenizerBase, AutoTokenizer

from .data import BASE_MODELS


@cache
def get_tokenizer(model: str) -> PreTrainedTokenizerBase:
    """
    Get the adjusted tokenizer for the model.
    :param model: Model name
    :return: Tokenizer
    """
    data = BASE_MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(data["model"], trust_remote_code=True)
    template_path = os.path.join(
        os.path.dirname(__file__),
        f"./chat_templates/chat_templates/{data['template']}.jinja",
    )
    template = open(template_path).read()
    template = template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = template
    return tokenizer


@dataclass
class GenerationConfig:
    system_prompt: Optional[str]
    stop_words: list[str]


@cache
def get_generation_config(model: str) -> GenerationConfig:
    """
    Get the generation config for the model, such as stop words.
    :param model: Model name
    :return: GenerationConfig
    """
    if BASE_MODELS[model]["config"]:
        config_path = os.path.join(
            os.path.dirname(__file__),
            f"./chat_templates/generation_configs/{BASE_MODELS[model]['config']}.json",
        )

        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"stop_str": None, "stop_token_ids": [], "system_prompt": None}

    stop_words: list[str] = []

    if config["stop_str"]:
        stop_words.append(str(config["stop_str"]))

    if config["stop_token_ids"]:
        tokenizer = get_tokenizer(model)
        for token_id in config["stop_token_ids"]:
            stop_words.append(tokenizer.decode(token_id))

    return GenerationConfig(
        system_prompt=config["system_prompt"],
        stop_words=stop_words,
    )


def apply_template(conversation: list, model: str) -> str:
    """
    Apply the chat template to the conversation
    :param conversation:  List of messages
    :param model: Model name on Hugging Face
    :return: Prepared prompt
    """
    if "system" not in {m["role"] for m in conversation}:
        config = get_generation_config(model)
        if config.system_prompt:
            conversation = [
                {"role": "system", "content": config.system_prompt}
            ] + conversation

    return str(
        get_tokenizer(model).apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    )


def to_role(role: str) -> str:
    role = role.lower()
    if "system" in role:
        return "system"
    if "assistant" in role:
        return "assistant"
    return "user"


def generic_clean(prompt: str) -> str:
    return prompt.replace("<|eot_id|>", "\n").strip("<s>").strip("</s>")


def cleanup_messages(messages: list) -> list:
    return [
        {"role": to_role(m["role"]), "content": generic_clean(m["content"])}
        for m in messages
        if generic_clean(m["content"])
    ]


class Parser:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.lstrip()

        self.messages = []

    def lstrip(self) -> None:
        self.prompt = self.prompt.lstrip()

    def skip(self, text: str) -> bool:
        if self.prompt.startswith(text):
            self.prompt = self.prompt[len(text) :]
            self.lstrip()
            return True
        return False

    def peeks(self, text: str) -> bool:
        return self.prompt.startswith(text)

    def read_until(self, *text: str, keep: bool = False) -> str:
        index = len(self.prompt)
        next_token = ""
        for t in text:
            i = self.prompt.find(t)
            if i != -1 and i < index:
                index = i
                next_token = t
        if not next_token:
            result = self.prompt
            self.prompt = ""
            return result
        result = self.prompt[:index]
        self.prompt = self.prompt[index + (0 if keep else len(next_token)) :]
        self.lstrip()
        return result

    def done(self) -> bool:
        return not self.prompt

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": to_role(role), "content": generic_clean(content)})


def _prompt_to_messages_llama3_instruct(parser: Parser) -> None:
    while not parser.done():
        parser.read_until("<|start_header_id|>")
        role = parser.read_until("<|end_header_id|>")
        content = parser.read_until("<|eot_id|>")
        parser.add(role, content)


def _prompt_to_messages_llama2_chat(parser: Parser) -> None:
    potential_system = parser.read_until("[INST]")

    # System messages
    if parser.skip("<<SYS>>"):
        parser.add("system", parser.read_until("<</SYS>>"))
    else:
        parser.add("system", potential_system)

    # User and assistant messages
    while not parser.done():
        parser.add("user", parser.read_until("[/INST]"))
        parser.add("assistant", parser.read_until("[INST]"))


def _prompt_to_messages_chatml(parser: Parser) -> None:
    parser.read_until("<|im_start|>")

    while not parser.done():
        role = parser.read_until("\n")
        content = parser.read_until("<|im_end|>")
        parser.add(role, content)


def _prompt_to_messages_openchat(parser: Parser) -> None:
    parser.skip("<s>")

    while not parser.done():
        if parser.skip("GPT4 Correct User:"):
            parser.add("user", parser.read_until("<|end_of_turn|>"))
        elif parser.skip("GPT4 Correct Assistant:"):
            parser.add("assistant", parser.read_until("<|end_of_turn|>"))
        elif len(parser.messages) == 0:
            parser.add("system", parser.read_until("<|end_of_turn|>"))
        else:
            parser.read_until("<|end_of_turn|>")


def _prompt_to_messages_gemma_instruct(parser: Parser) -> None:
    while not parser.done():
        parser.read_until("<start_of_turn>")
        if parser.skip("user"):
            parser.add("user", parser.read_until("<end_of_turn>"))
        elif parser.skip("model"):
            parser.add("assistant", parser.read_until("<end_of_turn>"))
        else:
            parser.read_until("<end_of_turn>")


def _prompt_to_messages_phi3(parser: Parser) -> None:
    parser.skip("<s>")

    while not parser.done():
        found = False
        for role in ["system", "user", "assistant"]:
            if parser.skip(f"<|{role}|>"):
                parser.add(role, parser.read_until("</s>", "<|end|>"))
                found = True
                break
        if not found:
            parser.read_until("<|end|>")


def _prompt_to_messages_saiga(parser: Parser) -> None:
    while not parser.done():
        parser.skip("<s>")
        if parser.skip("system"):
            parser.add("system", parser.read_until("</s>"))
        elif parser.skip("user"):
            parser.add("user", parser.read_until("</s>"))
        elif parser.skip("bot"):
            parser.add("assistant", parser.read_until("</s>"))
        else:
            parser.read_until("</s>")


def _prompt_to_messages_alpaca(
    parser: Parser,
    system: str = "### System:",
    user: str = "### Instruction:",
    assistant: str = "### Response:",
) -> None:
    parser.skip("<s>")
    parser.skip(system)
    parser.add("system", parser.read_until(user, keep=True))

    while not parser.done():
        if parser.skip(user):
            parser.add("user", parser.read_until(assistant, keep=True))
        elif parser.skip(assistant):
            parser.add("assistant", parser.read_until(user, keep=True))


def _prompt_to_messages_vicuna(parser: Parser) -> None:
    parser.add("system", parser.read_until("USER:"))
    while not parser.done():
        parser.add("user", parser.read_until("ASSISTANT:"))
        parser.add("assistant", parser.read_until("USER:"))


def _prompt_to_messages_chatqa(parser: Parser) -> None:
    parser.skip("<|begin_of_text|>")
    parser.skip("System:")

    potential_system = parser.read_until("User:", keep=True)

    while not parser.done():
        if parser.skip("System:"):
            parser.add("system", parser.read_until("User:", keep=True))
        else:
            if potential_system and len(parser.messages) == 0:
                parser.add("system", potential_system)
                potential_system = None

            if parser.skip("User:"):
                parser.add("user", parser.read_until("Assistant:", keep=True))
            elif parser.skip("Assistant:"):
                parser.add("assistant", parser.read_until("User:", keep=True))
            else:
                break


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    """
    Convert a prompt to a list of messages
    :param prompt: Prompt
    :return: List of messages
    """
    parser = Parser(prompt)
    if "<|start_header_id|>" in prompt and "<|end_header_id|>" in prompt:
        _prompt_to_messages_llama3_instruct(parser)
    elif "[INST]" in prompt and "[/INST]" in prompt:
        _prompt_to_messages_llama2_chat(parser)
    elif "<|im_start|>" in prompt and "<|im_end|>" in prompt:
        _prompt_to_messages_chatml(parser)
    elif "<|end_of_turn|>" in prompt:
        _prompt_to_messages_openchat(parser)
    elif "<end_of_turn>" in prompt:
        _prompt_to_messages_gemma_instruct(parser)
    elif "<|user|>" in prompt:
        _prompt_to_messages_phi3(parser)
    elif "<s>user" in prompt and "<s>bot" in prompt:
        _prompt_to_messages_saiga(parser)
    elif "### Instruction:" in prompt:
        _prompt_to_messages_alpaca(parser)
    elif "### User:" in prompt:
        _prompt_to_messages_alpaca(parser, user="### User:", assistant="### Assistant:")
    elif "###Human:" in prompt:
        _prompt_to_messages_alpaca(parser, user="###Human:", assistant="###Assistant:")
    elif "USER:" in prompt and "ASSISTANT:" in prompt:
        _prompt_to_messages_vicuna(parser)
    elif "User:" in prompt and "Assistant:" in prompt:
        _prompt_to_messages_chatqa(parser)
    else:
        parser.add("user", prompt)
    return cleanup_messages(parser.messages)
