import os
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAILLM:
    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model

    def complete(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


def get_llm(s: Settings) -> LLM:
    if s.llm_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(s.anthropic_model)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAILLM(s.openai_model)
