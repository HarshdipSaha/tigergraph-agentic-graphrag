from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agrag.config import settings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class FakeLLM:
    """Returns scripted answers in order. Token counts are word counts so tests can assert on them."""

    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        if not self._answers:
            raise RuntimeError("FakeLLM script exhausted")
        self.calls += 1
        self.prompts.append((system, user))
        text = self._answers.pop(0)
        return LLMResponse(text=text, input_tokens=len(system.split()) + len(user.split()), output_tokens=len(text.split()))


class GroqLLM:
    """Thin wrapper over the Groq chat-completions API (OpenAI-compatible). Free tier, no card required.
    Default model is settings.model (llama-3.3-70b-versatile); sign up at https://console.groq.com/keys."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        from groq import Groq  # lazy import so tests never need the SDK configured

        self._client = Groq(api_key=settings.groq_api_key or None)
        self.model = model or settings.model
        self.max_tokens = max_tokens or settings.max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
        return LLMResponse(text=text, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API, for anyone who prefers Claude over the free Groq default.
    Model defaults to settings.model, so override AGRAG_MODEL to a Claude model id when using this class."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        import anthropic  # lazy import so tests never need the SDK configured

        self._client = anthropic.Anthropic()
        self.model = model or settings.model
        self.max_tokens = max_tokens or settings.max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text").strip()
        return LLMResponse(text=text, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)


def build_llm(model: str | None = None, max_tokens: int | None = None) -> "LLM":
    """Construct the configured LLM (settings.llm_provider: "groq" default, or "anthropic")."""
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(model=model, max_tokens=max_tokens)
    return GroqLLM(model=model, max_tokens=max_tokens)
