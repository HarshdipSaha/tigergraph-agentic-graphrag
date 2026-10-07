from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agrag.config import settings

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.I)
_KEY_STATE_FILE = Path("data/.groq_key_state.json")


def _load_key_state() -> dict:
    try:
        return json.loads(_KEY_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_key_state(state: dict) -> None:
    try:
        _KEY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass  # best-effort; rotation still works within this process without persistence


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def complete(self, system: str, user: str, *, timeout_s: float | None = None) -> LLMResponse: ...


class LLMTimeoutError(TimeoutError):
    """A completion exceeded its caller-supplied wall-clock budget."""


class GroqKeyPoolExhausted(RuntimeError):
    """Every configured same-organization Groq project key is currently unavailable."""


class FakeLLM:
    """Returns scripted answers in order. Token counts are word counts so tests can assert on them."""

    def __init__(self, answers: list[str]):
        self._answers = list(answers)
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, timeout_s: float | None = None) -> LLMResponse:
        if not self._answers:
            raise RuntimeError("FakeLLM script exhausted")
        self.calls += 1
        self.prompts.append((system, user))
        text = self._answers.pop(0)
        return LLMResponse(text=text, input_tokens=len(system.split()) + len(user.split()), output_tokens=len(text.split()))


class GroqLLM:
    """Groq chat-completions client with same-organization project-key failover.

    Comma-separated keys can rotate across projects within one organization. Project-level limits can differ,
    while organization-level limits remain a shared ceiling. When every configured key is rate limited this
    wrapper raises a generic pool-exhausted error instead of sleeping indefinitely. Rotation state persists
    to data/.groq_key_state.json so a new eval process resumes at the last active key.

    `timeout_s` applies one monotonic deadline to SDK I/O and rate-limit handling. Each HTTP request receives
    the remaining time via the SDK's `with_options(timeout=...)` support.
    """

    def __init__(self, model: str | None = None, max_tokens: int | None = None, max_retries: int = 8):
        from groq import Groq  # lazy import so tests never need the SDK configured

        self.keys = settings.groq_api_keys or [settings.groq_api_key or None]
        state = _load_key_state()
        self.exhausted_until: dict[int, float] = {int(k): float(v) for k, v in state.get("exhausted_until", {}).items()}
        self.idx = state.get("idx", 0) % len(self.keys)
        self._client = Groq(api_key=self.keys[self.idx], max_retries=0)
        self.model = model or settings.model
        self.max_tokens = max_tokens or settings.max_tokens
        self.max_retries = max_retries

    def _persist(self) -> None:
        _save_key_state({"idx": self.idx, "exhausted_until": {str(k): v for k, v in self.exhausted_until.items()}})

    def _switch_key(self, idx: int) -> None:
        from groq import Groq

        self.idx = idx
        self._client = Groq(api_key=self.keys[self.idx], max_retries=0)
        self._persist()

    def _next_available_key(self, now: float) -> int | None:
        for offset in range(1, len(self.keys) + 1):
            cand = (self.idx + offset) % len(self.keys)
            if self.exhausted_until.get(cand, 0.0) <= now:
                return cand
        return None

    @staticmethod
    def _retry_after_seconds(error) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
        match = _RETRY_AFTER_RE.search(str(error))
        return float(match.group(1)) if match else None

    @staticmethod
    def _is_daily_limit(error) -> bool:
        message = str(error).lower()
        return any(token in message for token in ("tokens per day", " tpd", "requests per day", " rpd"))

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMTimeoutError("LLM completion timed out")
        return remaining

    def complete(self, system: str, user: str, *, timeout_s: float | None = None) -> LLMResponse:
        import groq

        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        attempts = 0
        while True:
            remaining = self._remaining(deadline)
            now = time.time()
            if self.exhausted_until.get(self.idx, 0.0) > now:
                nxt = self._next_available_key(now)
                if nxt is None:
                    raise GroqKeyPoolExhausted("all configured Groq project keys are rate limited")
                self._switch_key(nxt)
            try:
                client = self._client.with_options(timeout=remaining) if remaining is not None else self._client
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                text = (resp.choices[0].message.content or "").strip()
                return LLMResponse(text=text, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)
            except groq.RateLimitError as error:
                now = time.time()
                retry_after = self._retry_after_seconds(error)
                wait = (retry_after + 1.0) if retry_after is not None else (86_400.0 if self._is_daily_limit(error) else 60.0)
                self.exhausted_until[self.idx] = now + wait
                self._persist()
                nxt = self._next_available_key(now)
                if nxt is not None and len(self.keys) > 1:
                    self._switch_key(nxt)
                    attempts = 0
                    continue
                if all(self.exhausted_until.get(index, 0.0) > now for index in range(len(self.keys))):
                    raise GroqKeyPoolExhausted("all configured Groq project keys are rate limited") from None
                attempts += 1
                if attempts > self.max_retries:
                    raise
                remaining = self._remaining(deadline)
                pause = min(wait, remaining) if remaining is not None else wait
                if pause <= 0:
                    raise LLMTimeoutError("LLM completion timed out") from None
                time.sleep(pause)
            except groq.AuthenticationError:
                # A revoked or malformed project key must not prevent trying the other
                # configured keys. Same-organization rotation stays under the shared cap.
                self.exhausted_until[self.idx] = time.time() + 86_400.0
                self._persist()
                nxt = self._next_available_key(time.time())
                if nxt is None:
                    raise GroqKeyPoolExhausted("all configured Groq project keys are unavailable") from None
                self._switch_key(nxt)
                attempts += 1
                if attempts >= len(self.keys):
                    raise GroqKeyPoolExhausted("all configured Groq project keys are unavailable") from None
            except groq.APITimeoutError:
                if deadline is not None:
                    raise LLMTimeoutError("LLM completion timed out") from None
                raise


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        import anthropic  # lazy import so tests never need the SDK configured

        self._client = anthropic.Anthropic()
        self.model = model or settings.model
        self.max_tokens = max_tokens or settings.max_tokens

    def complete(self, system: str, user: str, *, timeout_s: float | None = None) -> LLMResponse:
        client = self._client.with_options(timeout=timeout_s) if timeout_s is not None else self._client
        resp = client.messages.create(
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
