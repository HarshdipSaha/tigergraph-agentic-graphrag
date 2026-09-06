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
    Default model is settings.model; sign up at https://console.groq.com/keys.

    The free tier has two rate-limit layers, both hit in practice on this project: a tokens-per-minute cap
    (observed: 8,000 TPM) that a handful of large-context requests can exceed within seconds, and a
    tokens-per-day cap (observed: 200,000 TPD, per model, per account) that a full ~150-question eval run
    can exceed regardless of pacing. complete() handles both: a TPM 429 is waited out on the same key using
    Groq's own stated wait ("Please try again in 3.89s"); a TPD 429 rotates immediately to the next key in
    GROQ_API_KEY (settings.groq_api_keys — several comma-separated keys from separate free-tier accounts),
    since waiting out a whole day is not practical. Rotation state persists to data/.groq_key_state.json so
    each separate `python -m agrag.eval.run` process (its own Python process, hence its own GroqLLM instance)
    resumes from the last known-good key instead of blindly restarting at key 0."""

    def __init__(self, model: str | None = None, max_tokens: int | None = None, max_retries: int = 8):
        from groq import Groq  # lazy import so tests never need the SDK configured

        self.keys = settings.groq_api_keys or [settings.groq_api_key or None]
        state = _load_key_state()
        self.exhausted_until: dict[int, float] = {int(k): v for k, v in state.get("exhausted_until", {}).items()}
        self.idx = state.get("idx", 0) % len(self.keys)
        self._client = Groq(api_key=self.keys[self.idx], max_retries=0)  # we retry/rotate ourselves, deliberately
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

    def complete(self, system: str, user: str) -> LLMResponse:
        import groq

        attempts = 0
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                text = (resp.choices[0].message.content or "").strip()
                return LLMResponse(text=text, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens)
            except groq.RateLimitError as e:
                msg = str(e)
                m = _RETRY_AFTER_RE.search(msg)
                now = time.time()
                is_daily = "tokens per day" in msg.lower() or " tpd" in msg.lower()
                wait = float(m.group(1)) + 1.0 if m else (3600.0 if is_daily else 5.0)
                if is_daily and len(self.keys) > 1:
                    self.exhausted_until[self.idx] = now + wait
                    self._persist()
                    nxt = self._next_available_key(now)
                    if nxt is not None:
                        self._switch_key(nxt)
                        continue  # retry immediately on the fresh key, no sleep
                    wait_for = max(1.0, min(self.exhausted_until.values()) - now)
                    time.sleep(wait_for)
                    self._switch_key(min(self.exhausted_until, key=self.exhausted_until.get))
                    continue
                attempts += 1
                if attempts > self.max_retries:
                    raise
                time.sleep(wait)


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
