"""Decision model abstraction and TypeSafe AI Jev (System One) integration.

Provides typed decisions (Choice, Noul, Score) inside software without free-form text generation.
Follows the specification in JEV.md §15.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agrag.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model: str = "unknown"
    raw_response: dict[str, Any] = field(default_factory=dict)


class DecisionModel(Protocol):
    def choice(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        question_key: str = "decision",
    ) -> Optional[DecisionResult]:
        ...

    def noul(
        self,
        state: str,
        instructions: str,
        question_key: str = "decision",
    ) -> Optional[float]:
        ...


class MockDecisionModel:
    """Deterministic fallback/mock decision model for unit testing and offline execution."""

    is_configured: bool = True

    def __init__(self, default_choice: str = "unknown"):
        self.default_choice = default_choice

    def choice(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        question_key: str = "decision",
    ) -> Optional[DecisionResult]:
        chosen = self.default_choice if self.default_choice in criteria else next(iter(criteria), "unknown")
        probs = {k: (1.0 if k == chosen else 0.0) for k in criteria}
        return DecisionResult(
            decision=chosen,
            confidence=1.0,
            probabilities=probs,
            model="mock",
        )

    def noul(
        self,
        state: str,
        instructions: str,
        question_key: str = "decision",
    ) -> Optional[float]:
        return 0.5


class JevDecisionModel:
    """TypeSafe AI Jev (System One) client for fast, non-autoregressive typed decisions."""

    API_URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = settings.jev_api_key or os.getenv("JEV_API_KEY", "")
        self.model = model or settings.jev_model or "jev-latest"
        self.timeout_sec = (timeout_ms or settings.jev_timeout_ms or 5000) / 1000.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def choice(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str],
        question_key: str = "decision",
    ) -> Optional[DecisionResult]:
        if not self.is_configured:
            logger.debug("JevDecisionModel skipped: JEV_API_KEY is not configured.")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agrag-tigergraph/1.0",
        }
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                question_key: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }

        import time
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                self.API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            ans = data.get("answers", {}).get(question_key, {})
            usage = data.get("usage", {})
            return DecisionResult(
                decision=ans.get("choice", ""),
                confidence=float(ans.get("confidence", 0.0)),
                probabilities=ans.get("probabilities", {}),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=elapsed_ms,
                model=data.get("model", self.model),
                raw_response=data,
            )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.warning("Jev API returned HTTP %d: %s", e.code, err_body)
            return None
        except Exception as e:
            logger.warning("Jev request failed: %s", e)
            return None

    def noul(
        self,
        state: str,
        instructions: str,
        question_key: str = "decision",
    ) -> Optional[float]:
        if not self.is_configured:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agrag-tigergraph/1.0",
        }
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                question_key: {
                    "type": "noul",
                    "instructions": instructions,
                }
            },
        }

        try:
            req = urllib.request.Request(
                self.API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ans = data.get("answers", {}).get(question_key, {})
            return float(ans.get("noul", 0.5))
        except Exception as e:
            logger.warning("Jev noul request failed: %s", e)
            return None


def get_decision_model() -> DecisionModel:
    """Factory returning configured DecisionModel or Mock fallback."""
    if settings.jev_enabled or settings.jev_api_key:
        model = JevDecisionModel()
        if model.is_configured:
            return model
    return MockDecisionModel()
