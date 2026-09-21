from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    tg_host: str = os.getenv("TG_HOST", "")
    tg_graph: str = os.getenv("TG_GRAPH", "OlympicsRAG")
    tg_secret: str = os.getenv("TG_SECRET", "")
    tg_username: str = os.getenv("TG_USERNAME", "")
    tg_password: str = os.getenv("TG_PASSWORD", "")
    llm_provider: str = os.getenv("AGRAG_LLM_PROVIDER", "groq")   # "groq" (free) or "anthropic"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("AGRAG_MODEL", "openai/gpt-oss-20b")
    max_tokens: int = int(os.getenv("AGRAG_MAX_TOKENS", "1024"))
    embed_model: str = os.getenv("AGRAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    jev_enabled: bool = os.getenv("JEV_ENABLED", "false").lower() in ("1", "true", "yes")
    jev_api_key: str = os.getenv("JEV_API_KEY", "")
    jev_model: str = os.getenv("JEV_MODEL", "jev-latest")
    jev_timeout_ms: int = int(os.getenv("JEV_TIMEOUT_MS", "5000"))

    @property
    def groq_api_keys(self) -> list[str]:
        """GROQ_API_KEY may hold several comma-separated keys (from separate free-tier accounts) so
        GroqLLM can rotate off a key that hits its daily cap instead of blocking the whole run."""
        return [k.strip() for k in self.groq_api_key.split(",") if k.strip()]


settings = Settings()
